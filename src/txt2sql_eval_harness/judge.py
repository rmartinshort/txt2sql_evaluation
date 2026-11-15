"""Hybrid funnel judge: Stage 0 execution, Stage 1 rules/alignment, Stage 2 packets, Stage 3 Jev, Stage 4 escalation."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import tempfile
import time
from typing import Any

from txt2sql_eval_harness.alignment import align_tables
from txt2sql_eval_harness.claims import check_d0, check_d2, extract_and_bind_claims
from txt2sql_eval_harness.clients import CallContext, call_model
from txt2sql_eval_harness.comparator import (
    DEFAULT_JUDGE_RHO,
    cells_equal,
    classify_cell,
    is_column_pair_exact,
)
from txt2sql_eval_harness.models import (
    Alignment,
    COUNTEREXAMPLE_ERROR_KINDS,
    ErrorKind,
    EvalConfig,
    GroundTruth,
    MethodName,
    ModeName,
    ModelCall,
    ResultTable,
    Step,
    ThresholdPair,
    Trace,
    Verdict,
    VerdictLabel,
)
from txt2sql_eval_harness.packet import PreparedCandidate, build_packet
from txt2sql_eval_harness.rubrics import PROMPT_HASH, RUBRICS, escalation_instructions
from txt2sql_eval_harness.sandbox import execute_sandboxed
from txt2sql_eval_harness.sql_analysis import (
    create_d5_deletion_copy,
    extract_contract,
    extract_schema_ddl,
    is_d1_superset_equivalent,
    is_det,
    is_rd_eligible,
    is_volatile,
    normalize_sql_n,
    parse_single_sqlite,
)


def over_input_limits(trace: Trace, cfg: EvalConfig) -> bool:
    """Enforce input ceilings on trace steps, SQL byte size, final answer chars, and column width."""
    if len(trace.steps) > cfg.max_steps:
        return True
    if trace.final_answer is not None and len(trace.final_answer) > cfg.max_answer_chars:
        return True
    for st in trace.steps:
        if len(st.sql.encode("utf-8")) > cfg.max_sql_bytes:
            return True
        if st.result is not None and len(st.result.columns) > cfg.max_columns:
            return True
    return False


def mode_of(trace: Trace) -> ModeName:
    """Return `'full_trace'` when a final answer is present, otherwise `'sql_only'`."""
    return "full_trace" if trace.final_answer is not None else "sql_only"


def _is_1x1_numeric(table: ResultTable | None) -> tuple[bool, Any]:
    """Return whether `table` is a complete 1x1 numeric table together with its scalar cell."""
    if (
        table is None
        or not table.complete
        or len(table.columns) != 1
        or len(table.rows) != 1
    ):
        return False, None
    cell = table.rows[0][0]
    cc = classify_cell(cell)
    if cc.kind in ("exact_number", "approx_number"):
        return True, cell
    return False, None


def _numeric_1x1_matches_any_scale(cell_a: Any, cell_b: Any, rho: Decimal) -> bool:
    """Check 1x1 numeric cell equality across unit, 100x, and 0.01x percent scales."""
    exact_1 = is_column_pair_exact([cell_a], [cell_b])
    if cells_equal(cell_a, cell_b, tolerant=not exact_1, rho=rho, scale_b=Decimal(1)):
        return True
    for scale in (Decimal(100), Decimal("0.01")):
        if cells_equal(cell_a, cell_b, tolerant=True, rho=rho, scale_b=scale):
            return True
    return False


def _is_scalar_mismatch(gt: ResultTable | None, step_res: ResultTable | None, rho: Decimal) -> bool:
    """Return whether two 1x1 numeric tables disagree even after percentage scaling."""
    ok_a, cell_a = _is_1x1_numeric(gt)
    ok_b, cell_b = _is_1x1_numeric(step_res)
    if ok_a and ok_b:
        return not _numeric_1x1_matches_any_scale(cell_a, cell_b, rho)
    return False


def _make_abstain(
    trace_id: str,
    method: MethodName,
    mode: ModeName,
    reason: str,
    cfg_hash: str,
    *,
    calls: list[ModelCall] | None = None,
    det_ms: int = 0,
    esc_prep_ms: int = 0,
    escalated: bool = False,
    cand: PreparedCandidate | None = None,
    packet_sha256: str | None = None,
) -> Verdict:
    """Construct an `ABSTAIN` verdict with aggregated cold/spend costs and optional trace evidence."""
    call_list = list(calls or [])
    billed = sum(c.cold.billed_usd for c in call_list)
    ambig = sum(c.cold.ambiguous_usd for c in call_list)
    spend = sum(
        c.cold.billed_usd + c.cold.ambiguous_usd
        for c in call_list
        if not c.from_cache
    )
    wall = sum(c.cold.wall_ms for c in call_list)
    return Verdict(
        trace_id=trace_id,
        method=method,
        mode=mode,
        label="ABSTAIN",
        decided_by="abstain",
        reason=reason,
        escalated=escalated,
        method_ok=cand.method_ok if cand is not None else None,
        flags=list(cand.flags) if cand is not None else [],
        claims=list(cand.claims) if cand is not None else [],
        alignments=[cand.alignment] if cand is not None else [],
        packet_sha256=packet_sha256,
        b4_context=None,
        b4_rows_shown=None,
        b4_rows_total=None,
        calls=call_list,
        det_ms=det_ms,
        esc_prep_ms=esc_prep_ms,
        cost_billed_usd=billed,
        cost_conservative_usd=billed + ambig,
        latency_ms=det_ms + esc_prep_ms + wall,
        spend_usd=spend,
        config_hash=cfg_hash,
    )


def _make_verdict(
    trace_id: str,
    method: MethodName,
    mode: ModeName,
    label: VerdictLabel,
    decided_by: Any,
    reason: str | None,
    cfg_hash: str,
    *,
    escalated: bool = False,
    flags: list[str] | None = None,
    calls: list[ModelCall] | None = None,
    det_ms: int = 0,
    esc_prep_ms: int = 0,
    cand: PreparedCandidate | None = None,
    packet_sha256: str | None = None,
) -> Verdict:
    """Construct a decided (`CORRECT`, `INCORRECT`, or `AMBIGUOUS`) verdict with aggregated cost metrics."""
    call_list = list(calls or [])
    billed = sum(c.cold.billed_usd for c in call_list)
    ambig = sum(c.cold.ambiguous_usd for c in call_list)
    spend = sum(
        c.cold.billed_usd + c.cold.ambiguous_usd
        for c in call_list
        if not c.from_cache
    )
    wall = sum(c.cold.wall_ms for c in call_list)
    return Verdict(
        trace_id=trace_id,
        method=method,
        mode=mode,
        label=label,
        decided_by=decided_by,
        reason=reason,
        escalated=escalated,
        method_ok=cand.method_ok if cand is not None else None,
        flags=list(flags if flags is not None else (cand.flags if cand is not None else [])),
        claims=list(cand.claims) if cand is not None else [],
        alignments=[cand.alignment] if cand is not None else [],
        packet_sha256=packet_sha256,
        b4_context=None,
        b4_rows_shown=None,
        b4_rows_total=None,
        calls=call_list,
        det_ms=det_ms,
        esc_prep_ms=esc_prep_ms,
        cost_billed_usd=billed,
        cost_conservative_usd=billed + ambig,
        latency_ms=det_ms + esc_prep_ms + wall,
        spend_usd=spend,
        config_hash=cfg_hash,
    )


def _extract_referenced_tables_ast(sql: str) -> set[str]:
    """Extract all table names referenced anywhere in a SQLite query AST."""
    tree = parse_single_sqlite(sql)
    if tree is None:
        return set()
    from sqlglot import exp

    names: set[str] = set()
    for tbl in tree.find_all(exp.Table):
        if tbl.name:
            names.add(tbl.name)
    return names


def _execute_steps(
    trace: Trace,
    db_path: Path | None,
    cfg: EvalConfig,
) -> tuple[list[Step], set[str], bool]:
    """Execute trace SQL steps lacking cached outputs and gather referenced tables and width overflows."""
    steps_out: list[Step] = []
    referenced: set[str] = set()
    col_overflow = False

    for st in trace.steps:
        vol = is_volatile(st.sql)
        if st.result is not None or st.error_kind is not None:
            if vol:
                res, err = None, "volatile"
            else:
                res, err = st.result, st.error_kind
            referenced |= _extract_referenced_tables_ast(st.sql)
        elif db_path is not None:
            res, err, objs = execute_sandboxed(db_path, st.sql, cfg)
            referenced |= objs
            referenced |= _extract_referenced_tables_ast(st.sql)
        else:
            res, err = None, None
            referenced |= _extract_referenced_tables_ast(st.sql)

        if res is not None and len(res.columns) > cfg.max_columns:
            col_overflow = True
        steps_out.append(Step(index=st.index, sql=st.sql, result=res, error_kind=err))

    return steps_out, referenced, col_overflow


def _evaluate_candidate(
    trace: Trace,
    gt: GroundTruth,
    gt_sql: str,
    gt_res: ResultTable | None,
    steps: list[Step],
    db_path: Path | None,
    schema_ddl: str,
    mode: ModeName,
    cfg: EvalConfig,
    cfg_hash: str,
    method: MethodName,
) -> PreparedCandidate:
    """Run Stage 1 rules, alignment, claims, and derivations for one GT query candidate."""
    gt_vol = is_volatile(gt_sql)
    if gt_vol:
        gt_res = None
    gt_det_flag = is_det(gt_sql)
    gt_rd_flag = is_rd_eligible(gt_sql)
    gt_n = normalize_sql_n(gt_sql)
    n_gt_cols = len(gt_res.columns) if gt_res is not None else 0
    contract = extract_contract(gt_sql, trace.question, n_cols=n_gt_cols)

    step_ns = [normalize_sql_n(st.sql) for st in steps]
    step_vols = [is_volatile(st.sql) for st in steps]
    step_dets = [is_det(st.sql) for st in steps]
    step_rds = [is_rd_eligible(st.sql) for st in steps]

    flags: list[str] = []
    if gt_n is None or any(sn is None and st.sql.strip() != "" for sn, st in zip(step_ns, steps)):
        flags.append("parser_unsupported")

    # Alignment
    row_diff = None
    coincident = False
    step_alignments: list[Alignment] = []
    method_ok: bool | None = None

    if mode == "sql_only":
        gen_step = steps[-1] if steps else None
        gen_res = gen_step.result if gen_step is not None else None
        gen_n = step_ns[-1] if step_ns else None
        gen_det_flag = step_dets[-1] if step_dets else False
        gen_rd_flag = step_rds[-1] if step_rds else False

        alignment, row_diff, align_flags = align_tables(
            gt_res, gen_res, contract, rho=DEFAULT_JUDGE_RHO
        )
        for fl in align_flags:
            if fl not in flags:
                flags.append(fl)

        d1_match = (
            gt_det_flag
            and gen_det_flag
            and gt_n is not None
            and gen_n is not None
            and gt_n == gen_n
        )
        if (
            gt_res is not None
            and gt_res.complete
            and gen_res is not None
            and gen_res.complete
            and alignment.relation == "equal"
            and not d1_match
        ):
            coincident = True
            flags.append("coincident_result")
    else:
        for st in steps:
            al, _, _ = align_tables(gt_res, st.result, contract, rho=DEFAULT_JUDGE_RHO)
            step_alignments.append(al)
        alignment = (
            step_alignments[-1]
            if step_alignments
            else Alignment(
                column_map={},
                map_status="unknown",
                relation="unknown",
                key_columns=None,
                witness_kind="none",
                rows_matched=0,
                rows_only_gt=0,
                rows_only_gen=0,
                order_ok=None,
                scale_match=None,
            )
        )
        successful = [
            (st, al)
            for st, al, sv in zip(steps, step_alignments, step_vols)
            if not sv and st.error_kind is None and st.result is not None and st.result.complete
        ]
        if successful:
            non_disjoint = any(
                al.relation != "disjoint" and not _is_scalar_mismatch(gt_res, st.result, DEFAULT_JUDGE_RHO)
                for st, al in successful
            )
            method_ok = True if non_disjoint else False

    # Claims & derivations for full-trace
    claims = []
    derivations = []
    if mode == "full_trace" and trace.final_answer is not None:
        claims, derivations = extract_and_bind_claims(
            trace.final_answer, trace.question, gt_res, steps
        )

    # Check evidence availability & completeness
    has_any_sql_or_res = any(
        st.sql.strip() != "" or st.result is not None or st.error_kind is not None for st in steps
    )
    if mode == "sql_only":
        no_evidence = not has_any_sql_or_res
        all_complete = (
            gt_res is not None
            and gt_res.complete
            and bool(steps)
            and steps[-1].error_kind is None
            and steps[-1].result is not None
            and steps[-1].result.complete
            and not step_vols[-1]
        )
    else:
        has_ans = trace.final_answer is not None and trace.final_answer.strip() != ""
        no_evidence = (not has_any_sql_or_res) and (not has_ans)
        all_complete = (
            gt_res is not None
            and gt_res.complete
            and not gt_vol
            and all(
                (not sv) and st.error_kind is None and st.result is not None and st.result.complete
                for st, sv in zip(steps, step_vols)
            )
        )

    cand = PreparedCandidate(
        trace=trace,
        gt=gt,
        gt_sql=gt_sql,
        gt_sql_n=gt_n,
        gt_result=gt_res,
        gt_volatile=gt_vol,
        gt_det=gt_det_flag,
        gt_rd_eligible=gt_rd_flag,
        contract=contract,
        schema_ddl=schema_ddl,
        steps=steps,
        step_sql_n=step_ns,
        step_volatile=step_vols,
        step_det=step_dets,
        step_rd_eligible=step_rds,
        alignment=alignment,
        step_alignments=step_alignments,
        row_diff=row_diff,
        coincident_result=coincident,
        claims=claims,
        derivations=derivations,
        flags=flags,
        rule_hit=None,
        method_ok=method_ok,
        no_evidence=no_evidence,
        all_complete=all_complete,
    )

    # Stage 1 deterministic rules in exact priority: D0, D3, D2, D4, D5, D1
    rule_hit: Verdict | None = None

    if mode == "full_trace" and check_d0(trace.final_answer):
        rule_hit = _make_verdict(
            trace.trace_id,
            method,
            mode,
            "INCORRECT",
            "rule:D0",
            "empty_answer",
            cfg_hash,
            flags=flags,
            cand=cand,
        )

    if rule_hit is None and mode == "sql_only" and steps:
        if steps[-1].error_kind in COUNTEREXAMPLE_ERROR_KINDS:
            rule_hit = _make_verdict(
                trace.trace_id,
                method,
                mode,
                "INCORRECT",
                "rule:D3",
                steps[-1].error_kind,
                cfg_hash,
                flags=flags,
                cand=cand,
            )

    if rule_hit is None and check_d2(mode, gt_sql, gt_res, steps):
        rule_hit = _make_verdict(
            trace.trace_id,
            method,
            mode,
            "INCORRECT",
            "rule:D2",
            "no_data",
            cfg_hash,
            flags=flags,
            cand=cand,
        )

    if rule_hit is None and mode == "sql_only" and steps:
        gen_step = steps[-1]
        if gt_rd_flag and step_rds[-1]:
            ok_gt_1x1, c_gt = _is_1x1_numeric(gt_res)
            ok_gen_1x1, c_gen = _is_1x1_numeric(gen_step.result)
            if ok_gt_1x1 and ok_gen_1x1 and not _numeric_1x1_matches_any_scale(
                c_gt, c_gen, DEFAULT_JUDGE_RHO
            ):
                rule_hit = _make_verdict(
                    trace.trace_id,
                    method,
                    mode,
                    "INCORRECT",
                    "rule:D4",
                    "scalar_counterexample",
                    cfg_hash,
                    flags=flags,
                    cand=cand,
                )

    if (
        rule_hit is None
        and mode == "sql_only"
        and cfg.enable_d5
        and db_path is not None
        and steps
        and gt_rd_flag
        and step_rds[-1]
    ):
        with tempfile.TemporaryDirectory(prefix="d5_") as tmpdir:
            d5_db = create_d5_deletion_copy(db_path, Path(tmpdir) / "d5.sqlite")
            if d5_db is not None:
                d5_gt_res, d5_gt_err, _ = execute_sandboxed(d5_db, gt_sql, cfg)
                d5_gen_res, d5_gen_err, _ = execute_sandboxed(d5_db, steps[-1].sql, cfg)
                if d5_gt_err is None and d5_gen_err is None:
                    ok_a, c_a = _is_1x1_numeric(d5_gt_res)
                    ok_b, c_b = _is_1x1_numeric(d5_gen_res)
                    if ok_a and ok_b and not _numeric_1x1_matches_any_scale(
                        c_a, c_b, DEFAULT_JUDGE_RHO
                    ):
                        rule_hit = _make_verdict(
                            trace.trace_id,
                            method,
                            mode,
                            "INCORRECT",
                            "rule:D5",
                            "perturbation_counterexample",
                            cfg_hash,
                            flags=flags,
                            cand=cand,
                        )

    if rule_hit is None and mode == "sql_only" and steps:
        d1_match = (
            gt_det_flag
            and step_dets[-1]
            and gt_n is not None
            and step_ns[-1] is not None
            and gt_n == step_ns[-1]
        )
        gen_res = steps[-1].result
        both_comp = (
            gt_res is not None
            and gt_res.complete
            and gen_res is not None
            and gen_res.complete
            and not gt_vol
            and not step_vols[-1]
        )
        if d1_match:
            if both_comp and alignment.relation != "equal":
                if "d1_result_conflict" not in flags:
                    flags.append("d1_result_conflict")
            else:
                rule_hit = _make_verdict(
                    trace.trace_id,
                    method,
                    mode,
                    "CORRECT",
                    "rule:D1",
                    None,
                    cfg_hash,
                    flags=flags,
                    cand=cand,
                )
        elif (
            method in ("H+E", "H")
            and both_comp
            and gt_det_flag
            and step_dets[-1]
            and alignment.relation in ("equal", "gt_subset_gen")
            and gt_res is not None
            and gen_res is not None
            and len(gt_res.rows) == len(gen_res.rows)
            and is_d1_superset_equivalent(
                gt_sql,
                steps[-1].sql,
                order_required=contract.order_required,
            )
        ):
            rule_hit = _make_verdict(
                trace.trace_id,
                method,
                mode,
                "CORRECT",
                "rule:D1",
                None,
                cfg_hash,
                flags=flags,
                cand=cand,
            )

    cand.rule_hit = rule_hit
    return cand


_RELATION_RANK: dict[str, int] = {
    "equal": 3,
    "gt_subset_gen": 2,
    "gen_subset_gt": 2,
    "overlap": 2,
    "unknown": 1,
    "disjoint": 0,
}


def prepare_best_candidate(
    trace: Trace,
    gt: GroundTruth,
    db_path: Path | None,
    cfg: EvalConfig,
    cfg_hash: str,
    method: MethodName,
) -> tuple[PreparedCandidate | None, bool]:
    """Execute Stage 0 and pick the best GT alternative by determinism, error status, and alignment."""
    mode = mode_of(trace)
    steps, step_refs, step_col_overflow = _execute_steps(trace, db_path, cfg)
    if step_col_overflow:
        return None, True

    gt_chains: list[list[str]] = [gt.sql_chain] + list(gt.alternatives)
    candidates_raw: list[tuple[str, ResultTable | None, set[str]]] = []

    for idx, chain in enumerate(gt_chains):
        if not chain:
            continue
        gt_sql = chain[-1]
        refs = _extract_referenced_tables_ast(gt_sql)
        gt_res: ResultTable | None = None
        if gt.results is not None and idx < len(gt.results):
            gt_res = gt.results[idx]
        elif is_volatile(gt_sql):
            gt_res = None
        elif db_path is not None:
            gt_res, gt_err, exec_refs = execute_sandboxed(db_path, gt_sql, cfg)
            refs |= exec_refs
            if gt_err is not None:
                gt_res = None

        if gt_res is not None and len(gt_res.columns) > cfg.max_columns:
            return None, True
        candidates_raw.append((gt_sql, gt_res, refs))

    if not candidates_raw:
        return None, False

    all_refs = set(step_refs)
    for _, _, r in candidates_raw:
        all_refs |= r

    schema_ddl = extract_schema_ddl(db_path, all_refs) if db_path is not None else ""

    evaluated: list[tuple[int, int, int, int, PreparedCandidate]] = []
    for idx, (gt_sql, gt_res, _) in enumerate(candidates_raw):
        cand = _evaluate_candidate(
            trace=trace,
            gt=gt,
            gt_sql=gt_sql,
            gt_res=gt_res,
            steps=steps,
            db_path=db_path,
            schema_ddl=schema_ddl,
            mode=mode,
            cfg=cfg,
            cfg_hash=cfg_hash,
            method=method,
        )
        if cand.rule_hit is not None:
            rank_outcome = 2 if cand.rule_hit.label == "CORRECT" else 0
        else:
            rank_outcome = 1
        if mode == "full_trace" and cand.step_alignments:
            rel_rank = max(
                (
                    _RELATION_RANK.get(al.relation, 0)
                    for st, al in zip(cand.steps, cand.step_alignments)
                    if st.error_kind is None and st.result is not None and st.result.complete
                ),
                default=_RELATION_RANK.get(cand.alignment.relation, 0),
            )
        else:
            rel_rank = _RELATION_RANK.get(cand.alignment.relation, 0)
        mapped_count = sum(1 for v in cand.alignment.column_map.values() if v is not None)
        evaluated.append((rank_outcome, rel_rank, mapped_count, -idx, cand))

    evaluated.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    return evaluated[0][4], False


def questions_for(mode: ModeName, cand: PreparedCandidate) -> list[str]:
    """Select the ordered Jev rubric question IDs required for the evaluation mode and table relation."""
    if mode == "sql_only":
        qs = ["Q1"]
        if cand.alignment.relation in {"gt_subset_gen", "gen_subset_gt", "overlap"}:
            qs.append("Q2")
        qs.append("Q5")
        return qs
    return ["Q3", "Q4", "Q6", "Q5"]


def _pick_exclusive_choice(probs: dict[str, float], tau_hi: float) -> str | None:
    """Return the unique highest-probability option meeting `tau_hi`, or None on tie/low confidence."""
    if not probs:
        return None
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_opt, top_p = ordered[0]
    if top_p < tau_hi:
        return None
    if len(ordered) > 1 and abs(ordered[1][1] - top_p) <= 1e-12:
        return None
    return top_opt


def sql_only_verdict_table(
    cand: PreparedCandidate,
    answers: dict[str, Any],
    tau: ThresholdPair,
) -> tuple[VerdictLabel | str, str | None]:
    """Map Jev SQL-only question probabilities (`Q1`, optional `Q2`, `Q5`) to a verdict or ESCALATE."""
    q1_p = float(answers["Q1"])
    q1_yes = q1_p >= tau.tau_hi
    q1_no = q1_p <= tau.tau_lo

    result_ok: bool | None
    if "Q2" in answers:
        q2_probs: dict[str, float] = answers["Q2"]
        good_set = {"equivalent_for_question", "acceptable_extra_rows"}
        bad_set = {"missing_required_rows", "wrong_rows"}
        q2_good = any(q2_probs.get(opt, 0.0) >= tau.tau_hi for opt in good_set) and not any(
            q2_probs.get(opt, 0.0) >= tau.tau_hi for opt in bad_set
        )
        q2_bad = any(q2_probs.get(opt, 0.0) >= tau.tau_hi for opt in bad_set) and not any(
            q2_probs.get(opt, 0.0) >= tau.tau_hi for opt in good_set
        )
        if q1_yes and q2_good:
            result_ok = True
        elif q1_no or q2_bad:
            result_ok = False
        else:
            result_ok = None
    else:
        if q1_yes:
            result_ok = True
        elif q1_no:
            result_ok = False
        else:
            result_ok = None

    if result_ok is True:
        return "CORRECT", None
    if result_ok is False:
        q5_probs: dict[str, float] = answers["Q5"]
        if _pick_exclusive_choice(q5_probs, tau.tau_hi) == "ambiguous_generated_reasonable":
            return "AMBIGUOUS", None
        if _pick_exclusive_choice(q5_probs, 0.0) == "ambiguous_generated_reasonable":
            return "ESCALATE", "uncertain_or_conflict"
        return "INCORRECT", None
    return "ESCALATE", "uncertain_or_conflict"


def full_trace_verdict_table(
    cand: PreparedCandidate | None,
    answers: dict[str, Any],
    tau: ThresholdPair,
) -> tuple[VerdictLabel | str, str | None, list[str]]:
    """Map Jev full-trace question probabilities (`Q3`, `Q4`, `Q6`, `Q5`) to a verdict or ESCALATE."""
    q3 = float(answers["Q3"])
    q4 = float(answers["Q4"])
    q6 = float(answers["Q6"])

    q3_yes, q4_yes, q6_yes = q3 >= tau.tau_hi, q4 >= tau.tau_hi, q6 >= tau.tau_hi
    q3_no, q4_no, q6_no = q3 <= tau.tau_lo, q4 <= tau.tau_lo, q6 <= tau.tau_lo

    extra_flags = list(cand.flags) if cand is not None else []

    if q3_yes and q4_yes and q6_yes:
        if cand is not None:
            successful = [
                (st, al)
                for st, al, sv in zip(cand.steps, cand.step_alignments, cand.step_volatile)
                if not sv and st.error_kind is None and st.result is not None and st.result.complete
            ]
            if successful:
                all_conflict = all(
                    al.relation == "disjoint"
                    or _is_scalar_mismatch(cand.gt_result, st.result, DEFAULT_JUDGE_RHO)
                    for st, al in successful
                )
                if all_conflict:
                    return "ESCALATE", "verdict_conflict", extra_flags
            if cand.method_ok is not True and "method_unverified" not in extra_flags:
                extra_flags.append("method_unverified")
        return "CORRECT", None, extra_flags

    if q3_no or q4_no or q6_no:
        if q6_no and (not q3_no) and (not q4_no):
            return "INCORRECT", "unsupported", extra_flags
        q5_probs: dict[str, float] = answers["Q5"]
        if _pick_exclusive_choice(q5_probs, tau.tau_hi) == "ambiguous_generated_reasonable":
            return "AMBIGUOUS", None, extra_flags
        if _pick_exclusive_choice(q5_probs, 0.0) == "ambiguous_generated_reasonable":
            return "ESCALATE", "uncertain_or_conflict", extra_flags
        return "INCORRECT", None, extra_flags

    return "ESCALATE", "uncertain_or_conflict", extra_flags


def single_verdict_v(
    answers: dict[str, Any],
    tau: ThresholdPair,
) -> VerdictLabel | str:
    """Map single-question baseline `V` probabilities (`B6` / `B6+E`) to a discrete verdict or ESCALATE."""
    probs: dict[str, float] = answers["V"]
    choice = _pick_exclusive_choice(probs, tau.tau_hi)
    if choice == "correct":
        return "CORRECT"
    if choice == "incorrect":
        return "INCORRECT"
    if choice == "ambiguous_reasonable":
        return "AMBIGUOUS"
    return "ESCALATE"


def judge_item(
    trace: Trace,
    gt: GroundTruth,
    method: MethodName,
    ctx: CallContext,
    *,
    db_path: Path | None = None,
    escalation: bool = True,
) -> Verdict:
    """Run the hybrid funnel judge for H+E, H, B6, or B6+E."""
    if method == "H":
        he_v = judge_item(trace, gt, "H+E", ctx, db_path=db_path, escalation=escalation)
        return derive_h_from_he(he_v)

    cfg = ctx.cfg
    cfg_hash = cfg.config_hash(PROMPT_HASH)
    mode = mode_of(trace)

    t_det0 = time.monotonic()
    if over_input_limits(trace, cfg):
        det_ms = max(0, int(round((time.monotonic() - t_det0) * 1000)))
        return _make_abstain(trace.trace_id, method, mode, "input_over_limit", cfg_hash, det_ms=det_ms)

    if mode == "full_trace" and check_d0(trace.final_answer):
        det_ms = max(0, int(round((time.monotonic() - t_det0) * 1000)))
        return _make_verdict(
            trace.trace_id,
            method,
            mode,
            "INCORRECT",
            "rule:D0",
            "empty_answer",
            cfg_hash,
            det_ms=det_ms,
        )

    cand, overflow = prepare_best_candidate(trace, gt, db_path, cfg, cfg_hash, method)
    det_ms = max(0, int(round((time.monotonic() - t_det0) * 1000)))

    if overflow:
        return _make_abstain(trace.trace_id, method, mode, "input_over_limit", cfg_hash, det_ms=det_ms)

    if cand is None or cand.no_evidence:
        return _make_abstain(trace.trace_id, method, mode, "no_evidence", cfg_hash, det_ms=det_ms, cand=cand)

    if cand.rule_hit is not None:
        return replace(cand.rule_hit, det_ms=det_ms, latency_ms=det_ms)

    if mode == "full_trace" and not cand.all_complete:
        return _make_abstain(
            trace.trace_id,
            method,
            mode,
            "incomplete_evidence",
            cfg_hash,
            det_ms=det_ms,
            cand=cand,
        )

    single = method in ("B6", "B6+E")
    tau_method_key = "B6" if single else "H"
    tau = cfg.tau.get(tau_method_key, {}).get(mode, ThresholdPair())

    packet = build_packet(cand, mode, "jev", cfg)
    calls: list[ModelCall] = []
    esc_reason = "exceeds_jev_tier"

    if packet.required_met:
        qs = ["V"] if single else questions_for(mode, cand)
        jev_req = {
            "state": packet.state,
            "questions": {qname: RUBRICS[qname].to_jev_spec() for qname in qs},
        }
        jev_call = call_model("jev", "jev", cfg.jev_model, jev_req, qs, ctx)
        calls.append(jev_call)
        if jev_call.outcome != "ok" or jev_call.answers is None:
            return _make_abstain(
                trace.trace_id,
                method,
                mode,
                jev_call.outcome,
                cfg_hash,
                calls=calls,
                det_ms=det_ms,
                cand=cand,
                packet_sha256=packet.packet_sha256,
            )

        out_flags = list(cand.flags)
        v_reason: str | None = None
        if single:
            dec = single_verdict_v(jev_call.answers, tau)
        elif mode == "sql_only":
            dec, v_reason = sql_only_verdict_table(cand, jev_call.answers, tau)
        else:
            dec, v_reason, out_flags = full_trace_verdict_table(cand, jev_call.answers, tau)

        if dec != "ESCALATE":
            return _make_verdict(
                trace.trace_id,
                method,
                mode,
                dec,  # type: ignore[arg-type]
                "jev",
                v_reason,
                cfg_hash,
                flags=out_flags,
                calls=calls,
                det_ms=det_ms,
                cand=cand,
                packet_sha256=packet.packet_sha256,
            )
        esc_reason = v_reason or "uncertain_or_conflict"

    if method == "B6" or (not escalation):
        abst_reason = "uncertain" if packet.required_met else "evidence_exceeds_bound"
        return _make_abstain(
            trace.trace_id,
            method,
            mode,
            abst_reason,
            cfg_hash,
            calls=calls,
            det_ms=det_ms,
            cand=cand,
            packet_sha256=packet.packet_sha256 if packet.required_met else None,
        )

    # Stage 4: Escalation
    t_esc0 = time.monotonic()
    big = build_packet(cand, mode, "escalation", cfg)
    esc_prep_ms = max(0, int(round((time.monotonic() - t_esc0) * 1000)))
    if not big.required_met:
        return _make_abstain(
            trace.trace_id,
            method,
            mode,
            "evidence_exceeds_bound",
            cfg_hash,
            calls=calls,
            det_ms=det_ms,
            esc_prep_ms=esc_prep_ms,
            escalated=True,
            cand=cand,
            packet_sha256=packet.packet_sha256 if packet.required_met else None,
        )

    esc_qs = questions_for(mode, cand)
    esc_req = {
        "instructions": escalation_instructions(mode, esc_qs),
        "state": big.state,
        "questions": esc_qs,
        "temperature": 0.0,
        "max_output_tokens": 512,
    }
    esc_call = call_model(
        "escalation", "generative", cfg.generative_model, esc_req, esc_qs, ctx
    )
    calls.append(esc_call)
    pkt_hash = packet.packet_sha256 if packet.required_met else big.packet_sha256

    if esc_call.outcome != "ok" or esc_call.answers is None:
        return _make_abstain(
            trace.trace_id,
            method,
            mode,
            esc_call.outcome,
            cfg_hash,
            calls=calls,
            det_ms=det_ms,
            esc_prep_ms=esc_prep_ms,
            escalated=True,
            cand=cand,
            packet_sha256=pkt_hash,
        )

    esc_label: VerdictLabel = esc_call.answers["label"]
    esc_flags = list(cand.flags)
    if (
        mode == "full_trace"
        and esc_label == "CORRECT"
        and cand.method_ok is not True
        and "method_unverified" not in esc_flags
    ):
        esc_flags.append("method_unverified")

    return _make_verdict(
        trace.trace_id,
        method,
        mode,
        esc_label,
        "escalation",
        esc_reason,
        cfg_hash,
        escalated=True,
        flags=esc_flags,
        calls=calls,
        det_ms=det_ms,
        esc_prep_ms=esc_prep_ms,
        cand=cand,
        packet_sha256=pkt_hash,
    )


def derive_h_from_he(he_verdict: Verdict) -> Verdict:
    """Derive the un-escalated H verdict strictly from an H+E Verdict without making new model calls."""
    if not he_verdict.escalated:
        return replace(he_verdict, method="H")

    non_esc_calls = [c for c in he_verdict.calls if c.stage != "escalation"]
    had_jev = any(c.stage == "jev" for c in non_esc_calls)
    h_reason = "uncertain" if had_jev else "evidence_exceeds_bound"

    billed = sum(c.cold.billed_usd for c in non_esc_calls)
    ambig = sum(c.cold.ambiguous_usd for c in non_esc_calls)
    spend = sum(
        c.cold.billed_usd + c.cold.ambiguous_usd
        for c in non_esc_calls
        if not c.from_cache
    )
    wall = sum(c.cold.wall_ms for c in non_esc_calls)

    return Verdict(
        trace_id=he_verdict.trace_id,
        method="H",
        mode=he_verdict.mode,
        label="ABSTAIN",
        decided_by="abstain",
        reason=h_reason,
        escalated=False,
        method_ok=he_verdict.method_ok,
        flags=list(he_verdict.flags),
        claims=list(he_verdict.claims),
        alignments=list(he_verdict.alignments),
        packet_sha256=he_verdict.packet_sha256 if had_jev else None,
        b4_context=None,
        b4_rows_shown=None,
        b4_rows_total=None,
        calls=non_esc_calls,
        det_ms=he_verdict.det_ms,
        esc_prep_ms=0,
        cost_billed_usd=billed,
        cost_conservative_usd=billed + ambig,
        latency_ms=he_verdict.det_ms + wall,
        spend_usd=spend,
        config_hash=he_verdict.config_hash,
    )
