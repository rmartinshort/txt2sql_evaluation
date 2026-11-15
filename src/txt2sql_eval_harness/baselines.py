"""Baseline judges: B1, B2, B-IBM-subset, B-IBM-judge, and B4."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
from pathlib import Path
import time
from typing import Any
import unicodedata

from txt2sql_eval_harness.alignment import align_tables
from txt2sql_eval_harness.clients import CallContext, call_model
from txt2sql_eval_harness.judge import (
    _execute_steps,
    _extract_referenced_tables_ast,
    _make_abstain,
    _make_verdict,
    mode_of,
    over_input_limits,
)
from txt2sql_eval_harness.models import (
    Cell,
    EvalConfig,
    GroundTruth,
    MethodName,
    ResultTable,
    Step,
    Trace,
    Verdict,
)
from txt2sql_eval_harness.renderer import count_tokens, encode_packet_cell, render_state
from txt2sql_eval_harness.rubrics import B4_INSTRUCTIONS, PROMPT_HASH
from txt2sql_eval_harness.sandbox import execute_sandboxed
from txt2sql_eval_harness.sql_analysis import extract_schema_ddl, is_volatile

IBM_JUDGE_TEMPLATE = (
    "You are grading a text-to-SQL prediction. Decide whether the predicted SQL and execution "
    "result answer the question equivalently to the ground-truth SQL and result.\n"
    "Respond with one verdict word on a line: Verdict: Yes, Verdict: No, or Verdict: Maybe.\n\n"
    "Question:\n{question}\n\n"
    "Generation Prompt / Hint:\n{generation_prompt}\n\n"
    "Ground Truth SQL:\n{ground_truth_sql}\n\n"
    "Ground Truth Result:\n{ground_truth_df}\n\n"
    "Predicted SQL / Trace:\n{predicted_sql}\n\n"
    "Predicted Result:\n{predicted_df}\n"
)


def _norm_sql_text(sql: str) -> str:
    """Normalize SQL whitespace and casing for lightweight syntactic matching."""
    return " ".join(sql.strip().lower().split())


def _canonical_bird_scalar(val: Cell) -> Any:
    """Canonicalize a cell for BIRD baseline equality (rounding floats to 6 decimal places)."""
    if val is None:
        return None
    if isinstance(val, int) and not isinstance(val, bool):
        return ("num", float(val))
    if isinstance(val, float):
        if not math.isfinite(val):
            return ("num", str(val))
        return ("num", round(val, 6))
    if isinstance(val, str):
        s = unicodedata.normalize("NFC", val).strip()
        try:
            f = float(s)
            if math.isfinite(f):
                return ("num", round(f, 6))
        except ValueError:
            pass
        return ("str", s)
    if isinstance(val, (bytes, bytearray)):
        return ("bytes", bytes(val))
    return ("other", str(val))


def _canon_ibm_scalar(val: Cell) -> Any:
    """Equivalent to IBM text2sql-eval-toolkit _canonical_scalar."""
    if val is None:
        return ("none", None)
    if isinstance(val, bool):
        return ("str", str(val))
    if isinstance(val, int):
        return ("num", int(val))
    if isinstance(val, float):
        if not math.isfinite(val):
            return ("str", str(val))
        if val.is_integer():
            return ("num", int(val))
        return ("num", round(val, 6))
    if isinstance(val, str):
        s = unicodedata.normalize("NFC", val).strip()
        try:
            f = float(s)
            if math.isfinite(f):
                return ("num", int(f)) if f.is_integer() else ("num", round(f, 6))
        except ValueError:
            pass
        return ("str", s)
    return ("bytes", bytes(val)) if isinstance(val, (bytes, bytearray)) else ("other", str(val))


def _resolve_gt_and_gen(
    trace: Trace,
    gt: GroundTruth,
    db_path: Path | None,
    cfg: EvalConfig,
) -> tuple[list[tuple[str, ResultTable | None]], list[Step], str, bool]:
    """Execute Stage 0 consistently for baselines."""
    steps, step_refs, col_overflow = _execute_steps(trace, db_path, cfg)
    if col_overflow:
        return [], steps, "", True

    gt_chains: list[list[str]] = [gt.sql_chain] + list(gt.alternatives)
    gt_pairs: list[tuple[str, ResultTable | None]] = []
    all_refs = set(step_refs)

    for idx, chain in enumerate(gt_chains):
        if not chain:
            continue
        gt_sql = chain[-1]
        all_refs |= _extract_referenced_tables_ast(gt_sql)
        gt_res: ResultTable | None = None
        if gt.results is not None and idx < len(gt.results):
            gt_res = gt.results[idx]
        elif is_volatile(gt_sql):
            gt_res = None
        elif db_path is not None:
            gt_res, err, exec_refs = execute_sandboxed(db_path, gt_sql, cfg)
            all_refs |= exec_refs
            if err is not None:
                gt_res = None
        if gt_res is not None and len(gt_res.columns) > cfg.max_columns:
            return [], steps, "", True
        gt_pairs.append((gt_sql, gt_res))

    schema_ddl = (
        extract_schema_ddl(db_path, all_refs, include_validity_note=True)
        if db_path is not None
        else ""
    )
    return gt_pairs, steps, schema_ddl, False


def judge_b1(
    trace: Trace,
    gt: GroundTruth,
    cfg: EvalConfig,
) -> Verdict:
    """B1: Exact SQL string match after case/whitespace normalization (SQL-only)."""
    t0 = time.monotonic()
    cfg_hash = cfg.config_hash(PROMPT_HASH)
    mode = mode_of(trace)
    if over_input_limits(trace, cfg):
        det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
        return _make_abstain(trace.trace_id, "B1", mode, "input_over_limit", cfg_hash, det_ms=det_ms)
    if mode != "sql_only":
        det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
        return _make_abstain(trace.trace_id, "B1", mode, "unsupported_mode", cfg_hash, det_ms=det_ms)
    if not trace.steps or not trace.steps[-1].sql.strip():
        det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
        return _make_abstain(trace.trace_id, "B1", mode, "no_evidence", cfg_hash, det_ms=det_ms)

    gen_norm = _norm_sql_text(trace.steps[-1].sql)
    gt_chains = [gt.sql_chain] + list(gt.alternatives)
    matched = any(chain and _norm_sql_text(chain[-1]) == gen_norm for chain in gt_chains)
    det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
    return _make_verdict(
        trace.trace_id,
        "B1",
        mode,
        "CORRECT" if matched else "INCORRECT",
        "baseline",
        None,
        cfg_hash,
        det_ms=det_ms,
    )


def judge_b2(
    trace: Trace,
    gt: GroundTruth,
    cfg: EvalConfig,
    *,
    db_path: Path | None = None,
) -> Verdict:
    """B2: BIRD-style execution accuracy (column-order & column-name invariant row bag match)."""
    t0 = time.monotonic()
    cfg_hash = cfg.config_hash(PROMPT_HASH)
    mode = mode_of(trace)
    if over_input_limits(trace, cfg):
        det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
        return _make_abstain(trace.trace_id, "B2", mode, "input_over_limit", cfg_hash, det_ms=det_ms)

    gt_pairs, steps, _, overflow = _resolve_gt_and_gen(trace, gt, db_path, cfg)
    if overflow:
        det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
        return _make_abstain(trace.trace_id, "B2", mode, "input_over_limit", cfg_hash, det_ms=det_ms)

    has_any_step = any(
        st.sql.strip() != "" or st.result is not None or st.error_kind is not None for st in steps
    )
    if not has_any_step:
        det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
        reason = "no_evidence" if trace.final_answer is None else "no_sql_step"
        return _make_abstain(trace.trace_id, "B2", mode, reason, cfg_hash, det_ms=det_ms)

    gen_step = steps[-1]
    gen_res = gen_step.result
    det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))

    if gen_step.error_kind is not None or gen_res is None:
        return _make_verdict(
            trace.trace_id,
            "B2",
            mode,
            "INCORRECT",
            "baseline",
            gen_step.error_kind or "no_result",
            cfg_hash,
            det_ms=det_ms,
        )

    complete_gt = [gt_res for _, gt_res in gt_pairs if gt_res is not None and gt_res.complete]
    if not gen_res.complete or not complete_gt:
        return _make_abstain(
            trace.trace_id,
            "B2",
            mode,
            "incomplete_evidence",
            cfg_hash,
            det_ms=det_ms,
        )

    def _bird_equal(a: ResultTable, b: ResultTable) -> bool:
        """Compare two tables using BIRD's unordered bag-of-rows canonical equality."""
        if len(a.columns) != len(b.columns) or len(a.rows) != len(b.rows):
            return False
        if not a.rows:
            return True
        n_cols = len(a.columns)
        a_cols = [
            [_canonical_bird_scalar(row[c]) for row in a.rows]
            for c in range(n_cols)
        ]
        b_cols = [
            [_canonical_bird_scalar(row[c]) for row in b.rows]
            for c in range(n_cols)
        ]
        # Check column-bag compatibility, then row-set equality via column permutation/greedy
        used_b: set[int] = set()
        col_perm: list[int] = []
        a_sorted = [Counter(col) for col in a_cols]
        b_sorted = [Counter(col) for col in b_cols]
        for ac in a_sorted:
            found = None
            for bi, bc in enumerate(b_sorted):
                if bi not in used_b and ac == bc:
                    found = bi
                    break
            if found is None:
                return False
            used_b.add(found)
            col_perm.append(found)
        a_rows = Counter(
            tuple(_canonical_bird_scalar(c) for c in row) for row in a.rows
        )
        b_rows = Counter(
            tuple(_canonical_bird_scalar(row[c]) for c in col_perm) for row in b.rows
        )
        return a_rows == b_rows

    matched = any(_bird_equal(gt_res, gen_res) for gt_res in complete_gt)
    return _make_verdict(
        trace.trace_id,
        "B2",
        mode,
        "CORRECT" if matched else "INCORRECT",
        "baseline",
        None,
        cfg_hash,
        det_ms=det_ms,
    )


def _ibm_subset_match(gt: ResultTable, gen: ResultTable) -> bool:
    """Reproduce IBM toolkit compare_dfs_ignore_colnames_subset logic cleanly."""
    if len(gt.rows) == 0:
        return len(gen.rows) == 0
    if len(gt.rows) > len(gen.rows) or len(gt.columns) > len(gen.columns):
        return False

    gt_cols_canon = [
        [_canon_ibm_scalar(row[c]) for row in gt.rows] for c in range(len(gt.columns))
    ]
    gen_cols_canon = [
        [_canon_ibm_scalar(row[c]) for row in gen.rows] for c in range(len(gen.columns))
    ]

    # Backtrack column assignment where every GT column multiset is a sub-multiset of a distinct gen column
    gt_counters = [Counter(col) for col in gt_cols_canon]
    gen_counters = [Counter(col) for col in gen_cols_canon]

    eligible: list[list[int]] = []
    for gc in gt_counters:
        cand_cols = [
            j
            for j, jc in enumerate(gen_counters)
            if all(jc[k] >= cnt for k, cnt in gc.items())
        ]
        if not cand_cols:
            return False
        eligible.append(cand_cols)

    # Order GT columns by fewest candidates for fast pruning
    order = sorted(range(len(gt.columns)), key=lambda i: len(eligible[i]))
    used: set[int] = set()
    assigned: list[int] = [-1] * len(gt.columns)

    def _search(pos: int, attempts: list[int]) -> bool:
        """Backtrack over candidate generated columns to find a multiset-subset column mapping."""
        if attempts[0] >= 128:
            return False
        if pos == len(order):
            attempts[0] += 1
            gt_rows_cnt = Counter(
                tuple(_canon_ibm_scalar(c) for c in row) for row in gt.rows
            )
            gen_proj_cnt = Counter(
                tuple(_canon_ibm_scalar(row[assigned[g]]) for g in range(len(gt.columns)))
                for row in gen.rows
            )
            return all(gen_proj_cnt[r] >= cnt for r, cnt in gt_rows_cnt.items())
        g_idx = order[pos]
        for j in eligible[g_idx]:
            if j in used:
                continue
            used.add(j)
            assigned[g_idx] = j
            if _search(pos + 1, attempts):
                return True
            used.remove(j)
        return False

    return _search(0, [0])


def judge_b_ibm_subset(
    trace: Trace,
    gt: GroundTruth,
    cfg: EvalConfig,
    *,
    db_path: Path | None = None,
) -> Verdict:
    """B-IBM-subset: IBM toolkit subset/superset execution match."""
    t0 = time.monotonic()
    cfg_hash = cfg.config_hash(PROMPT_HASH)
    mode = mode_of(trace)
    if over_input_limits(trace, cfg):
        det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
        return _make_abstain(
            trace.trace_id, "B-IBM-subset", mode, "input_over_limit", cfg_hash, det_ms=det_ms
        )

    gt_pairs, steps, _, overflow = _resolve_gt_and_gen(trace, gt, db_path, cfg)
    if overflow:
        det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
        return _make_abstain(
            trace.trace_id, "B-IBM-subset", mode, "input_over_limit", cfg_hash, det_ms=det_ms
        )

    has_any = any(
        st.sql.strip() != "" or st.result is not None or st.error_kind is not None for st in steps
    )
    if not has_any:
        det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
        return _make_abstain(
            trace.trace_id, "B-IBM-subset", mode, "no_evidence", cfg_hash, det_ms=det_ms
        )

    gen_step = steps[-1]
    det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
    if gen_step.error_kind is not None or gen_step.result is None:
        return _make_verdict(
            trace.trace_id,
            "B-IBM-subset",
            mode,
            "INCORRECT",
            "baseline",
            gen_step.error_kind or "no_result",
            cfg_hash,
            det_ms=det_ms,
        )

    complete_gt = [gt_res for _, gt_res in gt_pairs if gt_res is not None and gt_res.complete]
    if not gen_step.result.complete or not complete_gt:
        return _make_abstain(
            trace.trace_id,
            "B-IBM-subset",
            mode,
            "incomplete_evidence",
            cfg_hash,
            det_ms=det_ms,
        )

    matched = any(_ibm_subset_match(gt_res, gen_step.result) for gt_res in complete_gt)
    return _make_verdict(
        trace.trace_id,
        "B-IBM-subset",
        mode,
        "CORRECT" if matched else "INCORRECT",
        "baseline",
        None,
        cfg_hash,
        det_ms=det_ms,
    )


def _format_ibm_table(table: ResultTable | None) -> str:
    """Format a ResultTable into a bounded Markdown table preview for the IBM judge prompt."""
    if table is None:
        return "<unavailable>"
    cols = list(table.columns)
    total = len(table.rows)
    if total > 20:
        shown = list(table.rows[:10]) + [tuple("..." for _ in cols)] + list(table.rows[-10:])
    else:
        shown = list(table.rows)
    lines = [" | ".join(str(c) for c in cols)]
    lines.append(" | ".join("---" for _ in cols))
    for r in shown:
        lines.append(" | ".join("NULL" if x is None else str(x) for x in r))
    if not table.complete and table.total_rows is not None:
        lines.append(f"(prefix: {total} of {table.total_rows} rows)")
    return "\n".join(lines)


def _format_ibm_agent_trace(steps: list[Step], final_answer: str | None) -> str:
    """Format trace SQL steps and optional final answer into bounded IBM judge context lines."""
    chunks: list[str] = []
    for st in steps:
        msg = f"Step {st.index}: SQL: {st.sql[:500]}"
        if st.error_kind:
            msg += f" -> ERROR: {st.error_kind}"
        chunks.append(msg[:500])
    if final_answer is not None:
        chunks.append(f"Final Answer: {final_answer[:500]}")
    rendered = "\n".join(chunks)
    return rendered[:40_000]


def judge_b_ibm_judge(
    trace: Trace,
    gt: GroundTruth,
    ctx: CallContext,
    *,
    db_path: Path | None = None,
) -> Verdict:
    """B-IBM-judge: IBM toolkit v1.6.0 unframed prompt routed through call_model('ibm_judge', ...)."""
    t0 = time.monotonic()
    cfg = ctx.cfg
    cfg_hash = cfg.config_hash(PROMPT_HASH)
    mode = mode_of(trace)

    if over_input_limits(trace, cfg):
        det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
        return _make_abstain(
            trace.trace_id, "B-IBM-judge", mode, "input_over_limit", cfg_hash, det_ms=det_ms
        )

    gt_pairs, steps, schema_ddl, overflow = _resolve_gt_and_gen(trace, gt, db_path, cfg)
    if overflow:
        det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
        return _make_abstain(
            trace.trace_id, "B-IBM-judge", mode, "input_over_limit", cfg_hash, det_ms=det_ms
        )

    has_any = any(
        st.sql.strip() != "" or st.result is not None or st.error_kind is not None for st in steps
    ) or (trace.final_answer is not None and trace.final_answer.strip() != "")
    if not has_any or not gt_pairs:
        det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))
        return _make_abstain(
            trace.trace_id, "B-IBM-judge", mode, "no_evidence", cfg_hash, det_ms=det_ms
        )

    gt_sql, gt_res = gt_pairs[0]
    predicted_sql = (
        steps[-1].sql
        if mode == "sql_only" and steps
        else _format_ibm_agent_trace(steps, trace.final_answer)
    )
    predicted_df = _format_ibm_table(steps[-1].result if steps else None)
    generation_prompt = f"Hint: {trace.hint}\nSchema:\n{schema_ddl}"
    prompt_text = IBM_JUDGE_TEMPLATE.format(
        question=trace.question,
        generation_prompt=generation_prompt,
        ground_truth_sql=gt_sql,
        ground_truth_df=_format_ibm_table(gt_res),
        predicted_sql=predicted_sql,
        predicted_df=predicted_df,
    )
    det_ms = max(0, int(round((time.monotonic() - t0) * 1000)))

    req = {
        "prompt": prompt_text,
        "temperature": 0.0,
        "max_output_tokens": 512,
    }
    call = call_model("ibm_judge", "ibm_judge", cfg.generative_model, req, [], ctx)
    if call.outcome != "ok" or call.answers is None:
        return _make_abstain(
            trace.trace_id,
            "B-IBM-judge",
            mode,
            call.outcome,
            cfg_hash,
            calls=[call],
            det_ms=det_ms,
        )

    verdict_str = call.answers["verdict"]
    if verdict_str == "Maybe":
        return _make_abstain(
            trace.trace_id,
            "B-IBM-judge",
            mode,
            "maybe",
            cfg_hash,
            calls=[call],
            det_ms=det_ms,
        )
    lbl = "CORRECT" if verdict_str == "Yes" else "INCORRECT"
    return _make_verdict(
        trace.trace_id,
        "B-IBM-judge",
        mode,
        lbl,  # type: ignore[arg-type]
        "baseline",
        None,
        cfg_hash,
        calls=[call],
        det_ms=det_ms,
    )


def _encode_b4_table(table: ResultTable | None, row_limit: int | None) -> tuple[dict[str, Any] | None, int, int]:
    """Render table for B4, cutting to row_limit if specified and computing (shown, total_fetched)."""
    if table is None:
        return None, 0, 0
    avail = len(table.rows)
    if row_limit is None or avail <= row_limit:
        obj: dict[str, Any] = {
            "columns": list(table.columns),
            "rows": [[encode_packet_cell(c) for c in row] for row in table.rows],
        }
        if not table.complete:
            prefix_d: dict[str, Any] = {
                "n_fetched": avail,
                "total_rows": table.total_rows,
            }
            if table.cut_by is not None:
                prefix_d["cut_by"] = table.cut_by
            obj["prefix"] = prefix_d
        return obj, avail, avail

    cut_rows = table.rows[:row_limit]
    obj = {
        "columns": list(table.columns),
        "rows": [[encode_packet_cell(c) for c in row] for row in cut_rows],
        "prefix": {
            "n_fetched": len(cut_rows),
            "total_rows": table.total_rows if table.total_rows is not None else avail,
            "cut_by": "b4_context",
        },
    }
    return obj, len(cut_rows), avail


def judge_b4(
    trace: Trace,
    gt: GroundTruth,
    ctx: CallContext,
    *,
    db_path: Path | None = None,
) -> Verdict:
    """B4: Frontier generative judge with full context and binary-search equal-prefix truncation."""
    t0 = ctx.clock_fn()
    cfg = ctx.cfg
    cfg_hash = cfg.config_hash(PROMPT_HASH)
    mode = mode_of(trace)

    if over_input_limits(trace, cfg):
        det_ms = max(0, int(round((ctx.clock_fn() - t0) * 1000)))
        return _make_abstain(trace.trace_id, "B4", mode, "input_over_limit", cfg_hash, det_ms=det_ms)

    gt_pairs, steps, schema_ddl, overflow = _resolve_gt_and_gen(trace, gt, db_path, cfg)
    if overflow:
        det_ms = max(0, int(round((ctx.clock_fn() - t0) * 1000)))
        return _make_abstain(trace.trace_id, "B4", mode, "input_over_limit", cfg_hash, det_ms=det_ms)

    has_sql_or_step = any(
        st.sql.strip() != "" or st.result is not None or st.error_kind is not None for st in steps
    )
    has_answer = trace.final_answer is not None and trace.final_answer.strip() != ""
    if (not has_sql_or_step) and (not has_answer):
        det_ms = max(0, int(round((ctx.clock_fn() - t0) * 1000)))
        return _make_abstain(trace.trace_id, "B4", mode, "no_evidence", cfg_hash, det_ms=det_ms)

    gt_sql, gt_res = gt_pairs[0] if gt_pairs else (gt.sql_chain[-1] if gt.sql_chain else "", None)

    def _build_state_for_k(k: int | None) -> tuple[str, int, int, int]:
        """Serialize the B4 judge prompt payload capped at at most `k` rows per table."""
        shown_total = 0
        avail_total = 0

        gt_tbl_dict, s_gt, a_gt = _encode_b4_table(gt_res, k)
        shown_total += s_gt
        avail_total += a_gt

        obj: dict[str, Any] = {
            "QUESTION": trace.question,
            "HINT": trace.hint,
            "SCHEMA_DDL": schema_ddl,
            "REFERENCE_SQL": gt_sql,
            "REFERENCE_RESULT": gt_tbl_dict,
        }

        if mode == "sql_only":
            gen_st = steps[-1]
            gen_vol = is_volatile(gen_st.sql)
            gen_res = None if gen_vol else gen_st.result
            gen_tbl_dict, s_g, a_g = _encode_b4_table(gen_res, k)
            shown_total += s_g
            avail_total += a_g
            obj["GENERATED_SQL"] = gen_st.sql
            obj["GENERATED_RESULT"] = (
                gen_tbl_dict
                if gen_tbl_dict is not None
                else {
                    "available": False,
                    "error_kind": "volatile" if gen_vol else gen_st.error_kind,
                }
            )
        else:
            agent_list: list[dict[str, Any]] = []
            for st in steps:
                st_vol = is_volatile(st.sql)
                st_res = None if st_vol else st.result
                st_tbl_dict, s_st, a_st = _encode_b4_table(st_res, k)
                shown_total += s_st
                avail_total += a_st
                agent_list.append(
                    {
                        "step": st.index,
                        "sql": st.sql,
                        "result": (
                            st_tbl_dict
                            if st_tbl_dict is not None
                            else {
                                "available": False,
                                "error_kind": "volatile" if st_vol else st.error_kind,
                            }
                        ),
                    }
                )
            obj["AGENT_STEPS"] = agent_list
            obj["FINAL_ANSWER"] = trace.final_answer

        state = render_state(obj)
        return state, count_tokens(state), shown_total, avail_total

    full_state, full_tokens, rows_shown, rows_total = _build_state_for_k(None)
    if full_tokens <= cfg.b4_max_tokens:
        b4_state = full_state
        b4_context: str = "full"
    else:
        # Check k = 0 feasibility first
        zero_state, zero_tokens, z_shown, z_avail = _build_state_for_k(0)
        if zero_tokens > cfg.b4_max_tokens:
            det_ms = max(0, int(round((ctx.clock_fn() - t0) * 1000)))
            return _make_abstain(
                trace.trace_id,
                "B4",
                mode,
                "evidence_exceeds_bound",
                cfg_hash,
                det_ms=det_ms,
            )
        max_rows = max(
            [len(gt_res.rows) if gt_res is not None else 0]
            + [
                len(st.result.rows)
                for st in steps
                if st.result is not None and not is_volatile(st.sql)
            ],
            default=0,
        )
        lo, hi = 0, max(0, max_rows - 1)
        best_k = 0
        b4_state = zero_state
        rows_shown, rows_total = z_shown, z_avail
        while lo <= hi:
            mid = (lo + hi) // 2
            st_mid, tok_mid, sh_mid, av_mid = _build_state_for_k(mid)
            if tok_mid <= cfg.b4_max_tokens:
                best_k = mid
                b4_state = st_mid
                rows_shown, rows_total = sh_mid, av_mid
                lo = mid + 1
            else:
                hi = mid - 1
        _ = best_k
        b4_context = "truncated"

    det_ms = max(0, int(round((ctx.clock_fn() - t0) * 1000)))
    req = {
        "instructions": B4_INSTRUCTIONS,
        "state": b4_state,
        "temperature": 0.0,
        "max_output_tokens": 512,
    }
    call = call_model("b4", "generative", cfg.generative_model, req, [], ctx)
    if call.outcome != "ok" or call.answers is None:
        v = _make_abstain(
            trace.trace_id,
            "B4",
            mode,
            call.outcome,
            cfg_hash,
            calls=[call],
            det_ms=det_ms,
        )
        return replace(
            v,
            b4_context=b4_context,  # type: ignore[arg-type]
            b4_rows_shown=rows_shown,
            b4_rows_total=rows_total,
        )

    v = _make_verdict(
        trace.trace_id,
        "B4",
        mode,
        call.answers["label"],
        "baseline",
        None,
        cfg_hash,
        calls=[call],
        det_ms=det_ms,
    )
    return replace(
        v,
        b4_context=b4_context,  # type: ignore[arg-type]
        b4_rows_shown=rows_shown,
        b4_rows_total=rows_total,
    )
