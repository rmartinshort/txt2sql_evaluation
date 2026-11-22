"""Experiment runner: Common subset C, spend projection, threshold fitting, and resumable serial loop."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Callable, Iterable

from txt2sql_eval_harness.alignment import align_tables
from txt2sql_eval_harness.baselines import (
    judge_b1,
    judge_b2,
    judge_b4,
    judge_b_ibm_judge,
    judge_b_ibm_subset,
)
from txt2sql_eval_harness.cache import ResponseCache
from txt2sql_eval_harness.clients import CallContext, _get_api_key
from txt2sql_eval_harness.codec import decode_record, decode_run_identity, encode_record, encode_run_identity
from txt2sql_eval_harness.comparator import DEFAULT_JUDGE_RHO
from txt2sql_eval_harness.data import UnadmittedInputError, egress_decision, sha256_file
from txt2sql_eval_harness.judge import (
    derive_h_from_he,
    full_trace_verdict_table,
    judge_item,
    mode_of,
    single_verdict_v,
    sql_only_verdict_table,
)
from txt2sql_eval_harness.models import (
    EvalConfig,
    LabelledItem,
    ModeName,
    Record,
    RecordedMethodName,
    RunIdentity,
    ThresholdPair,
    Verdict,
)
from txt2sql_eval_harness.packet import PreparedCandidate
from txt2sql_eval_harness.renderer import count_tokens
from txt2sql_eval_harness.rubrics import PROMPT_HASH
from txt2sql_eval_harness.spend import SpendLedger
from txt2sql_eval_harness.sql_analysis import extract_contract, is_det, normalize_sql_n

logger = logging.getLogger(__name__)

G_HI: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)
G_LO: tuple[float, ...] = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


class IncompatibleRunDirError(ValueError):
    """Raised when an existing run directory has a mismatched RunIdentity."""


def _hash_with_salt(salt: str, value: str) -> str:
    """Compute a deterministic salted SHA-256 hex digest for stratified sampling order."""
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


def _c_sort_key(item: LabelledItem) -> tuple[str, str]:
    """Return a group-deterministic sort key for reproducible subset selection."""
    return (_hash_with_salt("C", item.group_id), item.item_id)


def tag_slices(
    items: list[LabelledItem],
    *,
    cfg: EvalConfig | None = None,
) -> list[LabelledItem]:
    """Compute code slice tags (`coincident`, `large_result`, `mid_table_diff`) deterministically."""
    config = cfg or EvalConfig()
    out: list[LabelledItem] = []
    for item in items:
        slices = list(item.slices)
        gt_res = item.gt.results[0] if item.gt.results else None
        if gt_res is not None:
            tot = gt_res.total_rows if gt_res.total_rows is not None else len(gt_res.rows)
            if tot > 1000 and "large_result" not in slices:
                slices.append("large_result")

        if item.trace.final_answer is None and item.trace.steps and item.gt.sql_chain:
            gt_sql = item.gt.sql_chain[-1]
            gen_sql = item.trace.steps[-1].sql
            gen_res = item.trace.steps[-1].result
            if (
                gt_res is not None
                and gt_res.complete
                and gen_res is not None
                and gen_res.complete
            ):
                contract = extract_contract(gt_sql, item.trace.question, n_cols=len(gt_res.columns))
                al, _, _ = align_tables(gt_res, gen_res, contract, rho=DEFAULT_JUDGE_RHO)
                d1 = (
                    is_det(gt_sql)
                    and is_det(gen_sql)
                    and normalize_sql_n(gt_sql) is not None
                    and normalize_sql_n(gt_sql) == normalize_sql_n(gen_sql)
                )
                if al.relation == "equal" and not d1 and item.set != "S2":
                    if "coincident" not in slices:
                        slices.append("coincident")

                # mid_table_diff: large S2 mutants whose differing rows lie strictly outside head/tail 10
                if (
                    item.set == "S2"
                    and len(gt_res.rows) > 20
                    and len(gen_res.rows) > 20
                    and al.relation != "equal"
                ):
                    if gt_res.rows[:10] == gen_res.rows[:10] and gt_res.rows[-10:] == gen_res.rows[-10:]:
                        if "mid_table_diff" not in slices:
                            slices.append("mid_table_diff")

        out.append(replace(item, slices=slices))
    return out


def select_common_subset_c(items: list[LabelledItem]) -> list[LabelledItem]:
    """Select the shared common test subset C across natural splits and synthetic quotas."""
    test_pool = [it for it in items if it.split == "test" and it.pair_of is None]
    selected: list[LabelledItem] = []
    seen_ids: set[str] = set()

    def _add(it: LabelledItem) -> None:
        """Append *it* to the subset once, preserving first-seen order."""
        if it.item_id not in seen_ids:
            seen_ids.add(it.item_id)
            selected.append(it)

    # S4 and S5: all test items
    for it in sorted((x for x in test_pool if x.set in ("S4", "S5")), key=_c_sort_key):
        _add(it)

    # S1 / S2 / S3: equal quota per family (40 / 60 / 40)
    set_quotas = {"S1": 40, "S2": 60, "S3": 40}
    for sname, total_q in set_quotas.items():
        by_fam: dict[str, list[LabelledItem]] = defaultdict(list)
        for it in test_pool:
            if it.set == sname:
                by_fam[it.family].append(it)
        if not by_fam:
            continue
        per_fam = max(1, total_q // len(by_fam))
        for fam in sorted(by_fam.keys()):
            fam_items = sorted(by_fam[fam], key=_c_sort_key)
            for it in fam_items[:per_fam]:
                _add(it)

    # S3m: up to 12 paired test base questions x 4 families = 48
    s3m_by_base: dict[str, list[LabelledItem]] = defaultdict(list)
    for it in test_pool:
        if it.set == "S3m":
            parts = it.item_id.split("__")
            base_key = parts[1] if len(parts) >= 2 else it.group_id
            s3m_by_base[base_key].append(it)
    s3m_groups = sorted(
        s3m_by_base.items(),
        key=lambda kv: (_hash_with_salt("C", kv[1][0].group_id), kv[0]),
    )
    taken_s3m = 0
    for _, grp_items in s3m_groups:
        if taken_s3m >= 12:
            break
        if len(grp_items) == 4:
            for it in sorted(grp_items, key=lambda x: x.family):
                _add(it)
            taken_s3m += 1

    # Slices: up to 30 test items per slice (overlaps counted once)
    for sl_name in ("coincident", "large_result", "mid_table_diff"):
        sl_items = [it for it in test_pool if sl_name in it.slices]
        sl_items.sort(key=_c_sort_key)
        for it in sl_items[:30]:
            _add(it)

    return selected


def project_spend_and_shrink_c(
    c_items: list[LabelledItem],
    e5b_items: list[LabelledItem],
    cfg: EvalConfig,
    *,
    budget_usd: float = 120.0,
) -> tuple[list[LabelledItem], dict[str, Any]]:
    """Locally project B4/IBM/Jev spend and shrink S1/S2/S3/slices if projection > budget_usd."""
    in_p, out_p = cfg.get_pricing(cfg.generative_model)
    jev_in_p, _ = cfg.get_pricing(cfg.jev_model)

    def _est_item_cost(it: LabelledItem) -> float:
        """Estimate combined B4, IBM judge, and Jev spend for one item from token count."""
        text = json.dumps(
            {
                "q": it.trace.question,
                "h": it.trace.hint,
                "s": [st.sql for st in it.trace.steps],
                "a": it.trace.final_answer,
                "n_gt_rows": len(it.gt.results[0].rows) if it.gt.results else 0,
            }
        )
        toks = count_tokens(text)
        # B4 + IBM judge + Jev/escalation proxy
        b4_cost = (toks * in_p + 300 * out_p) / 1_000_000.0
        ibm_cost = (min(toks, 8000) * in_p + 200 * out_p) / 1_000_000.0
        jev_cost = (min(toks, 6000) * jev_in_p) / 1_000_000.0
        return b4_cost + ibm_cost + jev_cost

    e5b_cost = sum(_est_item_cost(it) for it in e5b_items)
    c_costs = [(it, _est_item_cost(it)) for it in c_items]
    total_projected = e5b_cost + sum(c for _, c in c_costs)

    if total_projected <= budget_usd:
        return list(c_items), {
            "projected_usd": total_projected,
            "budget_usd": budget_usd,
            "shrunk": False,
            "n_before": len(c_items),
            "n_after": len(c_items),
        }

    # Preserve S4 and S5; proportionally keep lowest-hash S1/S2/S3/S3m items
    protected = [it for it in c_items if it.set in ("S4", "S5")]
    shrinkable = [it for it in c_items if it.set not in ("S4", "S5")]
    protected_cost = sum(_est_item_cost(it) for it in protected) + e5b_cost
    remaining_budget = max(0.0, budget_usd - protected_cost)
    shrinkable_cost = sum(_est_item_cost(it) for it in shrinkable)
    keep_ratio = min(1.0, remaining_budget / shrinkable_cost) if shrinkable_cost > 0 else 1.0

    by_stratum: dict[tuple[str, str], list[LabelledItem]] = defaultdict(list)
    for it in shrinkable:
        by_stratum[(it.set, it.family)].append(it)

    kept_shrinkable: list[LabelledItem] = []
    for key in sorted(by_stratum.keys()):
        group = sorted(by_stratum[key], key=_c_sort_key)
        k_keep = int(len(group) * keep_ratio)
        kept_shrinkable.extend(group[:k_keep])

    final_c = protected + kept_shrinkable
    new_projected = e5b_cost + sum(_est_item_cost(it) for it in final_c)
    return final_c, {
        "projected_usd": new_projected,
        "initial_projected_usd": total_projected,
        "budget_usd": budget_usd,
        "shrunk": True,
        "n_before": len(c_items),
        "n_after": len(final_c),
    }


def fit_thresholds(
    cal_records: list[tuple[LabelledItem, Verdict, PreparedCandidate | None]],
    method: str,
    mode: ModeName,
) -> tuple[ThresholdPair, dict[str, Any]]:
    """Fit (tau_hi, tau_lo) on calibration items via grid search with a 20-item floor."""
    # Collect calibration items whose Jev call succeeded and true label is in {"correct", "incorrect"}
    jev_items: list[tuple[LabelledItem, dict[str, Any], PreparedCandidate | None]] = []
    for item, verdict, cand in cal_records:
        if item.split != "calibration" or item.label in ("gt_wrong", "unlabelled"):
            continue
        if verdict.mode != mode:
            continue
        jev_call = next(
            (c for c in verdict.calls if c.stage == "jev" and c.outcome == "ok" and c.answers is not None),
            None,
        )
        if jev_call is None or jev_call.answers is None:
            continue
        jev_items.append((item, jev_call.answers, cand))

    default = ThresholdPair()
    single = method in ("B6", "B6+E")

    def _eval_at(answers: dict[str, Any], cand: PreparedCandidate | None, tau: ThresholdPair) -> str:
        """Evaluate the composed verdict table (or single-question V rule) at candidate *tau*."""
        if single:
            return str(single_verdict_v(answers, tau))
        if mode == "sql_only":
            res, _ = sql_only_verdict_table(cand, answers, tau)  # type: ignore[arg-type]
            return str(res)
        res, _, _ = full_trace_verdict_table(cand, answers, tau)  # type: ignore[arg-type]
        return str(res)

    # 1. Fit tau_hi
    tau_hi = default.tau_hi
    tau_hi_reason = "kept_default_fewer_than_20"
    for g_hi in G_HI:
        trial_tau = ThresholdPair(tau_hi=g_hi, tau_lo=default.tau_lo)
        marked_correct = [
            item
            for item, ans, cand in jev_items
            if _eval_at(ans, cand, trial_tau) == "CORRECT"
        ]
        if len(marked_correct) < 20:
            continue
        false_passes = sum(1 for it in marked_correct if it.label == "incorrect")
        rate = false_passes / len(marked_correct)
        if rate <= 0.05 + 1e-12:
            tau_hi = g_hi
            tau_hi_reason = f"fitted(n={len(marked_correct)},fp={rate:.4f})"
            break
        tau_hi_reason = "kept_default_no_grid_under_5pct"

    # 2. Fit tau_lo with tau_hi fixed
    tau_lo = default.tau_lo
    tau_lo_reason = "kept_default_fewer_than_20"
    for g_lo in sorted((g for g in G_LO if g < tau_hi), reverse=True):
        trial_tau = ThresholdPair(tau_hi=tau_hi, tau_lo=g_lo)
        marked_incorrect = [
            item
            for item, ans, cand in jev_items
            if _eval_at(ans, cand, trial_tau) == "INCORRECT"
        ]
        if len(marked_incorrect) < 20:
            continue
        false_fails = sum(1 for it in marked_incorrect if it.label == "correct")
        rate = false_fails / len(marked_incorrect)
        if rate <= 0.05 + 1e-12:
            tau_lo = g_lo
            tau_lo_reason = f"fitted(n={len(marked_incorrect)},ff={rate:.4f})"
            break
        tau_lo_reason = "kept_default_no_grid_under_5pct"

    return ThresholdPair(tau_hi=tau_hi, tau_lo=tau_lo), {
        "n_jev_cal": len(jev_items),
        "tau_hi": tau_hi,
        "tau_lo": tau_lo,
        "tau_hi_status": tau_hi_reason,
        "tau_lo_status": tau_lo_reason,
    }


def _git_info(repo_dir: Path) -> tuple[str, bool]:
    """Return `(commit_sha, is_dirty)` for *repo_dir*, or `('unknown', False)` outside git."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return head, bool(dirty_out)
    except Exception:
        return "unknown", False


def compute_items_sha256(items: list[LabelledItem]) -> str:
    """Hash sorted item IDs into a deterministic dataset-membership fingerprint."""
    ids = sorted(it.item_id for it in items)
    return hashlib.sha256(
        json.dumps(ids, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _prepare_records_resume(records_path: Path) -> dict[tuple[str, str], Record]:
    """Read valid lines from records.jsonl into an index, dropping a torn last line if present."""
    if not records_path.exists():
        return {}
    raw = records_path.read_bytes()
    if not raw:
        return {}

    lines = raw.splitlines(keepends=True)
    valid_bytes = 0
    done: dict[tuple[str, str], Record] = {}
    for idx, line_bytes in enumerate(lines):
        line_str = line_bytes.decode("utf-8", errors="replace").strip()
        if not line_str:
            valid_bytes += len(line_bytes)
            continue
        try:
            rec = decode_record(line_str)
        except Exception as exc:
            if idx == len(lines) - 1:
                # Truncate torn last line cleanly
                with open(records_path, "r+b") as fh:
                    fh.truncate(valid_bytes)
                    fh.flush()
                    os.fsync(fh.fileno())
                break
            raise ValueError(f"Corrupt record on non-final line {idx + 1}") from exc
        done[(rec.item_id, rec.method)] = rec
        valid_bytes += len(line_bytes)
    return done


def _dispatch_method(
    item: LabelledItem,
    method: RecordedMethodName,
    ctx: CallContext,
    db_dir: Path | None,
) -> Verdict:
    """Route a single `(item, method)` pair to the corresponding judge or baseline implementation."""
    db_path = (db_dir / item.trace.db_id / f"{item.trace.db_id}.sqlite") if db_dir is not None else None
    if db_path is not None and not db_path.exists():
        db_path = None
    if method in ("H+E", "B6", "B6+E"):
        return judge_item(item.trace, item.gt, method, ctx, db_path=db_path, escalation=True)
    if method == "B1":
        return judge_b1(item.trace, item.gt, ctx.cfg)
    if method == "B2":
        return judge_b2(item.trace, item.gt, ctx.cfg, db_path=db_path)
    if method == "B-IBM-subset":
        return judge_b_ibm_subset(item.trace, item.gt, ctx.cfg, db_path=db_path)
    if method == "B-IBM-judge":
        return judge_b_ibm_judge(item.trace, item.gt, ctx, db_path=db_path)
    if method == "B4":
        return judge_b4(item.trace, item.gt, ctx, db_path=db_path)
    raise ValueError(f"Unsupported recorded method: {method!r}")


@dataclass
class RunSummary:
    """Summary returned when `run_experiment` completes or stops at the spend cap."""

    run_dir: Path
    identity: RunIdentity
    written_count: int
    skipped_existing_count: int
    stopped_on_spend_cap: bool
    allow_remote: bool


def _fit_calibration_thresholds(
    cfg: EvalConfig,
    methods: list[RecordedMethodName],
    cal_items: list[LabelledItem],
    completed_records: dict[tuple[str, str], Record],
    run_dir: Path,
) -> EvalConfig:
    """Fit calibration thresholds for Jev methods and persist fitted_thresholds.json."""
    new_tau = {m: dict(modes) for m, modes in cfg.tau.items()}
    fitted_report: dict[str, Any] = {}
    for method in methods:
        if method not in ("H+E", "B6", "B6+E"):
            continue
        cal_tuples = [
            (it, completed_records[(it.item_id, method)].verdict, None)
            for it in cal_items
            if (it.item_id, method) in completed_records
        ]
        if not cal_tuples:
            continue
        tau_key = "B6" if method in ("B6", "B6+E") else "H"
        for mode in ("sql_only", "full_trace"):
            pair, info = fit_thresholds(cal_tuples, method, mode)
            new_tau.setdefault(tau_key, {})[mode] = pair
            if tau_key == "H":
                new_tau.setdefault("H+E", {})[mode] = pair
            else:
                new_tau.setdefault("B6+E", {})[mode] = pair
            fitted_report[f"{method}:{mode}"] = info

    if not fitted_report:
        return cfg

    (run_dir / "fitted_thresholds.json").write_text(
        json.dumps(fitted_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return replace(cfg, tau=new_tau)


def run_experiment(
    items: list[LabelledItem],
    methods: list[RecordedMethodName],
    cfg: EvalConfig,
    root: Path,
    run_dir: Path,
    *,
    manifest_path: Path | None = None,
    input_files: list[Path] | None = None,
    db_dir: Path | None = None,
    local_only: bool = False,
    cache_only: bool = False,
    cap_usd: float = 150.0,
    clients: dict[str, Any] | None = None,
    sleep_fn: Callable[[float], Any] | None = None,
    concurrency: int = 1,
    verbose: bool = False,
) -> RunSummary:
    """Resumable experiment runner enforcing lock, spend checkpoint, calibration fitting, and optional concurrency."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    root = Path(root)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Egress gate decision at start
    if input_files is not None and manifest_path is not None:
        allow_remote = egress_decision(
            input_files,
            manifest_path,
            local_only=local_only,
            base_dir=manifest_path.parent.parent if manifest_path.parent.name == "data" else root,
        )
    else:
        allow_remote = not local_only

    if manifest_path is not None and manifest_path.exists():
        manifest_sha = sha256_file(manifest_path)
    else:
        manifest_sha = hashlib.sha256(b"").hexdigest()

    commit, dirty = _git_info(root)
    ident = RunIdentity(
        config_hash=cfg.config_hash(PROMPT_HASH),
        allow_remote=allow_remote,
        items_sha256=compute_items_sha256(items),
        manifest_sha256=manifest_sha,
        code_commit=commit,
        code_dirty=dirty,
        cache_only=cache_only,
    )

    run_json = run_dir / "run.json"
    if run_json.exists():
        existing = decode_run_identity(json.loads(run_json.read_text(encoding="utf-8")))
        if existing != ident:
            diffs = [
                f
                for f in (
                    "config_hash",
                    "allow_remote",
                    "items_sha256",
                    "manifest_sha256",
                    "code_commit",
                    "code_dirty",
                    "cache_only",
                )
                if getattr(existing, f) != getattr(ident, f)
            ]
            raise IncompatibleRunDirError(f"incompatible_run_dir: mismatched {', '.join(diffs)}")
    else:
        run_json.write_text(
            json.dumps(encode_run_identity(ident), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    records_path = run_dir / "records.jsonl"
    completed_records = _prepare_records_resume(records_path)
    done = set(completed_records.keys())

    ordered_items = sorted(items, key=lambda it: it.item_id)
    cal_items = [it for it in ordered_items if it.split == "calibration"]
    test_items = [it for it in ordered_items if it.split != "calibration"]
    io_lock = threading.Lock()
    written = 0
    skipped = 0
    stopped_cap = False

    with SpendLedger(root, cap_usd=cap_usd) as ledger:
        cache = ResponseCache(root / "cache" / "responses.sqlite") if allow_remote else None
        try:
            ctx_kwargs: dict[str, Any] = {
                "cfg": cfg,
                "allow_remote": allow_remote,
                "cache_only": cache_only,
                "cache": cache,
                "ledger": ledger if (allow_remote and not cache_only) else None,
                "clients": dict(clients or {}),
            }
            if sleep_fn is not None:
                ctx_kwargs["sleep_fn"] = sleep_fn
            ctx = CallContext(**ctx_kwargs)

            if allow_remote and not cache_only:
                if any(m in ("H+E", "B6", "B6+E") for m in methods) and "jev" not in ctx.clients:
                    if cfg.jev_model.startswith("gemini-decision"):
                        if not _get_api_key("GEMINI_API_KEY"):
                            raise RuntimeError(
                                "GEMINI_API_KEY is required for remote Gemini Decision calls"
                            )
                    elif not _get_api_key("TYPESAFE_API_KEY"):
                        raise RuntimeError("TYPESAFE_API_KEY is required for remote Jev calls")
                if "B4" in methods and "generative" not in ctx.clients:
                    if not _get_api_key("GEMINI_API_KEY"):
                        raise RuntimeError("GEMINI_API_KEY is required for remote generative calls")
                if "B-IBM-judge" in methods and "ibm_judge" not in ctx.clients:
                    if not _get_api_key("GEMINI_API_KEY"):
                        raise RuntimeError("GEMINI_API_KEY is required for remote generative calls")

            def _eval_one(item: LabelledItem, method: RecordedMethodName) -> Verdict | None:
                """Evaluate one `(item, method)` pair while catching non-fatal errors for resume."""
                try:
                    return _dispatch_method(item, method, ctx, db_dir)
                except RuntimeError:
                    raise
                except Exception:
                    logger.exception(
                        "Unhandled error judging item=%s method=%s; skipping persistence so resume retries",
                        item.item_id,
                        method,
                    )
                    return None

            def _persist_one(fh: Any, item: LabelledItem, method: RecordedMethodName, verdict: Verdict) -> None:
                """Persist one completed Verdict atomically under `io_lock`."""
                nonlocal written
                rec = Record(item_id=item.item_id, method=method, verdict=verdict)
                with io_lock:
                    fh.write(encode_record(rec) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                    done.add((item.item_id, method))
                    completed_records[(item.item_id, method)] = rec
                    written += 1
                    if verbose:
                        reason_s = f", reason={verdict.reason}" if verdict.reason else ""
                        print(
                            f"[{written}] {item.item_id} | {method} -> {verdict.label} "
                            f"(decided_by={verdict.decided_by}{reason_s})",
                            file=sys.stderr,
                            flush=True,
                        )

            def _run_phase(fh: Any, phase_items: list[LabelledItem]) -> None:
                """Run all pending `(item, method)` pairs for *phase_items* serially or with bounded threads."""
                nonlocal skipped, stopped_cap
                pending: list[tuple[LabelledItem, RecordedMethodName]] = []
                seen_pending: set[tuple[str, str]] = set(done)
                for item in phase_items:
                    for method in methods:
                        key = (item.item_id, method)
                        if key in seen_pending:
                            skipped += 1
                            continue
                        seen_pending.add(key)
                        pending.append((item, method))

                if concurrency == 1:
                    for item, method in pending:
                        if stopped_cap:
                            break
                        verdict = _eval_one(item, method)
                        if verdict is None:
                            continue
                        if verdict.reason == "spend_cap":
                            stopped_cap = True
                            break
                        _persist_one(fh, item, method, verdict)
                    return

                it_pending = iter(pending)
                in_flight: dict[Any, tuple[LabelledItem, RecordedMethodName]] = {}

                def _submit_next(pool: ThreadPoolExecutor) -> bool:
                    """Submit the next pending pair unless spend cap is reached or work is exhausted."""
                    if stopped_cap:
                        return False
                    try:
                        item, method = next(it_pending)
                    except StopIteration:
                        return False
                    in_flight[pool.submit(_eval_one, item, method)] = (item, method)
                    return True

                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    for _ in range(concurrency):
                        if not _submit_next(pool):
                            break
                    while in_flight:
                        done_futs, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                        for fut in done_futs:
                            item, method = in_flight.pop(fut)
                            verdict = fut.result()
                            if verdict is None:
                                continue
                            if verdict.reason == "spend_cap":
                                stopped_cap = True
                                continue
                            _persist_one(fh, item, method, verdict)
                        if not stopped_cap:
                            while len(in_flight) < concurrency and _submit_next(pool):
                                pass

            with open(records_path, "a", encoding="utf-8") as fh:
                if cal_items and test_items:
                    _run_phase(fh, cal_items)
                    if not stopped_cap:
                        ctx.cfg = _fit_calibration_thresholds(
                            cfg, methods, cal_items, completed_records, run_dir
                        )
                        _run_phase(fh, test_items)
                else:
                    _run_phase(fh, ordered_items)
        finally:
            if cache is not None:
                cache.close()

    return RunSummary(
        run_dir=run_dir,
        identity=ident,
        written_count=written,
        skipped_existing_count=skipped,
        stopped_on_spend_cap=stopped_cap,
        allow_remote=allow_remote,
    )
