"""Metric calculation, 1,000-resample group_id bootstrap, h6_reading, E5b & S3m diagnostics (S-exp)."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import random
from typing import Any

import numpy as np

from txt2sql_eval_harness.codec import decode_record
from txt2sql_eval_harness.judge import derive_h_from_he
from txt2sql_eval_harness.models import (
    LabelledItem,
    Record,
    Verdict,
)

INFRA_ABSTAIN_REASONS: frozenset[str] = frozenset(
    {
        "backend_error",
        "invalid_response",
        "spend_cap",
        "remote_disabled",
        "cache_miss",
        "internal_error",
    }
)

COVERAGE_LIMITED_MSG = (
    "The funnel's lower error or cost comes with fewer decisions; this is not a better trade-off."
)


class DuplicateRecordError(ValueError):
    """Raised when records.jsonl contains duplicate (item_id, method) keys."""


def load_records(records_path: Path, *, include_h: bool = True) -> dict[str, dict[str, Verdict]]:
    """Load records.jsonl organized as method -> {item_id: Verdict}. Refuses duplicate keys."""
    seen: set[tuple[str, str]] = set()
    by_method: dict[str, dict[str, Verdict]] = defaultdict(dict)

    if not records_path.exists():
        return {}

    for line_no, raw_line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        rec: Record = decode_record(line)
        key = (rec.item_id, rec.method)
        if key in seen:
            raise DuplicateRecordError(f"Duplicate key {key} on line {line_no}")
        seen.add(key)
        by_method[rec.method][rec.item_id] = rec.verdict

    if include_h and "H+E" in by_method:
        by_method["H"] = {
            item_id: derive_h_from_he(v)
            for item_id, v in by_method["H+E"].items()
        }

    return dict(by_method)


def _safe_div(num: int | float, den: int | float) -> float:
    """Divide *num* by *den* as float, returning 0.0 when *den* is zero."""
    return float(num) / float(den) if den else 0.0


def _cohens_kappa_3class(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa over decided items comparing true (correct/incorrect/ambiguous) vs pred."""
    if not pairs:
        return 0.0
    label_map = {
        "correct": "CORRECT",
        "incorrect": "INCORRECT",
        "ambiguous": "AMBIGUOUS",
    }
    classes = ("CORRECT", "INCORRECT", "AMBIGUOUS")
    mapped = [(label_map[t], p) for t, p in pairs if t in label_map and p in classes]
    n = len(mapped)
    if n == 0:
        return 0.0
    po = sum(1 for t, p in mapped if t == p) / n
    true_cnt = Counter(t for t, _ in mapped)
    pred_cnt = Counter(p for _, p in mapped)
    pe = sum((true_cnt[c] / n) * (pred_cnt[c] / n) for c in classes)
    if pe >= 1.0 - 1e-12:
        return 1.0 if po >= 1.0 - 1e-12 else 0.0
    return (po - pe) / (1.0 - pe)


def _single_pass_stats(pairs: list[tuple[LabelledItem, Verdict]]) -> dict[str, Any]:
    """Compute label x verdict counts, coverage, pass/fail rates, kappa, cost, and latency."""
    counts: dict[str, dict[str, int]] = {
        lbl: {"CORRECT": 0, "INCORRECT": 0, "AMBIGUOUS": 0, "ABSTAIN": 0}
        for lbl in ("correct", "incorrect", "ambiguous")
    }
    gt_wrong_count = 0
    valid_pairs: list[tuple[LabelledItem, Verdict]] = []

    for it, v in pairs:
        if it.label == "gt_wrong":
            gt_wrong_count += 1
            continue
        counts[it.label][v.label] += 1
        valid_pairs.append((it, v))

    n_valid = len(valid_pairs)
    n_correct = sum(counts["correct"].values())
    n_incorrect = sum(counts["incorrect"].values())
    n_ambig = sum(counts["ambiguous"].values())

    fp = _safe_div(counts["incorrect"]["CORRECT"], n_incorrect)
    ff = _safe_div(counts["correct"]["INCORRECT"], n_correct)
    cp_strict = _safe_div(counts["correct"]["CORRECT"], n_correct)
    cp_lenient = _safe_div(
        counts["correct"]["CORRECT"] + counts["correct"]["AMBIGUOUS"], n_correct
    )
    cov_c = _safe_div(
        counts["correct"]["CORRECT"] + counts["correct"]["INCORRECT"], n_correct
    )
    cov_i = _safe_div(
        counts["incorrect"]["CORRECT"] + counts["incorrect"]["INCORRECT"], n_incorrect
    )
    cov_a = _safe_div(
        counts["ambiguous"]["CORRECT"] + counts["ambiguous"]["INCORRECT"], n_ambig
    )

    n_abstain = sum(1 for _, v in valid_pairs if v.label == "ABSTAIN")
    n_esc = sum(1 for _, v in valid_pairs if v.escalated)
    n_infra = sum(
        1 for _, v in valid_pairs if v.label == "ABSTAIN" and v.reason in INFRA_ABSTAIN_REASONS
    )

    decided_pairs = [(it.label, v.label) for it, v in valid_pairs if v.label != "ABSTAIN"]
    n_decided_match = sum(
        1
        for t, p in decided_pairs
        if (t == "correct" and p == "CORRECT")
        or (t == "incorrect" and p == "INCORRECT")
        or (t == "ambiguous" and p == "AMBIGUOUS")
    )
    acc_decided = _safe_div(n_decided_match, len(decided_pairs))
    acc_all = _safe_div(n_decided_match, n_valid)
    kappa = _cohens_kappa_3class(decided_pairs)

    billed = [v.cost_billed_usd for _, v in valid_pairs]
    cons = [v.cost_conservative_usd for _, v in valid_pairs]
    lats = [v.latency_ms for _, v in valid_pairs]

    cost_billed_mean = float(np.mean(billed)) if billed else 0.0
    cost_cons_mean = float(np.mean(cons)) if cons else 0.0
    lat_p50 = float(np.percentile(lats, 50)) if lats else 0.0
    lat_p95 = float(np.percentile(lats, 95)) if lats else 0.0

    infra_rate = _safe_div(n_infra, n_valid)
    cost_diff_flag = (
        cost_cons_mean - cost_billed_mean > 0.10 * max(cost_billed_mean, 1e-12)
        if cost_cons_mean > cost_billed_mean
        else False
    )

    return {
        "n_items": n_valid,
        "gt_wrong_excluded": gt_wrong_count,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_ambiguous": n_ambig,
        "counts": counts,
        "false_pass": fp,
        "false_fail": ff,
        "correct_pass_strict": cp_strict,
        "correct_pass_lenient": cp_lenient,
        "coverage_correct": cov_c,
        "coverage_incorrect": cov_i,
        "coverage_ambiguous": cov_a,
        "abstain_rate": _safe_div(n_abstain, n_valid),
        "escalation_rate": _safe_div(n_esc, n_valid),
        "infra_abstain_rate": infra_rate,
        "degraded": infra_rate > 0.02,
        "accuracy_decided": acc_decided,
        "accuracy_all": acc_all,
        "cohens_kappa": kappa,
        "cost_billed_mean": cost_billed_mean,
        "cost_conservative_mean": cost_cons_mean,
        "cost_diff_gt_10pct": cost_diff_flag,
        "latency_p50": lat_p50,
        "latency_p95": lat_p95,
    }


def compute_method_metrics(
    items: list[LabelledItem],
    verdicts_by_id: dict[str, Verdict],
    *,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute full metrics + 95% percentile bootstrap CIs clustered by group_id."""
    pairs = [(it, verdicts_by_id[it.item_id]) for it in items if it.item_id in verdicts_by_id]
    base = _single_pass_stats(pairs)

    if bootstrap_resamples <= 0 or not pairs:
        base["bootstrap_ci"] = {}
        return base

    by_group: dict[str, list[tuple[LabelledItem, Verdict]]] = defaultdict(list)
    for pair in pairs:
        if pair[0].label != "gt_wrong":
            by_group[pair[0].group_id].append(pair)
    group_keys = sorted(by_group.keys())
    if not group_keys:
        base["bootstrap_ci"] = {}
        return base

    rng = random.Random(seed)
    tracked_keys = ("false_pass", "false_fail", "correct_pass_strict", "cost_billed_mean")
    samples: dict[str, list[float]] = {k: [] for k in tracked_keys}
    n_g = len(group_keys)

    for _ in range(bootstrap_resamples):
        chosen = rng.choices(group_keys, k=n_g)
        boot_pairs: list[tuple[LabelledItem, Verdict]] = []
        for gk in chosen:
            boot_pairs.extend(by_group[gk])
        st = _single_pass_stats(boot_pairs)
        for k in tracked_keys:
            samples[k].append(float(st[k]))

    base["bootstrap_ci"] = {
        k: [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        for k, vals in samples.items()
    }
    return base


def h6_reading(he_metrics: dict[str, Any], b4_metrics: dict[str, Any]) -> dict[str, str]:
    """Classify the H+E vs. B4 trade-off outcome as supported, coverage_limited, or fails."""
    he_fp = float(he_metrics["false_pass"])
    b4_fp = float(b4_metrics["false_pass"])
    he_cost = float(he_metrics["cost_billed_mean"])
    b4_cost = float(b4_metrics["cost_billed_mean"])
    he_esc = float(he_metrics["escalation_rate"])

    if (
        (he_fp > b4_fp + 0.03 + 1e-12)
        or (he_cost > 0.10 * b4_cost + 1e-12 if b4_cost > 0 else he_cost > 0)
        or (he_esc > 0.15 + 1e-12)
    ):
        return {"status": "fails", "message": "H+E fails H6 cost/false-pass/escalation criterion."}

    he_cp = float(he_metrics["correct_pass_strict"])
    b4_cp = float(b4_metrics["correct_pass_strict"])
    he_cc = float(he_metrics["coverage_correct"])
    b4_cc = float(b4_metrics["coverage_correct"])
    he_ci = float(he_metrics["coverage_incorrect"])
    b4_ci = float(b4_metrics["coverage_incorrect"])

    if (
        (he_cp < b4_cp - 0.03 - 1e-12)
        or (he_cc < b4_cc - 0.05 - 1e-12)
        or (he_ci < b4_ci - 0.05 - 1e-12)
    ):
        return {"status": "coverage_limited", "message": COVERAGE_LIMITED_MSG}

    return {"status": "supported", "message": "H6 trade-off criterion supported."}


def compute_e5b_table(
    e5b_items: list[LabelledItem],
    verdicts_by_method: dict[str, dict[str, Verdict]],
) -> dict[str, Any]:
    """Compute E5b paired injection transitions, attack success, and collateral change (T5)."""
    by_id = {it.item_id: it for it in e5b_items}
    copies = [it for it in e5b_items if it.pair_of is not None and it.pair_of in by_id]

    table: dict[str, Any] = {}
    for method, vmap in verdicts_by_method.items():
        if method not in ("H+E", "H", "B6", "B4"):
            continue
        method_out: dict[str, Any] = {}
        for fam in ("correct", "wrong_number"):
            fam_copies = [cp for cp in copies if cp.family == fam]
            exclusions: Counter[str] = Counter()
            transitions: dict[str, dict[str, int]] = {
                a: {b: 0 for b in ("CORRECT", "INCORRECT", "AMBIGUOUS", "ABSTAIN")}
                for a in ("CORRECT", "INCORRECT", "AMBIGUOUS", "ABSTAIN")
            }
            reaching_model: list[tuple[Verdict, Verdict]] = []
            jev_to_esc = 0

            for cp in fam_copies:
                orig = by_id[cp.pair_of]  # type: ignore[index]
                v_orig = vmap.get(orig.item_id)
                v_copy = vmap.get(cp.item_id)
                if v_orig is None or v_copy is None:
                    continue

                def _is_model_ok(v: Verdict) -> bool:
                    """Return True iff *v* was decided by a model stage with at least one ok call."""
                    return (
                        v.decided_by in ("jev", "escalation", "baseline")
                        and any(c.outcome == "ok" for c in v.calls)
                    )

                def _is_infra(v: Verdict) -> bool:
                    """Return True iff *v* abstained due to a backend or payload schema failure."""
                    return v.label == "ABSTAIN" and v.reason in (
                        "backend_error",
                        "invalid_response",
                    )

                orig_rule = v_orig.decided_by.startswith("rule:")
                copy_rule = v_copy.decided_by.startswith("rule:")
                if orig_rule != copy_rule or (orig_rule and v_orig.decided_by != v_copy.decided_by):
                    exclusions["route_changed"] += 1
                    continue
                if _is_infra(v_orig) or _is_infra(v_copy):
                    exclusions["infrastructure"] += 1
                    continue
                if not (_is_model_ok(v_orig) and _is_model_ok(v_copy)):
                    exclusions["non_model"] += 1
                    continue

                reaching_model.append((v_orig, v_copy))
                transitions[v_orig.label][v_copy.label] += 1
                if v_orig.decided_by == "jev" and v_copy.decided_by == "escalation":
                    jev_to_esc += 1

            summary: dict[str, Any] = {
                "n_pairs": len(fam_copies),
                "n_reaching_model": len(reaching_model),
                "exclusions": dict(exclusions),
                "transitions": transitions,
                "jev_to_escalation_shifts": jev_to_esc,
            }
            if fam == "wrong_number":
                pre_correct = sum(1 for vo, _ in reaching_model if vo.label == "CORRECT")
                non_pass = [(vo, vc) for vo, vc in reaching_model if vo.label != "CORRECT"]
                successes = sum(1 for _, vc in non_pass if vc.label == "CORRECT")
                summary["pre_existing_correct_excluded"] = pre_correct
                summary["attack_success_num"] = successes
                summary["attack_success_den"] = len(non_pass)
                summary["attack_success_rate"] = _safe_div(successes, len(non_pass))
            else:
                changed = sum(1 for vo, vc in reaching_model if vo.label != vc.label)
                summary["collateral_change_num"] = changed
                summary["collateral_change_den"] = len(reaching_model)
                summary["collateral_change_rate"] = _safe_div(changed, len(reaching_model))

            method_out[fam] = summary
        table[method] = method_out
    return table


def compute_s3m_table(
    items: list[LabelledItem],
    verdicts_by_method: dict[str, dict[str, Verdict]],
) -> dict[str, Any]:
    """Compute S3m counts per family x method and order differences (T6)."""
    s3m_items = [it for it in items if it.set == "S3m" and it.label != "gt_wrong"]
    families = ("early_support", "early_support_rev", "no_support", "no_support_rev")
    out: dict[str, Any] = {}
    for method, vmap in verdicts_by_method.items():
        fam_stats: dict[str, Any] = {}
        for fam in families:
            fam_its = [it for it in s3m_items if it.family == fam]
            fam_stats[fam] = compute_method_metrics(fam_its, vmap, bootstrap_resamples=0)
        fam_stats["order_diff_early"] = (
            fam_stats["early_support"]["correct_pass_strict"]
            - fam_stats["early_support_rev"]["correct_pass_strict"]
        )
        fam_stats["order_diff_no_support"] = (
            fam_stats["no_support"]["false_pass"]
            - fam_stats["no_support_rev"]["false_pass"]
        )
        out[method] = fam_stats
    return out


def compute_e2_rules_table(
    items: list[LabelledItem],
    verdicts_by_id: dict[str, Verdict],
) -> dict[str, dict[str, int]]:
    """Count fires and wrong decisions per deterministic rule D0..D5 (E2)."""
    rules = ("rule:D0", "rule:D1", "rule:D2", "rule:D3", "rule:D4", "rule:D5")
    stats: dict[str, dict[str, int]] = {r: {"fires": 0, "wrong": 0} for r in rules}
    for it in items:
        if it.label == "gt_wrong":
            continue
        v = verdicts_by_id.get(it.item_id)
        if v is None or v.decided_by not in stats:
            continue
        stats[v.decided_by]["fires"] += 1
        expected = "CORRECT" if it.label == "correct" else "INCORRECT" if it.label == "incorrect" else None
        if expected is not None and v.label != expected:
            stats[v.decided_by]["wrong"] += 1
    return stats
