"""Report and chart generators (T1-T6 tables and F2-F6 matplotlib figures)."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from txt2sql_eval_harness.metrics import (
    compute_e2_rules_table,
    compute_e5b_table,
    compute_method_metrics,
    compute_s3m_table,
    h6_reading,
)
from txt2sql_eval_harness.models import LabelledItem, Verdict


def build_t1_summary(
    all_items: list[LabelledItem],
    c_items: list[LabelledItem],
    *,
    generator_discards: dict[str, int] | None = None,
    run_identities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """T1: counts per set/family/split/slice, exclusions, gt_wrong, run identities."""
    by_set_split: Counter[tuple[str, str]] = Counter()
    by_family: Counter[tuple[str, str]] = Counter()
    by_slice: Counter[str] = Counter()
    gt_wrong = 0
    for it in all_items:
        by_set_split[(it.set, it.split)] += 1
        by_family[(it.set, it.family)] += 1
        for sl in it.slices:
            by_slice[sl] += 1
        if it.label == "gt_wrong":
            gt_wrong += 1

    return {
        "total_items": len(all_items),
        "common_subset_c_size": len(c_items),
        "by_set_split": {f"{k[0]}:{k[1]}": v for k, v in sorted(by_set_split.items())},
        "by_family": {f"{k[0]}:{k[1]}": v for k, v in sorted(by_family.items())},
        "by_slice": dict(sorted(by_slice.items())),
        "gt_wrong_count": gt_wrong,
        "generator_discards": dict(generator_discards or {}),
        "run_identities": list(run_identities or []),
    }


def build_t2_table(
    c_items: list[LabelledItem],
    verdicts_by_method: dict[str, dict[str, Verdict]],
    *,
    bootstrap_resamples: int = 1000,
) -> dict[str, Any]:
    """T2: per method x mode metrics on C, B4 split (B4/B4-full/B4-truncated), and h6_reading."""
    out: dict[str, Any] = {}
    for mode in ("sql_only", "full_trace"):
        mode_items = [
            it
            for it in c_items
            if (it.trace.final_answer is not None) == (mode == "full_trace")
        ]
        if not mode_items:
            continue
        mode_rows: dict[str, Any] = {}
        for method, vmap in verdicts_by_method.items():
            mode_rows[method] = compute_method_metrics(
                mode_items, vmap, bootstrap_resamples=bootstrap_resamples
            )
            if method == "B4":
                b4_full_its = [
                    it
                    for it in mode_items
                    if it.item_id in vmap and vmap[it.item_id].b4_context == "full"
                ]
                b4_trunc_its = [
                    it
                    for it in mode_items
                    if it.item_id in vmap and vmap[it.item_id].b4_context == "truncated"
                ]
                mode_rows["B4-full"] = compute_method_metrics(
                    b4_full_its, vmap, bootstrap_resamples=bootstrap_resamples
                )
                mode_rows["B4-truncated"] = compute_method_metrics(
                    b4_trunc_its, vmap, bootstrap_resamples=bootstrap_resamples
                )
        if "H+E" in mode_rows and "B4" in mode_rows:
            mode_rows["h6_reading_HE_vs_B4"] = h6_reading(mode_rows["H+E"], mode_rows["B4"])
        if "B6+E" in mode_rows and "B4" in mode_rows:
            mode_rows["h6_reading_B6E_vs_B4"] = h6_reading(mode_rows["B6+E"], mode_rows["B4"])
        out[mode] = mode_rows
    return out


def build_t3_abstains(
    verdicts_by_method: dict[str, dict[str, Verdict]],
    *,
    cache_replay_mismatches: int = 0,
    cache_miss_count: int = 0,
    reproduced_count: int = 0,
) -> dict[str, Any]:
    """T3: abstain reasons per method and cache-only reproduction check."""
    by_method: dict[str, dict[str, int]] = {}
    for method, vmap in verdicts_by_method.items():
        reasons = Counter(v.reason or "unspecified" for v in vmap.values() if v.label == "ABSTAIN")
        by_method[method] = dict(sorted(reasons.items()))
    return {
        "abstain_reasons_by_method": by_method,
        "cache_only_reproduced": reproduced_count,
        "cache_only_misses": cache_miss_count,
        "cache_only_mismatches": cache_replay_mismatches,
    }


def build_t4_cost(
    spend_json_path: Path | None,
    verdicts_by_method: dict[str, dict[str, Verdict]],
) -> dict[str, Any]:
    """T4: committed spend by stage from spend.json and per-method cost breakdown."""
    spend_data: dict[str, Any] = {}
    if spend_json_path is not None and spend_json_path.exists():
        spend_data = json.loads(spend_json_path.read_text(encoding="utf-8"))

    by_method: dict[str, Any] = {}
    for method, vmap in verdicts_by_method.items():
        n = len(vmap) or 1
        billed = sum(v.cost_billed_usd for v in vmap.values())
        cons = sum(v.cost_conservative_usd for v in vmap.values())
        cached_calls = sum(1 for v in vmap.values() for c in v.calls if c.from_cache)
        net_calls = sum(1 for v in vmap.values() for c in v.calls if not c.from_cache)
        by_method[method] = {
            "total_billed_usd": billed,
            "total_conservative_usd": cons,
            "mean_billed_usd": billed / n,
            "mean_conservative_usd": cons / n,
            "cached_calls": cached_calls,
            "network_calls": net_calls,
        }
    return {
        "spend_checkpoint": spend_data,
        "by_method": by_method,
    }


def generate_all_reports_and_figures(
    out_dir: Path,
    all_items: list[LabelledItem],
    c_items: list[LabelledItem],
    verdicts_by_method: dict[str, dict[str, Verdict]],
    *,
    e5b_items: list[LabelledItem] | None = None,
    e5b_verdicts: dict[str, dict[str, Verdict]] | None = None,
    spend_json_path: Path | None = None,
    generator_discards: dict[str, int] | None = None,
    bootstrap_resamples: int = 1000,
) -> dict[str, Any]:
    """Write T1-T6 JSON tables + Markdown summary and F2-F6 charts into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = build_t1_summary(all_items, c_items, generator_discards=generator_discards)
    t2 = build_t2_table(c_items, verdicts_by_method, bootstrap_resamples=bootstrap_resamples)
    t3 = build_t3_abstains(verdicts_by_method)
    t4 = build_t4_cost(spend_json_path, verdicts_by_method)
    t5 = compute_e5b_table(e5b_items or [], e5b_verdicts or {})
    t6 = compute_s3m_table(c_items, verdicts_by_method)
    e2 = {
        method: compute_e2_rules_table(c_items, vmap)
        for method, vmap in verdicts_by_method.items()
        if method in ("H+E", "H", "B6", "B6+E")
    }

    report = {
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "T4": t4,
        "T5": t5,
        "T6": t6,
        "E2_rules": e2,
    }
    (out_dir / "tables.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Figures F2-F6
    _plot_f2(out_dir / "F2_cost_tradeoff.png", t2)
    _plot_f3(out_dir / "F3_coincident.png", c_items, verdicts_by_method)
    _plot_f4(out_dir / "F4_large_tables.png", c_items, verdicts_by_method)
    _plot_f5(out_dir / "F5_final_answers.png", c_items, verdicts_by_method)
    _plot_f6(out_dir / "F6_jev_probabilities.png", all_items, verdicts_by_method)

    return report


def _plot_f2(path: Path, t2: dict[str, Any]) -> None:
    """Render Figure F2: false-pass and strict correct-pass vs. log billed cost per trace."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric_key, title in [
        (axes[0], "false_pass", "(a) False-Pass vs Cost"),
        (axes[1], "correct_pass_strict", "(b) Strict Correct-Pass vs Cost"),
    ]:
        for mode, rows in t2.items():
            for method, stats in rows.items():
                if not isinstance(stats, dict) or "cost_billed_mean" not in stats:
                    continue
                x = max(stats["cost_billed_mean"], 1e-6)
                y = stats[metric_key]
                ax.scatter([x], [y], label=f"{method} ({mode})")
                ax.annotate(method, (x, y), fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("Billed USD / trace (log)")
        ax.set_ylabel(metric_key)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_f3(path: Path, c_items: list[LabelledItem], verdicts_by_method: dict[str, dict[str, Verdict]]) -> None:
    """Render Figure F3: false-pass rate by method on the coincident-result slice."""
    coinc = [it for it in c_items if "coincident" in it.slices]
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = sorted(verdicts_by_method.keys())
    fps = []
    for m in methods:
        st = compute_method_metrics(coinc, verdicts_by_method[m], bootstrap_resamples=0)
        fps.append(st["false_pass"])
    ax.bar(methods, fps)
    ax.set_ylabel("False-pass on coincident slice")
    ax.set_title("F3: Coincident Slice False-Pass")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_f4(path: Path, c_items: list[LabelledItem], verdicts_by_method: dict[str, dict[str, Verdict]]) -> None:
    """Render Figure F4: false-pass rate by method on the large / mid-table diff slice."""
    mid_items = [it for it in c_items if "mid_table_diff" in it.slices]
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = sorted(verdicts_by_method.keys())
    fps = []
    for m in methods:
        st = compute_method_metrics(mid_items, verdicts_by_method[m], bootstrap_resamples=0)
        fps.append(st["false_pass"])
    ax.bar(methods, fps)
    ax.set_ylabel("False-pass on mid_table_diff")
    ax.set_title("F4: Large / Mid-Table Diff False-Pass")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_f5(path: Path, c_items: list[LabelledItem], verdicts_by_method: dict[str, dict[str, Verdict]]) -> None:
    """Render Figure F5: false-pass rate by method on S3 and S5 full-trace final answers."""
    s3_s5 = [it for it in c_items if it.set in ("S3", "S5")]
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = sorted(verdicts_by_method.keys())
    fps = [
        compute_method_metrics(s3_s5, verdicts_by_method[m], bootstrap_resamples=0)["false_pass"]
        for m in methods
    ]
    ax.bar(methods, fps)
    ax.set_ylabel("False-pass on S3 + S5")
    ax.set_title("F5: Final Answers False-Pass")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_f6(path: Path, items: list[LabelledItem], verdicts_by_method: dict[str, dict[str, Verdict]]) -> None:
    """Render Figure F6: histogram of Jev p(true) scores split by correct vs. incorrect ground truth."""
    fig, ax = plt.subplots(figsize=(7, 4))
    he_map = verdicts_by_method.get("H+E", {})
    probs_correct: list[float] = []
    probs_incorrect: list[float] = []
    for it in items:
        v = he_map.get(it.item_id)
        if v is None:
            continue
        for c in v.calls:
            if c.stage == "jev" and c.answers:
                for q in ("Q1", "Q3"):
                    if q in c.answers and isinstance(c.answers[q], (int, float)):
                        if it.label == "correct":
                            probs_correct.append(float(c.answers[q]))
                        elif it.label == "incorrect":
                            probs_incorrect.append(float(c.answers[q]))
    if probs_correct:
        ax.hist(probs_correct, bins=10, alpha=0.5, label="correct")
    if probs_incorrect:
        ax.hist(probs_incorrect, bins=10, alpha=0.5, label="incorrect")
    ax.set_xlabel("Jev p(true)")
    ax.set_ylabel("Count")
    ax.set_title("F6: Jev Probability Distribution")
    if probs_correct or probs_incorrect:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
