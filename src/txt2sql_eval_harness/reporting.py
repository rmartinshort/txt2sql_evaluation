"""Report and chart generators (T1-T6/Assessment tables, SUMMARY.md, and F1-F6 charts)."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from txt2sql_eval_harness.metrics import (
    compute_assessment_summary,
    compute_e2_rules_table,
    compute_e5b_table,
    compute_method_metrics,
    compute_s3m_table,
    h6_reading,
)
from txt2sql_eval_harness.models import LabelledItem, Verdict

_CANONICAL_METHOD_ORDER: tuple[str, ...] = (
    "H+E",
    "H",
    "B6+E",
    "B6",
    "B4",
    "B-IBM-judge",
    "B-IBM-subset",
    "B2",
    "B1",
)


def _ordered_methods(methods: Any) -> list[str]:
    """Return method names sorted in canonical paper/funnel order followed by extras."""
    method_set = set(methods)
    ordered = [m for m in _CANONICAL_METHOD_ORDER if m in method_set]
    extras = sorted(m for m in method_set if m not in ordered)
    return ordered + extras


def _pct(val: float) -> float:
    """Convert a [0, 1] proportion to a [0, 100] percentage."""
    return 100.0 * float(val)


def _lenient_accuracy(stats: dict[str, Any]) -> float:
    """Compute overall accuracy counting AMBIGUOUS on `correct` items as non-errors."""
    counts = stats.get("counts", {})
    n_lab = (
        int(stats.get("n_correct", 0))
        + int(stats.get("n_incorrect", 0))
        + int(stats.get("n_ambiguous", 0))
    )
    if n_lab <= 0:
        return 0.0
    matched = (
        int(counts.get("correct", {}).get("CORRECT", 0))
        + int(counts.get("correct", {}).get("AMBIGUOUS", 0))
        + int(counts.get("incorrect", {}).get("INCORRECT", 0))
        + int(counts.get("ambiguous", {}).get("AMBIGUOUS", 0))
    )
    return float(matched) / float(n_lab)


def _render_empty_panel(ax: Any, title: str, message: str) -> None:
    """Render an informative centered placeholder message instead of a blank plot."""
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
        wrap=True,
    )


def _plot_grouped_metric_bars(
    ax: Any,
    groups: list[str],
    series: list[tuple[str, list[float], str]],
    *,
    title: str,
    ylabel: str,
    empty_msg: str = "No evaluated items available.",
    subtitle: str | None = None,
) -> None:
    """Plot a [0, 100]% grouped vertical bar chart with percentage annotations on every bar."""
    if not groups or not series:
        _render_empty_panel(ax, title, empty_msg)
        return

    n_groups = len(groups)
    n_series = len(series)
    total_width = 0.78
    bar_width = total_width / max(1, n_series)
    x_positions = list(range(n_groups))
    show_labels = (n_groups * n_series) <= 28

    for idx, (label, values, color) in enumerate(series):
        offsets = [x - total_width / 2 + (idx + 0.5) * bar_width for x in x_positions]
        bars = ax.bar(
            offsets,
            values,
            width=bar_width * 0.92,
            label=label,
            color=color,
            edgecolor="#ffffff",
            linewidth=0.5,
        )
        if show_labels:
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    min(val + 1.5, 103.5),
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.5 if n_series <= 3 else 7.0,
                    color="#222222",
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups, rotation=20 if n_groups > 4 else 0, ha="right" if n_groups > 4 else "center")
    ax.set_ylim(0.0, 112.0)
    ax.set_ylabel(ylabel)
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


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
    """T2: per method x mode metrics on item subset, B4 split, and h6_reading."""
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
    """Write T1-T6/Assessment JSON tables, SUMMARY.md, and F1-F6 charts into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = build_t1_summary(all_items, c_items, generator_discards=generator_discards)
    t2 = build_t2_table(c_items, verdicts_by_method, bootstrap_resamples=bootstrap_resamples)
    t2_all = build_t2_table(all_items, verdicts_by_method, bootstrap_resamples=0)
    assessment = compute_assessment_summary(all_items, verdicts_by_method)
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
        "T2_all": t2_all,
        "Assessment": assessment,
        "T3": t3,
        "T4": t4,
        "T5": t5,
        "T6": t6,
        "E2_rules": e2,
    }
    (out_dir / "tables.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fitted_path = out_dir.parent / "fitted_thresholds.json"
    fitted_thresholds: dict[str, Any] = {}
    if fitted_path.exists():
        try:
            fitted_thresholds = json.loads(fitted_path.read_text(encoding="utf-8"))
        except Exception:
            fitted_thresholds = {}

    _plot_f1(out_dir / "F1_judge_overview.png", all_items, verdicts_by_method, assessment)
    _plot_f2(out_dir / "F2_cost_tradeoff.png", t2, t2_all, assessment)
    _plot_f3(out_dir / "F3_coincident.png", all_items, verdicts_by_method, assessment)
    _plot_f4(out_dir / "F4_large_tables.png", all_items, verdicts_by_method)
    _plot_f5(out_dir / "F5_final_answers.png", all_items, verdicts_by_method)
    _plot_f6(
        out_dir / "F6_jev_probabilities.png",
        all_items,
        verdicts_by_method,
        fitted_thresholds=fitted_thresholds,
    )
    _write_summary_markdown(out_dir / "SUMMARY.md", report)

    return report


def _plot_f1(
    path: Path,
    all_items: list[LabelledItem],
    verdicts_by_method: dict[str, dict[str, Verdict]],
    assessment: dict[str, Any],
) -> None:
    """Render F1: Judge Assessment breakdown (CORRECT/INCORRECT/AMBIGUOUS/ABSTAIN) and Accuracy or By-DB summary."""
    methods = _ordered_methods(verdicts_by_method.keys())
    overall = assessment.get("overall", {})
    by_mode = assessment.get("by_mode", {})
    by_db = assessment.get("by_db", {})
    has_labels = any(bool(st.get("has_ground_truth_labels")) for st in overall.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, 0.45 * len(methods) + 3.2)))

    # Panel A: Horizontal stacked bar chart of Judge Assessment (% CORRECT / AMBIGUOUS / INCORRECT / ABSTAIN)
    ax_left = axes[0]
    row_labels: list[str] = []
    row_stats: list[dict[str, Any]] = []
    for m in reversed(methods):
        st_all = overall.get(m)
        if st_all and st_all.get("n_items", 0) > 0:
            row_labels.append(f"{m} (All, n={st_all['n_items']})")
            row_stats.append(st_all)
        for mode_name in ("sql_only", "full_trace"):
            st_mode = by_mode.get(mode_name, {}).get(m)
            if st_mode and st_mode.get("n_items", 0) > 0 and len(by_mode) > 1:
                short_mode = "SQL" if mode_name == "sql_only" else "Trace"
                row_labels.append(f"  {m} [{short_mode}, n={st_mode['n_items']}]")
                row_stats.append(st_mode)

    if not row_stats:
        _render_empty_panel(ax_left, "F1(a): Judge Assessment", "No evaluated items recorded.")
    else:
        y_pos = list(range(len(row_stats)))
        left_acc = [0.0] * len(row_stats)
        segments = [
            ("CORRECT", "prop_correct", "#2ca02c"),
            ("AMBIGUOUS", "prop_ambiguous", "#ffbf00"),
            ("INCORRECT", "prop_incorrect", "#d62728"),
            ("ABSTAIN", "prop_abstain", "#7f7f7f"),
        ]
        for seg_label, prop_key, color in segments:
            widths = [_pct(st["assessment"][prop_key]) for st in row_stats]
            ax_left.barh(
                y_pos,
                widths,
                left=left_acc,
                color=color,
                edgecolor="#ffffff",
                linewidth=0.6,
                label=seg_label,
            )
            for idx, (w, l_val) in enumerate(zip(widths, left_acc)):
                if w >= 8.0:
                    ax_left.text(
                        l_val + w / 2.0,
                        y_pos[idx],
                        f"{w:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="#ffffff" if color != "#ffbf00" else "#222222",
                        fontweight="bold",
                    )
            left_acc = [l_val + w for l_val, w in zip(left_acc, widths)]

        ax_left.set_yticks(y_pos)
        ax_left.set_yticklabels(row_labels, fontsize=8.5)
        ax_left.set_xlim(0.0, 100.0)
        ax_left.set_xlabel("% of Evaluated Items")
        ax_left.set_title("F1(a): Judge Assessment Breakdown\n(What the judge decided)", fontsize=11, fontweight="bold")
        ax_left.spines["top"].set_visible(False)
        ax_left.spines["right"].set_visible(False)
        ax_left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8)

    # Panel B: Judge Accuracy & Error Rates (if labelled) OR Assessment by DB (if unlabelled)
    ax_right = axes[1]
    if has_labels:
        eval_methods = [m for m in methods if overall.get(m, {}).get("n_items", 0) > 0]
        _plot_grouped_metric_bars(
            ax_right,
            eval_methods,
            [
                ("Strict Acc", [_pct(overall[m]["accuracy_all"]) for m in eval_methods], "#1f77b4"),
                ("Lenient Acc", [_pct(_lenient_accuracy(overall[m])) for m in eval_methods], "#2ca02c"),
                ("False Pass", [_pct(overall[m]["false_pass"]) for m in eval_methods], "#d62728"),
                ("False Fail", [_pct(overall[m]["false_fail"]) for m in eval_methods], "#ff7f0e"),
            ],
            title="F1(b): Judge Accuracy & Error Rates",
            subtitle=f"All labelled evaluated items (total n={len(all_items)})",
            ylabel="Percentage (%)",
        )
    else:
        primary = next((m for m in methods if overall.get(m, {}).get("n_items", 0) > 0), None)
        if primary is None or not by_db:
            _render_empty_panel(
                ax_right,
                "F1(b): Assessment by Database",
                "Unlabelled evaluation: no ground-truth labels or DB breakdown.",
            )
        else:
            dbs = sorted(by_db.keys())
            vals = [_pct(by_db[db][primary]["assessment"]["prop_correct"]) for db in dbs]
            labels = [f"{db} (n={by_db[db][primary]['n_items']})" for db in dbs]
            y_pos = list(range(len(dbs)))
            bars = ax_right.barh(y_pos, vals, color="#2ca02c", edgecolor="#ffffff")
            for bar, v in zip(bars, vals):
                ax_right.text(
                    min(v + 1.5, 95.0),
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{v:.1f}%",
                    va="center",
                    fontsize=8,
                )
            ax_right.set_yticks(y_pos)
            ax_right.set_yticklabels(labels, fontsize=8)
            ax_right.set_xlim(0.0, 108.0)
            ax_right.set_xlabel("% Judged CORRECT")
            ax_right.set_title(
                f"F1(b): Agent Pass Rate by Database ({primary})\n(Unlabelled assessment)",
                fontsize=11,
                fontweight="bold",
            )
            ax_right.spines["top"].set_visible(False)
            ax_right.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_f2(
    path: Path,
    t2: dict[str, Any],
    t2_all: dict[str, Any],
    assessment: dict[str, Any],
) -> None:
    """Render F2: Accuracy (or % CORRECT) vs. cost per item and decision-tier funnel breakdown."""
    source_t2 = t2_all if t2_all else t2
    overall = assessment.get("overall", {})
    has_labels = any(bool(st.get("has_ground_truth_labels")) for st in overall.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A: Accuracy (or % CORRECT) vs mean billed cost
    ax_left = axes[0]
    mode_markers = {"sql_only": ("o", "SQL-Only"), "full_trace": ("s", "Full-Trace")}
    method_colors = {
        "H+E": "#2ca02c",
        "H": "#17becf",
        "B6+E": "#9467bd",
        "B6": "#8c564b",
        "B4": "#d62728",
        "B-IBM-judge": "#ff7f0e",
        "B-IBM-subset": "#1f77b4",
        "B2": "#7f7f7f",
        "B1": "#bcbd22",
    }
    plotted = 0
    x_vals: list[float] = []
    points: list[tuple[float, float, str, str, str, str, float]] = []
    for mode, rows in source_t2.items():
        marker, mode_lbl = mode_markers.get(mode, ("^", mode))
        for method, stats in rows.items():
            if not isinstance(stats, dict) or "cost_billed_mean" not in stats or stats.get("n_items", 0) == 0:
                continue
            raw_cost = float(stats["cost_billed_mean"])
            x = max(raw_cost, 1e-6)
            y = _pct(
                float(stats["accuracy_all"])
                if has_labels and stats.get("has_ground_truth_labels")
                else float(stats.get("assessment", {}).get("prop_correct", 0.0))
            )
            points.append((x, y, marker, mode_lbl, method, method_colors.get(method, "#333333"), raw_cost))
            x_vals.append(x)
            plotted += 1

    max_x = max(x_vals) if x_vals else 1e-3
    min_x = min(x_vals) if x_vals else 1e-6
    for x, y, marker, mode_lbl, method, color, raw_cost in points:
        ax_left.scatter(
            [x],
            [y],
            marker=marker,
            s=75,
            color=color,
            edgecolors="#222222",
            linewidth=0.6,
            label=f"{method} ({mode_lbl})",
        )
        near_right = len(x_vals) > 1 and x >= max_x * 0.6
        ax_left.annotate(
            f"{method}\n({mode_lbl}, ${raw_cost:.4f})",
            (x, y),
            textcoords="offset points",
            xytext=(-6, -18) if near_right else (6, 4),
            ha="right" if near_right else "left",
            fontsize=7.5,
        )

    if plotted == 0:
        _render_empty_panel(ax_left, "F2(a): Accuracy vs Cost", "No evaluated method costs recorded.")
    else:
        ax_left.set_xscale("log")
        ax_left.set_xlim(min_x * 0.5, max_x * 2.5)
        ax_left.set_ylim(0.0, 108.0)
        ax_left.set_xlabel("Mean Billed Cost per Item (USD, log scale; $1e-6 = $0)")
        ax_left.set_ylabel("Strict Accuracy (%)" if has_labels else "% Judged CORRECT")
        ax_left.set_title("F2(a): Accuracy vs. Cost per Item\n(Top-left is best)", fontsize=11, fontweight="bold")
        ax_left.grid(True, linestyle="--", alpha=0.35)
        ax_left.spines["top"].set_visible(False)
        ax_left.spines["right"].set_visible(False)
        ax_left.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    # Panel B: Funnel decision-tier breakdown per method
    ax_right = axes[1]
    methods = [m for m in _ordered_methods(overall.keys()) if overall[m].get("n_items", 0) > 0]
    if not methods:
        _render_empty_panel(ax_right, "F2(b): Decision Funnel Breakdown", "No evaluated items recorded.")
    else:
        x_pos = list(range(len(methods)))
        bottoms = [0.0] * len(methods)
        tiers = [
            ("Deterministic Rules ($0)", "deterministic", "#1f77b4"),
            ("Jev Model", "jev", "#17becf"),
            ("Gemini Escalation", "escalation", "#9467bd"),
            ("Baseline Model/Heuristic", "baseline", "#ff7f0e"),
            ("Abstain", "abstain", "#7f7f7f"),
        ]
        for tier_lbl, tier_key, color in tiers:
            vals = [_pct(overall[m]["stage_breakdown"][tier_key]) for m in methods]
            if not any(v > 0.0 for v in vals):
                continue
            bars = ax_right.bar(
                x_pos,
                vals,
                bottom=bottoms,
                color=color,
                edgecolor="#ffffff",
                linewidth=0.6,
                label=tier_lbl,
            )
            for idx, (v, b) in enumerate(zip(vals, bottoms)):
                if v >= 8.0:
                    ax_right.text(
                        x_pos[idx],
                        b + v / 2.0,
                        f"{v:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="#ffffff",
                        fontweight="bold",
                    )
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        ax_right.set_xticks(x_pos)
        ax_right.set_xticklabels(methods, rotation=20 if len(methods) > 4 else 0)
        ax_right.set_ylim(0.0, 108.0)
        ax_right.set_ylabel("% of Evaluated Items")
        ax_right.set_title("F2(b): Where Verdicts Are Decided\n(Cost savings come from $0 Rules + Jev)", fontsize=11, fontweight="bold")
        ax_right.grid(axis="y", linestyle="--", alpha=0.35)
        ax_right.set_axisbelow(True)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)
        ax_right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_f3(
    path: Path,
    all_items: list[LabelledItem],
    verdicts_by_method: dict[str, dict[str, Verdict]],
    assessment: dict[str, Any],
) -> None:
    """Render F3: Judge accuracy and false-pass rate across benchmark sets and coincident-result items."""
    methods = _ordered_methods(verdicts_by_method.keys())
    by_set = assessment.get("by_set", {})
    coinc_ids = {
        it.item_id
        for it in all_items
        if "coincident" in it.slices
        or any("coincident_result" in verdicts_by_method[m].get(it.item_id, Verdict("", "B1", "sql_only", "ABSTAIN", "abstain")).flags for m in methods)
    }
    coinc_items = [it for it in all_items if it.item_id in coinc_ids and it.label != "gt_wrong"]

    slices: list[tuple[str, dict[str, Any]]] = []
    for s_name in sorted(by_set.keys()):
        n_s = max((by_set[s_name][m]["n_items"] for m in methods if m in by_set[s_name]), default=0)
        if n_s > 0:
            slices.append((f"{s_name}\n(n={n_s})", by_set[s_name]))
    if coinc_items:
        coinc_stats = {
            m: compute_method_metrics(coinc_items, verdicts_by_method[m], bootstrap_resamples=0)
            for m in methods
        }
        n_c = max((coinc_stats[m]["n_items"] for m in methods), default=0)
        if n_c > 0:
            slices.append((f"Coincident\n(n={n_c})", coinc_stats))

    has_labels = any(
        bool(st.get("has_ground_truth_labels"))
        for st in assessment.get("overall", {}).values()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    if not slices or not methods:
        _render_empty_panel(axes[0], "F3(a): Performance by Dataset Slice", "No evaluated slices available.")
        _render_empty_panel(axes[1], "F3(b): Error Rate by Dataset Slice", "No evaluated slices available.")
    else:
        palette = ["#2ca02c", "#17becf", "#9467bd", "#8c564b", "#d62728", "#ff7f0e", "#1f77b4", "#7f7f7f", "#bcbd22"]
        slice_labels = [lbl for lbl, _ in slices]
        acc_series = []
        fp_series = []
        for idx, m in enumerate(methods):
            color = palette[idx % len(palette)]
            acc_vals = [
                _pct(
                    smap[m]["accuracy_all"]
                    if has_labels and smap[m].get("has_ground_truth_labels")
                    else smap[m]["assessment"]["prop_correct"]
                )
                for _, smap in slices
            ]
            fp_vals = [
                _pct(
                    smap[m]["false_pass"]
                    if has_labels and smap[m].get("has_ground_truth_labels")
                    else smap[m]["assessment"]["prop_incorrect"]
                )
                for _, smap in slices
            ]
            acc_series.append((m, acc_vals, color))
            fp_series.append((m, fp_vals, color))

        _plot_grouped_metric_bars(
            axes[0],
            slice_labels,
            acc_series,
            title="F3(a): Strict Accuracy by Set & Coincident Slice"
            if has_labels
            else "F3(a): % Judged CORRECT by Set & Coincident Slice",
            ylabel="Strict Accuracy (%)" if has_labels else "% Judged CORRECT",
        )
        _plot_grouped_metric_bars(
            axes[1],
            slice_labels,
            fp_series,
            title="F3(b): False-Pass Rate (Lower is Better)"
            if has_labels
            else "F3(b): % Judged INCORRECT by Slice",
            ylabel="False-Pass Rate (%)" if has_labels else "% Judged INCORRECT",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_slice_breakdown_figure(
    path: Path,
    subset_items: list[LabelledItem],
    verdicts_by_method: dict[str, dict[str, Verdict]],
    *,
    title: str,
    subtitle: str,
    empty_msg: str,
) -> None:
    """Shared grouped bar chart plotting Accuracy, Correct-Pass, False-Pass, and False-Fail on a slice."""
    methods = [
        m
        for m in _ordered_methods(verdicts_by_method.keys())
        if any(it.item_id in verdicts_by_method[m] for it in subset_items)
    ]
    fig, ax = plt.subplots(figsize=(max(7.5, 1.35 * max(1, len(methods)) + 2.5), 4.5))
    if not subset_items or not methods:
        _render_empty_panel(ax, title, empty_msg)
    else:
        stats_by_m = {
            m: compute_method_metrics(subset_items, verdicts_by_method[m], bootstrap_resamples=0)
            for m in methods
        }
        has_labels = any(bool(st.get("has_ground_truth_labels")) for st in stats_by_m.values())
        if has_labels:
            series = [
                ("Strict Acc", [_pct(stats_by_m[m]["accuracy_all"]) for m in methods], "#1f77b4"),
                ("Correct-Pass", [_pct(stats_by_m[m]["correct_pass_strict"]) for m in methods], "#2ca02c"),
                ("False-Pass", [_pct(stats_by_m[m]["false_pass"]) for m in methods], "#d62728"),
                ("False-Fail", [_pct(stats_by_m[m]["false_fail"]) for m in methods], "#ff7f0e"),
            ]
        else:
            series = [
                ("% CORRECT", [_pct(stats_by_m[m]["assessment"]["prop_correct"]) for m in methods], "#2ca02c"),
                ("% INCORRECT", [_pct(stats_by_m[m]["assessment"]["prop_incorrect"]) for m in methods], "#d62728"),
                ("% ABSTAIN", [_pct(stats_by_m[m]["assessment"]["prop_abstain"]) for m in methods], "#7f7f7f"),
            ]
        n_eval = max(stats_by_m[m]["n_items"] for m in methods)
        _plot_grouped_metric_bars(
            ax,
            methods,
            series,
            title=title,
            subtitle=f"{subtitle} (evaluated n={n_eval})",
            ylabel="Percentage (%)",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_f4(
    path: Path,
    all_items: list[LabelledItem],
    verdicts_by_method: dict[str, dict[str, Verdict]],
) -> None:
    """Render F4: Judge accuracy & error rates on large_result and mid_table_diff items."""
    large_items = [
        it
        for it in all_items
        if ("mid_table_diff" in it.slices or "large_result" in it.slices) and it.label != "gt_wrong"
    ]
    subtitle = "Items with >1,000 rows (large_result) or non-empty mid-table diffs"
    if not large_items:
        large_items = [
            it for it in all_items if it.trace.final_answer is None and it.label != "gt_wrong"
        ]
        subtitle = "All SQL-only items (no large_result/mid_table_diff slice tagged)"
    _plot_slice_breakdown_figure(
        path,
        large_items,
        verdicts_by_method,
        title="F4: Large Result & Mid-Table Diff Slice",
        subtitle=subtitle,
        empty_msg="No large-result or mid-table-diff items evaluated.",
    )


def _plot_f5(
    path: Path,
    all_items: list[LabelledItem],
    verdicts_by_method: dict[str, dict[str, Verdict]],
) -> None:
    """Render F5: Judge accuracy & error rates on full-trace prose answers (S3, S3m, S5)."""
    ft_items = [
        it
        for it in all_items
        if (it.trace.final_answer is not None or it.set in ("S3", "S3m", "S5"))
        and it.label != "gt_wrong"
    ]
    _plot_slice_breakdown_figure(
        path,
        ft_items,
        verdicts_by_method,
        title="F5: Full-Trace & Prose Final Answer Evaluation",
        subtitle="Traces with SQL steps + natural-language final_answer (S3, S3m, S5)",
        empty_msg="No full-trace (prose final_answer) items in this dataset.",
    )


def _plot_f6(
    path: Path,
    items: list[LabelledItem],
    verdicts_by_method: dict[str, dict[str, Verdict]],
    *,
    fitted_thresholds: dict[str, Any] | None = None,
) -> None:
    """Render F6: Jev p(true) probability distribution with decision/escalation thresholds."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    target_method = next(
        (m for m in ("H+E", "H", "B6+E", "B6") if m in verdicts_by_method and verdicts_by_method[m]),
        None,
    )
    vmap = verdicts_by_method.get(target_method, {}) if target_method else {}
    probs_correct: list[float] = []
    probs_incorrect: list[float] = []
    probs_unlabelled: list[float] = []

    for it in items:
        v = vmap.get(it.item_id)
        if v is None:
            continue
        for c in v.calls:
            if c.stage == "jev" and c.answers:
                for q in ("Q1", "Q3", "Q4", "Q6"):
                    if q in c.answers and isinstance(c.answers[q], (int, float)):
                        pval = float(c.answers[q])
                        if it.label == "correct":
                            probs_correct.append(pval)
                        elif it.label == "incorrect":
                            probs_incorrect.append(pval)
                        elif it.label == "unlabelled":
                            probs_unlabelled.append(pval)

    if not (probs_correct or probs_incorrect or probs_unlabelled):
        _render_empty_panel(
            ax,
            "F6: Jev Probability Distribution",
            "No Jev model calls recorded in this run\n(e.g. deterministic-only baselines or --local-only).",
        )
    else:
        tau_lo = 0.15
        tau_hi = 0.85
        if fitted_thresholds and "H+E:sql_only" in fitted_thresholds:
            tau_lo = float(fitted_thresholds["H+E:sql_only"].get("tau_lo", tau_lo))
            tau_hi = float(fitted_thresholds["H+E:sql_only"].get("tau_hi", tau_hi))

        ax.axvspan(0.0, tau_lo, color="#d62728", alpha=0.07, label=f"Direct Reject (<= {tau_lo:.2f})")
        ax.axvspan(tau_lo, tau_hi, color="#ffbf00", alpha=0.10, label="Escalate to Gemini")
        ax.axvspan(tau_hi, 1.0, color="#2ca02c", alpha=0.07, label=f"Direct Pass (>= {tau_hi:.2f})")
        ax.axvline(tau_lo, color="#d62728", linestyle="--", linewidth=1.2)
        ax.axvline(tau_hi, color="#2ca02c", linestyle="--", linewidth=1.2)

        bins = [i / 20.0 for i in range(21)]
        if probs_correct:
            ax.hist(probs_correct, bins=bins, alpha=0.65, color="#2ca02c", label=f"True: correct (n={len(probs_correct)})")
        if probs_incorrect:
            ax.hist(probs_incorrect, bins=bins, alpha=0.65, color="#d62728", label=f"True: incorrect (n={len(probs_incorrect)})")
        if probs_unlabelled:
            ax.hist(probs_unlabelled, bins=bins, alpha=0.65, color="#1f77b4", label=f"Unlabelled (n={len(probs_unlabelled)})")

        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Jev Score p(true) across Q1, Q3, Q4, Q6")
        ax.set_ylabel("Question Score Count")
        ax.set_title(
            f"F6: Jev Confidence Score Distribution ({target_method})\n(Shaded bands show Direct Reject / Escalate / Direct Pass zones)",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper center", fontsize=7.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_summary_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a plain-English SUMMARY.md explaining judge assessment, accuracy, costs, and charts."""
    t1 = report.get("T1", {})
    t2 = report.get("T2", {})
    t4 = report.get("T4", {})
    assessment = report.get("Assessment", {})
    overall = assessment.get("overall", {})
    by_mode = assessment.get("by_mode", {})
    by_set = assessment.get("by_set", {})
    by_db = assessment.get("by_db", {})
    methods = _ordered_methods(overall.keys())

    lines: list[str] = [
        "# Text2SQL Evaluation Report Summary",
        "",
        f"- **Total items in dataset**: `{t1.get('total_items', 0)}` (Common Subset `C`: `{t1.get('common_subset_c_size', 0)}`)",
        f"- **Methods included**: {', '.join(f'`{m}`' for m in methods) if methods else 'none'}",
        "",
        "## 1. Judge Assessment (Proportion of Answers Judged Correct / Incorrect)",
        "*Shows what the judge concluded across all evaluated items (works both with and without ground-truth labels).*",
        "",
        "| Method | Mode | Evaluated | Judged `CORRECT` | Judged `AMBIGUOUS` | Judged `INCORRECT` | `ABSTAIN` |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for m in methods:
        for scope_lbl, st in [("All", overall.get(m)), *[
            (mode_name, by_mode.get(mode_name, {}).get(m)) for mode_name in ("sql_only", "full_trace")
        ]]:
            if not st or st.get("n_items", 0) == 0:
                continue
            ass = st["assessment"]
            lines.append(
                f"| `{m}` | `{scope_lbl}` | {st['n_items']} | "
                f"{ass['CORRECT']} ({_pct(ass['prop_correct']):.1f}%) | "
                f"{ass['AMBIGUOUS']} ({_pct(ass['prop_ambiguous']):.1f}%) | "
                f"{ass['INCORRECT']} ({_pct(ass['prop_incorrect']):.1f}%) | "
                f"{ass['ABSTAIN']} ({_pct(ass['prop_abstain']):.1f}%) |"
            )

    has_labels = any(bool(st.get("has_ground_truth_labels")) for st in overall.values())
    if has_labels:
        lines.extend(
            [
                "",
                "## 2. Judge Accuracy vs. Ground-Truth Labels (All Evaluated Items)",
                "*Compares the judge's verdicts against known ground-truth labels (`correct` / `incorrect`).*",
                "",
                "| Method | Mode | Labelled `n` | Strict Acc | Lenient Acc | False-Pass | False-Fail | Cohen's κ |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for m in methods:
            for scope_lbl, st in [("All", overall.get(m)), *[
                (mode_name, by_mode.get(mode_name, {}).get(m)) for mode_name in ("sql_only", "full_trace")
            ]]:
                if not st or not st.get("has_ground_truth_labels"):
                    continue
                n_lab = st["n_correct"] + st["n_incorrect"] + st["n_ambiguous"]
                lines.append(
                    f"| `{m}` | `{scope_lbl}` | {n_lab} | "
                    f"{_pct(st['accuracy_all']):.2f}% | "
                    f"{_pct(_lenient_accuracy(st)):.2f}% | "
                    f"{_pct(st['false_pass']):.2f}% | "
                    f"{_pct(st['false_fail']):.2f}% | "
                    f"{st['cohens_kappa']:.3f} |"
                )

        if t2:
            lines.extend(
                [
                    "",
                    "### Test Common Subset `C` Accuracy (`T2`)",
                    "",
                    "| Method | Mode | `n` | Strict Acc | False-Pass | False-Fail | Escalation % | Abstain % |",
                    "|---|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for mode_name in ("sql_only", "full_trace"):
                for m in methods:
                    st = t2.get(mode_name, {}).get(m)
                    if not isinstance(st, dict) or "accuracy_all" not in st:
                        continue
                    lines.append(
                        f"| `{m}` | `{mode_name}` | {st['n_items']} | "
                        f"{_pct(st['accuracy_all']):.2f}% | "
                        f"{_pct(st['false_pass']):.2f}% | "
                        f"{_pct(st['false_fail']):.2f}% | "
                        f"{_pct(st['escalation_rate']):.1f}% | "
                        f"{_pct(st['abstain_rate']):.1f}% |"
                    )

    if by_set:
        lines.extend(
            [
                "",
                "## 3. Breakdown by Dataset Set",
                "",
                "| Set | Method | `n` | Judged `CORRECT` | Strict Acc | False-Pass |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for s_name in sorted(by_set.keys()):
            for m in methods:
                st = by_set[s_name].get(m)
                if not st or st.get("n_items", 0) == 0:
                    continue
                acc_str = f"{_pct(st['accuracy_all']):.1f}%" if st.get("has_ground_truth_labels") else "N/A"
                fp_str = f"{_pct(st['false_pass']):.1f}%" if st.get("has_ground_truth_labels") else "N/A"
                lines.append(
                    f"| `{s_name}` | `{m}` | {st['n_items']} | "
                    f"{_pct(st['assessment']['prop_correct']):.1f}% | {acc_str} | {fp_str} |"
                )

    if by_db:
        primary = methods[0] if methods else None
        if primary:
            lines.extend(
                [
                    "",
                    f"## 4. Breakdown by Database (`{primary}`)",
                    "",
                    "| Database | `n` | Judged `CORRECT` | Judged `INCORRECT` | Strict Acc |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for db_id in sorted(by_db.keys()):
                st = by_db[db_id].get(primary)
                if not st or st.get("n_items", 0) == 0:
                    continue
                acc_str = f"{_pct(st['accuracy_all']):.1f}%" if st.get("has_ground_truth_labels") else "N/A"
                lines.append(
                    f"| `{db_id}` | {st['n_items']} | "
                    f"{_pct(st['assessment']['prop_correct']):.1f}% | "
                    f"{_pct(st['assessment']['prop_incorrect']):.1f}% | {acc_str} |"
                )

    t4_methods = t4.get("by_method", {})
    if overall:
        lines.extend(
            [
                "",
                "## 5. Funnel Stage & Cost/Latency Summary",
                "",
                "| Method | Local Rules ($0) | Jev | Gemini Escalation | Total Billed | Mean Cost / Item | Latency p50 | Latency p95 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for m in methods:
            st = overall.get(m)
            if not st or st.get("n_items", 0) == 0:
                continue
            sb = st["stage_breakdown"]
            tot_cost = float(t4_methods.get(m, {}).get("total_billed_usd", 0.0))
            mean_cost = float(st.get("cost_billed_mean", 0.0))
            lines.append(
                f"| `{m}` | {_pct(sb['deterministic']):.1f}% | {_pct(sb['jev']):.1f}% | "
                f"{_pct(sb['escalation']):.1f}% | ${tot_cost:.4f} | ${mean_cost:.5f} | "
                f"{st['latency_p50']:.0f} ms | {st['latency_p95']:.0f} ms |"
            )

    lines.extend(
        [
            "",
            "## 6. Chart Guide",
            "- **`F1_judge_overview.png`**: Left panel shows the Judge Assessment (% `CORRECT`, `AMBIGUOUS`, `INCORRECT`, `ABSTAIN`). Right panel shows Judge Accuracy & Error Rates (when labelled) or Agent Pass Rate by Database (when unlabelled).",
            "- **`F2_cost_tradeoff.png`**: Left panel plots Accuracy (or % `CORRECT`) against Mean Billed Cost per item. Right panel shows what fraction of items resolved at `$0` Deterministic Rules, Jev, or Gemini Escalation.",
            "- **`F3_coincident.png`**: Compares Strict Accuracy and False-Pass Rate across benchmark sets (`S1`–`S4`) and runtime `coincident_result` items.",
            "- **`F4_large_tables.png`**: Accuracy, Correct-Pass, False-Pass, and False-Fail rates on `large_result` (>1,000 rows) and `mid_table_diff` items.",
            "- **`F5_final_answers.png`**: Accuracy, Correct-Pass, False-Pass, and False-Fail rates on `full_trace` prose final answers (`S3`, `S3m`, `S5`).",
            "- **`F6_jev_probabilities.png`**: Distribution of Jev `p(true)` confidence scores split by ground-truth label, showing the Reject / Escalate / Pass threshold zones.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
