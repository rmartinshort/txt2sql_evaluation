"""Acceptance tests for baselines, B6/B6+E cache sharing, threshold fitting, subset C, metrics, H6, and T1-T6/F2-F6."""

from __future__ import annotations

import json
from pathlib import Path

from txt2sql_eval_harness.baselines import judge_b1, judge_b2, judge_b_ibm_judge, judge_b_ibm_subset
from txt2sql_eval_harness.cache import ResponseCache
from txt2sql_eval_harness.clients import CallContext, RawModelResponse
from txt2sql_eval_harness.judge import judge_item
from txt2sql_eval_harness.metrics import (
    COVERAGE_LIMITED_MSG,
    compute_e5b_table,
    compute_method_metrics,
    compute_s3m_table,
    h6_reading,
)
from txt2sql_eval_harness.models import (
    EvalConfig,
    GroundTruth,
    LabelledItem,
    ModelCall,
    CallMeasure,
    ResultTable,
    Step,
    Trace,
    Verdict,
)
from txt2sql_eval_harness.reporting import generate_all_reports_and_figures
from txt2sql_eval_harness.runner import fit_thresholds, select_common_subset_c


def _make_verdict(
    trace_id: str,
    method: str,
    label: str,
    cost: float,
    *,
    mode: str = "full_trace",
    decided_by: str = "baseline",
    reason: str | None = None,
    escalated: bool = False,
) -> Verdict:
    return Verdict(
        trace_id=trace_id,
        method=method,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        label=label,  # type: ignore[arg-type]
        decided_by=decided_by,  # type: ignore[arg-type]
        reason=reason,
        escalated=escalated,
        method_ok=None,
        flags=[],
        claims=[],
        alignments=[],
        packet_sha256=None,
        b4_context=None,
        b4_rows_shown=None,
        b4_rows_total=None,
        calls=[],
        det_ms=1,
        esc_prep_ms=0,
        cost_billed_usd=cost,
        cost_conservative_usd=cost,
        latency_ms=10,
        spend_usd=cost,
        config_hash="cfg",
    )


def test_b2_and_b_ibm_subset_and_ibm_judge_and_coincident() -> None:
    cfg = EvalConfig()
    # Coincident query: equal result, different SQL (region='West' AND refunded=0 vs OR)
    r_eq = ResultTable(["s"], [(100,)], 1, True)
    tr = Trace(
        "c1",
        "gc1",
        "Total?",
        "",
        "d",
        "0",
        [Step(1, "SELECT SUM(a) FROM s WHERE region='West' OR refunded=0", result=r_eq)],
    )
    gt = GroundTruth("c1", ["SELECT SUM(a) FROM s WHERE region='West' AND refunded=0"], results=[r_eq])

    assert judge_b1(tr, gt, cfg).label == "INCORRECT"
    assert judge_b2(tr, gt, cfg).label == "CORRECT"
    assert judge_b_ibm_subset(tr, gt, cfg).label == "CORRECT"

    class _IbmMock:
        def __init__(self, txt: str) -> None:
            self.txt = txt

        def send(self, req, *, timeout):
            return RawModelResponse(http_ok=True, payload=self.txt, input_tokens=100, output_tokens=5)

    assert judge_b_ibm_judge(tr, gt, CallContext(cfg, True, clients={"ibm_judge": _IbmMock("Verdict: Yes")})).label == "CORRECT"
    assert judge_b_ibm_judge(tr, gt, CallContext(cfg, True, clients={"ibm_judge": _IbmMock("Verdict: No")})).label == "INCORRECT"
    v_maybe = judge_b_ibm_judge(tr, gt, CallContext(cfg, True, clients={"ibm_judge": _IbmMock("Answer: Maybe")}))
    assert v_maybe.label == "ABSTAIN" and v_maybe.reason == "maybe"

    # S3m early_support scored by B2 (last step u != R*) -> INCORRECT vs early_support_rev -> CORRECT
    r_gt = ResultTable(["k", "v"], [("A", 10), ("B", 20)], 2, True)
    r_u = ResultTable(["city"], [("Paris",), ("Rome",)], 2, True)
    gt_s3m = GroundTruth("s3m", ["SELECT k, v FROM t"], results=[r_gt])
    tr_early = Trace("es", "gs", "?", "", "d", "0", [Step(1, "SELECT k, v FROM t", result=r_gt), Step(2, "SELECT city FROM c", result=r_u)], final_answer="A=10, B=20")
    tr_rev = Trace("er", "gs", "?", "", "d", "0", [Step(1, "SELECT city FROM c", result=r_u), Step(2, "SELECT k, v FROM t", result=r_gt)], final_answer="A=10, B=20")
    assert judge_b2(tr_early, gt_s3m, cfg).label == "INCORRECT"
    assert judge_b2(tr_rev, gt_s3m, cfg).label == "CORRECT"


def test_b6_and_b6_plus_e_share_cache_hits_with_b6_and_he(tmp_path: Path) -> None:
    cfg = EvalConfig()
    cache = ResponseCache(tmp_path / "shared_cache.sqlite")
    res_gt = ResultTable(["a"], [(1,)], 1, True)
    res_gen = ResultTable(["a"], [(2,)], 1, True)
    # Non-rd_eligible so D4 does not fire -> goes to Jev -> Jev returns uncertain -> escalates
    gt = GroundTruth("b6", ["SELECT a FROM t ORDER BY a LIMIT 1"], results=[res_gt])
    tr = Trace("b6", "gb6", "Top a?", "", "d", "0", [Step(1, "SELECT a FROM t ORDER BY b LIMIT 1", result=res_gen)])

    net_calls = {"jev": 0, "gen": 0}

    class _RouterMock:
        def __init__(self, kind: str) -> None:
            self.kind = kind

        def send(self, req, *, timeout):
            net_calls[self.kind] += 1
            if self.kind == "jev":
                if "V" in req["questions"]:
                    return RawModelResponse(
                        http_ok=True,
                        payload={"V": {"correct": 0.4, "incorrect": 0.4, "ambiguous_reasonable": 0.2}},
                        input_tokens=100,
                        reported_cost_usd=0.0005,
                    )
                return RawModelResponse(
                    http_ok=True,
                    payload={
                        "Q1": 0.5,
                        "Q2": {
                            "equivalent_for_question": 0.5,
                            "acceptable_extra_rows": 0.2,
                            "missing_required_rows": 0.2,
                            "wrong_rows": 0.1,
                        },
                        "Q5": {"not_ambiguous": 1.0, "ambiguous_generated_reasonable": 0.0, "ambiguous_generated_unreasonable": 0.0},
                    },
                    input_tokens=100,
                    reported_cost_usd=0.001,
                )
            return RawModelResponse(
                http_ok=True,
                payload='{"label": "INCORRECT", "rationale": "different column"}',
                input_tokens=300,
                output_tokens=20,
                reported_cost_usd=0.05,
            )

    ctx = CallContext(
        cfg=cfg,
        allow_remote=True,
        cache=cache,
        clients={"jev": _RouterMock("jev"), "generative": _RouterMock("gen")},
    )
    v_he = judge_item(tr, gt, "H+E", ctx)
    v_b6 = judge_item(tr, gt, "B6", ctx)
    assert v_he.decided_by == "escalation"
    assert v_b6.label == "ABSTAIN" and v_b6.reason == "uncertain"
    assert net_calls == {"jev": 2, "gen": 1}

    # Now B6+E hits BOTH B6's Jev call AND H+E's Escalation call -> 0 new network sends!
    v_b6e = judge_item(tr, gt, "B6+E", ctx)
    assert v_b6e.label == "INCORRECT" and v_b6e.decided_by == "escalation"
    assert net_calls == {"jev": 2, "gen": 1}
    assert all(c.from_cache for c in v_b6e.calls)
    assert v_b6e.cost_billed_usd > 0.0  # cold measurements preserved!
    cache.close()


def test_fit_thresholds_respects_20_item_minimum_and_5pct_rule() -> None:
    # Build 25 SQL-only calibration items reaching Jev V (for B6 fitting)
    def _rec(i: int, lbl: str, p_corr: float, p_inc: float):
        it = LabelledItem(
            item_id=f"c{i}",
            group_id=f"g{i}",
            split="calibration",
            set="S1",
            family="f",
            slices=[],
            trace=Trace(f"c{i}", f"g{i}", "?", "", "d", "0", []),
            gt=GroundTruth(f"c{i}", ["SELECT 1"]),
            label=lbl,  # type: ignore[arg-type]
        )
        mc = ModelCall(
            stage="jev",
            model="jev-1.14.0",
            request_sha256="x",
            questions=["V"],
            answers={"V": {"correct": p_corr, "incorrect": p_inc, "ambiguous_reasonable": max(0.0, 1.0 - p_corr - p_inc)}},
            outcome="ok",
            invalid_reason=None,
            from_cache=False,
            cold=CallMeasure([], 10, 0.0, 0.0),
        )
        v = Verdict(f"c{i}", "B6", "sql_only", "CORRECT", "jev", None, False, None, [], [], [], None, None, None, None, [mc], 1, 0, 0.0, 0.0, 10, 0.0, "h")
        return (it, v, None)

    small_recs = [_rec(i, "correct", 0.6, 0.4) for i in range(10)]
    tau_small, info_small = fit_thresholds(small_recs, "B6", "sql_only")
    assert tau_small.tau_hi == 0.85  # fewer than 20 -> keeps default
    assert "fewer_than_20" in info_small["tau_hi_status"]

    # 25 items: at 0.50 -> 2/22 false pass (>5%); at 0.70 -> 0/20 false pass (<=5%) -> picks 0.70
    recs = (
        [_rec(i, "correct", 0.72, 0.28) for i in range(20)]
        + [_rec(20, "incorrect", 0.65, 0.35), _rec(21, "incorrect", 0.65, 0.35)]
        + [_rec(22, "incorrect", 0.20, 0.80)]
    )
    tau_fit, info_fit = fit_thresholds(recs, "B6", "sql_only")
    assert tau_fit.tau_hi == 0.70
    assert "fitted" in info_fit["tau_hi_status"]


def test_coverage_report_fixture_ambiguous_e5b_and_charts(tmp_path: Path) -> None:
    # Synthetic coverage fixture: 100 correct + 100 incorrect full-trace items
    # Method X (in H+E slot) abstains on all 200 at $0.0001/trace
    # Method Y (in B4 slot) decides all 200: 95 correct-pass, 5 false-fail, 5 false-pass at $0.10/trace
    items: list[LabelledItem] = []
    vx: dict[str, Verdict] = {}
    vy: dict[str, Verdict] = {}

    for i in range(100):
        it_c = LabelledItem(
            f"c_{i}", f"gc_{i}", "test", "S3", "correct", [],
            Trace(f"c_{i}", f"gc_{i}", "Q?", "", "d", "0", [Step(1, "SELECT 1", result=ResultTable(["a"], [(1,)], 1, True))], final_answer="1"),
            GroundTruth(f"c_{i}", ["SELECT 1"], results=[ResultTable(["a"], [(1,)], 1, True)]),
            "correct",
        )
        it_inc = LabelledItem(
            f"w_{i}", f"gw_{i}", "test", "S3", "wrong_number", [],
            Trace(f"w_{i}", f"gw_{i}", "Q?", "", "d", "0", [Step(1, "SELECT 1", result=ResultTable(["a"], [(1,)], 1, True))], final_answer="2"),
            GroundTruth(f"w_{i}", ["SELECT 1"], results=[ResultTable(["a"], [(1,)], 1, True)]),
            "incorrect",
        )
        items.extend([it_c, it_inc])
        vx[it_c.item_id] = _make_verdict(it_c.item_id, "H+E", "ABSTAIN", 0.0001, decided_by="abstain", reason="uncertain")
        vx[it_inc.item_id] = _make_verdict(it_inc.item_id, "H+E", "ABSTAIN", 0.0001, decided_by="abstain", reason="uncertain")

        vy[it_c.item_id] = _make_verdict(it_c.item_id, "B4", "CORRECT" if i < 95 else "INCORRECT", 0.10)
        vy[it_inc.item_id] = _make_verdict(it_inc.item_id, "B4", "CORRECT" if i < 5 else "INCORRECT", 0.10)

    mx = compute_method_metrics(items, vx, bootstrap_resamples=50)
    my = compute_method_metrics(items, vy, bootstrap_resamples=50)

    assert mx["counts"]["correct"] == {"CORRECT": 0, "INCORRECT": 0, "AMBIGUOUS": 0, "ABSTAIN": 100}
    assert mx["counts"]["incorrect"] == {"CORRECT": 0, "INCORRECT": 0, "AMBIGUOUS": 0, "ABSTAIN": 100}
    assert mx["false_pass"] == 0.0 and mx["false_fail"] == 0.0
    assert mx["correct_pass_strict"] == 0.0 and mx["coverage_correct"] == 0.0 and mx["coverage_incorrect"] == 0.0
    assert my["correct_pass_strict"] == 0.95 and my["coverage_correct"] == 1.0

    reading = h6_reading(mx, my)
    assert reading["status"] == "coverage_limited"
    assert reading["message"] == COVERAGE_LIMITED_MSG

    # Same fixture with 30 correct items marked AMBIGUOUS -> lenient 0.30, strict 0.0, still coverage_limited
    vx_amb = dict(vx)
    for i in range(30):
        vx_amb[f"c_{i}"] = _make_verdict(f"c_{i}", "H+E", "AMBIGUOUS", 0.0001, decided_by="jev")
    mx_amb = compute_method_metrics(items, vx_amb, bootstrap_resamples=0)
    assert mx_amb["counts"]["correct"]["AMBIGUOUS"] == 30
    assert mx_amb["correct_pass_strict"] == 0.0
    assert mx_amb["correct_pass_lenient"] == 0.30
    assert mx_amb["coverage_correct"] == 0.0
    assert h6_reading(mx_amb, my)["status"] == "coverage_limited"

    # E5b T5 calculation: wrong_number excludes already-CORRECT original from attack_success denominator
    e5b_items = [
        LabelledItem("o1", "g1", "test", "S3", "wrong_number", [], Trace("o1", "g1", "?", "", "d", "0", [], "a"), GroundTruth("o1", ["SELECT 1"]), "incorrect"),
        LabelledItem("c1", "g1", "test", "S3", "wrong_number", [], Trace("c1", "g1", "?", "", "d", "0", [], "a\n\nP"), GroundTruth("o1", ["SELECT 1"]), "incorrect", pair_of="o1"),
        LabelledItem("o2", "g2", "test", "S3", "wrong_number", [], Trace("o2", "g2", "?", "", "d", "0", [], "a"), GroundTruth("o2", ["SELECT 1"]), "incorrect"),
        LabelledItem("c2", "g2", "test", "S3", "wrong_number", [], Trace("c2", "g2", "?", "", "d", "0", [], "a\n\nP"), GroundTruth("o2", ["SELECT 1"]), "incorrect", pair_of="o2"),
    ]
    ok_call = ModelCall("b4", "m", "h", [], {"label": "CORRECT"}, "ok", None, False, CallMeasure([], 1, 0.0, 0.0))
    e5b_verdicts = {
        "B4": {
            "o1": Verdict("o1", "B4", "full_trace", "CORRECT", "baseline", None, False, None, [], [], [], None, None, None, None, [ok_call], 1, 0, 0.0, 0.0, 1, 0.0, "h"),
            "c1": Verdict("c1", "B4", "full_trace", "CORRECT", "baseline", None, False, None, [], [], [], None, None, None, None, [ok_call], 1, 0, 0.0, 0.0, 1, 0.0, "h"),
            "o2": Verdict("o2", "B4", "full_trace", "INCORRECT", "baseline", None, False, None, [], [], [], None, None, None, None, [ok_call], 1, 0, 0.0, 0.0, 1, 0.0, "h"),
            "c2": Verdict("c2", "B4", "full_trace", "CORRECT", "baseline", None, False, None, [], [], [], None, None, None, None, [ok_call], 1, 0, 0.0, 0.0, 1, 0.0, "h"),
        }
    }
    t5 = compute_e5b_table(e5b_items, e5b_verdicts)
    wn = t5["B4"]["wrong_number"]
    assert wn["pre_existing_correct_excluded"] == 1
    assert wn["attack_success_num"] == 1 and wn["attack_success_den"] == 1

    # Full report + figures generation
    rep_dir = tmp_path / "rep"
    rep = generate_all_reports_and_figures(
        rep_dir,
        items,
        select_common_subset_c(items) or items,
        {"H+E": vx, "B4": vy},
        e5b_items=e5b_items,
        e5b_verdicts=e5b_verdicts,
        bootstrap_resamples=20,
    )
    assert "coverage_limited" == rep["T2"]["full_trace"]["h6_reading_HE_vs_B4"]["status"]
    assert (rep_dir / "F2_cost_tradeoff.png").exists()
    assert (rep_dir / "F6_jev_probabilities.png").exists()


def test_b2_and_ibm_subset_abstain_on_incomplete_tables() -> None:
    cfg = EvalConfig()
    comp = ResultTable(["a"], [(1,)], 1, True)
    incomp = ResultTable(["a"], [(1,)], 10, False)
    tr_incomp = Trace("inc1", "g1", "?", "", "d", "0", [Step(1, "SELECT a FROM t", result=incomp)])
    gt_comp = GroundTruth("inc1", ["SELECT a FROM t"], results=[comp])

    v_b2 = judge_b2(tr_incomp, gt_comp, cfg)
    assert v_b2.label == "ABSTAIN" and v_b2.reason == "incomplete_evidence"
    v_ibm = judge_b_ibm_subset(tr_incomp, gt_comp, cfg)
    assert v_ibm.label == "ABSTAIN" and v_ibm.reason == "incomplete_evidence"

    tr_comp = Trace("inc2", "g2", "?", "", "d", "0", [Step(1, "SELECT a FROM t", result=comp)])
    gt_incomp = GroundTruth("inc2", ["SELECT a FROM t"], results=[incomp])
    assert judge_b2(tr_comp, gt_incomp, cfg).reason == "incomplete_evidence"
    assert judge_b_ibm_subset(tr_comp, gt_incomp, cfg).reason == "incomplete_evidence"
