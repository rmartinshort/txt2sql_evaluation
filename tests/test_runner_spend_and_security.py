"""Acceptance tests for egress gate, network spy, manifest canary, cache-only, spend checkpoint, retries, and resume."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sqlite3

import pytest

from txt2sql_eval_harness.agent import run_reference_agent
from txt2sql_eval_harness.baselines import judge_b1, judge_b2, judge_b4, judge_b_ibm_judge, judge_b_ibm_subset
from txt2sql_eval_harness.cache import ResponseCache
from txt2sql_eval_harness.clients import CallContext, RawModelResponse, call_model
from txt2sql_eval_harness.data import UnadmittedInputError, egress_decision, load_manifest, record_manifest_entry, save_manifest
from txt2sql_eval_harness.judge import derive_h_from_he, judge_item
from txt2sql_eval_harness.metrics import load_records
from txt2sql_eval_harness.models import (
    EvalConfig,
    GroundTruth,
    LabelledItem,
    ResultTable,
    Step,
    Trace,
)
from txt2sql_eval_harness.runner import IncompatibleRunDirError, run_experiment
from txt2sql_eval_harness.spend import ExperimentBusyError, SpendCheckpointCorruptError, SpendLedger


def _sql_item(item_id: str = "it1", *, coinciding: bool = False) -> LabelledItem:
    gt_sql = "SELECT COUNT(*) FROM t WHERE a = 1"
    gen_sql = "SELECT COUNT(*) FROM t WHERE b = 2" if coinciding else "SELECT COUNT(*) FROM t WHERE a = 1 AND c > 0"
    res = ResultTable(["c"], [(10,)], 1, True)
    return LabelledItem(
        item_id=item_id,
        group_id=f"g_{item_id}",
        split="test",
        set="S4",
        family="flex",
        slices=[],
        trace=Trace(item_id, f"g_{item_id}", "How many?", "", "d", "0", [Step(1, gen_sql, result=res)]),
        gt=GroundTruth(item_id, [gt_sql], results=[res]),
        label="correct",
    )


def test_network_spy_local_only_zero_connections_zero_key_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    socket_calls = 0

    def _spy_fail(*args, **kwargs):
        nonlocal socket_calls
        socket_calls += 1
        raise AssertionError("Network access attempted under local-only!")

    monkeypatch.setattr(socket.socket, "connect", _spy_fail)
    monkeypatch.setattr(socket, "create_connection", _spy_fail)
    monkeypatch.setattr(socket, "getaddrinfo", _spy_fail)

    key_reads = 0
    real_get = os.environ.get

    def _spy_env_get(k, default=None):
        nonlocal key_reads
        if k in ("TYPESAFE_API_KEY", "GEMINI_API_KEY"):
            key_reads += 1
        return real_get(k, default)

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-dummy-format-1234")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-dummy-format-5678")
    monkeypatch.setattr(os.environ, "get", _spy_env_get)

    db_path = tmp_path / "d.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t(a INT, b INT, c INT);")
    conn.close()

    cfg = EvalConfig()
    item = _sql_item("spy1")
    ctx = CallContext(cfg=cfg, allow_remote=False)

    # All remote judges must abstain remote_disabled without creating cache or touching env/socket
    for meth in ("H+E", "B6", "B6+E"):
        v = judge_item(item.trace, item.gt, meth, ctx)  # type: ignore[arg-type]
        assert v.label == "ABSTAIN" and v.reason == "remote_disabled"
        assert all(c.outcome == "remote_disabled" and not c.cold.attempts for c in v.calls)

    v_b4 = judge_b4(item.trace, item.gt, ctx)
    assert v_b4.label == "ABSTAIN" and v_b4.reason == "remote_disabled"

    v_ibm = judge_b_ibm_judge(item.trace, item.gt, ctx)
    assert v_ibm.label == "ABSTAIN" and v_ibm.reason == "remote_disabled"

    # Generative prompt check & Jev calibration style call
    pc = call_model("prompt_check", "generative", cfg.generative_model, {"prompt": "hi"}, [], ctx)
    assert pc.outcome == "remote_disabled"

    # Reference agent returns remote_disabled and writes no trace
    tr_ag, _, err_ag = run_reference_agent("ag1", "gag1", "Count?", "", "d", "0", db_path, ctx)
    assert tr_ag is None and err_ag == "remote_disabled"

    # Local baselines still run identically
    assert judge_b1(item.trace, item.gt, cfg).label == "INCORRECT"
    assert judge_b2(item.trace, item.gt, cfg).label == "CORRECT"
    assert judge_b_ibm_subset(item.trace, item.gt, cfg).label == "CORRECT"

    # Runner under local_only must not create cache/responses.sqlite
    run_dir = tmp_path / "run_local"
    root = tmp_path / "exp_local"
    summary = run_experiment([item], ["H+E", "B4", "B-IBM-judge", "B2"], cfg, root, run_dir, local_only=True)
    assert not summary.allow_remote
    assert not (root / "cache" / "responses.sqlite").exists()

    assert socket_calls == 0
    assert key_reads == 0


def test_unadmitted_file_and_edited_text_canary(tmp_path: Path) -> None:
    manifest_path = tmp_path / "data" / "manifest.json"
    trace_file = tmp_path / "data" / "s5_traces.jsonl"
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text('{"trace_id": "1", "final_answer": "10 orders"}\n', encoding="utf-8")

    # 1. File absent from manifest -> refused without --local-only, allowed locally with it
    save_manifest(manifest_path, {"datasets": {}, "derivatives": {}, "databases": {}})
    with pytest.raises(UnadmittedInputError, match="unadmitted_input"):
        egress_decision([trace_file], manifest_path, local_only=False, base_dir=tmp_path)
    assert egress_decision([trace_file], manifest_path, local_only=True, base_dir=tmp_path) is False

    # 2. Record into manifest -> admitted -> edit one string ("10" -> "11") -> refused!
    record_manifest_entry(manifest_path, trace_file, "harness_derivative", base_dir=tmp_path)
    assert egress_decision([trace_file], manifest_path, local_only=False, base_dir=tmp_path) is True

    # Canary edit
    trace_file.write_text('{"trace_id": "1", "final_answer": "11 orders"}\n', encoding="utf-8")
    with pytest.raises(UnadmittedInputError, match="unadmitted_input"):
        egress_decision([trace_file], manifest_path, local_only=False, base_dir=tmp_path)
    assert egress_decision([trace_file], manifest_path, local_only=True, base_dir=tmp_path) is False


def test_b4_timeout_retry_reported_usage_and_cache_rerun(tmp_path: Path) -> None:
    cfg = EvalConfig()
    item = _sql_item("b4retry")
    cache = ResponseCache(tmp_path / "cache.sqlite")
    ledger = SpendLedger(tmp_path, cap_usd=150.0)

    clock_t = [0.0]

    def _clock() -> float:
        return clock_t[0]

    attempt_idx = [0]

    class _FlakyB4:
        def send(self, req, *, timeout):
            attempt_idx[0] += 1
            if attempt_idx[0] == 1:
                clock_t[0] += 60.0
                return RawModelResponse(http_ok=False, status_code=None, transport_error="timeout")
            clock_t[0] += 5.0
            return RawModelResponse(
                http_ok=True,
                payload='{"label": "CORRECT", "rationale": "ok"}',
                input_tokens=100,
                output_tokens=20,
                reported_cost_usd=0.0123,
            )

    try:
        ctx = CallContext(
            cfg=cfg,
            allow_remote=True,
            cache=cache,
            ledger=ledger,
            clients={"generative": _FlakyB4()},
            sleep_fn=lambda _: None,
            clock_fn=_clock,
        )
        v1 = judge_b4(item.trace, item.gt, ctx)
        assert v1.label == "CORRECT"
        assert v1.latency_ms >= 65_000
        call1 = v1.calls[0]
        assert not call1.from_cache
        assert len(call1.cold.attempts) == 2
        assert call1.cold.attempts[0].billing == "ambiguous"
        assert call1.cold.attempts[1].billing == "reported"
        assert call1.cold.billed_usd == pytest.approx(0.0123)
        assert v1.cost_conservative_usd > v1.cost_billed_usd

        # Cache rerun: identical cold latency & cost, from_cache=True, spend_usd=0
        ctx_cached = CallContext(
            cfg=cfg,
            allow_remote=True,
            cache_only=True,
            cache=cache,
            clock_fn=_clock,
        )
        v2 = judge_b4(item.trace, item.gt, ctx_cached)
        assert v2.label == "CORRECT"
        assert v2.calls[0].from_cache is True
        assert v2.calls[0].cold.wall_ms == v1.calls[0].cold.wall_ms
        assert v2.latency_ms == v1.latency_ms
        assert v2.cost_billed_usd == v1.cost_billed_usd
        assert v2.cost_conservative_usd == v1.cost_conservative_usd
        assert v2.spend_usd == 0.0
    finally:
        ledger.close()
        cache.close()


def test_jev_timeouts_invalid_responses_and_escalation_derive_h(tmp_path: Path) -> None:
    cfg = EvalConfig()
    item = _sql_item("jevfails")

    clock_t = [0.0]
    slept: list[float] = []

    class _AlwaysTimeoutJev:
        def send(self, req, *, timeout):
            clock_t[0] += 10.0
            return RawModelResponse(http_ok=False, status_code=None, transport_error="timeout")

    def _sleep(s: float) -> None:
        slept.append(s)

    ctx_to = CallContext(
        cfg=cfg,
        allow_remote=True,
        cache=ResponseCache(tmp_path / "c1.sqlite"),
        clients={"jev": _AlwaysTimeoutJev()},
        sleep_fn=_sleep,
        clock_fn=lambda: clock_t[0],
    )
    v_to = judge_item(item.trace, item.gt, "H+E", ctx_to)
    assert v_to.label == "ABSTAIN" and v_to.reason == "backend_error"
    assert not v_to.escalated
    assert slept == [1.0, 2.0]
    assert v_to.latency_ms >= 33_000
    assert v_to.cost_billed_usd == 0.0
    assert len(v_to.calls[0].cold.attempts) == 3
    assert all(a.billing == "ambiguous" for a in v_to.calls[0].cold.attempts)
    ctx_to.cache.close()  # type: ignore[union-attr]

    # Invalid Jev (missing question, then string probability) -> invalid_response, not cached, no escalation
    inv_responses = [
        RawModelResponse(http_ok=True, payload={"Q1": 0.9}),  # missing Q5
        RawModelResponse(
            http_ok=True,
            payload={
                "Q1": "0.9",  # invalid string!
                "Q5": {"not_ambiguous": 1.0, "ambiguous_generated_reasonable": 0.0, "ambiguous_generated_unreasonable": 0.0},
            },
        ),
    ]

    class _InvalidJev:
        def __init__(self) -> None:
            self.i = 0

        def send(self, req, *, timeout):
            r = inv_responses[self.i]
            self.i += 1
            return r

    cache2 = ResponseCache(tmp_path / "c2.sqlite")
    ctx_inv = CallContext(cfg=cfg, allow_remote=True, cache=cache2, clients={"jev": _InvalidJev()}, sleep_fn=lambda _: None)
    v_inv = judge_item(item.trace, item.gt, "H+E", ctx_inv)
    assert v_inv.label == "ABSTAIN" and v_inv.reason == "invalid_response"
    assert cache2.get(v_inv.calls[0].request_sha256) is None
    cache2.close()

    # Generative wrong-case label "Correct" twice -> invalid_response
    class _LowerCaseGen:
        def send(self, req, *, timeout):
            return RawModelResponse(http_ok=True, payload='{"label": "Correct", "rationale": "x"}')

    ctx_b4_inv = CallContext(cfg=cfg, allow_remote=True, clients={"generative": _LowerCaseGen()})
    assert judge_b4(item.trace, item.gt, ctx_b4_inv).reason == "invalid_response"

    # H+E escalates (0.4s Jev + 8s Escalation) -> derive_h_from_he drops escalation
    class _UncertainJev:
        def send(self, req, *, timeout):
            clock_t[0] += 0.4
            return RawModelResponse(
                http_ok=True,
                payload={
                    "Q1": 0.5,
                    "Q5": {"not_ambiguous": 1.0, "ambiguous_generated_reasonable": 0.0, "ambiguous_generated_unreasonable": 0.0},
                },
                input_tokens=100,
                reported_cost_usd=0.001,
            )

    class _EscGen:
        def send(self, req, *, timeout):
            clock_t[0] += 8.0
            return RawModelResponse(
                http_ok=True,
                payload='{"label": "CORRECT", "rationale": "ok"}',
                input_tokens=200,
                output_tokens=20,
                reported_cost_usd=0.05,
            )

    ctx_esc = CallContext(
        cfg=cfg,
        allow_remote=True,
        clients={"jev": _UncertainJev(), "generative": _EscGen()},
        clock_fn=lambda: clock_t[0],
    )
    v_he = judge_item(item.trace, item.gt, "H+E", ctx_esc)
    assert v_he.label == "CORRECT" and v_he.escalated
    assert v_he.cost_billed_usd == pytest.approx(0.051)
    v_h = derive_h_from_he(v_he)
    assert v_h.label == "ABSTAIN" and v_h.reason == "uncertain"
    assert not v_h.escalated
    assert v_h.esc_prep_ms == 0
    assert v_h.cost_billed_usd == pytest.approx(0.001)
    assert v_h.latency_ms == v_h.det_ms + v_he.calls[0].cold.wall_ms


def test_spend_checkpoint_orphans_cap_lock_and_runner_resume(tmp_path: Path) -> None:
    root = tmp_path / "exp"

    # 1. Locking: two concurrent SpendLedger on same root -> second raises experiment_busy
    with SpendLedger(root, cap_usd=150.0) as l1:
        with pytest.raises(ExperimentBusyError, match="experiment_busy"):
            SpendLedger(root, cap_usd=150.0)
        # Crash after reserve before settle -> orphan folding on reopen!
        assert l1.reserve("b4", 0.05)

    with SpendLedger(root, cap_usd=150.0) as l2:
        assert l2.checkpoint.reserved is None
        assert l2.checkpoint.by_stage["orphan"] == pytest.approx(0.05)
        assert l2.checkpoint.committed_usd == pytest.approx(0.05)

    # 2. Cap enforcement: committed = 149.98, upper = 0.05 -> stop without writing record
    root_cap = tmp_path / "exp_cap"
    with SpendLedger(root_cap, cap_usd=150.0) as lc:
        lc.reserve("b4", 149.98)
        lc.settle(149.98)

    # Inject mock generative client that costs >= $0.05 upper bound (say 15k tokens / high price)
    item = _sql_item("cap1")
    cfg_expensive = EvalConfig()
    # Set pricing so local request upper bound exceeds $0.02
    from txt2sql_eval_harness.models import ModelPricing

    cfg_expensive = EvalConfig(
        pricing={cfg_expensive.generative_model: ModelPricing(50.0, 50.0)}
    )
    run_dir = tmp_path / "run_cap"

    class _GoodGen:
        def send(self, req, *, timeout):
            return RawModelResponse(
                http_ok=True,
                payload='{"label": "CORRECT", "rationale": "ok"}',
                input_tokens=100,
                output_tokens=20,
                reported_cost_usd=0.005,
            )

    res_stop = run_experiment(
        [item],
        ["B4"],
        cfg_expensive,
        root_cap,
        run_dir,
        cap_usd=150.0,
        clients={"generative": _GoodGen()},
    )
    assert res_stop.stopped_on_spend_cap
    assert res_stop.written_count == 0
    assert (run_dir / "records.jsonl").read_text(encoding="utf-8") == ""

    # Raise cap and resume -> judges and writes it!
    res_ok = run_experiment(
        [item],
        ["B4"],
        cfg_expensive,
        root_cap,
        run_dir,
        cap_usd=200.0,
        clients={"generative": _GoodGen()},
    )
    assert not res_ok.stopped_on_spend_cap
    assert res_ok.written_count == 1

    # Resume again -> skips already written!
    res_idem = run_experiment(
        [item],
        ["B4"],
        cfg_expensive,
        root_cap,
        run_dir,
        cap_usd=200.0,
        clients={"generative": _GoodGen()},
    )
    assert res_idem.written_count == 0 and res_idem.skipped_existing_count == 1

    # Append torn last line to records.jsonl, then resume with another method -> drops torn line cleanly!
    with open(run_dir / "records.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"item_id": "broken_torn_line')
    res_b2 = run_experiment(
        [item],
        ["B4", "B2"],
        cfg_expensive,
        root_cap,
        run_dir,
        cap_usd=200.0,
        clients={"generative": _GoodGen()},
    )
    assert res_b2.written_count == 1
    loaded = load_records(run_dir / "records.jsonl", include_h=False)
    assert set(loaded.keys()) == {"B4", "B2"}

    # Config change (tau_hi) -> incompatible_run_dir
    cfg_changed = EvalConfig()
    from txt2sql_eval_harness.models import ThresholdPair

    cfg_changed.tau["H"]["sql_only"] = ThresholdPair(0.90, 0.15)
    with pytest.raises(IncompatibleRunDirError, match="incompatible_run_dir"):
        run_experiment([item], ["B4"], cfg_changed, root_cap, run_dir, cap_usd=200.0)


def test_cache_only_rerun_reproduces_verdicts_and_misses_on_failed_calls(tmp_path: Path) -> None:
    root = tmp_path / "exp_replay"
    run_src = tmp_path / "run_src"
    cfg = EvalConfig()

    it_ok = _sql_item("it_ok")
    it_fail = _sql_item("it_fail")

    calls_seen = []

    class _MixedJev:
        def send(self, req, *, timeout):
            calls_seen.append(req["state"])
            if "it_ok" in req["state"] or len(calls_seen) <= 1:
                return RawModelResponse(
                    http_ok=True,
                    payload={
                        "Q1": 0.95,
                        "Q5": {"not_ambiguous": 1.0, "ambiguous_generated_reasonable": 0.0, "ambiguous_generated_unreasonable": 0.0},
                    },
                    input_tokens=100,
                    reported_cost_usd=0.0002,
                )
            return RawModelResponse(http_ok=False, status_code=503)

    # Use distinct SQL/questions so request_sha256 differs between it_ok and it_fail
    it_ok = LabelledItem(
        item_id="a_ok",
        group_id="gok",
        split="test",
        set="S4",
        family="flex",
        slices=[],
        trace=Trace("a_ok", "gok", "Question OK?", "", "d", "0", [Step(1, "SELECT COUNT(*) FROM t WHERE a=1", result=ResultTable(["c"], [(10,)], 1, True))]),
        gt=GroundTruth("a_ok", ["SELECT COUNT(*) FROM t WHERE a=2"], results=[ResultTable(["c"], [(10,)], 1, True)]),
        label="correct",
    )
    it_fail = LabelledItem(
        item_id="b_fail",
        group_id="gfail",
        split="test",
        set="S4",
        family="flex",
        slices=[],
        trace=Trace("b_fail", "gfail", "Question FAIL?", "", "d", "0", [Step(1, "SELECT COUNT(*) FROM t WHERE x=1", result=ResultTable(["c"], [(10,)], 1, True))]),
        gt=GroundTruth("b_fail", ["SELECT COUNT(*) FROM t WHERE x=2"], results=[ResultTable(["c"], [(10,)], 1, True)]),
        label="correct",
    )

    run_experiment(
        [it_ok, it_fail],
        ["H+E"],
        cfg,
        root,
        run_src,
        clients={"jev": _MixedJev()},
        sleep_fn=lambda _: None,
    )
    src_recs = load_records(run_src / "records.jsonl", include_h=True)
    assert src_recs["H+E"]["a_ok"].label == "CORRECT"
    assert src_recs["H+E"]["b_fail"].label == "ABSTAIN" and src_recs["H+E"]["b_fail"].reason == "backend_error"

    # Cache-only rerun in fresh directory, NO clients and NO API keys!
    run_replay = tmp_path / "run_cache_only"
    run_experiment(
        [it_ok, it_fail],
        ["H+E"],
        cfg,
        root,
        run_replay,
        cache_only=True,
    )
    rep_recs = load_records(run_replay / "records.jsonl", include_h=True)
    assert rep_recs["H+E"]["a_ok"].label == "CORRECT"
    assert rep_recs["H+E"]["a_ok"].cost_billed_usd == src_recs["H+E"]["a_ok"].cost_billed_usd
    assert rep_recs["H+E"]["a_ok"].spend_usd == 0.0
    assert rep_recs["H+E"]["b_fail"].label == "ABSTAIN" and rep_recs["H+E"]["b_fail"].reason == "cache_miss"

    # Spend checkpoint corrupt error check
    (root / "spend.json").write_text("{corrupt", encoding="utf-8")
    with pytest.raises(SpendCheckpointCorruptError, match="spend_checkpoint_corrupt"):
        SpendLedger(root)


def test_preflight_missing_api_key_and_cli_missing_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from txt2sql_eval_harness.cli import main as cli_main
    from txt2sql_eval_harness.codec import encode_labelled_item

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("txt2sql_eval_harness.runner._get_api_key", lambda _k: None)
    monkeypatch.chdir(tmp_path)

    cfg = EvalConfig()
    item = _sql_item("k1")
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        run_experiment([item], ["H+E"], cfg, tmp_path / "exp_k", tmp_path / "run_k")

    # CLI run without --local-only and missing manifest must fail closed
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(json.dumps(encode_labelled_item(item)) + "\n", encoding="utf-8")
    with pytest.raises(UnadmittedInputError, match="manifest not found"):
        cli_main([
            "run",
            "--items", str(items_path),
            "--manifest", str(tmp_path / "missing_manifest.json"),
            "--root", str(tmp_path / "exp_cli"),
            "--run-dir", str(tmp_path / "run_cli"),
        ])


def test_eval_config_retry_policy_and_incompatible_run_dir_diag(tmp_path: Path) -> None:
    cfg = EvalConfig(jev_max_attempts=1)
    attempts_seen = 0

    class _FailOnceJev:
        def send(self, req, *, timeout):
            nonlocal attempts_seen
            attempts_seen += 1
            return RawModelResponse(http_ok=False, status_code=503)

    ctx = CallContext(cfg=cfg, allow_remote=True, clients={"jev": _FailOnceJev()}, sleep_fn=lambda _: None)
    call = call_model("jev", "jev", cfg.jev_model, {"state": "s", "questions": {}}, [], ctx)
    assert call.outcome == "backend_error"
    assert attempts_seen == 1

    # IncompatibleRunDirError lists differing fields
    root = tmp_path / "exp_diag"
    run_dir = tmp_path / "run_diag"
    item = _sql_item("d1")
    run_experiment([item], ["B1"], cfg, root, run_dir, local_only=True)
    with pytest.raises(IncompatibleRunDirError, match="config_hash"):
        run_experiment([item], ["B1"], EvalConfig(jev_max_attempts=2), root, run_dir, local_only=True)


def test_flex_loader_skips_invalid_labels(tmp_path: Path) -> None:
    from txt2sql_eval_harness.data import load_flex_dataset

    dbs_dir = tmp_path / "dbs"
    (dbs_dir / "d1").mkdir(parents=True)
    conn = sqlite3.connect(str(dbs_dir / "d1" / "d1.sqlite"))
    conn.execute("CREATE TABLE t(a INT)")
    conn.close()

    fp = tmp_path / "fp.jsonl"
    fn = tmp_path / "fn.jsonl"
    fp.write_text(
        json.dumps({"id": "1", "db_id": "d1", "question": "Q1", "gt_sql": "SELECT 1", "pred_sql": "SELECT 1", "human_judge": "yes"}) + "\n"
        + json.dumps({"id": "2", "db_id": "d1", "question": "Q2", "gt_sql": "SELECT 1", "pred_sql": "SELECT 1", "human_judge": "unknown_val"}) + "\n",
        encoding="utf-8",
    )
    fn.write_text("", encoding="utf-8")
    items, stats = load_flex_dataset(fp, fn, dbs_dir)
    assert len(items) == 1 and items[0].label == "correct"
    assert stats["invalid_label"] == 1


def test_concurrent_runner_and_calibration_threshold_fitting(tmp_path: Path) -> None:
    cfg = EvalConfig()
    cal_items: list[LabelledItem] = []
    # Build 22 correct + 5 incorrect calibration items where Jev gives Q1=0.60 on correct
    # (below default 0.85, above fitted 0.50) and Q1=0.10 on incorrect -> fits tau_hi=0.50!
    for i in range(22):
        res = ResultTable(["c"], [(10 + i,)], 1, True)
        cal_items.append(
            LabelledItem(
                f"cal_ok_{i}",
                f"g_cal_ok_{i}",
                "calibration",
                "S1",
                "add_columns",
                [],
                Trace(f"cal_ok_{i}", f"g_cal_ok_{i}", f"Q {i}?", "", "d", "0", [Step(1, f"SELECT a + {i} FROM g1", result=res)]),
                GroundTruth(f"cal_ok_{i}", [f"SELECT a + {i} FROM g2"], results=[res]),
                "correct",
            )
        )
    for i in range(5):
        res = ResultTable(["c"], [(200 + i,)], 1, True)
        cal_items.append(
            LabelledItem(
                f"cal_bad_{i}",
                f"g_cal_bad_{i}",
                "calibration",
                "S2",
                "agg_swap",
                [],
                Trace(f"cal_bad_{i}", f"g_cal_bad_{i}", f"Bad Q {i}?", "", "d", "0", [Step(1, f"SELECT b + {i} FROM g1 ORDER BY x LIMIT 1", result=res)]),
                GroundTruth(f"cal_bad_{i}", [f"SELECT b + {i} FROM g2 ORDER BY y LIMIT 1"], results=[res]),
                "incorrect",
            )
        )

    res_test = ResultTable(["c"], [(999,)], 1, True)
    test_item = LabelledItem(
        "test_0",
        "g_test_0",
        "test",
        "S1",
        "add_columns",
        [],
        Trace("test_0", "g_test_0", "Test Q?", "", "d", "0", [Step(1, "SELECT z FROM g1 ORDER BY k LIMIT 1", result=res_test)]),
        GroundTruth("test_0", ["SELECT z FROM g2 ORDER BY m LIMIT 1"], results=[res_test]),
        "correct",
    )

    esc_calls: list[str] = []

    class _MockJevCal:
        def send(self, req, *, timeout):
            state = req["state"]
            q1 = 0.10 if "Bad Q" in state else 0.60
            return RawModelResponse(
                http_ok=True,
                payload={
                    "Q1": q1,
                    "Q5": {
                        "not_ambiguous": 0.90,
                        "ambiguous_generated_reasonable": 0.05,
                        "ambiguous_generated_unreasonable": 0.05,
                    },
                },
                input_tokens=100,
                reported_cost_usd=0.0001,
            )

    class _MockGenCal:
        def send(self, req, *, timeout):
            esc_calls.append("esc")
            return RawModelResponse(
                http_ok=True,
                payload='{"label": "CORRECT", "rationale": "ok"}',
                input_tokens=150,
                output_tokens=20,
                reported_cost_usd=0.01,
            )

    root = tmp_path / "exp_cal"
    run_dir = tmp_path / "run_cal"
    summary = run_experiment(
        cal_items + [test_item],
        ["H+E"],
        cfg,
        root,
        run_dir,
        clients={"jev": _MockJevCal(), "generative": _MockGenCal()},
        sleep_fn=lambda _: None,
        concurrency=4,
    )
    assert summary.written_count == 28
    assert not summary.stopped_on_spend_cap
    fitted = json.loads((run_dir / "fitted_thresholds.json").read_text(encoding="utf-8"))
    assert fitted["H+E:sql_only"]["tau_hi"] == 0.5

    recs = load_records(run_dir / "records.jsonl", include_h=True)
    # test_0 had Q1=0.60, which meets fitted tau_hi=0.50 -> decided directly by Jev without escalation!
    assert recs["H+E"]["test_0"].label == "CORRECT"
    assert recs["H+E"]["test_0"].decided_by == "jev"
    assert not recs["H+E"]["test_0"].escalated
