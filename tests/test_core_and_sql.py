"""Acceptance tests for S-core and S-sql: codec, cmp, N, volatile/det/rd_eligible, sandbox, and D-rules."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import tempfile

import pytest

from txt2sql_eval_harness.clients import CallContext
from txt2sql_eval_harness.codec import decode_cell, decode_result_table, encode_cell, encode_result_table
from txt2sql_eval_harness.comparator import (
    DEFAULT_JUDGE_RHO,
    cells_equal,
    classify_cell,
    is_column_pair_exact,
    total_sort_key,
)
from txt2sql_eval_harness.judge import full_trace_verdict_table, judge_item, sql_only_verdict_table
from txt2sql_eval_harness.models import (
    EvalConfig,
    GroundTruth,
    ResultTable,
    Step,
    ThresholdPair,
    Trace,
)
from txt2sql_eval_harness.renderer import extract_data
from txt2sql_eval_harness.rubrics import VALIDITY_NOTE
from txt2sql_eval_harness.sandbox import execute_sandboxed
from txt2sql_eval_harness.sql_analysis import (
    is_d1_superset_equivalent,
    is_det,
    is_rd_eligible,
    is_volatile,
    normalize_sql_n,
)


def _create_sample_db(db_path: Path, *, strict: bool = False, check_gt0: bool = False) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        if strict:
            conn.execute("CREATE TABLE u(a INTEGER) STRICT;")
            conn.executemany("INSERT INTO u(a) VALUES (?)", [(1,), (2,), (None,)])
        else:
            conn.execute("CREATE TABLE u(a INTEGER);")
            conn.executemany("INSERT INTO u(a) VALUES (?)", [(1,), (2,), (None,), ("oops",)])

        if check_gt0:
            conn.execute("CREATE TABLE t(x INTEGER NOT NULL CHECK(x > 0));")
            conn.executemany("INSERT INTO t(x) VALUES (?)", [(1,), (5,), (10,)])
        else:
            conn.execute(
                "CREATE TABLE sales(id INTEGER PRIMARY KEY, region TEXT, refunded INTEGER, amount INTEGER);"
            )
            conn.executemany(
                "INSERT INTO sales VALUES (?, ?, ?, ?)",
                [
                    (1, "West", 0, 100),
                    (2, "East", 0, 50),
                    (3, "West", 1, 20),
                ],
            )
        conn.commit()
    finally:
        conn.close()


def test_codec_exact_64bit_int_and_bool_rejection() -> None:
    a = 9007199254740993
    b = 9007199254740992
    assert a != b
    enc_a = encode_cell(a)
    enc_b = encode_cell(b)
    assert enc_a == {"i": "9007199254740993"}
    assert enc_b == {"i": "9007199254740992"}
    assert decode_cell(enc_a) == a
    assert decode_cell(enc_b) == b

    tbl = ResultTable(columns=["big"], rows=[(a,), (b,)], total_rows=2, complete=True)
    round_trip = decode_result_table(json.loads(json.dumps(encode_result_table(tbl))))
    assert round_trip == tbl

    # float preservation via repr, blob via base64, bool rejected
    assert decode_cell(encode_cell(1.5)) == 1.5
    assert decode_cell(encode_cell(b"\x00\xff")) == b"\x00\xff"
    assert decode_cell(encode_cell(None)) is None
    with pytest.raises(Exception):
        encode_cell(True)  # type: ignore[arg-type]


def test_comparator_classes_and_total_order() -> None:
    assert classify_cell(9007199254740993).kind == "exact_number"
    assert classify_cell("42").kind == "exact_number"
    assert classify_cell("3.14").kind == "approx_number"
    assert classify_cell(3.14).kind == "approx_number"
    assert classify_cell("abc").kind == "text"

    # Adjacent large integers must never compare equal
    assert not is_column_pair_exact([9007199254740993], [9007199254740992]) or not cells_equal(
        9007199254740993, 9007199254740992, tolerant=False
    )
    assert not cells_equal(42, "42_text", tolerant=True)
    assert cells_equal(42, "42", tolerant=False)

    # Total order: null < exact_num < approx_num (at equal value) < text < blob
    items = [b"a", "a", 2.0, 2, None]
    ordered = sorted(items, key=total_sort_key)
    assert ordered[0] is None
    assert ordered[1] == 2 and type(ordered[1]) is int
    assert ordered[2] == 2.0 and type(ordered[2]) is float
    assert ordered[3] == "a"
    assert ordered[4] == b"a"


def test_sql_normal_form_and_predicates() -> None:
    q1 = "SELECT /* comment */ Amount AS total_amt FROM Sales WHERE Region = 'West'"
    q2 = "select amount from sales where region = 'West'"
    assert normalize_sql_n(q1) == normalize_sql_n(q2)

    # Quoted identifiers and literals stay case-exact
    q_quoted = 'SELECT "Amount" FROM Sales WHERE Region = \'West\''
    assert '"Amount"' in (normalize_sql_n(q_quoted) or "")
    assert normalize_sql_n('SELECT "amount" FROM Sales') != normalize_sql_n(
        'SELECT "Amount" FROM Sales'
    )

    # Alias kept if used elsewhere (e.g. ORDER BY alias)
    q_used_alias = "SELECT amount AS total_amt FROM sales ORDER BY total_amt"
    assert "total_amt" in (normalize_sql_n(q_used_alias) or "").lower()

    # Table aliases NOT normalized -> T1 vs a are distinct in N
    assert normalize_sql_n("SELECT T1.x FROM t AS T1") != normalize_sql_n(
        "SELECT a.x FROM t AS a"
    )
    # LIMIT/OFFSET NOT dropped
    assert normalize_sql_n("SELECT x FROM t LIMIT 1 OFFSET 0") != normalize_sql_n(
        "SELECT x FROM t LIMIT 1 OFFSET 1"
    )

    # Predicates
    assert is_volatile("SELECT RANDOM()")
    assert is_volatile("SELECT datetime()")
    assert not is_volatile("SELECT datetime('2024-01-01')")
    assert is_volatile("SELECT strftime('%Y')")
    assert not is_volatile("SELECT strftime('%Y', '2024-01-01')")

    assert not is_det("SELECT RANDOM()")
    assert not is_det("SELECT GROUP_CONCAT(x) FROM t")
    assert not is_det("SELECT ROW_NUMBER() OVER (ORDER BY x) FROM t")
    assert not is_det("SELECT x FROM t LIMIT 1")
    assert is_det("SELECT x FROM t ORDER BY x LIMIT 1")
    assert not is_rd_eligible("SELECT x FROM t ORDER BY x LIMIT 1")
    assert is_rd_eligible("SELECT x FROM t ORDER BY x")


def test_sandbox_non_read_timeout_and_prefix(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    _create_sample_db(db_path)

    # DROP TABLE / ATTACH rejected as non_read, DB unchanged
    for bad_sql in ("DROP TABLE sales", "ATTACH DATABASE ':memory:' AS m"):
        res, err, _ = execute_sandboxed(db_path, bad_sql)
        assert res is None
        assert err == "non_read"

    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 3
    conn.close()

    # Multi-statement
    res, err, _ = execute_sandboxed(db_path, "SELECT 1; SELECT 2;")
    assert res is None and err == "multi_statement"

    # Fetch cap prefix + COUNT(*)
    cfg_small_cap = EvalConfig(fetch_cap_rows=2, fetch_cap_bytes=1024 * 1024)
    res, err, refs = execute_sandboxed(db_path, "SELECT id FROM sales ORDER BY id", cfg_small_cap)
    assert err == "fetch_cap" and res is not None
    assert not res.complete
    assert len(res.rows) == 2
    assert res.total_rows == 3
    assert "sales" in refs

    # Recursive heavy query timeout
    cfg_fast = EvalConfig(sql_timeout_s=0.05)
    heavy_sql = (
        "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x < 10000000) "
        "SELECT SUM(x) FROM cnt"
    )
    res_t, err_t, _ = execute_sandboxed(db_path, heavy_sql, cfg_fast)
    assert res_t is None and err_t == "timeout"


def test_deterministic_rules_d1_d3_d4_d5_and_alternatives(tmp_path: Path) -> None:
    db_path = tmp_path / "sales.sqlite"
    _create_sample_db(db_path)
    cfg = EvalConfig()
    ctx = CallContext(cfg=cfg, allow_remote=False)

    def _trace(tid: str, sql: str) -> Trace:
        return Trace(
            trace_id=tid,
            group_id="g1",
            question="Total West non-refunded sales?",
            hint="",
            db_id="sales",
            db_sha256="0" * 64,
            steps=[Step(index=1, sql=sql)],
        )

    gt = GroundTruth(
        item_ref="q1",
        sql_chain=["SELECT SUM(amount) FROM sales WHERE region = 'West' AND refunded = 0"],
    )

    # D1 CORRECT on whitespace/comment/identifier case/unused outer alias
    v_d1 = judge_item(
        _trace(
            "t_d1",
            "select /* ok */ sum(AMOUNT) as s from SALES where REGION = 'West' and REFUNDED = 0",
        ),
        gt,
        "H+E",
        ctx,
        db_path=db_path,
    )
    assert v_d1.label == "CORRECT" and v_d1.decided_by == "rule:D1"

    # D3 non_read
    v_d3 = judge_item(_trace("t_d3", "DROP TABLE sales"), gt, "H+E", ctx, db_path=db_path)
    assert v_d3.label == "INCORRECT" and v_d3.decided_by == "rule:D3" and v_d3.reason == "non_read"

    # D4 scalar counterexample: 12 vs 13 (100 vs 50 here) vs 12 vs 1200 scale not D4
    gt_12 = GroundTruth(
        item_ref="g12",
        sql_chain=["SELECT COUNT(*) FROM sales"],
        results=[ResultTable(["c"], [(12,)], 1, True)],
    )
    tr_13 = Trace(
        trace_id="t13",
        group_id="g2",
        question="Count",
        hint="",
        db_id="sales",
        db_sha256="0",
        steps=[Step(1, "SELECT COUNT(*) + 1 FROM sales", result=ResultTable(["c"], [(13,)], 1, True))],
    )
    v_d4 = judge_item(tr_13, gt_12, "H+E", ctx)
    assert v_d4.label == "INCORRECT" and v_d4.decided_by == "rule:D4"

    # Exact 64-bit integer counterexample 9007199254740993 vs 9007199254740992 -> D4
    gt_big = GroundTruth(
        item_ref="gb",
        sql_chain=["SELECT x FROM t"],
        results=[ResultTable(["x"], [(9007199254740993,)], 1, True)],
    )
    tr_big = Trace(
        trace_id="tb",
        group_id="gb",
        question="x",
        hint="",
        db_id="d",
        db_sha256="0",
        steps=[Step(1, "SELECT y FROM t", result=ResultTable(["y"], [(9007199254740992,)], 1, True))],
    )
    v_big = judge_item(tr_big, gt_big, "H+E", ctx)
    assert v_big.label == "INCORRECT" and v_big.decided_by == "rule:D4"

    # Scale 12 vs 1200 -> NOT D4 (reaches Jev, here remote_disabled because allow_remote=False)
    tr_1200 = Trace(
        trace_id="t1200",
        group_id="g2",
        question="Pct",
        hint="",
        db_id="sales",
        db_sha256="0",
        steps=[Step(1, "SELECT COUNT(*) * 100 FROM sales", result=ResultTable(["c"], [(1200,)], 1, True))],
    )
    v_scale = judge_item(tr_1200, gt_12, "H+E", ctx)
    assert v_scale.label == "ABSTAIN" and v_scale.reason == "remote_disabled"

    # Non-rd_eligible ORDER BY ... LIMIT 1 differing 1x1 -> NO D4!
    gt_lim = GroundTruth(
        item_ref="glim",
        sql_chain=["SELECT amount FROM sales ORDER BY refunded DESC LIMIT 1"],
        results=[ResultTable(["amount"], [(20,)], 1, True)],
    )
    tr_lim = Trace(
        trace_id="tlim",
        group_id="glim",
        question="Top",
        hint="",
        db_id="sales",
        db_sha256="0",
        steps=[
            Step(
                1,
                "SELECT amount FROM sales ORDER BY refunded DESC, id ASC LIMIT 1",
                result=ResultTable(["amount"], [(100,)], 1, True),
            )
        ],
    )
    v_lim = judge_item(tr_lim, gt_lim, "H+E", ctx)
    assert v_lim.decided_by != "rule:D4" and v_lim.reason == "remote_disabled"


def test_volatile_and_strict_and_check_constraints_in_ddl(tmp_path: Path) -> None:
    db_chk = tmp_path / "chk.sqlite"
    _create_sample_db(db_chk, check_gt0=True)
    db_strict = tmp_path / "strict.sqlite"
    _create_sample_db(db_strict, strict=True)
    db_nonstrict = tmp_path / "nonstrict.sqlite"
    _create_sample_db(db_nonstrict, strict=False)

    cfg = EvalConfig(enable_d5=True)
    seen_packets: list[dict] = []

    class _MockJev:
        def send(self, req, *, timeout):
            seen_packets.append(extract_data(req["state"]))
            from txt2sql_eval_harness.clients import RawModelResponse

            return RawModelResponse(http_ok=True, payload={"Q1": 0.9, "Q5": {"not_ambiguous": 1.0, "ambiguous_generated_reasonable": 0.0, "ambiguous_generated_unreasonable": 0.0}}, input_tokens=50)

    ctx = CallContext(cfg=cfg, allow_remote=True, clients={"jev": _MockJev()})

    # 1. Volatile SQL vs Volatile SQL -> not D1, not D4, relation unknown, VOLATILE_NOTE present
    v_vol = judge_item(
        Trace("tvol", "gv", "Random", "", "chk", "0", [Step(1, "SELECT RANDOM()")]),
        GroundTruth("gvol", ["SELECT RANDOM()"]),
        "H+E",
        ctx,
        db_path=db_chk,
    )
    assert v_vol.decided_by == "jev"
    pkt_vol = seen_packets[-1]
    assert pkt_vol["RESULT_SUMMARY"]["relation"] == "unknown"
    assert pkt_vol["RESULT_SUMMARY"]["volatile_note"] is not None

    # 2. CHECK (x > 0): SELECT x FROM t WHERE x > 0 vs SELECT x FROM t -> coincident_result, DDL has CHECK, D5 cannot fire
    v_chk = judge_item(
        Trace("tchk", "gc", "x?", "", "chk", "0", [Step(1, "SELECT x FROM t")]),
        GroundTruth("gchk", ["SELECT x FROM t WHERE x > 0"]),
        "H+E",
        ctx,
        db_path=db_chk,
    )
    assert "coincident_result" in v_chk.flags
    assert v_chk.decided_by == "jev"
    pkt_chk = seen_packets[-1]
    assert "CHECK" in pkt_chk["SCHEMA_DDL"]["ddl"]

    # 3. STRICT vs non-STRICT DDL includes VALIDITY_NOTE & verbatim STRICT or lack of it
    judge_item(
        Trace("ts1", "gs", "a?", "", "strict", "0", [Step(1, "SELECT a FROM u WHERE a IS NULL OR typeof(a) = 'integer'")]),
        GroundTruth("gs1", ["SELECT a FROM u"]),
        "H+E",
        ctx,
        db_path=db_strict,
    )
    assert "STRICT" in seen_packets[-1]["SCHEMA_DDL"]["ddl"]
    assert seen_packets[-1]["SCHEMA_DDL"]["validity_note"] == VALIDITY_NOTE

    judge_item(
        Trace("ts2", "gns", "a?", "", "nonstrict", "0", [Step(1, "SELECT a FROM u WHERE a IS NULL OR typeof(a) = 'integer'")]),
        GroundTruth("gs2", ["SELECT a FROM u"]),
        "H+E",
        ctx,
        db_path=db_nonstrict,
    )
    assert "STRICT" not in seen_packets[-1]["SCHEMA_DDL"]["ddl"]


def test_is_volatile_ignores_comments_and_non_date_literals() -> None:
    from txt2sql_eval_harness.sql_analysis import is_volatile

    assert not is_volatile("SELECT * FROM t WHERE status = 'now'")
    assert not is_volatile("SELECT 1 -- random()")
    assert not is_volatile("SELECT /* current_timestamp */ 1")
    assert is_volatile("SELECT RANDOM()")
    assert is_volatile("SELECT datetime('now')")
    assert is_volatile("SELECT strftime('%Y', 'now')")


def test_prepare_best_candidate_prefers_equal_gt_alternative() -> None:
    from txt2sql_eval_harness.judge import prepare_best_candidate

    cfg = EvalConfig()
    r_gt_bad = ResultTable(["c"], [(99,)], 1, True)
    r_gt_good = ResultTable(["c"], [(10,)], 1, True)
    r_gen = ResultTable(["c"], [(10,)], 1, True)
    # Use LIMIT 1 so D4 does not fire on r_gt_bad
    gt = GroundTruth(
        "alt1",
        ["SELECT c FROM t ORDER BY a LIMIT 1"],
        alternatives=[["SELECT c FROM t ORDER BY b LIMIT 1"]],
        results=[r_gt_bad, r_gt_good],
    )
    tr = Trace("t_alt", "g_alt", "Top c?", "", "d", "0", [Step(1, "SELECT c FROM t ORDER BY x LIMIT 1", result=r_gen)])
    cand, overflow = prepare_best_candidate(tr, gt, None, cfg, cfg.config_hash(""), "H+E")
    assert not overflow and cand is not None
    assert cand.alignment.relation == "equal"
    assert cand.gt_sql == "SELECT c FROM t ORDER BY b LIMIT 1"


def test_sandbox_missing_db_and_memory_table_name(tmp_path: Path) -> None:
    from txt2sql_eval_harness.sandbox import execute_sandboxed

    res, err, _ = execute_sandboxed(tmp_path / "does_not_exist.sqlite", "SELECT 1")
    assert res is None and err == "runtime"

    db_path = tmp_path / "mem_tbl.sqlite"
    _create_sample_db(db_path)
    res2, err2, _ = execute_sandboxed(db_path, "SELECT * FROM memory_stats")
    assert res2 is None and err2 == "compile_error"


def test_d1_superset_equivalence_and_q5_conflict_escalation(tmp_path: Path) -> None:
    db_path = tmp_path / "sales_superset.sqlite"
    _create_sample_db(db_path)
    cfg = EvalConfig()
    ctx = CallContext(cfg=cfg, allow_remote=False)

    gt_sql = "SELECT SUM(amount) FROM sales WHERE id BETWEEN 1 AND 2"
    # Column addition + BETWEEN expansion -> resolved locally by rule:D1 without remote calls
    assert is_d1_superset_equivalent(
        gt_sql,
        "SELECT SUM(amount), COUNT(*) FROM sales WHERE id >= 1 AND id <= 2",
    )
    # Column permutation
    assert is_d1_superset_equivalent(
        "SELECT region, amount FROM sales ORDER BY id",
        "SELECT amount, region FROM sales ORDER BY id",
    )
    # DISTINCT with extra column must NOT match D1 superset
    assert not is_d1_superset_equivalent(
        "SELECT DISTINCT region FROM sales",
        "SELECT DISTINCT region, id FROM sales",
    )

    tr_add = Trace(
        trace_id="t_sup",
        group_id="g_sup",
        question="Total sales for ids 1 to 2?",
        hint="",
        db_id="sales",
        db_sha256="0" * 64,
        steps=[Step(1, "SELECT SUM(amount), COUNT(*) FROM sales WHERE id >= 1 AND id <= 2")],
    )
    v_sup = judge_item(tr_add, GroundTruth("q_sup", [gt_sql]), "H+E", ctx, db_path=db_path)
    assert v_sup.label == "CORRECT" and v_sup.decided_by == "rule:D1"

    # When Q1 <= tau_lo conflicts with plurality Q5['ambiguous_generated_reasonable'], escalate
    tau = ThresholdPair(tau_hi=0.85, tau_lo=0.15)
    ans_conflict = {
        "Q1": 0.12,
        "Q5": {
            "not_ambiguous": 0.25,
            "ambiguous_generated_reasonable": 0.57,
            "ambiguous_generated_unreasonable": 0.18,
        },
    }
    assert sql_only_verdict_table(None, ans_conflict, tau) == ("ESCALATE", "uncertain_or_conflict")  # type: ignore[arg-type]

    # When Q1 <= tau_lo and Q5 says not_ambiguous is the plurality winner, remain INCORRECT
    ans_unamb = {
        "Q1": 0.12,
        "Q5": {
            "not_ambiguous": 0.70,
            "ambiguous_generated_reasonable": 0.20,
            "ambiguous_generated_unreasonable": 0.10,
        },
    }
    assert sql_only_verdict_table(None, ans_unamb, tau) == ("INCORRECT", None)  # type: ignore[arg-type]

    # full_trace_verdict_table safely accepts cand=None (used by calibration threshold fitting)
    assert full_trace_verdict_table(
        None,
        {
            "Q3": 0.90,
            "Q4": 0.90,
            "Q6": 0.90,
            "Q5": {
                "not_ambiguous": 0.95,
                "ambiguous_generated_reasonable": 0.03,
                "ambiguous_generated_unreasonable": 0.02,
            },
        },
        tau,
    ) == ("CORRECT", None, [])
