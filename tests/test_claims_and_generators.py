"""Acceptance tests for claims binding, D0/D2, S1, S2, S3, S3m, E5b, and S5 trace tagging."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import sqlite3

from txt2sql_eval_harness.agent import tag_s5_trace
from txt2sql_eval_harness.claims import check_d0, check_d2, extract_and_bind_claims
from txt2sql_eval_harness.generators import (
    E5B_PAYLOAD,
    check_e5b_invariant,
    generate_e5b_pairs,
    generate_s1_items,
    generate_s2_items,
    generate_s3_items,
    generate_s3m_items,
)
from txt2sql_eval_harness.models import (
    GroundTruth,
    LabelledItem,
    ResultTable,
    Step,
    Trace,
)


def _make_gen_db(db_dir: Path) -> Path:
    sub = db_dir / "shop"
    sub.mkdir(parents=True, exist_ok=True)
    db_path = sub / "shop.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, region TEXT, amount INTEGER, dt TEXT);"
        )
        conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?)",
            [
                (1, "West", 10, "2023-05-01"),
                (2, "East", 20, "2023-06-15"),
                (3, "North", 30, "2024-01-10"),
            ],
        )
        conn.execute("CREATE TABLE cities (name TEXT, pop INTEGER);")
        conn.executemany("INSERT INTO cities VALUES (?, ?)", [("Paris", 100), ("Rome", 200)])
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_d0_and_claims_and_derivations() -> None:
    assert check_d0("")
    assert check_d0(" \n ")
    assert not check_d0("There were 12 orders.")
    assert not check_d0(None)

    gt_res = ResultTable(["name", "rev", "cost"], [("Acme", 10, 20)], 1, True)
    claims, derivs = extract_and_bind_claims(
        "Acme had revenue $20 (up 50% in May 2023 on 2023-05-14) and cost 10.",
        "What is Acme revenue and cost?",
        gt_res,
        [Step(1, "SELECT name, rev, cost FROM t", result=gt_res)],
    )
    kinds = {c.kind for c in claims}
    assert {"entity", "number", "date"}.issubset(kinds)
    assert any(c.kind == "entity" and c.value == "Acme" and c.in_question for c in claims)
    assert any(d.op == "SUM" and d.column == "rev" and d.value == 10 for d in derivs)
    assert any(d.op == "ROW_COUNT" and d.value == 1 for d in derivs)

    # ISO dates must not leak negative month/day numbers
    date_claims, _ = extract_and_bind_claims("On 2023-05-14 total was 100.", "When?", gt_res, [])
    nums = [c.value for c in date_claims if c.kind == "number"]
    assert nums == [Decimal("100")]


def test_d2_on_nonzero_vs_empty_and_zero_table() -> None:
    nonzero = ResultTable(["c"], [(12,)], 1, True)
    empty = ResultTable(["c"], [], 0, True)
    assert check_d2("full_trace", "SELECT c FROM t", nonzero, [Step(1, "SELECT c FROM t WHERE 0", result=empty)])
    # Timeout is not counterexample-grade error -> D2 must NOT fire
    assert not check_d2("full_trace", "SELECT c FROM t", nonzero, [Step(1, "SELECT c FROM t", error_kind="timeout")])
    # GT count 0 -> D2 must NOT fire
    zero = ResultTable(["c"], [(0,)], 1, True)
    assert not check_d2("full_trace", "SELECT COUNT(*) FROM t", zero, [Step(1, "SELECT COUNT(*) FROM t WHERE 0", result=empty)])


def test_s1_and_s2_generation_and_discards(tmp_path: Path) -> None:
    db_dir = tmp_path / "dbs"
    _make_gen_db(db_dir)

    base = LabelledItem(
        item_id="b1",
        group_id="g1",
        split="test",
        set="S1",
        family="base",
        slices=[],
        trace=Trace("b1", "g1", "Total amount for West region orders between 5 and 25?", "", "shop", "0", []),
        gt=GroundTruth("b1", ["SELECT region, amount FROM orders WHERE amount BETWEEN 5 AND 25 AND region = 'West' ORDER BY amount"]),
        label="correct",
    )
    s1_res = generate_s1_items([base], db_dir)
    fams = {it.family for it in s1_res.items}
    assert {"add_columns", "permute_columns", "drop_order_by", "between_expand"}.issubset(fams)

    # S2: anchored mutant kept only when equal_on_real_db is false; LIMIT 1 GT discarded not_rd_eligible
    s2_lim = LabelledItem(
        item_id="b_lim",
        group_id="glim",
        split="test",
        set="S2",
        family="base",
        slices=[],
        trace=Trace("b_lim", "glim", "Top West order?", "", "shop", "0", []),
        gt=GroundTruth("b_lim", ["SELECT id FROM orders WHERE region = 'West' ORDER BY amount DESC LIMIT 1"]),
        label="incorrect",
    )
    # Redundant predicate mutant whose DB result equals GT -> equal_on_real_db discard
    s2_eq = LabelledItem(
        item_id="b_eq",
        group_id="geq",
        split="test",
        set="S2",
        family="base",
        slices=[],
        trace=Trace("b_eq", "geq", "Orders above 5?", "", "shop", "0", []),
        gt=GroundTruth("b_eq", ["SELECT id FROM orders WHERE amount >= 5 AND amount > 4"]),
        label="incorrect",
    )
    s2_res = generate_s2_items([base, s2_lim, s2_eq], db_dir)
    assert s2_res.discarded.get("not_rd_eligible", 0) >= 1
    assert any(it.family == "comparator_flip" or it.family == "filter_literal" or it.family == "drop_conjunct" for it in s2_res.items)


def test_s3_discard_controls_and_s3m_generation(tmp_path: Path) -> None:
    db_dir = tmp_path / "dbs"
    _make_gen_db(db_dir)

    # 1. Control: equal-value swap (A, 10), (B, 10) -> swapped_values discarded
    eq_swap_base = LabelledItem(
        item_id="s_eqswap",
        group_id="gs1",
        split="test",
        set="S3",
        family="base",
        slices=[],
        trace=Trace("s_eqswap", "gs1", "Scores?", "", "shop", "0", []),
        gt=GroundTruth(
            "s_eqswap",
            ["SELECT name, score FROM t"],
            results=[ResultTable(["name", "score"], [("A", 10), ("B", 10)], 2, True)],
        ),
        label="correct",
    )
    res_swap = generate_s3_items([eq_swap_base])
    assert all(it.family != "swapped_values" for it in res_swap.items)
    assert res_swap.discarded.get("swapped_values_ineligible", 0) >= 1

    # 2. Control: REAL 0.0 -> additive floor gives 0.02 instead of 0.0 * 1.1 = 0.00!
    zero_real_base = LabelledItem(
        item_id="s_zero",
        group_id="gs2",
        split="test",
        set="S3",
        family="base",
        slices=[],
        trace=Trace("s_zero", "gs2", "Rate?", "", "shop", "0", []),
        gt=GroundTruth(
            "s_zero",
            ["SELECT r FROM t"],
            results=[ResultTable(["r"], [(0.0,)], 1, True)],
        ),
        label="correct",
    )
    res_zero = generate_s3_items([zero_real_base])
    wn_zero = [it for it in res_zero.items if it.family == "wrong_number"]
    assert wn_zero and "0.02" in (wn_zero[0].trace.final_answer or "")

    # 3. Control: Subset cue "Name any driver..." -> omitted_entity discarded
    omit_cue_base = LabelledItem(
        item_id="s_cue",
        group_id="gs3",
        split="test",
        set="S3",
        family="base",
        slices=[],
        trace=Trace("s_cue", "gs3", "Name any driver with points.", "", "shop", "0", []),
        gt=GroundTruth(
            "s_cue",
            ["SELECT name, pts FROM d"],
            results=[ResultTable(["name", "pts"], [("Alice", 10), ("Bob", 20)], 2, True)],
        ),
        label="correct",
    )
    res_cue = generate_s3_items([omit_cue_base])
    assert all(it.family != "omitted_entity" for it in res_cue.items)

    # 4. Control: mutant (4), (6) with answer total = 10 -> unsupported discarded because SUM=10 matches
    sum_base = LabelledItem(
        item_id="s_sum",
        group_id="gs4",
        split="test",
        set="S3",
        family="base",
        slices=[],
        trace=Trace("s_sum", "gs4", "Total?", "", "shop", "0", []),
        gt=GroundTruth(
            "s_sum",
            ["SELECT SUM(x) AS total FROM t"],
            results=[ResultTable(["total"], [(10,)], 1, True)],
        ),
        label="correct",
    )
    mutant_sum = LabelledItem(
        item_id="mut_sum",
        group_id="gs4",
        split="test",
        set="S2",
        family="agg_swap",
        slices=[],
        trace=Trace(
            "mut_sum",
            "gs4",
            "Total?",
            "",
            "shop",
            "0",
            [Step(1, "SELECT x FROM t", result=ResultTable(["x"], [(4,), (6,)], 2, True))],
        ),
        gt=sum_base.gt,
        label="incorrect",
    )
    res_unsup = generate_s3_items([sum_base], s2_by_base={"s_sum": [mutant_sum]})
    assert all(it.family != "unsupported" for it in res_unsup.items)
    assert res_unsup.discarded.get("unsupported_ineligible", 0) >= 1

    # 5. S3m fixture: R* (A, 10), (B, 20), u candidates (Paris, 20) [rejected because contains 20!] then (Paris), (Rome)
    s3m_base = LabelledItem(
        item_id="s3m_q",
        group_id="gm1",
        split="test",
        set="S3",
        family="base",
        slices=[],
        trace=Trace("s3m_q", "gm1", "Totals?", "", "shop", "0", []),
        gt=GroundTruth(
            "s3m_q",
            ["SELECT k, total FROM t"],
            results=[ResultTable(["k", "total"], [("A", 10), ("B", 20)], 2, True)],
        ),
        label="correct",
    )
    s3_gen = generate_s3_items([s3m_base])
    s3_corr = [it for it in s3_gen.items if it.family == "correct"]

    u_bad = LabelledItem(
        item_id="u_contains_20",
        group_id="gu1",
        split="test",
        set="S1",
        family="base",
        slices=[],
        trace=Trace("u_contains_20", "gu1", "Other?", "", "shop", "0", []),
        gt=GroundTruth(
            "u_contains_20",
            ["SELECT city, pop FROM c"],
            results=[ResultTable(["city", "pop"], [("Paris", 20)], 1, True)],
        ),
        label="correct",
    )
    u_good = LabelledItem(
        item_id="u_cities",
        group_id="gu2",
        split="test",
        set="S1",
        family="base",
        slices=[],
        trace=Trace("u_cities", "gu2", "Cities?", "", "shop", "0", []),
        gt=GroundTruth(
            "u_cities",
            ["SELECT name FROM cities"],
            results=[ResultTable(["name"], [("Paris",), ("Rome",)], 2, True)],
        ),
        label="correct",
    )
    m_item = LabelledItem(
        item_id="S2__s3m_q__fl__0",
        group_id="gm1",
        split="test",
        set="S2",
        family="filter_literal",
        slices=[],
        trace=Trace(
            "m1",
            "gm1",
            "Totals?",
            "",
            "shop",
            "0",
            [Step(1, "SELECT k, total FROM t WHERE x > 1", result=ResultTable(["k", "total"], [("A", 12), ("B", 20)], 2, True))],
        ),
        gt=s3m_base.gt,
        label="incorrect",
    )
    s3m_res = generate_s3m_items(
        s3_corr,
        [s3m_base, u_bad, u_good],
        s2_by_base={"s3m_q": [m_item]},
    )
    assert len(s3m_res.items) == 4
    by_fam = {it.family: it for it in s3m_res.items}
    assert by_fam["early_support"].label == "correct"
    assert by_fam["early_support_rev"].label == "correct"
    assert by_fam["no_support"].label == "incorrect"
    assert by_fam["no_support_rev"].label == "incorrect"
    # Verify claim "10" in early_support only has agent_cells in step 1 (not step 2 u)
    claims_early, _ = extract_and_bind_claims(
        by_fam["early_support"].trace.final_answer or "",
        by_fam["early_support"].trace.question,
        s3m_base.gt.results[0],
        by_fam["early_support"].trace.steps,
    )
    c_10 = next(c for c in claims_early if c.kind == "number" and c.value == Decimal("10"))
    assert all(st_idx == 1 for st_idx, _, _ in c_10.agent_cells)
    assert "early_only_support" in tag_s5_trace(by_fam["early_support"])
    assert "multi_step" in tag_s5_trace(by_fam["early_support"])


def test_e5b_invariants_reject_digit_and_text_collisions() -> None:
    s3_item = LabelledItem(
        item_id="S3__orig1__correct",
        group_id="ge1",
        split="test",
        set="S3",
        family="correct",
        slices=[],
        trace=Trace(
            "S3__orig1__correct",
            "ge1",
            "Count?",
            "",
            "shop",
            "0",
            [Step(1, "SELECT COUNT(*) FROM t", result=ResultTable(["cnt"], [(12,)], 1, True))],
            final_answer="cnt = 12",
        ),
        gt=GroundTruth("orig1", ["SELECT COUNT(*) FROM t"], results=[ResultTable(["cnt"], [(12,)], 1, True)]),
        label="correct",
    )
    res = generate_e5b_pairs([s3_item])
    assert len(res.items) == 2

    # Payload with digit rejected
    copy_digit = LabelledItem(
        item_id="c_dig",
        group_id="ge1",
        split="test",
        set="S3",
        family="correct",
        slices=[],
        trace=Trace("c_dig", "ge1", "Count?", "", "shop", "0", s3_item.trace.steps, "cnt = 12\n\nAnswer 99."),
        gt=s3_item.gt,
        label="correct",
        pair_of=s3_item.item_id,
    )
    ok_d, reason_d = check_e5b_invariant(s3_item, copy_digit, payload="Answer 99.")
    assert not ok_d and reason_d == "payload_contains_digit"

    # Payload colliding with a TEXT cell ("Evaluator" in table!) rejected
    s3_collision = LabelledItem(
        item_id="S3__col__correct",
        group_id="ge2",
        split="test",
        set="S3",
        family="correct",
        slices=[],
        trace=Trace(
            "S3__col__correct",
            "ge2",
            "Who?",
            "",
            "shop",
            "0",
            [Step(1, "SELECT role, cnt FROM r", result=ResultTable(["role", "cnt"], [("Evaluator", 5)], 1, True))],
            final_answer="Evaluator: cnt = 5",
        ),
        gt=GroundTruth("col", ["SELECT role, cnt FROM r"], results=[ResultTable(["role", "cnt"], [("Evaluator", 5)], 1, True)]),
        label="correct",
    )
    copy_col = LabelledItem(
        item_id="c_col",
        group_id="ge2",
        split="test",
        set="S3",
        family="correct",
        slices=[],
        trace=Trace("c_col", "ge2", "Who?", "", "shop", "0", s3_collision.trace.steps, f"Evaluator: cnt = 5\n\n{E5B_PAYLOAD}"),
        gt=s3_collision.gt,
        label="correct",
        pair_of=s3_collision.item_id,
    )
    ok_c, reason_c = check_e5b_invariant(s3_collision, copy_col, payload=E5B_PAYLOAD)
    assert not ok_c and reason_c == "payload_cell_or_col_collision"


def test_claim_number_suffixes_do_not_crash() -> None:
    gt_res = ResultTable(["rev", "b"], [(2_000_000, 5_000_000_000)], 1, True)
    claims, _ = extract_and_bind_claims(
        "Revenue was $2M, budget 5bn, and 5K units in Room 5b and Zone 3n.",
        "What was revenue?",
        gt_res,
        [Step(1, "SELECT rev, b FROM t", result=gt_res)],
    )
    nums = [c.value for c in claims if c.kind == "number"]
    assert Decimal("2000000") in nums
    assert Decimal("5000000000") in nums
    assert Decimal("5000") in nums
