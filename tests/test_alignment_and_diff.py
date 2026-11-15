"""Acceptance tests for alignment, matching, scales, row diffs, and coercion/prefix rules."""

from __future__ import annotations

from decimal import Decimal

from txt2sql_eval_harness.alignment import align_tables
from txt2sql_eval_harness.claims import check_d2
from txt2sql_eval_harness.models import Contract, ResultTable, Step


def _contract(n_cols: int, *, order_req: bool = False, limit_req: bool = False) -> Contract:
    return Contract(
        gt_columns=list(range(n_cols)),
        order_required=order_req,
        limit_required=limit_req,
    )


def test_numeric_only_unmatched_rows() -> None:
    gt = ResultTable(["v"], [(1,), (2,), (3,)], 3, True)
    gen = ResultTable(["v"], [(1,), (2,), (4,)], 3, True)
    al, rd, _ = align_tables(gt, gen, _contract(1))
    assert al.key_columns is None
    assert al.witness_kind == "unmatched_rows"
    assert al.relation == "overlap"
    assert rd is not None
    assert rd["rows_matched"] == 2
    assert rd["gt"]["only"] == 1
    assert rd["gen"]["only"] == 1
    assert rd["gt"]["witnesses"] == [{"row": [3], "n_gt": 1, "n_gen": 0}]
    assert rd["gen"]["witnesses"] == [{"row": [4], "n_gt": 0, "n_gen": 1}]


def test_strict_superset_and_duplicates_and_empty_side_zero_witnesses() -> None:
    # (1),(2) vs (1),(2),(3) -> gt_subset_gen; empty-diff side has only: 0 and 0 witnesses
    gt1 = ResultTable(["v"], [(1,), (2,)], 2, True)
    gen1 = ResultTable(["v"], [(1,), (2,), (3,)], 3, True)
    al1, rd1, _ = align_tables(gt1, gen1, _contract(1))
    assert al1.relation == "gt_subset_gen"
    assert rd1 is not None
    assert rd1["gt"]["only"] == 0 and rd1["gt"]["witnesses"] == []
    assert rd1["gen"]["only"] == 1 and rd1["gen"]["witnesses"] == [{"row": [3], "n_gt": 0, "n_gen": 1}]

    # (1),(1),(2) vs (1),(2) -> gen_subset_gt with duplicate multiplicity witness
    gt2 = ResultTable(["v"], [(1,), (1,), (2,)], 3, True)
    gen2 = ResultTable(["v"], [(1,), (2,)], 2, True)
    al2, rd2, _ = align_tables(gt2, gen2, _contract(1))
    assert al2.relation == "gen_subset_gt"
    assert rd2 is not None
    assert rd2["gt"]["only"] == 1
    assert rd2["gt"]["witnesses"] == [{"row": [1], "n_gt": 2, "n_gen": 1}]
    assert rd2["gen"]["only"] == 0 and rd2["gen"]["witnesses"] == []


def test_gt_zero_vs_empty_no_d2() -> None:
    gt = ResultTable(["z"], [(0,), (0,)], 2, True)
    gen = ResultTable(["z"], [], 0, True)
    assert not check_d2("sql_only", "SELECT z FROM t", gt, [Step(1, "SELECT z FROM t WHERE 0", result=gen)])
    al, rd, _ = align_tables(gt, gen, _contract(1))
    assert al.relation == "gen_subset_gt"
    assert rd is not None
    assert rd["gen"]["n_rows"] == 0
    assert rd["gen"]["witnesses"] == []
    assert rd["gen"]["stats"] is None


def test_prefix_250k_gives_unknown_and_no_row_diff() -> None:
    gt = ResultTable(["id"], [(1,), (2,)], 2, True)
    gen = ResultTable(["id"], [(1,), (2,)], 250_000, False)
    al, rd, _ = align_tables(gt, gen, _contract(1))
    assert al.relation == "unknown"
    assert rd is None


def test_real_rounding_boundary_straddle_unique_equal() -> None:
    gt = ResultTable(
        ["x"],
        [(1.0000049999,), (1.0000049999,), (1.0000049999,), (2.0,)],
        4,
        True,
    )
    gen = ResultTable(
        ["y"],
        [(1.0000050001,), (1.0000050001,), (1.0000050001,), (2.0,)],
        4,
        True,
    )
    al, rd, _ = align_tables(gt, gen, _contract(1), rho=Decimal("1e-9"))
    assert al.map_status == "unique"
    assert al.relation == "equal"
    assert rd is None


def test_key_aligned_paired_and_one_sided_witnesses() -> None:
    gt = ResultTable(["k", "v"], [("A", 10), ("B", 20)], 2, True)
    gen = ResultTable(["k", "v"], [("A", 10), ("B", 21)], 2, True)
    al, rd, _ = align_tables(gt, gen, _contract(2))
    assert al.witness_kind == "key_aligned"
    assert al.relation == "overlap"
    assert rd is not None
    assert rd["gt"]["witnesses"] == [{"key": ["B"], "gt": [20], "gen": [21]}]

    # One-sided extra row (C, 30)
    gen_sup = ResultTable(["k", "v"], [("A", 10), ("B", 20), ("C", 30)], 3, True)
    al_s, rd_s, _ = align_tables(gt, gen_sup, _contract(2))
    assert al_s.relation == "gt_subset_gen"
    assert rd_s is not None
    assert rd_s["gt"]["witnesses"] == []
    assert rd_s["gen"]["witnesses"] == [{"key": ["C"], "gen": [30]}]


def test_unmapped_rows_disjoint() -> None:
    gt = ResultTable(["name"], [("Alice",), ("Bob",)], 2, True)
    gen = ResultTable(["id"], [(1,), (2,)], 2, True)
    al, rd, _ = align_tables(gt, gen, _contract(1))
    assert al.map_status == "none"
    assert al.relation == "disjoint"
    assert al.witness_kind == "unmapped_rows"
    assert rd is not None
    assert len(rd["gt"]["witnesses"]) == 2
    assert len(rd["gen"]["witnesses"]) == 2


def test_adjacent_large_integers_column_not_equal() -> None:
    gt = ResultTable(["n"], [(9007199254740993,), (1,), (2,)], 3, True)
    gen = ResultTable(["n"], [(9007199254740992,), (1,), (2,)], 3, True)
    al, rd, _ = align_tables(gt, gen, _contract(1))
    assert al.relation != "equal"
    assert rd is not None
