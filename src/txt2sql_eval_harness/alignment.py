"""Tolerance-aware column alignment, maximum row matching, and ROW DIFF witness extraction."""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from decimal import Decimal
from typing import Any

from txt2sql_eval_harness.comparator import (
    DEFAULT_JUDGE_RHO,
    ClassifiedCell,
    classified_equal,
    classified_sort_key,
    classify_cell,
    is_column_pair_exact,
    sort_key_with_scale,
)
from txt2sql_eval_harness.models import Alignment, Cell, Contract, ResultTable

_SCALES: tuple[Decimal, ...] = (Decimal(1), Decimal(100), Decimal("0.01"))
_MAX_ASSIGNMENT_SEARCH = 64
_MAX_KUHN_GROUP = 64


def _encode_witness_cell(cell: Cell) -> Any:
    """Encode cell value for JSON-safe ROW DIFF witnesses inside DATA block."""
    if cell is None:
        return None
    if isinstance(cell, int) and not isinstance(cell, bool):
        return cell
    if isinstance(cell, float):
        return cell
    if isinstance(cell, str):
        return cell
    if isinstance(cell, bytes):
        return {"blob_bytes": len(cell)}
    return str(cell)


def _exact_hash_key(cc: ClassifiedCell) -> tuple:
    """Build a canonical, hashable key encoding storage type and normalized value."""
    if cc.kind == "null":
        return ("null", None)
    if cc.kind == "exact_number":
        return ("int", cc.exact_int)
    if cc.kind == "approx_number":
        return ("num", str(cc.num))
    if cc.kind == "text":
        return ("text", cc.text)
    return ("blob", cc.blob)


def _column_prefilter_passes(
    gt_cls: list[ClassifiedCell],
    gen_cls: list[ClassifiedCell],
    *,
    exact_pair: bool,
    scale: Decimal,
) -> bool:
    """Sound O(n) prefilter rejecting impossible equal-column pairs."""
    if len(gt_cls) != len(gen_cls):
        return False

    gt_nulls = sum(1 for c in gt_cls if c.kind == "null")
    gen_nulls = sum(1 for c in gen_cls if c.kind == "null")
    if gt_nulls != gen_nulls:
        return False

    gt_texts = Counter(c.text for c in gt_cls if c.kind == "text")
    gen_texts = Counter(c.text for c in gen_cls if c.kind == "text")
    if gt_texts != gen_texts:
        return False

    gt_blobs = Counter(c.blob for c in gt_cls if c.kind == "blob")
    gen_blobs = Counter(c.blob for c in gen_cls if c.kind == "blob")
    if gt_blobs != gen_blobs:
        return False

    gt_nums = [c for c in gt_cls if c.kind in ("exact_number", "approx_number")]
    gen_nums = [c for c in gen_cls if c.kind in ("exact_number", "approx_number")]
    if len(gt_nums) != len(gen_nums):
        return False

    if exact_pair and scale == 1:
        gt_ints = Counter(c.exact_int for c in gt_nums)
        gen_ints = Counter(c.exact_int for c in gen_nums)
        if gt_ints != gen_ints:
            return False

    return True


def _single_column_match_count(
    gt_cls: list[ClassifiedCell],
    gen_cls: list[ClassifiedCell],
    *,
    tolerant: bool,
    rho: Decimal,
    scale: Decimal,
) -> int:
    """Count maximum matched values between two single columns via sorted two-pointer pass."""
    gt_sorted = sorted(gt_cls, key=classified_sort_key)
    gen_sorted = sorted(gen_cls, key=lambda c: sort_key_with_scale(c, scale))
    i = 0
    j = 0
    matched = 0
    ng = len(gt_sorted)
    nc = len(gen_sorted)
    while i < ng and j < nc:
        ga = gt_sorted[i]
        gb = gen_sorted[j]
        if classified_equal(ga, gb, tolerant=tolerant, rho=rho, scale_b=scale):
            matched += 1
            i += 1
            j += 1
        elif classified_sort_key(ga) < sort_key_with_scale(gb, scale):
            i += 1
        else:
            j += 1
    return matched


def _columns_equal(
    gt_col: list[Cell],
    gen_col: list[Cell],
    gt_cls: list[ClassifiedCell],
    gen_cls: list[ClassifiedCell],
    *,
    rho: Decimal,
    scale: Decimal,
) -> tuple[bool, bool]:
    """Check whether sorted columns are elementwise equal under `cmp`.

    Returns ``(is_equal, is_tolerant)``.
    """
    exact_pair = is_column_pair_exact(gt_col, gen_col) and scale == 1
    tolerant = not exact_pair
    if not _column_prefilter_passes(gt_cls, gen_cls, exact_pair=exact_pair, scale=scale):
        return False, tolerant
    matched = _single_column_match_count(
        gt_cls, gen_cls, tolerant=tolerant, rho=rho, scale=scale
    )
    return matched == len(gt_cls), tolerant


def _kuhn_matching(
    gt_indices: list[int],
    gen_indices: list[int],
    gt_rows_cls: list[list[ClassifiedCell]],
    gen_rows_cls: list[list[ClassifiedCell]],
    tolerant_specs: list[tuple[int, int, Decimal]],
    rho: Decimal,
) -> tuple[int, set[int], set[int]]:
    """Run Kuhn's augmenting-path bipartite matching on one exact-column equivalence group."""
    n_gt = len(gt_indices)
    n_gen = len(gen_indices)
    adj: list[list[int]] = [[] for _ in range(n_gt)]
    for u, gi in enumerate(gt_indices):
        row_a = gt_rows_cls[gi]
        for v, cj in enumerate(gen_indices):
            row_b = gen_rows_cls[cj]
            ok = True
            for g_col, c_col, scale in tolerant_specs:
                if not classified_equal(
                    row_a[g_col],
                    row_b[c_col],
                    tolerant=True,
                    rho=rho,
                    scale_b=scale,
                ):
                    ok = False
                    break
            if ok:
                adj[u].append(v)

    match_gen = [-1] * n_gen

    def dfs(u: int, seen: list[bool]) -> bool:
        """Find an augmenting path from GT column index `u` in the bipartite graph."""
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_gen[v] == -1 or dfs(match_gen[v], seen):
                match_gen[v] = u
                return True
        return False

    matched_gt_local: set[int] = set()
    matched_gen_local: set[int] = set()
    for u in range(n_gt):
        seen = [False] * n_gen
        if dfs(u, seen):
            matched_gt_local.add(u)

    for v, u in enumerate(match_gen):
        if u != -1:
            matched_gen_local.add(v)

    matched_gt = {gt_indices[u] for u in matched_gt_local}
    matched_gen = {gen_indices[v] for v in matched_gen_local}
    return len(matched_gt), matched_gt, matched_gen


def _match_rows(
    gt_rows_cls: list[list[ClassifiedCell]],
    gen_rows_cls: list[list[ClassifiedCell]],
    col_specs: list[tuple[int, int, bool, Decimal]],
    rho: Decimal,
) -> tuple[int, set[int], set[int], bool]:
    """Compute maximum tolerance row matching across all rows.

    Returns ``(rows_matched, matched_gt_indices, matched_gen_indices, limit_hit)``.
    """
    exact_specs = [(g, c) for g, c, tol, scale in col_specs if not tol and scale == 1]
    tolerant_specs = [(g, c, scale) for g, c, tol, scale in col_specs if tol or scale != 1]

    gt_groups: dict[tuple, list[int]] = defaultdict(list)
    gen_groups: dict[tuple, list[int]] = defaultdict(list)

    for idx, row in enumerate(gt_rows_cls):
        key = tuple(_exact_hash_key(row[g]) for g, _ in exact_specs)
        gt_groups[key].append(idx)

    for idx, row in enumerate(gen_rows_cls):
        key = tuple(_exact_hash_key(row[c]) for _, c in exact_specs)
        gen_groups[key].append(idx)

    total_matched = 0
    matched_gt: set[int] = set()
    matched_gen: set[int] = set()

    for key, gt_idxs in gt_groups.items():
        gen_idxs = gen_groups.get(key)
        if not gen_idxs:
            continue

        if len(tolerant_specs) == 0:
            k = min(len(gt_idxs), len(gen_idxs))
            total_matched += k
            matched_gt.update(gt_idxs[:k])
            matched_gen.update(gen_idxs[:k])
        elif len(tolerant_specs) == 1:
            g_col, c_col, scale = tolerant_specs[0]
            gt_sorted = sorted(gt_idxs, key=lambda idx: classified_sort_key(gt_rows_cls[idx][g_col]))
            gen_sorted = sorted(
                gen_idxs, key=lambda idx: sort_key_with_scale(gen_rows_cls[idx][c_col], scale)
            )
            i = 0
            j = 0
            while i < len(gt_sorted) and j < len(gen_sorted):
                ga = gt_rows_cls[gt_sorted[i]][g_col]
                gb = gen_rows_cls[gen_sorted[j]][c_col]
                if classified_equal(ga, gb, tolerant=True, rho=rho, scale_b=scale):
                    total_matched += 1
                    matched_gt.add(gt_sorted[i])
                    matched_gen.add(gen_sorted[j])
                    i += 1
                    j += 1
                elif classified_sort_key(ga) < sort_key_with_scale(gb, scale):
                    i += 1
                else:
                    j += 1
        else:
            if len(gt_idxs) > _MAX_KUHN_GROUP or len(gen_idxs) > _MAX_KUHN_GROUP:
                return 0, set(), set(), True
            k, m_gt, m_gen = _kuhn_matching(
                gt_idxs, gen_idxs, gt_rows_cls, gen_rows_cls, tolerant_specs, rho
            )
            total_matched += k
            matched_gt.update(m_gt)
            matched_gen.update(m_gen)

    return total_matched, matched_gt, matched_gen, False


def _find_key_columns(
    gt_cols_cls: list[list[ClassifiedCell]],
    gen_cols_cls: list[list[ClassifiedCell]],
    gt_rows_cls: list[list[ClassifiedCell]],
    gen_rows_cls: list[list[ClassifiedCell]],
    column_map: dict[int, int | None],
) -> list[int] | None:
    """Find non-numeric mapped GT columns whose combination is unique in both results."""
    non_numeric_gt_cols: list[int] = []
    for g_idx, c_idx in sorted(column_map.items()):
        if c_idx is None:
            continue
        has_num = any(
            cc.kind in ("exact_number", "approx_number") for cc in gt_cols_cls[g_idx]
        ) or any(
            cc.kind in ("exact_number", "approx_number") for cc in gen_cols_cls[c_idx]
        )
        if not has_num:
            non_numeric_gt_cols.append(g_idx)

    if not non_numeric_gt_cols:
        return None

    def is_unique(cols: list[int]) -> bool:
        """Return whether the projected column tuple uniquely identifies every row across both tables."""
        gt_keys = {tuple(_exact_hash_key(row[g]) for g in cols) for row in gt_rows_cls}
        if len(gt_keys) != len(gt_rows_cls):
            return False
        gen_keys = {
            tuple(_exact_hash_key(row[column_map[g]]) for g in cols)  # type: ignore[index]
            for row in gen_rows_cls
        }
        return len(gen_keys) == len(gen_rows_cls)

    for g in non_numeric_gt_cols:
        if is_unique([g]):
            return [g]
    if len(non_numeric_gt_cols) > 1 and is_unique(non_numeric_gt_cols):
        return non_numeric_gt_cols
    return None


def _compute_col_stats(
    rows: list[tuple[Cell, ...]],
    col_indices: list[int | None],
) -> list[dict[str, Any] | None]:
    """Compute per-column min/max/sum/distinct over complete rows; None for empty sides."""
    if not rows:
        return [None for _ in col_indices]
    stats: list[dict[str, Any] | None] = []
    for col_idx in col_indices:
        if col_idx is None:
            stats.append(None)
            continue
        vals = [row[col_idx] for row in rows]
        cls_vals = [classify_cell(v) for v in vals]
        distinct = len({_exact_hash_key(c) for c in cls_vals})
        non_null_nums = [
            c.num for c in cls_vals if c.kind in ("exact_number", "approx_number") and c.num is not None
        ]
        non_null_sorted = sorted(
            [v for v in vals if v is not None], key=lambda x: classified_sort_key(classify_cell(x))
        )
        min_v = _encode_witness_cell(non_null_sorted[0]) if non_null_sorted else None
        max_v = _encode_witness_cell(non_null_sorted[-1]) if non_null_sorted else None
        sum_v: Any = None
        if non_null_nums and len(non_null_nums) == sum(1 for v in vals if v is not None):
            total = sum(non_null_nums, Decimal(0))
            if total.is_finite():
                sum_v = int(total) if total == int(total) else float(total)
        stats.append({"min": min_v, "max": max_v, "sum": sum_v, "distinct": distinct})
    return stats


def _build_witnesses(
    gt: ResultTable,
    gen: ResultTable,
    gt_cols_cls: list[list[ClassifiedCell]],
    gen_cols_cls: list[list[ClassifiedCell]],
    gt_rows_cls: list[list[ClassifiedCell]],
    gen_rows_cls: list[list[ClassifiedCell]],
    column_map: dict[int, int | None],
    map_status: str,
    relation: str,
    rows_only_gt: int,
    rows_only_gen: int,
    matched_gt: set[int],
    matched_gen: set[int],
    col_specs: list[tuple[int, int, bool, Decimal]],
    rho: Decimal,
) -> tuple[list[int] | None, str, dict[str, Any] | None]:
    """Build key_columns, witness_kind, and ROW DIFF payload with bounded witness rows per side."""
    if relation == "equal" or relation == "unknown":
        return None, "none", None

    n_gt_cols = len(gt.columns)
    mapped_complete = all(column_map.get(g) is not None for g in range(n_gt_cols))

    gt_stats = _compute_col_stats(gt.rows, list(range(n_gt_cols)))
    gen_mapped_cols = [column_map.get(g) for g in range(n_gt_cols)]
    gen_stats = _compute_col_stats(
        gen.rows,
        gen_mapped_cols if mapped_complete else list(range(len(gen.columns))),
    )

    key_cols: list[int] | None = None
    if mapped_complete and gt.rows and gen.rows:
        key_cols = _find_key_columns(
            gt_cols_cls, gen_cols_cls, gt_rows_cls, gen_rows_cls, column_map
        )

    gt_witnesses: list[dict[str, Any]] = []
    gen_witnesses: list[dict[str, Any]] = []

    if not mapped_complete or map_status == "none":
        witness_kind = "unmapped_rows"
        if rows_only_gt > 0:
            seen_gt: Counter[tuple] = Counter()
            order_gt: list[tuple[tuple, tuple[Cell, ...]]] = []
            for row, rcls in sorted(
                zip(gt.rows, gt_rows_cls),
                key=lambda p: tuple(classified_sort_key(c) for c in p[1]),
            ):
                k = tuple(_exact_hash_key(c) for c in rcls)
                if seen_gt[k] == 0:
                    order_gt.append((k, row))
                seen_gt[k] += 1
            for k, row in order_gt[:5]:
                gt_witnesses.append(
                    {
                        "row": [_encode_witness_cell(c) for c in row],
                        "n_gt": seen_gt[k],
                        "n_gen": 0,
                    }
                )
        if rows_only_gen > 0:
            seen_gen: Counter[tuple] = Counter()
            order_gen: list[tuple[tuple, tuple[Cell, ...]]] = []
            for row, rcls in sorted(
                zip(gen.rows, gen_rows_cls),
                key=lambda p: tuple(classified_sort_key(c) for c in p[1]),
            ):
                k = tuple(_exact_hash_key(c) for c in rcls)
                if seen_gen[k] == 0:
                    order_gen.append((k, row))
                seen_gen[k] += 1
            for k, row in order_gen[:5]:
                gen_witnesses.append(
                    {
                        "row": [_encode_witness_cell(c) for c in row],
                        "n_gt": 0,
                        "n_gen": seen_gen[k],
                    }
                )
    elif key_cols is not None:
        witness_kind = "key_aligned"
        gt_by_key: dict[tuple, tuple[int, tuple[Cell, ...]]] = {}
        for idx, (row, rcls) in enumerate(zip(gt.rows, gt_rows_cls)):
            k = tuple(_exact_hash_key(rcls[g]) for g in key_cols)
            gt_by_key[k] = (idx, row)

        gen_by_key: dict[tuple, tuple[int, tuple[Cell, ...]]] = {}
        for idx, (row, rcls) in enumerate(zip(gen.rows, gen_rows_cls)):
            k = tuple(_exact_hash_key(rcls[column_map[g]]) for g in key_cols)  # type: ignore[index]
            gen_by_key[k] = (idx, row)

        def fmt_key(row: tuple[Cell, ...], cols: list[int]) -> list[Any]:
            """Serialize key column values of a row for bounded row-diff witnesses."""
            return [_encode_witness_cell(row[c]) for c in cols]

        def fmt_vals(row: tuple[Cell, ...], cols: list[int]) -> list[Any]:
            """Serialize non-key column values of a row for bounded row-diff witnesses."""
            return [_encode_witness_cell(row[c]) for c in cols]

        val_gt_cols = [g for g in range(n_gt_cols) if g not in key_cols]
        val_gen_cols = [column_map[g] for g in val_gt_cols]  # type: ignore[misc]

        # Paired differences first (present on both sides, values differ)
        if rows_only_gt > 0 and rows_only_gen > 0:
            for k in sorted(set(gt_by_key) & set(gen_by_key), key=str):
                if len(gt_witnesses) >= 5 and len(gen_witnesses) >= 5:
                    break
                gi, grow = gt_by_key[k]
                ci, crow = gen_by_key[k]
                if gi in matched_gt and ci in matched_gen:
                    continue
                w = {
                    "key": fmt_key(grow, key_cols),
                    "gt": fmt_vals(grow, val_gt_cols),
                    "gen": fmt_vals(crow, val_gen_cols),  # type: ignore[arg-type]
                }
                if len(gt_witnesses) < 5:
                    gt_witnesses.append(w)
                if len(gen_witnesses) < 5:
                    gen_witnesses.append(w)

        if rows_only_gt > 0:
            for k in sorted(set(gt_by_key) - set(gen_by_key), key=str):
                if len(gt_witnesses) >= 5:
                    break
                _, grow = gt_by_key[k]
                gt_witnesses.append(
                    {"key": fmt_key(grow, key_cols), "gt": fmt_vals(grow, val_gt_cols)}
                )

        if rows_only_gen > 0:
            for k in sorted(set(gen_by_key) - set(gt_by_key), key=str):
                if len(gen_witnesses) >= 5:
                    break
                _, crow = gen_by_key[k]
                gen_witnesses.append(
                    {
                        "key": fmt_key(crow, [column_map[g] for g in key_cols]),  # type: ignore[arg-type,index]
                        "gen": fmt_vals(crow, val_gen_cols),  # type: ignore[arg-type]
                    }
                )
    else:
        witness_kind = "unmatched_rows"
        mapped_gen_idxs = [column_map[g] for g in range(n_gt_cols)]

        def canonical_gt_key(rcls: list[ClassifiedCell]) -> tuple:
            """Compute the exact canonical row key tuple for a classified GT row."""
            return tuple(_exact_hash_key(rcls[g]) for g in range(n_gt_cols))

        def canonical_gen_key(rcls: list[ClassifiedCell]) -> tuple:
            """Compute the exact canonical row key tuple for a mapped generated row."""
            return tuple(_exact_hash_key(rcls[c]) for c in mapped_gen_idxs)  # type: ignore[index]

        gt_mult = Counter(canonical_gt_key(rcls) for rcls in gt_rows_cls)
        gen_mult = Counter(canonical_gen_key(rcls) for rcls in gen_rows_cls)

        if rows_only_gt > 0:
            unmatched_gt_items = [
                (gt.rows[idx], gt_rows_cls[idx])
                for idx in range(len(gt.rows))
                if idx not in matched_gt
            ]
            unmatched_gt_items.sort(
                key=lambda p: tuple(classified_sort_key(c) for c in p[1])
            )
            seen_u_gt: set[tuple] = set()
            for row, rcls in unmatched_gt_items:
                ck = canonical_gt_key(rcls)
                if ck in seen_u_gt:
                    continue
                seen_u_gt.add(ck)
                gt_witnesses.append(
                    {
                        "row": [_encode_witness_cell(c) for c in row],
                        "n_gt": gt_mult[ck],
                        "n_gen": gen_mult.get(ck, 0),
                    }
                )
                if len(gt_witnesses) >= 5:
                    break

        if rows_only_gen > 0:
            unmatched_gen_items = [
                (gen.rows[idx], gen_rows_cls[idx])
                for idx in range(len(gen.rows))
                if idx not in matched_gen
            ]
            unmatched_gen_items.sort(
                key=lambda p: tuple(
                    classified_sort_key(p[1][c]) for c in mapped_gen_idxs  # type: ignore[index]
                )
            )
            seen_u_gen: set[tuple] = set()
            for row, rcls in unmatched_gen_items:
                ck = canonical_gen_key(rcls)
                if ck in seen_u_gen:
                    continue
                seen_u_gen.add(ck)
                gen_witnesses.append(
                    {
                        "row": [_encode_witness_cell(row[c]) for c in mapped_gen_idxs],  # type: ignore[index]
                        "n_gt": gt_mult.get(ck, 0),
                        "n_gen": gen_mult[ck],
                    }
                )
                if len(gen_witnesses) >= 5:
                    break

    # Enforce empty-difference invariant: a side with "only": 0 must emit zero witnesses.
    if rows_only_gt == 0:
        gt_witnesses = []
    if rows_only_gen == 0:
        gen_witnesses = []

    row_diff = {
        "rows_matched": sum(1 for _ in matched_gt),
        "witness_kind": witness_kind,
        "gt": {
            "only": rows_only_gt,
            "n_rows": len(gt.rows),
            "stats": gt_stats if gt.rows else None,
            "witnesses": gt_witnesses,
        },
        "gen": {
            "only": rows_only_gen,
            "n_rows": len(gen.rows),
            "stats": gen_stats if gen.rows else None,
            "witnesses": gen_witnesses,
        },
    }
    return key_cols, witness_kind, row_diff


def align_tables(
    gt: ResultTable | None,
    gen: ResultTable | None,
    contract: Contract,
    *,
    rho: Decimal = DEFAULT_JUDGE_RHO,
) -> tuple[Alignment, dict[str, Any] | None, list[str]]:
    """Align GT and generated ResultTables by inferring column bijection and row matching.

    Returns ``(alignment, row_diff_dict_or_none, flags)``.
    """
    flags: list[str] = []

    if gt is None or gen is None or not gt.complete or not gen.complete:
        return (
            Alignment(
                column_map={},
                map_status="unknown",
                relation="unknown",
                key_columns=None,
                witness_kind="none",
                rows_matched=0,
                rows_only_gt=0,
                rows_only_gen=0,
                order_ok=None,
                scale_match=None,
            ),
            None,
            flags,
        )

    n_gt_rows = len(gt.rows)
    n_gen_rows = len(gen.rows)
    n_gt_cols = len(gt.columns)
    n_gen_cols = len(gen.columns)

    # Empty sides are handled before column matching.
    if n_gt_rows == 0 and n_gen_rows == 0:
        cmap = {g: g if g < n_gen_cols else None for g in range(n_gt_cols)}
        mstatus = "unique" if all(v is not None for v in cmap.values()) else "none"
        return (
            Alignment(
                column_map=cmap,
                map_status=mstatus,  # type: ignore[arg-type]
                relation="equal",
                key_columns=None,
                witness_kind="none",
                rows_matched=0,
                rows_only_gt=0,
                rows_only_gen=0,
                order_ok=True if contract.order_required else None,
                scale_match=None,
            ),
            None,
            flags,
        )

    gt_cols_raw = [[row[g] for row in gt.rows] for g in range(n_gt_cols)]
    gen_cols_raw = [[row[c] for row in gen.rows] for c in range(n_gen_cols)]
    gt_cols_cls = [[classify_cell(v) for v in col] for col in gt_cols_raw]
    gen_cols_cls = [[classify_cell(v) for v in col] for col in gen_cols_raw]
    gt_rows_cls = [[gt_cols_cls[g][r] for g in range(n_gt_cols)] for r in range(n_gt_rows)]
    gen_rows_cls = [[gen_cols_cls[c][r] for c in range(n_gen_cols)] for r in range(n_gen_rows)]

    if n_gt_rows == 0 and n_gen_rows > 0:
        cmap = {g: None for g in range(n_gt_cols)}
        _, wkind, row_diff = _build_witnesses(
            gt,
            gen,
            gt_cols_cls,
            gen_cols_cls,
            gt_rows_cls,
            gen_rows_cls,
            cmap,
            "none",
            "gt_subset_gen",
            0,
            n_gen_rows,
            set(),
            set(),
            [],
            rho,
        )
        return (
            Alignment(
                column_map=cmap,
                map_status="none",
                relation="gt_subset_gen",
                key_columns=None,
                witness_kind=wkind,  # type: ignore[arg-type]
                rows_matched=0,
                rows_only_gt=0,
                rows_only_gen=n_gen_rows,
                order_ok=False if contract.order_required else None,
                scale_match=None,
            ),
            row_diff,
            flags,
        )

    if n_gt_rows > 0 and n_gen_rows == 0:
        cmap = {g: None for g in range(n_gt_cols)}
        _, wkind, row_diff = _build_witnesses(
            gt,
            gen,
            gt_cols_cls,
            gen_cols_cls,
            gt_rows_cls,
            gen_rows_cls,
            cmap,
            "none",
            "gen_subset_gt",
            n_gt_rows,
            0,
            set(),
            set(),
            [],
            rho,
        )
        return (
            Alignment(
                column_map=cmap,
                map_status="none",
                relation="gen_subset_gt",
                key_columns=None,
                witness_kind=wkind,  # type: ignore[arg-type]
                rows_matched=0,
                rows_only_gt=n_gt_rows,
                rows_only_gen=0,
                order_ok=False if contract.order_required else None,
                scale_match=None,
            ),
            row_diff,
            flags,
        )

    # Both sides non-empty: Step 3 Column alignment
    # Store equal candidates per GT column: list of (c_idx, tolerant, scale)
    equal_cands: list[list[tuple[int, bool, Decimal]]] = [[] for _ in range(n_gt_cols)]
    for g in range(n_gt_cols):
        # First try scale=1 across all generated columns
        unscaled_hits: list[tuple[int, bool, Decimal]] = []
        for c in range(n_gen_cols):
            eq, tol = _columns_equal(
                gt_cols_raw[g],
                gen_cols_raw[c],
                gt_cols_cls[g],
                gen_cols_cls[c],
                rho=rho,
                scale=Decimal(1),
            )
            if eq:
                unscaled_hits.append((c, tol, Decimal(1)))
        if unscaled_hits:
            equal_cands[g] = unscaled_hits
        else:
            # Try scales {100, 0.01}; scaled pair is always tolerant
            for sc in (Decimal(100), Decimal("0.01")):
                for c in range(n_gen_cols):
                    eq, _ = _columns_equal(
                        gt_cols_raw[g],
                        gen_cols_raw[c],
                        gt_cols_cls[g],
                        gen_cols_cls[c],
                        rho=rho,
                        scale=sc,
                    )
                    if eq:
                        equal_cands[g].append((c, True, sc))

    all_have_equal = n_gt_cols > 0 and all(len(cands) > 0 for cands in equal_cands)

    chosen_specs: dict[int, tuple[int, bool, Decimal]] = {}
    map_status: Literal["unique", "resolved_by_search", "partial", "none", "unknown"]

    if all_have_equal:
        if all(len(cands) == 1 for cands in equal_cands) and len(
            {cands[0][0] for cands in equal_cands}
        ) == n_gt_cols:
            for g in range(n_gt_cols):
                chosen_specs[g] = equal_cands[g][0]
            map_status = "unique"
        else:
            # Backtracking search over injective assignments in lexicographic gen column order (<= 64)
            valid_assignments: list[list[tuple[int, bool, Decimal]]] = []

            def backtrack(g_idx: int, used_c: set[int], cur: list[tuple[int, bool, Decimal]]) -> None:
                """Enumerate up to `_MAX_ASSIGNMENT_SEARCH` injective equal-column bijections."""
                if len(valid_assignments) >= _MAX_ASSIGNMENT_SEARCH:
                    return
                if g_idx == n_gt_cols:
                    valid_assignments.append(list(cur))
                    return
                for cand in equal_cands[g_idx]:
                    c_idx = cand[0]
                    if c_idx in used_c:
                        continue
                    used_c.add(c_idx)
                    cur.append(cand)
                    backtrack(g_idx + 1, used_c, cur)
                    cur.pop()
                    used_c.remove(c_idx)
                    if len(valid_assignments) >= _MAX_ASSIGNMENT_SEARCH:
                        return

            # Count total possible assignments up to 65 to detect truncation
            backtrack(0, set(), [])
            if len(valid_assignments) >= _MAX_ASSIGNMENT_SEARCH:
                # Check if at least one more exists
                flags.append("assignment_search_truncated")

            if valid_assignments:
                best_matched = -1
                best_assign = valid_assignments[0]
                for assign in valid_assignments:
                    specs = [(g, assign[g][0], assign[g][1], assign[g][2]) for g in range(n_gt_cols)]
                    m_cnt, _, _, lim = _match_rows(gt_rows_cls, gen_rows_cls, specs, rho)
                    if not lim and m_cnt > best_matched:
                        best_matched = m_cnt
                        best_assign = assign
                for g in range(n_gt_cols):
                    chosen_specs[g] = best_assign[g]
                map_status = "resolved_by_search"
            else:
                all_have_equal = False

    if not all_have_equal:
        # Try partial matching (>= 50% of GT rows matched on single column)
        used_gen: set[int] = set()
        # First lock in unique equal hits if injective
        for g in range(n_gt_cols):
            if len(equal_cands[g]) == 1 and equal_cands[g][0][0] not in used_gen:
                chosen_specs[g] = equal_cands[g][0]
                used_gen.add(equal_cands[g][0][0])

        threshold = (n_gt_rows + 1) // 2  # ceil(0.5 * n_gt_rows)
        for g in range(n_gt_cols):
            if g in chosen_specs:
                continue
            best_c: tuple[int, bool, Decimal] | None = None
            best_cnt = -1
            for c in range(n_gen_cols):
                if c in used_gen:
                    continue
                exact_pair = is_column_pair_exact(gt_cols_raw[g], gen_cols_raw[c])
                cnt = _single_column_match_count(
                    gt_cols_cls[g],
                    gen_cols_cls[c],
                    tolerant=not exact_pair,
                    rho=rho,
                    scale=Decimal(1),
                )
                if cnt >= threshold and cnt > best_cnt:
                    best_cnt = cnt
                    best_c = (c, not exact_pair, Decimal(1))
            if best_c is not None:
                chosen_specs[g] = best_c
                used_gen.add(best_c[0])

        if len(chosen_specs) == 0:
            map_status = "none"
        else:
            map_status = "partial"

    column_map: dict[int, int | None] = {
        g: chosen_specs[g][0] if g in chosen_specs else None for g in range(n_gt_cols)
    }
    scales_used = {chosen_specs[g][2] for g in chosen_specs if chosen_specs[g][2] != 1}
    scale_match = float(next(iter(scales_used))) if len(scales_used) == 1 else None

    if map_status == "none" or any(v is None for v in column_map.values()):
        # Disjoint because required GT columns are missing or unmapped
        _, wkind, row_diff = _build_witnesses(
            gt,
            gen,
            gt_cols_cls,
            gen_cols_cls,
            gt_rows_cls,
            gen_rows_cls,
            column_map,
            map_status,
            "disjoint",
            n_gt_rows,
            n_gen_rows,
            set(),
            set(),
            [],
            rho,
        )
        return (
            Alignment(
                column_map=column_map,
                map_status=map_status,
                relation="disjoint",
                key_columns=None,
                witness_kind=wkind,  # type: ignore[arg-type]
                rows_matched=0,
                rows_only_gt=n_gt_rows,
                rows_only_gen=n_gen_rows,
                order_ok=False if contract.order_required else None,
                scale_match=scale_match,
            ),
            row_diff,
            flags,
        )

    col_specs = [
        (g, chosen_specs[g][0], chosen_specs[g][1], chosen_specs[g][2]) for g in range(n_gt_cols)
    ]
    rows_matched, matched_gt, matched_gen, limit_hit = _match_rows(
        gt_rows_cls, gen_rows_cls, col_specs, rho
    )
    if limit_hit:
        flags.append("tolerance_match_limit")
        return (
            Alignment(
                column_map=column_map,
                map_status=map_status,
                relation="unknown",
                key_columns=None,
                witness_kind="none",
                rows_matched=0,
                rows_only_gt=0,
                rows_only_gen=0,
                order_ok=None,
                scale_match=scale_match,
            ),
            None,
            flags,
        )

    rows_only_gt = n_gt_rows - rows_matched
    rows_only_gen = n_gen_rows - rows_matched

    if rows_only_gt == 0 and rows_only_gen == 0:
        relation: Literal[
            "equal", "gt_subset_gen", "gen_subset_gt", "overlap", "disjoint", "unknown"
        ] = "equal"
    elif rows_only_gt == 0 and rows_only_gen > 0:
        relation = "gt_subset_gen"
    elif rows_only_gt > 0 and rows_only_gen == 0:
        relation = "gen_subset_gt"
    elif rows_matched > 0:
        relation = "overlap"
    else:
        relation = "disjoint"

    order_ok: bool | None = None
    if contract.order_required:
        if n_gt_rows != n_gen_rows or relation != "equal":
            order_ok = False
        else:
            order_ok = True
            for r in range(n_gt_rows):
                for g_col, c_col, tol, scale in col_specs:
                    if not classified_equal(
                        gt_rows_cls[r][g_col],
                        gen_rows_cls[r][c_col],
                        tolerant=tol or scale != 1,
                        rho=rho,
                        scale_b=scale,
                    ):
                        order_ok = False
                        break
                if not order_ok:
                    break

    key_cols, wkind, row_diff = _build_witnesses(
        gt,
        gen,
        gt_cols_cls,
        gen_cols_cls,
        gt_rows_cls,
        gen_rows_cls,
        column_map,
        map_status,
        relation,
        rows_only_gt,
        rows_only_gen,
        matched_gt,
        matched_gen,
        col_specs,
        rho,
    )

    return (
        Alignment(
            column_map=column_map,
            map_status=map_status,
            relation=relation,
            key_columns=key_cols,
            witness_kind=wkind,  # type: ignore[arg-type]
            rows_matched=rows_matched,
            rows_only_gt=rows_only_gt,
            rows_only_gen=rows_only_gen,
            order_ok=order_ok,
            scale_match=scale_match,
        ),
        row_diff,
        flags,
    )
