"""Cell classification, total sort ordering, and column-pair tolerance comparison."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal, Sequence

from txt2sql_eval_harness.models import Cell

CellClass = Literal["null", "exact_number", "approx_number", "text", "blob"]

_INT_RE = re.compile(r"^[+-]?\d+$", re.ASCII)
_APPROX_RE = re.compile(
    r"^(?:[+-]?(?:\d+\.\d*|\.\d+|\d+)[eE][+-]?\d+|[+-]?(?:\d+\.\d*|\.\d+))$",
    re.ASCII,
)
_ASCII_WS = " \t\n\r\f\v"

DEFAULT_JUDGE_RHO = Decimal("1e-9")
S2_LABEL_RHO = Decimal("1e-4")


@dataclass(frozen=True)
class ClassifiedCell:
    """Normalized representation of a Cell under the `cmp` specification."""

    kind: CellClass
    num: Decimal | None = None
    exact_int: int | None = None
    text: str | None = None
    blob: bytes | None = None


def classify_cell(cell: Cell) -> ClassifiedCell:
    """Classify a SQLite Cell into null, exact_number, approx_number, text, or blob."""
    if cell is None:
        return ClassifiedCell(kind="null")
    if isinstance(cell, bool):
        raise TypeError(f"bool is not a valid SQLite storage class: {cell!r}")
    if isinstance(cell, int):
        return ClassifiedCell(kind="exact_number", num=Decimal(cell), exact_int=cell)
    if isinstance(cell, float):
        return ClassifiedCell(kind="approx_number", num=Decimal(repr(cell)))
    if isinstance(cell, bytes):
        return ClassifiedCell(kind="blob", blob=cell)
    if isinstance(cell, str):
        stripped = cell.strip(_ASCII_WS)
        if _INT_RE.match(stripped):
            iv = int(stripped)
            return ClassifiedCell(kind="exact_number", num=Decimal(iv), exact_int=iv)
        if _APPROX_RE.match(stripped):
            try:
                dv = Decimal(stripped)
                return ClassifiedCell(kind="approx_number", num=dv)
            except InvalidOperation:
                pass
        return ClassifiedCell(kind="text", text=unicodedata.normalize("NFC", cell))
    raise TypeError(f"Unsupported cell type: {type(cell).__name__}")


def is_column_pair_exact(col_a: Sequence[Cell], col_b: Sequence[Cell]) -> bool:
    """Return True iff every numeric cell across both columns is an exact_number."""
    for c in col_a:
        cc = classify_cell(c)
        if cc.kind == "approx_number":
            return False
    for c in col_b:
        cc = classify_cell(c)
        if cc.kind == "approx_number":
            return False
    return True


def _decimal_equal(
    a: Decimal,
    b: Decimal,
    *,
    tolerant: bool,
    rho: Decimal,
    scale_b: Decimal,
) -> bool:
    """Compare two Decimal numbers with optional relative tolerance *rho* and right-hand scale factor."""
    if a.is_nan() or b.is_nan():
        return False
    if a.is_infinite() or b.is_infinite():
        if a.is_infinite() and b.is_infinite():
            # Apply scale sign so -inf * -1 == +inf
            b_neg = b.is_signed() ^ (scale_b < 0)
            return a.is_signed() == b_neg
        return False
    vb = b * scale_b
    if not tolerant and scale_b == 1:
        return a == vb
    diff = abs(a - vb)
    bound = rho * max(abs(a), abs(vb))
    return diff <= bound


def classified_equal(
    ca: ClassifiedCell,
    cb: ClassifiedCell,
    *,
    tolerant: bool,
    rho: Decimal = DEFAULT_JUDGE_RHO,
    scale_b: Decimal = Decimal(1),
) -> bool:
    """Compare two ClassifiedCells given pair exactness and optional scale factor."""
    if ca.kind == "null" or cb.kind == "null":
        return ca.kind == "null" and cb.kind == "null"
    if ca.kind in ("exact_number", "approx_number") and cb.kind in (
        "exact_number",
        "approx_number",
    ):
        assert ca.num is not None and cb.num is not None
        if not tolerant and scale_b == 1:
            if ca.kind == "exact_number" and cb.kind == "exact_number":
                return ca.exact_int == cb.exact_int
            return False
        return _decimal_equal(ca.num, cb.num, tolerant=True, rho=rho, scale_b=scale_b)
    if ca.kind == "text" and cb.kind == "text":
        return ca.text == cb.text
    if ca.kind == "blob" and cb.kind == "blob":
        return ca.blob == cb.blob
    return False


def cells_equal(
    a: Cell,
    b: Cell,
    *,
    tolerant: bool,
    rho: Decimal = DEFAULT_JUDGE_RHO,
    scale_b: Decimal = Decimal(1),
) -> bool:
    """Compare two raw Cells under `cmp` semantics."""
    return classified_equal(
        classify_cell(a),
        classify_cell(b),
        tolerant=tolerant,
        rho=rho,
        scale_b=scale_b,
    )


def total_sort_key(cell: Cell) -> tuple:
    """Total sort order: null < number (exact before approx at equal value) < text < blob."""
    cc = classify_cell(cell)
    return classified_sort_key(cc)


def classified_sort_key(cc: ClassifiedCell) -> tuple:
    """Total sort order key for a pre-classified cell."""
    if cc.kind == "null":
        return (0, 0, Decimal(0), 0, "", b"")
    if cc.kind in ("exact_number", "approx_number"):
        assert cc.num is not None
        exact_rank = 0 if cc.kind == "exact_number" else 1
        if cc.num.is_nan():
            return (1, 3, Decimal(0), exact_rank, "", b"")
        if cc.num.is_infinite():
            inf_rank = 0 if cc.num.is_signed() else 2
            return (1, inf_rank, Decimal(0), exact_rank, "", b"")
        return (1, 1, cc.num, exact_rank, "", b"")
    if cc.kind == "text":
        assert cc.text is not None
        return (2, 0, Decimal(0), 0, cc.text, b"")
    assert cc.blob is not None
    return (3, 0, Decimal(0), 0, "", cc.blob)


def sort_key_with_scale(cc: ClassifiedCell, scale: Decimal = Decimal(1)) -> tuple:
    """Total sort key after scaling numbers by *scale*."""
    if scale == 1 or cc.kind not in ("exact_number", "approx_number"):
        return classified_sort_key(cc)
    assert cc.num is not None
    if cc.num.is_nan():
        return (1, 3, Decimal(0), 1, "", b"")
    if cc.num.is_infinite():
        b_neg = cc.num.is_signed() ^ (scale < 0)
        inf_rank = 0 if b_neg else 2
        return (1, inf_rank, Decimal(0), 1, "", b"")
    return (1, 1, cc.num * scale, 1, "", b"")
