"""Deterministic checks D0/D2, derivations, and value-bound claim extraction (S-claims)."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
import unicodedata
from typing import Iterable

from txt2sql_eval_harness.comparator import classify_cell
from txt2sql_eval_harness.models import (
    COUNTEREXAMPLE_ERROR_KINDS,
    Cell,
    Claim,
    Derivation,
    ModeName,
    ResultTable,
    Step,
)
from txt2sql_eval_harness.sql_analysis import is_rd_eligible

_MULTIPLIERS: dict[str, Decimal] = {
    "thousand": Decimal("1000"),
    "k": Decimal("1000"),
    "million": Decimal("1000000"),
    "m": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
}

_MONTHS: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")
_MONTH_YEAR_RE = re.compile(
    r"\b("
    + "|".join(sorted(_MONTHS.keys(), key=len, reverse=True))
    + r")\.?\s+(\d{4})\b",
    re.IGNORECASE,
)

# Numeric token: optional sign (not glued to word/dot on the left), optional currency,
# digits with optional comma grouping or plain digits + optional fractional part,
# optional trailing % or magnitude suffix.
_NUM_TOKEN_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<sign1>[+-])?"
    r"(?P<currency>[$€£])?"
    r"(?P<sign2>[+-])?"
    r"(?P<mantissa>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?:\s*(?P<percent>%)|\s+(?P<mag_word>thousand|million|billion|bn)|(?P<mag_short>bn|[km])(?![A-Za-z0-9_]))?"
    r"(?![\w]|\.\d)",
    re.IGNORECASE,
)


def _is_zero_or_null(cell: Cell) -> bool:
    """Return True if *cell* is SQLite NULL or numerically equal to zero."""
    if cell is None:
        return True
    cc = classify_cell(cell)
    if cc.kind in ("exact_number", "approx_number") and cc.num is not None:
        return cc.num == 0
    return False


def check_d0(final_answer: str | None) -> bool:
    """D0 (full-trace): True iff final_answer is empty after NFC + strip."""
    if final_answer is None:
        return False
    return unicodedata.normalize("NFC", final_answer).strip() == ""


def check_d2(
    mode: ModeName,
    gt_sql: str,
    gt_result: ResultTable | None,
    steps: Iterable[Step],
) -> bool:
    """D2 (both modes): GT complete R* has >=1 non-null, non-zero cell, AND every step is
    complete empty or has a counterexample-grade error. SQL-only additionally requires
    both GT and generated SQL to be rd_eligible."""
    if gt_result is None or not gt_result.complete:
        return False

    has_nonzero_gt = any(
        not _is_zero_or_null(cell)
        for row in gt_result.rows
        for cell in row
    )
    if not has_nonzero_gt:
        return False

    step_list = list(steps)
    if not step_list:
        return False

    for step in step_list:
        if step.error_kind in COUNTEREXAMPLE_ERROR_KINDS:
            continue
        if (
            step.error_kind is None
            and step.result is not None
            and step.result.complete
            and len(step.result.rows) == 0
        ):
            continue
        return False

    if mode == "sql_only":
        if not is_rd_eligible(gt_sql):
            return False
        for step in step_list:
            if not is_rd_eligible(step.sql):
                return False

    return True


def _table_derivations(
    table: ResultTable,
    pool: str,
    step_idx: int | None,
) -> list[tuple[Derivation, Decimal | None]]:
    """Compute ROW_COUNT and per-numeric-column SUM/AVG/MIN/MAX/COUNT for a complete table."""
    if not table.complete:
        return []

    out: list[tuple[Derivation, Decimal | None]] = []
    row_count = table.total_rows if table.total_rows is not None else len(table.rows)
    out.append(
        (
            Derivation(
                pool=pool,  # type: ignore[arg-type]
                step=step_idx,
                op="ROW_COUNT",
                column=None,
                value=int(row_count),
            ),
            Decimal(row_count),
        )
    )

    n_cols = len(table.columns)
    for col_idx in range(n_cols):
        col_name = table.columns[col_idx]
        nums: list[tuple[Decimal, str, Cell]] = []
        has_non_numeric = False
        for row in table.rows:
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            if cell is None:
                continue
            cc = classify_cell(cell)
            if cc.kind in ("exact_number", "approx_number") and cc.num is not None:
                nums.append((cc.num, cc.kind, cell))
            else:
                has_non_numeric = True
                break
        if has_non_numeric or not nums:
            continue

        all_exact = all(k == "exact_number" for _, k, _ in nums)
        total = sum((val for val, _, _ in nums), Decimal(0))
        cnt = len(nums)
        avg_dec = total / Decimal(cnt)

        min_entry = min(nums, key=lambda item: item[0])
        max_entry = max(nums, key=lambda item: item[0])

        sum_cell: Cell
        if all_exact and total == total.to_integral_value():
            sum_cell = int(total)
        else:
            sum_cell = float(total)

        out.extend(
            [
                (
                    Derivation(
                        pool=pool,  # type: ignore[arg-type]
                        step=step_idx,
                        op="SUM",
                        column=col_name,
                        value=sum_cell,
                    ),
                    total,
                ),
                (
                    Derivation(
                        pool=pool,  # type: ignore[arg-type]
                        step=step_idx,
                        op="AVG",
                        column=col_name,
                        value=float(avg_dec),
                    ),
                    avg_dec,
                ),
                (
                    Derivation(
                        pool=pool,  # type: ignore[arg-type]
                        step=step_idx,
                        op="MIN",
                        column=col_name,
                        value=min_entry[2],
                    ),
                    min_entry[0],
                ),
                (
                    Derivation(
                        pool=pool,  # type: ignore[arg-type]
                        step=step_idx,
                        op="MAX",
                        column=col_name,
                        value=max_entry[2],
                    ),
                    max_entry[0],
                ),
                (
                    Derivation(
                        pool=pool,  # type: ignore[arg-type]
                        step=step_idx,
                        op="COUNT",
                        column=col_name,
                        value=int(cnt),
                    ),
                    Decimal(cnt),
                ),
            ]
        )

    return out


def compute_derivations(
    gt_result: ResultTable | None,
    steps: Iterable[Step],
) -> list[Derivation]:
    """Compute derivations for R* and every complete agent step."""
    pairs: list[tuple[Derivation, Decimal | None]] = []
    if gt_result is not None and gt_result.complete:
        pairs.extend(_table_derivations(gt_result, "reference", None))
    for step in steps:
        if step.error_kind is None and step.result is not None and step.result.complete:
            pairs.extend(_table_derivations(step.result, "agent", step.index))
    return [d for d, _ in pairs]


def _extract_dates_and_mask(text: str) -> tuple[list[tuple[int, int, str, str]], str]:
    """Return list of (start, end, raw_text, canonical_key) and text with date spans blanked."""
    matches: list[tuple[int, int, str, str]] = []
    occupied = [False] * len(text)

    for m in _ISO_DATE_RE.finditer(text):
        year, month, day = int(m.group(1)), int(m.group(2)), m.group(3)
        if not (1 <= month <= 12):
            continue
        if day is not None:
            d_int = int(day)
            if not (1 <= d_int <= 31):
                continue
            canon = f"{year:04d}-{month:02d}-{d_int:02d}"
        else:
            canon = f"{year:04d}-{month:02d}"
        matches.append((m.start(), m.end(), m.group(0), canon))
        for i in range(m.start(), m.end()):
            occupied[i] = True

    for m in _MONTH_YEAR_RE.finditer(text):
        if any(occupied[i] for i in range(m.start(), m.end())):
            continue
        month_num = _MONTHS[m.group(1).lower()]
        year = int(m.group(2))
        canon = f"{year:04d}-{month_num:02d}"
        matches.append((m.start(), m.end(), m.group(0), canon))
        for i in range(m.start(), m.end()):
            occupied[i] = True

    masked_chars = [" " if occupied[i] else ch for i, ch in enumerate(text)]
    matches.sort(key=lambda item: item[0])
    return matches, "".join(masked_chars)


def _extract_numbers(masked_text: str) -> list[tuple[int, str, Decimal, int, Decimal, bool]]:
    """Extract (start, raw_text, value_scaled, decimals, multiplier, is_percent)."""
    results: list[tuple[int, str, Decimal, int, Decimal, bool]] = []
    for m in _NUM_TOKEN_RE.finditer(masked_text):
        sign1 = m.group("sign1") or ""
        sign2 = m.group("sign2") or ""
        if sign1 and sign2:
            continue
        sign = sign1 or sign2
        mantissa_raw = m.group("mantissa").replace(",", "")
        if "." in mantissa_raw:
            decimals = len(mantissa_raw.split(".", 1)[1])
        else:
            decimals = 0
        try:
            mantissa = Decimal(sign + mantissa_raw)
        except InvalidOperation:
            continue

        is_percent = bool(m.group("percent"))
        mag_token = m.group("mag_word") or m.group("mag_short")
        multiplier = Decimal(1)
        if mag_token is not None and not is_percent:
            multiplier = _MULTIPLIERS[mag_token.lower()]

        value = mantissa * multiplier
        results.append((m.start(), m.group(0), value, decimals, multiplier, is_percent))
    return results


def _number_matches(
    claim_val: Decimal,
    decimals: int,
    multiplier: Decimal,
    is_percent: bool,
    cell_val: Decimal,
) -> bool:
    """Return True if *claim_val* matches *cell_val* (allowing x100 percent scaling) within half-ULP tolerance."""
    scales = (Decimal(1), Decimal(100)) if is_percent else (Decimal(1),)
    tol = Decimal("0.5") * (Decimal(10) ** (-decimals)) * multiplier
    for s in scales:
        if abs(claim_val - cell_val * s) <= tol:
            return True
    return False


def _cell_date_canons(cell_text: str) -> set[str]:
    """Return canonical YYYY-MM-DD and/or YYYY-MM strings represented by a TEXT cell."""
    canons: set[str] = set()
    s = unicodedata.normalize("NFC", cell_text).strip()
    m = re.match(r"^(\d{4})-(\d{2})(?:-(\d{2}))?(?:\b|[T\s])", s)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= month <= 12:
            canons.add(f"{year:04d}-{month:02d}")
            if day is not None and 1 <= int(day) <= 31:
                canons.add(f"{year:04d}-{month:02d}-{int(day):02d}")
    for _, _, _, canon in _extract_dates_and_mask(s)[0]:
        canons.add(canon)
    return canons


def _date_claim_matches_cell(claim_canon: str, cell: Cell) -> bool:
    """Return True if *cell* is a date-like TEXT cell matching canonical date string *claim_canon*."""
    if not isinstance(cell, str):
        return False
    cell_canons = _cell_date_canons(cell)
    if claim_canon in cell_canons:
        return True
    # Also allow literal month-year in answer to match full date in cell if YYYY-MM matches,
    # already handled because full ISO cells add YYYY-MM to cell_canons.
    return False


def extract_and_bind_claims(
    final_answer: str,
    question: str,
    gt_result: ResultTable | None,
    steps: Iterable[Step],
) -> tuple[list[Claim], list[Derivation]]:
    """Extract numeric, date, and entity claims from final_answer and bind them to R*,
    complete agent step tables, question, and derivations."""
    answer_nfc = unicodedata.normalize("NFC", final_answer)
    question_nfc = unicodedata.normalize("NFC", question)
    step_list = list(steps)

    # Precompute derivations with exact Decimal values for binding
    deriv_pairs: list[tuple[Derivation, Decimal | None]] = []
    if gt_result is not None and gt_result.complete:
        deriv_pairs.extend(_table_derivations(gt_result, "reference", None))
    for step in step_list:
        if step.error_kind is None and step.result is not None and step.result.complete:
            deriv_pairs.extend(_table_derivations(step.result, "agent", step.index))
    all_derivations = [d for d, _ in deriv_pairs]

    # Precompute question dates & numbers for in_question flags
    q_dates, q_masked = _extract_dates_and_mask(question_nfc)
    q_date_canons = {canon for _, _, _, canon in q_dates}
    q_numbers = _extract_numbers(q_masked)

    # Pre-index numeric cells and text cells in complete R* and agent tables
    gt_num_cells: list[tuple[int, int, Decimal]] = []
    gt_str_cells: list[tuple[int, int, str]] = []
    if gt_result is not None and gt_result.complete:
        for r_idx, row in enumerate(gt_result.rows):
            for c_idx, cell in enumerate(row):
                if cell is None:
                    continue
                if isinstance(cell, str):
                    gt_str_cells.append((r_idx, c_idx, cell))
                cc = classify_cell(cell)
                if cc.kind in ("exact_number", "approx_number") and cc.num is not None:
                    gt_num_cells.append((r_idx, c_idx, cc.num))

    agent_num_cells: list[tuple[int, int, int, Decimal]] = []
    agent_str_cells: list[tuple[int, int, int, str]] = []
    for step in step_list:
        if step.error_kind is None and step.result is not None and step.result.complete:
            for r_idx, row in enumerate(step.result.rows):
                for c_idx, cell in enumerate(row):
                    if cell is None:
                        continue
                    if isinstance(cell, str):
                        agent_str_cells.append((step.index, r_idx, c_idx, cell))
                    cc = classify_cell(cell)
                    if cc.kind in ("exact_number", "approx_number") and cc.num is not None:
                        agent_num_cells.append((step.index, r_idx, c_idx, cc.num))

    claims_with_pos: list[tuple[int, Claim]] = []

    # 1. Dates
    ans_dates, ans_masked = _extract_dates_and_mask(answer_nfc)
    for start, _, raw_date, canon in ans_dates:
        gt_matches = [
            (r, c) for r, c, cell_str in gt_str_cells if _date_claim_matches_cell(canon, cell_str)
        ]
        agent_matches = [
            (s, r, c)
            for s, r, c, cell_str in agent_str_cells
            if _date_claim_matches_cell(canon, cell_str)
        ]
        in_q = canon in q_date_canons or bool(
            re.search(rf"(?<!\w){re.escape(raw_date)}(?!\w)", question_nfc, re.IGNORECASE)
        )
        claims_with_pos.append(
            (
                start,
                Claim(
                    text=raw_date,
                    kind="date",
                    value=canon,
                    decimals=None,
                    in_question=in_q,
                    gt_cells=gt_matches,
                    agent_cells=agent_matches,
                    derivations=[],
                ),
            )
        )

    # 2. Numbers
    for start, raw_num, val, decimals, mult, is_pct in _extract_numbers(ans_masked):
        gt_matches = [
            (r, c)
            for r, c, cell_val in gt_num_cells
            if _number_matches(val, decimals, mult, is_pct, cell_val)
        ]
        agent_matches = [
            (s, r, c)
            for s, r, c, cell_val in agent_num_cells
            if _number_matches(val, decimals, mult, is_pct, cell_val)
        ]
        matched_derivs = [
            deriv
            for deriv, d_val in deriv_pairs
            if d_val is not None and _number_matches(val, decimals, mult, is_pct, d_val)
        ]
        in_q = any(
            q_val == val
            or (is_pct and _number_matches(val, decimals, mult, True, q_val))
            or (q_pct and _number_matches(q_val, q_dec, q_mult, True, val))
            for _, _, q_val, q_dec, q_mult, q_pct in q_numbers
        )
        claims_with_pos.append(
            (
                start,
                Claim(
                    text=raw_num,
                    kind="number",
                    value=val,
                    decimals=decimals,
                    in_question=in_q,
                    gt_cells=gt_matches,
                    agent_cells=agent_matches,
                    derivations=matched_derivs,
                ),
            )
        )

    # 3. Entities: distinct TEXT cells of length >= 3 found as case-insensitive whole word
    seen_entities: dict[str, str] = {}
    for _, _, cell_str in gt_str_cells:
        norm_cell = unicodedata.normalize("NFC", cell_str).strip()
        if len(norm_cell) >= 3:
            seen_entities.setdefault(norm_cell.casefold(), norm_cell)
    for _, _, _, cell_str in agent_str_cells:
        norm_cell = unicodedata.normalize("NFC", cell_str).strip()
        if len(norm_cell) >= 3:
            seen_entities.setdefault(norm_cell.casefold(), norm_cell)

    for entity_cf, entity_display in seen_entities.items():
        # Skip entities whose normalized form is already classified as pure number
        cc_ent = classify_cell(entity_display)
        if cc_ent.kind in ("exact_number", "approx_number"):
            continue
        pat = re.compile(rf"(?<!\w){re.escape(entity_display)}(?!\w)", re.IGNORECASE)
        m = pat.search(answer_nfc)
        if m is None:
            continue
        gt_matches = [
            (r, c)
            for r, c, cell_str in gt_str_cells
            if unicodedata.normalize("NFC", cell_str).strip().casefold() == entity_cf
        ]
        agent_matches = [
            (s, r, c)
            for s, r, c, cell_str in agent_str_cells
            if unicodedata.normalize("NFC", cell_str).strip().casefold() == entity_cf
        ]
        in_q = bool(pat.search(question_nfc))
        claims_with_pos.append(
            (
                m.start(),
                Claim(
                    text=m.group(0),
                    kind="entity",
                    value=entity_display,
                    decimals=None,
                    in_question=in_q,
                    gt_cells=gt_matches,
                    agent_cells=agent_matches,
                    derivations=[],
                ),
            )
        )

    claims_with_pos.sort(key=lambda item: item[0])
    return [c for _, c in claims_with_pos], all_derivations
