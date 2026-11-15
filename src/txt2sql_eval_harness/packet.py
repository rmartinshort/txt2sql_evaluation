"""Evidence packet assembly, cell/token bounds checking, and cut-order trimming."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
from typing import Any, Literal

from txt2sql_eval_harness.models import (
    Alignment,
    Claim,
    Contract,
    Derivation,
    ErrorKind,
    EvalConfig,
    EvidencePacket,
    GroundTruth,
    ModeName,
    ResultTable,
    Step,
    Trace,
    Verdict,
)
from txt2sql_eval_harness.renderer import count_tokens, encode_packet_cell, render_state
from txt2sql_eval_harness.rubrics import COINCIDENCE_NOTE, PARTIAL_FETCH_NOTE, VALIDITY_NOTE, VOLATILE_NOTE


@dataclass
class PreparedCandidate:
    """All Stage 0 and Stage 1 analysis artifacts for one Trace against one GT candidate."""

    trace: Trace
    gt: GroundTruth
    gt_sql: str
    gt_sql_n: str | None
    gt_result: ResultTable | None
    gt_volatile: bool
    gt_det: bool
    gt_rd_eligible: bool
    contract: Contract
    schema_ddl: str
    steps: list[Step]
    step_sql_n: list[str | None]
    step_volatile: list[bool]
    step_det: list[bool]
    step_rd_eligible: list[bool]
    alignment: Alignment
    step_alignments: list[Alignment] = field(default_factory=list)
    row_diff: dict[str, Any] | None = None
    coincident_result: bool = False
    claims: list[Claim] = field(default_factory=list)
    derivations: list[Derivation] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    rule_hit: Verdict | None = None
    method_ok: bool | None = None
    no_evidence: bool = False
    all_complete: bool = False


def _encode_table(table: ResultTable | None) -> dict[str, Any] | None:
    """Serialize a ResultTable into packet-ready JSON with optional partial-fetch metadata."""
    if table is None:
        return None
    obj: dict[str, Any] = {
        "columns": list(table.columns),
        "rows": [[encode_packet_cell(c) for c in row] for row in table.rows],
    }
    if not table.complete:
        prefix_obj: dict[str, Any] = {
            "n_fetched": len(table.rows),
            "total_rows": table.total_rows,
        }
        if table.cut_by is not None:
            prefix_obj["cut_by"] = table.cut_by
        obj["prefix"] = prefix_obj
    return obj


def _side_summary(
    table: ResultTable | None,
    error_kind: ErrorKind | None,
    volatile: bool,
    rd_eligible: bool,
) -> dict[str, Any]:
    """Summarize execution availability, completeness, and row counts for GT or generated SQL."""
    err: str | None = "volatile" if volatile else error_kind
    return {
        "available": (table is not None) and (not volatile) and (error_kind is None),
        "error_kind": err,
        "complete": bool(table.complete) if table is not None else False,
        "n_fetched": len(table.rows) if table is not None else 0,
        "total_rows": table.total_rows if table is not None else None,
        "rd_eligible": rd_eligible,
    }


def _claim_to_packet_dict(c: Claim) -> dict[str, Any]:
    """Serialize a bound Claim (and its matching derivations) into packet JSON format."""
    val: Any
    if isinstance(c.value, Decimal):
        val = str(c.value)
    else:
        val = c.value
    return {
        "text": c.text,
        "kind": c.kind,
        "value": val,
        "decimals": c.decimals,
        "in_question": c.in_question,
        "gt_cells": [[r, col] for r, col in c.gt_cells],
        "agent_cells": [[st, r, col] for st, r, col in c.agent_cells],
        "derivations": [
            {
                "pool": d.pool,
                "step": d.step,
                "op": d.op,
                "column": d.column,
                "value": encode_packet_cell(d.value),
            }
            for d in c.derivations
        ],
    }


def _derivation_to_packet_dict(d: Derivation) -> dict[str, Any]:
    """Serialize a precomputed Derivation (ROW_COUNT, SUM, MIN, MAX, diff) for the packet."""
    return {
        "pool": d.pool,
        "step": d.step,
        "op": d.op,
        "column": d.column,
        "value": encode_packet_cell(d.value),
    }


def _count_packet_cells(obj: dict[str, Any]) -> int:
    """Count total table and witness cells inside an evidence packet dictionary."""
    total = 0

    def _table_cells(tbl: dict[str, Any] | None) -> int:
        """Count rows * cells inside a serialized table dictionary."""
        if not tbl:
            return 0
        return sum(len(r) for r in tbl.get("rows", []))

    if "SMALL_RESULTS" in obj:
        sr = obj["SMALL_RESULTS"]
        total += _table_cells(sr.get("gt"))
        total += _table_cells(sr.get("gen"))

    if "PREFIX_ROWS" in obj:
        pr = obj["PREFIX_ROWS"]
        for side_key in ("gt", "gen"):
            if side_key in pr and pr[side_key] is not None:
                total += len(pr[side_key].get("rows", [])) * len(pr[side_key].get("columns", []))

    if "ROW_DIFF" in obj:
        rd = obj["ROW_DIFF"]
        for side_key in ("gt", "gen"):
            side = rd.get(side_key) or {}
            for w in side.get("witnesses", []):
                if "row" in w:
                    total += len(w["row"])
                else:
                    total += len(w.get("key", [])) + len(w.get("gt", [])) + len(w.get("gen", []))

    if "REFERENCE_RESULT" in obj:
        total += _table_cells(obj["REFERENCE_RESULT"].get("table"))

    if "AGENT_STEPS" in obj:
        for st in obj["AGENT_STEPS"]:
            total += _table_cells(st.get("table"))

    return total


def _trim_extra_witnesses(row_diff: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Trim ROW_DIFF witnesses beyond the first witness per side; return (trimmed, changed)."""
    trimmed = copy.deepcopy(row_diff)
    changed = False
    for side_key in ("gt", "gen"):
        side = trimmed.get(side_key)
        if side and isinstance(side.get("witnesses"), list) and len(side["witnesses"]) > 1:
            side["witnesses"] = side["witnesses"][:1]
            changed = True
    return trimmed, changed


def build_packet(
    cand: PreparedCandidate,
    mode: ModeName,
    tier: Literal["jev", "escalation"],
    cfg: EvalConfig,
) -> EvidencePacket:
    """Assemble and bound an EvidencePacket for H, H+E, B6, or B6+E.

    Identical inputs produce byte-identical packet state and sha256 across H, H+E, B6, and B6+E.
    """
    max_cells = cfg.jev_max_cells if tier == "jev" else cfg.escalation_max_cells
    max_tokens = cfg.jev_max_tokens if tier == "jev" else cfg.escalation_max_tokens

    base_obj: dict[str, Any] = {
        "QUESTION": cand.trace.question,
        "HINT": cand.trace.hint,
    }
    optional_sections: dict[str, Any] = {}

    if mode == "sql_only":
        gen_step = cand.steps[-1] if cand.steps else None
        gen_sql = gen_step.sql if gen_step is not None else None
        gen_sql_n = cand.step_sql_n[-1] if cand.step_sql_n else None
        gen_res = gen_step.result if gen_step is not None else None
        gen_err = gen_step.error_kind if gen_step is not None else None
        gen_vol = cand.step_volatile[-1] if cand.step_volatile else False
        gen_rd = cand.step_rd_eligible[-1] if cand.step_rd_eligible else False

        base_obj["SCHEMA_DDL"] = {
            "validity_note": VALIDITY_NOTE,
            "ddl": cand.schema_ddl,
        }
        base_obj["REFERENCE_SQL"] = {
            "raw": cand.gt_sql,
            "normalized": cand.gt_sql_n,
        }
        base_obj["GENERATED_SQL"] = {
            "raw": gen_sql,
            "normalized": gen_sql_n,
        }
        base_obj["RESULT_SUMMARY"] = {
            "gt": _side_summary(cand.gt_result, None, cand.gt_volatile, cand.gt_rd_eligible),
            "gen": _side_summary(gen_res, gen_err, gen_vol, gen_rd),
            "relation": cand.alignment.relation,
            "map_status": cand.alignment.map_status,
            "order_ok": cand.alignment.order_ok,
            "scale_match": cand.alignment.scale_match,
            "coincidence_note": COINCIDENCE_NOTE if cand.coincident_result else None,
            "volatile_note": VOLATILE_NOTE if (cand.gt_volatile or gen_vol) else None,
        }

        both_complete = (
            cand.gt_result is not None
            and cand.gt_result.complete
            and gen_res is not None
            and gen_res.complete
            and not cand.gt_volatile
            and not gen_vol
        )
        if both_complete and cand.alignment.relation != "equal" and cand.row_diff is not None:
            base_obj["ROW_DIFF"] = copy.deepcopy(cand.row_diff)

        # Optional sections (cut order 1): PREFIX_ROWS or SMALL_RESULTS
        if both_complete and len(cand.gt_result.rows) <= 20 and len(gen_res.rows) <= 20:  # type: ignore[arg-type]
            optional_sections["SMALL_RESULTS"] = {
                "gt": _encode_table(cand.gt_result),
                "gen": _encode_table(gen_res),
            }
        else:
            prefix_payload: dict[str, Any] = {}
            if cand.gt_result is not None and not cand.gt_result.complete:
                prefix_payload["gt"] = {
                    "note": PARTIAL_FETCH_NOTE,
                    "columns": list(cand.gt_result.columns),
                    "rows": [
                        [encode_packet_cell(c) for c in row] for row in cand.gt_result.rows[:5]
                    ],
                    "prefix": {
                        "n_fetched": len(cand.gt_result.rows),
                        "total_rows": cand.gt_result.total_rows,
                    },
                }
            if gen_res is not None and not gen_res.complete:
                prefix_payload["gen"] = {
                    "note": PARTIAL_FETCH_NOTE,
                    "columns": list(gen_res.columns),
                    "rows": [
                        [encode_packet_cell(c) for c in row] for row in gen_res.rows[:5]
                    ],
                    "prefix": {
                        "n_fetched": len(gen_res.rows),
                        "total_rows": gen_res.total_rows,
                    },
                }
            if prefix_payload:
                optional_sections["PREFIX_ROWS"] = prefix_payload
    else:
        base_obj["FINAL_ANSWER"] = cand.trace.final_answer
        base_obj["REFERENCE_RESULT"] = {
            "sql": cand.gt_sql,
            "table": _encode_table(cand.gt_result),
        }
        base_obj["AGENT_STEPS"] = [
            {
                "step": st.index,
                "sql": st.sql,
                "error_kind": st.error_kind,
                "table": _encode_table(st.result),
            }
            for st in cand.steps
        ]
        base_obj["CLAIMS"] = [_claim_to_packet_dict(c) for c in cand.claims]
        base_obj["DERIVATIONS"] = [_derivation_to_packet_dict(d) for d in cand.derivations]

    def _try_render(candidate_obj: dict[str, Any]) -> tuple[str, int, int, bool]:
        """Render *candidate_obj* and test cell and token budgets."""
        state = render_state(candidate_obj)
        n_cells = _count_packet_cells(candidate_obj)
        n_tokens = count_tokens(state)
        fits = (n_cells <= max_cells) and (n_tokens <= max_tokens)
        return state, n_cells, n_tokens, fits

    truncated: list[str] = []

    # Attempt 0: base + optional sections
    if optional_sections:
        full_obj = dict(base_obj)
        full_obj.update(optional_sections)
        state, n_cells, n_tokens, fits = _try_render(full_obj)
        if fits:
            return EvidencePacket(
                tier=tier,
                state=state,
                n_cells=n_cells,
                n_tokens=n_tokens,
                required_met=True,
                truncated=[],
                packet_sha256=hashlib.sha256(state.encode("utf-8")).hexdigest(),
            )
        # Cut order 1: drop optional sections
        for key in optional_sections:
            truncated.append(key.lower())

    # Attempt 1: required sections only
    state, n_cells, n_tokens, fits = _try_render(base_obj)
    if fits:
        return EvidencePacket(
            tier=tier,
            state=state,
            n_cells=n_cells,
            n_tokens=n_tokens,
            required_met=True,
            truncated=truncated,
            packet_sha256=hashlib.sha256(state.encode("utf-8")).hexdigest(),
        )

    # Cut order 2: trim extra ROW_DIFF witnesses beyond the first per side
    if "ROW_DIFF" in base_obj:
        trimmed_rd, changed = _trim_extra_witnesses(base_obj["ROW_DIFF"])
        if changed:
            trimmed_obj = dict(base_obj)
            trimmed_obj["ROW_DIFF"] = trimmed_rd
            truncated.append("extra_row_diff_witnesses")
            state, n_cells, n_tokens, fits = _try_render(trimmed_obj)
            return EvidencePacket(
                tier=tier,
                state=state,
                n_cells=n_cells,
                n_tokens=n_tokens,
                required_met=fits,
                truncated=truncated,
                packet_sha256=hashlib.sha256(state.encode("utf-8")).hexdigest(),
            )

    return EvidencePacket(
        tier=tier,
        state=state,
        n_cells=n_cells,
        n_tokens=n_tokens,
        required_met=False,
        truncated=truncated,
        packet_sha256=hashlib.sha256(state.encode("utf-8")).hexdigest(),
    )
