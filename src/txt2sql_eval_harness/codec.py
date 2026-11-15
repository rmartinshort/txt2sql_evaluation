"""Lossless JSONL cell codec and serialization helpers for core data contracts."""

from __future__ import annotations

import base64
import json
from decimal import Decimal
from typing import Any

from txt2sql_eval_harness.models import (
    Alignment,
    Attempt,
    CallMeasure,
    Cell,
    Claim,
    Derivation,
    GroundTruth,
    LabelledItem,
    ModelCall,
    Record,
    ResultTable,
    RunIdentity,
    Step,
    Trace,
    Verdict,
)


class CellCodecError(ValueError):
    """Raised when a value cannot be losslessly encoded or decoded as a SQLite Cell."""


def encode_cell(cell: Any) -> Any:
    """Encode a SQLite Cell losslessly for JSONL storage.

    NULL -> None
    INTEGER -> {"i": "<digits>"} (preserves integers > 2**53)
    REAL -> {"r": "<repr>"}
    TEXT -> str
    BLOB -> {"b": "<base64>"}
    Any other type (including bool) raises CellCodecError.
    """
    if cell is None:
        return None
    if isinstance(cell, bool):
        raise CellCodecError(f"bool is not a valid SQLite storage class: {cell!r}")
    if isinstance(cell, int):
        return {"i": str(cell)}
    if isinstance(cell, float):
        return {"r": repr(cell)}
    if isinstance(cell, str):
        return cell
    if isinstance(cell, bytes):
        return {"b": base64.b64encode(cell).decode("ascii")}
    raise CellCodecError(f"Unsupported cell type: {type(cell).__name__}")


def decode_cell(raw: Any) -> Cell:
    """Decode a JSONL-encoded cell back to a Python SQLite Cell."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict) and len(raw) == 1:
        if "i" in raw and isinstance(raw["i"], str):
            return int(raw["i"])
        if "r" in raw and isinstance(raw["r"], str):
            return float(raw["r"])
        if "b" in raw and isinstance(raw["b"], str):
            return base64.b64decode(raw["b"].encode("ascii"))
    raise CellCodecError(f"Invalid encoded cell representation: {raw!r}")


def encode_result_table(table: ResultTable | None) -> dict[str, Any] | None:
    """Serialize a ResultTable with lossless cell encoding."""
    if table is None:
        return None
    out: dict[str, Any] = {
        "columns": list(table.columns),
        "rows": [[encode_cell(c) for c in row] for row in table.rows],
        "total_rows": table.total_rows,
        "complete": table.complete,
    }
    if table.cut_by is not None:
        out["cut_by"] = table.cut_by
    return out


def decode_result_table(raw: dict[str, Any] | None) -> ResultTable | None:
    """Deserialize a ResultTable from JSON."""
    if raw is None:
        return None
    return ResultTable(
        columns=list(raw["columns"]),
        rows=[tuple(decode_cell(c) for c in row) for row in raw["rows"]],
        total_rows=raw.get("total_rows"),
        complete=bool(raw["complete"]),
        cut_by=raw.get("cut_by"),
    )


def encode_step(step: Step) -> dict[str, Any]:
    """Serialize a Step dataclass into a JSON-compatible dictionary."""
    return {
        "index": step.index,
        "sql": step.sql,
        "result": encode_result_table(step.result),
        "error_kind": step.error_kind,
    }


def decode_step(raw: dict[str, Any]) -> Step:
    """Deserialize a Step dataclass from decoded JSON."""
    return Step(
        index=int(raw["index"]),
        sql=str(raw["sql"]),
        result=decode_result_table(raw.get("result")),
        error_kind=raw.get("error_kind"),
    )


def encode_trace(trace: Trace) -> dict[str, Any]:
    """Serialize a Trace dataclass into a JSON-compatible dictionary."""
    return {
        "trace_id": trace.trace_id,
        "group_id": trace.group_id,
        "question": trace.question,
        "hint": trace.hint,
        "db_id": trace.db_id,
        "db_sha256": trace.db_sha256,
        "steps": [encode_step(s) for s in trace.steps],
        "final_answer": trace.final_answer,
    }


def decode_trace(raw: dict[str, Any]) -> Trace:
    """Deserialize a Trace dataclass from decoded JSON."""
    return Trace(
        trace_id=str(raw["trace_id"]),
        group_id=str(raw["group_id"]),
        question=str(raw["question"]),
        hint=raw.get("hint"),
        db_id=str(raw["db_id"]),
        db_sha256=str(raw["db_sha256"]),
        steps=[decode_step(s) for s in raw["steps"]],
        final_answer=raw.get("final_answer"),
    )


def encode_gt(gt: GroundTruth) -> dict[str, Any]:
    """Serialize a GroundTruth reference into a JSON-compatible dictionary."""
    return {
        "item_ref": gt.item_ref,
        "sql_chain": list(gt.sql_chain),
        "alternatives": [list(a) for a in gt.alternatives],
        "reference_answer": gt.reference_answer,
        "results": [encode_result_table(r) for r in gt.results]
        if gt.results is not None
        else None,
    }


def decode_gt(raw: dict[str, Any]) -> GroundTruth:
    """Deserialize a GroundTruth reference from decoded JSON."""
    raw_results = raw.get("results")
    results = (
        [decode_result_table(r) for r in raw_results]  # type: ignore[misc]
        if raw_results is not None
        else None
    )
    return GroundTruth(
        item_ref=str(raw["item_ref"]),
        sql_chain=list(raw["sql_chain"]),
        alternatives=[list(a) for a in raw.get("alternatives", [])],
        reference_answer=raw.get("reference_answer"),
        results=results,  # type: ignore[arg-type]
    )


def encode_labelled_item(item: LabelledItem) -> dict[str, Any]:
    """Serialize a LabelledItem benchmark record into a JSON-compatible dictionary."""
    return {
        "item_id": item.item_id,
        "group_id": item.group_id,
        "split": item.split,
        "set": item.set,
        "family": item.family,
        "slices": list(item.slices),
        "trace": encode_trace(item.trace),
        "gt": encode_gt(item.gt),
        "label": item.label,
        "pair_of": item.pair_of,
        "method_ok_label": item.method_ok_label,
        "snapshot": item.snapshot,
    }


def decode_labelled_item(raw: dict[str, Any]) -> LabelledItem:
    """Deserialize a LabelledItem benchmark record from decoded JSON."""
    return LabelledItem(
        item_id=str(raw["item_id"]),
        group_id=str(raw["group_id"]),
        split=raw["split"],
        set=raw["set"],
        family=str(raw["family"]),
        slices=list(raw.get("slices", [])),
        trace=decode_trace(raw["trace"]),
        gt=decode_gt(raw["gt"]),
        label=raw["label"],
        pair_of=raw.get("pair_of"),
        method_ok_label=raw.get("method_ok_label"),
        snapshot=str(raw.get("snapshot", "minidev")),
    )


def encode_derivation(d: Derivation) -> dict[str, Any]:
    """Serialize a precomputed Derivation record into a JSON-compatible dictionary."""
    return {
        "pool": d.pool,
        "step": d.step,
        "op": d.op,
        "column": d.column,
        "value": encode_cell(d.value),
    }


def decode_derivation(raw: dict[str, Any]) -> Derivation:
    """Deserialize a Derivation record from decoded JSON."""
    return Derivation(
        pool=raw["pool"],
        step=raw.get("step"),
        op=raw["op"],
        column=raw.get("column"),
        value=decode_cell(raw.get("value")),
    )


def encode_claim(c: Claim) -> dict[str, Any]:
    """Serialize a bound Claim (preserving exact Decimal values) into JSON."""
    return {
        "text": c.text,
        "kind": c.kind,
        "value": {"decimal": str(c.value)} if isinstance(c.value, Decimal) else c.value,
        "decimals": c.decimals,
        "in_question": c.in_question,
        "gt_cells": [list(p) for p in c.gt_cells],
        "agent_cells": [list(p) for p in c.agent_cells],
        "derivations": [encode_derivation(d) for d in c.derivations],
    }


def decode_claim(raw: dict[str, Any]) -> Claim:
    """Deserialize a bound Claim from decoded JSON."""
    raw_val = raw["value"]
    val: Decimal | str
    if isinstance(raw_val, dict) and "decimal" in raw_val:
        val = Decimal(raw_val["decimal"])
    else:
        val = str(raw_val)
    return Claim(
        text=str(raw["text"]),
        kind=raw["kind"],
        value=val,
        decimals=raw.get("decimals"),
        in_question=bool(raw["in_question"]),
        gt_cells=[(int(r), int(c)) for r, c in raw.get("gt_cells", [])],
        agent_cells=[(int(s), int(r), int(c)) for s, r, c in raw.get("agent_cells", [])],
        derivations=[decode_derivation(d) for d in raw.get("derivations", [])],
    )


def encode_alignment(a: Alignment) -> dict[str, Any]:
    """Serialize a table Alignment summary into a JSON-compatible dictionary."""
    return {
        "column_map": {str(k): v for k, v in a.column_map.items()},
        "map_status": a.map_status,
        "relation": a.relation,
        "key_columns": list(a.key_columns) if a.key_columns is not None else None,
        "witness_kind": a.witness_kind,
        "rows_matched": a.rows_matched,
        "rows_only_gt": a.rows_only_gt,
        "rows_only_gen": a.rows_only_gen,
        "order_ok": a.order_ok,
        "scale_match": a.scale_match,
    }


def decode_alignment(raw: dict[str, Any]) -> Alignment:
    """Deserialize a table Alignment summary from decoded JSON."""
    return Alignment(
        column_map={int(k): v for k, v in raw["column_map"].items()},
        map_status=raw["map_status"],
        relation=raw["relation"],
        key_columns=list(raw["key_columns"]) if raw.get("key_columns") is not None else None,
        witness_kind=raw["witness_kind"],
        rows_matched=int(raw["rows_matched"]),
        rows_only_gt=int(raw["rows_only_gt"]),
        rows_only_gen=int(raw["rows_only_gen"]),
        order_ok=raw.get("order_ok"),
        scale_match=raw.get("scale_match"),
    )


def encode_attempt(a: Attempt) -> dict[str, Any]:
    """Serialize an HTTP/API Attempt record into a JSON-compatible dictionary."""
    return {
        "n": a.n,
        "wait_ms": a.wait_ms,
        "elapsed_ms": a.elapsed_ms,
        "result": a.result,
        "input_tokens": a.input_tokens,
        "output_tokens": a.output_tokens,
        "billing": a.billing,
        "cost_usd": a.cost_usd,
    }


def decode_attempt(raw: dict[str, Any]) -> Attempt:
    """Deserialize an Attempt record from decoded JSON."""
    return Attempt(
        n=int(raw["n"]),
        wait_ms=int(raw["wait_ms"]),
        elapsed_ms=int(raw["elapsed_ms"]),
        result=raw["result"],
        input_tokens=int(raw["input_tokens"]),
        output_tokens=raw.get("output_tokens"),
        billing=raw["billing"],
        cost_usd=float(raw["cost_usd"]),
    )


def encode_call_measure(m: CallMeasure) -> dict[str, Any]:
    """Serialize a CallMeasure record into a JSON-compatible dictionary."""
    return {
        "attempts": [encode_attempt(a) for a in m.attempts],
        "wall_ms": m.wall_ms,
        "billed_usd": m.billed_usd,
        "ambiguous_usd": m.ambiguous_usd,
    }


def decode_call_measure(raw: dict[str, Any]) -> CallMeasure:
    """Deserialize a CallMeasure record from decoded JSON."""
    return CallMeasure(
        attempts=[decode_attempt(a) for a in raw.get("attempts", [])],
        wall_ms=int(raw["wall_ms"]),
        billed_usd=float(raw["billed_usd"]),
        ambiguous_usd=float(raw["ambiguous_usd"]),
    )


def encode_model_call(mc: ModelCall) -> dict[str, Any]:
    """Serialize a ModelCall telemetry record into a JSON-compatible dictionary."""
    return {
        "stage": mc.stage,
        "model": mc.model,
        "request_sha256": mc.request_sha256,
        "questions": list(mc.questions),
        "answers": mc.answers,
        "outcome": mc.outcome,
        "invalid_reason": mc.invalid_reason,
        "from_cache": mc.from_cache,
        "cold": encode_call_measure(mc.cold),
    }


def decode_model_call(raw: dict[str, Any]) -> ModelCall:
    """Deserialize a ModelCall telemetry record from decoded JSON."""
    return ModelCall(
        stage=raw["stage"],
        model=str(raw["model"]),
        request_sha256=str(raw["request_sha256"]),
        questions=list(raw["questions"]),
        answers=raw.get("answers"),
        outcome=raw["outcome"],
        invalid_reason=raw.get("invalid_reason"),
        from_cache=bool(raw["from_cache"]),
        cold=decode_call_measure(raw["cold"]),
    )


def encode_verdict(v: Verdict) -> dict[str, Any]:
    """Serialize a Verdict into a JSON-compatible dictionary."""
    return {
        "trace_id": v.trace_id,
        "method": v.method,
        "mode": v.mode,
        "label": v.label,
        "decided_by": v.decided_by,
        "reason": v.reason,
        "escalated": v.escalated,
        "method_ok": v.method_ok,
        "flags": list(v.flags),
        "claims": [encode_claim(c) for c in v.claims],
        "alignments": [encode_alignment(a) for a in v.alignments],
        "packet_sha256": v.packet_sha256,
        "b4_context": v.b4_context,
        "b4_rows_shown": v.b4_rows_shown,
        "b4_rows_total": v.b4_rows_total,
        "calls": [encode_model_call(mc) for mc in v.calls],
        "det_ms": v.det_ms,
        "esc_prep_ms": v.esc_prep_ms,
        "cost_billed_usd": v.cost_billed_usd,
        "cost_conservative_usd": v.cost_conservative_usd,
        "latency_ms": v.latency_ms,
        "spend_usd": v.spend_usd,
        "config_hash": v.config_hash,
    }


def decode_verdict(raw: dict[str, Any]) -> Verdict:
    """Deserialize a Verdict from decoded JSON."""
    return Verdict(
        trace_id=str(raw["trace_id"]),
        method=raw["method"],
        mode=raw["mode"],
        label=raw["label"],
        decided_by=raw["decided_by"],
        reason=raw.get("reason"),
        escalated=bool(raw.get("escalated", False)),
        method_ok=raw.get("method_ok"),
        flags=list(raw.get("flags", [])),
        claims=[decode_claim(c) for c in raw.get("claims", [])],
        alignments=[decode_alignment(a) for a in raw.get("alignments", [])],
        packet_sha256=raw.get("packet_sha256"),
        b4_context=raw.get("b4_context"),
        b4_rows_shown=raw.get("b4_rows_shown"),
        b4_rows_total=raw.get("b4_rows_total"),
        calls=[decode_model_call(mc) for mc in raw.get("calls", [])],
        det_ms=int(raw.get("det_ms", 0)),
        esc_prep_ms=int(raw.get("esc_prep_ms", 0)),
        cost_billed_usd=float(raw.get("cost_billed_usd", 0.0)),
        cost_conservative_usd=float(raw.get("cost_conservative_usd", 0.0)),
        latency_ms=int(raw.get("latency_ms", 0)),
        spend_usd=float(raw.get("spend_usd", 0.0)),
        config_hash=str(raw.get("config_hash", "")),
    )


def encode_record(rec: Record) -> str:
    """Serialize a Record to one compact JSONL line."""
    return json.dumps(
        {
            "item_id": rec.item_id,
            "method": rec.method,
            "verdict": encode_verdict(rec.verdict),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def decode_record(line: str) -> Record:
    """Deserialize one JSONL line into a Record."""
    raw = json.loads(line)
    return Record(
        item_id=str(raw["item_id"]),
        method=raw["method"],
        verdict=decode_verdict(raw["verdict"]),
    )


def encode_run_identity(ident: RunIdentity) -> dict[str, Any]:
    """Serialize a RunIdentity compatibility fingerprint into JSON."""
    return {
        "config_hash": ident.config_hash,
        "allow_remote": ident.allow_remote,
        "items_sha256": ident.items_sha256,
        "manifest_sha256": ident.manifest_sha256,
        "code_commit": ident.code_commit,
        "code_dirty": ident.code_dirty,
        "cache_only": ident.cache_only,
    }


def decode_run_identity(raw: dict[str, Any]) -> RunIdentity:
    """Deserialize a RunIdentity fingerprint from decoded run.json data."""
    return RunIdentity(
        config_hash=str(raw["config_hash"]),
        allow_remote=bool(raw["allow_remote"]),
        items_sha256=str(raw["items_sha256"]),
        manifest_sha256=str(raw["manifest_sha256"]),
        code_commit=str(raw["code_commit"]),
        code_dirty=bool(raw["code_dirty"]),
        cache_only=bool(raw["cache_only"]),
    )
