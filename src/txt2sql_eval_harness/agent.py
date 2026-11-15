"""Minimal bounded reference SQL agent and multi-step S5 trace tagger."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from txt2sql_eval_harness.claims import extract_and_bind_claims
from txt2sql_eval_harness.clients import CallContext, call_model
from txt2sql_eval_harness.models import LabelledItem, ModelCall, Step, Trace
from txt2sql_eval_harness.renderer import encode_packet_cell, render_state
from txt2sql_eval_harness.rubrics import AGENT_SYSTEM_PROMPT
from txt2sql_eval_harness.sandbox import execute_sandboxed


def _extract_full_ddl(db_path: Path) -> str:
    """Read schema DDL for all user tables and views from SQLite in read-only immutable mode."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table', 'view') "
            "AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    return "\n\n".join(str(r[0]) for r in rows)


def _preview_table(res: Any, max_rows: int = 20) -> dict[str, Any]:
    """Build a JSON-serializable table preview capped at *max_rows* rows for agent observations."""
    return {
        "columns": list(res.columns),
        "rows": [[encode_packet_cell(c) for c in row] for row in res.rows[:max_rows]],
        "total_rows": res.total_rows,
        "complete": res.complete,
    }


def _parse_agent_turn(raw: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract (sql_to_run, final_answer) from validated agent dict output."""
    if "final_answer" in raw and isinstance(raw["final_answer"], str):
        return None, raw["final_answer"]
    if "sql" in raw and isinstance(raw["sql"], str):
        return raw["sql"], None
    text = str(raw.get("text", "")).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("final_answer"), str):
                return None, parsed["final_answer"]
            if isinstance(parsed.get("sql"), str):
                return parsed["sql"], None
    except Exception:
        pass
    if text.upper().startswith(("SELECT ", "WITH ")):
        return text, None
    return None, text


def run_reference_agent(
    trace_id: str,
    group_id: str,
    question: str,
    hint: str,
    db_id: str,
    db_sha256: str,
    db_path: Path,
    ctx: CallContext,
    *,
    max_turns: int = 6,
) -> tuple[Trace | None, list[ModelCall], str | None]:
    """Run the reference agent via call_model('agent', ...); writes no trace when remote disabled."""
    schema_ddl = _extract_full_ddl(db_path)
    initial_block = render_state(
        {
            "QUESTION": question,
            "HINT": hint,
            "SCHEMA_DDL": schema_ddl,
        }
    )
    messages: list[str] = [initial_block]
    steps: list[Step] = []
    calls: list[ModelCall] = []
    final_answer: str | None = None

    for turn in range(1, max_turns + 1):
        req = {
            "system": AGENT_SYSTEM_PROMPT,
            "messages": "\n\n".join(messages),
            "temperature": 0.7,
            "max_output_tokens": 1024,
        }
        call = call_model("agent", "generative", ctx.cfg.agent_model, req, [], ctx)
        calls.append(call)
        if call.outcome != "ok" or call.answers is None:
            return None, calls, call.outcome

        sql_cmd, ans_text = _parse_agent_turn(call.answers)
        if sql_cmd is not None:
            res, err, _ = execute_sandboxed(db_path, sql_cmd, ctx.cfg)
            steps.append(Step(index=len(steps) + 1, sql=sql_cmd, result=res, error_kind=err))
            tool_obs = render_state(
                {
                    "STEP": len(steps),
                    "ERROR": err,
                    "RESULT_PREVIEW": _preview_table(res) if res is not None else None,
                }
            )
            messages.append(f"SQL executed:\n{render_state({'SQL': sql_cmd})}\nObservation:\n{tool_obs}")
        else:
            final_answer = ans_text or ""
            break

    if final_answer is None:
        return None, calls, "max_turns_reached"

    return (
        Trace(
            trace_id=trace_id,
            group_id=group_id,
            question=question,
            hint=hint,
            db_id=db_id,
            db_sha256=db_sha256,
            steps=steps,
            final_answer=final_answer,
        ),
        calls,
        None,
    )


def tag_s5_trace(item: LabelledItem) -> list[str]:
    """Compute structural tags (`multi_step`, `early_only_support`) for generated multi-step traces."""
    tags: list[str] = []
    steps = item.trace.steps
    if len(steps) >= 2:
        tags.append("multi_step")

    if len(steps) >= 2 and item.trace.final_answer:
        gt_res = item.gt.results[0] if item.gt.results else None
        claims, _ = extract_and_bind_claims(
            item.trace.final_answer, item.trace.question, gt_res, steps
        )
        last_step_idx = steps[-1].index
        agent_bound = [c for c in claims if c.agent_cells]
        if agent_bound and all(
            st_idx != last_step_idx
            for c in agent_bound
            for st_idx, _, _ in c.agent_cells
        ):
            tags.append("early_only_support")

    return tags
