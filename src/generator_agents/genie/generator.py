"""Genie Agent Mode S5 generator matching `generator_agents.langchain` and capturing full Genie traces."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Literal

from generator_agents.genie.client import parse_markdown_table
from generator_agents.genie.models import AgentResponse
from generator_agents.genie.orchestrator import GenieOrchestrator
from txt2sql_eval_harness.agent import tag_s5_trace
from txt2sql_eval_harness.codec import encode_labelled_item
from txt2sql_eval_harness.models import (
    EvalConfig,
    GroundTruth,
    LabelledItem,
    ResultTable,
    Step,
    Trace,
)
from txt2sql_eval_harness.runner import tag_slices
from txt2sql_eval_harness.sandbox import execute_sandboxed


_DATABRICKS_CITATION_RE = re.compile(r"\\?\[\[\d+\]\(https?://[^)]+\)\\?\]")


def _strip_databricks_citations(text: str) -> str:
    """Remove Databricks Genie markdown citation links so URL query IDs don't pollute claims."""
    return _DATABRICKS_CITATION_RE.sub("", text).strip()


def _sql_call_to_result_table(output_markdown: str | None) -> ResultTable | None:
    """Convert a Genie `function_call_output` markdown table into a harness ResultTable."""
    parsed = parse_markdown_table(output_markdown)
    if parsed is None:
        return None
    columns, rows = parsed
    return ResultTable(
        columns=columns,
        rows=rows,
        total_rows=len(rows),
        complete=True,
    )


def run_genie_agent(
    trace_id: str,
    group_id: str,
    question: str,
    hint: str | None,
    db_id: str,
    db_sha256: str,
    orchestrator: GenieOrchestrator,
    cfg: EvalConfig | None = None,
    *,
    agent_id: str | None = None,
    execution_mode: str | None = None,
    sql_only: bool = False,
) -> tuple[Trace | None, AgentResponse | None, dict[str, Any]]:
    """Run Genie Agent Mode on one question and return (Trace, full AgentResponse, info)."""
    config = cfg or EvalConfig()
    try:
        decision, resp = orchestrator.ask(
            question,
            hint=hint,
            agent_id=agent_id,
            new_conversation=True,
            execution_mode=execution_mode,
            allow_none=False,
        )
    except Exception as exc:
        return None, None, {"status": "agent_error", "error": f"{type(exc).__name__}: {exc}"}

    if resp is None:
        return (
            None,
            None,
            {
                "status": "routing_failed",
                "error": decision.reason,
                "routing": decision.to_dict(),
            },
        )

    steps: list[Step] = []
    for idx, sc in enumerate(resp.sql_calls[: config.max_steps], start=1):
        res_table = _sql_call_to_result_table(sc.output_markdown)
        err_kind = "runtime" if sc.status == "failed" else None
        steps.append(
            Step(
                index=idx,
                sql=sc.sql,
                result=res_table,
                error_kind=err_kind,
            )
        )

    final_answer = None if sql_only else _strip_databricks_citations(resp.answer_text)
    trace = Trace(
        trace_id=trace_id,
        group_id=group_id,
        question=question,
        hint=hint,
        db_id=db_id,
        db_sha256=db_sha256,
        steps=steps,
        final_answer=final_answer,
    )
    info: dict[str, Any] = {
        "status": "ok" if resp.status == "completed" else resp.status,
        "n_steps": len(steps),
        "n_thoughts": len(resp.thoughts),
        "routed_agent_id": decision.agent_id,
        "routed_agent_title": decision.agent_title,
        "route_confidence": decision.confidence,
    }
    return trace, resp, info


def _log_genie_progress(
    done_idx: int,
    total: int,
    item_id: str,
    db_id: str,
    question: str,
    trace: Trace | None,
    resp: AgentResponse | None,
    info: dict[str, Any],
    elapsed_s: float,
    *,
    verbose: bool,
) -> None:
    status = info.get("status", "unknown")
    n_steps = len(trace.steps) if trace is not None else 0
    n_thoughts = len(resp.thoughts) if resp is not None else 0
    routed = info.get("routed_agent_title") or info.get("routed_agent_id") or "-"
    conf = info.get("route_confidence")
    conf_str = f" p={conf:.2f}" if isinstance(conf, (int, float)) else ""
    print(
        f"[{done_idx}/{total}] {item_id} ({db_id}) -> {routed}{conf_str} "
        f"status={status} thoughts={n_thoughts} sql={n_steps} ({elapsed_s:.1f}s)",
        file=sys.stderr,
        flush=True,
    )
    if not verbose:
        return
    print(f"  Q: {question}", file=sys.stderr, flush=True)
    if resp is None:
        print(f"  ERROR: {info.get('error')}", file=sys.stderr, flush=True)
        return
    for th in resp.thoughts:
        print(f"  Thought {th.index}: {th.text}", file=sys.stderr, flush=True)
    for sc in resp.sql_calls:
        print(f"  SQL {sc.index} ({sc.title or 'query'}): {sc.sql}", file=sys.stderr, flush=True)
    print(f"  Final Answer: {resp.answer_text}", file=sys.stderr, flush=True)


def generate_genie_items(
    base_items: list[LabelledItem],
    orchestrator: GenieOrchestrator,
    *,
    db_dir: Path | None = None,
    cfg: EvalConfig | None = None,
    limit: int = 20,
    seed: int = 42,
    split: Literal["all", "calibration", "test"] = "test",
    db_id: str | None = None,
    agent_id: str | None = None,
    execution_mode: str | None = None,
    sql_only: bool = False,
    concurrency: int = 1,
    verbose: bool = False,
    out_path: Path | None = None,
    rich_out_path: Path | None = None,
) -> tuple[list[LabelledItem], list[AgentResponse], dict[str, Any]]:
    """Run Genie Agent Mode on sampled benchmark items, returning S5 LabelledItems + full AgentResponses."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    config = cfg or EvalConfig()
    orchestrator.list_agents()

    pool = [
        it
        for it in base_items
        if (split == "all" or it.split == split) and (db_id is None or it.trace.db_id == db_id)
    ]
    pool.sort(
        key=lambda it: (
            hashlib.sha256(f"{seed}:{it.group_id}:{it.item_id}".encode("utf-8")).hexdigest(),
            it.item_id,
        )
    )
    selected = pool[: max(0, limit)]

    mode_slug = (execution_mode or orchestrator.config.execution_mode).lower()
    stats: dict[str, Any] = {
        "router_model": orchestrator.config.router_model,
        "execution_mode": mode_slug,
        "requested": len(selected),
        "generated": 0,
        "routing_failed": 0,
        "agent_errors": 0,
        "total_steps": 0,
        "total_thoughts": 0,
    }
    total = len(selected)
    io_lock = threading.Lock()
    done_count = 0

    out_fh = None
    rich_fh = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fh = open(out_path, "w", encoding="utf-8")
        if rich_out_path is None:
            rich_out_path = out_path.with_suffix(".genie_responses.jsonl")
    if rich_out_path is not None:
        rich_out_path.parent.mkdir(parents=True, exist_ok=True)
        rich_fh = open(rich_out_path, "w", encoding="utf-8")

    def _run_one(
        idx: int, base: LabelledItem
    ) -> tuple[int, LabelledItem | None, AgentResponse | None, str]:
        nonlocal done_count
        t0 = time.monotonic()
        trace_id = f"s5_genie__{mode_slug}__{base.item_id}__{idx}"
        trace, resp, info = run_genie_agent(
            trace_id=trace_id,
            group_id=base.group_id,
            question=base.trace.question,
            hint=base.trace.hint,
            db_id=base.trace.db_id,
            db_sha256=base.trace.db_sha256,
            orchestrator=orchestrator,
            cfg=config,
            agent_id=agent_id,
            execution_mode=execution_mode,
            sql_only=sql_only,
        )
        if trace is None or resp is None:
            with io_lock:
                done_count += 1
                _log_genie_progress(
                    done_count,
                    total,
                    trace_id,
                    base.trace.db_id,
                    base.trace.question,
                    None,
                    None,
                    info,
                    time.monotonic() - t0,
                    verbose=verbose,
                )
            return idx, None, None, str(info.get("status", "agent_error"))

        gt = base.gt
        if gt.results is None and gt.sql_chain and db_dir is not None:
            db_path = db_dir / base.trace.db_id / f"{base.trace.db_id}.sqlite"
            if not db_path.exists():
                db_path = db_dir / f"{base.trace.db_id}.sqlite"
            if db_path.exists():
                gt_res, _, _ = execute_sandboxed(db_path, gt.sql_chain[-1], config)
                gt = GroundTruth(
                    item_ref=gt.item_ref,
                    sql_chain=list(gt.sql_chain),
                    alternatives=list(gt.alternatives),
                    reference_answer=gt.reference_answer,
                    results=[gt_res] if gt_res is not None else None,
                )

        item = LabelledItem(
            item_id=trace_id,
            group_id=base.group_id,
            split=base.split,
            set="S5",
            family=f"genie:{mode_slug}",
            slices=[],
            trace=trace,
            gt=gt,
            label="unlabelled",
            snapshot=base.snapshot,
        )
        item.slices = tag_s5_trace(item)
        item = tag_slices([item], cfg=config)[0]

        with io_lock:
            done_count += 1
            if out_fh is not None:
                out_fh.write(
                    json.dumps(encode_labelled_item(item), separators=(",", ":"), ensure_ascii=True)
                    + "\n"
                )
                out_fh.flush()
            if rich_fh is not None:
                rich_record = {
                    "item_id": trace_id,
                    "base_item_id": base.item_id,
                    "expected_db_id": base.trace.db_id,
                    **resp.to_dict(include_raw_events=False),
                }
                rich_fh.write(json.dumps(rich_record, ensure_ascii=False) + "\n")
                rich_fh.flush()
            _log_genie_progress(
                done_count,
                total,
                trace_id,
                base.trace.db_id,
                base.trace.question,
                trace,
                resp,
                info,
                time.monotonic() - t0,
                verbose=verbose,
            )
        return idx, item, resp, "ok"

    results: list[tuple[int, LabelledItem | None, AgentResponse | None, str]] = []
    try:
        if concurrency == 1:
            for idx, base in enumerate(selected):
                results.append(_run_one(idx, base))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futs = [executor.submit(_run_one, idx, base) for idx, base in enumerate(selected)]
                for fut in as_completed(futs):
                    results.append(fut.result())
    finally:
        if out_fh is not None:
            out_fh.close()
        if rich_fh is not None:
            rich_fh.close()

    results.sort(key=lambda r: r[0])
    out_items: list[LabelledItem] = []
    out_responses: list[AgentResponse] = []
    for _, item, resp, status in results:
        if status == "routing_failed":
            stats["routing_failed"] += 1
        elif status != "ok" or item is None or resp is None:
            stats["agent_errors"] += 1
        else:
            out_items.append(item)
            out_responses.append(resp)
            stats["generated"] += 1
            stats["total_steps"] += len(item.trace.steps)
            stats["total_thoughts"] += len(resp.thoughts)

    return out_items, out_responses, stats
