"""LangChain Text2SQL agent generator producing S5 Trace/LabelledItem records."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Literal

from txt2sql_eval_harness.agent import tag_s5_trace
from txt2sql_eval_harness.clients import _get_api_key
from txt2sql_eval_harness.codec import encode_labelled_item
from txt2sql_eval_harness.models import (
    DEFAULT_AGENT_MODEL,
    EvalConfig,
    GroundTruth,
    LabelledItem,
    Step,
    Trace,
)
from txt2sql_eval_harness.runner import tag_slices
from txt2sql_eval_harness.sandbox import execute_sandboxed


def _require_langchain() -> tuple[Any, Any, Any, Any, Any]:
    """Lazily import LangChain SQL agent dependencies or raise a clear install hint."""
    try:
        from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
        from langchain_community.utilities import SQLDatabase
        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError(
            "LangChain dependencies are not installed. "
            "Install them with: pip install -e '.[langchain]'"
        ) from exc
    return (
        SQLDatabase,
        SQLDatabaseToolkit,
        create_sql_agent,
        BaseCallbackHandler,
        ChatPromptTemplate,
        MessagesPlaceholder,
        ChatGoogleGenerativeAI,
    )


def build_chat_model(
    model: str = DEFAULT_AGENT_MODEL,
    *,
    temperature: float = 0.0,
    api_key: str | None = None,
    request_timeout: float = 60.0,
) -> Any:
    """Construct a Google Gemini LangChain chat model using GEMINI_API_KEY."""
    _, _, _, _, _, _, ChatGoogleGenerativeAI = _require_langchain()
    resolved_key = api_key or _get_api_key("GEMINI_API_KEY")
    if not resolved_key:
        raise RuntimeError("GEMINI_API_KEY is required for LangChain Gemini agent")
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=resolved_key,
        temperature=temperature,
        request_timeout=request_timeout,
    )


def _extract_sql_from_tool_input(tool_input: Any) -> str | None:
    """Extract a SQL string from LangChain `sql_db_query` tool input."""
    if isinstance(tool_input, str):
        s = tool_input.strip()
        return s or None
    if isinstance(tool_input, dict):
        for key in ("query", "sql", "input"):
            val = tool_input.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _normalize_output_text(output: Any) -> str:
    """Normalize LangChain agent `output` (str or multimodal content blocks) into plain text."""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, list):
        parts: list[str] = []
        for block in output:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(p for p in parts if p).strip()
    if output is None:
        return ""
    return str(output).strip()


def _open_readonly_sql_database(db_path: Path) -> Any:
    """Open a SQLite database in read-only immutable mode for LangChain SQLDatabase."""
    SQLDatabase, _, _, _, _, _, _ = _require_langchain()
    abs_posix = db_path.resolve().as_posix()
    return SQLDatabase.from_uri(
        f"sqlite:///file:{abs_posix}?mode=ro&immutable=1&uri=true",
    )


def run_langchain_sql_agent(
    trace_id: str,
    group_id: str,
    question: str,
    hint: str | None,
    db_id: str,
    db_sha256: str,
    db_path: Path,
    llm: Any,
    cfg: EvalConfig | None = None,
    *,
    max_iterations: int = 50,
    max_execution_time: float | None = 120.0,
    sql_only: bool = False,
) -> tuple[Trace | None, dict[str, Any]]:
    """Run LangChain's SQL agent on one question and capture sandboxed Steps + final_answer."""
    (
        _,
        SQLDatabaseToolkit,
        create_sql_agent,
        BaseCallbackHandler,
        ChatPromptTemplate,
        MessagesPlaceholder,
        _,
    ) = _require_langchain()
    config = cfg or EvalConfig()
    captured_sqls: list[str] = []

    class _SqlCaptureCallback(BaseCallbackHandler):  # type: ignore[misc, valid-type]
        def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            inputs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            tool_name = serialized.get("name", "") if isinstance(serialized, dict) else ""
            if tool_name == "sql_db_query":
                sql = _extract_sql_from_tool_input(inputs if inputs is not None else input_str)
                if sql is not None:
                    captured_sqls.append(sql)

    db = _open_readonly_sql_database(db_path)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    # Use a prompt ending with human + agent_scratchpad so Gemini never receives a trailing model turn.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an agent designed to interact with a SQLite database. "
                "Given an input question, inspect the schema using the available tools, "
                "write and execute syntactically correct SQLite SELECT queries to gather the answer, "
                "and then return a concise, accurate final answer grounded in the query results. "
                "Never issue DML/DDL statements.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="tool-calling",
        prompt=prompt,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        verbose=False,
        agent_executor_kwargs={"return_intermediate_steps": True},
    )

    prompt_input = question if not hint else f"{question}\n\nHint: {hint}"
    try:
        result = agent.invoke(
            {"input": prompt_input},
            config={"callbacks": [_SqlCaptureCallback()]},
        )
    except Exception as exc:
        return None, {"status": "agent_error", "error": f"{type(exc).__name__}: {exc}"}

    # Fall back to intermediate_steps if callback did not fire (e.g., custom mock executors)
    if not captured_sqls and isinstance(result, dict):
        for action, _obs in result.get("intermediate_steps", []):
            if getattr(action, "tool", None) == "sql_db_query":
                sql = _extract_sql_from_tool_input(getattr(action, "tool_input", None))
                if sql is not None:
                    captured_sqls.append(sql)

    steps: list[Step] = []
    for idx, sql in enumerate(captured_sqls[: config.max_steps], start=1):
        res, err, _ = execute_sandboxed(db_path, sql, config)
        steps.append(Step(index=idx, sql=sql, result=res, error_kind=err))

    final_answer: str | None = None
    if not sql_only:
        raw_out = result.get("output", "") if isinstance(result, dict) else result
        final_answer = _normalize_output_text(raw_out)

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
    return trace, {"status": "ok", "n_steps": len(steps)}


def _log_item_progress(
    done_idx: int,
    total: int,
    item_id: str,
    db_id: str,
    question: str,
    trace: Trace | None,
    info: dict[str, Any],
    elapsed_s: float,
    *,
    verbose: bool,
) -> None:
    """Print per-item progress (and full question/trace details when verbose=True) to stderr."""
    status = info.get("status", "unknown")
    n_steps = len(trace.steps) if trace is not None else 0
    print(
        f"[{done_idx}/{total}] {item_id} ({db_id}) status={status} steps={n_steps} ({elapsed_s:.1f}s)",
        file=sys.stderr,
        flush=True,
    )
    if not verbose:
        return
    print(f"  Q: {question}", file=sys.stderr, flush=True)
    if trace is None:
        print(f"  ERROR: {info.get('error')}", file=sys.stderr, flush=True)
        return
    for st in trace.steps:
        err_note = f" [ERROR: {st.error_kind}]" if st.error_kind else ""
        n_rows = len(st.result.rows) if st.result is not None else 0
        print(f"  Step {st.index}{err_note} (rows={n_rows}): {st.sql}", file=sys.stderr, flush=True)
    print(f"  Final Answer: {trace.final_answer}", file=sys.stderr, flush=True)


def generate_langchain_items(
    base_items: list[LabelledItem],
    db_dir: Path,
    llm: Any,
    model_name: str,
    cfg: EvalConfig | None = None,
    *,
    limit: int = 20,
    seed: int = 42,
    split: Literal["all", "calibration", "test"] = "test",
    db_id: str | None = None,
    max_iterations: int = 50,
    max_execution_time: float | None = 120.0,
    sql_only: bool = False,
    concurrency: int = 1,
    verbose: bool = False,
    out_path: Path | None = None,
) -> tuple[list[LabelledItem], dict[str, Any]]:
    """Run LangChain SQL agent on sampled benchmark items and return S5 LabelledItems."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    config = cfg or EvalConfig()
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

    model_slug = model_name.replace("/", "_").replace(":", "_")
    stats: dict[str, Any] = {
        "model": model_name,
        "requested": len(selected),
        "generated": 0,
        "agent_errors": 0,
        "missing_db": 0,
        "total_steps": 0,
    }
    total = len(selected)
    io_lock = threading.Lock()
    done_count = 0

    out_fh = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fh = open(out_path, "w", encoding="utf-8")

    def _run_one(idx: int, base: LabelledItem) -> tuple[int, LabelledItem | None, str]:
        nonlocal done_count
        t0 = time.monotonic()
        db_path = db_dir / base.trace.db_id / f"{base.trace.db_id}.sqlite"
        if not db_path.exists():
            alt = db_dir / f"{base.trace.db_id}.sqlite"
            if alt.exists():
                db_path = alt
            else:
                with io_lock:
                    done_count += 1
                    _log_item_progress(
                        done_count,
                        total,
                        base.item_id,
                        base.trace.db_id,
                        base.trace.question,
                        None,
                        {"status": "missing_db"},
                        time.monotonic() - t0,
                        verbose=verbose,
                    )
                return idx, None, "missing_db"

        trace_id = f"s5_lc__{model_slug}__{base.item_id}__{idx}"
        trace, info = run_langchain_sql_agent(
            trace_id=trace_id,
            group_id=base.group_id,
            question=base.trace.question,
            hint=base.trace.hint,
            db_id=base.trace.db_id,
            db_sha256=base.trace.db_sha256,
            db_path=db_path,
            llm=llm,
            cfg=config,
            max_iterations=max_iterations,
            max_execution_time=max_execution_time,
            sql_only=sql_only,
        )
        if trace is None:
            with io_lock:
                done_count += 1
                _log_item_progress(
                    done_count,
                    total,
                    trace_id,
                    base.trace.db_id,
                    base.trace.question,
                    None,
                    info,
                    time.monotonic() - t0,
                    verbose=verbose,
                )
            return idx, None, "agent_error"

        gt = base.gt
        if gt.results is None and gt.sql_chain:
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
            family=f"langchain:{model_slug}",
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
                out_fh.write(json.dumps(encode_labelled_item(item), separators=(",", ":"), ensure_ascii=True) + "\n")
                out_fh.flush()
            _log_item_progress(
                done_count,
                total,
                trace_id,
                base.trace.db_id,
                base.trace.question,
                trace,
                info,
                time.monotonic() - t0,
                verbose=verbose,
            )
        return idx, item, "ok"

    results: list[tuple[int, LabelledItem | None, str]] = []
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

    results.sort(key=lambda r: r[0])
    out_items: list[LabelledItem] = []
    for _, item, status in results:
        if status == "missing_db":
            stats["missing_db"] += 1
        elif status == "agent_error":
            stats["agent_errors"] += 1
        elif item is not None:
            out_items.append(item)
            stats["generated"] += 1
            stats["total_steps"] += len(item.trace.steps)

    return out_items, stats
