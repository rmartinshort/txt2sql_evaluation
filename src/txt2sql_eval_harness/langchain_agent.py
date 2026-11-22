"""Backwards-compatible re-export shim — canonical implementation lives in `generator_agents.langchain.agent`."""

from __future__ import annotations

from generator_agents.langchain.agent import (  # noqa: F401
    _extract_sql_from_tool_input,
    _log_item_progress,
    _normalize_output_text,
    _open_readonly_sql_database,
    _require_langchain,
    build_chat_model,
    generate_langchain_items,
    run_langchain_sql_agent,
)

__all__ = [
    "build_chat_model",
    "generate_langchain_items",
    "run_langchain_sql_agent",
]
