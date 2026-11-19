"""Structured data models for Genie agents, routing decisions, and full Agent Mode traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class GenieAgent:
    """Metadata for a single Databricks Genie Space / Agent."""

    agent_id: str
    title: str
    description: str = ""
    warehouse_id: str | None = None
    create_time: str | None = None
    update_time: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteDecision:
    """Routing decision produced by the Gemini Decision Model (`gemini-decision-1`)."""

    agent_id: str | None
    agent_title: str | None
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    is_followup: bool = False
    is_followup_prob: float | None = None
    needs_clarification: bool = False
    reason: str = ""
    router_model: str = "gemini-decision-1"
    pinned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThoughtStep:
    """One intermediate reasoning / thought item emitted by Genie Agent Mode."""

    index: int
    item_id: str
    text: str
    summary: list[str] = field(default_factory=list)
    status: str = "completed"
    sequence_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SqlCall:
    """One SQL query executed by Genie Agent Mode (`execute_sql` tool call + output)."""

    index: int
    call_id: str
    sql: str
    title: str | None = None
    status: str = "completed"
    statement_id: str | None = None
    output_markdown: str | None = None
    raw_arguments: str | None = None
    sequence_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimelineItem:
    """Ordered event capturing the exact sequence of Genie Agent Mode items."""

    index: int
    kind: Literal["thought", "sql_call", "sql_output", "message", "other"]
    item_id: str | None
    content: str
    status: str | None = None
    sequence_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResponse:
    """Complete captured output of a single Genie Agent Mode turn."""

    agent_id: str
    agent_title: str | None
    question: str
    conversation_id: str | None
    response_id: str | None
    status: str
    answer_text: str
    thoughts: list[ThoughtStep] = field(default_factory=list)
    sql_calls: list[SqlCall] = field(default_factory=list)
    timeline: list[TimelineItem] = field(default_factory=list)
    routing: RouteDecision | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    raw_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sql_queries(self) -> list[str]:
        """Convenience list of all SQL strings executed during this turn."""
        return [c.sql for c in self.sql_calls if c.sql]

    def to_dict(self, *, include_raw_events: bool = False) -> dict[str, Any]:
        """Serialize the complete captured Genie response to a JSON-friendly dict."""
        out: dict[str, Any] = {
            "agent_id": self.agent_id,
            "agent_title": self.agent_title,
            "question": self.question,
            "conversation_id": self.conversation_id,
            "response_id": self.response_id,
            "status": self.status,
            "answer_text": self.answer_text,
            "thoughts": [t.to_dict() for t in self.thoughts],
            "sql_calls": [s.to_dict() for s in self.sql_calls],
            "timeline": [ti.to_dict() for ti in self.timeline],
            "routing": self.routing.to_dict() if self.routing is not None else None,
            "error": self.error,
            "elapsed_s": round(self.elapsed_s, 3),
        }
        if include_raw_events:
            out["raw_events"] = list(self.raw_events)
        return out
