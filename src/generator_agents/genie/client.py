"""Databricks Genie Agent Mode HTTP/SSE client and full response parser."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Iterable, Sequence

import httpx

from generator_agents.genie.auth import DatabricksAuthProvider
from generator_agents.genie.models import (
    AgentResponse,
    GenieAgent,
    SqlCall,
    ThoughtStep,
    TimelineItem,
)

EventCallback = Callable[[dict[str, Any]], None]


class GenieApiError(RuntimeError):
    """Raised on non-2xx responses from the Databricks Genie API."""

    def __init__(self, status_code: int, message: str, body: str = "") -> None:
        super().__init__(f"Databricks Genie API HTTP {status_code}: {message}")
        self.status_code = status_code
        self.body = body


def _extract_reasoning_text(item: dict[str, Any]) -> tuple[str, list[str]]:
    """Extract full reasoning text and summary strings from a `reasoning` output item."""
    texts: list[str] = []
    for block in item.get("content") or []:
        if isinstance(block, dict):
            txt = block.get("text")
            if isinstance(txt, str) and txt.strip():
                texts.append(txt.strip())
        elif isinstance(block, str) and block.strip():
            texts.append(block.strip())

    summaries: list[str] = []
    for block in item.get("summary") or []:
        if isinstance(block, dict):
            txt = block.get("text")
            if isinstance(txt, str) and txt.strip():
                summaries.append(txt.strip())
        elif isinstance(block, str) and block.strip():
            summaries.append(block.strip())

    return "\n".join(texts).strip(), summaries


def _extract_message_text(item: dict[str, Any]) -> str:
    """Extract assistant message text from a `message` output item."""
    parts: list[str] = []
    for block in item.get("content") or []:
        if isinstance(block, dict):
            txt = block.get("text")
            if isinstance(txt, str) and txt:
                parts.append(txt)
        elif isinstance(block, str) and block:
            parts.append(block)
    return "\n".join(parts).strip()


def _parse_sql_arguments(raw_args: Any) -> tuple[str | None, str | None, str]:
    """Extract (sql, title, raw_arguments_str) from a Genie `function_call` item."""
    if isinstance(raw_args, dict):
        sql = raw_args.get("sql") or raw_args.get("query")
        title = raw_args.get("title")
        return (
            sql.strip() if isinstance(sql, str) and sql.strip() else None,
            title.strip() if isinstance(title, str) and title.strip() else None,
            json.dumps(raw_args, ensure_ascii=False),
        )
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                sql = parsed.get("sql") or parsed.get("query")
                title = parsed.get("title")
                return (
                    sql.strip() if isinstance(sql, str) and sql.strip() else None,
                    title.strip() if isinstance(title, str) and title.strip() else None,
                    raw_args,
                )
        except json.JSONDecodeError:
            pass
        return None, None, raw_args
    return None, None, ""


def _summarize_serialized_space(serialized_space: str | None) -> str:
    """Extract a concise table-list fallback description from Genie `serialized_space` JSON."""
    if not serialized_space:
        return ""
    try:
        parsed = json.loads(serialized_space)
    except json.JSONDecodeError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    data_sources = parsed.get("data_sources") or {}
    tables = data_sources.get("tables") if isinstance(data_sources, dict) else None
    table_names: list[str] = []
    if isinstance(tables, list):
        for tbl in tables:
            if isinstance(tbl, dict):
                ident = tbl.get("identifier")
                if isinstance(ident, str) and ident.strip():
                    table_names.append(ident.strip().split(".")[-1])
    if table_names:
        return f"Tables: {', '.join(table_names[:20])}."
    return ""


def _parse_cell_value(raw: str) -> None | int | float | str:
    """Convert a markdown table cell string into a typed scalar when unambiguous."""
    val = raw.strip()
    if val == "" or val.lower() in {"null", "none", "nan"}:
        return None
    if re.fullmatch(r"-?\d+", val):
        try:
            return int(val)
        except ValueError:
            return val
    if re.fullmatch(r"-?\d+\.\d+", val):
        try:
            return float(val)
        except ValueError:
            return val
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", val):
        clean = val.replace(",", "")
        try:
            return float(clean) if "." in clean else int(clean)
        except ValueError:
            return val
    return val


def parse_markdown_table(markdown: str | None) -> tuple[list[str], list[tuple[Any, ...]]] | None:
    """Parse the first GitHub-flavored markdown table from Genie `function_call_output`."""
    if not markdown:
        return None
    lines = [ln.strip() for ln in markdown.splitlines()]
    for idx in range(len(lines) - 1):
        hdr = lines[idx]
        sep = lines[idx + 1]
        if "|" not in hdr or "|" not in sep:
            continue
        sep_cells = [c.strip() for c in sep.strip("|").split("|")]
        if not sep_cells or not all(re.fullmatch(r":?-{2,}:?", c) for c in sep_cells):
            continue
        columns = [c.strip() for c in hdr.strip("|").split("|")]
        rows: list[tuple[Any, ...]] = []
        for row_line in lines[idx + 2 :]:
            if "|" not in row_line or not row_line.strip():
                break
            cells = [c.strip() for c in row_line.strip("|").split("|")]
            if len(cells) < len(columns):
                cells.extend([""] * (len(columns) - len(cells)))
            elif len(cells) > len(columns):
                cells = cells[: len(columns)]
            rows.append(tuple(_parse_cell_value(c) for c in cells))
        return columns, rows
    if re.search(r"(?im)^\s*no results\.?\s*$", markdown):
        return [], []
    return None


class _GenieStreamAssembler:
    """Incrementally assembles a complete AgentResponse from Genie Agent Mode SSE events."""

    def __init__(self, *, agent_id: str, agent_title: str | None, question: str) -> None:
        self.agent_id = agent_id
        self.agent_title = agent_title
        self.question = question
        self.conversation_id: str | None = None
        self.response_id: str | None = None
        self.status: str = "in_progress"
        self.error: str | None = None

        self._thoughts_by_id: dict[str, ThoughtStep] = {}
        self._thought_order: list[str] = []
        self._sql_by_call_id: dict[str, SqlCall] = {}
        self._sql_order: list[str] = []
        self._messages_by_id: dict[str, str] = {}
        self._message_order: list[str] = []
        self._timeline: list[TimelineItem] = []
        self._timeline_seen: set[tuple[str, str]] = set()
        self.raw_events: list[dict[str, Any]] = []

    def _record_item(self, item: dict[str, Any], seq: int | None = None) -> None:
        itype = str(item.get("type") or "")
        item_id = str(item.get("id") or "")
        status = str(item.get("status") or "completed")

        if itype == "reasoning":
            text, summaries = _extract_reasoning_text(item)
            if not text and summaries:
                text = "\n".join(summaries)
            if not text:
                return
            key = item_id or f"thought_{len(self._thought_order) + 1}"
            if key not in self._thoughts_by_id:
                self._thought_order.append(key)
                idx = len(self._thought_order)
            else:
                idx = self._thoughts_by_id[key].index
            self._thoughts_by_id[key] = ThoughtStep(
                index=idx,
                item_id=key,
                text=text,
                summary=summaries,
                status=status,
                sequence_number=seq,
            )
            tkey = ("thought", key)
            if tkey not in self._timeline_seen:
                self._timeline_seen.add(tkey)
                self._timeline.append(
                    TimelineItem(
                        index=len(self._timeline) + 1,
                        kind="thought",
                        item_id=key,
                        content=text,
                        status=status,
                        sequence_number=seq,
                    )
                )

        elif itype == "function_call":
            name = str(item.get("name") or "")
            call_id = str(item.get("call_id") or item_id or f"call_{len(self._sql_order) + 1}")
            sql, title, raw_args = _parse_sql_arguments(item.get("arguments"))
            if name == "execute_sql" or sql is not None:
                if call_id not in self._sql_by_call_id:
                    self._sql_order.append(call_id)
                    self._sql_by_call_id[call_id] = SqlCall(
                        index=len(self._sql_order),
                        call_id=call_id,
                        sql=sql or "",
                        title=title,
                        status=status,
                        raw_arguments=raw_args,
                        sequence_number=seq,
                    )
                else:
                    existing = self._sql_by_call_id[call_id]
                    if sql:
                        existing.sql = sql
                    if title:
                        existing.title = title
                    if raw_args:
                        existing.raw_arguments = raw_args
                    existing.status = status
                tkey = ("sql_call", call_id)
                if tkey not in self._timeline_seen:
                    self._timeline_seen.add(tkey)
                    self._timeline.append(
                        TimelineItem(
                            index=len(self._timeline) + 1,
                            kind="sql_call",
                            item_id=call_id,
                            content=sql or raw_args,
                            status=status,
                            sequence_number=seq,
                            metadata={"name": name, "title": title},
                        )
                    )

        elif itype == "function_call_output":
            call_id = str(item.get("call_id") or "")
            output_md = item.get("output")
            output_str = output_md if isinstance(output_md, str) else json.dumps(output_md)
            meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            stmt_id = meta.get("statement_id")
            if call_id and call_id in self._sql_by_call_id:
                sc = self._sql_by_call_id[call_id]
                sc.output_markdown = output_str
                sc.status = status
                if isinstance(stmt_id, str) and stmt_id:
                    sc.statement_id = stmt_id
            tkey = ("sql_output", item_id or call_id)
            if tkey not in self._timeline_seen and status == "completed":
                self._timeline_seen.add(tkey)
                self._timeline.append(
                    TimelineItem(
                        index=len(self._timeline) + 1,
                        kind="sql_output",
                        item_id=item_id or call_id,
                        content=output_str,
                        status=status,
                        sequence_number=seq,
                        metadata={"call_id": call_id, "statement_id": stmt_id},
                    )
                )

        elif itype == "message":
            role = str(item.get("role") or "assistant")
            if role != "assistant":
                return
            text = _extract_message_text(item)
            if not text:
                return
            key = item_id or f"msg_{len(self._message_order) + 1}"
            if key not in self._messages_by_id:
                self._message_order.append(key)
            self._messages_by_id[key] = text
            tkey = ("message", key)
            if tkey not in self._timeline_seen:
                self._timeline_seen.add(tkey)
                self._timeline.append(
                    TimelineItem(
                        index=len(self._timeline) + 1,
                        kind="message",
                        item_id=key,
                        content=text,
                        status=status,
                        sequence_number=seq,
                    )
                )

    def feed(self, evt: dict[str, Any]) -> None:
        """Process one parsed JSON SSE event payload."""
        self.raw_events.append(evt)
        etype = str(evt.get("type") or "")
        seq = evt.get("sequence_number") if isinstance(evt.get("sequence_number"), int) else None

        if etype in ("response.created", "response.in_progress", "response.completed", "response.failed"):
            resp = evt.get("response")
            if isinstance(resp, dict):
                if isinstance(resp.get("id"), str):
                    self.response_id = resp["id"]
                if isinstance(resp.get("conversation_id"), str):
                    self.conversation_id = resp["conversation_id"]
                if isinstance(resp.get("status"), str):
                    self.status = resp["status"]
                if resp.get("error"):
                    self.error = json.dumps(resp["error"])
                for item in resp.get("output") or []:
                    if isinstance(item, dict):
                        self._record_item(item, seq=seq)
            elif etype == "response.failed":
                self.status = "failed"
                self.error = json.dumps(evt.get("error") or evt)

        elif etype == "response.output_item.done":
            item = evt.get("item")
            if isinstance(item, dict):
                self._record_item(item, seq=seq)

        elif etype == "error":
            self.status = "failed"
            self.error = json.dumps(evt.get("error") or evt)

    def build(self, *, elapsed_s: float = 0.0) -> AgentResponse:
        """Finalize and return the assembled AgentResponse."""
        thoughts = [self._thoughts_by_id[k] for k in self._thought_order]
        sql_calls = [self._sql_by_call_id[k] for k in self._sql_order]
        answer_text = "\n\n".join(self._messages_by_id[k] for k in self._message_order).strip()
        return AgentResponse(
            agent_id=self.agent_id,
            agent_title=self.agent_title,
            question=self.question,
            conversation_id=self.conversation_id,
            response_id=self.response_id,
            status=self.status,
            answer_text=answer_text,
            thoughts=thoughts,
            sql_calls=sql_calls,
            timeline=list(self._timeline),
            error=self.error,
            elapsed_s=elapsed_s,
            raw_events=list(self.raw_events),
        )


def parse_sse_lines(
    lines: Iterable[str],
    *,
    agent_id: str,
    agent_title: str | None,
    question: str,
    on_event: EventCallback | None = None,
    elapsed_s: float = 0.0,
) -> AgentResponse:
    """Parse an iterable of SSE text lines into a full AgentResponse."""
    assembler = _GenieStreamAssembler(
        agent_id=agent_id,
        agent_title=agent_title,
        question=question,
    )
    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line or not line.startswith("data:"):
            continue
        data_str = line[len("data:") :].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            evt = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if isinstance(evt, dict):
            assembler.feed(evt)
            if on_event is not None:
                on_event(evt)
    return assembler.build(elapsed_s=elapsed_s)


class GenieAgentClient:
    """HTTP/SSE client for Databricks Genie Spaces discovery and Agent Mode execution."""

    def __init__(
        self,
        host: str,
        auth: DatabricksAuthProvider,
        *,
        timeout_s: float = 180.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.auth = auth
        self.timeout_s = timeout_s
        self._http = http_client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30.0))
        self._owns_http = http_client is None

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "GenieAgentClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.host}{path}"
        for attempt in range(2):
            headers = {
                **self.auth.auth_headers(force_refresh=(attempt > 0)),
                "Accept": "application/json",
            }
            resp = self._http.request(method, url, headers=headers, params=params, json=json_body)
            if resp.status_code == 401 and attempt == 0:
                continue
            if resp.status_code >= 400:
                raise GenieApiError(resp.status_code, resp.text[:300], body=resp.text)
            return resp.json() if resp.text.strip() else {}
        raise GenieApiError(401, "Authentication failed after token refresh.")

    def get_agent(self, agent_id: str, *, include_serialized_space: bool = False) -> GenieAgent:
        """Fetch live metadata (title, description, and optional table fallback) for one Genie agent."""
        params = {"include_serialized_space": "true"} if include_serialized_space else None
        raw = self._request_json("GET", f"/api/2.0/genie/spaces/{agent_id}", params=params)
        desc = str(raw.get("description") or "").strip()
        if not desc and include_serialized_space:
            desc = _summarize_serialized_space(raw.get("serialized_space"))
        return GenieAgent(
            agent_id=str(raw.get("space_id") or agent_id),
            title=str(raw.get("title") or agent_id),
            description=desc,
            warehouse_id=raw.get("warehouse_id"),
            create_time=raw.get("create_time"),
            update_time=raw.get("update_time"),
        )

    def list_agents(
        self,
        *,
        allowed_agent_ids: Sequence[str] | None = None,
        enrich_empty_descriptions: bool = True,
    ) -> list[GenieAgent]:
        """Discover Genie agents and pull their live descriptions from Databricks.

        When `allowed_agent_ids` is provided, only those IDs are returned (in config order).
        For any agent whose workspace `description` is blank, fetches `include_serialized_space=true`
        to populate a table-list description so the router knows its domain.
        """
        discovered_by_id: dict[str, GenieAgent] = {}
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {}
            if page_token:
                params["page_token"] = page_token
            data = self._request_json("GET", "/api/2.0/genie/spaces", params=params or None)
            for raw in data.get("spaces") or []:
                if not isinstance(raw, dict) or not raw.get("space_id"):
                    continue
                aid = str(raw["space_id"])
                discovered_by_id[aid] = GenieAgent(
                    agent_id=aid,
                    title=str(raw.get("title") or aid),
                    description=str(raw.get("description") or "").strip(),
                    warehouse_id=raw.get("warehouse_id"),
                    create_time=raw.get("create_time"),
                    update_time=raw.get("update_time"),
                )
            page_token = data.get("next_page_token")
            if not page_token:
                break

        if allowed_agent_ids:
            selected: list[GenieAgent] = []
            for aid in allowed_agent_ids:
                ag = discovered_by_id.get(aid)
                if ag is None:
                    # Fetch directly in case the space wasn't returned by list_spaces
                    ag = self.get_agent(aid, include_serialized_space=enrich_empty_descriptions)
                elif not ag.description and enrich_empty_descriptions:
                    ag = self.get_agent(aid, include_serialized_space=True)
                selected.append(ag)
            return selected

        out = list(discovered_by_id.values())
        if enrich_empty_descriptions:
            enriched: list[GenieAgent] = []
            for ag in out:
                if not ag.description:
                    try:
                        ag = self.get_agent(ag.agent_id, include_serialized_space=True)
                    except GenieApiError:
                        pass
                enriched.append(ag)
            return enriched
        return out

    def ask_agent(
        self,
        agent_id: str,
        question: str,
        *,
        agent_title: str | None = None,
        conversation_id: str | None = None,
        execution_mode: str = "STANDARD",
        enable_viz: bool = False,
        on_event: EventCallback | None = None,
    ) -> AgentResponse:
        """Invoke a Genie Agent in Agent Mode via SSE streaming and capture all outputs."""
        url = f"{self.host}/api/2.0/genie/agents/{agent_id}/responses"
        payload: dict[str, Any] = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": question}],
                }
            ],
            "execution_mode": execution_mode.upper(),
            "enable_viz": enable_viz,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        t0 = time.monotonic()
        assembler = _GenieStreamAssembler(
            agent_id=agent_id,
            agent_title=agent_title,
            question=question,
        )

        for attempt in range(2):
            headers = {
                **self.auth.auth_headers(force_refresh=(attempt > 0)),
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
            with self._http.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code == 401 and attempt == 0:
                    continue
                if resp.status_code >= 400:
                    body = resp.read().decode("utf-8", errors="replace")
                    raise GenieApiError(resp.status_code, body[:300], body=body)

                try:
                    for raw_line in resp.iter_lines():
                        line = raw_line.rstrip("\r\n")
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:") :].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            evt = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(evt, dict):
                            assembler.feed(evt)
                            if on_event is not None:
                                on_event(evt)
                except KeyboardInterrupt:
                    if assembler.conversation_id and assembler.response_id:
                        try:
                            self.cancel_response(
                                agent_id,
                                assembler.conversation_id,
                                assembler.response_id,
                            )
                        except Exception:
                            pass
                    raise
                return assembler.build(elapsed_s=time.monotonic() - t0)

        raise GenieApiError(401, "Authentication failed after token refresh.")

    def cancel_response(self, agent_id: str, conversation_id: str, response_id: str) -> dict[str, Any]:
        """Cancel an in-flight Genie Agent Mode response."""
        return self._request_json(
            "POST",
            f"/api/2.0/genie/agents/{agent_id}/conversations/{conversation_id}/responses/{response_id}/cancel",
        )

    def get_conversation_items(self, agent_id: str, conversation_id: str) -> dict[str, Any]:
        """Fetch chronological conversation items for an Agent Mode conversation."""
        return self._request_json(
            "GET",
            f"/api/2.0/genie/agents/{agent_id}/conversations/{conversation_id}/items",
        )
