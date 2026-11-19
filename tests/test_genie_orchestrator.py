"""Offline unit tests for `generator_agents.genie` (allowlist config, live descriptions, router, SSE, S5 generator)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from generator_agents.genie.auth import DatabricksAuthProvider
from generator_agents.genie.client import GenieAgentClient, parse_markdown_table, parse_sse_lines
from generator_agents.genie.config import OrchestratorConfig
from generator_agents.genie.generator import _strip_databricks_citations, generate_genie_items
from generator_agents.genie.models import AgentResponse, GenieAgent
from generator_agents.genie.orchestrator import GenieOrchestrator
from generator_agents.genie.router import GeminiDecisionRouter, NONE_OPTION
from txt2sql_eval_harness.alignment import align_tables
from txt2sql_eval_harness.judge import prepare_best_candidate
from txt2sql_eval_harness.models import (
    Claim,
    Contract,
    Derivation,
    EvalConfig,
    GroundTruth,
    LabelledItem,
    ResultTable,
    Step,
    Trace,
)
from txt2sql_eval_harness.packet import PreparedCandidate, build_packet


def _sample_sse_lines() -> list[str]:
    events = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": "resp_001",
                "status": "in_progress",
                "conversation_id": "conv_abc",
                "output": [],
            },
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "item": {
                "type": "reasoning",
                "id": "thought_1",
                "status": "completed",
                "content": [{"type": "reasoning_text", "text": "First I will check top products."}],
                "summary": [],
            },
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 2,
            "item": {
                "type": "function_call",
                "id": "call_1",
                "call_id": "call_1",
                "status": "completed",
                "name": "execute_sql",
                "arguments": json.dumps(
                    {
                        "title": "Top Products",
                        "sql": "SELECT product, SUM(totalPrice) AS total_sales FROM sales GROUP BY product ORDER BY total_sales DESC LIMIT 2",
                    }
                ),
            },
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 3,
            "item": {
                "type": "function_call_output",
                "id": "call_1_out",
                "call_id": "call_1",
                "status": "completed",
                "output": "**Top Products**\n\n| product | total_sales |\n| --- | --- |\n| Golden Gate Ginger | 11595 |\n| Outback Oatmeal | 11199 |",
                "metadata": {"statement_id": "stmt_1"},
            },
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 4,
            "item": {
                "type": "reasoning",
                "id": "thought_2",
                "status": "completed",
                "content": [{"type": "reasoning_text", "text": "Now I will verify total revenue."}],
                "summary": [],
            },
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 5,
            "item": {
                "type": "function_call",
                "id": "call_2",
                "call_id": "call_2",
                "status": "completed",
                "name": "execute_sql",
                "arguments": json.dumps(
                    {
                        "title": "Total Revenue",
                        "sql": "SELECT SUM(totalPrice) AS grand_total FROM sales",
                    }
                ),
            },
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 6,
            "item": {
                "type": "function_call_output",
                "id": "call_2_out",
                "call_id": "call_2",
                "status": "completed",
                "output": "| grand_total |\n| --- |\n| 50000.5 |",
                "metadata": {"statement_id": "stmt_2"},
            },
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 7,
            "item": {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The top product is Golden Gate Ginger ($11,595) out of $50,000.50 total.",
                    }
                ],
            },
        },
        {
            "type": "response.completed",
            "sequence_number": 8,
            "response": {
                "id": "resp_001",
                "status": "completed",
                "conversation_id": "conv_abc",
                "output": [],
            },
        },
    ]
    lines: list[str] = []
    for ev in events:
        lines.append(f"event:{ev['type']}")
        lines.append(f"data:{json.dumps(ev)}")
        lines.append("")
    return lines


def test_parse_sse_lines_captures_thoughts_sqls_outputs_and_answer() -> None:
    seen: list[str] = []
    resp = parse_sse_lines(
        _sample_sse_lines(),
        agent_id="agent_bakehouse",
        agent_title="Bakehouse Sales",
        question="What are the top products?",
        on_event=lambda e: seen.append(e["type"]),
    )
    assert resp.status == "completed"
    assert resp.conversation_id == "conv_abc"
    assert resp.response_id == "resp_001"
    assert len(resp.thoughts) == 2
    assert [t.text for t in resp.thoughts] == [
        "First I will check top products.",
        "Now I will verify total revenue.",
    ]
    assert len(resp.sql_calls) == 2
    assert resp.sql_calls[0].title == "Top Products"
    assert resp.sql_calls[0].statement_id == "stmt_1"
    assert "Golden Gate Ginger" in (resp.sql_calls[0].output_markdown or "")
    assert resp.sql_calls[1].statement_id == "stmt_2"
    assert len(resp.sql_queries) == 2
    assert "Golden Gate Ginger" in resp.answer_text
    assert [ti.kind for ti in resp.timeline] == [
        "thought",
        "sql_call",
        "sql_output",
        "thought",
        "sql_call",
        "sql_output",
        "message",
    ]
    assert len(seen) == 9


def test_parse_markdown_table() -> None:
    md = "**Results**\n\n| name | count | ratio | note |\n| --- | --- | --- | --- |\n| Alice | 42 | 0.75 | NULL |"
    parsed = parse_markdown_table(md)
    assert parsed is not None
    cols, rows = parsed
    assert cols == ["name", "count", "ratio", "note"]
    assert rows == [("Alice", 42, 0.75, None)]


def test_agents_yaml_allowlist_and_live_description_enrichment(tmp_path: Path) -> None:
    cfg_file = tmp_path / "agents.yaml"
    cfg_file.write_text(
        "databricks_host: https://example.cloud.databricks.com\n"
        "agents:\n"
        "  - id_bake\n"
        "  - id_f1\n",
        encoding="utf-8",
    )
    cfg = OrchestratorConfig.from_env(agents_config_path=cfg_file, databricks_token="dapi_test")
    assert cfg.allowed_agent_ids == ("id_bake", "id_f1")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/2.0/genie/spaces":
            return httpx.Response(
                200,
                json={
                    "spaces": [
                        {"space_id": "id_bake", "title": "Bakehouse", "description": "Bakery sales data."},
                        {"space_id": "id_excluded", "title": "Excluded", "description": "Should not appear."},
                        {"space_id": "id_f1", "title": "Text2SQL - formula_1", "description": ""},
                    ]
                },
            )
        if request.url.path == "/api/2.0/genie/spaces/id_f1":
            return httpx.Response(
                200,
                json={
                    "space_id": "id_f1",
                    "title": "Text2SQL - formula_1",
                    "description": "",
                    "serialized_space": json.dumps(
                        {
                            "data_sources": {
                                "tables": [
                                    {"identifier": "cat.gold.f1_drivers"},
                                    {"identifier": "cat.gold.f1_races"},
                                ]
                            }
                        }
                    ),
                },
            )
        return httpx.Response(404)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    auth = DatabricksAuthProvider(cfg.databricks_host, static_token="dapi_test")
    client = GenieAgentClient(cfg.databricks_host, auth, http_client=http_client)
    agents = client.list_agents(allowed_agent_ids=cfg.allowed_agent_ids, enrich_empty_descriptions=True)

    assert [a.agent_id for a in agents] == ["id_bake", "id_f1"]
    assert agents[0].description == "Bakery sales data."
    assert "f1_drivers" in agents[1].description
    assert "f1_races" in agents[1].description


def test_auth_provider_caches_and_refreshes() -> None:
    calls: list[list[str]] = []

    def fake_cli(cmd: list[str]) -> str:
        calls.append(cmd)
        return json.dumps({"access_token": f"tok_{len(calls)}", "expiry": "2099-01-01T00:00:00Z"})

    auth = DatabricksAuthProvider(
        "https://example.cloud.databricks.com",
        profile="DEFAULT",
        cli_runner=fake_cli,
    )
    assert auth.get_token() == "tok_1"
    assert auth.get_token() == "tok_1"
    assert len(calls) == 1
    assert auth.get_token(force_refresh=True) == "tok_2"
    assert len(calls) == 2

    static_auth = DatabricksAuthProvider("https://example.cloud.databricks.com", static_token="dapi_xyz")
    assert static_auth.get_token() == "dapi_xyz"


class _FakeDecisionTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def decide(self, *, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"state": state, "questions": questions})
        return self._responses.pop(0)


def test_gemini_decision_router_routes_and_detects_followups() -> None:
    agents = [
        GenieAgent(agent_id="a_bake", title="Bakehouse", description="Bakery sales data"),
        GenieAgent(agent_id="a_hero", title="Superhero", description="Superhero powers and attributes"),
    ]
    transport = _FakeDecisionTransport(
        [
            {"route": {"probabilities": {"a_bake": 0.94, "a_hero": 0.04, "none": 0.02}}},
            {
                "route": {"probabilities": {"a_bake": 0.88, "a_hero": 0.06, "none": 0.06}},
                "is_followup": {"noul": 0.91},
            },
            {"route": {"probabilities": {"none": 0.82, "a_bake": 0.10, "a_hero": 0.08}}},
        ]
    )
    router = GeminiDecisionRouter(transport=transport, min_confidence=0.35, followup_threshold=0.65)

    d1 = router.route("What are the top selling cookies?", agents)
    assert d1.agent_id == "a_bake"
    assert d1.confidence == 0.94
    assert not d1.needs_clarification
    assert "is_followup" not in transport.calls[0]["questions"]

    d2 = router.route(
        "Now break that down by franchise",
        agents,
        active_agent=agents[0],
        last_question="What are the top selling cookies?",
        last_answer="Golden Gate Ginger",
    )
    assert d2.agent_id == "a_bake"
    assert d2.is_followup is True
    assert d2.is_followup_prob == 0.91
    assert "is_followup" in transport.calls[1]["questions"]

    d3 = router.route("What is the weather in Tokyo?", agents)
    assert d3.agent_id is None
    assert d3.needs_clarification is True


class _FakeGenieClient:
    def __init__(self, agents: list[GenieAgent]) -> None:
        self._agents = agents
        self.calls: list[dict[str, Any]] = []

    def list_agents(self, **_: Any) -> list[GenieAgent]:
        return list(self._agents)

    def ask_agent(
        self,
        agent_id: str,
        question: str,
        *,
        agent_title: str | None = None,
        conversation_id: str | None = None,
        execution_mode: str = "STANDARD",
        enable_viz: bool = False,
        on_event: Any = None,
    ) -> AgentResponse:
        self.calls.append(
            {
                "agent_id": agent_id,
                "question": question,
                "conversation_id": conversation_id,
                "execution_mode": execution_mode,
            }
        )
        return parse_sse_lines(
            _sample_sse_lines(),
            agent_id=agent_id,
            agent_title=agent_title,
            question=question,
            on_event=on_event,
        )

    def close(self) -> None:
        pass


def test_orchestrator_reuses_conversation_on_followup_and_generates_s5(tmp_path: Path) -> None:
    agents = [
        GenieAgent(agent_id="a_bake", title="Bakehouse", description="Bakery sales"),
        GenieAgent(agent_id="a_hero", title="Superhero", description="Superheroes"),
    ]
    fake_client = _FakeGenieClient(agents)
    transport = _FakeDecisionTransport(
        [
            {"route": {"probabilities": {"a_bake": 0.95, "a_hero": 0.03, "none": 0.02}}},
            {
                "route": {"probabilities": {"a_bake": 0.90, "a_hero": 0.05, "none": 0.05}},
                "is_followup": {"noul": 0.88},
            },
            {"route": {"probabilities": {"a_bake": 0.93, "a_hero": 0.05, "none": 0.02}}},
        ]
    )
    router = GeminiDecisionRouter(transport=transport)
    orch = GenieOrchestrator(
        config=OrchestratorConfig(),
        client=fake_client,  # type: ignore[arg-type]
        router=router,
    )

    d1, r1 = orch.ask("Top products?")
    assert d1.agent_id == "a_bake"
    assert r1 is not None and r1.conversation_id == "conv_abc"
    assert fake_client.calls[0]["conversation_id"] is None

    d2, r2 = orch.ask("Break that down by store")
    assert d2.is_followup is True
    assert r2 is not None
    assert fake_client.calls[1]["conversation_id"] == "conv_abc"

    base_item = LabelledItem(
        item_id="base_1",
        group_id="g1",
        split="test",
        set="S1",
        family="bird",
        slices=[],
        trace=Trace(
            trace_id="base_1",
            group_id="g1",
            question="Top products?",
            hint=None,
            db_id="bakehouse",
            db_sha256="0" * 64,
            steps=[Step(index=1, sql="SELECT 1")],
            final_answer=None,
        ),
        gt=GroundTruth(item_ref="gt_1", sql_chain=["SELECT 1"]),
        label="unlabelled",
        snapshot="snap",
    )
    out_jsonl = tmp_path / "genie_s5.jsonl"
    items, responses, stats = generate_genie_items(
        [base_item],
        orch,
        limit=1,
        out_path=out_jsonl,
    )
    assert stats["generated"] == 1
    assert stats["total_steps"] == 2
    assert stats["total_thoughts"] == 2
    assert len(items) == 1
    assert len(items[0].trace.steps) == 2
    assert items[0].trace.steps[0].result is not None
    assert items[0].trace.steps[0].result.columns == ["product", "total_sales"]
    assert len(responses) == 1

    rich_jsonl = out_jsonl.with_suffix(".genie_responses.jsonl")
    assert out_jsonl.exists()
    assert rich_jsonl.exists()
    rich_rec = json.loads(rich_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert len(rich_rec["thoughts"]) == 2
    assert len(rich_rec["sql_calls"]) == 2
    assert rich_rec["routing"]["agent_id"] == "a_bake"


def test_parse_markdown_table_no_results_and_commas() -> None:
    assert parse_markdown_table("**Query Title**\n\nNo results") == ([], [])
    assert parse_markdown_table("**Query Title**\n\nNo results.") == ([], [])

    md = "| label | count | amount |\n| --- | --- | --- |\n| Single | 1,408 | 12,345.67 |"
    parsed = parse_markdown_table(md)
    assert parsed == (["label", "count", "amount"], [("Single", 1408, 12345.67)])


def test_strip_databricks_citations_and_preserves_rich_answer(tmp_path: Path) -> None:
    raw = (
        "The answer is **42**."
        "\\[[1](https://dbc-89003fab-33b9.cloud.databricks.com/genie/rooms/abc/chats/def?o=42481210739289)\\]"
    )
    cleaned = _strip_databricks_citations(raw)
    assert "42481210739289" not in cleaned
    assert "https://" not in cleaned
    assert "**42**" in cleaned

    class _CitationClient(_FakeGenieClient):
        def ask_agent(self, agent_id: str, question: str, **kwargs: Any) -> AgentResponse:
            resp = super().ask_agent(agent_id, question, **kwargs)
            resp.answer_text = raw
            return resp

    agents = [GenieAgent(agent_id="a_bake", title="Bakehouse", description="Bakery sales")]
    transport = _FakeDecisionTransport([{"route": {"probabilities": {"a_bake": 0.99}}}])
    orch = GenieOrchestrator(
        config=OrchestratorConfig(),
        client=_CitationClient(agents),  # type: ignore[arg-type]
        router=GeminiDecisionRouter(transport=transport),
    )
    base_item = LabelledItem(
        item_id="cite_1",
        group_id="g1",
        split="test",
        set="S1",
        family="bird",
        slices=[],
        trace=Trace("cite_1", "g1", "Count?", None, "bakehouse", "0" * 64, [Step(1, "SELECT 1")]),
        gt=GroundTruth("gt_1", ["SELECT 1"]),
        label="unlabelled",
        snapshot="snap",
    )
    items, responses, _ = generate_genie_items([base_item], orch, limit=1)
    assert items[0].trace.final_answer == "The answer is **42**."
    assert "42481210739289" in responses[0].answer_text


def test_router_allow_none_false_omits_none_and_forces_best_agent() -> None:
    agents = [
        GenieAgent(agent_id="a_club", title="student_club", description="Student club budgets"),
        GenieAgent(agent_id="a_fin", title="financial", description="Bank accounts"),
    ]
    transport = _FakeDecisionTransport(
        [{"route": {"probabilities": {NONE_OPTION: 0.65, "a_club": 0.25, "a_fin": 0.10}}}]
    )
    router = GeminiDecisionRouter(transport=transport, min_confidence=0.35)
    decision = router.route("What were the fundraising notes?", agents, allow_none=False)
    assert NONE_OPTION not in transport.calls[0]["questions"]["route"]["criteria"]
    assert decision.agent_id == "a_club"
    assert decision.confidence == 0.25
    assert decision.needs_clarification is False


def test_judge_all_complete_allows_recovery_and_volatile_gt() -> None:
    cfg = EvalConfig()
    gt = GroundTruth("g_rec", ["SELECT COUNT(*) FROM t WHERE date('now') > '2020-01-01'"])
    tr = Trace(
        "t_rec",
        "g_rec",
        "How many rows?",
        None,
        "db",
        "0" * 64,
        [
            Step(1, "SELECT bad_col FROM t", result=None, error_kind="runtime"),
            Step(2, "SELECT COUNT(*) FROM t", result=ResultTable(["cnt"], [(5,)], 1, True)),
        ],
        final_answer="There are 5 rows.",
    )
    cand, overflow = prepare_best_candidate(tr, gt, None, cfg, cfg.config_hash(""), "H+E")
    assert not overflow and cand is not None
    assert cand.all_complete is True


def test_full_trace_packet_compacts_extra_claim_bindings() -> None:
    cfg = EvalConfig(escalation_max_cells=1200, escalation_max_tokens=8000)
    gt_tbl = ResultTable(["v"], [(1,)], 1, True)
    step_tbl = ResultTable(["v"], [(1,)], 1, True)
    al, rd, _ = align_tables(gt_tbl, step_tbl, Contract([0], False, False))
    many_cells = [(0, 0) for _ in range(30)]
    many_agent_cells = [(1, 0, 0) for _ in range(30)]
    many_derivs = [Derivation("agent", 1, "COUNT", "v", 1) for _ in range(10)]
    claims = [
        Claim(
            text=f"val_{i}",
            kind="number",
            value=i,
            decimals=0,
            in_question=False,
            gt_cells=many_cells,
            agent_cells=many_agent_cells,
            derivations=many_derivs,
        )
        for i in range(60)
    ]
    cand = PreparedCandidate(
        trace=Trace("t_big_claims", "g", "Q?", None, "d", "0", [Step(1, "SELECT 1", result=step_tbl)], final_answer="Answer."),
        gt=GroundTruth("t_big_claims", ["SELECT 1"], results=[gt_tbl]),
        gt_sql="SELECT 1",
        gt_sql_n="SELECT 1",
        gt_result=gt_tbl,
        gt_volatile=False,
        gt_det=True,
        gt_rd_eligible=True,
        contract=Contract([0], False, False),
        schema_ddl="",
        steps=[Step(1, "SELECT 1", result=step_tbl)],
        step_sql_n=["SELECT 1"],
        step_volatile=[False],
        step_det=[True],
        step_rd_eligible=[True],
        alignment=al,
        row_diff=rd,
        claims=claims,
        derivations=[],
    )
    pkt = build_packet(cand, "full_trace", "escalation", cfg)
    assert "extra_claim_bindings" in pkt.truncated
    assert pkt.required_met is True
    assert pkt.n_tokens <= cfg.escalation_max_tokens
