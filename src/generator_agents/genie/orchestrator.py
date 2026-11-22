"""Genie Agent Orchestrator coordinating configured allowlist, live description pull, decision routing, and Agent Mode calls."""

from __future__ import annotations

from typing import Any

from generator_agents.genie.auth import DatabricksAuthProvider
from generator_agents.genie.client import EventCallback, GenieAgentClient
from generator_agents.genie.config import OrchestratorConfig
from generator_agents.genie.models import AgentResponse, GenieAgent, RouteDecision
from generator_agents.genie.router import GeminiDecisionRouter


class GenieOrchestrator:
    """Routes user questions via `gemini-decision-1` and invokes Databricks Genie in Agent Mode."""

    def __init__(
        self,
        *,
        config: OrchestratorConfig | None = None,
        client: GenieAgentClient | None = None,
        router: GeminiDecisionRouter | None = None,
        agents: list[GenieAgent] | None = None,
    ) -> None:
        self.config = config or OrchestratorConfig.from_env()
        if client is not None:
            self.client = client
            self._owns_client = False
        else:
            auth = DatabricksAuthProvider(
                self.config.databricks_host,
                profile=self.config.databricks_profile,
                static_token=self.config.databricks_token,
            )
            self.client = GenieAgentClient(
                self.config.databricks_host,
                auth,
                timeout_s=self.config.request_timeout_s,
            )
            self._owns_client = True

        self._router = router
        self._agents: list[GenieAgent] | None = list(agents) if agents is not None else None
        self._conversations: dict[str, str] = {}
        self._last_agent_id: str | None = None
        self._last_question: str | None = None
        self._last_answer: str | None = None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "GenieOrchestrator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def router(self) -> GeminiDecisionRouter:
        if self._router is None:
            self._router = GeminiDecisionRouter(
                api_key=self.config.gemini_api_key,
                model=self.config.router_model,
                min_confidence=self.config.min_route_confidence,
                followup_threshold=self.config.followup_threshold,
            )
        return self._router

    def list_agents(self, *, refresh: bool = False) -> list[GenieAgent]:
        """Return configured Genie agents with live titles & descriptions pulled from Databricks."""
        if self._agents is None or refresh:
            allowed = self.config.allowed_agent_ids or None
            self._agents = self.client.list_agents(
                allowed_agent_ids=allowed,
                enrich_empty_descriptions=True,
            )
        return list(self._agents)

    def reset_conversations(self, agent_id: str | None = None) -> None:
        """Clear saved conversation state for one agent or all agents."""
        if agent_id is None:
            self._conversations.clear()
            self._last_agent_id = None
            self._last_question = None
            self._last_answer = None
        else:
            self._conversations.pop(agent_id, None)
            if self._last_agent_id == agent_id:
                self._last_agent_id = None
                self._last_question = None
                self._last_answer = None

    def route_question(
        self,
        question: str,
        *,
        hint: str | None = None,
        use_session_context: bool = True,
        allow_none: bool = True,
    ) -> RouteDecision:
        """Route a question using the Gemini Decision Model without executing Genie yet."""
        agents = self.list_agents()
        agent_by_id = {a.agent_id: a for a in agents}
        active_agent = (
            agent_by_id.get(self._last_agent_id)
            if (use_session_context and self._last_agent_id)
            else None
        )
        return self.router.route(
            question,
            agents,
            hint=hint,
            active_agent=active_agent,
            last_question=self._last_question if active_agent is not None else None,
            last_answer=self._last_answer if active_agent is not None else None,
            allow_none=allow_none,
        )

    def ask(
        self,
        question: str,
        *,
        hint: str | None = None,
        agent_id: str | None = None,
        new_conversation: bool = False,
        execution_mode: str | None = None,
        enable_viz: bool | None = None,
        allow_none: bool = True,
        on_event: EventCallback | None = None,
    ) -> tuple[RouteDecision, AgentResponse | None]:
        """Route a question (unless pinned) and invoke the chosen Genie agent in Agent Mode."""
        agents = self.list_agents()
        agent_by_id = {a.agent_id: a for a in agents}

        if agent_id is not None:
            chosen = agent_by_id.get(agent_id)
            decision = RouteDecision(
                agent_id=agent_id,
                agent_title=chosen.title if chosen is not None else agent_id,
                confidence=1.0,
                probabilities={agent_id: 1.0},
                is_followup=(not new_conversation and agent_id in self._conversations),
                needs_clarification=False,
                reason="Agent explicitly pinned by caller.",
                router_model="pinned",
                pinned=True,
            )
        else:
            decision = self.route_question(
                question,
                hint=hint,
                use_session_context=not new_conversation,
                allow_none=allow_none,
            )

        if decision.needs_clarification or not decision.agent_id:
            return decision, None

        target_id = decision.agent_id
        conv_id = None if new_conversation else self._conversations.get(target_id)
        if not decision.pinned and not decision.is_followup:
            conv_id = None

        prompt_text = question if not hint else f"{question}\n\nHint: {hint}"
        resp = self.client.ask_agent(
            target_id,
            prompt_text,
            agent_title=decision.agent_title,
            conversation_id=conv_id,
            execution_mode=execution_mode or self.config.execution_mode,
            enable_viz=self.config.enable_viz if enable_viz is None else enable_viz,
            on_event=on_event,
        )
        resp.routing = decision

        if resp.conversation_id:
            self._conversations[target_id] = resp.conversation_id
        self._last_agent_id = target_id
        self._last_question = question
        self._last_answer = resp.answer_text
        return decision, resp
