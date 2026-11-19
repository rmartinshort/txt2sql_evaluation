"""CLI and interactive REPL for the Databricks Genie Agent Orchestrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from generator_agents.genie.config import OrchestratorConfig
from generator_agents.genie.generator import generate_genie_items
from generator_agents.genie.models import AgentResponse, RouteDecision
from generator_agents.genie.orchestrator import GenieOrchestrator


def _make_live_printer(*, show_thoughts: bool = True, show_sql: bool = True):
    """Build an SSE callback that prints live Genie thoughts and SQL calls to stderr."""

    def _on_event(evt: dict[str, Any]) -> None:
        if evt.get("type") != "response.output_item.done":
            return
        item = evt.get("item")
        if not isinstance(item, dict):
            return
        itype = item.get("type")
        if itype == "reasoning" and show_thoughts:
            texts = [
                b.get("text", "").strip()
                for b in (item.get("content") or [])
                if isinstance(b, dict) and b.get("text")
            ]
            if texts:
                print(f"[thought] {' '.join(texts)}", file=sys.stderr, flush=True)
        elif itype == "function_call" and show_sql:
            raw_args = item.get("arguments")
            title = ""
            sql = ""
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                    title = parsed.get("title") or ""
                    sql = parsed.get("sql") or ""
                except json.JSONDecodeError:
                    sql = raw_args
            hdr = f"[sql: {title}]" if title else "[sql]"
            print(f"{hdr}\n{sql.strip()}", file=sys.stderr, flush=True)

    return _on_event


def _print_full_response(
    decision: RouteDecision,
    resp: AgentResponse | None,
    *,
    show_thoughts: bool = True,
    show_sql: bool = True,
) -> None:
    """Print routing decision, captured thoughts, SQL queries, and final text answer."""
    route_label = decision.agent_title or decision.agent_id or "NONE"
    print(f"\n=== Route: {route_label} (p={decision.confidence:.2f}, model={decision.router_model}) ===")
    if decision.reason:
        print(f"Reason: {decision.reason}")
    if decision.is_followup_prob is not None:
        print(f"Follow-up: {decision.is_followup} (p={decision.is_followup_prob:.2f})")

    if resp is None:
        print("\nNo Genie agent was invoked (low confidence or no matching domain).")
        return

    if show_thoughts and resp.thoughts:
        print(f"\n=== Intermediate Thoughts ({len(resp.thoughts)}) ===")
        for th in resp.thoughts:
            print(f"{th.index}. {th.text}")

    if show_sql and resp.sql_calls:
        print(f"\n=== Executed SQL Queries ({len(resp.sql_calls)}) ===")
        for sc in resp.sql_calls:
            title = f" — {sc.title}" if sc.title else ""
            print(f"\n-- Query {sc.index}{title} (status={sc.status}) --")
            print(sc.sql)
            if sc.output_markdown:
                print(f"\nResult:\n{sc.output_markdown}")

    print(f"\n=== Final Answer ({resp.elapsed_s:.1f}s, status={resp.status}) ===")
    print(resp.answer_text or "(empty answer)")
    if resp.error:
        print(f"\nError: {resp.error}", file=sys.stderr)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agents-config",
        type=Path,
        default=None,
        help="Path to Genie agents YAML config (default: src/generator_agents/genie/agents.yaml)",
    )
    parser.add_argument("--host", default=None, help="Databricks workspace URL")
    parser.add_argument("--profile", default=None, help="~/.databrickscfg profile name")
    parser.add_argument("--router-model", default=None, help="Gemini Decision model (default: gemini-decision-1)")
    parser.add_argument("--execution-mode", default=None, help="Genie Agent Mode (default: STANDARD)")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Minimum routing probability")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genie-orchestrator",
        description="Databricks Genie Agent Mode orchestrator routed by Gemini Decision Model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_agents = sub.add_parser("agents", help="List configured Genie agents with live descriptions")
    _add_common_args(p_agents)
    p_agents.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON output")

    p_ask = sub.add_parser("ask", help="Route a single question and invoke Genie in Agent Mode")
    _add_common_args(p_ask)
    p_ask.add_argument("question", help="Natural-language question to ask")
    p_ask.add_argument("--hint", default=None, help="Optional hint / domain clarification")
    p_ask.add_argument("--agent", default=None, dest="agent_id", help="Pin a specific Genie agent_id")
    p_ask.add_argument("--json", action="store_true", dest="as_json", help="Emit full JSON record")
    p_ask.add_argument("--include-raw-events", action="store_true", help="Include raw SSE events in JSON")
    p_ask.add_argument("--quiet-stream", action="store_true", help="Disable live stderr progress stream")
    p_ask.add_argument("--hide-thoughts", action="store_true", help="Omit thoughts from summary output")
    p_ask.add_argument("--hide-sql", action="store_true", help="Omit SQL queries from summary output")

    p_chat = sub.add_parser("chat", help="Interactive multi-turn Genie orchestrator REPL")
    _add_common_args(p_chat)
    p_chat.add_argument("--agent", default=None, dest="agent_id", help="Optionally pin an agent_id")

    p_gen = sub.add_parser("generate", help="Run Genie Agent Mode over benchmark items (S5)")
    _add_common_args(p_gen)
    p_gen.add_argument("--items", type=Path, default=None, help="Input benchmark LabelledItem JSONL")
    p_gen.add_argument(
        "--dataset",
        choices=["bird", "flex"],
        default="bird",
        help="Built-in dataset to load when --items is omitted (default: bird)",
    )
    p_gen.add_argument("--data-dir", type=Path, default=Path("data"), help="Data dir containing raw/ and dbs/")
    p_gen.add_argument("--db-dir", type=Path, default=None, help="Optional SQLite DB dir for GT execution")
    p_gen.add_argument("--out", type=Path, default=Path("data/genie_items.jsonl"), help="Output S5 JSONL")
    p_gen.add_argument("--rich-out", type=Path, default=None, help="Output rich Genie responses JSONL")
    p_gen.add_argument("--limit", type=int, default=20)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--split", choices=["all", "calibration", "test"], default="test")
    p_gen.add_argument("--db-id", default=None)
    p_gen.add_argument("--agent", default=None, dest="agent_id", help="Pin a single Genie agent_id")
    p_gen.add_argument("--sql-only", action="store_true")
    p_gen.add_argument("--concurrency", type=int, default=1)
    p_gen.add_argument("--verbose", action="store_true")

    return parser


def _make_orchestrator(args: argparse.Namespace) -> GenieOrchestrator:
    cfg = OrchestratorConfig.from_env(
        agents_config_path=args.agents_config,
        databricks_host=args.host,
        databricks_profile=args.profile,
        router_model=args.router_model,
        execution_mode=args.execution_mode,
        min_route_confidence=args.min_confidence,
    )
    return GenieOrchestrator(config=cfg)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    with _make_orchestrator(args) as orch:
        if args.command == "agents":
            agents = orch.list_agents()
            if args.as_json:
                print(json.dumps([a.to_dict() for a in agents], indent=2))
            else:
                for idx, ag in enumerate(agents, start=1):
                    desc = f" — {ag.description}" if ag.description else ""
                    print(f"{idx:2d}. {ag.agent_id}  {ag.title}{desc}")
            return 0

        if args.command == "ask":
            on_event = None if (args.as_json or args.quiet_stream) else _make_live_printer(
                show_thoughts=not args.hide_thoughts,
                show_sql=not args.hide_sql,
            )
            decision, resp = orch.ask(
                args.question,
                hint=args.hint,
                agent_id=args.agent_id,
                new_conversation=True,
                on_event=on_event,
            )
            if args.as_json:
                payload = {
                    "routing": decision.to_dict(),
                    "response": (
                        resp.to_dict(include_raw_events=args.include_raw_events)
                        if resp is not None
                        else None
                    ),
                }
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                _print_full_response(
                    decision,
                    resp,
                    show_thoughts=not args.hide_thoughts,
                    show_sql=not args.hide_sql,
                )
            return 0 if resp is not None and resp.status == "completed" else 1

        if args.command == "chat":
            pinned_agent: str | None = args.agent_id
            show_thoughts = True
            show_sql = True
            agents = orch.list_agents()
            print(
                f"Genie Orchestrator ready ({len(agents)} configured agents, router={orch.config.router_model}, "
                f"mode={orch.config.execution_mode}). Commands: /agents, /use <id|auto>, /reset, /thoughts, /sql, /quit"
            )
            while True:
                try:
                    user_in = input("\ngenie> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user_in:
                    continue
                if user_in in {"/quit", "/exit"}:
                    break
                if user_in == "/agents":
                    for idx, ag in enumerate(orch.list_agents(), start=1):
                        print(f"{idx:2d}. {ag.agent_id}  {ag.title}")
                    continue
                if user_in == "/reset":
                    orch.reset_conversations()
                    print("Conversation state cleared.")
                    continue
                if user_in == "/thoughts":
                    show_thoughts = not show_thoughts
                    print(f"Show thoughts: {show_thoughts}")
                    continue
                if user_in == "/sql":
                    show_sql = not show_sql
                    print(f"Show SQL: {show_sql}")
                    continue
                if user_in.startswith("/use "):
                    target = user_in[len("/use ") :].strip()
                    pinned_agent = None if target.lower() in {"auto", "none", ""} else target
                    print(f"Pinned agent: {pinned_agent or 'auto (Gemini Decision Router)'}")
                    continue

                decision, resp = orch.ask(
                    user_in,
                    agent_id=pinned_agent,
                    on_event=_make_live_printer(show_thoughts=show_thoughts, show_sql=show_sql),
                )
                _print_full_response(
                    decision,
                    resp,
                    show_thoughts=show_thoughts,
                    show_sql=show_sql,
                )
            return 0

        if args.command == "generate":
            from txt2sql_eval_harness.cli import _bird_records_to_items
            from txt2sql_eval_harness.codec import decode_labelled_item
            from txt2sql_eval_harness.data import (
                load_bird_mini_dev,
                load_flex_dataset,
                record_manifest_entry,
            )
            from txt2sql_eval_harness.models import EvalConfig

            db_dir = args.db_dir or (args.data_dir / "dbs")
            if args.items is not None:
                base_items = [
                    decode_labelled_item(json.loads(line))
                    for line in args.items.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            elif args.dataset == "bird":
                bird_records, _ = load_bird_mini_dev(args.data_dir / "raw" / "mini_dev_sqlite.json", db_dir)
                base_items = _bird_records_to_items(bird_records, EvalConfig())
            else:
                base_items, _ = load_flex_dataset(
                    args.data_dir / "raw" / "flex_fp.jsonl",
                    args.data_dir / "raw" / "flex_fn.jsonl",
                    db_dir,
                    experts_csv=args.data_dir / "raw" / "experts_agreement_200.csv",
                )
            _, _, stats = generate_genie_items(
                base_items,
                orch,
                db_dir=db_dir,
                limit=args.limit,
                seed=args.seed,
                split=args.split,
                db_id=args.db_id,
                agent_id=args.agent_id,
                execution_mode=args.execution_mode,
                sql_only=args.sql_only,
                concurrency=args.concurrency,
                verbose=args.verbose,
                out_path=args.out,
                rich_out_path=args.rich_out,
            )
            manifest = args.data_dir / "manifest.json"
            if manifest.exists():
                record_manifest_entry(manifest, args.out, "harness_derivative")
            print(json.dumps({**stats, "out": str(args.out)}, indent=2))
            return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
