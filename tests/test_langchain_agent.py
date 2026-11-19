"""Tests for the LangChain Text2SQL agent generator and CLI integration."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from txt2sql_eval_harness.cli import main as cli_main
from txt2sql_eval_harness.codec import decode_labelled_item
from txt2sql_eval_harness.data import load_manifest
from txt2sql_eval_harness.judge import judge_item
from txt2sql_eval_harness.clients import CallContext
from txt2sql_eval_harness.langchain_agent import (
    _open_readonly_sql_database,
    generate_langchain_items,
    run_langchain_sql_agent,
)
from txt2sql_eval_harness.metrics import compute_method_metrics
from txt2sql_eval_harness.models import EvalConfig, GroundTruth, LabelledItem, Trace


class _FakeToolCallingChatModel(BaseChatModel):
    """Deterministic fake chat model that issues one sql_db_query tool call, then answers."""

    sql_to_run: str
    answer_text: str

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        has_tool_result = any(isinstance(m, ToolMessage) for m in messages)
        if not has_tool_result:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_query",
                        "args": {"query": self.sql_to_run},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            msg = AIMessage(content=self.answer_text)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _make_test_db(db_dir: Path, db_id: str = "shop") -> Path:
    db_path = db_dir / db_id / f"{db_id}.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, region TEXT, amount INTEGER)")
    conn.executemany(
        "INSERT INTO orders (id, region, amount) VALUES (?, ?, ?)",
        [(1, "West", 10), (2, "West", 20), (3, "East", 50)],
    )
    conn.commit()
    conn.close()
    return db_path


def test_readonly_db_and_run_langchain_sql_agent(tmp_path: Path) -> None:
    db_dir = tmp_path / "dbs"
    db_path = _make_test_db(db_dir)

    db = _open_readonly_sql_database(db_path)
    assert "orders" in db.get_usable_table_names()

    llm = _FakeToolCallingChatModel(
        sql_to_run="SELECT SUM(amount) AS total FROM orders WHERE region = 'West'",
        answer_text="The total West amount is 30.",
    )
    trace, info = run_langchain_sql_agent(
        trace_id="t1",
        group_id="g1",
        question="What is the total amount for West?",
        hint=None,
        db_id="shop",
        db_sha256="0" * 64,
        db_path=db_path,
        llm=llm,
    )
    assert info["status"] == "ok"
    assert trace is not None
    assert len(trace.steps) == 1
    assert trace.steps[0].result is not None
    assert trace.steps[0].result.rows == [(30,)]
    assert trace.final_answer == "The total West amount is 30."


def test_generate_langchain_items_and_judge_and_cli(tmp_path: Path, monkeypatch: Any) -> None:
    db_dir = tmp_path / "data" / "dbs"
    _make_test_db(db_dir)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Minimal BIRD-style JSON fixture
    bird_json = raw_dir / "mini_dev_sqlite.json"
    bird_json.write_text(
        json.dumps(
            [
                {
                    "question_id": 101,
                    "db_id": "shop",
                    "question": "Total West amount?",
                    "evidence": "",
                    "SQL": "SELECT SUM(amount) FROM orders WHERE region = 'West'",
                }
            ]
        ),
        encoding="utf-8",
    )

    llm = _FakeToolCallingChatModel(
        sql_to_run="SELECT SUM(amount) FROM orders WHERE region = 'West'",
        answer_text="30",
    )
    base_item = LabelledItem(
        item_id="bird:101",
        group_id="g101",
        split="test",
        set="S1",
        family="base",
        slices=[],
        trace=Trace("bird:101", "g101", "Total West amount?", None, "shop", "0", []),
        gt=GroundTruth("bird:101", ["SELECT SUM(amount) FROM orders WHERE region = 'West'"]),
        label="correct",
    )

    items, stats = generate_langchain_items(
        [base_item],
        db_dir,
        llm,
        "gemini-2.5-flash",
        limit=5,
        split="test",
    )
    assert stats["generated"] == 1
    assert len(items) == 1
    it = items[0]
    assert it.set == "S5"
    assert it.label == "unlabelled"
    assert it.family == "langchain:gemini-2.5-flash"
    assert it.gt.results is not None and it.gt.results[0].rows == [(30,)]

    # Verify it can be judged and summarized as an unlabelled item
    v = judge_item(it.trace, it.gt, "H+E", CallContext(cfg=EvalConfig(), allow_remote=False))
    assert v.mode == "full_trace"
    m = compute_method_metrics(items, {it.item_id: v}, bootstrap_resamples=0)
    assert m["n_unlabelled"] == 1

    # Verify CLI generate-langchain writes JSONL and updates manifest.json
    manifest_path = tmp_path / "data" / "manifest.json"
    manifest_path.write_text(json.dumps({"entries": {}, "databases": {}}), encoding="utf-8")
    out_path = tmp_path / "data" / "lc_out.jsonl"
    monkeypatch.setattr(
        "txt2sql_eval_harness.langchain_agent.build_chat_model",
        lambda model, temperature=0.0: llm,
    )
    rc = cli_main(
        [
            "generate-langchain",
            "--dataset",
            "bird",
            "--data-dir",
            str(tmp_path / "data"),
            "--model",
            "gemini-2.5-flash",
            "--split",
            "all",
            "--limit",
            "1",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    lines = [decode_labelled_item(json.loads(line)) for line in out_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0].set == "S5"
    manifest = load_manifest(manifest_path)
    assert len(manifest["entries"]) == 1
