"""Write-once SQLite response cache for validated model responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from txt2sql_eval_harness.models import Attempt, CallMeasure


@dataclass(frozen=True)
class CachedEntry:
    """Immutable cache row holding validated answers and original cold-call measurements."""

    request_sha256: str
    stage: str
    model: str
    answers: dict[str, Any]
    cold: CallMeasure


def _attempt_to_dict(a: Attempt) -> dict[str, Any]:
    """Serialize an Attempt dataclass into a JSON-compatible dictionary."""
    return {
        "n": a.n,
        "wait_ms": a.wait_ms,
        "elapsed_ms": a.elapsed_ms,
        "result": a.result,
        "input_tokens": a.input_tokens,
        "output_tokens": a.output_tokens,
        "billing": a.billing,
        "cost_usd": a.cost_usd,
    }


def _dict_to_attempt(d: dict[str, Any]) -> Attempt:
    """Reconstruct an Attempt dataclass from a decoded JSON dictionary."""
    return Attempt(
        n=int(d["n"]),
        wait_ms=int(d["wait_ms"]),
        elapsed_ms=int(d["elapsed_ms"]),
        result=d["result"],
        input_tokens=int(d["input_tokens"]),
        output_tokens=int(d["output_tokens"]) if d["output_tokens"] is not None else None,
        billing=d["billing"],
        cost_usd=float(d["cost_usd"]),
    )


def _measure_to_json(m: CallMeasure) -> str:
    """Serialize a CallMeasure into canonical ASCII JSON without NaNs."""
    payload = {
        "attempts": [_attempt_to_dict(a) for a in m.attempts],
        "wall_ms": m.wall_ms,
        "billed_usd": m.billed_usd,
        "ambiguous_usd": m.ambiguous_usd,
    }
    return json.dumps(payload, separators=(",", ":"), allow_nan=False, ensure_ascii=True)


def _json_to_measure(raw: str) -> CallMeasure:
    """Deserialize a CallMeasure from its stored JSON representation."""
    d = json.loads(raw)
    return CallMeasure(
        attempts=[_dict_to_attempt(a) for a in d["attempts"]],
        wall_ms=int(d["wall_ms"]),
        billed_usd=float(d["billed_usd"]),
        ambiguous_usd=float(d["ambiguous_usd"]),
    )


class ResponseCache:
    """Lazy-opening write-once SQLite cache storing only validated 'ok' model responses."""

    def __init__(self, db_path: Path) -> None:
        """Store the cache database path without opening or creating the file until needed."""
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _ensure_conn(self) -> sqlite3.Connection:
        """Open the SQLite cache lazily and initialize the responses table on first write/read."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    request_sha256 TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    model TEXT NOT NULL,
                    answers_json TEXT NOT NULL,
                    cold_measure_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
            self._conn = conn
        return self._conn

    def get(self, request_sha256: str) -> CachedEntry | None:
        """Return a cached response for *request_sha256*, or None without touching disk if absent."""
        with self._lock:
            # Avoid creating an empty SQLite file during --local-only or cold read checks.
            if self._conn is None and not self.db_path.exists():
                return None
            conn = self._ensure_conn()
            row = conn.execute(
                """
                SELECT request_sha256, stage, model, answers_json, cold_measure_json
                FROM responses
                WHERE request_sha256 = ?
                """,
                (request_sha256,),
            ).fetchone()
            if row is None:
                return None
            return CachedEntry(
                request_sha256=str(row[0]),
                stage=str(row[1]),
                model=str(row[2]),
                answers=json.loads(row[3]),
                cold=_json_to_measure(row[4]),
            )

    def put_once(
        self,
        request_sha256: str,
        stage: str,
        model: str,
        answers: dict[str, Any],
        cold: CallMeasure,
    ) -> bool:
        """Insert a validated response once; existing keys are never overwritten."""
        with self._lock:
            conn = self._ensure_conn()
            answers_json = json.dumps(
                answers, separators=(",", ":"), allow_nan=False, ensure_ascii=True
            )
            cold_json = _measure_to_json(cold)
            created_at = datetime.now(timezone.utc).isoformat()
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO responses (
                    request_sha256, stage, model, answers_json, cold_measure_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (request_sha256, stage, model, answers_json, cold_json, created_at),
            )
            conn.commit()
            return cur.rowcount == 1

    def close(self) -> None:
        """Close the underlying SQLite connection if it was opened."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> ResponseCache:
        """Enter context manager returning self."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit context manager and close any open SQLite connection."""
        self.close()
