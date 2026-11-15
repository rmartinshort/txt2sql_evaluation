"""Sandboxed read-only SQLite execution with authorizer, timeouts, and fetch caps."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from txt2sql_eval_harness.models import Cell, ErrorKind, EvalConfig, ResultTable
from txt2sql_eval_harness.sql_analysis import is_volatile

_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        getattr(sqlite3, "SQLITE_RECURSIVE", 33),
    }
)


def _has_code_tokens(part: str) -> bool:
    """Return True if *part* contains non-whitespace characters outside comments."""
    i = 0
    n = len(part)
    while i < n:
        ch = part[i]
        nxt = part[i + 1] if i + 1 < n else ""
        if ch == "-" and nxt == "-":
            nl = part.find("\n", i + 2)
            i = n if nl == -1 else nl + 1
            continue
        if ch == "/" and nxt == "*":
            end = part.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if not ch.isspace():
            return True
        i += 1
    return False


def _split_top_level_statements(sql: str) -> list[str]:
    """Split SQL text by semicolons outside string literals, comments, and quoted identifiers."""
    stmts: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(sql)
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if in_line_comment:
            buf.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            buf.append(ch)
            if ch == "*" and nxt == "/":
                buf.append(nxt)
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_single:
            buf.append(ch)
            if ch == "'" and nxt == "'":
                buf.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            buf.append(ch)
            if ch == '"' and nxt == '"':
                buf.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "-" and nxt == "-":
            in_line_comment = True
            buf.append(ch)
            buf.append(nxt)
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            buf.append(ch)
            buf.append(nxt)
            i += 2
            continue
        if ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = True
            buf.append(ch)
            i += 1
            continue
        if ch == ";":
            part = "".join(buf).strip()
            if _has_code_tokens(part):
                stmts.append(part)
            buf.clear()
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if _has_code_tokens(tail):
        stmts.append(tail)
    return stmts


def _cell_size_bytes(val: Any) -> int:
    """Estimate storage size in bytes for a single decoded SQLite cell value."""
    if val is None:
        return 0
    if isinstance(val, bytes):
        return len(val)
    if isinstance(val, str):
        return len(val.encode("utf-8", errors="replace"))
    return 8


def _open_readonly_connection(db_path: Path, heap_limit_bytes: int) -> sqlite3.Connection:
    """Open an immutable read-only SQLite connection with extensions disabled and a heap limit."""
    abs_posix = db_path.resolve().as_posix()
    uri = f"file:{abs_posix}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    if hasattr(conn, "enable_load_extension"):
        conn.enable_load_extension(False)
    conn.execute("PRAGMA temp_store = MEMORY;")
    try:
        conn.execute(f"PRAGMA hard_heap_limit = {int(heap_limit_bytes)};")
    except sqlite3.Error:
        pass
    return conn


def execute_sandboxed(
    db_path: Path,
    sql: str,
    cfg: EvalConfig | None = None,
) -> tuple[ResultTable | None, ErrorKind | None, set[str]]:
    """Execute *sql* read-only inside the sandboxed SQLite runtime.

    Returns ``(result_table, error_kind, referenced_objects)``.
    """
    cfg = cfg or EvalConfig()
    referenced_objects: set[str] = set()

    if not sql or not sql.strip():
        return None, "compile_error", referenced_objects
    if len(sql.encode("utf-8", errors="replace")) > cfg.max_sql_bytes:
        return None, "resource_limit", referenced_objects

    stmts = _split_top_level_statements(sql)
    if len(stmts) == 0:
        return None, "compile_error", referenced_objects
    if len(stmts) > 1:
        return None, "multi_statement", referenced_objects
    single_sql = stmts[0]

    try:
        conn = _open_readonly_connection(db_path, cfg.sqlite_heap_limit_bytes)
    except sqlite3.Error:
        return None, "runtime", referenced_objects
    non_read_denied = False

    def authorizer(
        action: int,
        arg1: str | None,
        arg2: str | None,
        db_name: str | None,
        view_or_trigger: str | None,
    ) -> int:
        """Allow read-only SQLite opcodes and record referenced tables/views."""
        nonlocal non_read_denied
        if action in _ALLOWED_ACTIONS:
            if action == sqlite3.SQLITE_READ:
                if arg1:
                    referenced_objects.add(arg1)
                if view_or_trigger:
                    referenced_objects.add(view_or_trigger)
            return sqlite3.SQLITE_OK
        non_read_denied = True
        return sqlite3.SQLITE_DENY

    try:
        conn.set_authorizer(authorizer)
        try:
            conn.execute(f"EXPLAIN {single_sql}").fetchall()
        except sqlite3.DatabaseError as exc:
            msg = str(exc).lower()
            if non_read_denied or "not authorized" in msg:
                return None, "non_read", referenced_objects
            return None, "compile_error", referenced_objects
        except sqlite3.ProgrammingError:
            return None, "multi_statement", referenced_objects

        if non_read_denied:
            return None, "non_read", referenced_objects

        if is_volatile(single_sql):
            return None, "volatile", referenced_objects

        deadline = time.monotonic() + cfg.sql_timeout_s

        def progress_handler() -> int:
            """Interrupt SQLite VM execution once the per-query wall-clock deadline elapses."""
            if time.monotonic() > deadline:
                return 1
            return 0

        conn.set_progress_handler(progress_handler, 1000)

        try:
            cur = conn.execute(single_sql)
            if cur.description is None:
                return None, "non_read", referenced_objects
            columns = [desc[0] for desc in cur.description]
            if len(columns) > cfg.max_columns:
                return None, "resource_limit", referenced_objects

            rows: list[tuple[Cell, ...]] = []
            total_bytes = 0
            truncated = False

            while True:
                batch = cur.fetchmany(1000)
                if not batch:
                    break
                for row in batch:
                    row_tuple: list[Cell] = []
                    row_bytes = 0
                    for val in row:
                        if isinstance(val, memoryview):
                            val = bytes(val)
                        sz = _cell_size_bytes(val)
                        if sz > cfg.max_cell_bytes:
                            return None, "resource_limit", referenced_objects
                        row_bytes += sz
                        row_tuple.append(val)
                    if len(rows) >= cfg.fetch_cap_rows or (
                        total_bytes + row_bytes > cfg.fetch_cap_bytes
                    ):
                        truncated = True
                        break
                    rows.append(tuple(row_tuple))
                    total_bytes += row_bytes
                if truncated:
                    break

            if not truncated:
                return (
                    ResultTable(
                        columns=columns,
                        rows=rows,
                        total_rows=len(rows),
                        complete=True,
                    ),
                    None,
                    referenced_objects,
                )

            # Prefix reached: compute total_rows under the same deadline.
            total_rows: int | None = None
            try:
                count_sql = f"SELECT COUNT(*) FROM ({single_sql})"
                count_row = conn.execute(count_sql).fetchone()
                if count_row is not None and isinstance(count_row[0], int):
                    total_rows = count_row[0]
            except sqlite3.OperationalError:
                total_rows = None

            return (
                ResultTable(
                    columns=columns,
                    rows=rows,
                    total_rows=total_rows,
                    complete=False,
                ),
                "fetch_cap",
                referenced_objects,
            )
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "interrupted" in msg:
                return None, "timeout", referenced_objects
            if (
                getattr(exc, "sqlite_errorcode", None) == getattr(sqlite3, "SQLITE_NOMEM", -1)
                or "out of memory" in msg
            ):
                return None, "resource_limit", referenced_objects
            if non_read_denied or "not authorized" in msg:
                return None, "non_read", referenced_objects
            return None, "runtime", referenced_objects
        except sqlite3.DatabaseError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg:
                return None, "resource_limit", referenced_objects
            return None, "runtime", referenced_objects
    finally:
        try:
            conn.set_authorizer(None)
            conn.set_progress_handler(None, 0)
        except Exception:
            pass
        conn.close()
