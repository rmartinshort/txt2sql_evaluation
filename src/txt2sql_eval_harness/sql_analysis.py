"""SQL parsing, normalization (N), volatility/determinism predicates, Contract, and DDL extraction."""

from __future__ import annotations

import os
import random
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Iterable

import sqlglot
from sqlglot import exp

from txt2sql_eval_harness.models import Contract

VALIDITY_NOTE = (
    "Valid data means any rows SQLite would accept for these statements with default settings: "
    "NOT NULL, PRIMARY KEY and UNIQUE (NULLs are distinct), CHECK (a NULL result passes) and "
    "STRICT column types are enforced; an INTEGER PRIMARY KEY is never NULL; FOREIGN KEY is not "
    "enforced; outside STRICT tables a declared type is only an affinity, so any value type can occur."
)

_VOLATILE_CALL_RE = re.compile(
    r"\b(?:random|randomblob|changes|total_changes|last_insert_rowid)\s*\(",
    re.IGNORECASE,
)
_VOLATILE_DATE_FN_RE = re.compile(
    r"\b(?:date|time|datetime|julianday|unixepoch|strftime)\s*\(",
    re.IGNORECASE,
)
_VOLATILE_NOW_RE = re.compile(r"'now'", re.IGNORECASE)
_VOLATILE_CURRENT_RE = re.compile(
    r"\b(?:current_date|current_time|current_timestamp)\b",
    re.IGNORECASE,
)
_VOLATILE_ZERO_ARG_DATE_RE = re.compile(
    r"\b(?:date|time|datetime|julianday|unixepoch)\s*\(\s*\)",
    re.IGNORECASE,
)
_STRFTIME_RE = re.compile(r"\bstrftime\s*\(", re.IGNORECASE)

_ORDER_CUES = frozenset(
    {
        "top",
        "highest",
        "lowest",
        "most",
        "least",
        "largest",
        "smallest",
        "first",
        "last",
        "rank",
        "ranked",
        "order",
        "sorted",
        "ascending",
        "descending",
    }
)

_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "thirteen|fourteen|fifteen|twenty|thirty|fifty|hundred|thousand"
)
_LIMIT_NUM_PLURAL_RE = re.compile(
    rf"\b(?:\d+|{_NUMBER_WORDS})\s+[a-z]+s\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def _ascii_lower(text: str) -> str:
    """Lowercase only ASCII A-Z characters, preserving non-ASCII characters."""
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in text)


def _has_one_arg_strftime(sql: str) -> bool:
    """Detect `strftime(...)` calls with exactly one argument."""
    for match in _STRFTIME_RE.finditer(sql):
        idx = match.end()
        depth = 1
        commas = 0
        in_single = False
        in_double = False
        inner_chars: list[str] = []
        n = len(sql)
        while idx < n and depth > 0:
            ch = sql[idx]
            if in_single:
                if ch == "'" and idx + 1 < n and sql[idx + 1] == "'":
                    inner_chars.append("''")
                    idx += 2
                    continue
                if ch == "'":
                    in_single = False
                inner_chars.append(ch)
            elif in_double:
                if ch == '"' and idx + 1 < n and sql[idx + 1] == '"':
                    inner_chars.append('""')
                    idx += 2
                    continue
                if ch == '"':
                    in_double = False
                inner_chars.append(ch)
            else:
                if ch == "'":
                    in_single = True
                    inner_chars.append(ch)
                elif ch == '"':
                    in_double = True
                    inner_chars.append(ch)
                elif ch == "(":
                    depth += 1
                    inner_chars.append(ch)
                elif ch == ")":
                    depth -= 1
                    if depth > 0:
                        inner_chars.append(ch)
                elif ch == "," and depth == 1:
                    commas += 1
                else:
                    inner_chars.append(ch)
            idx += 1
        if depth == 0 and commas == 0 and "".join(inner_chars).strip():
            return True
    return False


def _strip_comments_and_literals(sql: str) -> tuple[str, str]:
    """Return `(sql_without_comments, sql_without_comments_or_quoted_literals)`."""
    no_comments: list[str] = []
    no_literals: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if ch == "-" and nxt == "-":
            nl = sql.find("\n", i + 2)
            end = n if nl == -1 else nl
            no_comments.append(" " * (end - i))
            no_literals.append(" " * (end - i))
            i = end
            continue
        if ch == "/" and nxt == "*":
            end_idx = sql.find("*/", i + 2)
            end = n if end_idx == -1 else end_idx + 2
            no_comments.append(" " * (end - i))
            no_literals.append(" " * (end - i))
            i = end
            continue
        if ch in ("'", '"'):
            quote = ch
            j = i + 1
            while j < n:
                if sql[j] == quote:
                    if j + 1 < n and sql[j + 1] == quote:
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            span = sql[i:j]
            no_comments.append(span)
            no_literals.append(quote + (" " * max(0, len(span) - 2)) + (quote if len(span) >= 2 else ""))
            i = j
            continue
        no_comments.append(ch)
        no_literals.append(ch)
        i += 1
    return "".join(no_comments), "".join(no_literals)


def _has_now_in_date_fn(no_comments_sql: str) -> bool:
    """Detect `'now'` passed inside a SQLite date/time function call."""
    n = len(no_comments_sql)
    for match in _VOLATILE_DATE_FN_RE.finditer(no_comments_sql):
        idx = match.end()
        depth = 1
        in_single = False
        in_double = False
        arg_start = idx
        while idx < n and depth > 0:
            ch = no_comments_sql[idx]
            if in_single:
                if ch == "'" and idx + 1 < n and no_comments_sql[idx + 1] == "'":
                    idx += 2
                    continue
                if ch == "'":
                    in_single = False
            elif in_double:
                if ch == '"' and idx + 1 < n and no_comments_sql[idx + 1] == '"':
                    idx += 2
                    continue
                if ch == '"':
                    in_double = False
            else:
                if ch == "'":
                    in_single = True
                elif ch == '"':
                    in_double = True
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
            idx += 1
        if _VOLATILE_NOW_RE.search(no_comments_sql[arg_start:idx]):
            return True
    return False


def is_volatile(sql: str) -> bool:
    """Return True if SQL references non-deterministic SQLite functions (RANDOM, date/time/now)."""
    no_comments, no_literals = _strip_comments_and_literals(sql)
    if _VOLATILE_CALL_RE.search(no_literals):
        return True
    if _VOLATILE_CURRENT_RE.search(no_literals):
        return True
    if _VOLATILE_ZERO_ARG_DATE_RE.search(no_literals):
        return True
    if _has_now_in_date_fn(no_comments):
        return True
    if _has_one_arg_strftime(no_comments):
        return True
    return False


def parse_single_sqlite(sql: str) -> exp.Expression | None:
    """Parse a single SQLite statement with sqlglot; return None on failure or multi-statement."""
    if not sql or not sql.strip():
        return None
    try:
        stmts = sqlglot.parse(sql, read="sqlite")
    except Exception:
        return None
    if len(stmts) != 1 or stmts[0] is None:
        return None
    return stmts[0]


def _normalize_tree_in_place(tree: exp.Expression) -> None:
    """Strip comments, ASCII-lowercase unquoted identifiers, and drop unused outer column aliases."""
    for node in tree.walk():
        node.comments = None
        if isinstance(node, exp.Identifier) and not node.quoted and isinstance(node.this, str):
            node.set("this", _ascii_lower(node.this))

    if isinstance(tree, exp.Select):
        outer_alias_id_nodes: set[int] = set()
        for expr in tree.expressions:
            if isinstance(expr, exp.Alias):
                alias_node = expr.args.get("alias")
                if alias_node is not None:
                    outer_alias_id_nodes.add(id(alias_node))

        other_names: set[str] = set()
        for node in tree.walk():
            if isinstance(node, exp.Identifier) and id(node) not in outer_alias_id_nodes:
                if isinstance(node.this, str):
                    key = node.this if node.quoted else _ascii_lower(node.this)
                    other_names.add(key)

        new_expressions: list[exp.Expression] = []
        for expr in tree.expressions:
            if isinstance(expr, exp.Alias):
                alias_node = expr.args.get("alias")
                alias_name: str | None = None
                if isinstance(alias_node, exp.Identifier) and isinstance(alias_node.this, str):
                    alias_name = (
                        alias_node.this if alias_node.quoted else _ascii_lower(alias_node.this)
                    )
                elif isinstance(alias_node, str):
                    alias_name = _ascii_lower(alias_node)
                if alias_name is not None and alias_name not in other_names:
                    new_expressions.append(expr.this)
                    continue
            new_expressions.append(expr)
        tree.set("expressions", new_expressions)


def normalize_sql_n(sql: str) -> str | None:
    """Compute canonical presentation-normalized SQL form N for rule D1.

    Drops comments, ASCII-lowercases unquoted identifiers while preserving
    quoted identifiers and literals verbatim, retains table aliases, and strips
    redundant outer column aliases not referenced elsewhere in the query.
    """
    tree = parse_single_sqlite(sql)
    if tree is None:
        return None
    _normalize_tree_in_place(tree)
    try:
        return tree.sql(dialect="sqlite", comments=False)
    except Exception:
        return None


def _canon_select_tree(sql: str, *, drop_order: bool = False) -> exp.Select | None:
    """Parse and canonicalize a single SQLite SELECT AST for projection-superset checks."""
    tree = parse_single_sqlite(sql)
    if not isinstance(tree, exp.Select):
        return None
    _normalize_tree_in_place(tree)

    # Normalize `x BETWEEN low AND high` into `(x >= low AND x <= high)`, unwrapping
    # redundant parentheses around conjunctions inside WHERE/AND chains.
    for btw in list(tree.find_all(exp.Between)):
        x = btw.this.copy()
        low = btw.args["low"].copy()
        high = btw.args["high"].copy()
        btw.replace(
            exp.Paren(
                this=exp.And(
                    this=exp.GTE(this=x, expression=low),
                    expression=exp.LTE(this=x.copy(), expression=high),
                )
            )
        )
    for paren in list(tree.find_all(exp.Paren)):
        if isinstance(paren.this, exp.And) and isinstance(paren.parent, (exp.Where, exp.And)):
            paren.replace(paren.this)

    if drop_order and not tree.args.get("limit") and not tree.args.get("offset"):
        tree.set("order", None)
    return tree


def is_d1_superset_equivalent(
    gt_sql: str,
    gen_sql: str,
    *,
    order_required: bool = True,
) -> bool:
    """Return True if *gen_sql* has identical relational body to *gt_sql* and superset SELECT columns.

    Handles column additions, column permutations, and `BETWEEN` expansions when
    all `FROM`, `JOIN`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`, and CTE clauses match.
    If either query uses `DISTINCT`, exact SELECT expression-set equality is required.
    """
    drop_order = not order_required
    t_gt = _canon_select_tree(gt_sql, drop_order=drop_order)
    t_gen = _canon_select_tree(gen_sql, drop_order=drop_order)
    if t_gt is None or t_gen is None:
        return False

    try:
        gt_exprs = [e.sql(dialect="sqlite", comments=False) for e in t_gt.expressions]
        gen_exprs = [e.sql(dialect="sqlite", comments=False) for e in t_gen.expressions]
        if not gt_exprs or not set(gt_exprs).issubset(set(gen_exprs)):
            return False

        if (t_gt.args.get("distinct") or t_gen.args.get("distinct")) and set(gt_exprs) != set(
            gen_exprs
        ):
            return False

        t_gt_rest = t_gt.copy()
        t_gen_rest = t_gen.copy()
        t_gt_rest.set("expressions", [exp.Literal.number(1)])
        t_gen_rest.set("expressions", [exp.Literal.number(1)])
        return t_gt_rest.sql(dialect="sqlite", comments=False) == t_gen_rest.sql(
            dialect="sqlite", comments=False
        )
    except Exception:
        return False


def is_det(sql: str) -> bool:
    """Return True iff *sql* parses, is not volatile, and avoids non-deterministic constructs."""
    if is_volatile(sql):
        return False
    tree = parse_single_sqlite(sql)
    if tree is None:
        return False

    for node in tree.walk():
        if isinstance(node, (exp.GroupConcat, exp.Window, exp.Anonymous)):
            return False
        if isinstance(node, (exp.Limit, exp.Offset)):
            parent = node.parent
            if parent is None or not parent.args.get("order"):
                return False
    return True


def is_rd_eligible(sql: str) -> bool:
    """Return True iff *sql* is deterministic and contains no LIMIT/OFFSET in any scope."""
    if not is_det(sql):
        return False
    tree = parse_single_sqlite(sql)
    if tree is None:
        return False
    for node in tree.walk():
        if isinstance(node, (exp.Limit, exp.Offset)):
            return False
    return True


def extract_contract(gt_sql: str, question: str, n_cols: int = 0) -> Contract:
    """Derive the must-match Contract from the question cue lexicon and GT SQL."""
    words = {w.lower() for w in _WORD_RE.findall(question)}
    order_cued = bool(words & _ORDER_CUES)
    limit_cued = order_cued or bool(_LIMIT_NUM_PLURAL_RE.search(question))

    tree = parse_single_sqlite(gt_sql)
    has_order = False
    has_limit = False
    if tree is not None:
        has_order = bool(tree.args.get("order"))
        has_limit = bool(tree.args.get("limit"))

    return Contract(
        gt_columns=list(range(n_cols)),
        order_required=order_cued and has_order,
        limit_required=limit_cued and has_limit,
    )


def extract_schema_ddl(
    db_path: Path,
    referenced_objects: Iterable[str],
    *,
    include_validity_note: bool = False,
) -> str:
    """Build SCHEMA DDL block: verbatim sqlite_master SQL for referenced objects."""
    refs_lower = {r.lower() for r in referenced_objects if r}
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    ddl_parts: list[str] = []
    try:
        rows = conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL ORDER BY rowid"
        ).fetchall()
        for obj_type, name, tbl_name, sql_text in rows:
            name_l = (name or "").lower()
            tbl_l = (tbl_name or "").lower()
            if obj_type in ("table", "view") and name_l in refs_lower:
                ddl_parts.append(sql_text.strip())
            elif obj_type == "trigger" and tbl_l in refs_lower:
                ddl_parts.append(sql_text.strip())
            elif obj_type == "index" and tbl_l in refs_lower:
                if "unique" in sql_text.lower().split("index", 1)[0]:
                    ddl_parts.append(sql_text.strip())
    finally:
        conn.close()

    body = "\n\n".join(ddl_parts)
    if not include_validity_note:
        return body
    if not body:
        return VALIDITY_NOTE
    return VALIDITY_NOTE + "\n\n" + body


def create_d5_deletion_copy(
    db_path: Path,
    out_path: Path,
    *,
    seed: int = 101,
    delete_fraction: float = 0.30,
) -> Path | None:
    """Create a seeded 30% row-deletion database copy for rule D5 counterexample checking.

    Returns None if the source database contains triggers, since deletions could
    violate trigger-maintained invariants.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    shutil.copy2(db_path, out_path)

    conn = sqlite3.connect(str(out_path))
    try:
        trig_count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger'"
        ).fetchone()[0]
        if trig_count > 0:
            conn.close()
            out_path.unlink(missing_ok=True)
            return None

        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        rng = random.Random(seed)
        for tbl in tables:
            quoted = '"' + tbl.replace('"', '""') + '"'
            try:
                rowids = [
                    r[0]
                    for r in conn.execute(f"SELECT rowid FROM {quoted} ORDER BY rowid").fetchall()
                ]
            except sqlite3.OperationalError:
                # WITHOUT ROWID table: skip deletion cleanly
                continue
            k = int(len(rowids) * delete_fraction)
            if k <= 0:
                continue
            to_delete = rng.sample(rowids, k)
            for idx in range(0, len(to_delete), 500):
                chunk = to_delete[idx : idx + 500]
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(f"DELETE FROM {quoted} WHERE rowid IN ({placeholders})", chunk)
        conn.commit()
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise RuntimeError(f"D5 integrity_check failed on {out_path}: {integrity}")
    finally:
        conn.close()
    return out_path
