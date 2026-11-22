# Text2SQL Decision-Model Judge Harness (`txt2sql-eval-harness`)

A deterministic-first hybrid evaluation harness for judging Text-to-SQL outputs and multi-step SQL agent traces against ground-truth references. It combines sandboxed SQL execution, syntactic/semantic verification rules, exact column/row table alignment, and numerical claim binding with calibrated probabilistic evaluation via **TypeSafe Jev** (`jev-1.14.0`) and selective generative escalation (`gemini-4-argon`).

---

## Overview & 5-Stage Hybrid Judge Funnel

Evaluating Text-to-SQL predictions on execution equality alone often misses subtle SQL bugs (coincidental result agreement on a single database, dropped ordering constraints, duplicate/missing rows, or unverified extra columns) and cannot grade multi-step agent traces that end in natural-language answers. Conversely, sending raw full tables and prompts directly to a frontier LLM judge on every example is expensive and brittle on large outputs or numeric rounding.

This harness solves both problems with a staged verification funnel (`judge_item` in `src/txt2sql_eval_harness/judge.py`):

1. **Stage 0 — Sandbox Execution & Schema Extraction**:
   - Rejects obviously-oversized inputs before doing any work (`max_steps=8` agent steps, `max_sql_bytes=32 KiB` per SQL statement, `max_answer_chars=20,000` for the agent's final answer, `max_columns=64` per result table).
   - Executes any ground-truth or candidate SQL step that doesn't already have a cached result, inside a locked-down SQLite connection (`execute_sandboxed` in `sandbox.py`). Concretely, this means:
     - **Opcode authorizer**: SQLite calls a callback before running every low-level operation in a query; the callback allows only read opcodes (`SELECT`, `READ`, `FUNCTION`, `RECURSIVE`) and denies everything else (`INSERT`, `UPDATE`, `DELETE`, `ATTACH`, pragmas that mutate state, etc.), so a candidate query cannot write to or corrupt the database even if it tries.
     - **Progress timeout (`sql_timeout_s=10.0`)**: a progress handler checks a wall-clock deadline every 1,000 VM instructions and aborts the query if it runs longer than 10 seconds, so a slow or infinite query can't hang the harness.
     - **Heap limit (`256 MiB`)**: `PRAGMA hard_heap_limit` caps how much memory SQLite itself can allocate while running the query, so a pathological query (e.g. a huge cross join) fails fast with an out-of-memory error instead of consuming unbounded host memory.
     - **Bounded row/byte fetch caps (`200,000` rows / `16 MiB`)**: rows are streamed out in batches and fetching stops once either cap is hit; the result is marked `complete=False` and the harness separately asks SQLite for the true total row count, so the rest of the pipeline still knows how big the real result would have been.
     - The database itself is opened read-only and `immutable=1` (SQLite assumes the file won't change underneath it, which also lets it skip locking), and extension loading is disabled, as further defense in depth on top of the authorizer.
   - If a ground-truth item supplies more than one acceptable SQL formulation (`gt.alternatives`), each is executed and scored, and `prepare_best_candidate` (`judge.py`) keeps the single best one to evaluate the candidate against. "Best" is decided by a fixed priority order, with no LLM involved:
     1. Prefer an alternative whose deterministic rules (`D0`–`D5`, below) already produced a `CORRECT` verdict over one that produced `INCORRECT`, over one that reached no rule at all.
     2. Among ties, prefer the alternative whose result table has a closer relationship to the candidate's result table (`equal` beats `overlap`/`subset` beats `unknown` beats `disjoint`).
     3. Among further ties, prefer the alternative with more columns successfully matched to the candidate's columns.
     4. Any remaining tie is broken by picking the earliest-listed alternative.

2. **Stage 1 — Deterministic Analysis, Alignment, Claims & Rules (`D0`–`D5`)**:
   - Derives a per-question `Contract` (`sql_analysis.py:extract_contract`) purely by pattern-matching the question text against the ground-truth SQL's own structure — **no LLM is involved**. Concretely: the question's words are checked against a fixed cue lexicon (`top`, `highest`, `most`, `first`, `ranked`, `sorted`, etc., or a number-of-things pattern like `"top 5 cities"`); `order_required` is only set `True` if the question contains one of those cues **and** the ground-truth SQL itself has an `ORDER BY`; `limit_required` is only set `True` if the question is order/count-cued **and** the ground-truth SQL itself has a `LIMIT`. In other words, the contract never requires something the reference query doesn't already do — it only flags, from the question's own wording, which of the reference's existing guarantees (ordering, row limit) the candidate is also expected to honor when grading row alignment.
   - Extracts the **schema DDL** (`sql_analysis.py:extract_schema_ddl`): the verbatim `CREATE TABLE` / `CREATE VIEW` / `CREATE TRIGGER` / `CREATE UNIQUE INDEX` statements, copied as-is from SQLite's own `sqlite_master` catalog, for every table/view touched by either the ground-truth or candidate SQL (tables are identified by watching which objects the sandbox's opcode authorizer actually reads, plus a syntactic scan of the SQL). This is included in the evidence packet (Stage 2) so the downstream judge can see each column's declared type/constraints, not just column names, without needing the full database dump.
   - Computes an exact column bijection and row alignment/diff between ground-truth and candidate `ResultTable` outputs (`alignment.py` & `comparator.py`) — i.e. which generated column corresponds to which reference column, and whether the generated rows are the same set, a subset, a superset, overlapping, or disjoint from the reference rows — including unit-scale matching (values that match after multiplying by `100` or `0.01`, which catches percentage-vs-fraction mismatches) and relative numerical tolerance (`rho = 1e-9`) for floating-point comparisons.
   - Extracts and binds numerical, date, and aggregate claims from agent `final_answer` text against execution tables (`claims.py`): the harness scans the natural-language answer for numbers/dates it asserts, then tries to find a matching cell in the ground-truth or step result tables (or a code-computed aggregate — see `DERIVATIONS` below) that would justify that claim, so unsupported claims can be caught deterministically.
   - Applies deterministic short-circuit rules, in priority order, before any remote model call — if one of these fires, the verdict is decided without ever calling Jev or an LLM:
     - **D0 (`rule:D0`)**: Immediate `INCORRECT` in `full_trace` mode if `final_answer` is missing or empty — an agent that gives no answer cannot be correct.
     - **D3 (`rule:D3`)**: Immediate `INCORRECT` in `sql_only` mode if the candidate SQL fails to compile, or is denied by the sandbox authorizer as a non-read statement.
     - **D2 (`rule:D2`)**: Immediate `INCORRECT` if the candidate query returns no usable rows/result while the reference result is non-empty.
     - **D4 (`rule:D4`)**: Immediate `INCORRECT` when both the reference and candidate queries are single-number (1x1) results that disagree even after trying the `x100`/`x0.01` scale adjustments — i.e. a genuine numeric mismatch, not just a units difference.
     - **D5 (`rule:D5`, opt-in via `--enable-d5` / `enable_d5=True`)**: Re-runs both the reference and candidate queries against a seeded copy of the database with a random 30% of rows deleted from every table (`create_d5_deletion_copy`). If they now disagree, their original agreement was coincidental (e.g. both queries happened to return the same count on this particular dataset without being logically equivalent), so the item is marked `INCORRECT`.
     - **D1 (`rule:D1`)**: Immediate `CORRECT` when both the reference and candidate SQL are deterministic (no `RANDOM()`, no unordered `LIMIT`, etc.) and reduce to identical SQL after presentation normalization (`normalize_sql_n` — lowercasing keywords/identifiers, stripping comments, stripping redundant aliases), provided complete execution results don't actually contradict that equality.

3. **Stage 2 — Bounded Canonical Evidence Packet (`packet.py` & `renderer.py`)**:
   - Packs `QUESTION`, `HINT`, `SCHEMA_DDL`, normalized/original SQL, execution errors, alignment relation, bounded witness rows/column stats, bound claims, code-computed `DERIVATIONS`, and flags into a token-budgeted canonical JSON packet — this is the *only* evidence the Jev/escalation judge ever sees; it never gets the raw database or the full result tables. The packet is capped at `jev_max_cells=400` cells / `jev_max_tokens=6,000` tokens for normal Jev scoring, with a larger `escalation_max_cells=1,200` / `escalation_max_tokens=16,000` tier reserved for escalation. If a packet doesn't fit, sections are dropped in a fixed order (full result tables first, then extra row-diff witnesses) until it does, or the item abstains if it still doesn't fit (see `b4_max_tokens` for how the `B4` baseline instead truncates rather than abstaining). See [**Example Evidence Packet**](#example-evidence-packet) below for exactly what this looks like for a real `sql_only` item.

4. **Stage 3 — TypeSafe Jev Calibrated Scoring (`clients.py` & `rubrics.py`)**:
   - Evaluates structured rubric questions via TypeSafe Jev (`jev-1.14.0`):
     - **`sql_only` mode**: `Q1` (`same_intent`), conditional `Q2` (`row_difference` when alignment relation is `gt_subset_gen`, `gen_subset_gt`, or `overlap`), and `Q5` (`ambiguity`).
     - **`full_trace` mode**: `Q3` (`answer_true`), `Q4` (`answer_complete`), `Q6` (`answer_supported`), and `Q5` (`ambiguity`).
   - Maps calibrated probabilities against abstain thresholds (`ThresholdPair(tau_hi=0.85, tau_lo=0.15)` by default) into `CORRECT`, `INCORRECT`, `AMBIGUOUS`, or `ESCALATE`.

5. **Stage 4 — Selective Generative Escalation**:
   - Invoked when evidence exceeds Jev bounds, Jev returns an uncertain probability band, or execution evidence conflicts with Jev probabilities (`verdict_conflict`).
   - Calls Gemini (`gemini-4-argon` by default) to produce a strict JSON verdict (`CORRECT`, `INCORRECT`, `AMBIGUOUS`), or returns `ABSTAIN` when escalation is disabled (`H`) or remote calls are disallowed (`--local-only`).

---

## Example Evidence Packet

This is exactly what gets sent to the TypeSafe Jev judge for a real `sql_only` item (produced by actually calling `build_packet` — not hand-written). The question asks for each customer's total spend; the reference query returns 2 rows, and the candidate query (which additionally picks up a customer with no matching orders rows, i.e. `Carol`) returns a superset of 3 rows with renamed columns:

- **Reference SQL**: `SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.name` → `[("Alice", 120.50), ("Bob", 75.25)]`
- **Candidate SQL**: `SELECT c.name AS customer_name, SUM(o.amount) AS total_spent FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.name` → `[("Alice", 120.50), ("Bob", 75.25), ("Carol", 10.0)]`

Running this through `build_packet(cand, "sql_only", "jev", cfg)` produces a packet with `n_cells=12`, `n_tokens=612`, well under the `jev_max_cells=400` / `jev_max_tokens=6,000` caps, so no sections get trimmed. The exact JSON body sent inside the `<<DATA>>` / `<</DATA>>` framing markers (see `renderer.py` — the framing text and markers tell the model "this is data, not instructions", and all `<`/`>`/`&` characters inside are escaped so nothing in the question, SQL, or table data can forge a fake closing marker) is:

```json
{
  "QUESTION": "Which customers have placed orders, and what is their total amount spent?",
  "HINT": null,
  "SCHEMA_DDL": {
    "validity_note": "Valid data means any rows SQLite would accept for these statements with default settings: NOT NULL, PRIMARY KEY and UNIQUE (NULLs are distinct), CHECK (a NULL result passes) and STRICT column types are enforced; an INTEGER PRIMARY KEY is never NULL; FOREIGN KEY is not enforced; outside STRICT tables a declared type is only an affinity, so any value type can occur.",
    "ddl": "CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT);\n\nCREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);"
  },
  "REFERENCE_SQL": {
    "raw": "SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.name",
    "normalized": "SELECT c.name, SUM(o.amount) FROM customers AS c JOIN orders AS o ON o.customer_id = c.id GROUP BY c.name"
  },
  "GENERATED_SQL": {
    "raw": "SELECT c.name AS customer_name, SUM(o.amount) AS total_spent FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.name",
    "normalized": "SELECT c.name AS customer_name, SUM(o.amount) AS total_spent FROM customers AS c JOIN orders AS o ON o.customer_id = c.id GROUP BY c.name"
  },
  "RESULT_SUMMARY": {
    "gt": {"available": true, "error_kind": null, "complete": true, "n_fetched": 2, "total_rows": 2, "rd_eligible": true},
    "gen": {"available": true, "error_kind": null, "complete": true, "n_fetched": 3, "total_rows": 3, "rd_eligible": true},
    "relation": "gt_subset_gen",
    "map_status": "partial",
    "order_ok": null,
    "scale_match": null,
    "coincidence_note": null,
    "volatile_note": null
  },
  "ROW_DIFF": {
    "rows_matched": 2,
    "witness_kind": "key_aligned",
    "gt": {"only": 0, "n_rows": 2, "stats": [{"min": "Alice", "max": "Bob", "sum": null, "distinct": 2}, {"min": 75.25, "max": 120.5, "sum": 195.75, "distinct": 2}], "witnesses": []},
    "gen": {"only": 1, "n_rows": 3, "stats": [{"min": "Alice", "max": "Carol", "sum": null, "distinct": 3}, {"min": 10.0, "max": 120.5, "sum": 205.75, "distinct": 3}], "witnesses": [{"key": ["Carol"], "gen": [10.0]}]}
  },
  "SMALL_RESULTS": {
    "gt": {"columns": ["customer_name", "total_spent"], "rows": [["Alice", 120.5], ["Bob", 75.25]]},
    "gen": {"columns": ["customer_name", "total_spent"], "rows": [["Alice", 120.5], ["Bob", 75.25], ["Carol", 10.0]]}
  }
}
```

A few things worth pointing out about what's actually in here:
- **`SCHEMA_DDL`** is the literal `CREATE TABLE` text pulled from `sqlite_master` for the two tables referenced by either query (see Stage 1 above) — the judge sees column names *and* types/constraints, not just a bare column list.
- **`RESULT_SUMMARY.relation: "gt_subset_gen"`** is the code-computed alignment verdict: every reference row was found inside the candidate's rows, but the candidate has an extra row the reference doesn't. `map_status: "partial"` means the column bijection matched by position/type since the names differ (`name`→`customer_name`, `SUM(o.amount)`→`total_spent`).
- **`ROW_DIFF`** gives the judge a *bounded* summary of the mismatch instead of two full tables: counts of rows only-on-one-side, per-column min/max/sum/distinct-count stats for context, and up to one example "witness" row per side showing a concrete row that didn't match (here, `Carol`'s row, which only the candidate produced).
- **`SMALL_RESULTS`** includes the full tables verbatim because both are small enough (≤20 rows) to fit the budget; for larger results this section is dropped first and replaced by `PREFIX_ROWS` (a capped preview) if the packet would otherwise exceed `jev_max_cells`/`jev_max_tokens`.
- Nothing here is a judgment call — `Q1`/`Q2`/`Q5` (Stage 3) are the first point where a model is asked "does this still count as correct given the extra row?"; everything above is produced by deterministic code in `alignment.py`, `comparator.py`, and `sql_analysis.py`.

**What determines which questions get asked.** The packet alone never produces a verdict — `questions_for(mode, cand)` in `judge.py` looks at `cand.alignment.relation` (the exact value shown in `RESULT_SUMMARY.relation` above) and decides which rubric questions go into the Stage 3/4 request. `align_tables()` (`alignment.py`) can produce six possible `relation` values:

| `relation` | What it means | Sent to Jev/escalation? |
| --- | --- | --- |
| `equal` | Row sets match exactly (after tolerance/scale matching) | `Q1`, `Q5` only — the row sets already agree, so there's nothing for `Q2` to adjudicate |
| `gt_subset_gen` | Every reference row is present in the candidate's rows, plus the candidate has extra rows (**this example**) | `Q1`, **`Q2`**, `Q5` |
| `gen_subset_gt` | Every candidate row is present in the reference's rows, but the candidate is missing some | `Q1`, **`Q2`**, `Q5` |
| `overlap` | Some rows match on both sides, some don't | `Q1`, **`Q2`**, `Q5` |
| `disjoint` | No rows match at all | `Q1`, `Q5` — such a strong signal of a wrong result that `Q2`'s nuance isn't needed; items like this are also likely to already be caught by rule `D2`/`D4` |
| `unknown` | Alignment couldn't be computed (an incomplete/missing table on one side, or the row-matching search exceeded its size limit) | Items with `unknown` alignment normally never reach Stage 3 at all — `cand.no_evidence`/incomplete-evidence checks in `judge_item` route them to `ABSTAIN` first |

So for this example, `relation == "gt_subset_gen"` is exactly why the packet is scored against three questions (`Q1`, `Q2`, `Q5`) rather than two — see [**Rubric Questions**](#rubric-questions-q1q6-v) below for what each one actually asks.

---

## Rubric Questions (`Q1`–`Q6`, `V`)

Every question sent to TypeSafe Jev is a fixed, hash-verified instruction string defined once in `rubrics.py` (`RUBRICS`) — the same text is reused verbatim for Stage 4 generative escalation (see **Escalation reuses the same questions** below), so a `CORRECT`/`INCORRECT` grading standard never silently differs between Jev and the escalation LLM. Two question *types* exist:
- **`noul`** ("number, unbounded likelihood") questions ask for a single calibrated probability that the answer is "yes".
- **`choice`** questions ask for a probability distribution over a fixed, mutually-exclusive set of labelled options (`criteria`).

| Question | Used in | Type | Asks |
| --- | --- | --- | --- |
| **`Q1`** `same_intent` | `sql_only` (always) | `noul` | Does the generated SQL compute the *same thing* as the reference SQL — same measures, filters, time window, grouping, and joins — on **any** valid database matching the schema, not just the rows actually shown? Column names, column order, extra non-row-affecting columns, formatting, and rounding are explicitly ignored. |
| **`Q2`** `row_difference` | `sql_only`, only when `relation` is `gt_subset_gen` / `gen_subset_gt` / `overlap` | `choice` | Given that the two result sets differ (see `ROW_DIFF`), which best describes the generated result: `equivalent_for_question` (the difference doesn't change the answer), `acceptable_extra_rows` (has every required row, plus rows the question doesn't exclude), `missing_required_rows`, or `wrong_rows`? |
| **`Q3`** `answer_true` | `full_trace` (always) | `noul` | Is every factual statement in the agent's final answer actually true of the (complete) reference result, with each value correctly attached to its entity, measure, and time period? The judge is told to use the harness's own `CLAIMS`/`DERIVATIONS` rather than recomputing aggregates itself. |
| **`Q4`** `answer_complete` | `full_trace` (always) | `noul` | Does the final answer state *every* specific value/entity the question asks for — not a subset, and not just a description of the method used to get there? |
| **`Q5`** `ambiguity` | both modes (always) | `choice` | Could the question reasonably be read more than one way, and if so, is the generated work's interpretation one of the reasonable ones? Options: `not_ambiguous`, `ambiguous_generated_reasonable`, `ambiguous_generated_unreasonable`. |
| **`Q6`** `answer_supported` | `full_trace` (always) | `noul` | Is every value/entity in the final answer actually shown in, or derivable from, the agent's own intermediate SQL step results (not just coincidentally equal to the reference result)? This is deliberately independent of `Q3`/correctness — an answer can agree with the reference by luck while still being unsupported by what the agent's own queries actually returned. |
| **`V`** `verdict` | `B6`/`B6+E` single-question baseline only | `choice` | A single collapsed question — is the generated work correct overall? — used in place of the structured `Q1`–`Q6` breakdown, with options `correct`, `incorrect`, `ambiguous_reasonable`. |

`sql_only` items are therefore graded on `{Q1, [Q2], Q5}` and `full_trace` items on `{Q3, Q4, Q6, Q5}` (see `questions_for()` above); `Q2`'s inclusion is the one conditional case, gated on the `relation` table above.

**Escalation reuses the same questions.** When an item escalates to the generative model (Stage 4), `judge_item` computes the *same* `questions_for(mode, cand)` list and passes it to `escalation_instructions()` (`rubrics.py`), which stitches each selected question's exact `instructions` text (and `criteria` option names, for `choice` questions) into the escalation prompt, followed by a fixed instruction to return strict JSON (`{"label": "CORRECT"|"INCORRECT"|"AMBIGUOUS", "rationale": "<=80 words"}`). So a `gt_subset_gen` item escalated to Gemini is asked to reason about exactly `Q1` + `Q2` + `Q5`, worded identically to what Jev would have seen — the only difference is Jev returns calibrated per-question probabilities mapped through fixed thresholds (`sql_only_verdict_table`/`full_trace_verdict_table` in `judge.py`), while escalation returns one collapsed label directly.

---

## Evaluation Modes & Methods

### Modes
- **`sql_only`** (`trace.final_answer is None`): Grades a single generated SQL statement against the reference query chain using deterministic rules `D1`–`D5`, alignment, and rubric questions `Q1`, `Q2`, `Q5`.
- **`full_trace`** (`trace.final_answer is not None`): Grades an agent trace (`steps` + `final_answer`) by checking intermediate SQL steps (`method_ok`), numeric/date claim bindings (`claims.py`), code-computed aggregates (`DERIVATIONS`), rule `D0`/`D2`, and rubric questions `Q3`, `Q4`, `Q6`, `Q5`.

### Supported Methods
All methods can be invoked programmatically (`judge_item` or `baselines.py`) or via `txt2sql-judge run --methods ...`:

| Method | Description |
| --- | --- |
| `H+E` | Full hybrid judge: Stage 0–1 deterministic funnel + TypeSafe Jev (`Q1`–`Q6`) + selective Gemini escalation |
| `H` | Deterministic funnel + TypeSafe Jev without escalation (uncertain/oversized cases yield `ABSTAIN`; derived offline from `H+E` records via `derive_h_from_he`) |
| `B6` | Hybrid funnel replacing the structured `Q1`–`Q6` rubric with single-question Jev grading (`V`) without escalation |
| `B6+E` | Single-question Jev (`V`) baseline with generative escalation enabled |
| `B1` | Exact normalized SQL text match (grades `sql_only` as `CORRECT`/`INCORRECT`; abstains in `full_trace` or when empty) |
| `B2` | BIRD-style execution multiset equality (`_canonical_bird_scalar` rounding floats to 6 decimals; evaluates final step SQL result) |
| `B-IBM-subset` | Execution subset/superset heuristic with column permutation matching |
| `B-IBM-judge` | Prompted generative baseline grading formatted Markdown result tables and trace steps |
| `B4` | Frontier generative judge (`gemini-4-argon`) with binary-search prefix truncation fitting within `b4_max_tokens=100,000` |

---

## Datasets & Synthetic Benchmark Suites

The harness integrates pinned datasets and deterministic synthetic suites (`data.py` & `generators.py`):

- **Pinned Upstream Datasets (`txt2sql-judge download`)**:
  - **BIRD Mini-Dev V2**: Pinned to revision `5b256b5a9c17478f06ed6912612e9a14d17065c8` of `birdsql/bird_mini_dev` (`mini_dev_sqlite-00000-of-00001.json` + SQLite databases verified against SHA-256 checksums in `data/manifest.json`).
  - **FLEX**: Pinned commit `270abdba59306134adf0928b2ce92acb8fbd94eb` providing `sql_only` expert-annotated items (`S4`, loaded via `load_flex_dataset`). Multi-step `S5` traces can be synthesized and tagged programmatically via the `agent.py` library helpers (`run_reference_agent`, `tag_s5_trace`).
- **Synthetic & Challenge Suites (`txt2sql-judge generate`)**:
  - **`S1`**: Meaning-preserving SQLite AST rewrites (`add_columns`, `permute_columns`, `drop_order_by`, `between_expand`) labeled `correct`.
  - **`S2`**: Question-anchored faulty SQL mutants (`filter_literal`, `comparator_flip`, `drop_conjunct`, `agg_swap`, `time_shift`) verified by exhaustive table diff on the real SQLite database at relative tolerance `rho = 1e-4`.
  - **`S3`**: Templated `full_trace` answers derived from complete ground-truth tables (`correct`, wrong number/entity/omission perturbations, unsupported claims) bounded to eligible tables with 1–20 rows and unique entity keys.
  - **`S3m`**: Multi-step traces pairing exploratory or mistaken intermediate queries with right/wrong SQL and right/wrong final answers (`right_sql_right_ans`, `wrong_sql_wrong_ans`, `misleading_trace`, `good_sql_wrong_ans`, `unsupported_right_ans`).
  - **`E5b`**: Injection-resistance paired items embedding fixed benign carrier text (`E5B_PAYLOAD`) into final answers without altering SQL semantics, data facts, or numeric values.
- **Leakage-Free Splits & Common Subset `C`**:
  - Items are assigned deterministically to `calibration` or `test` splits (50/50) keyed by `group_id = sha256(db_id + "\x1f" + norm(question))[:16]`, enforced via `assert_disjoint_splits`.
  - `select_common_subset_c` samples a balanced evaluation subset `C` stratified by family with deterministic hashing.
  - `fit_thresholds` and `project_spend_and_shrink_c` (`runner.py`) are available as Python library helpers for offline threshold calibration and budget projection (not invoked automatically by `txt2sql-judge run`).

---

## Installation

Requires **Python >= 3.10**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Key runtime dependencies defined in [`pyproject.toml`](pyproject.toml): `sqlglot`, `tiktoken`, `httpx`, `google-genai`, `pandas`, `numpy`, `matplotlib`, and `pyyaml`.

### Environment Variables

Remote calls are gated and lazy-initialized inside `client_for` (`clients.py`):

- **`TYPESAFE_API_KEY`**: Bearer token for TypeSafe Jev requests (`https://api.typesafe.ai/v1/systemone` by default). Required for Jev stages (`H+E`, `H`, `B6`, `B6+E`) unless running `--local-only` or serving all responses from cache.
- **`GEMINI_API_KEY`**: API key passed to `google-genai` for generative escalation (`H+E`, `B6+E`), `B4`, `B-IBM-judge`, and agent trace synthesis (`agent.py`). Required when those stages invoke uncached remote calls.

---

## CLI Usage

Installing the package registers the `txt2sql-judge` console script (`txt2sql_eval_harness.cli:main`).

### 1. Download Datasets

```bash
txt2sql-judge download --data-dir data
```

Downloads BIRD Mini-Dev V2 and FLEX into `--data-dir` (default: `data`) and writes `data/manifest.json` with SHA-256 digests used by the egress admission gate.

### 2. Generate Synthetic & Adversarial Benchmarks

```bash
txt2sql-judge generate \
  --data-dir data \
  --out data/items.jsonl \
  --e5b-out data/e5b_items.jsonl
```

Loads BIRD Mini-Dev + FLEX, generates `S1`, `S2`, `S3`, `S3m`, tags evaluation slices, creates paired `E5b` carrier items, and prints emission/discard counts as JSON.

### 2b. Generate & Assess LangChain Text2SQL Agent Traces (Optional `[langchain]` extra)

Install the optional LangChain extra (`pip install -e ".[dev,langchain]"`), then run LangChain's SQL agent against sampled benchmark questions with any Gemini model (`GEMINI_API_KEY`):

```bash
txt2sql-judge generate-langchain \
  --dataset bird \
  --model gemini-2.5-flash \
  --limit 20 \
  --out data/langchain_gemini.jsonl
```

This records each `sql_db_query` step via `execute_sandboxed` and the natural-language `final_answer` as `S5` (`label="unlabelled"`) items, and registers the file in `data/manifest.json` so you can immediately score and report on the agent:

```bash
txt2sql-judge run --items data/langchain_gemini.jsonl --run-dir runs/lc_gemini --methods H+E B2 B4
txt2sql-judge report --items data/langchain_gemini.jsonl --records runs/lc_gemini/records.jsonl --out-dir runs/report_lc_gemini
```

### 3. Run Evaluation Methods

```bash
txt2sql-judge run \
  --items data/items.jsonl \
  --root experiments/default \
  --run-dir runs/default \
  --db-dir data/bird/databases \
  --manifest data/manifest.json \
  --methods H+E B6 B6+E B4 B1 B2 B-IBM-subset B-IBM-judge \
  --cap-usd 150.0
```

Useful flags for offline development, determinism checks, and budget control:
- `--local-only`: Disables all remote HTTP/SDK egress and skips API key lookup; deterministic rules still run while model-dependent items return `ABSTAIN` (`remote_disabled`).
- `--cache-only`: Serves hits from the SQLite `ResponseCache` in `--root` and abstains (`cache_miss`) instead of hitting remote APIs.
- `--enable-d5`: Enables rule `D5` (30% row-deletion database copy counterexample checks).
- `--subset-c`: Restricts execution to the deterministic common subset `C` (`select_common_subset_c`).
- `--cap-usd`: Hard spend limit enforced atomically by `SpendLedger` (default: `150.0`). Resuming into an existing `--run-dir` validates `RunIdentity` (`git_sha`, `config_hash`, `prompt_hash`, `items_sha256`) and continues where the prior run left off.

### 4. Generate Reports & Figures

```bash
txt2sql-judge report \
  --items data/items.jsonl \
  --records runs/default/records.jsonl \
  --out-dir runs/report \
  --spend runs/default/spend.json \
  --bootstrap-resamples 1000
```

Outputs `tables.json` (containing `T1`–`T6` and rule summary `E2_rules`) plus 5 publication figures in `--out-dir`:
- `F2_cost_tradeoff.png`: False-pass and strict correct-pass rate vs. log billed cost per trace.
- `F3_coincident.png`: Performance on coincidentally matching SQL queries.
- `F4_large_tables.png`: Performance on truncated / large-table items.
- `F5_final_answers.png`: False-pass rate on `S3` and `S5` full-trace answers.
- `F6_jev_probabilities.png`: Histogram of Jev `p(true)` scores across correct vs. incorrect items.

---

## Python API Quickstart

You can call `judge_item` directly without network access (`allow_remote=False`) to inspect Stage 0–2 deterministic verdicts, table alignment, and extracted claims:

```python
from txt2sql_eval_harness import (
    EvalConfig,
    GroundTruth,
    ResultTable,
    Step,
    Trace,
)
from txt2sql_eval_harness.clients import CallContext
from txt2sql_eval_harness.judge import judge_item

gt = GroundTruth(
    item_ref="ex_1",
    sql_chain=["SELECT COUNT(*) AS n FROM orders WHERE status = 'shipped'"],
    results=[ResultTable(columns=["n"], rows=[(42,)], total_rows=1, complete=True)],
)

trace = Trace(
    trace_id="trace_1",
    group_id="grp_1",
    question="How many shipped orders are there?",
    hint=None,
    db_id="shop",
    db_sha256="dummy",
    steps=[
        Step(
            index=1,
            sql="SELECT COUNT(*) FROM orders WHERE status = 'shipped'",
            result=ResultTable(columns=["COUNT(*)"], rows=[(42,)], total_rows=1, complete=True),
        )
    ],
)

ctx = CallContext(cfg=EvalConfig(local_only=True), allow_remote=False)
verdict = judge_item(trace, gt, method="H+E", ctx=ctx)

print(verdict.label)       # -> "CORRECT"
print(verdict.decided_by)  # -> "rule:D1"
```

When remote APIs are enabled (`allow_remote=True` with `ResponseCache` and `SpendLedger`), `judge_item` seamlessly progresses through Jev rubric scoring and Gemini escalation while recording exact cold latency/cost (`calls`, `cost_billed_usd`, `cost_conservative_usd`) and cache hit provenance.

---

## Safety, Reproducibility & Spend Controls

- **Read-Only Sandboxed SQL Execution (`sandbox.py`)**: Connections open with `mode=ro&immutable=1`, `enable_load_extension(False)`, strict `set_authorizer` read allowlist (`SELECT`, `READ`, `FUNCTION`, `PRAGMA` allowlist, etc.), `set_progress_handler` deadline enforcement, and `PRAGMA hard_heap_limit`.
- **Dataset Egress Admission Gate (`data.py`)**: `egress_decision` ensures remote calls are only permitted on input files verified in `data/manifest.json` (with matching whole-file SHA-256 and `kind` in `{"pinned_download", "harness_derivative"}`), blocking unverified files from being sent to model APIs.
- **Deterministic Request Cache (`cache.py`)**: `ResponseCache` stores only validated `"ok"` responses keyed by canonical request SHA-256 (`canonical_request_sha256`), preserving cold wall time and cold token/USD cost on cache replays.
- **Conservative Spend Accounting (`spend.py`)**: `SpendLedger` reserves worst-case upper-bound USD before every remote request and persists atomic state to disk (`spend.json`) so timeouts/disconnections (`billing="ambiguous"`) count against the budget rather than risking overspend.
- **Prompt & Framing Integrity (`rubrics.py` & `renderer.py`)**: All prompt constants and rubric hashes (`PROMPT_HASH`) pass ASCII-only / framing-token integrity checks (`verify_prompt_constants`) so model-visible instructions cannot be forged by untrusted table text or answers.

---

## SQL Dialect Scope & Extensibility

Currently, the harness **targets SQLite exclusively**:
- `sql_analysis.py` (`parse_single_sqlite`, `normalize_sql_n`, `is_volatile`, `extract_schema_ddl`, `create_d5_deletion_copy`) parses and normalizes with `sqlglot(..., read="sqlite")`, queries `sqlite_master`, and assumes SQLite type affinity / constraint rules in `VALIDITY_NOTE`.
- `sandbox.py` executes against local `.sqlite` files via Python's built-in `sqlite3`.
- `comparator.py` and `renderer.py` restrict `Cell` types to SQLite's five storage classes (`None`, `int`, `float`, `str`, `bytes`), explicitly rejecting `bool`.

Adapting the harness to compare SQL across multiple dialects (e.g., PostgreSQL, DuckDB, MySQL, BigQuery, Snowflake) requires adding dialect configuration on `Trace`/`GroundTruth`/`EvalConfig`, transpile-or-execute adapters in `sql_analysis.py` and `sandbox.py`, and richer storage-type mapping in `comparator.py` and `renderer.py`.

---

## Package Structure

### Source Modules (`src/txt2sql_eval_harness/`)
- [`__init__.py`](src/txt2sql_eval_harness/__init__.py) — Top-level package exports and version.
- [`agent.py`](src/txt2sql_eval_harness/agent.py) — Gemini-backed `full_trace` multi-step SQL agent synthesis.
- [`alignment.py`](src/txt2sql_eval_harness/alignment.py) — Column bijection search, multiset/ordered row matching, and bounded row diffs.
- [`baselines.py`](src/txt2sql_eval_harness/baselines.py) — Baseline evaluators (`B1`, `B2`, `B-IBM-subset`, `B-IBM-judge`, `B4`).
- [`cache.py`](src/txt2sql_eval_harness/cache.py) — SQLite write-once `ResponseCache` preserving cold call measurements.
- [`claims.py`](src/txt2sql_eval_harness/claims.py) — Numeric/date/aggregate claim extraction, cell binding, `DERIVATIONS`, and `check_d0`/`check_d2`.
- [`cli.py`](src/txt2sql_eval_harness/cli.py) — `txt2sql-judge` CLI (`download`, `generate`, `run`, `report`).
- [`clients.py`](src/txt2sql_eval_harness/clients.py) — `JevHttpClient`, `GeminiGenerativeClient`, retry/cost policy, and egress-gated `call_model`.
- [`codec.py`](src/txt2sql_eval_harness/codec.py) — Exact JSONL codecs preserving numeric distinctions (`int` vs. `float`) and `RunIdentity`.
- [`comparator.py`](src/txt2sql_eval_harness/comparator.py) — SQLite cell classification, deterministic total sort key, and relative tolerance comparison.
- [`data.py`](src/txt2sql_eval_harness/data.py) — Dataset downloader, SHA-256 manifest, split assignment, and BIRD/FLEX loaders.
- [`generators.py`](src/txt2sql_eval_harness/generators.py) — Generators for `S1`, `S2`, `S3`, `S3m`, and `E5b` plus invariant validation.
- [`judge.py`](src/txt2sql_eval_harness/judge.py) — 5-stage hybrid funnel (`judge_item`), deterministic rules `D0`–`D5`, verdict tables, and `derive_h_from_he`.
- [`metrics.py`](src/txt2sql_eval_harness/metrics.py) — Coverage, false-pass, strict/lenient correct-pass, bootstrap CIs, threshold fitting, and `h6_reading`.
- [`models.py`](src/txt2sql_eval_harness/models.py) — Core dataclasses (`Trace`, `GroundTruth`, `ResultTable`, `Verdict`, `EvalConfig`, etc.).
- [`packet.py`](src/txt2sql_eval_harness/packet.py) — Canonical evidence packet builder with automatic witness/row trimming under token/cell caps.
- [`renderer.py`](src/txt2sql_eval_harness/renderer.py) — Strict framed prompt renderer (`OPEN`/`CLOSE`/`FRAMING`) and `tiktoken` counting.
- [`reporting.py`](src/txt2sql_eval_harness/reporting.py) — Report generator (`T1`–`T6`, `E2_rules`) and matplotlib figures (`F2`–`F6`).
- [`rubrics.py`](src/txt2sql_eval_harness/rubrics.py) — Fixed Jev rubric definitions (`Q1`–`Q6`, `V`), prompt templates, and `PROMPT_HASH`.
- [`runner.py`](src/txt2sql_eval_harness/runner.py) — Common subset `C` selection, threshold calibration, spend projection, and resumable experiment loop.
- [`sandbox.py`](src/txt2sql_eval_harness/sandbox.py) — Read-only immutable SQLite execution sandbox and opcode authorizer.
- [`spend.py`](src/txt2sql_eval_harness/spend.py) — Persistent JSON `SpendLedger` with pre-send reservations and cap checks.
- [`sql_analysis.py`](src/txt2sql_eval_harness/sql_analysis.py) — SQLite parsing (`sqlglot`), normalization `N`, volatility detection, `Contract`, and `D5` copy creation.

### Test Suite (`tests/`)
- [`test_alignment_and_diff.py`](tests/test_alignment_and_diff.py) — Table alignment, column bijections, scale matching, and row-diff invariants.
- [`test_baselines_and_metrics.py`](tests/test_baselines_and_metrics.py) — Baseline judges, metric calculations, threshold calibration, and `T1`–`T6`/`F2`–`F6` reporting.
- [`test_claims_and_generators.py`](tests/test_claims_and_generators.py) — Claim binding, derivations, and `S1`/`S2`/`S3`/`S3m`/`E5b` synthetic generators.
- [`test_core_and_sql.py`](tests/test_core_and_sql.py) — Codec round-trips, SQL normalization `N`, volatility checks, sandbox isolation, and `D5` copies.
- [`test_renderer_and_packets.py`](tests/test_renderer_and_packets.py) — Packet construction, token/cell trimming, prompt framing guarantees, and `PROMPT_HASH`.
- [`test_runner_spend_and_security.py`](tests/test_runner_spend_and_security.py) — Egress gates, cache write-once semantics, spend reservations/caps, resumability, and hybrid funnel integration.

---

## Running Tests

Run the offline acceptance test suite (uses local SQLite fixtures and deterministic transport fakes; requires no remote API keys):

```bash
.venv/bin/pytest -v
```
