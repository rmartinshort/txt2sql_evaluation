# Text2SQL Decision-Model Judge Harness (`txt2sql-eval-harness`)

A deterministic-first hybrid evaluation harness for judging Text-to-SQL outputs and multi-step SQL agent traces against ground-truth references. It combines sandboxed SQL execution, syntactic/semantic verification rules, exact column/row table alignment, and numerical claim binding with calibrated probabilistic evaluation via **TypeSafe Jev** (`jev-1.14.0`) and selective generative escalation (`gemini-4-argon`).

---

## Overview & 5-Stage Hybrid Judge Funnel

Evaluating Text-to-SQL predictions on execution equality alone often misses subtle SQL bugs (coincidental result agreement on a single database, dropped ordering constraints, duplicate/missing rows, or unverified extra columns) and cannot grade multi-step agent traces that end in natural-language answers. Conversely, sending raw full tables and prompts directly to a frontier LLM judge on every example is expensive and brittle on large outputs or numeric rounding.

This harness solves both problems with a staged verification funnel (`judge_item` in `src/txt2sql_eval_harness/judge.py`):

1. **Stage 0 — Sandbox Execution & Schema Extraction**:
   - Validates input size limits (`max_steps=8`, `max_sql_bytes=32 KiB`, `max_answer_chars=20,000`, `max_columns=64`).
   - Executes missing ground-truth and candidate SQL steps inside a read-only, immutable SQLite sandbox (`execute_sandboxed` in `sandbox.py`) enforcing an opcode authorizer, progress timeout (`sql_timeout_s=10.0`), heap limit (`256 MiB`), and bounded row/byte fetch caps (`200,000` rows / `16 MiB`).
   - Picks the best ground-truth SQL alternative by determinism, execution status, and table alignment.

2. **Stage 1 — Deterministic Analysis, Alignment, Claims & Rules (`D0`–`D5`)**:
   - Infers a question-derived must-match `Contract` (`order_required`, `limit_required`) and extracts referenced schema DDL (`sql_analysis.py`).
   - Computes an exact column bijection and row alignment/diff between ground-truth and candidate `ResultTable` outputs (`alignment.py` & `comparator.py`), including unit-scale (`x100` / `x0.01`) and relative numerical tolerance (`rho = 1e-9`).
   - Extracts and binds numerical, date, and aggregate claims from agent `final_answer` text against execution tables (`claims.py`).
   - Applies deterministic short-circuit rules before any remote model call:
     - **D0 (`rule:D0`)**: Immediate `INCORRECT` in `full_trace` mode if `final_answer` is missing or empty.
     - **D3 (`rule:D3`)**: Immediate `INCORRECT` in `sql_only` mode on SQL compile or non-read authorization errors.
     - **D2 (`rule:D2`)**: Immediate `INCORRECT` if the generated query returns no usable result/data against non-empty reference results.
     - **D4 (`rule:D4`)**: Immediate `INCORRECT` scalar counterexample when deterministic 1x1 numeric queries disagree across unit and percentage scales.
     - **D5 (`rule:D5`, opt-in via `--enable-d5` / `enable_d5=True`)**: Runs both deterministic queries on a seeded 30% row-deletion database copy (`create_d5_deletion_copy`) to catch coincidental equality via a counterexample database.
     - **D1 (`rule:D1`)**: Immediate `CORRECT` when both ground-truth and candidate SQL are deterministic and share identical presentation-normalized SQL (`normalize_sql_n`), provided complete execution results do not contradict equality.

3. **Stage 2 — Bounded Canonical Evidence Packet (`packet.py` & `renderer.py`)**:
   - Packs `QUESTION`, `HINT`, `SCHEMA_DDL`, normalized/original SQL, execution errors, alignment relation, bounded witness rows/column stats, bound claims, code-computed `DERIVATIONS`, and flags into a token-budgeted canonical JSON packet (`jev_max_cells=400`, `jev_max_tokens=6,000`, with a larger tier `escalation_max_cells=1,200`, `escalation_max_tokens=16,000` for escalation).

4. **Stage 3 — TypeSafe Jev Calibrated Scoring (`clients.py` & `rubrics.py`)**:
   - Evaluates structured rubric questions via TypeSafe Jev (`jev-1.14.0`):
     - **`sql_only` mode**: `Q1` (`same_intent`), conditional `Q2` (`row_difference` when alignment relation is `gt_subset_gen`, `gen_subset_gt`, or `overlap`), and `Q5` (`ambiguity`).
     - **`full_trace` mode**: `Q3` (`answer_true`), `Q4` (`answer_complete`), `Q6` (`answer_supported`), and `Q5` (`ambiguity`).
   - Maps calibrated probabilities against abstain thresholds (`ThresholdPair(tau_hi=0.85, tau_lo=0.15)` by default) into `CORRECT`, `INCORRECT`, `AMBIGUOUS`, or `ESCALATE`.

5. **Stage 4 — Selective Generative Escalation**:
   - Invoked when evidence exceeds Jev bounds, Jev returns an uncertain probability band, or execution evidence conflicts with Jev probabilities (`verdict_conflict`).
   - Calls Gemini (`gemini-4-argon` by default) to produce a strict JSON verdict (`CORRECT`, `INCORRECT`, `AMBIGUOUS`), or returns `ABSTAIN` when escalation is disabled (`H`) or remote calls are disallowed (`--local-only`).

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
