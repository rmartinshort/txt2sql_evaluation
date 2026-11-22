# Method Walkthroughs: Two Real Examples

This guide walks through how every evaluation method in `txt2sql-eval-harness` processes two concrete items from `data/items.jsonl` (both based on `bird_minidev:1471` in `debit_card_specializing`):

1. **Example 1 (`sql_only`)**: `S2__bird_minidev:1471__agg_swap__0` — a single generated SQL query with a subtle aggregation bug (`COUNT(IIF(...))` instead of `SUM(IIF(...))`).
2. **Example 2 (`full_trace`)**: `S3m__bird_minidev:1471__early_support` — a 2-step agent trace where Step 1 computes the right answer, Step 2 runs an unrelated query, and `final_answer` reports the Step 1 result.

**`H+E`** is the flagship hybrid method and is covered in full 5-stage detail for both examples, followed by how each of the other 8 methods (`H`, `B6`, `B6+E`, `B1`, `B2`, `B-IBM-subset`, `B-IBM-judge`, `B4`) behaves on the same inputs.

---

## Example 1 (`sql_only`): `S2__bird_minidev:1471__agg_swap__0`

- **Question**: *"What is the ratio of customers who pay in EUR against customers who pay in CZK?"*
- **Hint**: *"ratio of customers who pay in EUR against customers who pay in CZK = count(Currency = 'EUR') / count(Currency = 'CZK')."*
- **Ground Truth SQL**:
  ```sql
  SELECT CAST(SUM(IIF(Currency = 'EUR', 1, 0)) AS FLOAT) / SUM(IIF(Currency = 'CZK', 1, 0)) AS ratio
  FROM customers
  ```
  - **GT Result**: `1x1` table `ratio = 0.06572769953051644`
- **Candidate SQL**:
  ```sql
  SELECT CAST(SUM(IIF(Currency = 'EUR', 1, 0)) AS REAL) / COUNT(IIF(Currency = 'CZK', 1, 0)) AS ratio
  FROM customers
  ```
  - **Candidate Result**: `1x1` table `ratio = 0.06167400881057269` (`COUNT(IIF(..., 1, 0))` counts `0`s as non-null rows, inflating the denominator).
- **True Label**: `incorrect` (`S2:agg_swap`)

### Flagship Method: `H+E` (5-Stage Funnel)

1. **Stage 0 — Sandbox Execution & Validation (`judge.py`, `sandbox.py`)**
   - Checks input limits (`max_steps=8`, `max_sql_bytes=32 KiB`, `max_columns=64`).
   - Both GT and candidate already have complete `1x1` `ResultTable`s attached (if missing, `execute_sandboxed` runs them in a read-only, immutable SQLite connection with a 10s timeout and 256 MiB heap limit).
2. **Stage 1 — Deterministic Analysis, Alignment & Rules (`sql_analysis.py`, `alignment.py`, `comparator.py`, `judge.py`)**
   - Infers `Contract(order_required=False, limit_required=False)` from the question.
   - Aligns the `1x1` tables (`ratio` ↔ `ratio`).
   - Compares the two scalar values (`0.06572769953051644` vs `0.06167400881057269`) across unit and percentage scales (`x1`, `x100`, `x0.01`) at relative tolerance `rho = 1e-9`. They disagree on all scales.
   - **Rule `D4` fires**: Because both queries are deterministic `1x1` numeric queries that disagree across unit/percentage scales, `H+E` immediately short-circuits and returns:
     - `label = "INCORRECT"`, `decided_by = "rule:D4"`, **0 remote API calls, $0.00 cost**.
3. **Stages 2–4 (Skipped due to `D4`)**
   - No `EvidencePacket` needs to be sent to Jev or Gemini.

### How the Other Methods Handle Example 1

| Method | Behavior on `S2__bird_minidev:1471__agg_swap__0` |
| --- | --- |
| **`H`** | Identical to `H+E` here: short-circuits at `rule:D4` with `INCORRECT` and `$0.00` cost. |
| **`B6` / `B6+E`** | Runs the same Stage 0–1 deterministic funnel, so `rule:D4` also short-circuits to `INCORRECT` before Jev (`V`) or Gemini is called. |
| **`B1`** | Normalizes both SQL strings (`normalize_sql_n`). Since `SUM(IIF(...))` != `COUNT(IIF(...))`, returns `INCORRECT` (`decided_by="baseline"`). |
| **`B2`** | Compares execution multisets with `_canonical_bird_scalar` (6-decimal float rounding): `0.065728 != 0.061674` → returns `INCORRECT`. |
| **`B-IBM-subset`** | Checks subset/superset execution match with column permutation. The single row values differ → returns `INCORRECT`. |
| **`B-IBM-judge`** | Formats the question, SQL, and Markdown tables into the unframed IBM v1.6.0 prompt and calls `gemini-4-argon` (`ibm_judge` stage). |
| **`B4`** | Renders the framed `<<DATA>>` packet (up to `b4_max_tokens=100,000`) and calls `gemini-4-argon` directly to grade. |

---

## Example 2 (`full_trace`): `S3m__bird_minidev:1471__early_support`

- **Question & GT**: Same as Example 1 (`ratio = 0.06572769953051644`).
- **Candidate Trace**:
  - **Step 1**: Runs the correct SQL (`SUM(IIF(Currency = 'EUR', 1, 0)) / SUM(IIF(Currency = 'CZK', 1, 0))`), returning `ratio = 0.06572769953051644`.
  - **Step 2**: Runs an unrelated query on `Segment` and `Consumption` (a distractor/exploratory step).
  - **`final_answer`**: States the ratio from Step 1 (`~0.0657`).
- **True Label**: `correct` (`S3m:early_support`)

### Flagship Method: `H+E` (5-Stage Funnel)

1. **Stage 0 — Sandbox Execution & Validation**
   - Validates `len(steps) == 2 <= max_steps`, `len(final_answer) <= 20,000`, and ensures all step results are populated.
2. **Stage 1 — Deterministic Analysis, Claims & Rules (`claims.py`, `judge.py`)**
   - **`D0` / `D2` checks**: `final_answer` is non-empty (`D0` passes) and Step 1 produced a usable result (`D2` passes).
   - **Claim Extraction & Binding (`claims.py`)**: Extracts the numeric claim (`0.0657...`) from `final_answer` and binds it to Step 1's `ResultTable` cell (`0.06572769953051644`), recording `DERIVATIONS` and `method_ok`.
   - **`D1` / `D4` do not short-circuit**: In `full_trace` mode, `H+E` must verify that `final_answer` actually answers the question accurately and is supported by the trace, so it proceeds to Stage 2.
3. **Stage 2 — Bounded Canonical Evidence Packet (`packet.py`, `renderer.py`)**
   - Assembles `QUESTION`, `HINT`, `SCHEMA_DDL`, `STEPS` (Step 1 + Step 2 SQL and tables), `CLAIMS`, `DERIVATIONS`, and `FLAGS` into a prompt-injection-safe `<<DATA>>` JSON block.
   - Checks `jev_max_cells=400` and `jev_max_tokens=6,000`. If oversized, trims optional sections first, then extra `ROW_DIFF` witnesses.
4. **Stage 3 — TypeSafe Jev Calibrated Scoring (`clients.py`, `rubrics.py`)**
   - Calls `jev-1.14.0` on the `full_trace` rubric questions:
     - `Q3` (`answer_true`): Does `final_answer` match the ground truth?
     - `Q4` (`answer_complete`): Does `final_answer` cover all parts of the question?
     - `Q6` (`answer_supported`): Is `final_answer` supported by the executed trace steps (Step 1)?
     - `Q5` (`ambiguity`): Is the question/evidence ambiguous?
   - Compares calibrated probabilities against `ThresholdPair(tau_hi=0.85, tau_lo=0.15)`.
   - If all probabilities are decisive (e.g., `Q3, Q4, Q6 >= 0.85` and `Q5 <= 0.15`) and don't conflict with deterministic flags, returns `CORRECT` (`decided_by="jev"`).
5. **Stage 4 — Selective Generative Escalation (`clients.py`)**
   - Triggered **only if** the packet exceeded Jev bounds, Jev returned an uncertain probability in `(0.15, 0.85)`, or `verdict_conflict` occurred.
   - Re-packs at the larger escalation tier (`escalation_max_cells=1,200`, `escalation_max_tokens=16,000`) and calls `gemini-4-argon` for a strict JSON verdict (`CORRECT`, `INCORRECT`, or `AMBIGUOUS`).

### How the Other Methods Handle Example 2

| Method | Behavior on `S3m__bird_minidev:1471__early_support` |
| --- | --- |
| **`H`** | Runs Stages 0–3 identically to `H+E`. If Jev is decisive, returns `CORRECT`; if Jev is uncertain or oversized, **abstains** (`ABSTAIN`) instead of escalating to Gemini. |
| **`B6`** | Runs Stages 0–2, then asks Jev a **single** overall question (`V`) instead of `Q3`/`Q4`/`Q6`/`Q5`, without escalation. |
| **`B6+E`** | Same as `B6`, but escalates to `gemini-4-argon` if `V` is uncertain or oversized. |
| **`B1`** | **Abstains** (`unsupported_mode`) — `B1` only supports `sql_only`. |
| **`B2`** | Evaluates **only the final step** (`steps[-1]`, i.e., Step 2's unrelated query) against GT → returns `INCORRECT` (false fail). |
| **`B-IBM-subset`** | Also evaluates **only the final step** (`steps[-1]`) against GT → returns `INCORRECT` (false fail). |
| **`B-IBM-judge`** | Sends the full trace steps + `final_answer` in the unframed IBM v1.6.0 prompt to `gemini-4-argon`. |
| **`B4`** | Sends the framed `<<DATA>>` packet (up to `100,000` tokens) directly to `gemini-4-argon` without Jev pre-scoring. |

---

## Example 3 (Multi-Column Table): `S2__bird_minidev:1457__agg_swap__0`

- **Question**: *"Give the full name and contact number of members who had to spend more than average on each expense."*
- **Ground Truth SQL**:
  ```sql
  SELECT DISTINCT T3.first_name, T3.last_name, T3.phone
  FROM expense AS T1
  INNER JOIN budget AS T2 ON T1.link_to_budget = T2.budget_id
  INNER JOIN member AS T3 ON T3.member_id = T1.link_to_member
  WHERE T1.cost > (SELECT AVG(T1.cost) FROM expense AS T1 ...)
  ```
  - **GT Result (`3x3` table)**:
    - Columns: `["first_name", "last_name", "phone"]`
    - Rows: `("Sacha", "Harrison", "840-555-4781")`, `("Phillip", "Cullen", "905-555-5035")`, `("Elijah", "Allen", "724-555-5721")`
- **Candidate SQL**: Replaces `AVG(T1.cost)` with `SUM(T1.cost)` in the subquery.
  - **Candidate Result (`0x3` table)**:
    - Columns: `["first_name", "last_name", "phone"]`
    - Rows: `[]` (empty, because no single expense exceeds the sum of all expenses)
- **True Label**: `incorrect` (`S2:agg_swap`)

### How Multi-Column Tables Are Assessed (`H+E`)

1. **Stage 1 — Column Bijection & Row Alignment (`alignment.py`, `comparator.py`)**
   - **Column alignment**: `alignment.py` searches for a 1-to-1 column mapping (bijection) between GT columns and candidate columns (tolerant of reordered/aliased columns, up to `_MAX_ASSIGNMENT_SEARCH = 64`). Here, `first_name`, `last_name`, and `phone` map 1-to-1.
   - **Row matching**: Classifies each cell (`null`, `exact_number`, `approx_number`, `text`, `blob`) via `comparator.py` and computes maximum row matching (respecting `Contract.order_required`). Here, GT has 3 rows and candidate has 0 rows, so the alignment relation is `disjoint` / empty candidate.
   - **Rule `D2` fires**: Because the candidate returned 0 rows against a non-empty GT table (`3` rows), `H+E` immediately short-circuits with `INCORRECT` (`decided_by = "rule:D2"`, `$0.00` cost).
2. **What Happens When Both Multi-Column Tables Are Non-Empty & Differ (Stages 2–4)**
   - If the candidate had returned *some* matching rows and *some* differing rows (e.g., `gt_subset_gen`, `gen_subset_gt`, or `overlap`), `D2` would not fire. Instead:
     - `alignment.py` extracts bounded **`ROW_DIFF` witnesses** (example rows only in GT vs. only in candidate) rather than sending the entire tables.
     - `packet.py` packs the alignment relation, column mapping, and `ROW_DIFF` witnesses into `<<DATA>>`, enforcing `jev_max_cells=400` / `jev_max_tokens=6,000` (dropping optional sections first, then trimming extra `ROW_DIFF` witnesses beyond the first per side).
     - **Stage 3 (Jev)** evaluates `Q1` (`same_intent`), `Q2` (`row_difference`), and `Q5` (`ambiguity`); if uncertain or oversized, **Stage 4** escalates to `gemini-4-argon` (`1,200` cells / `16,000` tokens).

### How the Other Methods Handle Example 3

| Method | Behavior on `S2__bird_minidev:1457__agg_swap__0` |
| --- | --- |
| **`H` / `B6` / `B6+E`** | All run Stage 0–1 first, so `rule:D2` short-circuits to `INCORRECT` at `$0.00` cost. |
| **`B1`** | Normalized SQL differs (`AVG` vs `SUM`) → `INCORRECT`. |
| **`B2`** | Compares execution multisets (`3` rows vs `0` rows) → `INCORRECT`. |
| **`B-IBM-subset`** | Checks subset/superset match across column permutations (`0` rows vs `3` rows) → `INCORRECT`. |
| **`B-IBM-judge`** | Formats both multi-column tables as Markdown and sends them to `gemini-4-argon`. |
| **`B4`** | Packs the full `3x3` and `0x3` tables into `<<DATA>>` (truncating rows via binary search only if >`100,000` tokens) and sends to `gemini-4-argon`. |

---

## Exact Python Inputs & Outputs (`judge_item` Offline Run)

Running `judge_item(item.trace, item.gt, method="H+E", ctx=CallContext(cfg=EvalConfig(local_only=True), allow_remote=False))` on the three examples above produces:

```text
S2__bird_minidev:1471__agg_swap__0    -> label=INCORRECT, decided_by=rule:D4,  reason=scalar_counterexample
S2__bird_minidev:1457__agg_swap__0    -> label=INCORRECT, decided_by=rule:D2,  reason=no_data
S3m__bird_minidev:1471__early_support -> label=ABSTAIN,   decided_by=abstain,  reason=remote_disabled
```

- **Why this matters**: Both `S2` SQL bugs are caught deterministically in Stage 1 (`$0.00` cost, 0 LLM calls), while `S3m` (which requires grading natural-language `final_answer` against multi-step SQL evidence) cleanly progresses to Stage 3–4 (and abstains only because `allow_remote=False` in offline mode).
- See [notebooks/he_walkthrough.ipynb](notebooks/he_walkthrough.ipynb) for a runnable, step-by-step Python notebook.
