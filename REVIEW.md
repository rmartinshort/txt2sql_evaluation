# Code Review: `src/txt2sql_eval_harness` & `src/generator_agents`

> **Status Update (2026-11-18):**
> - **Historical harness items resolved:** H1 (`claims.py` case-insensitive multipliers), H2 (`runner.py` skips persisting `internal_error` and logs tracebacks + prefights keys), H3 (`cli.py`/`runner.py` enforce manifest check), H4 (`judge.py` `prepare_best_candidate` ranks by `_RELATION_RANK`), H5 (`runner.py` runs `fit_thresholds` on calibration split; S5 traces produced by `generator_agents`), M3 (`sql_analysis.py` strips comments/literals before `is_volatile`), M7/M8 (`clients.py` reads `EvalConfig` retry/timeout fields and passes `HttpOptions(timeout=...)`), M9 (item-major runner), M11 (`IncompatibleRunDirError` lists mismatched fields), and L1 (`sandbox.py` checks `SQLITE_NOMEM`).
> - **`src/generator_agents/` added:** Consolidates `genie/` (Databricks Genie Agent Mode SSE client, `agents.yaml` allowlist + live description pull, `gemini-decision-1` router, `GenieOrchestrator`, and S5 generator) alongside `langchain/`.
> - **Seam & packet fixes applied:**
>   1. `generator_agents/genie/client.py`: `parse_markdown_table` handles `"No results"` outputs as empty complete tables `([], [])` and parses comma-formatted numbers (`1,408`).
>   2. `generator_agents/genie/generator.py`: Strips Databricks citation links (`[1](https://...)`) from `Trace.final_answer` (preserving raw citations in `.genie_responses.jsonl`) and sets `allow_none=False` during benchmark generation.
>   3. `generator_agents/genie/router.py` & `orchestrator.py`: Added `allow_none` control for `GeminiDecisionRouter`.
>   4. `txt2sql_eval_harness/judge.py`: Updated `full_trace` `all_complete` so multi-step self-correction traces (with an early failed/empty step followed by a complete step) and volatile GT SQL do not falsely abstain as `incomplete_evidence`.
>   5. `txt2sql_eval_harness/packet.py`: Added `extra_claim_bindings` cut stage in `full_trace` mode when `n_cells <= max_cells` and `n_tokens > max_tokens` (preserving `evidence_exceeds_bound` for cell-overflow packets).

Reviewer notes for the author of the code (original baseline review below). Scope: all of `src/` (about 10.3k lines, 22 modules). The test suite passes (33 tests).

**Depth of review:**
- Read in full: `judge.py`, `runner.py`, `claims.py`, `comparator.py`, `sandbox.py`, `sql_analysis.py`, `clients.py`, `spend.py`, `cache.py`, `metrics.py`, `alignment.py` (the `align_tables` flow and helpers).
- Read in part: `baselines.py`, `cli.py`, `data.py`, `agent.py`, `models.py`.
- Only skimmed or not read: `generators.py`, `packet.py`, `renderer.py`, `reporting.py`, `rubrics.py`, `codec.py`.
- Findings in the unread files will be missing.
- Items marked **[verified]** were reproduced by running code. The rest come from reading the code.

---

## 1. What the code does

It is a hybrid, deterministic-first judge for Text-to-SQL. It labels a candidate (the last SQL step, or a multi-step trace plus a natural-language final answer) as `CORRECT`, `INCORRECT`, `AMBIGUOUS` or `ABSTAIN`. It compares the candidate against a reference ("ground truth", GT) SQL and result. It also includes benchmark-item generation, baseline judges, an experiment runner and reporting.

**Funnel (`judge.judge_item`)**
1. **Stage 0, execution.** `sandbox.execute_sandboxed` runs SQL read-only on SQLite. Safeguards:
   - immutable URI;
   - authorizer allow-list;
   - progress-handler timeout;
   - heap limit;
   - row, byte and cell caps;
   - a single-statement check.
   `judge._execute_steps` and `prepare_best_candidate` execute the steps and GT alternatives. They pick the best GT alternative.
2. **Stage 1, deterministic analysis.**
   - `sql_analysis` parses with sqlglot and computes:
     - a normalised form `N`;
     - volatility and determinism predicates;
     - a question-derived `Contract` (order and limit required).
   - `alignment.align_tables` finds a column bijection with unit-scale (x100, x0.01) and relative tolerance. It then does row matching and builds witnesses.
   - `claims` extracts numbers, dates and entities from the final answer and binds them to table cells.
   - Short-circuit rules run in this order:
     - D0: empty answer is INCORRECT.
     - D3: compile or non-read error is INCORRECT.
     - D2: no data against non-empty GT is INCORRECT.
     - D4: scalar mismatch is INCORRECT.
     - D5: scalar mismatch on a 30%-row-deleted DB copy is INCORRECT.
     - D1: identical normalised SQL is CORRECT.
3. **Stage 2.** `packet.build_packet` builds a token- and cell-bounded evidence packet.
4. **Stage 3.** Calibrated rubric probabilities from TypeSafe Jev (`clients.JevHttpClient`) are mapped through the `ThresholdPair` (`tau_hi`, `tau_lo`) by `sql_only_verdict_table` and `full_trace_verdict_table`.
5. **Stage 4.** Uncertain or over-bound items escalate to a Gemini judge. `H` is derived from `H+E` by dropping the escalation.

**Supporting pieces**
- `clients.call_model`: single egress-gated entry point with a response cache, retry policy, pre-send spend reservation and settlement.
- `spend.SpendLedger`: flock'd, fsync'd spend ledger.
- `cache.ResponseCache`: write-once SQLite cache.
- `runner.run_experiment`: serial, crash-resumable loop over methods and items.
- `runner.tag_slices`, `select_common_subset_c`: slice tagging and common-subset selection.
- `runner.project_spend_and_shrink_c`, `runner.fit_thresholds`: budget shrinking and threshold calibration.
- `baselines`: B1 (exact SQL), B2 (BIRD execution accuracy), IBM subset and judge, B4 (LLM judge).
- `data` and `generators`: dataset download and manifest, S1 to S3m and E5b synthetic suites, FLEX loader.
- `metrics` and `reporting`: bootstrap CIs, kappa, the H6 reading, T1 to T6 tables.

The architecture is good: layered, with a single egress choke point, content-addressed caching and crash-safe I/O. The problems below are mostly in the seams between layers and in the data-handling edge cases.

---

## 2. Logical flaws (ordered by severity)

### HIGH

**H1. [verified] The claim number extractor crashes on common suffixes.** `claims.py:75` and `claims.py:331-335`.
- The regex uses `re.IGNORECASE` and allows the short suffix `[kMbn]`. The multiplier lookup then does one of two things:
  - `mag_token in ("k","M")` is followed by `_MULTIPLIERS["M"]`, but the dict has only lowercase keys, so **`"2M"` raises `KeyError('M')`**.
  - `"5b"` and `"3n"` raise `KeyError('b')` and `KeyError('n')`, because `b` and `n` are not dict keys.
  - `"5bn"` matches nothing at all.
- Reproduced: `_extract_numbers("5K and 2M")` raises `KeyError('M')`, and `"Room 5b is open"` raises `KeyError('b')`.
- Any `full_trace` answer such as "$2M", "Room 5b" or "Section 3n" throws inside `_evaluate_candidate`.
- `runner.run_experiment` catches it as `internal_error` and records an `ABSTAIN` (see H2). Real answers are silently converted to abstentions.
- `b`/`n` are not sensible magnitude suffixes. `m` is ambiguous (metres or minutes versus million).
- Fix: normalise with `.lower()` for every lookup. Restrict suffixes to `k`, `m`, `bn`, `b` (the last one only after a currency symbol), or drop short suffixes. Add tests for `$2M`, `5K`, `3b`, `5bn` and `Room 5b`.

**H2. `internal_error` abstentions are permanent, and they hide bugs and misconfiguration.** `runner.py:525-563`, `clients.py:637`.
- `run_experiment` catches every `Exception` from `_dispatch_method`, writes an `ABSTAIN/internal_error` record, adds it to `done`, and fsyncs it.
- On resume those items are skipped, so transient failures and bugs are never retried.
- `client_for` raises `RuntimeError` for a missing `TYPESAFE_API_KEY` or `GEMINI_API_KEY`. `call_model` does not catch that. A run started without keys therefore "completes" with every model-backed item recorded as `internal_error`.
- It is worse when the egress gate has `allow_remote=True`, because no preflight check runs.
- Recommendations:
  - Do not persist `internal_error` as a completed record. Either re-run it on resume, or write it to a separate `errors.jsonl`. Alternatively, abort on the first non-judge exception unless `--continue-on-error` is set.
  - Log the traceback. Currently only `type(exc).__name__` is kept.
  - Preflight the API keys and SDK imports before the loop starts.
  - Treat `internal_error` and the infra abstains as run-degrading. `metrics.INFRA_ABSTAIN_REASONS` already does this, but only after the fact.

**H3. The egress gate is silently bypassed when no manifest is available.** `cli.py:213`, `runner.py:457-465`.
- The CLI passes `manifest_path=args.manifest if args.manifest.exists() else None`.
- `run_experiment` only calls `egress_decision` when `input_files` and `manifest_path` are both set. Otherwise it uses `allow_remote = not local_only`.
- `egress_decision` itself raises `UnadmittedInputError` when the manifest is missing and not `--local-only`. The runner never reaches it, so a missing manifest turns a hard error into "send everything to remote models".
- Fix: in the CLI, pass the path unconditionally. Make `run_experiment` fail closed if `local_only` is False and either argument is missing.
- Also, only `args.items` is checked. The DB-derived `SCHEMA_DDL` and the generated E5b items are also model-visible.

**H4. GT alternative selection does not look at the alignment result.** `judge.prepare_best_candidate`, `judge.py:605-628`.
- The docstring and README say "by determinism, execution status, and table alignment".
- The rank key is `(rank_outcome, mapped_count, -idx)`. `rank_outcome` is 2 only when a D1 rule hit says CORRECT, 0 for an INCORRECT rule hit, and 1 otherwise.
- `alignment.relation` is never used.
- Consequence: when no rule fires, an alternative whose result is `equal` to the candidate has the same rank as one that is `disjoint`, as long as the mapped column count ties. The first listed alternative wins. That alternative can send the item to Jev with a bad comparison, even though a better GT alternative matches exactly.
- Fix: add a relation-based tier, for example `equal > gt_subset_gen/gen_subset_gt/overlap > unknown > disjoint`, before `mapped_count`. In `full_trace` mode, apply it to the last successful step or to the best step.

**H5. The `generate` and `run` flows never use `fit_thresholds`, and S5 is never produced.**
- No call sites exist in the package for `runner.fit_thresholds`, `runner.project_spend_and_shrink_c`, `agent.run_reference_agent` or `agent.tag_s5_trace`.
- Consequences:
  - Every run uses the default `tau` map. The calibration split is built but never used for calibration.
  - The $120 budget shrink is never applied before running.
  - The README claims S5 traces come from FLEX, but `data.load_flex_dataset` emits only `set="S4"`, and nothing else sets `S5`.
- Either wire these into the CLI (`calibrate`, `plan` subcommands, or inside `run`), or drop the claims from the README and delete dead code.

### MEDIUM

**M1. The claim binder is very permissive, so it emits "supported" evidence for coincidental matches.** `claims._number_matches`, `claims.py:342-355`.
- Tolerance is half a unit in the last place of the claim, times the multiplier.
- "5" matches any cell in (4.5, 5.5). "1.2 million" matches 1,150,000 to 1,250,000. Rounded and truncated forms are both accepted without telling the model which.
- Small integers (1, 2, 3, years, "top 3") will bind to arbitrary cells in any table. Binding is to any cell anywhere (all rows, all steps, all derivations), with no column or header context.
- The `gt_cells`/`agent_cells` lists then reach Jev. A hallucinated "3" can appear as grounded.
- Mitigations:
  - Require the column or row context to match, or at least rank by closeness (an exact match is better than a rounding match).
  - Do not bind numbers with magnitude below some threshold, or mark them `weak`.
  - Record the match type (`exact`, `rounded`, `scaled`) on the `Claim`.

**M2. [verified] `classify_cell` treats numeric-looking text as numbers, so identifiers collide.** `comparator.py:49-60`.
- `"007"` equals `7`, and `"00123"` equals `123`, under both `tolerant=False` and `tolerant=True`.
- ZIP codes, product codes and IDs that are zero-padded strings will be reported equal to differently padded or integer IDs.
- This is probably deliberate for BIRD-style leniency, but `claims._extract_entities` and `alignment` both use it. Zero-padded or very long digit strings are a false-pass risk.
- Suggestion: only coerce TEXT to number if it has no leading zeros (except for `0.x`), is shorter than N digits, and the column is otherwise numeric. Or make it a config switch and flag `text_numeric_coercion` when it is the deciding factor.

**M3. `is_volatile` runs regexes on raw SQL, so comments and string literals cause false positives.** `sql_analysis.py:25-37, 128-140`.
- `WHERE status = 'now'`, `-- uses random()`, and a literal `'current_date'` all match. [verified: `is_volatile("... status='now'")` is `True`, and `"SELECT 1 -- random()"` is `True`.]
- A volatile verdict nulls the result in `_execute_steps` (`judge.py:238-241`), and the sandbox rejects the query as `volatile`. The item becomes unjudgeable for a trivially benign reason.
- Fix: detect volatility from the sqlglot AST (function and keyword nodes), and fall back to a token scan that skips comments and string literals.

**M4. D5 is narrower than documented, and expensive.** `judge.py:488-518`, `sql_analysis.create_d5_deletion_copy`.
- The README says D5 "runs both deterministic queries". The code only compares if both results are 1x1 numeric. Multi-row results never trigger a counterexample.
- It copies the whole SQLite file (`shutil.copy2`) and runs `DELETE` plus `PRAGMA integrity_check` for every item. For BIRD-sized DBs this is slow and I/O-heavy. The perturbed DB depends only on `(db_path, seed)`, so it could be built once per `db_id` and reused.
- Deleting 30% of rows from every table ignores FK integrity. This is acceptable for a "counterexample" test only if both queries are fully determined by the DB state, which `is_rd_eligible` approximates.
- Suggestion: generalise D5 to `align_tables(...).relation == "equal"` versus not-equal on the perturbed copy, and cache the copies.

**M5. D1 and the "coincident" flag ignore ordering.** `judge.py:318-326, 538`.
- `d1_match` conflicts and `coincident_result` use `alignment.relation == "equal"`.
- `align_tables` can return `relation="equal"` with `order_ok=False` when the contract requires order (for example "top 3"). So D1 can fire CORRECT, and `coincident` can be missed, for results that have the right rows in the wrong order.
- D1 requires identical normalised SQL, so this is unlikely to bite. But `coincident` is used for slice tagging (`tag_slices`) and for sql-only reporting. Treat `order_ok is False` as non-equal wherever `relation == "equal"` is used as an equality predicate.

**M6. The `Alignment.relation` value `disjoint` conflates different situations.** `alignment.py:856-890`.
- If any GT column cannot be mapped, the relation is `"disjoint"` with `rows_matched=0`, even when every other column matches exactly. Missing a single column is not "disjoint" from the GT, but the `method_ok` logic and `full_trace_verdict_table` conflict check (`al.relation == "disjoint"`) treat it as such.
- Suggest a distinct relation such as `missing_columns`, and keep `disjoint` for "columns map but no rows match".

**M7. [verified] Config values that are hashed but never used.** `clients._policy_for_stage` (`clients.py:449-463`).
- `EvalConfig` has `jev_max_attempts`, `jev_timeout_s`, `jev_backoff_s`, `generative_max_attempts`, `generative_timeout_s` and `max_retry_after_s`. All of them are part of `config_hash`.
- The retry policy hard-codes `3 / 10.0 / (1.0, 2.0)`, `2 / 60.0`, and a 30 s `Retry-After` cap.
- Changing config changes the hash (so `RunIdentity` is invalidated and the cache semantics change) but not the behaviour. Read these from `cfg`.

**M8. The Gemini transport ignores the timeout.** `clients.py:319-365`.
- `GeminiGenerativeClient.send(request, *, timeout)` never passes `timeout` to the SDK, so the 60 s policy does nothing.
- `except TimeoutError` will not catch SDK timeouts. A hang blocks the serial runner indefinitely.
- Pass `http_options` with a timeout to the client or request.
- Also, a safety-blocked response gives `resp.text == None`, which becomes `""` and is reported as `invalid_json`. Consider a distinct `blocked` outcome.

**M9. Method-major iteration biases partial runs.** `runner.py:518-521`.
- The loop is `for method: for item:`. If the spend cap is hit mid-run, early methods have full coverage and later ones have none.
- Resumed or capped runs then cannot be compared fairly. Iterate item-major, or at least interleave methods per item, so every method sees the same prefix of items.
- Cheap local methods (B1, B2, IBM subset) have no spend, so they could be run first without prejudice.

**M10. `fit_thresholds` has an inconsistent error definition and misleading status strings.** `runner.py:255-335`.
- Fitting constrains `false_passes / marked_correct <= 5%`, which is precision (FDR). `metrics._single_pass_stats` reports `false_pass` as `fp / n_incorrect` (FPR over all incorrect items, including those abstained). These share a name but not a definition.
- `tau_hi_reason` is overwritten on each grid step. If no grid point reaches 20 marked items, the status says "no_grid_under_5pct" when it should say "too_few_items". The `continue` on `<20` falls through to the final assignment.
- The `tau_lo` search iterates `g_lo < tau_hi` in descending order and takes the first success, which is the loosest (highest) `tau_lo`. Check that this matches the intent ("most coverage subject to FF <= 5%").
- Ambiguous-labelled items count as correct or incorrect verdicts but are not counted in either error. This is probably intended, but add a comment.

**M11. A run-directory identity mismatch gives no diagnostic.** `runner.py:483-487`.
- `RunIdentity` includes `code_commit` and `code_dirty`. Any edit or commit during development makes resume impossible, and the error is just `"incompatible_run_dir"`.
- At least list which fields differ. Consider making `code_commit` and `code_dirty` warnings (or opt-out) rather than hard failures.

**M12. FLEX label parsing defaults to `incorrect`.** `data.py:~340-355`.
- Anything that is not in `("1","true","correct","yes")` becomes `"incorrect"`, including `None`, empty strings and unrecognised text. `"ambiguous"` and `"gt_wrong"` are in the `Literal` type but cannot be produced here.
- Missing or unparseable labels become false incorrect labels, which silently skews calibration and false-pass or false-fail metrics. Reject or skip rows with an unknown label, count them in a `discarded` report, and decide explicitly how expert overrides map to `ambiguous` and `gt_wrong`.

**M13. B6 baseline shares deterministic rules with the funnel.** `judge_item`.
- `B6` and `B6+E` go through `_evaluate_candidate` and its D0 to D5 rules, so D-rule hits return before the single-question Jev call.
- If B6 is meant to be "Jev alone" for comparison, the rules inflate it. If it is meant to include them, the name and README should say so. Please confirm and document.

**M14. Truncated GT in baselines.** `baselines._resolve_gt_and_gen`, `judge_b2`, `_ibm_subset_match`.
- B2 and IBM-subset compare `gt_res` to `gen_res` without checking `.complete`. If either is truncated at the fetch cap, the verdict is made on a prefix.
- The funnel handles `complete=False` through `align_tables` ("unknown"), but the baselines do not. Abstain or flag when either side is incomplete.

### LOW

- **L1.** `sandbox.py:302`: `if "out of memory" in msg or "memory" in msg` makes any message containing the word "memory" a `resource_limit`. This includes `no such table: memory_stats`. The second test makes the first redundant. Use SQLite's `SQLITE_NOMEM` and error codes (`exc.sqlite_errorcode` on Python 3.11+) instead of string matching.
- **L2.** `sandbox._open_readonly_connection` runs outside the `try/finally`. A `sqlite3.OperationalError` ("unable to open database file") propagates as an exception, which H2 turns into `internal_error`. Map it to a distinct `db_unavailable` error.
- **L3.** `judge._evaluate_candidate` returns `(cand, False)`. The bool is always `False` and ignored (`prepare_best_candidate` does `cand, _ = ...`). `cand.flags = flags` after the D1 conflict is a no-op because `cand.flags` is the same list. `if contract.limit_required and rows_only_gen > 0 and rows_only_gt == 0: relation = "gt_subset_gen"` in `alignment.py:933` is already the result of the branch above and does nothing.
- **L4.** `check_d0` runs twice per item (`judge_item` and `_evaluate_candidate`). `judge_item` has an inner `if method == "B6"` abstain branch that is shadowed by the later `if method == "B6" or not escalation` check.
- **L5.** `all_complete` is computed for `sql_only` but only enforced for `full_trace`. SQL-only items with a truncated GT or candidate go to Jev with partial data. This may be fine (packet flags truncation), but the asymmetry should be documented.
- **L6.** Heuristic limit cue (`_LIMIT_NUM_PLURAL_RE`): `"among the 3 schools"` or `"2 years"` mark the limit as required. `_ORDER_CUES` includes `first`, `last` and `most`, which are very common in non-ordering contexts. Only matters when the GT also has `ORDER BY`/`LIMIT`, but it is a precision risk.
- **L7.** Performance: `claims.extract_and_bind_claims` is O(claims x cells), with a regex compile per entity and a regex match per date and cell. Caps allow 200k rows. `alignment` re-sorts every column for every GT x generated pair, up to 64 x 64. Index cells once (dict by canonical value) and cap or sample.
- **L8.** `alignment._single_column_match_count` and `_match_rows` use two-pointer greedy matching under tolerance. With non-transitive tolerance, this is not a guaranteed maximum matching, although the docstrings say "maximum".
- **L9.** `metrics.load_records` silently overwrites any recorded `H` method with a derived value. The derived `H` also relies on `H+E` being present.
- **L10.** `cli._bird_records_to_items` drops the GT execution error (`_err`) and sets `results=None` silently. The seed items use `set="S1"`, `family="base"`. This naming overlaps with the real S1 rewrites. Use a distinct set name.
- **L11.** `_get_api_key` reads `.env` from the CWD, `src/.env` and the package parent. This is convenient but surprising and weakens the stated "client_for is the only place that touches keys". Consider an explicit `--env-file`.
- **L12.** `spend.SpendLedger._open_or_init` silently adopts a new `cap_usd`. A cap raised between runs is not recorded anywhere.

---

## 3. Structure and clarity recommendations

1. **Split `judge.py` (1019 lines).**
   - Move the verdict-table logic (`sql_only_verdict_table`, `full_trace_verdict_table`, `single_verdict_v`) into `verdicts.py`.
   - Move `_make_abstain` and `_make_verdict` to a `verdict_factory.py` or onto `Verdict` as constructors.
   - Move candidate preparation (`_execute_steps`, `prepare_best_candidate`, `_evaluate_candidate`) into `candidate.py`.
   - Leave `judge_item` as the orchestrator.
   - Rule evaluation inside `_evaluate_candidate` is a 130-line if-chain. Extract each rule into a function, `rule_d0(ctx) -> Verdict | None`, and iterate through an ordered list. This also makes rule order testable and removes the repeated `_make_verdict(...)` blocks.
2. **Fix the dependency direction.**
   - `baselines.py` imports private helpers from `judge.py` (`_execute_steps`, `_make_verdict`, and others). Make them public in a shared module.
   - `metrics.py` imports `judge.derive_h_from_he`. Move it to `verdicts`.
   - `clients.py` has imports in the middle of the module (lines 25-27), after type aliases.
3. **Remove the `Verdict` construction boilerplate.** `Verdict(...)` is built with about 25 keyword arguments in four places (`_make_abstain`, `_make_verdict`, `derive_h_from_he`, `runner` fallback). The runner copy even spells out the zero fields. Add `Verdict.abstain(...)` and `Verdict.decided(...)` classmethods with defaults for the `b4_*`, cost and timing fields.
4. **Replace magic strings with enums or Literals.**
   - Verdict labels, `decided_by` (`"rule:D1"`), reasons (`"internal_error"`, `"spend_cap"`) and relation names are bare strings spread across `judge`, `runner`, `metrics` and `baselines`.
   - `INFRA_ABSTAIN_REASONS` in `metrics` has to track strings produced elsewhere. A typo fails silently.
   - Define the constants once in `models.py`.
5. **Centralise policy numbers.**
   - `G_HI`/`G_LO`, quotas (40/60/40, 12 groups, 30 per slice), `0.05`, `20`, the 1000-row `large_result` threshold, the 0.03 and 0.05 H6 margins, and `1.2x` upper bounds are scattered.
   - Move them to named constants, or into `EvalConfig` where they affect results (and thus `config_hash`).
6. **Dead and unwired code.**
   - `fit_thresholds`, `project_spend_and_shrink_c`, `run_reference_agent`, `tag_s5_trace`, and the `S5` and `gt_wrong` labels are implemented but never reached from the CLI.
   - Either add subcommands or mark them as library-only in the docstrings and README.
7. **Make the README's rule semantics match reality.**
   - D5 (1x1 only).
   - GT alternative ranking.
   - Config-driven retry settings.
   - The S5 source.
   - Whether B6 includes D-rules.
8. **Error handling.**
   - There are 13 `except Exception` sites and 7 bare `pass` handlers (`sandbox.py`, `baselines.py`, `data.py`, `agent.py`).
   - Narrow them where possible, and log or flag instead of passing silently. For example `execute_sandboxed` hides `PRAGMA hard_heap_limit` failure, so the heap limit may not be enforced and nobody knows.
   - Surface it as a flag (`heap_limit_unavailable`).
9. **Types and signatures.**
   - `_evaluate_candidate` takes 11 parameters. Bundle `(trace, gt, db_path, cfg, cfg_hash, method)` into a `JudgeContext` dataclass.
   - `sql_only_verdict_table` returns `VerdictLabel | str`. Use a proper `Literal["CORRECT","INCORRECT","AMBIGUOUS","ESCALATE"]` and avoid `# type: ignore`.
10. **Tests to add.**
    - Claims extraction edge cases (H1).
    - GT-alternative ranking (H4).
    - A runner test with a missing API key, which must not write `internal_error` records (H2).
    - A CLI test with no manifest (H3).
    - Volatility false positives (M3).
    - Order-sensitive equality (M5).
    - `fit_thresholds` status strings (M10).
    - A test that asserts retry-policy fields come from `EvalConfig` (M7).

## 4. Suggested fix order

1. H1 (crash, easy fix). Then H2, H3, H4.
2. M7, M8, M9, M12. These affect result validity and cost control.
3. M1, M2, M3, M5, M6. These affect judge precision.
4. Structural work (sections 3.1 to 3.3), once the behaviour is pinned by tests.
