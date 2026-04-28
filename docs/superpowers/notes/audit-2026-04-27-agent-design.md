# Agent-System Design Audit — deriva-ml-mcp + deriva-ml-skills

**Date:** 2026-04-27
**Scope:** deriva-ml-mcp v1.x tools/resources/prompts + deriva-ml-skills 23 skills
**Auditor perspective:** An LLM (Claude) trying to do real ML-development work end-to-end

---

## User-Goal Walkthroughs

### Goal 1: Explore an unfamiliar catalog

**Walkthrough.** The LLM arrives with only a hostname and a catalog ID. There is no `connect_catalog` step — this is called out explicitly in both the `deriva_ml_getting_started` prompt and the `deriva-ml-context` always-on skill. The stateless model is actually well-documented once you read those two sources. Assuming the LLM has the prompt loaded (it is not automatically sent — the user must invoke `deriva_ml_getting_started` or the always-on context skill must trigger), the first move is clearly: read `deriva://catalog/{h}/{c}/ml/registries` (vocabs), `deriva://catalog/{h}/{c}/ml/datasets`, `deriva://catalog/{h}/{c}/ml/workflows`, and `deriva://catalog/{h}/{c}/ml/executions` as a four-resource cold-start sweep. These resources are free reads, uncapped only at 1000, and return structured JSON.

The `registries` resource bundles dataset types, workflow types, asset types, and execution statuses in a single fetch — this is genuinely useful for orientation. It returns only `{name, rid}` per term by design (description is elided), which is a reasonable tradeoff. If the LLM wants descriptions it falls back to `rag_search` or `list_vocabulary_terms`.

For schema discovery there is a gap: the MCP surface has no tool for "list domain tables" or "what tables carry my research data?" That is handled by the tier-1 `deriva-skills` plugin's schema tools (`browse-erd`, `route-catalog-schema`). An LLM using only `deriva-ml-mcp` tools directly cannot get a table list without calling the generic `get_table_sample_data` or `rag_search`. The `deriva-ml-context` skill bridges this — it says use `/deriva:browse-erd` or `/deriva:route-catalog-schema` for schema introspection — but only if that skill is loaded.

Naming conventions are documented in the `api-naming-conventions` always-on skill and in the `deriva_ml_getting_started` prompt's "RESOLVING USER-MENTIONED NAMES TO CATALOG IDENTIFIERS" section. The RAG coverage index is also documented there with freshness caveats. This is strong.

`rag_search` is the recommended semantic entry point, and the getting-started prompt explains two doc types: `catalog-data` (Dataset/Workflow/Execution rows) and `catalog-schema` (tables, vocab terms). The LLM is correctly steered to use RAG before paginated list tools.

**Friction points:**
1. The cold-start requires knowing to invoke `deriva_ml_getting_started` or having the always-on `deriva-ml-context` skill trigger. If neither fires, the LLM sees 45+ tools with no orientation.
2. No "list domain tables" tool exists in this plugin. Schema discovery requires the tier-1 companion.
3. The `registries` resource elides term descriptions. For "what does this workflow type mean?" the LLM must fire a follow-up `rag_search` or `list_vocabulary_terms`.
4. Resource URIs must be fully known — the LLM cannot discover available resources by browsing; it must know the template from the prompt.

**Score: 3/5.** Resources are well-structured, the getting-started prompt is comprehensive, and RAG is usable. But the cold-start assumes prompt consumption, schema discovery leaks to the tier-1 layer, and the resource URI namespace is not self-discoverable.

---

### Goal 2: Develop a new model

**Walkthrough.** The user says "train a CNN on the Image dataset to predict cell types." The recommended path from `model-development-workflow` and `execution-lifecycle` is:

1. Find the relevant data: `rag_search("Image cell type dataset", doc_type="catalog-data")`, then `deriva_ml_list_datasets` or `deriva://catalog/{h}/{c}/ml/datasets`. Good — the RAG and resource coverage makes data discovery tractable.
2. Inspect features: `deriva_ml_list_features(table="Image")` or `deriva://catalog/{h}/{c}/ml/features/Image`. The resource has the right shape. `deriva_ml_get_feature(table="Image", feature_name="Cell_Type")` returns term_columns/value_columns/asset_columns with nullok flags — enough for the LLM to build valid feature records.
3. Register a workflow: `deriva_ml_create_workflow(name=..., workflow_type=..., url=..., checksum=...)`. The `deriva_ml_workflow_dedup` prompt cleanly documents idempotency. However, the LLM must supply the Git URL and checksum itself — the tool cannot compute them. The `execution-lifecycle` skill says "code must be committed" and explains `deriva-ml-run`, but does not give the LLM a tool to fetch the Git hash. The LLM must ask the user or assume they computed it.
4. Create an execution: `deriva_ml_create_execution(workflow_rid=..., dataset_rids=["4HM@1.0.0"], ...)`. The `"RID@version"` shorthand for dataset inputs is documented in the execution tool docstring. `dry_run=True` is supported.
5. Start the execution: `deriva_ml_start_execution(execution_rid=...)`.
6. Recording results: `deriva_ml_create_feature(target_table="Image", feature_name="Cell_Type_Prediction", terms=["Cell_Type"], ...)` then `deriva_ml_add_feature_values(table="Image", feature_name="Cell_Type_Prediction", execution_rid=..., entries=[...])`. The hybrid dispatch is documented in the `deriva_ml_execution_lifecycle` prompt — if the execution is already `Running`, call `add_feature_values` directly; if it is `Created`, the tool auto-wraps.
7. Commit: `deriva_ml_commit_execution(execution_rid=...)`.

**Where the LLM must guess or hand off:**

- **File I/O is entirely out of scope for the MCP surface.** The model checkpoint file cannot be registered through MCP tools alone. The `work-with-assets` skill generates Python the user runs locally. This boundary is documented but requires the user to run code outside Claude.
- **The `workflow_name` phantom parameter.** The `model-development-workflow` skill (lines 146, 191) passes `workflow_name=` to `deriva_ml_create_execution`. That parameter does not exist — the real signature is `workflow_rid=`. This will cause a runtime error if the LLM follows the skill exactly. A real hallucination risk baked into the skill text itself.
- **`entries=` vs `values=`.** The `create-feature` skill uses `values=` in all its `deriva_ml_add_feature_values` examples (lines 166–188). The actual tool parameter is `entries=`. This mismatch will generate a validation error.
- **The Git commit boundary.** `deriva-ml-run` enforces a clean working tree; MCP tools do not. An LLM trying to do the whole workflow via MCP tools without the CLI can succeed, but provenance is weaker. This boundary is explained well in `execution-lifecycle` but is a UX cliff the first time a user hits `DerivaMLDirtyWorkflowError`.
- **`create_execution` does not accept a workflow name string.** The LLM must first resolve or create the workflow and hold its RID. The tool is correctly designed for this, but the step ordering is non-obvious without the prompts.

**Score: 3/5.** The happy path is well-covered by the prompts and skills if the LLM reads them carefully. The `workflow_name` phantom and `entries`/`values` mismatch are real bugs in skill content that will cause tool call failures. File I/O hand-off to local Python is a coherent design decision but a significant capability gap.

---

### Goal 3: Compare model runs

**Walkthrough.** "Show me which of the last 5 training runs got the best F1 score."

1. Find executions: `deriva_ml_list_executions(status="Uploaded", limit=5)` — sorted by RID ascending, so "last 5" by RID is feasible. However, execution summary fields are `{rid, workflow_rid, status, description, start_time, stop_time, duration}`. There is no sorting by `start_time` descending; pagination is RID-ordered ascending. To get the 5 most recent, the LLM must either do a preflight count and page to the end, or use `rag_search("recent training executions", doc_type="catalog-data")` and hope the index is fresh.
2. Identify the F1 feature: `rag_search("F1 score feature", doc_type="catalog-schema")` or `deriva_ml_list_features(table="Image")`. If the catalog has a feature named `Metrics` with a `f1_score` value column, the LLM can find it via RAG. If the feature name is not intuitive, this step requires trial and iteration.
3. Query feature values per execution: `deriva_ml_list_feature_values(table="Image", feature_name="Metrics", selector="by_execution", selector_execution_rid="1-EXEC")` — one call per execution. For 5 executions this is 5 calls. The selector enum is documented in the tool's `Args:` block inline: `none | newest | first | latest | majority_vote | by_workflow | by_execution`. Clear and actionable.
4. Ranking: The LLM must aggregate the responses across 5 tool calls and sort. There is no "query feature values across multiple executions" tool. This is a real gap — the LLM cannot do an in-catalog comparison query. It must materialize all the values into context and compute the ranking itself.

**Friction points:**
- No reverse-chronological ordering on list tools. "Last 5" requires either the preflight-count dance or a RAG hit.
- No cross-execution feature value query. The LLM must loop.
- Feature discovery relies on naming conventions or RAG. If the metric column is named `val_f1` in the feature value record, the LLM must know that from `deriva_ml_get_feature` before it can extract it.
- The `selector="by_execution"` path works well once the LLM knows the execution RIDs and the feature name.

**Score: 2/5.** This is the weakest goal. The surface has the building blocks but no "compare across executions" primitive. The LLM must do 5+ sequential tool calls, aggregate in context, and rank manually. On large catalogs where the last 5 runs require paging, the friction compounds. RAG freshness also matters here — stale RAG on a shared catalog means recent runs may not be discoverable.

---

### Goal 4: Manage catalog growth

**Walkthrough.** "We need a new feature schema for storing model uncertainty per-prediction."

1. Design: The `create-feature` skill documents the decision tree (term-based / asset-based / scalar / mixed). A per-prediction scalar uncertainty score is a value column (`float4`), making this a metadata-only feature. The LLM would call `deriva_ml_create_feature(target_table="Image", feature_name="Uncertainty", metadata=["uncertainty_score"], comment="...")`. The `metadata=` parameter accepts `list[str | dict]` — the type union is documented in the tool's `Args:` block, but "metadata" here means "extra scalar columns"; the naming is not self-explanatory. The `create-feature` skill explains it as "metadata columns" with a type table, which is better.
2. Vocabulary: If the feature is term-based (e.g., uncertainty category), `add_term(schema="deriva-ml", table="...", ...)` via the tier-1 surface. This hand-off is documented in `deriva-ml-context`.
3. Verify: Read `deriva://catalog/{h}/{c}/ml/features/Image` to confirm the feature appears. Good.
4. Re-index: The LLM must call `deriva_ml_reindex_vocabularies()` after adding terms. But `deriva_ml_create_feature` does not trigger a schema RAG re-index (that is a tier-1 responsibility). The getting-started prompt says: "To re-index after adding a term via core's `add_term`, call `deriva_ml_reindex_vocabularies` first." But creating a new feature table is a schema mutation — the schema RAG (`catalog-schema` doc type) needs refreshing too. The prompt says fresh-on-first-connect for tables/columns, but does not say how to force a schema re-index after creating a new feature table mid-session. `rag_index_schema()` is mentioned in `model-development-workflow` Phase 1, but is it a tool available to the LLM? It exists as a tier-1 tool (`rag_index_schema`) but is not called out in the feature skill.
5. Communication to other users: Nothing in the surface addresses this. The cross-user freshness gap (v1.4 known limitation) means other users will not see the new feature in their RAG index until they reconnect. `deriva_ml_resync_indexes` is the manual bridge, but it is not mentioned in the create-feature skill.

**Score: 3/5.** Feature creation is well-supported. The friction is at the edges: the `metadata=` parameter name is misleading for scalar columns, schema RAG re-index after feature creation is not clearly documented in the create-feature skill, and cross-user propagation is invisible unless you know about `deriva_ml_resync_indexes`.

---

### Goal 5: Recover from failure

**Walkthrough.** "My training run crashed mid-epoch. Execution shows Running but process is dead."

1. Diagnose: `deriva_ml_get_execution(hostname=..., catalog_id=..., execution_rid="1-EXEC")` — returns `{rid, workflow_rid, status, description, start_time, stop_time, duration}`. Or read `deriva://catalog/{h}/{c}/ml/execution/1-EXEC` for the richer detail including inputs/outputs. Status will be `Running`.
2. The `troubleshoot-execution` skill covers "Execution Stuck in Running" explicitly with the right diagnostic sequence.
3. Decision: abort. `deriva_ml_abort_execution(execution_rid="1-EXEC", reason="Process died mid-epoch")`. This is clean and idempotent — if the status is already `Aborted`, it no-ops with `status="already_aborted"`. The state machine is enforced.
4. **Critical bug in the skill.** The `troubleshoot-execution` skill (line 129) lists this option: `deriva_ml_update_execution(hostname, catalog_id, execution_rid, status="Failed", message="Manually marked as failed")`. That tool signature is wrong. `deriva_ml_update_execution` only accepts `description=` — it does not accept `status=` or `message=`. If the LLM follows this skill text, it will generate a tool call that fails. `update_execution` explicitly rejects status edits with: "description must be provided; no other fields are editable on Execution (status changes go through start/commit/abort)".
5. Recovery: `deriva_ml_create_execution(workflow_rid=..., dataset_rids=..., ...)` — fresh execution with same inputs. The troubleshoot skill documents this "re-run after abort" pattern correctly (lines 159–163).
6. Verify: `deriva_ml_get_execution(execution_rid="<new_exec>")` — confirm status is `Created`.

**Friction points:**
- The bogus `status="Failed"` option in `troubleshoot-execution` is a live trap. The LLM reading the skill might try the "arbitrary status" path before the abort path. It will fail with an error envelope, at which point it should fall back to `abort_execution` — but that requires the LLM to correctly interpret the error and try the right alternative.
- `deriva_ml_get_execution` returns a flat summary (no inputs/outputs). The resource `deriva://catalog/{h}/{c}/ml/execution/{rid}` returns inputs+outputs+experiment. For diagnosing "what did this run have as inputs?" the resource is better but requires knowing the URI template.
- The orphaned local working directory (staged files) is not addressable from the MCP surface at all. The `troubleshoot-execution` skill mentions `derive://storage/execution-dirs` but that resource is from the tier-1 plugin, and the cleanup is manual.

**Score: 3/5.** The abort path is solid, the state machine enforcement is good, and the re-run pattern is documented. The bogus `status=` option in the troubleshoot skill is a real hazard that needs fixing.

---

### Score Summary

| Goal | Score | Top friction |
|------|-------|-------------|
| 1 — Explore unfamiliar catalog | 3/5 | No cold-start prompt auto-injection; schema discovery leaks to tier-1 |
| 2 — Develop new model | 3/5 | `workflow_name` phantom param; `entries`/`values` mismatch in skills |
| 3 — Compare model runs | 2/5 | No reverse-sort on list tools; no cross-execution feature query |
| 4 — Manage catalog growth | 3/5 | `metadata=` naming; schema RAG re-index gap after feature creation |
| 5 — Recover from failure | 3/5 | Bogus `status=` option in troubleshoot skill; resource vs tool gap for full detail |

---

## Cross-Cutting Findings

### A. Tool naming + discoverability

The `deriva_ml_` prefix is consistent across all 45 tools and is the right call — it prevents collision with tier-1 tools and makes grep/autocomplete reliable.

**Verb clarity is high.** `create` / `list` / `get` / `find` / `delete` / `add` / `update` / `commit` / `abort` are all standard REST-flavored verbs that translate directly to intent.

**Two close-sibling pairs require docstring disambiguation:**

- `deriva_ml_create_execution` vs `deriva_ml_create_execution_dataset`: the latter creates an output dataset linked to a completed execution, not a general dataset. An LLM trying to "create a dataset from this execution" will likely pick the right one — the name is explicit — but the distinction between "input dataset" (supplied at `create_execution` time via `dataset_rids=`) and "output dataset" (created post-hoc via `create_execution_dataset`) is subtle enough to cause mistakes. The getting-started prompt documents this cleanly.
- `deriva_ml_list_executions` vs `deriva_ml_find_workflow_executions`: these are nearly identical in behavior (the latter is `list_executions(workflow_rid=...)`). The getting-started prompt justifies this as "different LLM intent" — "runs of this workflow" vs "browse executions." This is debatable; in practice an LLM that sees both tools may invoke either for either intent.

**`deriva_ml_list_feature_values` with `selector=` is the most complex tool in the surface.** The `Literal[...]` type annotation with 7 values plus two conditional required parameters (`selector_workflow`, `selector_execution_rid`) is hard for an LLM to get right without the `create-feature` skill loaded. Without the skill, the LLM might invoke `selector="by_workflow"` without `selector_workflow=` and get a clean error message — which is good.

### B. Tool argument design

**Strengths:**
- The `entries=` parameter in `deriva_ml_add_feature_values` carries a clear inline error if empty.
- Pagination is uniform across all list tools (`limit=`, `after_rid=`, `preflight_count=`). The contract is documented once in the getting-started prompt and not repeated in every docstring — good DRY discipline.
- The `workflow_rid=` name in `deriva_ml_create_execution` is precise; early versions of the code comment explain that passing a raw string would silently route to `lookup_workflow_by_url` instead of `lookup_workflow`, which would be a confusing failure. The resolved path is the right design.
- Status enum values (`Created`, `Running`, `Stopped`, `Pending_Upload`, `Uploaded`, `Failed`, `Aborted`) are enumerated in the `deriva_ml_execution_lifecycle` prompt and in `_list_executions_impl`'s Args docstring. The string values match `ExecutionStatus.value` exactly.

**Weaknesses:**
- `deriva_ml_create_feature`'s `metadata=` parameter accepts `list[str | dict]`. "Metadata" is ambiguous — it sounds like key-value metadata about the feature, but it means "extra scalar or FK columns added to the feature table." The `create-feature` skill explains this, but the tool docstring does not.
- `deriva_ml_create_workflow`'s `workflow_type=` accepts `str | list[str]`. This is good for single vs multi-type, but the valid values are not listed in the docstring (`Args:` says "Each must be a term in the `Workflow_Type` vocabulary" without enumerating the terms). The LLM must do a prior `rag_search` or `registries` fetch to know valid values.
- `deriva_ml_split_dataset`'s parameters are not visible from this audit; it is in `complex.py` which was not fully read. The skill mentions `stratify_by_column` but does not enumerate required vs optional args.
- `deriva_ml_cache_dataset`'s `materialize=` parameter defaults to `True` (full asset download). An LLM doing a preflight check that calls `cache_dataset` without `materialize=False` will trigger a full download — potentially gigabytes. The docstring says "If False, only fetch the bag metadata" but does not warn "this downloads everything." The `execution-lifecycle` skill's Phase 1 table says to use `bag_info` for size checking before `cache_dataset`, which partially mitigates this.

### C. Resource utility

Resources are actively used across the skill surface. Counting explicit resource references in skills:

- `execution-lifecycle` SKILL.md: 3 resource references (execution detail, executions list, workflows list, registries)
- `dataset-lifecycle` SKILL.md: 2 resource references (datasets list, registries)
- `create-feature` SKILL.md: 0 explicit `deriva://` resource references (recommends RAG and typed tools instead)
- `troubleshoot-execution` SKILL.md: 1 resource reference (execution detail in Phase 3 verify)
- `model-development-workflow` SKILL.md: 0 explicit resource references

Resources are better used in the getting-started prompt (10 URIs enumerated) than in individual skills. The skills lean more toward typed tools (`deriva_ml_get_execution`, `deriva_ml_list_features`) than their corresponding resources (`deriva://catalog/.../ml/execution/{rid}`, `deriva://catalog/.../ml/features/{table}`). This is defensible — tools are stateful (they handle errors cleanly) and skills document the typed tools for predictability. But the resources are faster (one network call vs tool overhead) and return richer detail for execution (inputs + outputs bundled). The `execution-lifecycle` skill does point to the detail resource for post-run verification, which is correct.

**Underused resources:** `deriva://catalog/{h}/{c}/ml/asset/{rid}` and `deriva://catalog/{h}/{c}/ml/asset-tables` are mentioned only in the getting-started prompt. No skill explicitly says "use the asset resource for this." The `work-with-assets` skill was not fully read but likely covers this.

### D. Prompt utility

**`deriva_ml_getting_started`** is the strongest asset in the system. It covers: the stateless model, the pagination contract, the error envelope, the five ML domains with verb tables, the resource URI namespace, RAG discovery patterns, name resolution order, RAG freshness with cross-user caveats, the mutation chain, and the asset file I/O split. At approximately 370 lines it is long but dense and mostly non-redundant. Grounding on this prompt gives an LLM an accurate mental model of the surface.

**`deriva_ml_execution_lifecycle`** correctly describes the state machine with concrete state names, the two state-set constants (`_START_REJECT_STATES`, `_COMMIT_ALLOWED_STATES`), the five lifecycle tools, the hybrid dispatch on `add_feature_values`, and the two pitfalls (no manual status edit, always commit). This is load-bearing; without it the LLM will likely mishandle the `Created -> Running` boundary.

**`deriva_ml_workflow_dedup`** is short and correct. The anti-pattern / correct pattern pair is exactly what an LLM needs to avoid the double-call trap.

**Missing prompts:**

- **`deriva_ml_feature_design`** — There is no prompt for the feature design mental model (term vs asset vs scalar, single vs multi-column, selector strategy). The `create-feature` skill covers this, but a prompt would help an LLM that encounters a feature question without loading the skill.
- **`deriva_ml_dataset_versioning`** — The versioning rules (always pin versions in configs, increment after changes, semantic version convention) are important enough for a prompt. They are currently spread across `dataset-lifecycle` skill, `model-development-workflow` skill, and the `execution-lifecycle` skill's pre-flight section.
- **`deriva_ml_rag_freshness`** — The cross-user freshness gap and the `resync_indexes` recovery pattern deserve a prompt. The getting-started prompt covers this in one paragraph, but it is easy to miss.

### E. Error message quality

The `_error_envelope` implementation calls `str(exc)` on the caught exception. Error quality depends on what `deriva-ml` raises.

**Sampled error paths:**

1. **Unknown workflow RID in `create_execution`.** `ml.lookup_workflow(workflow_rid)` will raise `DerivaMLException`. The error envelope returns `{"error": "Workflow with RID '<rid>' not found in catalog"}` (inferred from the comment in the source: `"Workflow with URL or checksum '<rid>' not found in catalog"` — but this is the URL path, not the RID path). The RID lookup path raises a different exception. The error text may say "not found" without telling the LLM what to do next. Actionability: low.

2. **Wrong execution state for `start_execution`.** This path is handled before `_error_envelope`: the tool returns `{"error": "cannot start execution in state Stopped; only Created (will start) or Running (no-op) are valid"}`. This is actionable — it tells the LLM exactly what states are valid and what the rejected state was.

3. **Empty `entries` in `add_feature_values`.** Returns `{"error": "entries must be a non-empty list.", "attempted_count": 0}`. Clear and actionable.

4. **Unknown workflow_type term in `create_workflow`.** Will propagate from `ml._add_workflow()` as a `DerivaMLException` or catalog FK violation. The `str()` of the underlying exception is catalog-layer text — likely something like "unknown term in Workflow_Type vocabulary." Not specifically actionable (doesn't say "call `add_term` to add it").

5. **`commit_execution` on an already-aborted execution.** Returns `{"error": "cannot commit execution in state Aborted; only Created, Running, Stopped, Pending_Upload, or Uploaded (additive upload) are valid"}`. This is excellent — the valid state list is enumerated, the rejected state is named, and the additive-upload edge case is explained inline.

**Assessment:** The tool-internal validation paths (states, empty lists) produce actionable errors. The passa-through from `deriva-ml` library errors are less predictable — they give the LLM the raw exception message which may or may not say what to do next. No error message in the surface says "call `rag_search` to find the correct value" or "use `list_vocabulary_terms` to find valid terms." That gap means the LLM must synthesize the recovery step from general knowledge rather than the error text.

### F. Provenance + reproducibility

The surface makes provenance **hard to skip by accident**:

- `deriva_ml_add_feature_values` requires `execution_rid=` — there is no way to add feature values without an execution context.
- `deriva_ml_create_execution_dataset` requires `execution_rid=` — output datasets must be linked to an execution.
- The `deriva-ml-context` always-on skill's steering principle explicitly warns: "raw `insert_entities` bypasses provenance tracking" and lists the specific harms.
- `deriva_ml_create_execution` accepts a `dry_run=True` flag that skips catalog writes and audit — explicitly documented as "test without recording."

**Gap:** `deriva_ml_create_dataset` (the standalone tool) does not require an execution context. You can create a dataset that is not linked to any execution. The skills recommend creating an execution first ("create a workflow and execution for provenance tracking") but the tool does not enforce it. This is a correct design decision for bootstrap scenarios (the first dataset of a project cannot be linked to an execution that consumed it), but it is a place where the LLM can accidentally skip provenance for datasets it is curating.

**Git boundary:** `DerivaMLDirtyWorkflowError` is enforced by `deriva-ml-run` but not by the MCP tools. The MCP execution tools do not check for dirty working trees. This means an LLM driving the execution lifecycle entirely through MCP tools (without the CLI) will happily create executions without a clean git state. The provenance record will not have a verifiable git hash. The skills document this but the tools cannot enforce it.

### G. Pagination + scale

**Implemented correctly:**
- All list tools support `limit=` (default 100, max 1000) and `after_rid=` cursor pagination.
- Resources are capped at 1000 rows with `truncated: bool` signal.
- `preflight_count=True` allows the LLM to learn the count before committing to a page size.
- `_paginate` implementation: `truncated = len(page) == limit` — this is a false-positive edge case (exactly 1000 results looks truncated), matching deriva-mcp-core convention. Acceptable.
- Feature values have the same pagination (`after_rid=` is by RID for feature records, which have RIDs).

**Gap:** `deriva_ml_list_feature_values` materializes ALL matching records into memory before pagination (`records = list(ds.feature_values(...))`). For a catalog with 10M feature values, this will OOM or timeout before pagination kicks in. The tool does not mention this limitation. A `preflight_count=True` call has the same problem — it counts a fully-materialized list. This is a known scalability risk.

**Ordering:** List tools sort by RID ascending. There is no `sort_by=` or `order=` parameter. For "show me the most recent executions" the LLM must either page to the end of a large set or rely on RAG freshness. This is a real gap for time-ordered exploration.

### H. Failure modes specific to LLMs

**RID hallucination:** `deriva_ml_get_execution(execution_rid="1-XXXX")` on a non-existent RID will return `{"error": "<DerivaMLException message>"}`. The LLM has no tool to validate a RID before using it except `get_entities` from the tier-1 surface. `validate_rids` was removed. The typed lookups (`deriva_ml_get_dataset`, `deriva_ml_get_execution`, etc.) are the correct recovery — the skills document this correctly. The LLM is steered to look up before operating on a RID.

**Column/feature name hallucination:** `deriva_ml_get_feature(table="Image", feature_name="Quality_Score")` on a non-existent feature returns an error. The LLM should then use `deriva_ml_list_features(table="Image")` to find the real name. The `create-feature` skill says to use `rag_search` first, which catches this class of hallucination.

**Vocabulary term hallucination:** An LLM passing an invented `workflow_type="Model_Training_CNN"` to `deriva_ml_create_workflow` will get a catalog FK violation error. The error text from `deriva-ml` may not say "this is not in the Workflow_Type vocabulary." The LLM must infer that and call `list_vocabulary_terms` or `rag_search`. The getting-started prompt documents the vocabulary as searchable but doesn't enumerate terms (by design — terms vary per catalog). The `registries` resource gives the LLM all valid terms in one read.

**Status enum hallucination:** `deriva_ml_list_executions(status="Complete")` instead of `status="Uploaded"` will raise `ValueError` in `ExecutionStatus("Complete")`. The error envelope will return `{"error": "'Complete' is not a valid ExecutionStatus"}` (Python's enum error text). The valid values are enumerated in the execution lifecycle prompt. Moderately actionable.

**The `entries`/`values` and `workflow_name` bugs documented above** are literal skill-induced hallucinations — the skill tells the LLM to pass parameters that do not exist. These are the highest-risk findings in the audit.

---

## Top-Line Findings

1. **Two skill content bugs will cause runtime failures.** The `model-development-workflow` skill passes `workflow_name=` to `deriva_ml_create_execution` (the parameter is `workflow_rid=`) and the `create-feature` skill uses `values=` in `add_feature_values` examples (the parameter is `entries=`). Both will produce tool-call errors on first use. Fix these immediately.

2. **The troubleshoot-execution skill contains a bogus recovery option.** Line 129 suggests `deriva_ml_update_execution(..., status="Failed", message="...")`. That tool accepts only `description=`. An LLM reading this skill may try the wrong tool before trying `abort_execution`. Fix by removing the status/message option entirely.

3. **No reverse-chronological ordering makes "recent runs" queries expensive.** Every list tool paginates by RID ascending. Finding the last 5 executions on a large catalog requires knowing the total count and paging to the end, or relying on RAG freshness. A `sort_desc=True` or `order="desc"` parameter on `list_executions` would eliminate the most common cross-run comparison pattern.

4. **Feature value retrieval at scale is unsafe.** `deriva_ml_list_feature_values` materializes all records into memory before pagination. On catalogs with millions of feature rows this will OOM or timeout. The tool needs a server-side cursor or a maximum pre-materialization guard.

5. **The cross-user freshness gap is real and underexplained in individual skills.** Only the getting-started prompt documents `deriva_ml_resync_indexes`. Skills like `create-feature` and `troubleshoot-execution` do not mention it. A colleague adding a vocabulary term or creating a feature is invisible to the LLM until reconnect unless the LLM proactively resyncs. The skills should add a one-line reminder.

6. **The `deriva_ml_getting_started` prompt is the system's most important defense against LLM confusion** — and it is not automatically loaded. If the user does not invoke it and the `deriva-ml-context` always-on skill does not fire (e.g., the user asks a very specific tool question without DerivaML-flavored phrasing), the LLM operates on 45 tools with no grounding. Consider making the prompt part of the framework's server-startup greeting or the always-on context skill's content.

7. **File I/O is a hard boundary that the surface does not bridge gracefully.** The asset upload/download split (MCP for catalog metadata, local Python for bytes) is architecturally correct but requires the user to context-switch from the Claude conversation to a local script. The `work-with-assets` skill bridges this by generating the Python, but the LLM cannot verify the script ran, cannot read the output, and cannot diagnose upload failures. A structured "here is the Python you need to run; paste the output back" pattern would help — this is a UX convention the skills could encode more explicitly.
