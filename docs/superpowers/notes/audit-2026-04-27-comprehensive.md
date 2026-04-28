# Comprehensive Audit — deriva-ml-mcp + deriva-ml-skills

**Date:** 2026-04-27
**Scope:** `deriva-ml-mcp` v3.0.0 (MCP plugin) + `deriva-ml-skills` v1.1.0 (Claude Code plugin)
**Method:** Five independent subagent audits, synthesized below
**Source documents:**
- `audit-2026-04-27-architecture.md` — software architecture + deriva-ml API coverage
- `audit-2026-04-27-tests.md` — test coverage (95% line, 287 tests)
- `audit-2026-04-27-core-integration.md` — deriva-mcp-core boundary discipline
- `audit-2026-04-27-skills.md` — skill coverage of tools and workflows
- `audit-2026-04-27-agent-design.md` — LLM-leverage walkthroughs of 5 user goals

---

## Executive summary

The `deriva-ml-mcp` plugin and its companion `deriva-ml-skills` are in good shape architecturally and well-tested. The plugin holds the contract with `deriva-mcp-core` cleanly, exposes the right ~70-90% of the `deriva-ml` Python API, and gets 95% line coverage from a 287-test suite that pins all 11 v3.0 wire breaks.

The skills layer, however, contains **three documented bugs that will cause runtime failures** when an LLM follows them: a `workflow_name=` parameter that doesn't exist, a `values=` parameter that should be `entries=`, and a `status=`/`message=` argument set on `update_execution` that the tool doesn't accept. These ship in skill text across at least 6 files and represent the highest-severity findings in the audit.

The agent-system design has one structural friction at its core: **no reverse-chronological ordering on list tools** makes "show me the last 5 runs" — one of the most common ML-development queries — expensive on any catalog of meaningful size. Combined with `list_feature_values` materializing all records before pagination, the system has scalability rough edges that will surface as catalogs grow.

The remaining findings are smaller. Coverage gaps in tests are confined to a handful of error paths and reindex-failure swallows. Coverage gaps in tools are three real but bounded missing capabilities (`Dataset.list_executions`, `find_assets` cross-table, `find_incomplete_executions`). Skill content drift from the v3.0 wire breaks exists but is mostly contained.

**Findings sorted by severity:**

| # | Severity | Finding | Source |
|---|---|---|---|
| 1 | **Critical (correctness)** | 4 skill content bugs cause tool-call failures | Skills + Agent |
| 2 | **High (UX)** | No reverse-chrono sort; "last N" requires page-to-end | Agent |
| 3 | **High (scalability)** | `list_feature_values` materializes all rows pre-pagination | Agent |
| 4 | **High (UX)** | `resync_indexes` cross-user freshness has zero skill coverage | Skills + Agent |
| 5 | **Medium (UX)** | `getting_started` prompt is load-bearing but not auto-injected | Agent |
| 6 | **Medium (correctness)** | 3 missing deriva-ml tools (`list_executions` per dataset, `find_assets`, `find_incomplete_executions`) | Architecture |
| 7 | **Medium (deployment)** | `deriva-mcp-core` declared without lower-bound version pin | Core integration |
| 8 | **Medium (test gap)** | 3 resource error paths, 1 preflight branch, 5 reindex-swallow branches untested | Tests |
| 9 | **Low (technical debt)** | 3 sites access `deriva-ml` private attributes | Core integration |
| 10 | **Low (technical debt)** | Mock factory duplication across 4 test files | Tests |

---

## 1. Software architecture

### What works

- **Module structure is coherent.** Domain-per-file factoring is consistent (`tools/dataset/{read,mutate,complex}.py`, `tools/feature.py`, `tools/workflow.py`, `tools/execution.py`, `tools/asset.py`). Only `execution.py` at 1338 lines is approaching the threshold that triggered the dataset split (~1745 lines). A future read/mutate split for execution is the natural next refactor but isn't urgent.
- **`_helpers.py` is genuinely shared** across all five domains (pagination, error envelope, table utilities, RID extraction). No copy-paste duplication.
- **Dual-mode policy (tools + resources sharing `_*_impl` helpers) is followed in 10 of 11 resource+tool pairs.** The one intentional deviation is `deriva_ml_get_execution` (returns `ExecutionSummary`) vs `ml/execution/{rid}` (returns `ExecutionDetail`). The deviation is documented but the tool docstring should note that callers wanting full detail should use the resource.
- **The Pydantic response-model migration is ~95% complete after v3.0.** All 25 mutating tools, all `_*_impl` helpers, and all summary helpers return Pydantic instances. Remaining `json.dumps` calls are: argument-validation early returns (which structurally don't need a model) and 4 small read paths (`bag_info`, `get_dataset_spec`, `list_dataset_members` summary mode, `lookup_workflow_by_url` no-result branch).
- **Audit + re-index pattern is consistently applied.** Every `mutates=True` tool emits `audit_event` on success and routes failures through `_error_envelope` (which emits `<op>_failed`). All catalog-mutating tools call `_reindex_<entity>` after the catalog mutation. Lazy imports inside tool bodies (per the documented anti-circular-import pattern) are followed everywhere.
- **No cycles, no leaky abstraction of deriva-py exceptions, no orphaned helpers.**

### What needs attention

**`execution.py` size.** At 1338 lines with 11 tools and 5 helpers, this file is on the same trajectory that triggered the dataset split. Not urgent (the file is still readable), but worth keeping on the radar for the next major refactor pass.

**Three `deriva-ml` private-attribute accesses.** Not boundary violations against `deriva-mcp-core`, but real coupling risks:
- `tools/workflow.py:392`: `ml._add_workflow(wf)`
- `tools/execution.py:746`: `ml._execution = None`
- `tools/dataset/mutate.py:529`: `_set_row_description(ml, ml._dataset_table, ...)`

These represent gaps in `deriva-ml`'s public API. They should be tracked as upstream issues in `deriva-ml` to expose proper public methods. A future `deriva-ml` refactor that renames any of these would silently break the plugin.

**`_summarize_upload_dict` returns a plain dict, not Pydantic.** Minor inconsistency with the v2.2 sweep. The dict is immediately wrapped in `CommitExecutionReport(**summary)` so the wire boundary is still typed; but the helper itself is untyped. Cleanup item.

---

## 2. Coverage of `deriva-ml`

### Coverage by mixin

| Mixin | Coverage | Notes |
|---|---|---|
| `WorkflowMixin` | ~90% | Only internal `_add_workflow` omitted |
| `FeatureMixin` | ~80% | Minor gap: `Feature.list_workflow_executions` |
| `DatasetMixin` (mixin + class) | ~70% | One real gap: `Dataset.list_executions` |
| `ExecutionMixin` (mixin + class) | ~65% | Two gaps: `find_incomplete_executions`, `find_experiments` |
| `AssetMixin` | ~55% | One real gap: `find_assets` (cross-table) |
| `VocabularyMixin` | 0% (delegated to core) | Correct by design |
| `DerivaML` base | ~10% | Most omissions intentional (filesystem, schema admin, Python-only APIs) |

### Three real coverage gaps worth filing as issues

1. **`Dataset.list_executions(dataset_rid)`** — no tool to enumerate which executions produced (or consumed) a given dataset. Callers must filter `deriva_ml_list_executions` results manually. Useful for "who used this dataset?" lineage queries. Suggested tool: `deriva_ml_list_dataset_executions(dataset_rid=...)`.

2. **`AssetMixin.find_assets()`** — `deriva_ml_list_assets` requires a known `asset_table` argument. There's no cross-table search. An LLM that has a RID or filename and doesn't know which asset table it lives in cannot discover the table without iterating. Suggested tool: `deriva_ml_find_assets(rid=..., filename=..., asset_type=...)`.

3. **`ExecutionMixin.find_incomplete_executions()`** — no tool to surface executions stuck in non-terminal states (`Created`, `Running`, `Stopped`, `Pending_Upload`). Useful for recovery workflows. Suggested tool: `deriva_ml_list_incomplete_executions()`.

### Intentional non-coverage (correct)

All file-I/O methods (`download_dataset_bag`, `add_files`, `upload_assets`, `cache_denormalized`), Python-session objects (`pathBuilder`, DataFrame returns), destructive maintenance (`gc_executions`, `clean_execution_dirs`), and schema/annotation admin are correctly omitted. The MCP server has no writable local filesystem from the calling user's perspective; the rest belong to `deriva-skills` (tier-1) or `deriva-mcp-core`.

---

## 3. Test coverage

### Quantitative

- **287 tests pass, 0 fail, 0 skip.** 14 deselected (integration suite gated by `@pytest.mark.integration`).
- **95% overall line coverage** (1763 / 1848 statements).
- **Per-module:** every module ≥ 86% coverage except auto-generated `_version.py`. `_helpers.py` 86%, `resources/ml.py` 95%, `resources/rag.py` 94%, all tool modules 91-100%.
- **All 11 v3.0 wire breaks are pinned by exact value-level assertions.** This is the strongest part of the suite.

### Qualitative strengths

- **Wire-shape discipline is uniform.** Every test does `payload = json.loads(result)` and asserts on JSON keys/values. No test inspects internal Python state in place of the wire shape.
- **Audit assertions on both branches.** Every `mutates=True` tool has both a success-path audit assertion (`_success_calls(mock_audit, "deriva_ml_<op>")`) and a failure-path audit assertion (`<op>_failed`). The dual-patch helper handles the two-bind-site problem cleanly.
- **Async harness is uniform** (`asyncio_mode = "auto"`), no test mixes asyncio with anyio or sync wrappers.
- **Tests are independent.** Function-scoped fixtures everywhere; `_set_plugin_context(None)` cleanup in `conftest.py`. Running any single file passes standalone.
- **Test naming and organization are intuitive** — `test_<tool>_<scenario>` with section-comment delimiters. No orphan tests.

### Gaps to close

1. **3 resource error paths untested** — `ml/asset-tables`, `ml/asset/{rid}`, `ml/registries` in `resources/ml.py:320-321, 348-349, 376-377` have no exception-path tests. One test each would close this. Estimated effort: 30 minutes.

2. **`update_workflow` reindex untested** — workflow.py:511-513 calls `_reindex_workflow` on success, but no test asserts it. The analogous `create_workflow` reindex test exists. Same for 4 dataset mutation tools (`add_dataset_members`, `delete_dataset_members`, `update_dataset`, `increment_dataset_version`) which all have untested reindex exception-swallow branches.

3. **`find_workflow_executions` missing preflight test** — execution.py:528-530 is the only uncovered preflight branch across all paginated tools. A two-line test modeled on `test_list_executions_preflight` would close it.

4. **`create_execution_dataset` `dataset_types=None` case** — the v3.0 contract says the field is always present (null when omitted). Only the non-null branch is currently tested.

5. **Mock factory duplication.** `_make_dataset_mock`, `_make_workflow_mock`, and `_make_execution_record_mock` are defined in 2-3 separate test files. A shared `tests/_factories.py` would consolidate them. Negligible runtime impact; matters for maintainability.

### Test design quality

Low brittleness, high assertion specificity, focused fixtures. Two minor exceptions: (a) preflight `action_required` assertions check key presence but not message content (a broken preflight returning empty string would pass), and (b) error-path tests use substring matching on exception messages (`assert "kaboom" in payload["error"]`), which is slightly brittle to rewording.

---

## 4. `deriva-mcp-core` integration

### Boundary discipline — fully compliant

- **Connection management** centralized in `ml_context.py` via `get_request_credential()`. No raw catalog opens.
- **Auth/credential handling** entirely delegated to core. No `os.environ` reads, no `keyring`.
- **Audit machinery** entirely from `deriva_mcp_core.telemetry`. No competing implementation.
- **RAG per-user safety contract is the most rigorous item.** `ctx.rag_dataset_indexer` is provably unused (two CI pins: one at the plugin level, one at the RAG module level). Per-RID source naming is pinned end-to-end. The `vocab:` prefix carve-out for catalog-public content is separately pinned. This is the most security-sensitive boundary and it is the most thoroughly tested.
- **Vocabulary primitives** delegated to core's `add_term`/`lookup_term`/`delete_term`. The plugin's `tools/maintenance.py` ships only RAG-indexing tools (`reindex_vocabularies`, `resync_indexes`), not term CRUD.
- **Generic Deriva primitives** never duplicated. Where a low-level write is needed (Asset/Dataset description has no setter in `deriva-ml`), the plugin uses `ml.pathBuilder()` — `deriva-ml` API, not raw ERMrest.

### Plugin-authoring-guide conformance — fully compliant

- All 46 tool registrations carry explicit `mutates=True/False` (no bare `@ctx.tool()` that would raise `TypeError` at startup).
- All DERIVA I/O wrapped in `with deriva_call():`.
- All mutating tools emit audit on both success and failure (failure path centralized in `_error_envelope`, which makes it structurally impossible to forget).
- Tool naming, prompt naming, and audit event naming all follow the `deriva_ml_<op>` convention.

### Findings

**Two upstream-tracked gaps:**
- `# TODO(upstream-rag-doctype)` in `resources/rag.py:428, 903` — tracks `deriva-mcp-core#2` (no per-table `doc_type` filter on RAG chunks).
- `# TODO(deriva-ml-execution-metadata-api)` in `tools/execution.py:275` — no `deriva-ml` API to enumerate `Execution_Metadata` files. The `ml/execution/{rid}` resource omits the `metadata` key as a result. This is a `deriva-ml` upstream gap, not a `deriva-mcp-core` one.

**Version-pinning concern.** `pyproject.toml:26` declares `deriva-mcp-core` with no version constraint. In development the `[tool.uv.sources]` editable-install path covers this, but a published release could be installed against a too-old core that lacks `resolve_user_identity`, `get_rag_store`, etc. **Recommend adding a lower-bound pin** like `"deriva-mcp-core>=<current-version>"`.

**Import-surface contract uncertainty.** Five RAG-extension imports from internal submodule paths (`deriva_mcp_core.rag.chunker`, `.data`, `.store`) are not enumerated as stable public exports in core's `__init__.py`. If core reorganizes its RAG package, the plugin's import paths break. Either confirm these paths are intentionally stable, or request that core re-exports them from `deriva_mcp_core.rag`.

---

## 5. Skills coverage

### What works

- **Tool coverage is broad.** 41 of 43 mutating/reading tools (excluding the two maintenance tools) are mentioned by at least one skill. Most are covered by 3-7 skills with clear primary teaching skills.
- **`api-naming-conventions`, `dataset-lifecycle`, `execution-lifecycle`, `create-feature`, `troubleshoot-execution`, `model-development-workflow`** form a solid primary teaching set covering the major workflows.
- **Cross-references between skills are consistent.** `execution-lifecycle` correctly points to `dataset-lifecycle`, `create-feature`, `ml-data-engineering`, `configure-experiment`, etc. Tier-1 references use the `(tier-1; deriva-skills)` annotation convention.
- **`deriva-ml-context` (always-on) carries its load-bearing role correctly.** The steering principle (DerivaML abstractions > raw primitives), the stateless model, and the vocabulary-extension pattern are all present and accurate. This skill alone prevents the most common LLM mistake: using raw entity CRUD for ML domain objects.

### What's broken

**Critical: skill content bugs that will cause tool-call failures.**

1. **`workflow_name=` phantom parameter.** `model-development-workflow/SKILL.md` (lines 146, 191) passes `workflow_name=` to `deriva_ml_create_execution`. The actual parameter is `workflow_rid=`. An LLM following this skill will produce a tool call that errors immediately. **Fix:** rename to `workflow_rid=` in both call sites.

2. **`values=` vs `entries=` mismatch.** `create-feature/SKILL.md` (lines 166-188) uses `values=` in all `deriva_ml_add_feature_values` examples. The actual parameter is `entries=`. **Fix:** rename to `entries=` in all examples.

3. **`update_execution` accepts `status=`/`message=` (false).** Multiple skill files document the call signature `deriva_ml_update_execution(..., status="Failed", message="...")`. The actual tool only accepts `description=`. Status changes go through `start_execution`/`commit_execution`/`abort_execution`. Affected files:
   - `troubleshoot-execution/SKILL.md:129`
   - `execution-lifecycle/references/workflow.md:38, 123`
   - `execution-lifecycle/references/concepts.md:114, 120`
   - `run-notebook/references/workflow.md:262`
   - `api-naming-conventions/SKILL.md:179`

4. **Phantom `status` field in `denormalize_dataset` example.** `ml-data-engineering/references/denormalize-guide.md:91` shows `"status": "success"` in a `deriva_ml_denormalize_dataset` response. The tool has no `status` field — it uses `mode` as the discriminator.

**Two completely orphan tools.**

5. **`deriva_ml_reindex_vocabularies`** has zero skill coverage. After adding/removing vocabulary terms via core's `add_term`/`delete_term`, this tool must be called to refresh the RAG vocab index. Skills teach term addition (via `add_term`) but never mention the follow-up reindex.

6. **`deriva_ml_resync_indexes`** has zero skill coverage. This is the documented v1.4 cross-user freshness bridge — when user B needs to see user A's mutations to shared-visible datasets, this tool refreshes B's per-user RAG sources. The plugin's CLAUDE.md explicitly documents this as the manual bridge for the cross-user staleness gap, but no skill surfaces it. **The most impactful UX gap in the system.** A user experiencing "I can't see the dataset my colleague just created" has no skill-level guidance.

**Smaller skill issues.**

7. **`deriva_ml_cache_dataset(asset_rid="...")`** — `manage-storage/SKILL.md:184` shows an `asset_rid=` argument. The actual parameter is `dataset_rid=`. Minor but real.

8. **Singular vs plural in `deriva-ml-context`.** The abstractions table lists `deriva_ml_add_feature_value` (singular). Correct name is `deriva_ml_add_feature_values` (plural).

9. **Skill count discrepancy.** CLAUDE.md mentions 24 tier-2 skills; the actual count is 23. Off-by-one in the prose.

### Workflow coverage scorecard

10 ML/catalog workflows scored:

| # | Workflow | Status |
|---|---|---|
| 1 | Create dataset from catalog data | ✅ Covered |
| 2 | Register new model architecture | ⚠ Partial — missing Git URL/checksum recipe |
| 3 | Run training execution | ✅ Covered |
| 4 | Define + record features | ✅ Covered |
| 5 | Compare model runs | ⚠ Partial — pieces exist in separate skills, no synthesis |
| 6 | Cache dataset locally | ✅ Covered |
| 7 | Split dataset (train/test/val) | ✅ Covered |
| 8 | Recover from failed execution | ✅ Covered (with bug #3 above) |
| 9 | Cross-user dataset visibility | ❌ Not covered |
| 10 | Explore unfamiliar catalog | ⚠ Partial — pieces in individual skills, no first-visit recipe |

**Score: 4/10 fully covered, 4/10 partially, 2/10 not.**

Most-impactful gaps to close:
- **Workflow 9 (cross-user visibility).** A `cross-user-data-sync` skill (or a section in `troubleshoot-execution`) that says "when collaborator changes are missing, call `deriva_ml_resync_indexes`" would close this.
- **Workflow 5 (compare runs).** A "compare model runs" section in `create-feature` or `execution-lifecycle` showing the find-executions → query-features-per-execution → aggregate pattern.
- **Workflow 2 (register workflow).** A concrete recipe for computing `git rev-parse HEAD` and the GitHub URL before calling `deriva_ml_create_workflow`.

---

## 6. Agent-system design

This is the most consequential perspective: does the surface give an LLM enough leverage to do real work? The answer is **mostly yes, with three structural friction points and the four content bugs already named**.

### Per-goal scores

| Goal | Score | Top friction |
|---|---|---|
| 1 — Explore unfamiliar catalog | 3/5 | No cold-start prompt auto-injection; schema discovery leaks to tier-1 |
| 2 — Develop new model | 3/5 | `workflow_name` phantom; `entries`/`values` mismatch in skills |
| 3 — Compare model runs | **2/5** | No reverse-sort on list tools; no cross-execution feature query |
| 4 — Manage catalog growth | 3/5 | `metadata=` naming ambiguity; schema RAG re-index gap after feature creation |
| 5 — Recover from failure | 3/5 | Bogus `status=` option in `troubleshoot-execution` |

The lowest score (Goal 3, comparing runs) reflects two structural issues that aren't bugs but are real frictions:

### The three structural frictions

**A. No reverse-chronological ordering on list tools.** Every `list_*` tool paginates by RID ascending. Finding "the last 5 executions" on a catalog of any meaningful size requires either:
- A preflight count, then paging to the end (expensive and round-trip-heavy), or
- Reliance on RAG freshness (subject to the cross-user freshness gap)

This is a real UX cliff for what should be the most natural ML-development query. Adding `sort_desc=True` or `order="desc"` to `list_executions` (and ideally all list tools) would eliminate it. Estimated implementation effort: medium — needs upstream `deriva-ml` support for descending iteration, or post-fetch reversal with a cap.

**B. `list_feature_values` materializes all rows before pagination.** `records = list(ds.feature_values(...))` loads everything into memory, then paginates the list. On a catalog with millions of feature values this will OOM or timeout before the LLM ever sees a page. Even `preflight_count=True` triggers full materialization. **This is a known scalability cliff.** Solutions: server-side cursor in `deriva-ml`, or a tool-level guard that refuses to materialize beyond a threshold and returns a clear "use a server-side query" error.

**C. No cross-execution feature value query primitive.** Comparing metrics across 5 runs requires 5 sequential `list_feature_values(selector="by_execution", ...)` calls + in-context aggregation. There's no "give me the F1 score from each of these executions" tool. For comparison workflows this means N+1 round-trips and N×payload context cost. A `deriva_ml_compare_executions(execution_rids=[...], feature_name=..., column=...)` tool would close this — the use case is concrete and the cost is bounded.

### Tool design — strong overall, two blind spots

**Strong:**
- `deriva_ml_` prefix is consistent and prevents tier-1 collisions.
- Verb clarity is high (`create`/`list`/`get`/`find`/`delete`/`add`/`update`/`commit`/`abort`).
- Pagination contract is uniform across all list tools.
- Status enum values are enumerated in the execution-lifecycle prompt and in tool docstrings.
- Inline error messages for state-machine rejections are excellent (e.g., `"cannot start execution in state Stopped; only Created (will start) or Running (no-op) are valid"`).

**Two blind spots:**
- **`metadata=` parameter naming on `create_feature`.** Accepts `list[str | dict]` for "extra scalar or FK columns added to the feature table." The name "metadata" is ambiguous — sounds like key-value metadata about the feature itself, not extra columns. The skill explains this; the tool docstring doesn't.
- **`workflow_type=` valid values not enumerated in docstring.** `create_workflow` accepts `str | list[str]` but the docstring says "Each must be a term in the `Workflow_Type` vocabulary" without listing terms. Vocabulary varies per catalog so the docstring can't enumerate them all, but pointing to `registries` resource or `list_vocabulary_terms` would help. The LLM has to do a prior lookup or risk an FK violation.

### Resource utility

Resources are used in skills but unevenly:
- `execution-lifecycle`: 3 resource references
- `dataset-lifecycle`: 2 resource references
- `create-feature`: 0 resource references (uses RAG + typed tools)
- `troubleshoot-execution`: 1 resource reference
- `model-development-workflow`: 0 resource references

The `getting_started` prompt enumerates 10 resource URIs cleanly. Skills lean toward typed tools, which is defensible (tools handle errors uniformly) but means resources are underused. Resources are typically faster (one network call) and return richer detail (execution detail bundles inputs+outputs+experiment). The `execution-lifecycle` skill correctly points to the detail resource for post-run verification.

**Underused resources:** `ml/asset/{rid}` and `ml/asset-tables` are only mentioned in the prompt, never in `work-with-assets`.

### Prompts — three load-bearing, three missing

**The 3 existing prompts are good:**
- **`deriva_ml_getting_started`** (the most important) is comprehensive: stateless model, pagination contract, error envelope, five domains, resource URIs, RAG patterns, name resolution, mutation chain, asset I/O split. ~370 lines but dense and mostly non-redundant. **Without this prompt the LLM operates on 45+ tools with no grounding** — but the prompt is not auto-injected. Either users invoke it explicitly or the always-on `deriva-ml-context` skill must trigger.
- **`deriva_ml_execution_lifecycle`** correctly describes the state machine, the lifecycle tools, the hybrid dispatch on `add_feature_values`, and the two pitfalls.
- **`deriva_ml_workflow_dedup`** is short and correct; the anti-pattern/correct-pattern pair is exactly what's needed.

**Three prompts that should exist:**
- **`deriva_ml_feature_design`** — feature design mental model (term vs asset vs scalar, single vs multi-column, selector strategy). Currently only in the `create-feature` skill, which the LLM may not load.
- **`deriva_ml_dataset_versioning`** — semantic versioning, when to bump, how to pin in configs. Currently spread across 3 skills.
- **`deriva_ml_rag_freshness`** — the cross-user freshness gap and `resync_indexes` recovery. Easy to miss in the getting-started prompt.

### Provenance + reproducibility

**The surface makes provenance hard to skip by accident:**
- `add_feature_values` requires `execution_rid=`.
- `create_execution_dataset` requires `execution_rid=`.
- The `deriva-ml-context` skill explicitly warns against raw `insert_entities`.
- `dry_run=True` is the documented "test without recording" path.

**One soft gap:** `deriva_ml_create_dataset` does not require an execution context. Datasets can be created without provenance linkage. This is correct for bootstrap (the first dataset of a project has no prior execution to link to), but it's the one place an LLM can accidentally skip provenance.

**Hard gap:** `DerivaMLDirtyWorkflowError` is enforced by `deriva-ml-run` (the CLI) but not by MCP execution tools. An LLM driving the lifecycle entirely through MCP can create executions without a clean git state. The provenance record won't have a verifiable git hash. The skills document this; the tools can't enforce it.

### Failure modes specific to LLMs

| Hallucination type | Protection? |
|---|---|
| RID hallucination | Tool returns `{"error": ...}`; skills steer to lookup-before-operate |
| Feature/column name hallucination | `rag_search` and `list_features` cover this; skills teach the lookup-first pattern |
| Vocabulary term hallucination | Catalog FK violation propagates; error text isn't always actionable |
| Status enum hallucination | Python enum error text propagates; valid values are in the prompt |
| **Skill-induced phantom params** | **No protection — skills tell the LLM to use parameters that don't exist** |

The last row is the most dangerous because it's the hardest to recover from. An LLM that gets `{"error": "unexpected keyword argument 'workflow_name'"}` will likely retry rather than question whether the skill text is wrong.

---

## Action items (prioritized)

### Critical — fix before any new user onboards

1. **Fix the `workflow_name=` phantom parameter** in `model-development-workflow/SKILL.md` (rename to `workflow_rid=`).
2. **Fix the `values=` → `entries=` rename** in `create-feature/SKILL.md`.
3. **Fix the `update_execution(status=, message=)` false signature** across 5 files (remove the option entirely; redirect to `abort_execution`).
4. **Fix the phantom `status` field in `denormalize_dataset`** example in `ml-data-engineering/references/denormalize-guide.md`.

These are 4 file edits totaling maybe 20 lines of content. They prevent runtime tool-call failures.

### High — close the largest UX gaps

5. **Add a `cross-user-data-sync` skill** (or a section in `troubleshoot-execution`) covering `deriva_ml_resync_indexes`. This closes Workflow 9 and surfaces orphan tool #6.
6. **Add a `compare-model-runs` section** to `create-feature` or `execution-lifecycle` that synthesizes find-executions → query-features-per-execution → aggregate.
7. **Auto-inject the `getting_started` prompt** into the `deriva-ml-context` always-on skill, so an LLM doesn't depend on the user invoking it.

### High — close the structural frictions

8. **Add `sort_desc=True` (or `order=`) to `list_executions`** (and ideally all list tools). The "last N runs" query is the most common ML-development pattern; the current design forces an expensive workaround.
9. **Add a server-side cursor or pre-materialization guard to `list_feature_values`.** OOM on millions of rows is a real risk.
10. **Add a `compare_executions` tool** for cross-execution feature value queries.

### Medium — coverage gaps

11. **File three deriva-ml-mcp issues:**
   - `deriva_ml_list_dataset_executions(dataset_rid)` for dataset → executions lineage
   - `deriva_ml_find_assets(rid=, filename=, ...)` for cross-table asset search
   - `deriva_ml_list_incomplete_executions()` for recovery workflows

12. **Three test-coverage micro-gaps:**
   - 3 resource error-path tests (`ml/asset-tables`, `ml/asset/{rid}`, `ml/registries`)
   - 1 `find_workflow_executions` preflight test
   - `update_workflow` reindex assertion + 4 `_reindex_dataset` exception-swallow tests

13. **Add `deriva-mcp-core` lower-bound version pin** to `pyproject.toml`.

### Low — technical debt

14. **Track three deriva-ml private-attribute accesses as upstream issues** (`ml._add_workflow`, `ml._execution`, `ml._dataset_table`). Either expose proper public methods upstream, or document the workarounds with explicit deriva-ml version compatibility notes.

15. **Consolidate mock factories** into `tests/_factories.py` to remove duplication across `test_dataset.py`, `test_resources.py`, `test_execution.py`, `test_workflow.py`.

16. **Sweep `_summarize_upload_dict` to return Pydantic** for v2.2 consistency. Minor.

17. **Add three new prompts** (`deriva_ml_feature_design`, `deriva_ml_dataset_versioning`, `deriva_ml_rag_freshness`).

18. **Update CLAUDE.md** to fix the 24-vs-23 skill-count off-by-one and update the resource-count from 9 to 11 in `resources/ml.py` module docstring.

---

## What's notably good

To balance the action-item list, three things deserve explicit recognition:

1. **The RAG per-user safety contract is exemplary.** The architectural decision to ban `ctx.rag_dataset_indexer` (which would leak per-user ACL rows), the per-RID source-naming scheme, the vocab `vocab:` prefix carve-out, and the dual CI pins enforcing both the ban and the naming scheme — this is the most security-sensitive boundary in the system and it's the most thoroughly tested. The plugin's `CLAUDE.md` explains the rationale clearly enough that a future contributor can't accidentally regress.

2. **The v3.0 wire-break test coverage is unusually disciplined.** All 11 wire breaks are pinned by exact value-level assertions. This is the kind of contract testing that prevents silent regressions; most projects this size have a few wire breaks that drift undetected.

3. **`deriva-ml-context` is the right load-bearing always-on skill.** The steering principle ("DerivaML abstractions take precedence over raw catalog primitives") is stated cleanly with six concrete reasons, the stateless model is explained, and the vocabulary-extension pattern is correct. This single skill prevents the most common LLM mistake in deriva-ml-loaded catalogs.

---

## Cross-document references

- Architecture details + per-mixin coverage tables: `audit-2026-04-27-architecture.md`
- Test coverage report + per-module breakdown: `audit-2026-04-27-tests.md`
- Boundary discipline + plugin-authoring-guide conformance: `audit-2026-04-27-core-integration.md`
- Skill coverage matrix (40 tools × 23 skills) + workflow-by-workflow notes: `audit-2026-04-27-skills.md`
- Per-goal LLM walkthroughs + tool-by-tool design notes: `audit-2026-04-27-agent-design.md`
