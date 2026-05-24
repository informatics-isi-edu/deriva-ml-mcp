# Architecture & Coverage Audit — deriva-ml-mcp

*Audited: 2026-04-27. Auditor: Claude Sonnet 4.6 (independent pass).*

> **Status note (2026-05-24):** This audit was originally drafted
> against deriva-ml's pre-v1.39 surface. Updated 2026-05-24 to reflect
> the unified `commit_output_assets` surface that shipped in deriva-ml
> v1.39.0 (see
> [ADR-0009](https://github.com/informatics-isi-edu/deriva-ml/blob/main/docs/adr/0009-unified-commit-output-assets.md)).
> Method names in the coverage tables below are current as of that
> release: `upload_execution_outputs` / `upload_outputs` /
> `upload_pending` are gone, replaced by `commit_output_assets` (on
> `Execution`) and `commit_pending_executions` (on `DerivaML`).

---

## Part A: Software Architecture

### Module structure

- **Line counts (approx):** `execution.py` 1338 lines, `resources/rag.py` 1078 lines, `_response_models.py` 1125 lines, `tools/dataset/read.py` 746, `tools/dataset/mutate.py` 734, `tools/dataset/complex.py` 539, `tools/feature.py` 696, `tools/workflow.py` 532, `tools/asset.py` 525. Only `execution.py` and `resources/rag.py` are above 800 lines.

- **`execution.py` at 1338 lines** is the one module worth flagging. It contains 5 read tools, 6 mutation tools, and 5 shared helpers (`_summarize_execution`, `_list_executions_impl`, `_get_execution_detail_impl`, `_summarize_upload_dict`, and a `register` aggregator). The dataset domain was already split into read/mutate/complex at 17 tools/1745 lines; execution at 11 tools/1338 lines is growing toward the same threshold. A read/mutate split would bring it under 700 lines per file, but it isn't urgent today.

- **`resources/rag.py` at 1078 lines** is large but not incoherent — it's a single subsystem (per-user RAG hooks, vocabulary indexing, and surgical `_reindex_*` helpers) with no natural split point. The complexity is inherent to the per-user safety contract; the module docstring thoroughly explains the design.

- **`_response_models.py` at 1125 lines** is an intentional single-file Pydantic model registry. All ~50 models in one file means `grep` and cross-referencing stay trivial; no split is warranted.

- **Dataset subpackage** (`__init__.py` + read/mutate/complex) is coherent. The package-level `audit_event` re-export trick (`_pkg.audit_event`) to maintain a single patch site across three submodules is slightly non-obvious but documented in the `__init__.py` docstring and preserves a clean test surface.

- **No cycle / coupling issues found.** The one intentional import-cycle (`mutate.py` imports `deriva_ml_mcp.tools.dataset` at module load time) is documented and resolved by deferring submodule imports inside `register()`.

### Separation of concerns

- Each module has one clear responsibility: `ml_context.py` builds DerivaML instances, `_helpers.py` owns pagination/error-envelope/table-utils, `_response_models.py` owns wire shapes, `tools/<domain>` files own tool registration, `resources/` files own resource registration and RAG indexing.

- `_helpers.py` is genuinely shared: `_error_envelope`, `_paginate`, `_read_rid`, `_table_qname`, `_table_to_dict`, `_set_row_description`, `_row_rid_for` are all called from multiple domain modules. No copy-paste found.

- `_summarize_*` helpers in each domain file are domain-specific (not copy-pasted cross-module) and return typed Pydantic instances after the v2.2 sweep.

- The `_summarize_upload_dict` helper in `execution.py` (line 314) returns a plain `dict`, not Pydantic. It feeds `CommitExecutionReport(**summary)` immediately, so the Pydantic boundary is still honored; but the helper itself is untyped. Minor inconsistency with the v2.2 pattern; not a correctness issue.

### Dual-mode policy compliance

| Domain | List tool | Detail tool | List resource | Detail resource | Shared `_*_impl` helper |
|---|---|---|---|---|---|
| Dataset | `deriva_ml_list_datasets` | `deriva_ml_get_dataset` | `ml/datasets` | `ml/dataset/{rid}` | `_list_datasets_impl`, `_get_dataset_detail_impl` |
| Workflow | `deriva_ml_list_workflows` | `deriva_ml_get_workflow` | `ml/workflows` | `ml/workflow/{rid}` | `_list_workflows_impl`, `_get_workflow_impl` |
| Execution | `deriva_ml_list_executions` | `deriva_ml_get_execution` | `ml/executions` | `ml/execution/{rid}` | `_list_executions_impl`, `_get_execution_detail_impl` |
| Feature | `deriva_ml_list_features` | `deriva_ml_get_feature` | `ml/features/{table}` | — | `_list_features_impl` |
| Asset | `deriva_ml_list_assets` | `deriva_ml_lookup_asset` | `ml/asset-tables` | `ml/asset/{rid}` | `_list_asset_tables_impl`, `_get_asset_detail_impl` |

**One gap:** features have no per-feature detail resource (`ml/feature/{table}/{name}`). The tool `deriva_ml_get_feature` exists and returns the full schema, but there is no corresponding `deriva://catalog/{h}/{c}/ml/feature/{table}/{name}` resource. Features are schema objects, not data rows, so a resource here would be unusual (no stable RID-based URI), but it's worth noting for consistency. Not a defect under the current policy — the CLAUDE.md says "fixed-shape with no filters = resource" and feature detail has two required keys (table + name), making it more tool-like.

**`deriva_ml_get_execution` tool returns `ExecutionSummary` shape** (7 fields: rid/workflow_rid/status/description/start_time/stop_time/duration), while the `ml/execution/{rid}` resource returns `ExecutionDetail` (summary + inputs + outputs + experiment). The `deriva_ml_get_execution` tool does NOT call `_get_execution_detail_impl`. This is a documented intentional difference — `get_execution` is a lightweight lookup, the resource is the bundled view. The CLAUDE.md's dual-mode policy says shapes MUST match when the helper is shared; here the tool intentionally returns a subset, which is permissible but the docstring (line 456–466) should note that callers wanting full detail should use the resource or call `_get_execution_detail_impl` explicitly.

### Audit + re-index pattern compliance

Scan results for `mutates=True` tools:

| Tool | audit_event on success | audit_event on failure (via _error_envelope) | _reindex_* call |
|---|---|---|---|
| `deriva_ml_create_dataset` | yes (line 128) | yes | yes (`_reindex_dataset`, line 144) |
| `deriva_ml_delete_dataset` | yes (line 201) | yes | yes (`_delete_dataset_source`, line 213) |
| `deriva_ml_add_dataset_members` | yes (line 309) | yes | yes (`_reindex_dataset`, line 319) |
| `deriva_ml_delete_dataset_members` | yes (line 389) | yes | yes (`_reindex_dataset`, line 399) |
| `deriva_ml_update_dataset` | yes (line 535) | yes | yes (`_reindex_dataset`, line 547) |
| `deriva_ml_add_dataset_element_type` | yes (line 630) | yes | yes (`_reindex_dataset`, line 711) |
| `deriva_ml_increment_dataset_version` | yes (line 700) | yes | yes (`_reindex_dataset`, line 711) |
| `deriva_ml_cache_dataset` | yes (line 131) | yes | no — intentional: `cache_dataset` does not change catalog data rows, only local filesystem |
| `deriva_ml_split_dataset` | yes (line 489) | yes | yes (`_reindex_dataset` for each split RID, line 510) |
| `deriva_ml_denormalize_dataset` | `mutates=False` | — | — |
| `deriva_ml_create_workflow` | yes (line 394) | yes | yes (`_reindex_workflow`, line 407) |
| `deriva_ml_update_workflow` | yes (line 500) | yes | yes (`_reindex_workflow`, line 511) |
| `deriva_ml_create_execution` | yes (line 752) | yes | yes (`_reindex_execution`, line 764) |
| `deriva_ml_start_execution` | yes (line 851) | yes | yes (`_reindex_execution`, line 859) |
| `deriva_ml_commit_execution` | yes (line 964) | yes | yes (`_reindex_execution`, line 975) |
| `deriva_ml_update_execution` | yes (line 1058) | yes | yes (`_reindex_execution`, line 1067) |
| `deriva_ml_abort_execution` | yes (line 1143) | yes | yes (`_reindex_execution`, line 1152) |
| `deriva_ml_create_execution_dataset` | yes (line 1218) | yes | yes (both `_reindex_dataset` + `_reindex_execution`, lines 1232-1250) |
| `deriva_ml_add_nested_execution` | yes (line 1306) | yes | yes (`_reindex_execution`, line 1316) |
| `deriva_ml_create_feature` | yes (line 493) | yes | no — intentional: features are schema objects, not RAG-indexed data rows |
| `deriva_ml_delete_feature` | yes (line 549) | yes | no — same rationale |
| `deriva_ml_add_feature_values` | yes (line 669) | yes | no — feature records are not RAG-indexed |
| `deriva_ml_update_asset` | yes (line 495) | yes | no — assets are not in the per-user RAG trio (dataset/workflow/execution) |

**No missing audit_event calls found.** The `cache_dataset` and `denormalize_dataset` omissions are consistent with their documented purposes. All `mutates=True` tools emit `_error_envelope(audit=True)` on failure via `_helpers.py`.

**One idempotent-no-op deviation:** `deriva_ml_start_execution` returns `{"error": "cannot start execution in state ..."}` via raw `json.dumps` (line 840) for terminal-state rejection. This is the error path, but it doesn't go through `_error_envelope` and doesn't emit an audit row. The same pattern appears in `deriva_ml_commit_execution` (line 923) and `deriva_ml_update_execution` (line 1042 — the `description is None` guard). These are argument-validation returns, not exception-caught returns, so the `_error_envelope` wrapping would be unusual here; but the inconsistency means some validation errors are not logged at `ERROR` level. Minor.

### Error handling

- `_error_envelope` is used consistently across all tool try/except blocks. Every exception is caught at the outer level and serialized to `{"error": str(exc)}`, with the full traceback logged server-side (`exc_info=True`). No raw exception trace reaches the wire.

- The bare `except Exception:` with `# noqa: BLE001` comments in `_get_execution_detail_impl` and `_get_asset_detail_impl` are intentional best-effort swallows for optional sub-payloads (inputs/outputs/experiment on execution detail; executions list on asset detail). These are acceptable because the outer payload is still valid and the catch is narrow in scope.

- No leaky abstraction of deriva-py exceptions found. The `DerivaMLException` import in `workflow.py` (line 25) is used to check for a specific exception type; it doesn't re-raise it raw.

### Pydantic response-model coverage

- **All `mutates=True` tool returns** use `model_dump_json(by_alias=True)` as of v3.0. No raw dict returns for mutation responses.

- **Remaining `json.dumps` calls** are confined to:
  1. Argument-validation error returns that pre-date `_error_envelope` and use `json.dumps({"error": ...})` directly — `execution.py` lines 840, 923, 1042; `feature.py` lines 355, 363, 372, 634; `dataset/complex.py` lines 231, 237, 272, 335, 351. These are structurally identical `{"error": ...}` responses; no Pydantic model wraps them (there's no `ValidationErrorResponse` model).
  2. Two read tools that return ad-hoc dicts: `deriva_ml_list_dataset_members` in summary mode (lines 415, 424, 452 — no Pydantic model for the summary sub-path) and `deriva_ml_list_dataset_element_types` (line 608). `deriva_ml_bag_info` (line 667) does `json.dumps(info, default=str)` because the `bag_info()` return shape is upstream-defined and varies. `deriva_ml_get_dataset_spec` (line 728) returns a plain dict. `deriva_ml_list_feature_values` result rows (line 412) call `r.model_dump()` then `json.dumps` — this is fine (the inner records ARE Pydantic).
  3. `workflow.py` lines 481, 485 — the `lookup_workflow_by_url` no-result branch returns `{"url_or_checksum": ..., "message": ...}` as a raw dict instead of a named model.

- The CLAUDE.md's v2.2 note explicitly calls out `_summarize_*` helpers as swept to Pydantic and the ad-hoc payload sites as retired in v2.3. The remaining `json.dumps` uses (listed above) are post-v2.3 survivors that weren't part of those sweeps. Not regressions, but the migration isn't 100% complete.

- **No cases found** where a `_*_impl` helper returns a Pydantic model but the wrapper then calls `json.dumps` on it (which would lose the alias-map). The aliased `model_dump_json(by_alias=True)` pattern is applied correctly everywhere a Pydantic model is returned from an impl helper.

### Lazy imports

- All `from deriva_ml_mcp.resources.rag import _reindex_*` calls are inside tool function bodies (not at module level), consistent with the lazy-import pattern described in CLAUDE.md. This applies across all tool modules: `execution.py` (7 lazy imports), `workflow.py` (2), `dataset/mutate.py` (6), `dataset/complex.py` (1), `tools/maintenance.py` (2).

- `resources/rag.py` itself uses top-level imports from `deriva_mcp_core.rag`, `deriva_ml_mcp._helpers`, `deriva_ml_mcp.ml_context`, and the `tools/dataset`, `tools/execution`, `tools/workflow` impl helpers. These are all intra-plugin imports; the RAG indexer is not loaded at plugin import time because `resources/rag.py` is only loaded when `plugin.py` calls `resources_rag.register(ctx)`, not at MCP server start. Fine.

- `ctx.rag_dataset_indexer` is NOT called anywhere in the plugin source. The only reference is in the module docstring and a comment in `rag.py`'s module docstring explaining why it's banned. The safety contract is preserved.

---

## Part B: deriva-ml Coverage

### DerivaML class methods (from `core/base.py` and mixin classes)

**DatasetMixin (`core/mixins/dataset.py`)**

| Method | Exposed? | Tool name | Notes |
|---|---|---|---|
| `find_datasets` | Yes | `deriva_ml_list_datasets` | |
| `lookup_dataset` | Yes | `deriva_ml_get_dataset` | |
| `delete_dataset` | Yes | `deriva_ml_delete_dataset` | |
| `list_dataset_element_types` | Yes | `deriva_ml_list_dataset_element_types` | |
| `add_dataset_element_type` | Yes | `deriva_ml_add_dataset_element_type` | |
| `download_dataset_bag` | No | — | Intentional: requires local filesystem access on the client side, not the MCP server |
| `estimate_bag_size` | Partial | `deriva_ml_bag_info` | `bag_info` wraps this; `estimate_bag_size` standalone is not exposed |
| `bag_info` | Yes | `deriva_ml_bag_info` | |
| `estimate_denormalized_size` | Yes | `deriva_ml_denormalize_dataset` (size-only mode) | Exposed via the tool's `describe_only=True` branch |
| `cache_dataset` | Yes | `deriva_ml_cache_dataset` | |

**Dataset class methods (`dataset/dataset.py`)**

| Method | Exposed? | Tool name | Notes |
|---|---|---|---|
| `create_dataset` | Yes | `deriva_ml_create_dataset` | Via `Execution.create_dataset` in MCP; direct `DatasetMixin.create_dataset` is not separately exposed but the effect is the same |
| `add_dataset_type` | Partial | `deriva_ml_update_dataset` | Set-style diff exposes the behavior |
| `remove_dataset_type` | Partial | `deriva_ml_update_dataset` | Same |
| `increment_dataset_version` | Yes | `deriva_ml_increment_dataset_version` | |
| `list_dataset_members` | Yes | `deriva_ml_list_dataset_members` | |
| `dataset_history` | Yes | `deriva_ml_get_dataset(include_history=True)` | |
| `current_version` | Yes | Included in `_summarize_dataset` return | |
| `get_chaise_url` | Yes | Included in `deriva_ml_get_dataset` return | |
| `list_dataset_parents` | Yes | `deriva_ml_list_dataset_relations(direction="parents")` | |
| `list_dataset_children` | Yes | `deriva_ml_list_dataset_relations(direction="children")` | |
| `list_executions` | No | — | No tool to list executions for a given dataset RID. Gap — callers must filter `deriva_ml_list_executions` by result. |
| `add_dataset_members` | Yes | `deriva_ml_add_dataset_members` | |
| `delete_dataset_members` | Yes | `deriva_ml_delete_dataset_members` | |
| `get_denormalized_as_dict` | Yes | `deriva_ml_denormalize_dataset` | |
| `get_denormalized_as_dataframe` | No | — | Intentional: DataFrames not serializable to JSON |
| `describe_denormalized` | Yes | `deriva_ml_denormalize_dataset(describe_only=True)` | |
| `list_denormalized_columns` | No | — | No direct exposure; callers use `describe_denormalized` |
| `list_schema_paths` | No | — | Internal schema introspection; rarely needed externally |
| `cache_denormalized` | No | — | Requires local filesystem; intentional |
| `to_markdown` / `display_markdown` | No | — | Intentional: presentation-layer, not catalog-state |
| `prefetch` | No | — | Low-level; no clear LLM use case |
| `find_features` | Partial | `deriva_ml_list_features` (table filter) | Dataset-scoped feature listing is exposed via the Dataset object's `feature_values` (see below) |
| `feature_values` | Yes | `deriva_ml_list_feature_values(dataset_rid=...)` | Dataset-scoped |

**ExecutionMixin (`core/mixins/execution.py`)**

| Method | Exposed? | Tool name | Notes |
|---|---|---|---|
| `create_execution` | Yes | `deriva_ml_create_execution` | |
| `lookup_execution` | Yes | `deriva_ml_get_execution` | |
| `list_executions` | Yes | `deriva_ml_list_executions` | |
| `pending_summary` | No | — | Workspace-level diagnostic; useful but LLM context unclear |
| `find_incomplete_executions` | No | — | Gap: useful for recovery workflows; no LLM tool equivalent |
| `resume_execution` | No (internal) | — | Used internally by start/commit/abort tools; not directly exposed as an "resume" tool, which is correct |
| `gc_executions` | No | — | Intentional: destructive GC not safe to expose unguarded |
| `find_executions` | Yes | `deriva_ml_list_executions` + `deriva_ml_find_workflow_executions` | |
| `lookup_experiment` | Partial | Embedded in `_get_execution_detail_impl` | Not a standalone tool; accessible only via the execution detail resource |
| `find_experiments` | No | — | Gap: no tool to list experiments independent of executions |
| `commit_pending_executions` | No | — | Workspace-wide drain (deriva-ml v1.39+; replaced legacy `upload_pending`). Per-execution work is covered by `commit_execution`, which calls `commit_output_assets()` on the resumed execution. |

**Execution class methods (`execution/execution.py`)**

| Method | Exposed? | Tool name | Notes |
|---|---|---|---|
| `execution_start` | Yes | `deriva_ml_start_execution` | |
| `commit_output_assets` | Yes | `deriva_ml_commit_execution` | One method drives the full lifecycle bracket (Running -> Stopped -> Pending_Upload -> Uploaded, asset descriptions, Upload_Duration, optional folder cleanup). deriva-ml v1.39+ replaced the legacy `execution_stop` + `upload_execution_outputs` two-step with this single call. |
| `abort` | Yes | `deriva_ml_abort_execution` | |
| `add_features` | Yes | `deriva_ml_add_feature_values` | Staged via execution |
| `create_dataset` | Yes | `deriva_ml_create_execution_dataset` | |
| `add_nested_execution` | Yes | `deriva_ml_add_nested_execution` | |
| `list_input_datasets` | Yes | Embedded in execution detail resource | |
| `list_assets` | Yes | Embedded in execution detail resource | |
| `add_files` | No | — | Intentional: requires local filesystem; file upload is a client-side operation |
| `upload_assets` | No | — | Same |
| `download_dataset_bag` | No | — | Same |
| `asset_file_path` / `metrics_file` | No | — | Filesystem path accessors; MCP server has no local working dir |
| `pending_summary` | No | — | Low-level; execution internals |

**WorkflowMixin (`core/mixins/workflow.py`)**

| Method | Exposed? | Tool name | Notes |
|---|---|---|---|
| `find_workflows` | Yes | `deriva_ml_list_workflows` | |
| `lookup_workflow` | Yes | `deriva_ml_get_workflow` | |
| `create_workflow` | Yes | `deriva_ml_create_workflow` | |
| `lookup_workflow_by_url` | Yes | `deriva_ml_find_workflow_by_url` | |
| `_add_workflow` | No | — | Internal; not public API |

**FeatureMixin (`core/mixins/feature.py`)**

| Method | Exposed? | Tool name | Notes |
|---|---|---|---|
| `create_feature` | Yes | `deriva_ml_create_feature` | |
| `delete_feature` | Yes | `deriva_ml_delete_feature` | |
| `lookup_feature` | Yes | `deriva_ml_get_feature` | |
| `find_features` | Yes | `deriva_ml_list_features` | |
| `feature_values` | Yes | `deriva_ml_list_feature_values` | |
| `add_features` | Yes | `deriva_ml_add_feature_values` | Via execution |
| `feature_record_class` | No | — | Intentional: Python class object; used internally to build records |
| `fetch_table_features` | No | — | Low-level batch accessor; no clear LLM-facing use |
| `list_feature_values` | Partial | `deriva_ml_list_feature_values` | Overlaps with `feature_values`; CLAUDE.md notes the distinction |
| `list_workflow_executions` | No | — | On the Feature object, not the mixin; minor gap |
| `select_by_workflow` | Yes | Exposed as `selector="by_workflow"` parameter in `deriva_ml_list_feature_values` | |

**AssetMixin (`core/mixins/asset.py`)**

| Method | Exposed? | Tool name | Notes |
|---|---|---|---|
| `create_asset` | No | — | Intentional: requires local file staging; out of scope (documented in `asset.py` module docstring) |
| `list_assets` | Yes | `deriva_ml_list_assets` | |
| `list_asset_executions` | Partial | Embedded in `_get_asset_detail_impl` via `asset.list_executions()` | Not independently queryable; the asset detail bundles execution refs |
| `lookup_asset` | Yes | `deriva_ml_lookup_asset` | |
| `list_asset_tables` | Yes | `deriva_ml_list_asset_tables` | |
| `find_assets` | No | — | Gap: `list_assets` requires a specific table; `find_assets` does cross-table search. No MCP equivalent for searching across all asset tables |
| `asset_record_class` | No | — | Python class; intentional |

**VocabularyMixin (`core/mixins/vocabulary.py`)**

All vocabulary operations (`add_term`, `lookup_term`, `list_vocabulary_terms`, `delete_term`) are explicitly delegated to `deriva-mcp-core` per the Boundary Rules. The MCP `registries` resource bundles vocabulary term names for quick reference; the core `add_term` / `delete_term` tools handle mutations. Coverage is correct by design.

**DerivaML base class (`core/base.py`) — catalog administration methods**

| Method | Exposed? | Notes |
|---|---|---|
| `refresh_schema` | No | Server-side cache invalidation; no LLM use case |
| `pin_schema` / `unpin_schema` | No | Schema administration; delegated to `deriva-mcp-core` or `deriva-skills` |
| `validate_schema` | No | Intentional: schema admin, not ML-domain |
| `apply_catalog_annotations` | No | Intentional: annotation admin, not ML-domain |
| `create_vocabulary` | No | Intentional: delegated to `deriva-mcp-core` |
| `create_table` | No | Intentional: delegated to `deriva-mcp-core` |
| `clear_cache` | No | Local filesystem; no MCP equivalent needed |
| `get_cache_size` | No | Same |
| `list_execution_dirs` | No | Local filesystem |
| `clean_execution_dirs` | No | Destructive; intentional omission |
| `get_storage_summary` | No | Local filesystem |
| `workspace` / `download_dir` | No | Path accessors; filesystem |
| `pathBuilder` | No | Intentional: query builder, returns Python object |
| `chaise_url` | No | Covered by individual entity `get_chaise_url()` |
| `cite` | No | Intentional: Python-session convenience method |

### Coverage summary

- **DatasetMixin**: ~70% of public methods exposed. Gaps: `download_dataset_bag` (intentional), `cache_denormalized` (intentional), `list_schema_paths` / `list_denormalized_columns` (low LLM value), `Dataset.list_executions` (actual gap — see below).
- **ExecutionMixin**: ~65% exposed. Intentional omissions: `gc_executions`, `commit_pending_executions` (deriva-ml v1.39+ — workspace-wide drain has no MCP analog; per-execution work goes through `commit_execution`), `pending_summary`. Actual gaps: `find_incomplete_executions`, `find_experiments`.
- **Execution class**: ~60% of public methods exposed. Intentional: file I/O methods, path accessors.
- **WorkflowMixin**: ~90% exposed (all meaningful public methods). Only internal `_add_workflow` omitted.
- **FeatureMixin**: ~80% exposed. Minor gap: `list_workflow_executions` on the Feature object.
- **AssetMixin**: ~55% exposed. Intentional omission: `create_asset` (file staging). Actual gap: `find_assets` for cross-table search.
- **VocabularyMixin**: 0% directly — correctly delegated to `deriva-mcp-core`. Not a gap.
- **DerivaML base**: ~10% exposed. Most omissions intentional (filesystem, schema admin, Python-only APIs).

**Most-significant gaps (methods where an LLM would have to work around):**

1. `Dataset.list_executions()` — no tool to list executions that produced a given dataset. Callers must query `deriva_ml_list_executions` and filter manually, or fetch the execution detail resource if they know execution RIDs. A `deriva_ml_list_dataset_executions(dataset_rid)` tool would close this.

2. `AssetMixin.find_assets()` — the only cross-table asset search. `deriva_ml_list_assets` requires a specific `asset_table`. An LLM that doesn't know which table an asset lives in has no tool to discover it by RID or filename cross-table. `find_assets` supports filtering by RID or other criteria across all asset tables.

3. `ExecutionMixin.find_incomplete_executions()` — no tool to surface executions stuck in non-terminal states (Created/Running/Stopped/Pending_Upload). Useful for recovery workflows but not exposed.

4. `ExecutionMixin.find_experiments()` — experiments are accessible via `lookup_experiment` embedded in the execution detail, but there is no tool to list or search experiments independently.

**Things intentionally not exposed (with justification):**

- All file I/O methods (`download_dataset_bag`, `add_files`, `upload_assets`, `cache_denormalized`, etc.): the MCP server runs without a writable local filesystem from the calling user's perspective. These methods require a local working directory. Correct to omit.
- `pathBuilder()`, `get_table_as_dataframe()`: Python-session objects not serializable to JSON.
- `gc_executions()`, `clean_execution_dirs()`: destructive maintenance operations that should not be exposed to LLMs without explicit human confirmation workflows.
- All vocabulary mutations: correctly delegated to `deriva-mcp-core`.
- All annotation/schema administration methods: correctly delegated to `deriva-mcp-core` / `deriva-skills`.

---

## Top-line findings

1. **Architecture is sound.** The module split, helper factoring, dual-mode policy, and audit + re-index pattern are consistently applied across all five domains. The `_*_impl` shared-helper rule is followed without drift in 10 of 11 resource+tool pairs (`deriva_ml_get_execution` vs `ml/execution/{rid}` being the documented intentional exception where the tool returns a summary, not the full detail).

2. **`execution.py` at 1338 lines is the only oversize file.** It is on the same trajectory that triggered the dataset read/mutate/complex split at ~1745 lines. A read/mutate split for execution would be the natural next structural refactor, but it is not urgent.

3. **Pydantic migration is ~95% complete.** All mutation responses and all list/detail responses from `_*_impl` helpers are Pydantic-typed. Remaining `json.dumps` uses are: (a) argument-validation error returns (structurally fine, no model needed), (b) two read tools with ad-hoc sub-paths (bag_info, dataset_spec, list_dataset_members summary mode, list_dataset_element_types, lookup_workflow_by_url no-result branch). These are post-v2.3 survivors, not regressions.

4. **Three actual coverage gaps worth filing issues for:** (a) `Dataset.list_executions` — no tool to enumerate which executions produced a given dataset; (b) `find_assets` — no cross-table asset search; (c) `find_incomplete_executions` — no tool for recovery-workflow discovery. All other omissions are intentional.

5. **Lazy-import and per-user RAG safety contracts are correctly enforced.** `ctx.rag_dataset_indexer` is never called; all `_reindex_*` imports are inside tool function bodies; the `data:{host}:{cat}:{user_id}:{table}:{rid}` per-RID source naming is used throughout `rag.py`. The CI test pins (`test_register_does_not_use_rag_dataset_indexer`, `test_data_sources_use_per_rid_naming`) cover the regressions that matter most.
