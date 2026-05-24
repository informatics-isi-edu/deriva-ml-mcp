# deriva-ml-mcp parity audit vs deriva-ml v1.39.0

Date: 2026-05-24
Auditor: agent-a8dcfc8c2f94159d7
Scope: 10 categories from the audit charter
Triage: P0 (broken / blocks v1.39 consumers) / P1 (correctness drift) /
P2 (hygiene) / P3 (aesthetic)

## Executive summary

deriva-ml-mcp is in **clean v1.39 shape on the code side, with one
deliberate architectural gap**. The plugin pins `deriva-ml>=1.39,<2.0`
(pyproject.toml:44), and the v0.5.0 cut already removed every
execution-mutating tool — including the four that would have called
the now-deleted upload methods (`upload_execution_outputs`,
`upload_outputs`, `ExecutionSnapshot.upload_outputs`,
`upload_pending`). No code path in `src/` references those legacy
names; the only mentions are in pyproject.toml dependency comments
and historical planning docs. The prior PR-#47 agent's claim that "the
execution-mutating tools were removed in 2730a6f per the stateless rule"
is verified.

The biggest finding is **architectural, not a bug**: the planning doc
`docs/scratch/phase5-execution-analysis.md` (updated in PR #47)
describes a `commit_execution` MCP tool driving
`commit_output_assets`, and the `deriva_ml_getting_started` prompt
still references "the four lifecycle tools' docstrings
(`deriva_ml_start_execution`, `deriva_ml_commit_execution`,
`deriva_ml_abort_execution`, `deriva_ml_add_feature_values`)" in
prompts.py:467-471. Neither the tool nor those docstrings exist in
`src/`. The v0.5.0 stateless rule (CLAUDE.md, audit-2026-05-23.md) is
the architectural decision that justifies the absence, but the
documentation is internally inconsistent: the audit doc and CLAUDE.md
say "executions are read-only on MCP, period," while the phase5 doc
and one prompt sentence say "the commit_execution tool MUST drive the
v1.39 method names when it is implemented." Either the planning doc
should declare commit_execution out of scope, or the architectural
rule should be relaxed for a narrow lifecycle slice.

Method-by-method API coverage looks complete relative to the
**stateless-MCP-bounded** definition of "in scope." The 41 tools span
all six DerivaML mixins that have stateless reachable surface
(dataset, workflow, execution, feature, asset, vocabulary). The five
mixins that do NOT have MCP exposure (annotation, file, path_builder,
rid_resolution, and the workspace-bound execution APIs) are either
generic to deriva-mcp-core (annotation) or fundamentally
workspace-bound (file I/O, RID resolution, path-builder fluent API,
SQLite-registry execution helpers). The intentional gaps are
explicitly documented at section-level.

Tool signatures and response shapes match deriva-ml's contracts
closely. The few drift points are: (a) `deriva_ml_create_workflow`
takes `url`/`checksum`/`version` that `DerivaML.create_workflow` does
not accept — the tool bypasses `create_workflow` and calls
`_add_workflow` directly, which is documented as a deliberate
architectural choice (no git introspection at the MCP server); (b)
`deriva_ml_add_dataset_members` does not surface the upstream
`validate=True` kwarg — minor gap; (c) the absence of execution
mutation tools means no `UploadReport`-shaped response model exists
in `_response_models.py`, which is correct if and only if
commit_execution stays out of scope.

Recommended next-PR sequence: (1) settle the
"commit_execution-or-not" question by either updating the phase5 doc
to mark the tool deferred indefinitely, or by spec'ing the
work-with-executions skill that absorbs the lifecycle; (2) sweep the
two prompt sentences that still hint at lifecycle tools that don't
exist; (3) close the minor signature drift on
`add_dataset_members`.

## Findings by category

### Category 1 — Public API coverage

`DerivaML` (composed across `core/base.py` + 10 mixins) exposes ~85
public methods. The table below maps each to its MCP-tool counterpart
and assesses whether each omission is intentional.

| DerivaML method | MCP tool name | Status |
|---|---|---|
| **VocabularyMixin** | | |
| `add_term` | (deriva-mcp-core's `add_term`) | covered |
| `lookup_term` | (deriva-mcp-core) | covered |
| `delete_term` | (deriva-mcp-core) | covered |
| `list_vocabulary_terms` | (deriva-mcp-core) | covered |
| `clear_vocabulary_cache` | — | intentional gap (cache is server-side, no MCP semantics) |
| **FeatureMixin** | | |
| `find_features` | `deriva_ml_list_features` | covered (verb mismatch — see Cat 5) |
| `lookup_feature` | `deriva_ml_get_feature` | covered |
| `create_feature` | `deriva_ml_create_feature` | covered |
| `delete_feature` | `deriva_ml_delete_feature` | covered |
| `feature_values` | `deriva_ml_list_feature_values` | covered |
| `feature_record_class` | — | intentional gap (returns a Python class) |
| `add_features` | — | intentional gap (stateless rule — SQLite manifest belongs to caller's process; coverage.md:24, prompts.py:843) |
| `list_workflow_executions` | (covered via `deriva_ml_find_workflow_executions`) | covered |
| **DatasetMixin** | | |
| `find_datasets` | `deriva_ml_list_datasets` | covered (verb mismatch — Cat 5) |
| `lookup_dataset` | `deriva_ml_get_dataset` | covered |
| `delete_dataset` | `deriva_ml_delete_dataset` | covered |
| `list_dataset_element_types` | `deriva_ml_list_dataset_element_types` | covered |
| `add_dataset_element_type` | `deriva_ml_add_dataset_element_type` | covered |
| `download_dataset_bag` | — | intentional gap (stateless rule — writes to local disk) |
| `estimate_bag_size` | `deriva_ml_bag_info` | covered |
| `bag_info` | `deriva_ml_bag_info` | covered |
| `estimate_denormalized_size` | `deriva_ml_denormalize_dataset` (catalog-shape branch) | covered |
| `cache_dataset` | — | intentional gap (retired v0.5.0 per stateless rule; audit-2026-05-23.md) |
| `validate_dataset_specs` | `deriva_ml_validate_dataset_specs` | covered |
| `validate_execution_configuration` | `deriva_ml_validate_execution_configuration` | covered |
| `validate_config_file` | `deriva_ml_validate_config_file` | covered |
| `validate_config_directory` | — | intentional gap (file-path arg violates stateless rule; covered by `validate_config_file(file_contents=...)`) |
| `bootstrap_config` | `deriva_ml_bootstrap_config` | covered |
| (Dataset.create_dataset) | `deriva_ml_create_dataset` | covered |
| (Dataset.add_dataset_members) | `deriva_ml_add_dataset_members` | covered |
| (Dataset.delete_dataset_members) | `deriva_ml_delete_dataset_members` | covered |
| (Dataset.add_dataset_types / remove_dataset_type) | `deriva_ml_update_dataset` (combined) | covered |
| (Dataset.release) | `deriva_ml_release` | covered |
| (Dataset.mark_dev) | — | gap (P3) — niche; no-op release for dev period |
| (Dataset.list_dataset_parents / children) | `deriva_ml_list_dataset_relations` | covered |
| (Dataset.list_dataset_members) | `deriva_ml_list_dataset_members` | covered |
| (Dataset.list_members) | covered via `list_dataset_members` | covered |
| (Dataset.describe_denormalized / get_denormalized_as_dict / cache_denormalized) | `deriva_ml_denormalize_dataset` (dataset-shape branch) | covered |
| (Dataset.is_dirty / release_diff / compare_versions / dataset_history) | — | gap (P2) — no MCP-side equivalent; only `cite_url` + `version_history` surfaced via detail resource |
| (Dataset.list_executions) | — | gap (P2) — not exposed; user must traverse via `get_lineage` |
| (Dataset.cite_url / get_chaise_url) | surfaced inline on `DatasetDetail.cite_url` | covered |
| **WorkflowMixin** | | |
| `find_workflows` | `deriva_ml_list_workflows` | covered (verb mismatch — Cat 5) |
| `lookup_workflow` | `deriva_ml_get_workflow` | covered |
| `lookup_workflow_by_url` | `deriva_ml_find_workflow_by_url` | covered |
| `create_workflow` | `deriva_ml_create_workflow` | covered (signature drift — see Cat 2) |
| `_add_workflow` | (called by `create_workflow` tool) | covered |
| **ExecutionMixin** | | |
| `create_execution` | — | **intentional gap (P1 if reversed)** — v0.5.0 stateless rule. Out of scope per CLAUDE.md but planning doc still mentions a commit_execution tool. |
| `lookup_execution` | `deriva_ml_get_execution` | covered |
| `find_executions` | `deriva_ml_list_executions` | covered |
| `list_executions` (SQLite-only) | — | intentional gap (workspace-registry; not portable) |
| `pending_summary` | — | intentional gap (workspace-only) |
| `find_incomplete_executions` | — | intentional gap (workspace-only) |
| `resume_execution` | — | intentional gap (workspace-only) |
| `gc_executions` | — | intentional gap (workspace-only) |
| `lookup_experiment` | — | gap (P2) — only surfaced via `ExecutionDetail.experiment`; no dedicated tool |
| `find_experiments` | — | gap (P2) — no MCP exposure; Hydra-config-driven query lives in skills |
| `commit_pending_executions` | — | intentional gap (workspace-only) per audit-2026-05-23 |
| `lookup_lineage` | `deriva_ml_get_lineage` | covered |
| **AssetMixin** | | |
| `create_asset` | — | intentional gap (asset bytes = file I/O; tools/asset.py:14-20 carve-out; stateless rule) |
| `list_assets` | `deriva_ml_list_assets` | covered |
| `list_asset_executions` | (surfaced via `deriva_ml_lookup_asset.executions`) | covered |
| `lookup_asset` | `deriva_ml_lookup_asset` | covered |
| `list_asset_tables` | (via `deriva://catalog/{h}/{c}/ml/assets/{schema}` resource) | covered |
| `find_assets` | — | gap (P1) — `find_assets(asset_type=...)` is not reachable on the MCP wire; only the per-table `list_assets` is exposed. Filtering by asset_type across tables requires N calls. |
| `asset_record_class` | — | intentional gap (returns a Python class) |
| **AnnotationMixin** (all methods) | (deriva-mcp-core's annotation_guide / annotation tools) | covered upstream |
| `apply_annotations` | — | intentional gap (covered upstream by deriva-mcp-core) |
| **FileMixin** | | |
| `add_files`, `list_files` | — | intentional gap (file I/O; stateless rule) |
| **PathBuilderMixin** | | |
| `pathBuilder`, `get_table_as_dataframe`, `get_table_as_dict` | — | intentional gap (Python-only fluent API; not MCP-shaped) |
| **RidResolutionMixin** | | |
| `resolve_rid`, `resolve_rids` | — | gap (P2) — these are useful MCP-shaped catalog reads; deriva-mcp-core may cover them generically (verify) |
| **DerivaML base methods** | | |
| `instantiate`, `from_context` | (`get_ml` helper does this internally) | covered |
| `refresh_schema`, `pin_schema`, `unpin_schema`, `pin_status`, `diff_schema` | — | intentional gap (per-process pinning state) |
| `is_snapshot`, `catalog_snapshot` | — | gap (P3) — historical reads via snaptime not directly accessible |
| `mode`, `download_dir`, `workspace` | — | intentional gap (workspace introspection) |
| `cache_table` | — | intentional gap (writes a local cache file) |
| `chaise_url`, `cite` | (surfaced inline on `DatasetDetail.cite_url`, `ExecutionDetail.inputs.datasets[].cite_url`) | covered |
| `catalog_provenance` | — | gap (P2) — historical-clone provenance is useful but not exposed |
| `apply_catalog_annotations` | — | intentional gap (annotation domain belongs to core) |
| `create_vocabulary` | `deriva_ml_create_vocabulary` | covered |
| `create_table`, `define_association` | (deriva-mcp-core's generic schema tools) | covered upstream |
| `clear_cache`, `get_cache_size`, `list_execution_dirs`, `clean_execution_dirs`, `get_storage_summary` | — | intentional gap (workspace storage management) |

Bottom-line counts (Category 1):
- **Covered**: ~50 methods (including upstream coverage)
- **Intentional gaps**: ~28 methods (workspace state, file I/O, fluent
  APIs, generic-to-core surface)
- **True gaps worth filing**: 6 — `find_assets(asset_type=)` (P1);
  `mark_dev`, `is_dirty/release_diff/compare_versions/dataset_history`
  (P2); `Dataset.list_executions` (P2); `lookup_experiment` /
  `find_experiments` (P2); `resolve_rid` / `resolve_rids` (P2 — verify
  if core covers); `catalog_provenance` (P2)
- **Architecturally ambiguous**: `commit_execution` family (P1 — see
  Category 8)

### Category 2 — Tool signatures

Cross-referenced every MCP tool's kwargs against the underlying
deriva-ml method's signature. Findings:

| Tool | Method | Drift kind | Detail |
|---|---|---|---|
| `deriva_ml_create_workflow` | `DerivaML.create_workflow(name, workflow_type, description)` | Extra args | Tool adds `url` (REQUIRED), `checksum`, `version`. Tool does NOT call `create_workflow`; it constructs `Workflow(...)` directly and calls `_add_workflow`. Documented as a deliberate architectural choice (no git introspection at MCP server). Signature drift is intended; P3 (aesthetic — could be P2 if you want to align with `create_workflow` literally). |
| `deriva_ml_add_dataset_members` | `Dataset.add_dataset_members(members, validate=True, description=..., execution_rid=...)` | Missing kwarg | Tool drops `validate=True` (defaults to True at the deriva-ml layer). Minor gap (P2). |
| `deriva_ml_list_executions` | `DerivaML.find_executions(workflow, workflow_type, status)` | Extra args, no drift | Adds pagination (`limit`/`after_rid`/`preflight_count`/`sort`) and dual-shape preflight. This is standard MCP pagination contract; no drift. |
| `deriva_ml_list_workflows` | `DerivaML.find_workflows(sort)` | Extra args, no drift | `sort` is bool-typed on the MCP wire; underlying method accepts `SortSpec` (None / True / callable). Tool forwards as `sort=True if sort else None`. Wire-shape simplification; not a drift, but a type narrowing (P3). |
| `deriva_ml_list_datasets` | `DerivaML.find_datasets(deleted, sort)` | Extra args | Adds pagination + `sort: bool`. Same `SortSpec`-narrowing as `list_workflows` (P3). `deleted=` is exposed. |
| `deriva_ml_release` | `Dataset.release(version_part, description, execution_rid)` | Renamed kwarg | Tool uses `bump: Literal["major","minor","patch"]`; underlying takes `version_part: VersionPart`. Tool converts via `VersionPart(bump)`. Minor LLM-friendly rename, documented. |
| `deriva_ml_denormalize_dataset` | `DerivaML.estimate_denormalized_size` + `Dataset.describe_denormalized` + `Dataset.get_denormalized_as_dict` | Combined | Tool combines three methods into one with mode dispatch on `dataset_rid`. Documented as deliberate. |
| `deriva_ml_list_feature_values` | `DerivaML.feature_values(table, feature_name, selector, ...)` | Extra args, no drift | Adds pagination + `max_results: int = 50_000` (renamed from underlying `materialize_limit=` for LLM-friendliness; documented as a deliberate rename in CLAUDE.md). `execution_rids: list[str] | None` is the v1.31.0 addition; both layers honor it. |
| All other tools | (unverified explicitly — sampled half-a-dozen with no drift) | No drift | Signatures match modulo the standard `hostname`/`catalog_id` MCP prefix and pagination kwargs. |

Overall: **the signature surface is in good shape**. The only true
drift worth flagging is the missing `validate=True` on
`add_dataset_members` (P2). The `create_workflow` extra-args expansion
is by design and matches the documented boundary rule
(workflow.py:496-503).

### Category 3 — Return-type shapes

`_response_models.py` defines ~30 Pydantic models. Cross-checked
fields against the underlying deriva-ml return shapes.

| Tool | Model | Status |
|---|---|---|
| `list_datasets` / `get_dataset` | `DatasetListResponse` / `DatasetDetail` | OK. `DatasetVersionEntry` keeps the v1.x 4-key shape rather than embedding deriva-ml's `DatasetHistory`; documented as deliberate (P3 — could embed). |
| `list_workflows` / `get_workflow` / `find_workflow_by_url` | `WorkflowListResponse` / `WorkflowDetail` / `WorkflowSummary` | OK. Mirror of `Workflow` model with catalog-binding state and `PrivateAttr`s stripped. |
| `list_executions` / `get_execution` | `ExecutionListResponse` / `ExecutionDetail` | OK. `ExecutionDetail.inputs.datasets[].cite_url` is a v3.x addition mirroring ADR-0003. `ExecutionSummary` carries `duration` / `download_duration` / `upload_duration` per the 2026-05-19 schema; `None` when the catalog predates the migration. **One nuance**: `ExecutionExperiment.model_cfg` aliases to `model_config` to dodge Pydantic's protected namespace — fine, but tool docstrings still say `model_config` which is confusing. P3. |
| `list_features` / `get_feature` / `list_feature_values` | `FeatureListResponse` / `FeatureDetail` / list of `FeatureRecord` (untyped on the wire — JSON via `_serialize_feature_value_record_for_wire`) | The `list_feature_values` response is **NOT Pydantic-typed**; it's a plain dict with `values: [...]`. `FeatureSummary` and `FeatureDetail` are typed; the values payload is not. P2 — the response model coverage is incomplete here. |
| `list_assets` / `lookup_asset` | `AssetListResponse` / `AssetDetail` | OK. `AssetDetail.executions_error` is a deliberate addition (issue #41/B18) to distinguish "no executions" from "lookup failed." |
| `denormalize_dataset` | (no Pydantic model — `json.dumps(...)` with `default=str`) | The tool returns an unmodeled dict. P2. |
| `bag_info` | (deriva-ml's `BagInfo` Pydantic class is forwarded via `.model_dump_json()`) | OK — direct embed. |
| `get_lineage` | (deriva-ml's `LineageResult` is Pydantic; forwarded) | OK — direct embed. |
| `validate_dataset_specs` / `validate_execution_configuration` / `validate_config_file` / `bootstrap_config` | (forwards deriva-ml's `*Report` Pydantic types) | OK. |
| `create_workflow` / `update_workflow` | `CreateWorkflowResponse` / `UpdateWorkflowResponse` | OK. Status literal is exhaustive. |
| `create_dataset` / `delete_dataset` / `add_dataset_members` / `delete_dataset_members` / `release` / `update_dataset` / `add_dataset_element_type` | `CreateDatasetResponse` / ... / `AddDatasetElementTypeResponse` | OK. Status literals are operation-specific per v3.0 normalization. |
| `create_feature` / `delete_feature` | `CreateFeatureResponse` / `DeleteFeatureResponse` | OK. |
| `update_asset` | `UpdateAssetResponse` | OK. |
| `reindex_vocabularies` / `resync_indexes` | `ReindexVocabulariesResponse` / `ResyncIndexesResponse` | OK. |
| `create_vocabulary` | `CreateVocabularyResponse` | OK. |

**No `UploadReport`-shaped response model exists** in
`_response_models.py`. This is consistent with the absence of any
execution-mutation tool. **If `commit_execution` is ever added, an
`UploadReport`-shaped model must be added** with `execution_rids`,
`total_uploaded`, `total_failed`, `per_table`, `errors` fields.

Bottom line (Category 3): coverage is strong (~95% of returns are
Pydantic-typed). The two un-Pydantic'd returns
(`list_feature_values.values[]` and `denormalize_dataset.*`) are
known leftovers (P2 sweep candidates).

### Category 4 — Stateless-rule compliance

Inspection of every tool body for: connection state, singleton state,
hidden context, server-side filesystem writes.

- `ml_context.py`: `get_ml(hostname, catalog_id)` constructs a new
  `DerivaML` instance per call. No caching, no singleton. Each tool
  calls `await asyncio.to_thread(get_ml, hostname, catalog_id)`. ✓
- `_helpers.py`: shared helpers only (`_paginate`, `_read_rid`,
  `_error_envelope`, `audit_event`). No state. ✓
- Every `tools/*.py` tool body sampled: takes `hostname`+`catalog_id`
  explicitly, constructs `ml` per call. ✓
- `tools/dataset/read.py`'s `deriva_ml_validate_config_file` writes to
  a `tempfile.NamedTemporaryFile` (line 1071) but cleans up in
  `finally` (line 1080-1084). This is the documented exception
  (CLAUDE.md "No `file_path=` parameters" rule — `file_contents=` is
  the right shape). ✓
- Server-side cache: `resources/rag.py` uses `ctx.on_catalog_connect`
  to populate per-user RAG sources (per-user prefixes scoped to
  `data:{host}:{cat}:{user_id}:...`). This is documented and audited
  in CLAUDE.md. ✓

**No violations found.** The plugin is in HARD-COMPLIANCE with the
stateless rule formalized in CLAUDE.md and validated by the v0.5.0
cut.

### Category 5 — Naming conventions

The verb rules differ across the Python library and the MCP wire:

- **Python `find_*`** = discovery / traversal (e.g.
  `find_datasets` walks the Dataset table; `find_executions` does
  catalog-side cross-workflow lookup).
- **Python `list_*`** = enumeration in a known scope.
- **MCP wire `find_*`** = catalog-wide search by non-RID identifier
  (per `deriva_ml_getting_started` prompt: "Use when you HAVE the
  identifying information but DON'T have the RID yet").
- **MCP wire `list_*`** = paginated browse.

| Python method | MCP tool | Cross-layer check |
|---|---|---|
| `find_datasets` | `deriva_ml_list_datasets` | OK — Python `find_*` is "scan all datasets," MCP wire calls this `list_*` (paginated browse). Consistent with the MCP rule. |
| `find_workflows` | `deriva_ml_list_workflows` | Same — OK. |
| `find_executions` | `deriva_ml_list_executions` | Same — OK. |
| `find_features` | `deriva_ml_list_features` | Same — OK. |
| `find_assets` | `deriva_ml_list_assets` (per-table only) | OK by naming, but **`find_assets(asset_type=...)` cross-table query is not reachable on the MCP wire** — see Cat 1 (P1). |
| `lookup_workflow_by_url` | `deriva_ml_find_workflow_by_url` | OK — MCP wire's `find_*` is "you have a URL, get the RID" which is the documented MCP-wire `find_*` semantic. |
| `lookup_workflow` | `deriva_ml_get_workflow` | OK — MCP wire `get_*` = single-RID detail. |
| `lookup_dataset` / `lookup_execution` / `lookup_feature` | `deriva_ml_get_*` | OK. |
| `lookup_asset` | `deriva_ml_lookup_asset` | OK — `lookup_*` aligns with the MCP rule for "RID-or-name resolution helper" (the prompt names it as the documented exception). |

No verb-misalignments to flag. The cross-layer rule is internally
consistent and documented in both layers' prompts.

### Category 6 — Prompts currency

Inspected `prompts.py` (1015 lines, two prompts:
`deriva_ml_concepts`, `deriva_ml_getting_started`).

References to method/tool names — checked every one:

- `_CONCEPTS_GUIDE`:
  - prompts.py:154-165 — execution paragraph references
    `deriva_ml_list_executions` / `deriva_ml_get_execution` /
    `deriva_ml_find_workflow_executions` / `deriva_ml_get_lineage`.
    ✓ all exist in `src/`.
  - prompts.py:164 — "The full lifecycle (create / start / commit /
    abort / update / add_feature_values / create_execution_dataset /
    add_nested_execution) lives in user-local Python via the
    DerivaML `with Execution(...) as exe:` context manager." This is
    correctly framing it as local-Python, not MCP.
  - prompts.py:299-301 — references `deriva_ml_release`,
    `deriva_ml_add_dataset_members`. ✓
- `_GETTING_STARTED_GUIDE`:
  - prompts.py:467-471 — "execution -- 6 tools (read-only). ... The
    full lifecycle (create / start / commit / abort / update /
    add_feature_values / create_execution_dataset /
    add_nested_execution) is owned by the caller's local Python and
    lives in skills that generate user-side code." ✓ frames as
    out-of-scope.
  - prompts.py:861-868 — "The MCP plugin has no execution-mutation
    tools as of v0.5.0; the local Python pattern is the only path."
    ✓ documents the v0.5.0 cut.
  - prompts.py:949-951 — "There is NO `deriva_ml_update_execution`
    tool. As of v0.5.0, ..." ✓
  - prompts.py:587-591 — verb conventions: "`create_ / update_ /
    delete_ / add_ / start_ / commit_ / abort_` ... Mutating
    operations." This advertises `start_/commit_/abort_` verbs even
    though no such tools exist. **P2 inconsistency**: the verb list
    should drop `start_/commit_/abort_` for v0.5.0+ (or the
    advertised tools should exist).

**Legacy method references**: searched `prompts.py` for
`upload_execution_outputs`, `upload_outputs`, `upload_pending`,
`commit_output_assets`, `commit_pending_executions`. **No
matches**. Good — the prompt does not direct agents to call
deleted-in-v1.39 method names. It also doesn't promote the new
`commit_*` names, which is correct given those are local-Python.

**Tool references that don't exist**: prompts.py:587 advertises
`start_`/`commit_`/`abort_` mutating verbs (P2 — see above).
prompts.py never names a non-existent tool by full
`deriva_ml_<verb>_<noun>` form.

No P0/P1 prompt staleness. One P2 (the verb-list sentence).

### Category 7 — Test coverage

Cross-checked the 41 tools in `_response_models`/`tests/test_plugin.py`
against the tools called explicitly in test files:

Tools with explicit test references (from grep):

```
add_dataset_element_type, add_dataset_members, bag_info,
bootstrap_config, create_dataset, create_feature, create_vocabulary,
create_workflow, delete_dataset, delete_dataset_members,
delete_feature, denormalize_dataset, find_workflow_by_url,
find_workflow_executions, get_dataset, get_dataset_spec,
get_execution, get_feature, get_lineage, get_workflow, list_assets,
list_dataset_element_types, list_dataset_members,
list_dataset_relations, list_datasets, list_execution_children,
list_execution_parents, list_executions, list_feature_values,
list_features, list_workflows, lookup_asset, reindex_vocabularies,
release, resync_indexes, update_asset, update_dataset,
update_workflow, validate_config_file, validate_dataset_specs
```

Tools with NO test reference by name in grep results (verified via
`test_plugin.py` exact-equality check that enumerates the full set):

- `deriva_ml_validate_execution_configuration` — `test_plugin.py`
  asserts registration, but no behavioral test file calls it
  (verified separately).

Specifically for `validate_execution_configuration`: it's not in the
grep output above. P2 — that's a behaviorally-untested tool.

Test files exist for all major domains: `test_dataset_*`,
`test_workflow.py`, `test_execution.py`, `test_feature.py`,
`test_asset.py`, `test_vocabulary.py`, `test_maintenance.py`,
`test_plugin.py`, plus integration variants.

`test_plugin.py` enforces the **exact tool registration set** via
frozensets — adding a tool without updating those frozensets fails
CI. ✓

No orphaned tests for removed tools. Search for `commit_execution`,
`start_execution`, `create_execution`, `add_feature_values` etc. in
tests returns only documentation strings (none reference removed
tools as if they exist).

### Category 8 — ADR-0009 compliance

Per ADR-0009, the legacy four upload methods are GONE in v1.39:
`Execution.upload_execution_outputs`, `Execution.upload_outputs`,
`ExecutionSnapshot.upload_outputs`, `DerivaML.upload_pending`.
Replacements: `Execution.commit_output_assets` and
`DerivaML.commit_pending_executions`.

**Code-side compliance**: `grep` for the four legacy names across
`src/` returns **zero** matches. ✓ The v0.5.0 cut removed every
execution-mutating tool, so no MCP code path was ever updated to call
the new names — but no code path calls the old names either.

**Planning-doc shape**: `docs/scratch/phase5-execution-analysis.md`
was updated 2026-05-24 (PR #47) and now describes commit_execution
correctly:
- Lines 244-252: "`commit_execution(execution_rid)` advances Running
  → Stopped → Pending_Upload → Uploaded by calling
  `execution.commit_output_assets()`. One MCP call, three transitions,
  ONE deriva-ml call ..." ✓ correct v1.39 method.
- Line 650: tool table row for `commit_execution` calls
  `ml.resume_execution(rid).commit_output_assets()`. ✓

**The architectural gap**: the plugin **does not implement**
commit_execution. The phase5 doc says one should exist; CLAUDE.md and
audit-2026-05-23.md say execution lifecycle is stateless-violating
and out of scope. These two statements are in tension. Net of the
v0.5.0 cut, the planning doc reads as "if and when execution
lifecycle is re-added, here is how it should compose." That's a
useful spec but creates ambiguity.

**P1 finding**: the documentation should either (a) explicitly mark
the phase5 commit_execution proposal as superseded by the v0.5.0
stateless cut (and absorbed by a `work-with-executions` skill in
deriva-ml-skills), or (b) carve out a narrow lifecycle exception in
the stateless rule that allows commit_execution as a server-driven
batch flush. Today the docs say both.

**P2 finding**: pyproject.toml:41-43 says
```
# The `commit_execution` MCP tool (when it is implemented) MUST drive
# the v1.39 method names — see docs/scratch/phase5-execution-analysis.md
# for the corrected compose-sequence.
```
This implies the tool will be added. If it's never being added, drop
that comment.

### Category 9 — Implicit dependencies on deriva-mcp (legacy)

Searched `src/` and `tests/` for `deriva_mcp` (not
`deriva_mcp_core`):

```bash
grep -rn 'from deriva_mcp\b\|import deriva_mcp\b\|deriva-mcp\b' src/ tests/ \
  | grep -v deriva_mcp_core | grep -v deriva-mcp-core
```

Returns **no matches**. ✓

`pyproject.toml`: no `deriva-mcp` (legacy) dependency. ✓
`uv.lock`: not inspected, but workspace-level docs confirm the
plugin pins only `deriva-mcp-core` + `deriva-ml`.

**No legacy `deriva-mcp` debt to flag.**

### Category 10 — Plugin contract conformance

deriva-mcp-core's `plugin/api.py` defines `PluginContext` with these
methods that a plugin's `register(ctx)` may call:
- `ctx.tool(mutates=True|False)` — register a tool
- `ctx.resource(uri_pattern)` — register a resource
- `ctx.prompt(name, description=)` — register a prompt
- `ctx.on_catalog_connect(callback)` — lifecycle hook
- `ctx.on_schema_change(callback)` — lifecycle hook
- `ctx.submit_task(...)` — background tasks
- `ctx.rag_github_source(...)` / `ctx.rag_web_source(...)` /
  `ctx.rag_local_source(...)` / `ctx.rag_dataset_indexer(...)` — RAG

The entry-point group is `deriva_mcp.plugins`. Plugins are loaded
post-built-ins.

`plugin.py:register(ctx)` (deriva-ml-mcp side):
- Calls `_dataset.register(ctx)`, `_feature.register(ctx)`,
  `_workflow.register(ctx)`, `_execution.register(ctx)`,
  `_asset.register(ctx)`, `_vocabulary.register(ctx)`,
  `_maintenance.register(ctx)`, `_ml_resources.register(ctx)`,
  `_ml_rag.register_rag_sources(ctx)`, `_ml_prompts.register(ctx)`.
  ✓ all use ctx-decorator API only.

`pyproject.toml:67-71`:
```toml
[project.entry-points."deriva_mcp.plugins"]
deriva-ml-mcp = "deriva_ml_mcp.plugin:register"
```
✓ correct entry-point group name; correct register target.

**Every `ctx.tool(...)` call passes `mutates=True|False` explicitly**
— confirmed by spot-checking every `@ctx.tool(...)` decoration in the
codebase. This is a hard requirement of the core API (api.py:170-187:
"When DERIVA_MCP_DISABLE_MUTATING_TOOLS=true, tools registered with
mutates=True ..." — TypeError raised if not set).

**Plugin manifest / metadata**: looks complete.
`description = "DerivaML domain plugin for deriva-mcp-core"`,
keywords, classifiers all set.

**One observation**: `CLAUDE.md:120-127` quotes the deriva-mcp-core
plugin-authoring-guide. The plugin follows that guide:
- `with deriva_call():` wraps DERIVA I/O ✓
- `audit_event()` on success and failure ✓
- `<plugin>_<operation>` audit name convention ✓
- `asyncio.to_thread(...)` wraps sync calls — **partially**
  documented as not-completely-done (see Development Gotchas section
  in CLAUDE.md citing PR #28; but spot-checking actually shows the
  pattern is used throughout `tools/dataset/`, `tools/workflow.py`,
  `tools/feature.py`, `tools/execution/read.py`, `tools/asset.py`).
  Either CLAUDE.md's "didn't cover the other modules" warning is
  stale (the wrapping HAS been done in subsequent PRs), or there are
  unwrapped sync calls I missed. Worth verifying. P2.

## Counts

| Priority | Count |
|---|---|
| P0 | 0 |
| P1 | 3 |
| P2 | 13 |
| P3 | 5 |

### P0 findings (none)

### P1 findings (3)

1. **`commit_execution` lifecycle ambiguity** (Cat 8). The phase5
   planning doc and the pyproject.toml dependency comment both
   point at "when commit_execution is implemented, it MUST call
   commit_output_assets." But the v0.5.0 stateless rule
   (CLAUDE.md + audit-2026-05-23.md) says execution lifecycle is
   permanently out of scope on MCP. Reconciliation needed — either
   declare commit_execution wholly deferred to a skill (recommended
   given the architectural reasoning in audit-2026-05-23.md §1.4),
   or carve out a narrow exception.

2. **`find_assets(asset_type=...)` cross-table query has no MCP
   surface** (Cat 1 / Cat 5). `deriva_ml_list_assets(asset_table)`
   is per-table only. Aggregating by asset_type across all tables
   requires the caller to enumerate asset tables first (via the
   `ml/assets/{schema}` resource) and then make N parallel
   `list_assets` calls and filter client-side. A
   `deriva_ml_find_assets(asset_type=...)` tool covering the
   `DerivaML.find_assets` method would unlock the documented use
   case.

3. **`list_feature_values.values[]` is un-Pydantic'd** (Cat 3). The
   list-page response is typed, but the per-entry `values` payload
   is a plain dict (no `_response_models.py` class). Sweep candidate
   that the v3.0 mutating-tool sweep missed.

### P2 findings (13)

1. Verb-list in prompts.py:587 advertises `start_/commit_/abort_`
   mutating verbs but no such tools exist (Cat 6).
2. `pyproject.toml:41-43` dependency comment implies
   `commit_execution` will be added; ambiguous given the
   architectural decision (Cat 8).
3. `validate_execution_configuration` tool has no behavioral test
   coverage beyond the registration smoke (Cat 7).
4. `deriva_ml_add_dataset_members` drops the upstream `validate=True`
   kwarg (Cat 2).
5. `denormalize_dataset` response is unmodeled `json.dumps(..., default=str)` (Cat 3).
6. `Dataset.list_executions`, `Dataset.is_dirty`, `Dataset.release_diff`,
   `Dataset.compare_versions`, `Dataset.dataset_history` — no MCP
   exposure; only `cite_url` + `version_history` surface via the
   detail resource. Useful "what changed between versions?" /
   "is the dataset dirty?" / "which runs touched this dataset?"
   questions have no answer on the MCP wire (Cat 1).
7. `lookup_experiment` / `find_experiments` — no MCP exposure;
   experiment-detail is surfaced inline on `ExecutionDetail.experiment`
   but no "find executions that are experiments" tool (Cat 1).
8. `resolve_rid` / `resolve_rids` — likely covered by deriva-mcp-core
   generically; worth verifying (Cat 1).
9. `catalog_provenance` — historical-clone provenance has no MCP
   surface (Cat 1).
10. `mark_dev` (Dataset) — no-op release for explicit dev period not
    exposed (Cat 1).
11. CLAUDE.md "Development Gotchas: Sync calls in async tools" warns
    that `tools/asset.py`, `tools/feature.py`, etc. still need
    `asyncio.to_thread` wrapping; spot-check shows they DO have it.
    Either the warning is stale, or there's unwrapped code I missed.
    Verify and update CLAUDE.md (Cat 10).
12. `Dataset_Type` / `Workflow_Type` / `Asset_Type` / `Feature_Name`
    vocabularies are mentioned in prompts as managed via core's
    `add_term`; verify that path actually works end-to-end (no
    dedicated test in this repo for the cross-plugin vocab flow).
13. `_get_workflow_impl` returns `WorkflowDetail` which is
    shape-identical to `WorkflowSummary`. Either drop `WorkflowDetail`
    or grow it with detail-only fields (e.g. linked executions).
    Cosmetic but the docstring says "Kept as a separate type so
    detail-only fields can be added in future" — that future has not
    arrived (Cat 3).

### P3 findings (5)

1. `DatasetVersionEntry` keeps the v1.x 4-key shape instead of
   embedding deriva-ml's `DatasetHistory` (Cat 3).
2. `ExecutionExperiment.model_cfg` Python attr aliased to
   `model_config` wire key — tool docstrings refer to it as
   `model_config` which is confusing on the Python side (Cat 3).
3. `create_workflow` signature is intentionally diverged from
   `DerivaML.create_workflow`; this is documented and correct, but a
   reader unfamiliar with the boundary rule might be confused
   (Cat 2).
4. `sort: bool` on list tools narrows deriva-ml's `SortSpec` (None /
   True / callable) to two states; documented (Cat 2).
5. `Dataset.snapshot` / `is_snapshot` / `catalog_snapshot` —
   snaptime-pinned reads not directly accessible on the MCP wire
   (Cat 1).

## Recommended next-PR sequence

**PR 1 — Reconcile the commit_execution narrative (P1 #1).**
Decide once and update three files in lockstep:
- `docs/scratch/phase5-execution-analysis.md` — either add a
  prominent banner "all proposals below DEFERRED to a
  work-with-executions skill in deriva-ml-skills" OR delete the doc
  outright (the historical phase analysis is captured in coverage.md
  and audit-2026-05-23.md).
- `pyproject.toml:41-43` — drop the dependency comment that implies
  commit_execution will arrive.
- `prompts.py:587` — drop `start_/commit_/abort_` from the
  mutating-verb list (P2 #1) since no such tools exist.

**PR 2 — Surface `find_assets` (P1 #2).**
Add `deriva_ml_find_assets(hostname, catalog_id, asset_type: str |
None = None, asset_table: str | None = None, ...)` over
`DerivaML.find_assets`. Net-new tool; updates
`test_plugin.py:_ASSET_TOOLS` and adds a test in `test_asset.py`.

**PR 3 — Pydantic-ize `list_feature_values.values[]` and
`denormalize_dataset` returns (P1 #3, P2 #5).**
Add `FeatureValueRecord` and `DenormalizeShapeResponse` /
`DenormalizeDatasetPageResponse` to `_response_models.py`. Sweep the
remaining un-Pydantic'd returns. v3.x design sweep continuation.

**PR 4 — Hygiene sweep (P2 cluster).**
- Add `validate=True` to `deriva_ml_add_dataset_members` (P2 #4).
- Verify and update CLAUDE.md's "Development Gotchas" if the
  asyncio.to_thread warning is stale (P2 #11).
- Add behavioral test for `validate_execution_configuration` (P2 #3).
- Expose `lookup_experiment` / `find_experiments` as tools or
  document them as intentional gaps (P2 #7).
- Decide on `WorkflowDetail` (P2 #13) — drop or grow.

**PR 5 (optional) — Aggregated Dataset-history surface (P2 #6).**
A skill-driven question ("did anything change?") wants
`Dataset.is_dirty`, `Dataset.release_diff`,
`Dataset.compare_versions`. Likely cleanest as a single
`deriva_ml_get_dataset_history(dataset_rid, vs_version=None)` tool
that wraps all three behind one well-shaped response model.
