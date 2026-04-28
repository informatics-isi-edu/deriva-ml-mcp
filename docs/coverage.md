# Coverage Report — deriva-mcp → deriva-ml-mcp

Tracks the disposition of every tool and resource from the old `deriva-mcp` repo
during the migration to `deriva-ml-mcp` + `deriva-mcp-core`. Built incrementally
during implementation; reviewed at every phase boundary.

See [the design spec](superpowers/specs/2026-04-24-deriva-ml-mcp-design.md#4-coverage-report-docscoveragemd)
for column meanings, disposition definitions, and maintenance rules.

## Tools

| old_name | old_module | disposition | new_name | new_module | signature_change | notes |
|---|---|---|---|---|---|---|
| add_term | vocabulary.py | dropped-to-core | add_term | (core) | none | Core covers all ML vocab needs |
| create_vocabulary | vocabulary.py | dropped-to-core | create_vocabulary | (core) | none | Core covers all ML vocab needs |
| add_synonym | vocabulary.py | dropped-to-core | add_synonym | (core) | none | Core covers all ML vocab needs |
| remove_synonym | vocabulary.py | dropped-to-core | remove_synonym | (core) | none | Core covers all ML vocab needs |
| update_term_description | vocabulary.py | dropped-to-core | update_term_description | (core) | none | Core covers all ML vocab needs |
| delete_term | vocabulary.py | dropped-to-core | delete_term | (core) | none | Core covers all ML vocab needs |
| create_dataset | dataset.py | renamed | deriva_ml_create_dataset | tools/dataset/mutate.py | execution_rid now explicit arg (was implicit from connection-manager) | Old form pulled active execution from connection singleton; new form requires explicit `execution_rid` (boundary rule) |
| get_dataset_spec | dataset.py | kept | deriva_ml_get_dataset_spec | tools/dataset/read.py | hostname/catalog_id added | Returns DatasetSpecConfig(...) string for hydra-zen configs |
| add_dataset_members | dataset.py | kept | deriva_ml_add_dataset_members | tools/dataset/mutate.py | hostname/catalog_id added | Same intent and shape |
| delete_dataset_members | dataset.py | kept | deriva_ml_delete_dataset_members | tools/dataset/mutate.py | hostname/catalog_id added | Same intent and shape |
| increment_dataset_version | dataset.py | kept | deriva_ml_increment_dataset_version | tools/dataset/mutate.py | hostname/catalog_id added | Catalog-admin lens |
| delete_dataset | dataset.py | kept | deriva_ml_delete_dataset | tools/dataset/mutate.py | hostname/catalog_id added | Soft delete with nesting refusal — distinct from generic delete_entities |
| set_dataset_description | dataset.py | dropped-to-core | update_entities | (core) | none | Pure column update; use update_entities(schema="deriva-ml", table="Dataset", entities=[{"RID": rid, "Description": "..."}]) |
| add_dataset_type | dataset.py | merged | deriva_ml_update_dataset | tools/dataset/mutate.py | combined add+remove into one tool; v1.2 renamed `update_dataset_types` -> `update_dataset` and switched to set-style `dataset_types=[...]` (BREAKING) | Single-action sugar over the bulk add_dataset_types/remove_dataset_type pair. v1.2.0 release notes document the rename. |
| remove_dataset_type | dataset.py | merged | deriva_ml_update_dataset | tools/dataset/mutate.py | combined add+remove into one tool; v1.2 renamed `update_dataset_types` -> `update_dataset` and switched to set-style `dataset_types=[...]` (BREAKING) | Same merge as add_dataset_type. v1.2.0 release notes document the rename. |
| set_dataset_description (v1.2 reabsorption) | dataset.py | merged | deriva_ml_update_dataset | tools/dataset/mutate.py | description folded into the curation-symmetric `update_dataset(description=?)` shape | v1.0 routed bare description edits to core's `update_entities`; v1.2 widened `update_dataset` to take `description` so all per-entity curation goes through one tool per type. The `update_entities` path still works (raw catalog write). |
| add_dataset_element_type | dataset.py | kept | deriva_ml_add_dataset_element_type | tools/dataset/mutate.py | hostname/catalog_id added | ML-schema mutation |
| add_dataset_child | dataset.py | dropped-redundant | deriva_ml_add_dataset_members | tools/dataset/mutate.py | (none) | A nested dataset is just a member of element-type Dataset; add_dataset_members covers it |
| list_dataset_parents | dataset.py | merged | deriva_ml_list_dataset_relations | tools/dataset/read.py | combined parents+children into one tool with direction param | Both walk Dataset_Dataset assoc; one tool, direction= "parents" or "children" |
| estimate_bag_size | dataset.py | merged | deriva_ml_bag_info | tools/dataset/read.py | dropped — bag_info returns same info plus cache status | Strict subset of bag_info |
| bag_info | dataset.py | kept | deriva_ml_bag_info | tools/dataset/read.py | hostname/catalog_id added | Comprehensive read for "describe a dataset bag" |
| cache_dataset | dataset.py | split | deriva_ml_cache_dataset (dataset only) | tools/dataset/complex.py | dataset branch kept; asset branch dropped to core hatrac | Old tool was 2-in-1 (dataset OR asset); asset path belongs in core |
| preview_denormalized_dataset | dataset.py | renamed | deriva_ml_denormalize_dataset | tools/dataset/complex.py | uses current describe_denormalized / get_denormalized_as_dict API; row mode adds limit/after_rid/preflight_count pagination | Two clean modes: catalog-shape (no rid) vs. dataset-described (with rid). Schema-path discovery is a skill/in-process concern, not an MCP tool. |
| create_dataset_type_term | dataset.py | dropped-to-core | add_term | (core) | none | Vocab-domain; core's add_term("Dataset_Type", ...) covers it |
| delete_dataset_type_term | dataset.py | dropped-to-core | delete_term | (core) | none | Vocab-domain; core's delete_term("Dataset_Type", ...) covers it |
| split_dataset | dataset.py | kept | deriva_ml_split_dataset | tools/dataset/complex.py | hostname/catalog_id added; selection_fn dropped (Python-only callable) | High-value ML-domain operation; sklearn-style API |
| create_feature | feature.py | kept | deriva_ml_create_feature | tools/feature.py | hostname/catalog_id added; drops connection-manager active-execution behavior; drops side-channel RAG warnings | Same intent. Schema mutation creating an association table + a Feature_Name term. |
| delete_feature | feature.py | kept | deriva_ml_delete_feature | tools/feature.py | hostname/catalog_id added | Drops the feature's association table; same intent as the old tool. |
| add_feature_value | feature.py | merged | deriva_ml_add_feature_values | tools/feature.py | renamed (singular -> plural); merged with add_feature_value_record into one tool taking a list of field dicts; execution_rid REQUIRED (no implicit active execution) | Old "value shorthand" branch absorbed -- caller passes `{target_rid: ..., <column>: ...}` for the simple case. Goes through `Execution.restore_execution + exe.add_features` (the only supported write path). |
| add_feature_value_record | feature.py | merged | deriva_ml_add_feature_values | tools/feature.py | combined with add_feature_value | Same merge as add_feature_value; the multi-column entry shape was already a superset of the single-column shape. |
| lookup_workflow_by_url | workflow.py | renamed | deriva_ml_find_workflow_by_url | tools/workflow.py | hostname/catalog_id added; arg renamed `url` -> `url_or_checksum` (matches underlying Python helper which already accepted either) | "find_*" verb aligns with Phase 2 cohort (`find_*` = search by predicate, `get_*` = by RID, `list_*` = page through all). Returns the catalog-bound Workflow row as JSON. |
| create_workflow | workflow.py | kept | deriva_ml_create_workflow | tools/workflow.py | hostname/catalog_id added; `url` REQUIRED (and optional `checksum`, `version`) — old tool relied on server-side git introspection which is unsafe over MCP; drops connection-manager singleton; bundles `ml.create_workflow(...)` + `ml._add_workflow(workflow)` into one call (only useful endpoint from MCP — an unregistered Workflow has no MCP surface) | Boundary rule: caller (deriva-skills / notebook) computes git URL/hash locally and passes them in. |
| set_workflow_description | workflow.py | merged | deriva_ml_update_workflow | tools/workflow.py | combined description update + workflow_type list replacement into one tool | Same merge shape as Phase 2's `update_dataset_types`. Uses Workflow's catalog-bound setters (which encapsulate the Workflow_Workflow_Type association table dance). Empty workflow_type list is rejected with `{"error": ...}` per Phase 2's argument-validation pattern. |
| add_workflow_type | workflow.py | dropped-to-core | add_term | (core) | none | `Workflow_Type` is a vocab table; `add_term("Workflow_Type", ...)` covers it identically. Mirrors the dataset-type vocab drops. |
| create_execution | execution.py | renamed | deriva_ml_create_execution | tools/execution.py | takes `workflow_rid` (not workflow_name + workflow_type — those are Phase 4's `create_workflow`); `dataset_rids` / `asset_rids` explicit; drops connection-singleton stash; `dry_run=True` doesn't audit | Workflow registration moved out per boundary discipline; caller composes `create_workflow` then `create_execution(workflow_rid=...)`. |
| start_execution | execution.py | renamed | deriva_ml_start_execution | tools/execution.py | takes explicit `execution_rid` (was zero-arg using connection singleton); rejects status != Created with `{"error": ...}` directly | Long-running mode opener per Q1 hybrid resolution. Calls `Execution.execution_start()` (Created → Running). |
| stop_execution | execution.py | merged | deriva_ml_commit_execution | tools/execution.py | new tool combines stop + upload (Stopped → Pending_Upload → Uploaded); takes explicit `execution_rid` and optional `retry_failed`; returns UploadReport summary | Old `stop_execution` only advanced to Stopped, leaving rows staged in SQLite but not flushed. Real user intent is "I'm done — make my data durable", which is the full upload sequence. See Q1 / Q3 / Q4 resolutions. |
| update_execution_status | execution.py | dropped | (none) | (none) | (none) | Free-form status setter is dangerous (lets LLM pick illegal targets); redundant once `start_execution` / `commit_execution` / `abort_execution` cover the user-facing transitions. The "set error message" use case lives on `abort_execution(reason=...)`. The "manual Pending_Upload after a hard crash" path is a Phase 7+ admin tool, not user-facing. |
| set_execution_description | execution.py | dropped-to-core | update_entities | (core) | none | Pure column update; use `update_entities("deriva-ml", "Execution", entities=[{"RID": rid, "Description": "..."}])`. Mirrors Phase 2's `set_dataset_description` decision. |
| restore_execution | execution.py | dropped | (none) | (none) | (none) | The "active execution" connection singleton is dead with the boundary-rule. Once every mutation tool takes `execution_rid` explicitly, there's no state to restore. Caller passes the RID; `list_executions` / `find_workflow_executions` finds it if forgotten. (NB: the old tool also calls `ml.restore_execution` which was renamed upstream to `ml.resume_execution` — the old tool was bit-rotted as well as redundant.) |
| create_execution_dataset | execution.py | renamed | deriva_ml_create_execution_dataset | tools/execution.py | takes explicit `execution_rid`; drops connection-singleton stash | Same intent. Calls `Execution.create_dataset(...)` to create a dataset with execution-provenance linkage. |
| add_nested_execution | execution.py | kept | deriva_ml_add_nested_execution | tools/execution.py | hostname/catalog_id added | Already takes both RIDs explicitly. Same intent; no signature change beyond the boundary args. |
| list_nested_executions | execution.py | renamed | deriva_ml_list_execution_children | tools/execution.py | name change mirrors upstream rename `Execution.list_nested_executions` → `ExecutionRecord.list_execution_children`; aligns with dataset-hierarchy template (`list_dataset_children`); hostname/catalog_id added | Net-new symmetric complement `list_execution_parents` (catalog-side parent query) lands alongside this in Phase 5.2 but is NOT a port of any old tool — it has no row here. |

### Net-new tools (post-v1.0 additions)

Tools not present in the old `deriva-mcp` repo. Added because v1.0+
landed new infrastructure (RAG, on_catalog_connect hooks, etc.) that
had no corresponding old-tool to migrate from.

| name | module | mutates | added_in | notes |
|---|---|---|---|---|
| deriva_ml_reindex_vocabularies | tools/maintenance.py | False | v1.1 (issue #5) | Force re-index of vocabulary tables in the RAG vector store. The on_catalog_connect hook bulk-indexes all vocabs on first connect; this tool is the manual bridge for after-the-fact mutations via core's `add_term` / `delete_term` (which don't fire any framework lifecycle hook -- tracked upstream as deriva-mcp-core#3). `mutates=False` because the vector store is a cache; the audit log is for catalog state changes and a cache refresh isn't one. Optional `vocab="schema.table"` arg restricts to one vocab; default re-indexes all. |
| deriva_ml_list_asset_tables | tools/asset.py | False | v1.2 (issue #2) | List the asset tables in the catalog. No pagination -- catalogs typically have a handful of asset tables. Resource form: `deriva://catalog/{h}/{c}/ml/asset-tables`. |
| deriva_ml_list_assets | tools/asset.py | False | v1.2 (issue #2) | Paginated list of rows in one asset table. Cursor pattern shared with `list_datasets` / `list_workflows`: `limit` / `after_rid` / `preflight_count`, capped at 1000. |
| deriva_ml_lookup_asset | tools/asset.py | False | v1.2 (issue #2) | Bundled per-asset detail (filename, length, md5, url, description, asset_types, executions). Same shape as the `deriva://catalog/{h}/{c}/ml/asset/{rid}` resource via the shared `_get_asset_detail_impl` helper. |
| deriva_ml_update_asset | tools/asset.py | True | v1.2 (issue #2) | Curation tool for asset metadata: `asset_types` (set-style diff -> add/remove) and `description` (free-text overwrite via pathBuilder, since `Asset` doesn't expose a write-through setter the way `Workflow` and `ExecutionRecord` do). At least one kwarg must be non-None. |
| deriva_ml_update_execution | tools/execution.py | True | v1.2 (issue #2) | Curation tool for execution metadata. Description-only by design: there is no `Execution_Type` vocabulary, and `Status` edits remain forbidden (state-machine territory driven by `start_execution` / `commit_execution` / `abort_execution`). Closes the curation-symmetry gap surfaced during v1.2 design. |
| deriva_ml_resync_indexes | tools/maintenance.py | False | v1.4 (issue #9) | Cross-user freshness bridge for the per-user-trio (Dataset / Workflow / Execution) RAG sources. v1.3's surgical re-index handles the calling user's OWN mutations; this tool refreshes after-the-fact mutations by OTHER users (or by non-MCP clients like Chaise). `target=None` (default) refreshes all of the calling user's per-user sources for the catalog; `target="<table>:<rid>"` refreshes one source surgically. `mutates=False` -- cache refresh, not catalog state change. Returns `{"resynced": {"dataset": N, "workflow": M, "execution": K}}` with success counts. Per-RID failures are swallowed and logged; partial success is normal. Mirrors the v1.1 `deriva_ml_reindex_vocabularies` precedent for "framework can't trigger automatically; ship a manual bridge." Automatic cross-user fan-out (Options 1-3 in #9) deferred -- load profile is deployment-specific and no demand signal yet. |

### Pre-existing core-owned tools (Phase 7.1 backfill)

These 52 old `deriva-mcp` tools were never analyzed in the per-domain phases (2-5) because they have no ML-domain semantics — they're generic Deriva primitives that live in `deriva-mcp-core` (or were dropped with the connection-singleton architecture). Phase 7.1's `scripts/check_coverage.py` flagged them as missing from `coverage.md`; this section catalogues the disposition for completeness.

| old_name | old_module | disposition | new_name | new_module | notes |
|---|---|---|---|---|---|
| add_asset_type | catalog.py | dropped | (none) | (none) | DerivaML-specific asset typing concept; the new `Asset_Type` vocabulary is managed via core's `add_term`. |
| add_asset_type_to_asset | catalog.py | dropped | (none) | (none) | Same — the new architecture exposes asset-type assignment via generic `update_entities` on the asset row's `Type` column. |
| add_column | schema.py | dropped-to-core | add_column | (core) tools/schema.py | Generic schema CRUD. |
| add_visible_column | annotation.py | dropped-to-core | add_visible_column | (core) tools/annotation.py | Generic annotation. |
| add_visible_foreign_key | annotation.py | dropped-to-core | add_visible_foreign_key | (core) tools/annotation.py | Generic annotation. |
| apply_annotations | annotation.py | dropped | (none) | (none) | Bulk annotation apply was a connection-singleton helper; core's per-annotation tools cover the surface without the bulk wrapper. |
| apply_catalog_annotations | annotation.py | dropped | (none) | (none) | Same as `apply_annotations` but at catalog scope; superseded by core's per-annotation tools. |
| cancel_task | background_tasks.py | dropped-to-core | cancel_task | (core) tools/tasks.py | Generic background-task management. |
| cite | catalog.py | dropped-to-core | cite | (core) tools/catalog.py | Generic citation builder. |
| clone_catalog | catalog.py | dropped-to-core | clone_catalog | (core) tools/catalog.py | Generic catalog management. |
| clone_catalog_async | catalog.py | dropped-to-core | clone_catalog | (core) tools/catalog.py | Async variant collapsed into the single `clone_catalog` in core. |
| connect_catalog | catalog.py | dropped | (none) | (none) | Connection-singleton primitive; the new architecture takes `(hostname, catalog_id)` per call. |
| create_asset_table | catalog.py | dropped | (none) | (none) | DerivaML-specific table-creation helper; replaced by generic `create_table` + asset-type vocabulary registration. |
| create_catalog | catalog.py | dropped-to-core | create_catalog | (core) tools/catalog.py | Generic catalog management. |
| create_catalog_alias | catalog.py | dropped-to-core | create_catalog_alias | (core) tools/catalog.py | Generic catalog management. |
| create_table | schema.py | dropped-to-core | create_table | (core) tools/schema.py | Generic schema CRUD. |
| delete_catalog | catalog.py | dropped-to-core | delete_catalog | (core) tools/catalog.py | Generic catalog management. |
| delete_catalog_alias | catalog.py | dropped-to-core | delete_catalog_alias | (core) tools/catalog.py | Generic catalog management. |
| disconnect_catalog | catalog.py | dropped | (none) | (none) | Connection-singleton primitive; nothing to disconnect under the per-call boundary. |
| get_handlebars_template_variables | annotation.py | dropped-to-core | get_handlebars_template_variables | (core) tools/annotation.py | Generic annotation introspection. |
| get_record | data.py | dropped-to-core | get_entities | (core) tools/entity.py | Renamed in core. |
| get_table_sample_data | annotation.py | dropped-to-core | get_table_sample_data | (core) tools/annotation.py | Generic annotation introspection. |
| get_task_status | background_tasks.py | dropped-to-core | get_task_status | (core) tools/tasks.py | Generic background-task management. |
| insert_records | data.py | dropped-to-core | insert_entities | (core) tools/entity.py | Renamed in core. |
| invalidate_cache | cache.py | dropped | (none) | (none) | Per-connection result cache from the dead connection-singleton era; no successor concept. |
| list_asset_executions | catalog.py | dropped | (none) | (none) | DerivaML-specific asset/execution join; the new architecture exposes this via `list_executions` + per-execution assets in the execution detail resource. |
| list_cached_results | cache.py | dropped | (none) | (none) | Connection-singleton result cache; gone with that era. |
| list_catalog_registry | catalog.py | dropped | (none) | (none) | The catalog registry is exposed as a core MCP resource (`deriva://registry/{hostname}`) instead of a tool. |
| list_tasks | background_tasks.py | dropped-to-core | list_tasks | (core) tools/tasks.py | Generic background-task management. |
| preview_table | data.py | dropped-to-core | get_entities | (core) tools/entity.py | Use `get_entities(..., limit=N)` in core. |
| query_cached_result | cache.py | dropped | (none) | (none) | Connection-singleton result cache; gone with that era. |
| remove_asset_type_from_asset | catalog.py | dropped | (none) | (none) | Asset-type removal goes through generic `update_entities` setting the `Type` column to null. |
| remove_visible_column | annotation.py | dropped-to-core | remove_visible_column | (core) tools/annotation.py | Generic annotation. |
| remove_visible_foreign_key | annotation.py | dropped-to-core | remove_visible_foreign_key | (core) tools/annotation.py | Generic annotation. |
| reorder_visible_columns | annotation.py | dropped-to-core | reorder_visible_columns | (core) tools/annotation.py | Generic annotation. |
| reorder_visible_foreign_keys | annotation.py | dropped-to-core | reorder_visible_foreign_keys | (core) tools/annotation.py | Generic annotation. |
| set_active_catalog | catalog.py | dropped | (none) | (none) | Connection-singleton primitive; gone with the per-call boundary. |
| set_column_description | schema.py | dropped-to-core | set_column_description | (core) tools/schema.py | Generic schema CRUD. |
| set_column_display | annotation.py | dropped-to-core | set_column_display | (core) tools/annotation.py | Generic annotation. |
| set_column_display_name | annotation.py | dropped-to-core | set_column_display_name | (core) tools/annotation.py | Generic annotation. |
| set_column_nullok | schema.py | dropped-to-core | set_column_nullok | (core) tools/schema.py | Generic schema CRUD. |
| set_default_schema | catalog.py | dropped | (none) | (none) | Connection-singleton primitive; the new architecture passes `(schema, table)` per call. |
| set_display_annotation | annotation.py | dropped-to-core | set_display_annotation | (core) tools/annotation.py | Generic annotation. |
| set_row_name_pattern | annotation.py | dropped-to-core | set_row_name_pattern | (core) tools/annotation.py | Generic annotation. |
| set_table_description | schema.py | dropped-to-core | set_table_description | (core) tools/schema.py | Generic schema CRUD. |
| set_table_display | annotation.py | dropped-to-core | set_table_display | (core) tools/annotation.py | Generic annotation. |
| set_table_display_name | annotation.py | dropped-to-core | set_table_display_name | (core) tools/annotation.py | Generic annotation. |
| set_visible_columns | annotation.py | dropped-to-core | set_visible_columns | (core) tools/annotation.py | Generic annotation. |
| set_visible_foreign_keys | annotation.py | dropped-to-core | set_visible_foreign_keys | (core) tools/annotation.py | Generic annotation. |
| update_catalog_alias | catalog.py | dropped-to-core | update_catalog_alias | (core) tools/catalog.py | Generic catalog management. |
| update_record | data.py | dropped-to-core | update_entities | (core) tools/entity.py | Renamed in core. |
| validate_rids | data.py | dropped | (none) | (none) | Bulk-RID validation was a convenience over per-call lookups; use core's `get_entities` per RID. |

## Resources

| old_uri | old_module | disposition | new_uri | new_module | notes |
|---|---|---|---|---|---|
| deriva://server/version | resources.py | dropped-to-core | deriva://server/status | (core) | Core's `deriva://server/status` already returns version + auth + RAG state |
| deriva://config/deriva-ml-template | resources.py | deferred | (none) | (none) | Hydra-zen Python template; belongs in `deriva-skills` (consumer of these patterns), not as an MCP resource. Phase 6+ follow-up via prompts |
| deriva://config/dataset-spec-template | resources.py | deferred | (none) | (none) | Hydra-zen Python template; same disposition as `config/deriva-ml-template` |
| deriva://config/execution-template | resources.py | deferred | (none) | (none) | Hydra-zen Python template; same disposition as `config/deriva-ml-template` |
| deriva://config/model-template | resources.py | deferred | (none) | (none) | Hydra-zen Python template; same disposition as `config/deriva-ml-template` |
| deriva://config/experiment-template | resources.py | deferred | (none) | (none) | Hydra-zen Python template; same disposition as `config/deriva-ml-template` |
| deriva://config/multirun-template | resources.py | deferred | (none) | (none) | Hydra-zen Python template; same disposition as `config/deriva-ml-template` |
| deriva://catalog/schema | resources.py | dropped-to-core | deriva://catalog/{h}/{c}/schema | (core) | Core ships full ERMrest schema JSON keyed on hostname/catalog_id |
| deriva://catalog/vocabularies | resources.py | dropped-redundant | (none) | (none) | Iterating all vocab tables can be done with core `lookup_term` per-vocab plus the `registries` snapshot for the 4 ML vocabs. No global "all vocabs" payload needed at MCP level — it grows unboundedly |
| deriva://catalog/datasets | resources.py | kept | deriva://catalog/{h}/{c}/ml/datasets | resources/ml.py | Lists datasets with rid/description/types/version. Same payload shape, catalog-scoped |
| deriva://catalog/dataset-element-types | resources.py | dropped-redundant | deriva_ml_list_dataset_element_types | tools/dataset/read.py | Already exposed as a tool in Phase 2; resource form is duplicate surface |
| deriva://catalog/workflows | resources.py | kept | deriva://catalog/{h}/{c}/ml/workflows | resources/ml.py | Lists workflows with rid/name/url/type/description. Same payload, catalog-scoped |
| deriva://catalog/workflow-types | resources.py | merged | deriva://catalog/{h}/{c}/ml/registries | resources/ml.py | Folded into the `registries` one-shot snapshot alongside dataset_types/asset_types/execution_statuses |
| deriva://catalog/features | resources.py | dropped-redundant | deriva_ml_list_features | tools/feature.py | `list_features` (Phase 3) returns feature names; resource form duplicates the tool |
| deriva://catalog/tables | resources.py | dropped-to-core | deriva://catalog/{h}/{c}/tables | (core) | Core ships flat list of {schema, table, comment} for every non-system table |
| deriva://catalog/dataset-types | resources.py | merged | deriva://catalog/{h}/{c}/ml/registries | resources/ml.py | Folded into the `registries` one-shot snapshot |
| deriva://dataset/{dataset_rid} | resources.py | kept | deriva://catalog/{h}/{c}/ml/dataset/{dataset_rid} | resources/ml.py | Detail payload: description, types, current_version, member_counts, children, parents, version_history. Drops the old `_related_docs`/`_related_data` RAG side-channel — RAG enrichment is now caller-driven via `rag_search` |
| deriva://dataset/{dataset_rid}/members | resources.py | kept | deriva://catalog/{h}/{c}/ml/dataset/{dataset_rid}/members | resources/ml.py | Members grouped by table with RIDs and counts. Same shape, catalog-scoped |
| deriva://dataset/{dataset_rid}/versions | resources.py | merged | deriva://catalog/{h}/{c}/ml/dataset/{dataset_rid} | resources/ml.py | Version history is included as `version_history` key in the dataset detail payload (matches old `dataset/{rid}` behavior). No separate URI needed — full history is bounded per dataset |
| deriva://dataset/{dataset_rid}/bag-preview | resources.py | deferred | (none) | (none) | FK-path debugging for bag exports. Specialized concern; revisit when bag-export tooling matures |
| deriva://catalog/element-type-paths | resources.py | deferred | (none) | (none) | FK-path debugging across all element types. Same disposition as `bag-preview` |
| deriva://table/{table_name}/features | resources.py | kept | deriva://catalog/{h}/{c}/ml/features/{table_name} | resources/ml.py | Features defined on a target table; same payload shape (name, target_table, feature_table, asset/term/value columns) |
| deriva://feature/{table_name}/{feature_name} | resources.py | dropped-redundant | deriva_ml_get_feature | tools/feature.py | `get_feature` (Phase 3) returns the same column-detail payload (term/asset/value cols, required_fields). The resource form is a tool duplicate |
| deriva://feature/{table_name}/{feature_name}/values | resources.py | dropped-redundant | deriva_ml_list_feature_values | tools/feature.py | `list_feature_values` covers the same data with selector and pagination support |
| deriva://table/{table_name}/feature-values | resources.py | dropped-redundant | deriva_ml_list_feature_values | tools/feature.py | `list_feature_values` (no selector) returns all rows; the per-table grouping is a client-side concern |
| deriva://table/{table_name}/feature-values/newest | resources.py | dropped-redundant | deriva_ml_list_feature_values | tools/feature.py | `list_feature_values(selector="newest")` covers it |
| deriva://table/{table_name}/feature-values/first | resources.py | dropped-redundant | deriva_ml_list_feature_values | tools/feature.py | `list_feature_values(selector="first")` covers it |
| deriva://table/{table_name}/feature-values/majority_vote | resources.py | dropped-redundant | deriva_ml_list_feature_values | tools/feature.py | `list_feature_values(selector="majority_vote")` covers it |
| deriva://vocabulary/{vocab_name} | resources.py | dropped-to-core | lookup_term / list_vocabulary_terms | (core) | Per-vocab term listing belongs in core's vocab tools; the 4 ML-specific vocabs are also surfaced via `registries` |
| deriva://vocabulary/{vocab_name}/{term_name} | resources.py | dropped-to-core | lookup_term | (core) | Single-term lookup is a core vocab tool |
| deriva://table/{table_name}/schema | resources.py | dropped-to-core | deriva://catalog/{h}/{c}/table/{schema}/{table} | (core) | Core ships full ERMrest table definition (columns, FKs, annotations, keys) |
| deriva://table/{table_name}/assets | resources.py | deferred | (none) | (none) | Asset listing per table. No ML-side asset tooling planned for Phase 6; revisit if asset workflows need MCP surface |
| deriva://workflow/{workflow_rid} | resources.py | kept | deriva://catalog/{h}/{c}/ml/workflow/{workflow_rid} | resources/ml.py | Detail payload: name, type, description, url, checksum, version, is_notebook |
| deriva://table/{table_name}/annotations | resources.py | dropped-to-core | deriva://catalog/{h}/{c}/table/{schema}/{table} | (core) | Annotations are part of the core table-definition payload |
| deriva://table/{table_name}/column/{column_name}/annotations | resources.py | dropped-to-core | deriva://catalog/{h}/{c}/table/{schema}/{table} | (core) | Column annotations are nested in the core table-definition payload under `column_definitions` |
| deriva://table/{table_name}/foreign-keys | resources.py | dropped-to-core | deriva://catalog/{h}/{c}/table/{schema}/{table} | (core) | Foreign keys (outbound and inbound) are part of the core table-definition payload |
| deriva://docs/annotation-contexts | resources.py | dropped | (none) | (none) | Static reference doc about Chaise annotation contexts. Annotation tooling is core's territory; if needed, lives in core's docs subsystem, not the ML plugin |
| deriva://docs/{suffix} (loop, 17 endpoints) | resources.py | renamed | rag_search(source="github:informatics-isi-edu/deriva-ml") | (core RAG) | The 17 docs endpoints (DerivaML overview/datasets/features/hydra-zen/notebooks/etc., ERMrest, Chaise, deriva-py guides) collapse into one GitHub-backed RAG source registered via `ctx.rag_github_source(...)` in Phase 6.3. Callers query via `rag_search` instead of one URI per topic |
| deriva://catalog/asset-tables | resources.py | kept | deriva://catalog/{h}/{c}/ml/asset-tables | resources/ml.py | v1.2 (issue #2): asset-table listing landed alongside the asset MCP surface. Returns `{asset_tables: [{name, schema}], count}`. Tool form: `deriva_ml_list_asset_tables`. |
| deriva://catalog/assets | resources.py | deferred | (none) | (none) | Per-table asset summaries with counts. Use `deriva_ml_list_assets(asset_table=...)` per-table instead -- the cross-table aggregate stayed out of v1.2 scope. |
| deriva://asset/{asset_rid} | resources.py | kept | deriva://catalog/{h}/{c}/ml/asset/{asset_rid} | resources/ml.py | v1.2 (issue #2): bundled asset detail (filename, length, md5, url, description, asset_types, executions). Tool form: `deriva_ml_lookup_asset`. The two share an internal helper so payloads cannot drift. |
| deriva://catalog/executions | resources.py | kept | deriva://catalog/{h}/{c}/ml/executions | resources/ml.py | Recent executions (rid/workflow/status/description). Old form had a 50-row cap; new form will paginate via standard query params |
| deriva://execution/{execution_rid} | resources.py | kept | deriva://catalog/{h}/{c}/ml/execution/{execution_rid} | resources/ml.py | Detail payload absorbs old `execution/{rid}` plus inputs+outputs+metadata as nested keys (see merged rows below) |
| deriva://catalog/info | resources.py | dropped | (none) | (none) | Old "active connection" catalog summary. The implicit-connection model is dead; hostname/catalog_id are explicit on every resource URI now |
| deriva://catalog/users | resources.py | dropped | (none) | (none) | Per-catalog user listing. No identified ML caller; if needed, belongs in core (catalog-level concern, not ML-domain) |
| deriva://catalog/connections | resources.py | dropped | (none) | (none) | Old connection-singleton state. The new architecture has no implicit connections to enumerate |
| deriva://chaise-url/{table_or_rid} | resources.py | dropped-to-core | (core) | (core) | Chaise URL synthesis is a generic catalog concern, not ML-specific. If core ships a `chaise_url` tool/resource, it lives there |
| deriva://rid/{rid} | resources.py | dropped-to-core | (core) | (core) | RID resolution to {schema, table, url} is a generic catalog concern; belongs alongside core's catalog tools |
| deriva://cite/{rid} | resources.py | dropped-to-core | cite | (core) | The `cite` tool already exists in the core MCP surface; no resource form needed |
| deriva://registry/{hostname} | resources.py | dropped-to-core | list_catalog_registry | (core) | Server-level catalog/alias discovery is core's territory; tool form already exists |
| deriva://alias/{hostname}/{alias_name} | resources.py | dropped-to-core | (core) | (core) | Per-alias metadata; same disposition as `registry/{hostname}` |
| deriva://execution/{execution_rid}/inputs | resources.py | merged | deriva://catalog/{h}/{c}/ml/execution/{execution_rid} | resources/ml.py | Input datasets and assets returned as `inputs` key in the execution detail payload |
| deriva://execution/{execution_rid}/outputs | resources.py | merged | deriva://catalog/{h}/{c}/ml/execution/{execution_rid} | resources/ml.py | Output assets grouped by table returned as `outputs` key in the execution detail payload |
| deriva://execution/{execution_rid}/metadata | resources.py | merged | deriva://catalog/{h}/{c}/ml/execution/{execution_rid} | resources/ml.py | Auto-created metadata files (Deriva_Config, Hydra_Config, Execution_Config, Runtime_Env) returned as `metadata` key in the execution detail payload |
| deriva://experiment/{execution_rid} | resources.py | merged | deriva://catalog/{h}/{c}/ml/execution/{execution_rid} | resources/ml.py | Experiment is just an execution with Hydra config; the summary fields fold into the execution detail payload's `experiment` key when present |
| deriva://catalog/experiments | resources.py | dropped-redundant | deriva://catalog/{h}/{c}/ml/executions | resources/ml.py | An "experiment" is an execution with Hydra config; callers can filter the executions list. No separate URI |
| deriva://storage/summary | resources.py | dropped | (none) | (none) | Local filesystem state (~/.deriva-ml/), not catalog state. Out of MCP scope |
| deriva://storage/cache | resources.py | dropped | (none) | (none) | Local filesystem cache stats; same disposition as `storage/summary` |
| deriva://storage/execution-dirs | resources.py | dropped | (none) | (none) | Local filesystem execution directories; same disposition as `storage/summary` |
| deriva://cache/results | resources.py | dropped | (none) | (none) | Per-connection result cache from the dead connection-singleton era; no successor concept in the new architecture |

## v1.3 surgical per-RID re-index design

Each Dataset / Workflow / Execution chunk now lives in its own per-user
source named `data:{host}:{cat}:{user_id}:{table}:{rid}` (one source per
user-table-RID tuple) rather than the v1.0 bulk-per-user shape
`data:{host}:{cat}:{user_id}`. Mutating tools call `_reindex_<entity>`
(or `_delete_dataset_source` for `deriva_ml_delete_dataset`) inline
after the catalog mutation succeeds so the affected source is refreshed
immediately and `rag_search` finds the change on the very next call from
the same user. The re-index is best-effort: a failure logs but does NOT
propagate to the tool's success path. The audit event for the catalog
mutation always fires before the re-index call so audit captures the
success regardless of cache state. Required a small upstream change to
the `rag_search` user-id filter (extended from exact equality to also
prefix-match `data:{host}:{cat}:{user_id}:` for the v1.3 per-RID shape;
the trailing colon prevents accidental string-prefix overlap from
weakening per-user isolation). Cross-user freshness (user A's mutation
-> user B's RAG view) is still imperfect and tracked separately as #9
for v1.4.

## Upstream gaps (Phase 6 RAG)

Two open `deriva-mcp-core` limitations. The other (issue #1) was filed during
Phase 6.6 and landed upstream in commit
[`0aef9fc`](https://github.com/informatics-isi-edu/deriva-mcp-core/commit/0aef9fc)
during the v1.0 polish sprint -- `rag_search` now filters `data:` sources by
user_id symmetrically with the schema-source filter, closing the cross-user
read leak. The plugin no longer carries a TODO marker for it.

Still open:

- **`index_table_data` hardcodes `doc_type="catalog-data"`**
  ([deriva-mcp-core#2](https://github.com/informatics-isi-edu/deriva-mcp-core/issues/2);
  `rag/data.py` lines 138, 142). Plugins cannot tag per-user catalog chunks
  with a domain-specific doc_type, so `rag_search(doc_type="ml-dataset")`
  cannot distinguish Dataset chunks from Workflow or Execution chunks. The
  rendered Markdown header (`## Dataset: <RID>`) still tags chunks
  inline for the LLM. Fix is a one-line `doc_type` parameter addition
  upstream. The plugin marks the gap with `# TODO(upstream-rag-doctype)` at
  the `index_table_data` call site in `resources/rag.py`.

- **No `on_data_change` lifecycle hook**
  ([deriva-mcp-core#3](https://github.com/informatics-isi-edu/deriva-mcp-core/issues/3)).
  Vocabulary mutations via core's `add_term` / `delete_term` do not fire
  `on_schema_change` (verified by reading core's `vocabulary.py`), and there
  is no `on_data_change` equivalent. The v1.1 vocab indexer in
  `resources/rag.py` therefore can't auto-refresh after term mutations.
  Bridge: the `deriva_ml_reindex_vocabularies` tool exposes a manual
  re-index that callers (or the LLM) can fire after `add_term`. Long-term
  fix is the upstream hook; the manual tool stays as the targeted-refresh
  affordance even after the hook lands.

## Validation (Phase 7+)

Once `scripts/check_coverage.py` lands in Task 7.1, run:

```bash
uv run python scripts/check_coverage.py
```

It walks the old `deriva-mcp` tool/resource definitions and confirms every name
appears exactly once in this file. Until then, the coverage table is reviewed
manually at each phase boundary (see Definition of Done in the design spec).
