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
| create_dataset | dataset.py | renamed | create_dataset | tools/dataset.py | execution_rid now explicit arg (was implicit from connection-manager) | Old form pulled active execution from connection singleton; new form requires explicit `execution_rid` (boundary rule) |
| get_dataset_spec | dataset.py | kept | get_dataset_spec | tools/dataset.py | hostname/catalog_id added | Returns DatasetSpecConfig(...) string for hydra-zen configs |
| add_dataset_members | dataset.py | kept | add_dataset_members | tools/dataset.py | hostname/catalog_id added | Same intent and shape |
| delete_dataset_members | dataset.py | kept | delete_dataset_members | tools/dataset.py | hostname/catalog_id added | Same intent and shape |
| increment_dataset_version | dataset.py | kept | increment_dataset_version | tools/dataset.py | hostname/catalog_id added | Catalog-admin lens |
| delete_dataset | dataset.py | kept | delete_dataset | tools/dataset.py | hostname/catalog_id added | Soft delete with nesting refusal — distinct from generic delete_entities |
| set_dataset_description | dataset.py | dropped-to-core | update_entities | (core) | none | Pure column update; use update_entities(schema="deriva-ml", table="Dataset", entities=[{"RID": rid, "Description": "..."}]) |
| add_dataset_type | dataset.py | merged | update_dataset_types | tools/dataset.py | combined add+remove into one tool | Single-action sugar over the bulk add_dataset_types/remove_dataset_type pair |
| remove_dataset_type | dataset.py | merged | update_dataset_types | tools/dataset.py | combined add+remove into one tool | Same merge as add_dataset_type |
| add_dataset_element_type | dataset.py | kept | add_dataset_element_type | tools/dataset.py | hostname/catalog_id added | ML-schema mutation |
| add_dataset_child | dataset.py | dropped-redundant | add_dataset_members | tools/dataset.py | (none) | A nested dataset is just a member of element-type Dataset; add_dataset_members covers it |
| list_dataset_parents | dataset.py | merged | list_dataset_relations | tools/dataset.py | combined parents+children into one tool with direction param | Both walk Dataset_Dataset assoc; one tool, direction= "parents" or "children" |
| estimate_bag_size | dataset.py | merged | bag_info | tools/dataset.py | dropped — bag_info returns same info plus cache status | Strict subset of bag_info |
| bag_info | dataset.py | kept | bag_info | tools/dataset.py | hostname/catalog_id added | Comprehensive read for "describe a dataset bag" |
| cache_dataset | dataset.py | split | cache_dataset (dataset only) | tools/dataset.py | dataset branch kept; asset branch dropped to core hatrac | Old tool was 2-in-1 (dataset OR asset); asset path belongs in core |
| preview_denormalized_dataset | dataset.py | renamed | denormalize_dataset | tools/dataset.py | uses current describe_denormalized / get_denormalized_as_dict API | Same intent (preview wide table); covers list_schema_paths via mode flag |
| create_dataset_type_term | dataset.py | dropped-to-core | add_term | (core) | none | Vocab-domain; core's add_term("Dataset_Type", ...) covers it |
| delete_dataset_type_term | dataset.py | dropped-to-core | delete_term | (core) | none | Vocab-domain; core's delete_term("Dataset_Type", ...) covers it |
| split_dataset | dataset.py | kept | split_dataset | tools/dataset.py | hostname/catalog_id added; selection_fn dropped (Python-only callable) | High-value ML-domain operation; sklearn-style API |

## Resources

| old_uri | old_module | disposition | new_uri | new_module | notes |
|---|---|---|---|---|---|

## Validation (Phase 7+)

Once `scripts/check_coverage.py` lands in Task 7.1, run:

```bash
uv run python scripts/check_coverage.py
```

It walks the old `deriva-mcp` tool/resource definitions and confirms every name
appears exactly once in this file. Until then, the coverage table is reviewed
manually at each phase boundary (see Definition of Done in the design spec).
