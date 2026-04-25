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
