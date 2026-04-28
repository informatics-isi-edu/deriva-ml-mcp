# CLAUDE.md

This file provides guidance to Claude Code when working with the deriva-ml-mcp codebase.

See [`../CLAUDE.md`](../CLAUDE.md) for shared workspace conventions: `uv` for everything,
Google-style docstrings with examples, no backwards-compat shims, no over-engineering,
shared `bump-version` rules, and standard `pytest` / `ruff` invocations.

## Project Overview

`deriva-ml-mcp` is a plugin for [`deriva-mcp-core`](../deriva-mcp-core) that exposes
DerivaML domain workflows (datasets, features, workflows, executions) as MCP tools
and resources. The plugin is discovered via the `deriva_mcp.plugins` entry-point
group at server startup.

## Architecture

Current shape (v1.1.x):

```
src/deriva_ml_mcp/
├── plugin.py            # register(ctx) entry point -- dispatches to module registrars
├── ml_context.py        # The single helper that builds a DerivaML from core's credential
├── _helpers.py          # _error_envelope, _paginate, _read_rid, _table_qname,
│                        # _table_to_dict, _MAX_LIMIT (shared)
├── prompts.py           # 3 MCP prompts: deriva_ml_getting_started /
│                        #   _execution_lifecycle / _workflow_dedup
├── tools/               # Per-domain tool modules (40 tools across 5 domains)
│   ├── dataset/         #   17 tools, split into focused submodules:
│   │   ├── __init__.py  #     register aggregator + helper re-exports
│   │   ├── read.py      #     7 read tools + 4 shared helpers
│   │   ├── mutate.py    #     7 simple mutation tools
│   │   └── complex.py   #     3 complex tools (cache, denormalize, split)
│   ├── feature.py       #    6 tools
│   ├── workflow.py      #    5 tools
│   ├── execution.py     #   11 tools
│   └── vocabulary.py    #    1 tool (deriva_ml_reindex_vocabularies)
└── resources/           # MCP resources + per-user RAG
    ├── ml.py            #   9 read-only resources under deriva://catalog/{h}/{c}/ml/...
    └── rag.py           #   1 GitHub doc source +
                         #   3 per-user on_catalog_connect hooks (Dataset/Workflow/Execution)
                         # + 1 catalog-public on_catalog_connect hook (vocabularies)
```

History: Phase 0 shipped `plugin.py` and the empty `tools/`/`resources/`
package markers. `ml_context.py` arrived in Phase 1; the domain tool
modules in Phases 2-5; `resources/ml.py` and `resources/rag.py` in
Phase 6; v1.0 polish added `prompts.py` + the `deriva_ml_*` tool name
prefix; v1.1 added vocabulary RAG indexing + `tools/vocabulary.py` +
the discovery-pattern prompt section; v1.x split `tools/dataset.py`
into a focused-submodule package.

**Per-user RAG safety.** `resources/rag.py` does NOT use
`ctx.rag_dataset_indexer(...)` -- that API produces a single global
enriched source shared across users (`enriched:{host}:{cat}:{schema}:{table}`),
which leaks data for any catalog table where rows have user-specific
ACLs. Instead, the plugin uses `ctx.on_catalog_connect(...)` with
direct `store.delete_source + store.add` writes so each user's chunks
land under per-RID source names of the shape
`data:{host}:{cat}:{user_id}:{table}:{rid}` (v1.3) and ACL is applied
at fetch time using the calling user's credential. The `data:` prefix
keeps upstream's `rag_search` user-id filter in play (it accepts both
the legacy bulk form `data:{host}:{cat}:{user_id}` and prefix-match
on the per-RID form). The plugin-level test
`test_register_does_not_use_rag_dataset_indexer` pins the unsafe-API
ban; `test_data_sources_use_per_rid_naming` pins the v1.3 source-name
shape; future commits accidentally calling the unsafe API or
regressing the source-name shape will fail CI.

**Surgical per-RID re-index (v1.3).** Each mutating tool that affects
a Dataset / Workflow / Execution row calls `_reindex_<entity>` (a
lazy import inside the tool body, mirroring the v1.1 vocab indexer
pattern) immediately after the catalog mutation succeeds. The
re-index is best-effort: any failure is logged but does NOT propagate
to the tool's success path (the catalog mutation already succeeded).
The audit event for the catalog mutation always fires before the
re-index call so audit captures the success regardless of cache
state. Result: a freshly created or modified row is searchable via
`rag_search` on the very next call from the same user.

**Cross-user freshness gap (v1.4 — known limitation).** Per-user
surgical re-index covers the calling user's OWN mutations. It does
NOT propagate across users: when user A mutates a dataset visible
to user B, B's per-user sources remain stale until B reconnects.
Same gap applies to mutations from non-MCP clients (Chaise UI,
ERMrest direct). v1.4 ships a manual bridge: the
`deriva_ml_resync_indexes(hostname, catalog_id, target=None)` tool
in `tools/vocabulary.py` re-runs `_reindex_*` for the calling
user's per-user sources -- either all (target=None) or one
(target="dataset:1-AAAA"). The `deriva_ml_getting_started` prompt
documents the verify-with-`get_*` recovery pattern for shared-visible
rows. Automatic cross-user fan-out is deferred per the design
discussion in #9 -- the load profile is deployment-specific and
no demand signal yet.

Vocabularies are the documented exception: vocab content is catalog-
public (no per-user ACL), so the plugin writes vocab terms directly
to the vector store under a custom `vocab:{host}:{cat}:{schema}.{table}`
prefix that bypasses upstream's `data:` user-id filter and serves
chunks to all users in the catalog. See `resources/rag.py`'s module
docstring for the full rationale.

One upstream gap remains: deriva-mcp-core#2 (`index_table_data`
should accept a `doc_type` parameter so per-table RAG chunks can
be filtered by `rag_search(doc_type="ml-dataset")`). Tracked
inline at the call site with `# TODO(upstream-rag-doctype)`.

**Boundary rules.** This plugin **never** duplicates anything that lives in
`deriva-mcp-core`:

- Connection management (use `get_request_credential()` via `ml_context.py`)
- Auth, audit machinery, RAG subsystem
- Vocabulary tools (core's `add_term`/`lookup_term`/etc. cover all ML needs)
- Generic Deriva primitives (entity CRUD, schema introspection, hatrac)

If a primitive is needed and core lacks it, file an issue against core — do not
work around it here.

## Tool Implementation Rules

Inherited from `deriva-mcp-core`'s plugin authoring guide
(`../deriva-mcp-core/docs/plugin-authoring-guide.md`):

- Every tool registered with explicit `mutates=True` or `mutates=False`. Omitting
  raises `TypeError` at server startup.
- Wrap DERIVA I/O in `with deriva_call():` from `deriva_mcp_core`.
- Every `mutates=True` tool emits `audit_event(...)` on both success and failure.
- Audit event names use the `<plugin>_<operation>` convention: `deriva_ml_<op>`
  on success, `deriva_ml_<op>_failed` on failure.
- All tool names are prefixed `deriva_ml_<verb>` (e.g. `deriva_ml_create_dataset`).
  Tool names, audit event names, and prompt names use the same convention so
  searches and log greps line up cleanly. Sibling-plugin collisions are
  prevented by construction.

## Tool / Resource Dual-Mode Policy

For every read-only ML domain object (dataset, workflow, execution, feature),
the plugin ships **both** a paginated tool *and* a fixed-shape resource:

| Use case | Reach for |
|---|---|
| Filtered scan (status=Failed, workflow=X, page through 5K rows) | `deriva_ml_list_<x>` tool with `limit` / `after_rid` / domain filters |
| "What's in this catalog right now?" snapshot for the LLM to ground on | `deriva://catalog/{h}/{c}/ml/<x>s` resource (capped at 1000) |
| One known RID, full detail | `deriva://catalog/{h}/{c}/ml/<x>/{rid}` resource |
| Semantic discovery ("workflows that train CNN models") | `rag_search(doc_type="ml-docs"|"catalog-data")` |

Resources are intentionally not parameterized beyond the URI template — they
have no query string. They are cheap, cacheable, audit-free reads. Tools carry
the filter / sort / pagination surface and the audit-on-failure path.

When adding a new read endpoint:

- If it's a fixed-shape "give me X by Y" with no filters, ship as a resource.
- If it benefits from `limit` / `after_rid` / domain filters, ship as a tool.
- If both, ship both — they MUST share the data-fetch helper (`_<verb>_impl`
  in the relevant `tools/<domain>.py`) so the resource and tool responses can
  never drift in shape.

This was decided after the v1.0 reviews surfaced dataset/workflow/execution
read endpoints landing as both tools (paginated) and resources (snapshot).
Don't replay the debate — write the helper once, register it twice.

### Response models (Pydantic, v1.6+)

Every list / detail / summary response shape is declared as a
`pydantic.BaseModel` in `src/deriva_ml_mcp/_response_models.py`. The shared
`_<verb>_impl` helpers return Pydantic instances; the wrapper and resource
serialize via `payload.model_dump_json(by_alias=True)` (replaces the v1.5
pattern of `json.dumps(payload)` over a plain dict).

This is a **named, validated** contract -- helpers declare `-> DatasetSummary`
instead of `-> dict[str, Any]`, and `model_config = ConfigDict(extra="forbid")`
on every model catches helper-side typos and upstream-shape drift at
construction time (raises `ValidationError` -- caught by the wrapper's
`_error_envelope` and returned as `{"error": "..."}` on the wire).

When adding a new shared helper:

1. Add a Pydantic model to `_response_models.py` for the response shape.
   Use `extra="forbid"` and explicit field types. For nullable fields, use
   `T | None` and provide a `= None` default if the helper might omit the
   field.
2. Annotate the helper's return type as the new model.
3. Construct the model directly: `return DatasetListResponse(datasets=...,
   count=..., truncated=..., next_after_rid=...)`.
4. In wrappers and resources, replace `json.dumps(payload)` with
   `payload.model_dump_json(by_alias=True)` (the `by_alias=True` flag
   handles fields like `AssetTableRef.schema_` that alias to wire keys
   like `"schema"`).
5. The intermediate `_summarize_*` helpers (e.g. `_summarize_dataset`)
   still return `dict[str, Any]` — they're called from many ad-hoc payload
   sites and converting at every call site is out of scope for v1.6 (PR 2
   in the v2.0 sprint will sweep those). The list response helpers
   construct the model from the dicts these helpers produce:
   `DatasetSummary(**_summarize_dataset(d))`.

v2.0 (PR 2 of #6) extended the Pydantic coverage to two more shapes
and shipped two breaking wire-shape changes:

- `_get_execution_detail_impl` is now Pydantic-typed (`ExecutionDetail`
  with `ExecutionInputs` / `ExecutionOutputs` / `ExecutionExperiment`
  sub-models). Wire shape: same nested structure as v1.x, but the
  `experiment` field is now ALWAYS present (set to `null` when the
  execution is not a Hydra-driven experiment) instead of conditionally
  omitted. Consumers that did `if "experiment" in payload:` should
  switch to `if payload.get("experiment") is not None:`.
- The `deriva_ml_get_dataset` tool wrapper now returns the same wire
  shape as the `deriva://catalog/{h}/{c}/ml/dataset/{rid}` resource:
  the field is `version_history` (was `history` in v1.x), and it's
  always present (empty list when `include_history=False`). This
  unifies the tool + resource shapes so consumers can switch freely
  between them.

v2.1 introduced `PreflightCountResponse` — one shared model used by
the `preflight_count=True` branch of every paginated tool. Removes
the DRY violation where each paginated tool built the same
`{total_count, entities_fetched, action_required}` dict inline.
The model uses `extra="allow"` to permit per-call-site context
fields (`element_table`, `mode`, `dataset_rid`).

v2.2 swept the `_summarize_*` building-block helpers to return
Pydantic instances (`DatasetSummary`, `WorkflowSummary`,
`ExecutionSummary`, `FeatureSummary`, `AssetSummary`). Consumers
that previously dict-accessed the summary now use attribute access
or call `.model_dump()` to get a plain dict back. Ad-hoc payload
sites (parents/children in `deriva_ml_list_dataset_relations`,
inline sub-payloads in `deriva_ml_find_workflow_executions` /
`_list_execution_children` / `_list_execution_parents`) call
`.model_dump()` to bridge to the surrounding `json.dumps(...)`. The
RAG indexer's `_fetch_*_rows` and `_reindex_*` helpers were updated
to use `.model_dump(mode="json")` (replaces the v1.x
`json.loads(json.dumps(row, default=str))` round-trip; cleaner
single-call coercion of datetimes etc.).

Future v2.x sweeps (each its own PR):

- v2.3: ad-hoc payload Pydantic models. The list_dataset_relations,
  find_workflow_executions, list_execution_children, and
  list_execution_parents wrappers each build inline
  `{<plural>: [...], count, ...}` shapes with one-off context fields
  that should become named models. v2.2's `.model_dump()` bridge
  gets replaced by directly constructing the model.
- v3.0 (PR 3 of #6): mutating-tool response shapes (currently
  heterogeneous: `{"status": "created", ...}`, `{"status": "deleted",
  ...}`, partial-result shapes for failures). Their own coherent
  design pass.

## Coverage Report

`docs/coverage.md` records what happened to every old `deriva-mcp` tool. Update it
incrementally as each per-concept analysis pass completes (during phases 2-5).
Reviewed at every phase boundary — see Definition of Done in the design spec.

## Development Workflow

Use feature branches in the main checkout — **not** git worktrees:

```bash
git checkout -b feature/phase-N-<short-name>
uv sync --extra dev
# ... work, commit ...
git checkout main
git merge --no-ff feature/phase-N-<short-name>
git branch -d feature/phase-N-<short-name>
```

Why not worktrees? `[tool.uv.sources]` declares `path = "../deriva-mcp-core"`
and `path = "../deriva-ml"`, which is correct from the main repo location
but resolves to non-existent dirs from any worktree under `worktrees/<branch>/`.
We tried worktrees during Phase 0/1 and the temp-edit-paths-then-revert dance
plus uv.lock churn at every merge made it more friction than it was worth.

`worktrees/` remains in `.gitignore` so a one-off `git worktree add` (e.g. for
parallel review of a different branch) still won't leak into commits.

## Development Gotchas

- **`src/deriva_ml_mcp/_version.py` is auto-generated by hatch-vcs.** It's
  gitignored (`.gitignore: src/*/_version.py`) and excluded from ruff via
  `extend-exclude` in `[tool.ruff]`. Don't commit it; don't try to format it.
  If `ruff format --check` ever flags it, the exclude has been broken.

- **Annotations: prefer unquoted under `from __future__ import annotations`.**
  House style is `from __future__ import annotations` everywhere; under that
  PEP 563 opt-in, all annotations are lazy strings at runtime, so quoting is
  redundant and ruff `UP037` flags it. The canonical form is
  `def register(ctx: PluginContext) -> None:` with the `PluginContext`
  import guarded by `if TYPE_CHECKING:`. Don't "fix" the unquoted form back
  to quoted strings.

## Spec & Plan

- Design: [`docs/superpowers/specs/2026-04-24-deriva-ml-mcp-design.md`](docs/superpowers/specs/2026-04-24-deriva-ml-mcp-design.md)
- Plan: [`docs/superpowers/plans/2026-04-24-deriva-ml-mcp.md`](docs/superpowers/plans/2026-04-24-deriva-ml-mcp.md)
