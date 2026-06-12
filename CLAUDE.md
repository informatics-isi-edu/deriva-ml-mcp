# CLAUDE.md

This file provides guidance to Claude Code when working with the deriva-ml-mcp-plugin codebase.

See [`../CLAUDE.md`](../CLAUDE.md) for shared workspace conventions: `uv` for everything,
Google-style docstrings with examples, no backwards-compat shims, no over-engineering,
shared `bump-version` rules, and standard `pytest` / `ruff` invocations.

## Project Overview

`deriva-ml-mcp-plugin` is a plugin for [`deriva-mcp-core`](../deriva-mcp-core) that exposes
DerivaML domain workflows (datasets, features, workflows, executions) as MCP tools
and resources. The plugin is discovered via the `deriva_mcp.plugins` entry-point
group at server startup.

## Architecture

Current shape (v1.1.x):

```
src/deriva_ml_mcp_plugin/
├── plugin.py            # register(ctx) entry point -- dispatches to module registrars
├── ml_context.py        # The single helper that builds a DerivaML from core's credential
├── _helpers.py          # _error_envelope, _paginate, _read_rid, _table_qname,
│                        # _table_to_dict, _MAX_LIMIT (shared)
├── prompts.py           # 3 MCP prompts (deriva_ml_concepts /
│                        #   _getting_started / _primer) + the
│                        #   deriva_ml_primer + deriva_ml_get_guide orientation tools
├── tools/               # Per-domain tool modules (50 tools total)
│   ├── dataset/         #   20 tools, split into focused submodules:
│   │   ├── __init__.py  #     register aggregator + helper re-exports
│   │   ├── read.py      #     12 read tools + shared helpers
│   │   ├── mutate.py    #     7 mutation tools
│   │   └── complex.py   #     1 complex tool (denormalize)
│   ├── feature.py       #    6 tools
│   ├── workflow.py      #    5 tools
│   ├── execution/       #    8 read-only tools (lifecycle is local Python)
│   ├── asset.py         #    5 tools
│   ├── vocabulary.py    #    1 tool (deriva_ml_create_vocabulary)
│   ├── maintenance.py   #    2 tools (reindex_vocabularies, reindex_rows)
│   └── resolve.py       #    1 tool (deriva_ml_describe_rid -- RID -> kind + routing)
└── resources/           # MCP resources + per-user RAG
    ├── ml.py            #   19 resources (16 catalog-scoped under
    │                    #   deriva://catalog/{h}/{c}/deriva-ml/... + 3 static)
    └── rag.py           #   1 GitHub doc source +
                         #   read-through per-user-per-RID writers (Dataset/Workflow/Execution,
                         #   warmed on list/get/find + surgically on mutate; no on-connect pass)
                         # + 1 catalog-public on_catalog_connect hook (vocabularies)
```

History: Phase 0 shipped `plugin.py` and the empty `tools/`/`resources/`
package markers. `ml_context.py` arrived in Phase 1; the domain tool
modules in Phases 2-5; `resources/ml.py` and `resources/rag.py` in
Phase 6; v1.0 polish added `prompts.py` + the `deriva_ml_*` tool name
prefix; v1.1 added vocabulary RAG indexing + `tools/vocabulary.py` (renamed to
`tools/maintenance.py` in v1.4 when `deriva_ml_reindex_rows` joined it) +
the discovery-pattern prompt section; v1.x split `tools/dataset.py`
into a focused-submodule package.

**Per-user RAG safety.** `resources/rag.py` does NOT use
`ctx.rag_dataset_indexer(...)` -- that API produces a single global
enriched source shared across users (`enriched:{host}:{cat}:{schema}:{table}`),
which leaks data for any catalog table where rows have user-specific
ACLs. Instead, the plugin writes these rows via direct
`store.delete_source + store.add` calls (from the read-through and
mutate paths, not an on-connect hook) so each user's chunks land under
per-RID source names of the shape
`data:{host}:{cat}:{user_id}:{table}:{rid}` (v1.3) and ACL is applied
at fetch time using the calling user's credential. The `data:` prefix
keeps upstream's `rag_search` user-id filter in play (it accepts both
the legacy bulk form `data:{host}:{cat}:{user_id}` and prefix-match
on the per-RID form). The plugin-level test
`test_register_does_not_use_rag_dataset_indexer` pins the unsafe-API
ban; `test_data_sources_use_per_rid_naming` pins the v1.3 source-name
shape; future commits accidentally calling the unsafe API or
regressing the source-name shape will fail CI.

**Read-through (index-on-find) indexing (v1.5).** Dataset / Workflow /
Execution rows are NO LONGER bulk-indexed on connect (the three
first-connect passes were removed in v1.5 because they re-fetched and
re-embedded every visible row on every catalog connect). A row's chunk
is warmed lazily instead: the list/get/find tools call
`_index_rows_on_find(...)` after a read to schedule a best-effort
re-index for each row just seen (in the order the read returned them --
there is NO newest-first sort), and each mutating tool calls
`_reindex_<entity>` surgically right after its catalog mutation
succeeds. Both paths write per-RID sources of the shape
`data:{host}:{cat}:{user_id}:{table}:{rid}`. The re-index is
best-effort: any failure is logged but does NOT propagate to the
tool's success path (the catalog mutation already succeeded). The
audit event always fires before the re-index call so audit captures
the success regardless of cache state. Result: a row becomes
`rag_search`-able once it's been listed/fetched (which warms it) or
the moment the calling user creates/mutates it.

**Cross-user freshness gap (v1.4 — known limitation).** The
read-through warm and surgical re-index cover only the rows the calling
user has read or mutated under their OWN credential. They do NOT
propagate across users: when user A mutates a dataset visible to user
B, B's per-user sources stay stale until B next lists/fetches that row
(or resyncs). Same gap applies to mutations from non-MCP clients
(Chaise UI, ERMrest direct). The manual bridge is
`deriva_ml_reindex_rows(hostname, catalog_id, target=None)` in
`tools/maintenance.py` -- the "warm everything for this catalog now"
button, re-running `_reindex_*` over every visible row for the calling
user (target=None) or one (target="dataset:1-AAAA"). The
`deriva_ml_getting_started` prompt documents the verify-with-`get_*`
recovery pattern for shared-visible rows. Automatic cross-user fan-out
is deferred per the design discussion in #9 -- the load profile is
deployment-specific and no demand signal yet.

Vocabularies are the documented exception: vocab content is catalog-
public (no per-user ACL), so the plugin writes vocab terms directly
to the vector store under a custom `vocab:{host}:{cat}:{schema}.{table}`
prefix that bypasses upstream's `data:` user-id filter and serves
chunks to all users in the catalog. See `resources/rag.py`'s module
docstring for the full rationale.

ML-kind doc_types (issue #7, resolved plugin-side): because the
plugin writes chunks directly to the store, it tags them
`ml-dataset` / `ml-workflow` / `ml-execution` / `ml-vocab` so
`rag_search(doc_type=...)` filters by kind. deriva-mcp-core#2
(`index_table_data` doc_type parameter) no longer blocks anything
here -- it remains open for plugins that use that path. Rows
indexed before the tags shipped carry the legacy `catalog-data`
doc_type until re-indexed (read-through refresh / reindex tools).

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
- **Every synchronous deriva-ml or deriva-py call inside an `async def` tool
  MUST be wrapped in `await asyncio.to_thread(...)`.** No exceptions. The
  plugin's tool functions are coroutines invoked by an asyncio-based MCP host;
  any sync call that takes more than a few hundred milliseconds blocks the
  event loop and starves the host's permission/heartbeat machinery. Symptom:
  intermittent silent tool rejections in clients (the host's permission stream
  closes mid-flight; pending tool calls reject with "user doesn't want to
  proceed" even though no user action occurred). See
  Development Gotchas → "Sync calls in async tools" for history.

## Naming conventions (provenance: which plugin does a name come from?)

Multiple plugins co-load on one `deriva-mcp-core` server (this plugin,
the eye-ai plugin, future domain plugins). A client/LLM must be able to
tell, from a name alone, which plugin exposed it. The convention that
makes that work, confirmed by the 2026-05-30 naming audit:

**Tools — `deriva_ml_<verb>[_<noun>]`.** Every tool this plugin
registers carries the `deriva_ml_` prefix (e.g. `deriva_ml_list_datasets`,
`deriva_ml_create_workflow`). This distinguishes them from:
- **core's built-in tools**, which are bare verbs (`get_entities`,
  `add_term`, `create_table`, `query_attribute`) plus the
  self-namespaced `rag_*` family; and
- **sibling plugins**, which use their own `deriva_<domain>_` prefix
  (the eye-ai plugin uses `deriva_eye_ai_*`).

The general rule for any deriva-mcp-core domain plugin is
`deriva_<domain>_<verb>`: a shared `deriva_` root + a domain token. New
tools MUST follow it. There are zero unprefixed tools in this plugin and
that is load-bearing — `tests/test_plugin.py`'s exact-equality frozensets
pin it.

**Prompts — `deriva_ml_<name>`.** `deriva_ml_concepts`,
`deriva_ml_getting_started`. Same prefix rule as tools. (Core's prompts
use the `*_guide` suffix: `query_guide`, `entity_guide`, etc.)

**Audit events — `deriva_ml_<op>` / `deriva_ml_<op>_failed`.** See Tool
Implementation Rules above. Core's own events use `<module>_<op>`
(`catalog_create`, `vocabulary_add_term`) with no top-level prefix, so
the `deriva_ml_` prefix keeps this plugin's events distinguishable in
the audit log.

**Resource URIs — `deriva://catalog/{h}/{c}/deriva-ml/...`.** The
catalog-scoped resource family uses a `/deriva-ml/` path segment (NOT a
terse `/ml/`). This was renamed from `/ml/` in the 2026-05-30 audit
cleanup precisely for provenance clarity: the URI root
`deriva://catalog/{h}/{c}/...` is SHARED with core (core ships
`deriva://catalog/{h}/{c}/schema|tables`), so the plugin-naming segment
is the only origin marker — it must name the plugin, not read as
generic. The two static cold-start resources use
`deriva://deriva-ml/concepts` and `deriva://deriva-ml/getting-started`
(authority segment names the plugin). New resources MUST use a
`deriva-ml`-naming segment; never reintroduce a bare `/ml/`.

**RAG source-name prefixes.** Per-user catalog rows use core's
`data:{host}:{cat}:{user_id}:...` prefix (gated by core's user-id
filter). Catalog-public content uses a plugin-custom prefix that bypasses
that filter: this plugin uses `vocab:` for vocabularies; the eye-ai
plugin uses `eye-ai:`. These bypass prefixes are independent carve-outs
from core's filter (which only gates `schema:` / `data:` / `enriched:`).
A new public-bypass prefix must be unique across co-loaded plugins;
embed `{host}:{cat}` in the source name so same-prefix content from
different catalogs never collides. GitHub doc sources are named
`deriva-ml-docs` / `deriva-ml-mcp-docs` (the `-docs` suffix + `deriva-ml`
token make them self-identifying).

**Entry-point name == package name.** The `deriva_mcp.plugins`
entry-point name is `deriva-ml-mcp-plugin` (matches the PyPI/package name), so
the deriva-docker `DERIVA_MCP_PLUGIN_ALLOWLIST` value works without the
name-vs-package confusion. (See Development Gotchas for the war story.)

## Stateless / bounded-resource rule for MCP operations

**MCP operations must be stateless on the server side and consume bounded
resources per call.** Operations that require significant data localization,
server-side persistent storage, or managed state (per-user SQLite registries,
``~/.deriva-ml/`` caches, local git checkout dependencies, materializing
unbounded data per call) must NOT be exposed as MCP tools or resources. They
belong in skill-driven Python that the user runs locally.

**Rationale.** The MCP server runs in a Docker container with no per-user
persistent workspace and may serve many clients simultaneously. Any operation
that needs the caller's local filesystem doesn't work in this environment;
any operation that materializes unbounded data per call breaks the
bounded-resource contract. Tools that violate the rule appear to "work" in
single-machine development (where the MCP server happens to be the user's
machine) but break the moment the deployment is multi-tenant.

**Concrete guidance for new tools.**

1. **No reads from `ml.workspace.*` or `_manifest_store`** — these are the
   per-user SQLite registries; what they hold is bound to whichever process
   created it, not the calling client.
2. **No writes to local disk** — no `Path.write_bytes()`, no
   `bag.materialize()`, no `download_*()`. File I/O belongs in a skill
   like ``work-with-assets`` that generates Python the user runs locally.
3. **No git introspection** — no `_get_python_script()`, no
   `Workflow._github_url()` (that path is the bug fixed in commit
   ``976875e``). The caller computes the Git URL, checksum, and version
   locally and passes them in.
4. **No `file_path=` parameters that the server reads from disk.** Take
   ``file_contents=`` (a string) instead, write to a temp file the server
   manages, and clean up in ``finally``.
5. **Bounded responses.** Every potentially-unbounded read MUST cap at
   ``_MAX_LIMIT`` (1000 rows by default) and surface ``truncated`` so the
   client knows to switch to the cursor-paginated tool form. The
   PAGINATION CONTRACT in ``deriva_ml_getting_started`` is the wire
   guarantee that backs this.
6. **No state surfaced from the server's own cache** — fields like
   ``cache_status`` / ``cache_path`` describe the server's local cache,
   not the user's. They mislead a client that expects to see its own
   state and should be stripped from any response surfaced via MCP.

**Good models.** ``deriva_ml_list_*`` (cursor-paginated, ``_MAX_LIMIT``
bounded, no filesystem); ``deriva_ml_validate_dataset_specs`` (pure
metadata round-trips); ``deriva_ml_list_feature_values`` (caps via
``max_results`` + propagates ``DerivaMLMaterializeLimitExceeded``);
the ``Asset`` module's explicit carve-out at ``tools/asset.py:14-20``
("file I/O lives in a skill, not the MCP surface").

**Known violations** are catalogued in
``docs/audit-2026-05-23.md`` § "Stateless / bounded-resource rule"
along with the design notes for moving each to a skill. The biggest
cluster is the staged-execution lifecycle
(``add_feature_values`` + ``commit_execution``) which inherits
per-server SQLite manifest state from deriva-ml's
``Execution._manifest_store``; deprecating that pair and moving it
to a ``work-with-executions`` skill is the load-bearing fix.

## Out of Scope: Local Storage Management

The deriva-ml 1.46 cache/storage APIs (`list_cached_bags`,
`list_cached_assets`, `delete_cached_bag`, `delete_cached_asset`,
`clear_cache`, `clean_execution_dirs`, `get_storage_summary`) are
**deliberately not exposed** as MCP tools or resources. They operate
on the filesystem of the machine where the deriva-ml Python library
runs — the user's machine — and this plugin's server does not share
that filesystem. Storage management is the `manage-deriva-storage`
skill's job (it runs Python locally). Do not add `storage/*`
resources or cache-cleanup tools here; see
[`docs/adr/0001-local-storage-management-out-of-mcp-scope.md`](docs/adr/0001-local-storage-management-out-of-mcp-scope.md)
for the full decision, including the recorded tension with the
pre-existing `deriva_ml_cache_dataset` / `bag_info` cache fields.

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

**Read tools have `_impl` helpers; mutate tools don't.** This asymmetry is
intentional. Mutate tools emit audit events inline and handle the
`deriva_call` context per call; resources are read-only by MCP convention,
so the mutate→resource path doesn't exist. If you find yourself extracting
a helper from a mutate tool, either it's a candidate for splitting
(separate that read piece into its own tool) or you've found a
mutate-tool→mutate-tool sharing case that the convention should grow to
cover. As of v3.3, every mutate tool keeps its body inline; 13 read tools
have `_impl` helpers. The pattern is well-followed; new tools should
follow existing files as templates.

### Response models (Pydantic, v1.6+)

Every list / detail / summary response shape is declared as a
`pydantic.BaseModel` in `src/deriva_ml_mcp_plugin/_response_models.py`. The shared
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

v2.3 retired the four ad-hoc payload sites listed in the v2.2 note.
Each wrapper now constructs a named Pydantic model and serializes
via `.model_dump_json(by_alias=True)`:

- `deriva_ml_find_workflow_executions` reuses `ExecutionListResponse`
  (its response is shape-identical to `_list_executions_impl`'s,
  just filtered by workflow).
- `deriva_ml_list_execution_children` returns `ExecutionChildrenResponse`
  (`{parent_rid, recurse, count, children: list[ExecutionSummary]}`).
- `deriva_ml_list_execution_parents` returns `ExecutionParentsResponse`
  (`{child_rid, recurse, count, parents: list[ExecutionSummary]}`).
- `deriva_ml_list_dataset_relations` returns `DatasetRelationsResponse`
  with all-optional fields (`parents`, `parents_truncated`,
  `children`, `children_truncated`, `warning`). The wrapper passes
  `exclude_none=True` to `model_dump_json` so the v1.x wire shape
  is preserved -- when `direction="parents"` the response has no
  `children`/`children_truncated` keys at all (not present as
  `null`). The conditional-key contract is documented on the model
  docstring.

v3.0 swept the mutating-tool response shapes. Every `mutates=True`
tool now returns a named Pydantic response model with a
`Literal[...]` `status` discriminator and `extra="forbid"`. The
discriminator vocabulary was normalized -- the generic `"success"`
status was retired in favor of operation-specific verbs (`created`,
`added`, `removed`, `incremented`, `cached`, `split`). Idempotent
no-ops on `start_execution` / `abort_execution` are now signaled
via dedicated status values (`already_running` / `already_aborted`)
instead of an optional `note` field. `update_dataset`'s
conditional-key fields (`dataset_types`, `added`, `removed`) are
now always present (`null` when the description-only branch ran).
`cache_dataset` nests upstream bag-info keys under a `bag_info`
field instead of spreading them top-level.

v3.0 wire-break migration map for clients:

- `add_dataset_members`: `status="success"` -> `status="added"`.
- `delete_dataset_members`: `status="success"` -> `status="removed"`.
- `add_dataset_element_type`: `status="success"` -> `status="created"`.
- `increment_dataset_version`: `status="success"` -> `status="incremented"`.
- `cache_dataset`: `status="success"` -> `status="cached"`; bag-info
  keys moved from top-level to `payload["bag_info"][<key>]`.
- `split_dataset`: `status="success"` -> `status="split"`.
- `start_execution` no-op: `{"status": "running", "note": "already running"}`
  -> `{"status": "already_running"}`.
- `abort_execution` no-op: `{"status": "aborted", "note": "already aborted"}`
  -> `{"status": "already_aborted", "reason": null}`.
- `update_dataset` description-only edit: `dataset_types` / `added` /
  `removed` keys are present as `null` instead of being omitted.
- `abort_execution`: `reason` is always present (`null` when no
  reason was given) instead of being omitted on the no-op branch.
- `create_execution_dataset`: `dataset_types` is always present
  (`null` when omitted by the caller) instead of being omitted.

This closes PR 3 of #6. The response-shape Pydantic migration is
complete -- every `_*_impl` helper, every list/detail/summary
shape, and every mutating-tool response is now Pydantic-typed.

v3.1 exposes three new optional parameters from the deriva-ml v1.31.0
release through the MCP tool surface:

- `deriva_ml_list_executions`, `deriva_ml_list_datasets`,
  `deriva_ml_list_workflows` (and `deriva_ml_find_workflow_executions`
  which shares the executions impl) accept an optional `sort: bool =
  False`. Default `False` preserves the current RID-ascending order
  used for stable cursor pagination. `True` returns results
  newest-first by record creation time (RCT desc) -- recommended for
  "show me what's recent" queries. The `_list_*_impl` helpers were
  extended in lockstep so the resource layer (which calls them
  without `sort=`) keeps its current snapshot stability.

- `deriva_ml_list_feature_values` accepts:
  - `execution_rids: list[str] | None = None` -- server-side filter
    to a known set of execution RIDs. Empty list short-circuits.
    Recommended for cross-execution comparison queries (e.g. "give
    me the F1 score for each of these 5 runs in one round-trip").
  - `max_results: int = 50_000` -- caller-controlled cap on rows
    materialized before pagination. The MCP wrapper translates
    deriva-ml's `DerivaMLMaterializeLimitExceeded` into a clear
    error envelope. Behavior tightening: queries that previously
    OOMed silently now return `{"error": "...exceeds max_results..."}`.
    Note the deliberate naming asymmetry -- the MCP wire uses
    `max_results` (LLM-friendly name) and forwards as
    `materialize_limit=max_results` to deriva-ml (library-internal
    name).

- `deriva-ml` is now pinned to `>=1.31.0` (was unpinned;
  audit-flagged deployment risk).

Resource shapes are unchanged. Resources keep RID-ascending defaults
for snapshot stability; users who want recent-first content use the
tool with `sort=True`.

This release does NOT add a `compare_metrics` tool. The
cross-execution comparison workflow is taught at the skill layer
(`compare-model-runs` skill in `deriva-ml-skills`, follow-up) which
respects both metric-storage patterns: features-as-scalars (use
`execution_rids=` on `list_feature_values`) and metrics-as-JSONL-
asset files (download via `work-with-assets`, parse locally).

## Cross-Repo Sync: `deriva_ml_concepts` prompt ↔ `deriva-ml-context` skill

The `_CONCEPTS_GUIDE` constant in `src/deriva_ml_mcp_plugin/prompts.py`
(rendered as the `deriva_ml_concepts` MCP prompt) and the
`skills/deriva-ml-context/SKILL.md` file in the companion
`deriva-ml-skills` Claude Code plugin share their conceptual content
DELIBERATELY. Both must explain:

- What DerivaML is (one paragraph)
- The five core abstractions (Dataset, Workflow, Execution, Feature, Asset)
- The provenance principle (every artifact links to its producing Execution)
- The vocabulary-extension pattern (use core's `add_term` with `schema="deriva-ml"`)

The duplication is intentional. The two surfaces serve different LLM
clients with different invocation models:

- **Claude Code clients** with the `deriva-ml-skills` plugin loaded
  get the conceptual frame pushed into context proactively via the
  always-on `deriva-ml-context` skill (the audit-named "load-bearing"
  path).
- **Non-Claude-Code clients** (Cursor, SDK-based agents, raw FastMCP
  clients, etc.) pull the same frame in via the `deriva_ml_concepts`
  prompt over the MCP wire.

The skill is RICHER than the prompt — it adds tool-selection guidance,
cross-references to other skills, the worked "when to reach back to
the raw catalog surface" table, and other Claude-Code-specific
value-add. The prompt is the conceptual FLOOR; the skill is floor +
Claude-Code value-add.

**When updating the abstractions** (rare — they're fundamental),
update BOTH:

1. `_CONCEPTS_GUIDE` in `src/deriva_ml_mcp_plugin/prompts.py` (this repo)
2. `skills/deriva-ml-context/SKILL.md` in `../deriva-ml-skills`

Both files carry an inline comment block at their top pointing at the
other side. The matching comment lives in
`../deriva-ml-skills/CLAUDE.md` under the same section heading so the
constraint is visible from either repo.

### v3.x update: prompts went from 4 to 2

The plugin originally shipped four prompts; v3.x removed two of them
when the [Round 2 audit cleanup](https://github.com/informatics-isi-edu/deriva-ml-skills/blob/main/docs/superpowers/plans/2026-05-02-tier-2-audit-cleanup-plan-round-2-refinement.md)
identified them as architecturally mis-shaped per FastMCP guidance
(prompts should be user-controlled parameterized templates, not
static reference documents):

- `deriva_ml_workflow_dedup` → content moved to the
  `deriva_ml_create_workflow` and `deriva_ml_find_workflow_by_url`
  tool docstrings. Per-tool LLM-trap warnings belong on the trap.
- `deriva_ml_execution_lifecycle` → per-tool warnings moved to the
  four lifecycle tools' docstrings (`deriva_ml_start_execution`,
  `deriva_ml_commit_execution`, `deriva_ml_abort_execution`,
  `deriva_ml_add_feature_values`); cross-cutting state-machine
  depth covered by the RAG-indexed `user-guide/executions.md` doc
  in the deriva-ml repo.

The remaining two prompts (`_CONCEPTS_GUIDE`, `_GETTING_STARTED_GUIDE`)
serve cold-start orientation for non-Claude-Code clients and have no
clean alternative without a deriva-mcp-core API addition for plugin-
contributed server `instructions=` content. The architectural target
is to migrate them to that field once the API exists; the ask has
been raised with the deriva-mcp-core maintainer (hook is being added
upstream; no tracking issue number yet). Until then the two
remaining prompts are the right home for cold-start orientation.

### Workaround: cold-start frame inlined in `deriva_ml_list_datasets`

The two cold-start prompts (`deriva_ml_concepts`,
`deriva_ml_getting_started`) carry the conceptual frame (five
abstractions, provenance principle, vocabulary-extension pattern,
pagination contract). The 2026-05-13 e2e platform test surfaced a
real symptom of clients ignoring these prompts: the journal author
saw the deriva-mcp-core server-level `instructions=` string (which
names the four core prompts), did NOT fetch any prompt, and only
learned the pagination contract by trial-and-error.

Root cause: deriva-mcp-core's `instructions=` is closed to plugins
(see `server.py:237-243` in deriva-mcp-core), so this plugin's two
prompt names are never advertised at server-init. A client that
ignores the named-prompt hint never discovers the unnamed ones.

Until the upstream `instructions=` hook lands, the entire cold-start
frame is **inlined verbatim into `deriva_ml_list_datasets`'s
docstring** (`tools/dataset/read.py`). That tool was chosen because
"what's in this catalog?" is the canonical cold-start opening move
for a DerivaML LLM client -- any first-time agent will fetch its
schema. The inlined block is fenced with `[Cold-start orientation
-- WORKAROUND ...]` / `[End cold-start block.]` markers so removal
is mechanical when the upstream hook arrives.

**When the deriva-mcp-core hook lands, do all three:**

1. Delete the bracketed cold-start block from
   `deriva_ml_list_datasets` docstring (and any other tool that
   grew one in the meantime).
2. Register the conceptual frame at server-init time via the new
   `instructions=` extension hook, in `plugin.py:register(ctx)`.
3. Update this section of CLAUDE.md to reflect the migration; the
   inline workaround is no longer the architectural truth.

The full prompt text stays in `prompts.py` -- prompts are the
authoritative source for any deeper detail clients can fetch on
demand, both before and after the hook lands. The inline frame is
only the load-bearing FACTS that a client must see *without
fetching anything*.

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

### Versioning convention (post-2026-05-23 consolidation)

The plugin is pre-release. Tags follow **`v0.x.y` semver** anchored
at `v0.5.0` (the post-audit-cycle consolidation point — May 2026).
Patch (`v0.x.y+1`) for fixes / doc updates / behavior-preserving
refactors; minor (`v0.x+1.0`) for new tools / new resources / wire
shape additions; major (`v0.x.0` → `v1.0.0`) is reserved for the
eventual "we have downstream users" stability promise.

Pre-`v0.5.0` history: there used to be `v0.1.0`, `v1.0.0`...`v3.4.1`,
`v4.0.0`...`v5.0.1` tags. All deleted in the 2026-05-23 cleanup
because the plugin had no downstream consumers and the
multi-major-version history misrepresented the project's actual
maturity. Older doc references to those tags (in `coverage.md`,
the audit doc, prompt-content version-history notes, etc.) are
preserved as a **narrative chronology** — they read as "this
change landed in development phase X" rather than as live tag
pointers. Recover via commit hashes if you need the historical
diff; `git log --oneline` works the same.

`bump-version` defaults to deriving the next tag from the highest
existing tag, so a `uv run bump-version patch` from current state
lands at `v0.5.1` cleanly. No manual `[tool.bumpversion]
current_version` editing needed.

The CHANGELOG (when we have one — not yet) and any future
deprecation notes should reference `v0.5.0` as the architectural
baseline rather than the deleted pre-consolidation tags.

## Running Under Docker (deriva-docker)

The plugin is loaded into the `deriva-mcp-test` service of a `deriva-docker`
deployment via the `DERIVA_MCP_EXTRA_PACKAGES` env var (set in the env file,
typically `~/.deriva-docker/env/localhost.env`). The package list must
include the **deriva-py `deriva-ml` branch** explicitly — `pip` inside the
container does not honor this repo's `[tool.uv] override-dependencies`, so
without that explicit pin pip resolves the conflict between
`deriva-mcp-core`'s `@master` pin and `deriva-ml`'s `@deriva-ml` pin and
fails. The README's `DERIVA_MCP_EXTRA_PACKAGES` value documents the
correct three-package incantation.

To pick up new versions of any of those packages (or any code change in
this repo when installed `@main`), rebuild and restart the service from
inside a `deriva-docker` checkout (where `docker-compose.yml` lives):

```bash
docker-compose --env-file ~/.deriva-docker/env/localhost.env down deriva-mcp-test
docker-compose --env-file ~/.deriva-docker/env/localhost.env build deriva-mcp-test --no-cache
docker-compose --env-file ~/.deriva-docker/env/localhost.env up -d deriva-mcp-test
```

`--no-cache` is required — without it, pip's resolver reuses the cached
wheel layer and your version bump silently doesn't land.

`scripts/rebuild-deriva-docker-mcp.sh` in this repo wraps those three
commands. Its first argument selects an MCP-server **target**:
`localhost` (deriva-ml plugin only, auths against the in-container
keycloak → `~/.deriva-docker/env/localhost.env`) or `eye-ai` (eye-ai +
deriva-ml plugins, auths against `dev.eye-ai.org` →
`~/.deriva-docker/env/eye-ai.env`). Default is `localhost`. Both
targets rebuild the same shared `deriva-mcp-test` service and differ
only in the MCP package set + auth host. A first argument that looks
like a path (contains `/` or ends in `.env`) is still honored verbatim.
The README has the full README context (env-var format, pre-release status
note). When `deriva-docker` ships final support for this plugin, the
helper script and this section will be replaced by the canonical
deriva-docker workflow.

## Development Gotchas

- **`src/deriva_ml_mcp_plugin/_version.py` is auto-generated by hatch-vcs.** It's
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

- **Sync calls in async tools — wrap with `asyncio.to_thread`.** This is a
  load-bearing rule (also stated under Tool Implementation Rules), but the
  history is worth recording so the same mistake isn't repeated.

  Tool functions are `async def` because the MCP host (deriva-mcp-core) is
  asyncio-based. The deriva-ml and deriva-py libraries are synchronous —
  `ml.find_datasets()`, `ds.list_dataset_members()`, `ml.lookup_dataset()`,
  etc. all block. Calling these directly inside an async coroutine blocks the
  event loop for the duration of the catalog call (often seconds; sometimes
  minutes for `cache_dataset` or large queries).

  While the loop is blocked, the host cannot service its other coroutines —
  including the **permission stream** that delivers user approve/deny
  decisions for tool calls. If that stream's heartbeat or response window
  expires while the loop is stalled, the host closes the stream and rejects
  every pending tool call with "Tool permission stream closed before response
  received" — surfaced to the model as "The user doesn't want to proceed
  with this tool use." The rejection is silent (no UI prompt), looks
  identical to user cancellation, and is intermittent (depends on whether
  the loop block crosses the threshold).

  **Pre-PR-#28 history:** the original implementation (Phases 2–5, v1.0)
  shipped every tool calling deriva-ml directly inside the async def. The
  symptom didn't surface during initial development because dev tests used
  small catalogs where calls returned in milliseconds. It surfaced under
  realistic catalog load — long-running tools (`cache_dataset`, large
  `find_datasets`) blocked the loop long enough for hosts to drop pending
  calls.

  **PR #28 (`fix(complex): wrap sync deriva-ml calls in asyncio.to_thread`)**
  fixed this in `tools/dataset/complex.py` for the three known-slow tools
  (`cache_dataset`, `denormalize_dataset`, `split_dataset`). Subsequent
  passes extended the wrapping to every remaining tool module
  (`asset.py`, `feature.py`, `maintenance.py`, `workflow.py`,
  `dataset/mutate.py`, `dataset/read.py`, `execution/read.py`, and
  `resources/ml.py`). As of the P2 hygiene sweep (2026-05-24), the AST
  test in `tests/test_async_thread_wrap.py` confirms **all modules are
  clean** — no unwrapped sync deriva-ml call exists inside any
  `with deriva_call():` block in any checked module.

  **The lesson — for plugin design.** Async-defined tools that call sync
  libraries are a footgun. The rule must be applied uniformly the first
  time, not retrofitted module-by-module. When adding a new tool, the
  template should already include `asyncio.to_thread`; when reviewing a
  PR that adds a tool, the absence of `asyncio.to_thread` around any
  sync deriva-ml/deriva-py call is a blocking review comment.
  `tests/test_async_thread_wrap.py` enforces the rule structurally so
  CI catches regressions automatically.

## Spec & Plan

- Design: [`docs/superpowers/specs/2026-04-24-deriva-ml-mcp-design.md`](docs/superpowers/specs/2026-04-24-deriva-ml-mcp-design.md)
- Plan: [`docs/superpowers/plans/2026-04-24-deriva-ml-mcp.md`](docs/superpowers/plans/2026-04-24-deriva-ml-mcp.md)
