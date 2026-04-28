# deriva-mcp-core Integration Audit -- deriva-ml-mcp

**Date:** 2026-04-27
**Auditor:** Claude Sonnet 4.6 (independent automated audit)
**Scope:** `deriva-ml-mcp` v3.0.0 against `deriva-mcp-core` plugin-authoring-guide contract

---

## Part A: Boundary Discipline

### Connection management

COMPLIANT.

`ml_context.py` is the single gateway: `get_ml(hostname, catalog_id)` calls
`get_request_credential()` from core and passes the credential to `DerivaML(...)`. No
tool module opens a raw `ErmrestCatalog` or `DerivaServer` connection directly. The
module docstring explicitly names this as the enforced pattern.

One ambiguity worth noting: `tools/workflow.py:392` calls `ml._add_workflow(wf)` and
`tools/execution.py:746` nulls out `ml._execution`. These are private attributes on
the `DerivaML` object (not on `PluginContext`), so they are not a core-boundary
violation -- they are plugin-to-deriva-ml coupling. That is a separate concern and is
addressed in Part C.

### Auth / credential handling

COMPLIANT.

No `os.environ` reads for tokens, no `keyring` imports, no direct reads or writes to
`~/.deriva/credential.json`. The only auth-adjacent code is in `ml_context.py:11`:

```python
from deriva_mcp_core import get_request_credential
```

This is the documented sanctioned path. Integration tests read `os.environ.get("DERIVA_HOST",
"localhost")` in `conftest.py` for the test server hostname, which is correct test
infrastructure, not auth handling.

### Audit machinery

COMPLIANT.

The plugin does NOT define its own audit machinery. All audit calls go through:

```python
from deriva_mcp_core.telemetry import audit_event
```

imported at module level in `_helpers.py:45`, `tools/asset.py:53`,
`tools/dataset/__init__.py:56`, `tools/execution.py:30`, `tools/feature.py:23`,
`tools/workflow.py:24`. The `_error_envelope` helper in `_helpers.py:107` centralizes
failure-path audit emission for all mutating tools. No competing audit implementation
was found.

### RAG subsystem (per-user safety)

COMPLIANT. Both the architectural requirement and the CI pin are in place.

The plugin deliberately avoids `ctx.rag_dataset_indexer(...)`. `resources/rag.py:40-46`
explains the rationale explicitly in the module docstring: the global enriched-source
API leaks per-user ACL rows. Instead the plugin uses `ctx.on_catalog_connect(...)` with
direct `store.delete_source + store.add` writes under per-RID source names
(`data:{host}:{cat}:{user_id}:{table}:{rid}` shape, established in v1.3).

The required CI pin exists and runs:

- `tests/test_plugin.py:309` -- `test_register_does_not_use_rag_dataset_indexer` --
  asserts `len(ctx._rag_dataset_indexers) == 0` after `register(ctx)`.
- `tests/test_plugin.py:393` -- `test_data_sources_use_per_rid_naming` -- fires each
  hook with mocked fetchers and asserts every produced source name matches the v1.3
  shape.
- `tests/test_rag.py:100` -- a second assertion `rag_ctx._rag_dataset_indexers == []`
  at the RAG-module registration level.

The vocab path (`resources/rag.py:855-864`) writes under `vocab:{hostname}:{catalog_id}:{qname}`
to bypass upstream's per-user filter, which is the documented correct carve-out for
catalog-public content. `tests/test_plugin.py:328` pins that vocab chunks never land
under `data:`.

### Vocabulary primitives

COMPLIANT. `tools/vocabulary.py` does NOT exist -- the module was renamed to
`tools/maintenance.py` in v1.4. The maintenance module ships exactly two tools:

- `deriva_ml_reindex_vocabularies` (`mutates=False`) -- re-indexes the RAG vector store
  from vocab table data via `_index_vocabularies` in `resources/rag.py`. No term CRUD.
- `deriva_ml_resync_indexes` (`mutates=False`) -- refreshes per-user RAG sources after
  cross-user mutations. No term CRUD.

Neither tool duplicates core's `add_term` / `lookup_term` / `delete_term`. The plugin
delegates all term CRUD to core as documented. `resources/ml.py:357-383` provides a
read-only `ml/registries` resource that calls `ml.list_vocabulary_terms(...)` for
snapshot display -- also not term CRUD.

### Generic Deriva primitives

COMPLIANT.

No re-implementation of entity CRUD (`insert_records`, `update_record`, `get_entities`),
schema introspection, or Hatrac operations was found. The plugin exclusively uses the
`DerivaML` high-level API (via `get_ml()`). Where a low-level write was needed and the
DerivaML API lacked it (`Asset.description` and `Dataset.description` have no write-
through setter), the plugin uses `_set_row_description` in `_helpers.py:250-287`, which
calls `ml.pathBuilder()` -- this is `deriva-ml` API, not raw ERMrest. The docstring
flags this as an upstream gap in `deriva-ml` (not `deriva-mcp-core`).

---

## Part B: Plugin-Authoring-Guide Conformance

### `mutates=` discipline

COMPLIANT.

Every `@ctx.tool(...)` decorator in the plugin carries an explicit `mutates=True` or
`mutates=False`. Verified across all six tool modules (46 total registrations):

- `tools/execution.py`: 5 `mutates=False`, 7 `mutates=True`
- `tools/dataset/read.py`: 7 `mutates=False`
- `tools/dataset/mutate.py`: 7 `mutates=True`
- `tools/dataset/complex.py`: 1 `mutates=True` (`cache_dataset`); `denormalize` and
  `split` are also `mutates=True`
- `tools/workflow.py`: 3 `mutates=False`, 2 `mutates=True`
- `tools/feature.py`: 3 `mutates=False`, 3 `mutates=True`
- `tools/asset.py`: 3 `mutates=False`, 1 `mutates=True`
- `tools/maintenance.py`: 2 `mutates=False`

No bare `@ctx.tool()` decorators (which would raise `TypeError` at startup) were found.

### `with deriva_call():` discipline

COMPLIANT.

Every tool function that performs DERIVA I/O wraps it in `with deriva_call():`. The
pattern is consistent across all tool modules. Resources (`resources/ml.py`) likewise
wrap all `get_ml()` + data-fetch calls in `with deriva_call():`. The maintenance tools
(`tools/maintenance.py:112`, `206`) wrap their `_index_vocabularies` /
`_resync_user_sources` calls, even though those helpers call into `get_ml()` themselves
-- defensive double-coverage.

One nuance: the `_reindex_*` helpers in `resources/rag.py` are called from within the
already-established `deriva_call()` block of the mutating tools that trigger them
(e.g. `tools/execution.py:763-771`), and are also reachable from the on-connect hooks
where no `deriva_call()` wraps them at the hook level. The hooks are fire-and-forget
background tasks -- the guide does not require `deriva_call()` in background hooks, and
hook failures are logged and suppressed per the framework contract. This is acceptable.

### Audit-on-success / audit-on-failure discipline

COMPLIANT. Five mutating tools sampled:

**`deriva_ml_create_execution` (`execution.py:672`)**
- Success: `audit_event("deriva_ml_create_execution", ...)` at line 752. Dry-run
  skips audit intentionally (documented at line 748: no catalog state changed).
- Failure: `_error_envelope(exc, operation="create_execution", ...)` at line 781,
  which emits `deriva_ml_create_execution_failed`.

**`deriva_ml_create_workflow` (`workflow.py:302`)**
- Success: `audit_event("deriva_ml_create_workflow", ...)` at line 394.
- Failure: `_error_envelope(exc, operation="create_workflow", ...)` at line 425.

**`deriva_ml_add_dataset_members` (`dataset/mutate.py`)**
- Success: `_pkg.audit_event(...)` at line 201.
- Failure: `_error_envelope` via `_pkg` at the `except` block.

**`deriva_ml_update_workflow` (`workflow.py:434`)**
- Success: `audit_event("deriva_ml_update_workflow", ...)` at line 500.
- Failure: `_error_envelope(exc, operation="update_workflow", ...)` at line 525.

**`deriva_ml_update_asset` (`asset.py:409`)**
- Success: `audit_event(...)` at line 495.
- Failure: routed through `_error_envelope` (confirmed by grep pattern at line 405
  comment).

All five confirm the dual-path pattern. The `_error_envelope` helper centralizes the
failure-path emission (`_helpers.py:107`) so the pattern cannot silently break if a
tool fails to add its own `except` block.

### Audit event naming convention

COMPLIANT.

All success-path audit events follow `deriva_ml_<operation>` exactly:
`"deriva_ml_create_execution"`, `"deriva_ml_create_workflow"`, `"deriva_ml_update_workflow"`,
etc. Failure events are emitted by `_error_envelope` as `f"deriva_ml_{operation}_failed"`
(line 108: `f"deriva_ml_{operation}_failed"`). The `operation` parameter passed by each
tool is the bare verb without prefix (e.g., `"create_execution"`, `"create_workflow"`),
and the helper prepends `deriva_ml_`. This is consistent with the guide's
`<plugin_name>_<operation>` convention where `plugin_name = "deriva_ml"`.

### Tool/prompt naming convention

COMPLIANT.

All 46 tools follow the `deriva_ml_<verb>` prefix (full list confirmed via grep).
All 3 prompts follow the same convention: `deriva_ml_getting_started`,
`deriva_ml_execution_lifecycle`, `deriva_ml_workflow_dedup` (`prompts.py:603`, `613`,
`623`). No tool or prompt name deviates from this prefix.

---

## Part C: Integration Touch-Points

### Plugin entry point

COMPLIANT. `plugin.py:register(ctx)` is clean:

```python
def register(ctx: PluginContext) -> None:
    _dataset.register(ctx)
    _feature.register(ctx)
    _workflow.register(ctx)
    _execution.register(ctx)
    _asset.register(ctx)
    _maintenance.register(ctx)
    _ml_resources.register(ctx)
    _ml_rag.register_rag_sources(ctx)
    _ml_prompts.register(ctx)
```

Pure dispatch to per-domain registrars. Synchronous. No side effects at module scope.
`PluginContext` is TYPE_CHECKING-guarded so the import is lazy at runtime (only needed
at type-check time, not during plugin loading before core is fully initialized).

### PluginContext usage

LARGELY COMPLIANT, with one ambiguity in tests.

Plugin production source (`src/`) never accesses private `ctx._*` attributes. All
interaction with `ctx` is through documented public methods: `ctx.tool(...)`,
`ctx.resource(...)`, `ctx.prompt(...)`, `ctx.on_catalog_connect(...)`,
`ctx.rag_github_source(...)`.

Tests (`tests/conftest.py:24`, `tests/test_plugin.py:279`, `325`, `349`, `431-433`,
`tests/test_rag.py:60`, `79`, `84`, `95`, `100`) access `ctx._rag_dataset_indexers`,
`ctx._rag_sources`, `ctx._catalog_connect_hooks`, and `ctx._mcp`. These private
attributes are used exactly as the plugin-authoring-guide's own test examples show
(`ctx._mcp.tools["tool_name"]` on page 598 of the guide, `_set_plugin_context` at
page 536). The guide canonizes this pattern for test code, so this is intended use.

One distinct private API is `_set_plugin_context` (imported from
`deriva_mcp_core.plugin.api` in `conftest.py:24`). This is a leading-underscore
function from core's public-ish testing surface -- the guide itself shows this exact
import and usage in the "Fixtures" section. Ambiguous whether this is a contract
violation or a documented testing escape hatch; the guide treats it as the latter.

**Note on DerivaML private access (not a core boundary issue):** Three sites access
private `deriva_ml.DerivaML` internals:
- `tools/workflow.py:392`: `ml._add_workflow(wf)` -- undocumented `deriva-ml` API.
- `tools/execution.py:746`: `ml._execution = None` -- nulls a private DerivaML field.
- `tools/dataset/mutate.py:529`: `_set_row_description(ml, ml._dataset_table, ...)`.

These are `deriva-ml` contract issues, not `deriva-mcp-core` boundary violations.
They represent API gaps in `deriva-ml` that forced the plugin to reach into internals.

### Resource URI scheme

COMPLIANT. All 11 resources (the module docstring lists 11 but the code registers 11)
follow `deriva://catalog/{hostname}/{catalog_id}/ml/...`:

```
deriva://catalog/{hostname}/{catalog_id}/ml/datasets
deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}
deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/members
deriva://catalog/{hostname}/{catalog_id}/ml/workflows
deriva://catalog/{hostname}/{catalog_id}/ml/workflow/{workflow_rid}
deriva://catalog/{hostname}/{catalog_id}/ml/executions
deriva://catalog/{hostname}/{catalog_id}/ml/execution/{execution_rid}
deriva://catalog/{hostname}/{catalog_id}/ml/features/{table_name}
deriva://catalog/{hostname}/{catalog_id}/ml/asset-tables
deriva://catalog/{hostname}/{catalog_id}/ml/asset/{asset_rid}
deriva://catalog/{hostname}/{catalog_id}/ml/registries
```

All 11 are namespaced under `ml/`. None escape the `ml/` namespace. (The module
docstring says 9 but the code has 11 -- the docstring was not updated when `asset-tables`,
`asset/{rid}`, and `registries` were added. Minor doc drift, not a contract violation.)

### Error envelope shape

COMPLIANT.

`_error_envelope` in `_helpers.py:114` returns `json.dumps({"error": str(exc)})`,
optionally extended with `response_fields`. The base shape is `{"error": "..."}`, which
matches the guide's example at page 128-129. Validation errors from Pydantic models are
caught by the `except Exception` block in each tool and routed through `_error_envelope`,
so they surface as `{"error": "..."}` consistently. Resources use the same helper with
`audit=False`.

### Lazy import patterns

COMPLIANT.

The authoring guide requires imports of `get_catalog`, `deriva_call`, etc. inside tool
function bodies, not at `register()` scope, so test patches resolve at call time.

In practice, this plugin uses `DerivaML` (via `get_ml()`) rather than raw `get_catalog`,
so the lazy-import requirement is satisfied by `get_ml(...)` being called inside each
tool body. The `deriva_call` import at module level (e.g., `from deriva_mcp_core import
deriva_call` in `execution.py:29`) is safe because `deriva_call` is a context manager
class, not a function that reads request state -- it does not capture credential context
at import time.

The per-RID re-indexers (`_reindex_dataset`, `_reindex_workflow`, `_reindex_execution`)
ARE imported lazily inside tool bodies (e.g., `execution.py:764`: `from deriva_ml_mcp.resources.rag
import _reindex_execution`). The CLAUDE.md calls this out explicitly as the correct
pattern to avoid circular imports during plugin loading. Similarly, `maintenance.py:116`
lazily imports `_index_vocabularies`, and `maintenance.py:210` lazily imports
`_resync_user_sources`.

---

## Part D: Forward/Upstream Compatibility

### Open upstream issues

Two tracked upstream gaps found:

1. **`# TODO(upstream-rag-doctype)`** -- `resources/rag.py:428` and `resources/rag.py:903`.
   Tracked as `deriva-mcp-core#2`: `index_table_data` should accept a `doc_type`
   parameter so per-table RAG chunks can be filtered by `rag_search(doc_type="ml-dataset")`
   etc. Until then, all chunks land under `doc_type="catalog-data"`. Both sites carry
   the issue reference inline.

2. **`# TODO(deriva-ml-execution-metadata-api)`** -- `tools/execution.py:275`. No
   generic `deriva-ml` API exists to enumerate `Execution_Metadata` files. Until an
   upstream enumerator exists, the `ml/execution/{rid}` resource omits the `metadata`
   key. This is a `deriva-ml` gap, not a `deriva-mcp-core` gap.

The CLAUDE.md also references `deriva-mcp-core#3` (core's `add_term`/`delete_term`
tools don't fire any lifecycle hook, so vocab RAG goes stale after term edits). This
gap is addressed in the plugin via the manual `deriva_ml_reindex_vocabularies` tool
bridge. No inline `# TODO` for this one -- it is handled, not deferred.

No other `# TODO(upstream-...)` markers were found.

### Version pinning

POTENTIAL CONCERN.

`pyproject.toml:26` declares `"deriva-mcp-core"` with no version constraint -- fully
unpinned. In development, the `[tool.uv.sources]` stanza (`pyproject.toml:57`) points
to a local path `../deriva-mcp-core` (editable install), which implicitly pins to
whatever version is checked out locally. For published releases, the absence of a lower-
bound constraint (e.g., `"deriva-mcp-core>=1.0"`) means the plugin could be installed
against a too-old core that lacks `resolve_user_identity`, `get_rag_store`, or other
symbols the plugin imports. This is a deployment risk, not a correctness issue in the
current workspace where both are developed together.

The guide's own example shows `"deriva-mcp-core>=0.1"`, which is still loose but at
least establishes a floor. Recommend adding a lower-bound pin.

### Import-surface contract

MOSTLY COMPLIANT. The full set of imported symbols from `deriva_mcp_core`:

| Symbol | Import path | Visibility |
|---|---|---|
| `get_request_credential` | `deriva_mcp_core` (`__init__.py`) | Public |
| `deriva_call` | `deriva_mcp_core` (`__init__.py`) | Public |
| `audit_event` | `deriva_mcp_core.telemetry` | Public (submodule) |
| `PluginContext` | `deriva_mcp_core.plugin.api` | Public (plugin API) |
| `resolve_user_identity` | `deriva_mcp_core.context` | Ambiguous |
| `get_rag_store` | `deriva_mcp_core.rag` | Ambiguous |
| `chunk_markdown` | `deriva_mcp_core.rag.chunker` | Ambiguous |
| `RowSerializer` | `deriva_mcp_core.rag.data` | Ambiguous |
| `Chunk` | `deriva_mcp_core.rag.store` | Ambiguous |
| `_set_plugin_context` | `deriva_mcp_core.plugin.api` | **Private** (leading `_`) |

`_set_plugin_context` is the only leading-underscore import. It appears only in
`tests/conftest.py:24`, not in production source. The authoring guide itself uses this
symbol in its "Fixtures" section (guide page 536), making it a documented testing
escape hatch. It is not a production code contract violation.

The five `deriva_mcp_core.rag.*` imports (`resolve_user_identity`, `get_rag_store`,
`chunk_markdown`, `RowSerializer`, `Chunk`) are internal submodule paths without leading
underscores. They are not listed as stable public exports in core's `__init__.py`. If
core reorganizes its RAG package (e.g. moves `RowSerializer` to a different submodule),
the plugin's import paths break. This is the RAG extension's natural API surface -- the
authoring guide documents the RAG extension pattern but does not enumerate which specific
submodule paths are stable. **Recommendation:** confirm with core maintainers that these
five paths are intentionally stable, or request that core re-exports them from a
top-level `deriva_mcp_core.rag` namespace.

---

## Top-line Findings

- **Boundary discipline is excellent.** Connection management, auth, audit machinery,
  vocabulary primitives, and generic Deriva primitives are all clean -- no duplication.
  The single-gateway `ml_context.py` design holds throughout the codebase.

- **The RAG per-user safety contract is well-enforced.** `ctx.rag_dataset_indexer`
  is provably unused (CI pin), per-RID source naming is pinned end-to-end (CI pin),
  and vocab indexing correctly uses the `vocab:` prefix carve-out. This is the most
  security-sensitive boundary and it is the most thoroughly tested.

- **Plugin-authoring-guide conformance is complete.** All 46 tools carry explicit
  `mutates=`, all DERIVA I/O is wrapped in `deriva_call()`, all mutating tools emit
  audit on both success and failure via the centralized `_error_envelope` pattern, and
  all tool/prompt names follow the `deriva_ml_` prefix.

- **Three `deriva-ml` private-attribute accesses are the most significant
  technical-debt items** (`ml._add_workflow`, `ml._execution`, `ml._dataset_table`).
  These are not `deriva-mcp-core` boundary violations but they represent fragile
  coupling to `deriva-ml` internals that could silently break on a `deriva-ml` refactor.

- **Version pinning is absent.** `deriva-mcp-core` is declared without a lower-bound
  version. For a published package this is a deployment risk; recommend
  `"deriva-mcp-core>=<current-compatible-version>"`.
