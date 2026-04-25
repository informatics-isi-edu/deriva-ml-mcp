# deriva-ml-mcp — Design Spec

**Status:** Draft for review
**Date:** 2026-04-24
**Author:** Carl + Claude
**Workspace:** `/Users/carl/GitHub/DerivaML/`

## 1. Goal & Scope

### Goal

Create `deriva-ml-mcp`, a standalone repository and `deriva-mcp-core`
plugin that exposes DerivaML domain workflows (datasets, executions,
features, workflows) as MCP tools and resources. The tool surface is
designed fresh against the current `deriva-ml` API. The existing
`deriva-mcp` repository is used only as a checklist to confirm nothing
important was forgotten — not as a port list.

### In Scope

- Tools and resources that express **ML domain concepts** on top of
  generic Deriva primitives: dataset versioning, feature definitions,
  execution lifecycle, workflow registration.
- Test suite (unit + integration) modelled on `deriva-mcp-core/tests/`
  for unit mechanics and `deriva-mcp/tests/test_integration.py` for
  integration mechanics.
- Plugin packaging via Python entry point (`deriva_mcp.plugins`).
- A coverage report (`docs/coverage.md`) recording what happened to
  every old `deriva-mcp` tool: kept, renamed, merged, split, dropped to
  core, dropped redundant, dropped deprecated, or deferred.

### Out of Scope

- **Generic Deriva primitives** (entity CRUD, attribute query, schema
  introspection, annotations, hatrac, vocabulary mechanics). These come
  from core. If the plugin needs a primitive that core lacks, file a
  core gap; do not ship it in the plugin.
- **Connection management.** Core owns connections via
  `deriva_mcp_core.context`. The plugin uses `get_catalog()` and
  `get_request_credential()` through a single helper module; it never
  reimplements connection logic.
- **RAG management.** Core owns the RAG subsystem. If ML-specific RAG
  sources are needed, they are declared via the plugin authoring
  guide's `ctx.register_rag_source(...)` API — not reimplemented.
- **Updates to `deriva-skills`** or any prompts that reference old tool
  names. Handled in a separate follow-up project after this work
  completes.
- **Any change to the existing `deriva-mcp` repository.** It stays
  running, untouched, and is deprecated as a unit when `deriva-ml-mcp`
  is complete.
- **Vocabulary tools.** Fully covered by core. See section 4 for
  details.
- **RAG, prompts, result caching, background tasks, github_docs.**
  Either core's territory or out of scope entirely.

### Success Criteria

1. Every concept in current `deriva-ml`'s public API that LLMs need to
   drive ML workflows is reachable through (a) a plugin tool that
   encodes ML domain semantics, (b) a plugin resource, or (c) direct
   composition of core primitives by the LLM (with the composition
   pattern recorded in coverage.md for the skill-migration follow-up).
2. Every tool ships with a unit test (mocked `DerivaML`) and, where it
   touches a catalog, an integration test.
3. Coverage report exists and shows, for each old `deriva-mcp` tool,
   what happened to it (one row per old tool, exactly).
4. Plugin loads cleanly under `deriva-mcp-core`'s entry-point loader
   and registers without errors.
5. `uv run pytest` passes; `uv run pytest -m integration` passes
   against a local Deriva catalog.

## 2. Repo & Package Structure

### Layout

```
/Users/carl/GitHub/DerivaML/deriva-ml-mcp/
├── CLAUDE.md                  # Defers to ../CLAUDE.md for shared conventions
├── README.md                  # Installation and usage for operators
├── LICENSE
├── pyproject.toml             # uv-managed; depends on deriva-mcp-core, deriva-ml
├── uv.lock
├── docs/
│   ├── superpowers/specs/     # Design specs live here (this spec is the first)
│   ├── coverage.md            # Per-old-tool disposition table
│   └── workplan.md            # Phase-by-phase implementation plan
├── src/
│   └── deriva_ml_mcp/
│       ├── __init__.py
│       ├── plugin.py          # register(ctx) entry point
│       ├── ml_context.py      # The single helper for obtaining a DerivaML instance
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── feature.py
│       │   ├── workflow.py
│       │   └── execution.py
│       └── resources/
│           ├── __init__.py
│           └── ml.py          # deriva-ml://... resources
└── tests/
    ├── conftest.py            # mock_ml fixture, _CapturingMCP plugin context
    ├── test_plugin.py         # Entry-point smoke test
    ├── test_dataset.py
    ├── test_feature.py
    ├── test_workflow.py
    ├── test_execution.py
    └── test_integration.py    # @pytest.mark.integration
```

### Naming

| Aspect | Value |
|--------|-------|
| Repo / git | `deriva-ml-mcp` |
| Distribution name | `deriva-ml-mcp` |
| Import name | `deriva_ml_mcp` |
| Entry-point group | `deriva_mcp.plugins` |
| Entry-point name | `deriva-ml` (used in `DERIVA_MCP_PLUGIN_ALLOWLIST`) |
| Resource URI scheme | `deriva-ml://...` |

### Plugin Entry Point

```toml
[project.entry-points."deriva_mcp.plugins"]
deriva-ml = "deriva_ml_mcp.plugin:register"
```

`plugin.py:register(ctx)` calls each tool module's own `register(ctx)`,
mirroring `deriva-mcp-core/src/deriva_mcp_core/tools/__init__.py`.
Modules stay self-contained and testable in isolation.

### `ml_context.py`

The only module that touches core's connection plumbing. Provides a
single helper that returns a `DerivaML` instance built from
`get_catalog()` and `get_request_credential()`. Every tool obtains its
`DerivaML` through this helper. No tool re-implements connection logic
or duplicates core's auth.

### Versioning

`setuptools_scm` from git tags. `uv run bump-version` per workspace
conventions. Tag/commit auto-pushed.

### Things Deliberately Absent

- No `connection.py` (core owns connections).
- No `rag/` (core owns RAG).
- No `prompts.py` (deferred to skills/prompts follow-up).
- No `tasks.py`, `result_cache.py`, `github_docs.py` (core or out of
  scope).
- No `vocabulary.py` (core fully covers; see section 4).

## 3. Tool Surface Methodology

### Source-of-Truth Order

When deciding what tools exist, authority decreases down this list:

1. **`deriva-ml`'s public API** — the `DerivaML` class methods and
   supporting types. If a method exists and matters for an LLM-driven
   workflow, it deserves consideration. If it doesn't exist, we don't
   expose it.
2. **`deriva-ml`'s `CHANGELOG.md` and docs** — orient on what's new vs
   legacy.
3. **`deriva-mcp-core`'s built-in tools** — anything covered here is
   dropped from our surface (with a note in coverage.md).
4. **Existing `deriva-mcp` tools** — used as a "did we forget anything?"
   checklist at the END, not as a port list.
5. **`deriva-skills` skill content** — read to understand workflows
   users actually drive. Influences shape, not inclusion.

### Per-Concept Process

For each ML domain area (dataset, feature, workflow, execution):

1. Read the relevant `DerivaML` API surface and write down operations.
2. Cross out anything covered by a core primitive (note core tool name).
3. Cluster remaining operations into tools at the right granularity
   (see "Tool Granularity Rules" below).
4. Cross-check against the old `deriva-mcp` module — for each old tool,
   decide: covered (new name), redundant, dropped to core, or new (we
   forgot it).
5. Record outcomes in `coverage.md` immediately, while context is
   fresh.

### Tool Granularity Rules

- **One tool per user intent**, not one tool per `DerivaML` method. If
  three methods together accomplish what an LLM thinks of as a single
  act, they merge into one tool with sensible defaults.
- **Reads can be coarser than writes.** A single `get_dataset` returning
  the full shape is fine; mutations should be specific so `mutates=True`
  semantics and audit events are precise.
- **No tool that just renames or wraps a core primitive.** Both core's
  tools and the plugin's tools are registered against the same
  `PluginContext` — the LLM sees one unified tool list. The plugin only
  ships tools that encode genuine ML domain semantics core can't
  express. Operations the LLM should drive via core go in coverage.md
  as `dropped-to-core` (with the core tool name) so the skill-migration
  follow-up can update skills accordingly.

### Mutation & Audit Policy

Inherited from core's plugin authoring guide:

- Every tool registered with explicit `mutates=True` or `mutates=False`.
  Omitting raises `TypeError` at server startup.
- Every `mutates=True` tool wraps DERIVA I/O in `with deriva_call():`
  from `deriva_mcp_core.context`.
- Every `mutates=True` tool emits `audit_event(...)` on success and on
  failure with target identifiers (no payload values).

### Resources

Read-only views into ML-domain catalog state. URI patterns to plan for
(concrete list finalized during the per-concept analysis):

- `deriva-ml://datasets` — list active datasets (current connection)
- `deriva-ml://dataset/{rid}` — one dataset's detail
- `deriva-ml://features/{table}` — features defined on a table
- `deriva-ml://workflows` — registered workflows
- `deriva-ml://executions` — execution history

## 4. Coverage Report (`docs/coverage.md`)

### Purpose

Single source of truth for what happened to every old `deriva-mcp` tool
and resource. Built incrementally during implementation. Enables the
skill-migration follow-up project to mechanically translate skill
content.

### Format

One row per old tool, one row per old resource. Columns:

| Column | Meaning |
|--------|---------|
| `old_name` | Tool/resource name in `deriva-mcp` |
| `old_module` | File it lived in |
| `disposition` | `kept`, `renamed`, `merged`, `split`, `dropped-to-core`, `dropped-redundant`, `dropped-deprecated`, or `deferred` |
| `new_name` | Tool name in `deriva-ml-mcp`; or core tool name if `dropped-to-core`; or empty |
| `new_module` | Where it lives now |
| `signature_change` | `none` or brief description |
| `notes` | One-line rationale, especially for `dropped-*`, `merged`, `split` |

### Disposition Meanings

- **`kept`** — same name, same intent, possibly minor signature tweaks.
- **`renamed`** — same intent, new name.
- **`merged`** — folded into another tool. `new_name` is the survivor.
- **`split`** — broke into multiple tools. `new_name` is comma-separated.
- **`dropped-to-core`** — covered by a `deriva-mcp-core` built-in.
  `new_name` is the core tool.
- **`dropped-redundant`** — was a thin wrapper over a primitive; LLM
  composes from core. `notes` says which primitive(s).
- **`dropped-deprecated`** — old API no longer exists in current
  `deriva-ml`. `notes` explains.
- **`deferred`** — punted to a later phase or out of scope. `notes`
  says why.

### Confirmed Up-Front Decisions

The vocabulary investigation during brainstorming established that
core's vocabulary tools are a strict superset of `deriva-mcp`'s and
that `DerivaML`'s vocabulary mixin adds no ML-specific semantics that
need a tool wrapper. Coverage.md will record:

| old_name | old_module | disposition | new_name | notes |
|---|---|---|---|---|
| `add_term` | `vocabulary.py` | `dropped-to-core` | `add_term` | Core covers all ML vocab needs |
| `create_vocabulary` | `vocabulary.py` | `dropped-to-core` | `create_vocabulary` | Core covers all ML vocab needs |
| `add_synonym` | `vocabulary.py` | `dropped-to-core` | `add_synonym` | Core covers all ML vocab needs |
| `remove_synonym` | `vocabulary.py` | `dropped-to-core` | `remove_synonym` | Core covers all ML vocab needs |
| `update_term_description` | `vocabulary.py` | `dropped-to-core` | `update_term_description` | Core covers all ML vocab needs |
| `delete_term` | `vocabulary.py` | `dropped-to-core` | `delete_term` | Core covers all ML vocab needs |

### Maintenance

- Created during implementation. Each per-concept analysis pass writes
  the rows for that domain immediately, while the rationale is fresh.
- Reviewed at every phase boundary as part of Definition of Done.
- Kept current as a working document — discipline, not automation.

### Resource Coverage

Same table format, separate section. Old `deriva://...` URIs map to
new `deriva-ml://...` URIs (or `dropped-to-core` for ones core covers).

## 5. Testing Strategy

### Unit Tests

- **Fixture: `_CapturingMCP`-style plugin context.** A `PluginContext`
  whose `tool()` decorator stores the registered function in a dict
  keyed by name. Tests fetch tools by name and call them directly.
  Pattern from `deriva-mcp-core/tests/test_tools.py`.
- **Fixture: `mock_ml`.** A `MagicMock` standing in for `DerivaML`.
  Per-test setup configures return values for the methods being
  exercised.
- **Fixture: `ml_context_with_mock`.** Wires `mock_ml` so the plugin's
  `ml_context.py` helper returns it without touching real connections.
- **One test file per tool module** (`test_dataset.py`, etc.).
- **Coverage per tool:** success path, error path (`DerivaML` raises),
  and — for `mutates=True` tools — assertion that an `audit_event` was
  emitted on success and on failure.
- **Disconnected-state test:** when core's `get_catalog()` returns no
  active connection, the tool returns the standard error shape. Match
  the pattern from `deriva-mcp/tests/conftest.py`'s
  `disconnected_conn_manager`.

### Integration Tests

- Single file: `tests/test_integration.py`.
- Marked with `@pytest.mark.integration` (registered in `pyproject.toml`
  so `pytest -m "not integration"` is the default behaviour).
- Skip-logic copied from `deriva-mcp/tests/test_integration.py`: quick
  TCP connect to `localhost:443`; skip the whole module if no Deriva
  server is reachable.
- Honors `DERIVA_HOST` env var; defaults to localhost.
- Uses a `CatalogManager` fixture that creates and tears down a test
  catalog per session (same pattern as `deriva-mcp/tests/conftest.py`).
- **Coverage target:** at least one happy-path round trip per ML domain
  area. Not exhaustive — proves tools talk to a real catalog. More
  scenarios added during implementation as gaps are discovered.

### Plugin Smoke Test

`tests/test_plugin.py` verifies:

- `register(ctx)` runs without error.
- All expected tools are registered.
- Every registered tool has `mutates` set explicitly (catches the
  `TypeError` core raises at startup if omitted).

### Audit-Event Tests

For every `mutates=True` tool:

- Success: `audit_event` called with `<module>_<operation>` and target
  identifiers.
- Failure: `audit_event` called with `<module>_<operation>_failed` and
  `error_type`.

Tested by mocking `audit_event` and asserting on call arguments.

### Things Deliberately Not Tested

- Core's tools (have their own tests in `deriva-mcp-core`).
- Connection management, auth, RAG, audit machinery (also core's).
- `DerivaML`'s methods (tested in the `deriva-ml` repo).

### `uv` Invocation

Per workspace conventions:

```bash
uv run pytest                      # unit tests only (default skips integration)
uv run pytest -m integration       # live catalog tests
uv run pytest --cov                # with coverage
```

## 6. Phasing & Documentation

### Phase Order

| Phase | Domain | Why this order |
|-------|--------|----------------|
| 0 | Repo scaffold + plugin smoke | Empty plugin loads under core; CI green. Foundation for everything. |
| 1 | `ml_context.py` + connection helper | Every tool needs this. Tested in isolation before any tool depends on it. |
| 2 | `dataset` | Largest module in old `deriva-mcp` (1124 lines). Touches versioning, members, splits. Sets patterns later phases follow. |
| 3 | `feature` | Builds on dataset (features attach to dataset members). |
| 4 | `workflow` | Smaller, mostly registration. Independent of dataset/feature. |
| 5 | `execution` | Depends on workflow. Touches state machine, lifecycle, asset uploads — finishes the ML loop. |
| 6 | Resources | `deriva-ml://datasets`, `deriva-ml://dataset/{rid}`, etc. After tools so we know what reads are common. |
| 7 | Coverage validation pass | Walk every old tool name; confirm each has a row in coverage.md; close out. |

Each phase produces: tools + unit tests + integration tests +
coverage.md updates + phase-end commit. Each phase is shippable in
isolation.

### Phase 0 Deliverables (Concrete)

- Repo at `/Users/carl/GitHub/DerivaML/deriva-ml-mcp/` with
  `pyproject.toml`, `uv.lock`, `CLAUDE.md` (defers to `../CLAUDE.md`),
  `README.md`, `LICENSE`.
- Empty `src/deriva_ml_mcp/plugin.py:register(ctx)` registering nothing.
- Entry point declared in `pyproject.toml`.
- `tests/test_plugin.py` smoke test: plugin loads, register runs.
- `docs/coverage.md` with table headers and the vocabulary rows
  pre-populated.
- `docs/workplan.md` (produced by `writing-plans` skill after spec
  approval).
- `uv run pytest` passes.

### Documentation Deliverables

- `docs/superpowers/specs/2026-04-24-deriva-ml-mcp-design.md` — this
  spec, committed to git.
- `docs/coverage.md` — built incrementally during implementation.
- `docs/workplan.md` — produced by `writing-plans` skill.
- `README.md` — installation and usage for operators.
- `CLAUDE.md` — repo-specific guidance, defers to `../CLAUDE.md`.

### Per-Phase Definition of Done

1. All unit tests for tools added in this phase pass.
2. At least one integration test exercises a happy-path round trip for
   the phase's domain.
3. Every tool registered with explicit `mutates=`.
4. Every `mutates=True` tool has audit-event tests (success and
   failure).
5. Coverage.md updated for every old `deriva-mcp` tool in this phase's
   domain (exactly one row each).
6. `uv run ruff check` and `uv run ruff format --check` pass.
7. Commit on a feature branch; PR description summarizes coverage
   decisions made in this phase.

### Parallel Work

None planned. Phases are linear because each depends on patterns set
by the previous one. If the workplan finds parallelizable work, it
surfaces there.

## 7. Open Items Carried Into Implementation

- Final list of resources (URI patterns above are planning-stage only;
  finalized during phase 6).
- Whether any per-concept analysis turns up an ML-specific need that
  belongs in core rather than the plugin. If so, file as a core gap;
  do not work around it in the plugin.
- Specific `DerivaML` API methods that are private but driven through
  public methods may need re-evaluation as tools are designed.

## 8. Non-Decisions (Explicit)

To avoid relitigating in implementation:

- Approach 3 (domain-first redesign) was chosen over lift-and-shift.
- `deriva-skills` updates are a separate follow-up project, not part
  of this spec.
- `deriva-mcp` will be deprecated as a unit when this is complete; no
  incremental edits to the old repo during this work.
- Core owns connection management, auth, RAG, vocabulary tools, and
  generic Deriva primitives. The plugin never duplicates these.
- Coverage.md is a working document, not an executable test artifact.
