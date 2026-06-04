# Read-through RAG indexing: cut connect-time bulk passes

**Date:** 2026-06-04
**Repo:** `deriva-ml-mcp-plugin`
**Status:** Approved (design)

## Problem

Catalog connect has too much startup overhead. On every first connect
per (user, catalog), `register_rag_sources`
([resources/rag.py](../../../src/deriva_ml_mcp_plugin/resources/rag.py))
runs **three per-user bulk indexers** — Dataset, Workflow, Execution —
each of which fetches up to `_MAX_LIMIT` rows under the caller's
credential and embeds every row into the vector store. That fetch+embed
work is the dominant connect-time cost.

Two goals must survive the fix:

1. **Avoid duplicate creation** of datasets, vocabulary terms,
   workflows, and executions — the `semantic-awareness`
   find-before-you-create guardrail, which runs `rag_search` *before* a
   create.
2. **Find by description** — datasets, workflows, executions, tables
   must be discoverable from free-text, not just exact metadata.

The naive fix (delete the data-row indexers) breaks both goals for
workflows and executions. The constraint is real: anything fuzzy-
searchable by `rag_search` must be in the index, but bulk-indexing it on
connect is what costs.

## Decision

Replace **connect-time bulk indexing** with **read-through (index-on-
find) indexing**, and keep vocabulary always-indexed on connect.

- **Schema** — already indexed by `deriva-mcp-core` on connect
  (`schema:` prefix, catalog-public, TTL-gated). Untouched; free.
- **Vocabulary** — stays bulk-indexed on connect via this plugin's
  existing vocab hook (`vocab:` prefix, catalog-public, cheap). This is
  "vocabulary is always indexed": the term-dedup guardrail is never
  cold.
- **Dataset / Workflow / Execution** — **no bulk pass on connect.**
  Instead each row is indexed (per-user, `data:` prefix) at two moments:
  - **on create / mutate** — via the existing surgical `_reindex_*`
    helpers already wired into the mutation tools (unchanged); and
  - **on find / list / get** — NEW: the shared read path indexes each
    row it returns, fire-and-forget, so reads warm the index.

The index-on-find hook lands at the **`_list_*_impl` / `_get_*_impl`
layer** (`tools/dataset/read.py`, `tools/workflow.py`,
`tools/execution/read.py`). Both the `deriva_ml_*` MCP tools and the
`deriva://.../deriva-ml/...` resources call through these impls
([resources/ml.py:82-90](../../../src/deriva_ml_mcp_plugin/resources/ml.py)),
so hooking the impl layer gives auto-indexing to tools and resources
with one implementation per entity.

### Why this satisfies every goal

- **Startup:** connect does schema (core) + vocab only. The three
  per-user bulk fetch+embed passes are gone. This is the win.
- **Find by description (incl. workflows + executions):** all three
  entity types remain in `catalog-data` RAG, so `rag_search` ranks them
  by free-text exactly as today. The index is populated by the reads a
  session already performs (a `list`/`get`/resource-fetch warms the rows
  before any `rag_search` would want them).
- **Dedup guardrail:** intact for all four entity types — vocab terms
  (always on connect) plus datasets/workflows/executions (warmed by the
  list/find that `semantic-awareness` already does as step 1). Workflows
  additionally retain the deterministic `find_workflow_by_url`.
- **Cold-start gap closes naturally:** pre-existing rows enter a user's
  index the first time that user lists/browses them — no separate
  warm-up step, no bulk connect cost.

## Mechanism details

### Index-on-find hook

A small helper in `resources/rag.py`, e.g.
`_index_rows_background(hostname, catalog_id, table_token, rows,
serializer)`, that:

- resolves the user identity once,
- for each row, schedules `_write_row_chunk` under the per-RID `data:`
  source name,
- runs as a fire-and-forget background task (`asyncio.create_task` or
  equivalent) so it **never blocks or fails the read's response**,
- swallows all errors (best-effort, logged at debug), mirroring the
  existing `_reindex_*` contract.

Each `_list_*_impl` / `_get_*_impl`, after building its summary rows,
calls this helper with the rows it is already returning. No extra
catalog fetch — it reuses the rows in hand.

Idempotency: `_write_row_chunk` already does `delete_source` +
`add`, so re-indexing an unchanged row is a cheap no-op-equivalent.
Re-warming the same rows on every list is acceptable; if profiling shows
it matters, add a short per-process (host,cat,user,rid) seen-set to skip
recently-indexed RIDs — deferred until measured.

### Background warm order: follow the read, don't re-sort

The index fills incrementally in the background: `store.add` offloads
the ONNX embedding via `asyncio.to_thread`
([core store.py](../../../../deriva-mcp-core/src/deriva_mcp_core/rag/store.py))
and the writer adds in batches, so each row becomes `rag_search`-able as
soon as *its* write lands — not after the whole set finishes.

The warm indexes rows **in the order the read returned them** — it does
NOT impose its own sort. An earlier draft of this design had the warm
re-sort newest-first (record-creation descending) as a heuristic to make
recently-created rows searchable first during the brief request/index
overlap window. That heuristic is **dropped**: (1) "most recent = most
relevant" is false for the dominant use case (find-before-create matches
*semantic similarity*, where row age is irrelevant); (2) a single list
page warms in seconds, so the ordering only affects which rows are
searchable in the first second vs. the next few — a narrow, low-value
window; and (3) no summary model carries a record-creation timestamp, so
the sort was inert anyway. Removing it deletes dead plumbing (`rct_key`)
rather than adding a wire-shape field to honor a questionable heuristic.

If a caller genuinely wants recent-first overlap, the read already
supports it: `deriva_ml_list_*` exposes `sort=True` (RCT-descending page
order), and the warm follows whatever order the page came in. So the
recency preference lives at the call site, where the user's intent
actually is — not hardcoded in the indexer. The read's default order
(RID-ascending, the stable-cursor contract) is unchanged; the warm
simply mirrors the returned page.

### What is removed

`resources/rag.py`:
- The three `ctx.on_catalog_connect(_make_hook(...))` registrations for
  Dataset / Workflow / Execution.
- The `_make_hook` **bulk-pass factory** (the on-connect bulk indexer).
  This is the only thing whose removal is the actual startup win.

Note the `_fetch_*_rows` fetchers are **NOT** removed — they are still
used by the kept `deriva_ml_resync_indexes` path (`_resync_one_table`
enumerates a table via its `_fetch_*_rows` fetcher). Only the
`_make_hook` on-connect wiring goes.

`tools/maintenance.py`:
- (nothing removed) — `deriva_ml_resync_indexes` is **kept** as an
  explicit "warm all of my datasets/workflows/executions for this
  catalog now" button. Index-on-find makes it non-essential for steady-
  state use, but it stays as the deterministic operator warm-up after a
  restart (and as the manual bridge for `add_term`-style mutations that
  fire no lifecycle hook). Its `_resync_user_sources` /
  `_resync_one_table` machinery in `rag.py` therefore also **stays**.

### What stays (unchanged)

- The surgical `_reindex_dataset/workflow/execution`,
  `_delete_dataset_source`, `_row_source_name`, `_write_row_chunk`,
  and the per-table serializers (`_DatasetSerializer`,
  `_WorkflowSerializer`, `_ExecutionSerializer`) — now the *primary*
  index writers, reused by both the create/mutate path and the new
  index-on-find path.
- The mutation-tool reindex call sites in `tools/dataset/mutate.py`
  (×5) and `tools/workflow.py` (×2).
- All vocab machinery + `deriva_ml_reindex_vocabularies` + both
  `rag_github_source` declarations.
- Core's schema indexing.

## Retrieval modes: structured vs. fuzzy vs. hybrid

A "find" specified by structured attributes (type, status, RID, URL)
does **not** need the fuzzy index — the catalog answers it
deterministically. RAG is only for the free-text/description residue.
The skill routes by query shape:

1. **Structured** — "find Training datasets", "Failed executions",
   "the workflow at this URL". → deterministic `find_*` / `list_*` with
   a type/status/RID/URL filter. Exact; no embedding consulted. (Also
   warms the index newest-first as a side effect — see index-on-find.)
2. **Free-text** — "the run about retinopathy screening". → `rag_search`
   for fuzzy ranking over the indexed description/name.
3. **Hybrid** — "all Training datasets matching <description text>". →
   structured filter narrows the candidate set, then fuzzy ranks within
   it. First-class mode: the structured filter MUST exist for the
   narrow-then-rank flow to work.

We still index all three entity types (so free-text and the fuzzy half
of hybrid work); routing just avoids paying for fuzzy when the query is
fully structured.

### Consistency fix: type filters on datasets and workflows (required)

Type-based lookup ("find Training datasets") will be common, and the
hybrid mode depends on a structured type filter existing. Today the
library is uneven:

- `find_executions(workflow_type=, status=, workflow_rid=, dataset=)` —
  structured filters already exist; executions read path exposes them.
- `find_datasets(deleted=, sort=)` — **no `dataset_type` filter.**
- `find_workflows(sort=)` — **no `workflow_type` filter.**

Add **client-side** type filtering to the plugin read helpers (no
library change needed):

- `_list_datasets_impl(..., dataset_type: str | None = None)` — filter
  the rows `find_datasets()` already returns. `DatasetSummary` already
  carries `dataset_types` (a **list** — a dataset may have several), so
  the filter is a **membership test** (`dataset_type in
  row.dataset_types`), done in-memory before pagination. No extra
  catalog fetch.
- `_list_workflows_impl(..., workflow_type: str | None = None)` —
  same pattern; `WorkflowSummary.workflow_type` is likewise a **list**,
  so the filter is the same membership test (`workflow_type in
  row.workflow_type`).

Surface the new param on the corresponding `deriva_ml_list_datasets` /
`deriva_ml_list_workflows` tools (and the list resources). This gives
all three ML entities a **consistent** structured-filter surface and
makes mode 1 + mode 3 real for datasets and workflows, not just
executions.

(If the deriva-ml library later grows native `dataset_type` /
`workflow_type` filter params, the impls switch to forwarding them and
drop the client-side pass — but that is a separate, optional follow-on,
not a blocker here.)

## Skill / prompt alignment (deriva-skills, separate repo)

`semantic-awareness` (and `references/find-before-you-create.md`)
currently states the RAG index "covers schema, vocabulary terms, and
data" and reaches for `rag_search` as step 1. Two refinements under this
design:

1. **Route by query shape (structured / fuzzy / hybrid).** When the
   find is fully specified by a type/status/RID/URL, use the
   deterministic `find_*` / `list_*` filter — do not `rag_search`. Use
   `rag_search` for free-text description matching, and for the hybrid
   case (`find Training datasets matching <text>`) apply the structured
   type filter first, then rank the narrowed set with `rag_search`.
2. **Warm-before-rank ordering.** The structured `list`/`find` step also
   warms the per-user index (index-on-find), so doing it first means the
   subsequent `rag_search` sees freshly-indexed rows. Vocab terms are
   always warm (on-connect); data rows warm on first browse. Workflow
   dedup additionally routes to the content-addressed
   `find_workflow_by_url`.

(Tracked as a follow-on change to
`deriva-skills`; out of scope for the plugin code change but required
for end-to-end correctness.)

## Test impact

- `test_rag.py` — drop the three bulk-hook tests (`test_*_hook_writes_
  one_per_rid_source_per_row`, fetch-exception, per-user-partition bulk
  tests) and the `_fetch_*_rows` tests. Keep the serializer tests, the
  per-RID source-naming test, the `_reindex_*` tests, and all vocab
  tests. Add tests for the new index-on-find helper (fires per row,
  fire-and-forget, swallows errors, never raises into the read).
- `test_plugin.py` —
  `test_register_wires_four_catalog_connect_hooks` → one-hook assertion
  (vocab only); drop `test_data_sources_use_per_rid_naming` (no bulk
  hooks); keep `test_vocab_hook_writes_to_vocab_prefix_not_data_prefix`
  (adjust hook index `[3]` → `[0]`); keep
  `test_register_does_not_use_rag_dataset_indexer`.
- `test_maintenance.py` — unchanged: both `deriva_ml_resync_indexes`
  and `deriva_ml_reindex_vocabularies` tests stay (resync is kept).
- Add: read-tool tests asserting that `deriva_ml_list_datasets` /
  `list_workflows` / `list_executions` / `get_execution` schedule the
  index-on-find write for each returned row (and that a write failure
  does not affect the tool's returned payload).
- `test_dataset_mutate.py` / workflow mutate tests — unchanged (the
  create/mutate reindex path stays).
- Add: type-filter tests for `_list_datasets_impl(dataset_type=...)` and
  `_list_workflows_impl(workflow_type=...)` — membership match against
  the list-valued type field, no match → empty page, `None` → unchanged
  behavior, and the filter runs before pagination (so `limit` applies to
  the filtered set, not the pre-filter set).

## Resolved decisions

- **Newest-first background warm order** — yes, the default (no config
  knob).
- **`deriva_ml_resync_indexes`** — **kept** as the explicit "warm
  everything now" operator button; the resync machinery stays.
- **deriva-skills `semantic-awareness` update** — accepted as a separate
  follow-on change in that repo.

## Out of scope

- eye-ai-deriva-mcp-plugin (clinical-row indexer already empty/opt-in).
- The vocabulary indexing mechanism (preserved verbatim).
- deriva-mcp-core / framework changes.
- The deriva-skills doc change is required for correctness but is a
  separate repo + separate change.
