# Vocabulary and Asset Resources — Design

**Status:** Draft (brainstorming complete)
**Author:** Carl Kesselman
**Date:** 2026-05-05
**Target version:** v3.4

## Summary

Add four new MCP resources to `deriva-ml-mcp` that expose vocabulary
tables and asset tables in a schema-scoped hierarchy, and remove two
existing resources whose surface they supersede.

| URI | Returns |
|---|---|
| `ml/vocabularies/{schema}` | Vocabulary tables in a schema, each `{name, rid, term_count}` |
| `ml/vocabularies/{schema}/{vocab_name}` | Full terms in one vocabulary, each `{name, rid, description, synonyms, id, uri}` |
| `ml/assets/{schema}` | Asset tables in a schema, each `{name}` |
| `ml/assets/{schema}/{asset_table}` | Assets in one asset table, each `AssetSummary` (paginated cap) |

**Removed:**

- `ml/registries` (curated four-vocabulary bundle, superseded by `ml/vocabularies/{schema}`)
- `ml/asset-tables` (flat all-schemas list, superseded by `ml/assets/{schema}`)

This is a wire-breaking change. Existing callers of the removed URIs
will get "resource not found" after the cut. Acceptable per the
plugin's pre-1.0 evolution posture.

## Motivation

The plugin currently exposes ML-domain resources for datasets,
workflows, executions, features, and individual assets, but the
**discovery surface** for vocabularies and asset tables is inconsistent:

- Vocabularies have a single `ml/registries` resource that bundles only
  the four hardcoded `MLVocab` enum entries (`Dataset_Type`,
  `Workflow_Type`, `Asset_Type`, `Execution_Status`). User-defined
  domain vocabularies (e.g. `Image_Annotation_Type`) don't appear, and
  the per-term shape is stripped to `{name, rid}` only — descriptions,
  synonyms, CURIE, and URI are unavailable through the resource.
- Asset tables have `ml/asset-tables` (flat all-schemas list) and
  `ml/asset/{rid}` (single-asset detail), but no resource for "the
  contents of one asset table" — the `deriva_ml_list_assets` tool is
  the only path, which forces pagination handling for what is often a
  reasonable snapshot fetch.

Both gaps push callers either to raw `deriva-mcp-core` schema-walking
tools or to RAG-based semantic search, neither of which gives the
canonical structural answer to "what vocabularies / asset tables exist
in this schema, and what's in them."

The design fills both gaps with a consistent schema-scoped hierarchy
and a singular-vs-plural URI convention (singular for one-thing-by-RID,
plural for browsing a collection).

## Design

### URI hierarchy

The four new URIs follow a strict pattern:

```
deriva://catalog/{hostname}/{catalog_id}/ml/vocabularies/{schema}
deriva://catalog/{hostname}/{catalog_id}/ml/vocabularies/{schema}/{vocab_name}
deriva://catalog/{hostname}/{catalog_id}/ml/assets/{schema}
deriva://catalog/{hostname}/{catalog_id}/ml/assets/{schema}/{asset_table}
```

Schema is honored literally — the resource resolves the table via
`ml.model.schemas[schema].tables[name]`, NOT via deriva-ml's
`name_to_table()` (which silently searches across schemas in priority
order). Two same-named vocabularies in different schemas are
disambiguated by the URI's schema segment.

The existing `ml/asset/{rid}` (singular) resource for single-asset
detail is unchanged. Singular-vs-plural carries semantic weight:

- `ml/asset/{rid}` — one asset, addressed by its RID
- `ml/assets/{schema}` — collection of asset tables, scoped by schema
- `ml/assets/{schema}/{asset_table}` — collection of assets in one table

The same convention applies to vocabularies. There is no
`ml/vocabulary/{rid}` (a single-term resource) because terms are not
canonically RID-addressable in user workflows; callers needing one
term use core's `lookup_term` tool.

### Response shapes

All response shapes are Pydantic models in
`src/deriva_ml_mcp/_response_models.py` with `extra="forbid"` per the
v1.6+ contract. Wire serialization via
`payload.model_dump_json(by_alias=True)`.

#### `ml/vocabularies/{schema}`

```python
class VocabularyTableSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    rid: str
    term_count: int | None  # None on best-effort fetch failure

class VocabularyTablesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_: str = Field(alias="schema")
    count: int
    vocabularies: list[VocabularyTableSummary]
```

`term_count` is computed via `len(ml.list_vocabulary_terms(table))` per
vocab. Vocab term counts are bounded (typically tens, occasionally
hundreds) so the cost is acceptable. Per-vocab failures (transient
catalog hiccup, malformed table) surface as `term_count = None` rather
than aborting the snapshot — same best-effort pattern as the existing
`_vocab_terms` helper.

#### `ml/vocabularies/{schema}/{vocab_name}`

```python
class VocabularyTermDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    rid: str
    description: str | None
    synonyms: list[str]
    id: str | None      # CURIE
    uri: str | None     # full URI

class VocabularyTermsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_: str = Field(alias="schema")
    vocabulary: str
    count: int
    terms: list[VocabularyTermDetail]
```

All six fields from deriva-ml's `VocabularyTerm` are surfaced. The name
`VocabularyTermDetail` (rather than `VocabularyTerm`) avoids visual
collision with deriva-ml's own `VocabularyTerm` Pydantic class.

The list is bounded by `_MAX_LIMIT` (1000) for consistency with other
list-style resources. Vocabularies almost always fit under the cap; if
truncation ever fires, callers fall back to core's
`list_vocabulary_terms` tool for paginated access.

#### `ml/assets/{schema}`

```python
class AssetTableNameRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    rid: str

class AssetTablesInSchemaResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_: str = Field(alias="schema")
    count: int
    asset_tables: list[AssetTableNameRef]
```

No `asset_count` (in deliberate contrast to vocabularies'
`term_count`). Asset tables can have millions of rows; deriva-ml's
`list_assets` materializes rows to count them, and pulling in raw
deriva-py count machinery to avoid that would introduce boundary
tension with the deriva-ml API surface this plugin is built on. The
`ml/assets/{schema}/{asset_table}` resource carries `truncated` /
`next_after_rid` so callers learn "this table is huge" the moment they
fetch the snapshot.

#### `ml/assets/{schema}/{asset_table}`

```python
class AssetTableContentsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_: str = Field(alias="schema")
    asset_table: str
    count: int
    truncated: bool
    next_after_rid: str | None
    assets: list[AssetSummary]  # existing model
```

Reuses the existing `AssetSummary` shape so the resource and the
paginated `deriva_ml_list_assets` tool cannot drift. Cap at
`_MAX_LIMIT` mirrors `ml/datasets`, `ml/workflows`, etc.

### Validation flow

For both `{schema}/{vocab_name}` and `{schema}/{asset_table}`, the
wrapper checks:

1. `schema in ml.model.schemas?` — else error envelope: `"schema not found: {schema}"`
2. `vocab_name in ml.model.schemas[schema].tables?` (or `asset_table in ...`) — else error: `"vocabulary not found: {schema}.{name}"` / `"asset table not found: {schema}.{name}"`
3. `table.is_vocabulary()` (or `table.is_asset()`) — else error: `"table {schema}.{name} is not a vocabulary"` / `"table {schema}.{name} is not an asset table"`
4. Fetch contents via `ml.list_vocabulary_terms(table)` / `ml.list_assets(table)`

The schema-scoped list resources (`ml/vocabularies/{schema}`,
`ml/assets/{schema}`) only do step 1; an existing schema with zero
matching tables returns an empty list (not an error).

All deriva-ml synchronous calls are wrapped in `await asyncio.to_thread(...)`
per the plugin's async discipline (CLAUDE.md → "Sync calls in async
tools").

### Helper structure

All helpers inline in `resources/ml.py`. No `tools/vocabulary.py` or
`tools/asset.py` additions for the new endpoints — these are
resource-only reads, no paginated tool counterpart, and CLAUDE.md's
"Read tools have `_impl` helpers; mutate tools don't" pattern
addresses tool/resource sharing, not resource-only code.

If a paginated `deriva_ml_list_vocabularies` or
`deriva_ml_list_schema_assets` tool is added later, the helper can be
extracted at that point — premature extraction is over-engineering per
the workspace's "no over-engineering" principle.

The `_get_ml_registries_impl` and `_vocab_terms` helpers are deleted
along with `MLRegistriesResponse` and `VocabularyTermRef`. The
`_list_asset_tables_impl` helper in `tools/asset.py` is deleted along
with `ml/asset-tables`.

### RAG considerations

These resources are pure read operations on
`ml.model.schemas[...]` and `ml.list_vocabulary_terms(...)` /
`ml.list_assets(...)`. They do **not** write to the vector store, do
**not** trigger re-index hooks, do **not** fetch RAG-indexed data.
They are orthogonal to the RAG subsystem.

Existing RAG state is unchanged:

- Vocabularies remain catalog-public RAG-indexed under the `vocab:`
  source prefix on first catalog connect, with manual refresh via
  `deriva_ml_reindex_vocabularies`.
- The Dataset/Workflow/Execution per-user-per-RID indexer is
  untouched.

One divergence worth noting: the existing vocab indexer at
`resources/rag.py` writes only `{name, description, synonyms, rid}`
per term — it does not surface `id` (CURIE) or `uri`. The new
`VocabularyTermDetail` model returns all six fields. This means the
resource exposes more than `rag_search` returns. This is intentional:
the resource is the canonical structural source, RAG is approximate.
A future maintainer should not "fix" this divergence without weighing
the cost of re-indexing all vocab content.

#### Out of scope: asset RAG indexing

Asset tables and asset rows are **not** currently RAG-indexed. The
Dataset/Workflow/Execution trio is per-user-per-RID indexed, and
vocabularies are catalog-public indexed, but the asset surface has no
equivalent.

The new `ml/assets/{schema}/{asset_table}` resource gives **structural**
access to assets but does not introduce semantic search over them.

Adding asset RAG indexing is a separate decision deferred for now:
asset rows are heavy file metadata with little descriptive content, so
the return on investment for RAG-indexing them is unclear. Including
this note explicitly so a future maintainer reading this spec
understands the omission is deliberate, not an oversight.

## Tests

Approximately 14 unit tests (~7 vocabulary + ~7 asset), parallel
across the two domains:

**Vocabulary:**

- `ml/vocabularies/{schema}` returns vocab list for a schema with
  multiple vocabs (happy path)
- `ml/vocabularies/{schema}` returns empty list for a schema with no
  vocabs
- `ml/vocabularies/{schema}` returns error envelope when schema
  doesn't exist
- `ml/vocabularies/{schema}` falls back to `term_count = None` when
  one vocab's term fetch raises (best-effort)
- `ml/vocabularies/{schema}/{name}` returns full term list with all
  six fields populated
- `ml/vocabularies/{schema}/{name}` returns error envelope when vocab
  table doesn't exist in the schema
- `ml/vocabularies/{schema}/{name}` returns error envelope when table
  exists but `is_vocabulary()` is False

**Asset:**

- `ml/assets/{schema}` returns asset table list for a schema with
  multiple asset tables (happy path)
- `ml/assets/{schema}` returns empty list for a schema with no asset
  tables
- `ml/assets/{schema}` returns error envelope when schema doesn't
  exist
- `ml/assets/{schema}/{table}` returns asset list with `truncated` /
  `next_after_rid` populated correctly when the table fits under
  `_MAX_LIMIT`
- `ml/assets/{schema}/{table}` returns `truncated=True` when the table
  exceeds `_MAX_LIMIT`
- `ml/assets/{schema}/{table}` returns error envelope when asset
  table doesn't exist in the schema
- `ml/assets/{schema}/{table}` returns error envelope when table
  exists but `is_asset()` is False

The full test suite must pass after the deletions
(`MLRegistriesResponse`, `VocabularyTermRef`, `_get_ml_registries_impl`,
`_vocab_terms`, `_list_asset_tables_impl`, `AssetTablesResponse`,
`AssetTableRef`) — these are not separately tested but their removal
must not break unrelated tests via stale imports or test fixtures.

## Skill update follow-ups

These changes affect skills in the companion `deriva-ml-skills` and
`deriva-skills` Claude Code plugins. The implementation plan
(writing-plans phase) will pick these up as a separate block of tasks
after the `deriva-ml-mcp` changes land and a release is tagged.

**`deriva-ml-skills` updates (canonical paths):**

- `skills/work-with-assets/SKILL.md`
- `skills/work-with-assets/references/concepts.md`
- `skills/work-with-assets/references/workflow.md`
  → Update `ml/asset-tables` references to `ml/assets/{schema}`. Add
  the new `ml/assets/{schema}/{asset_table}` snapshot resource as the
  "browse this asset table" path. **Highest priority** — directly
  impacted by the asset URI changes.

- `skills/dataset-lifecycle/SKILL.md`
- `skills/dataset-lifecycle/evals/evals.json`
- `skills/dataset-lifecycle/references/concepts.md`
- `skills/dataset-lifecycle/references/workflow.md`
- `skills/dataset-lifecycle/references/bags.md`
  → Update any `ml/asset-tables` references to `ml/assets/{schema}`.

- `skills/execution-lifecycle/SKILL.md`
- `skills/debug-bag-contents/SKILL.md`
  → Verify and update any asset URI references.

- `skills/deriva-ml-context/SKILL.md`
  → The always-on context skill. Add the new vocabulary/asset URIs
  under "core ML resources" so the LLM knows they exist. Scrub
  references to the removed `ml/registries` and `ml/asset-tables`.

- `skills/api-naming-conventions/SKILL.md`
  → Verify the singular/plural URI pattern matches what we just
  locked (`asset` singular for RID, `assets` plural for collection;
  same for `vocabulary` / `vocabularies`).

**`deriva-skills` updates:**

- `skills/deriva-context/references/concepts.md`
  → Verify only. Vocabulary discovery via `find_vocabularies()` is a
  deriva-ml concept, not core, so likely no change needed.

**No new skills proposed.** The new resources slot into existing
skill content; vocabularies are covered by the `deriva-ml-context`
discovery surface, and assets are covered by the `work-with-assets`
canonical skill.

## Out of scope

- A paginated `deriva_ml_list_vocabularies` or
  `deriva_ml_list_schema_assets` tool. The current set of bounded
  resources is sufficient; if a future use case demands cursor
  pagination over hundreds of vocabularies in one schema, the helper
  can be extracted from the inline resource implementation at that
  point.
- A `ml/vocabulary/{rid}` (single-term-by-RID) resource. Use core's
  `lookup_term` tool.
- Asset RAG indexing (see RAG considerations above).
- Migration shims for the removed URIs. Per the plugin's pre-1.0
  posture, callers update to the new URIs at the cutover.

## Implementation notes

- Insertion order in `resources/ml.py`: place the four new
  `@ctx.resource(...)` blocks after the existing
  `ml_features_for_table` block (alongside `ml_asset_detail`, before
  `ml_registries`'s removal site). Keep registration order stable
  for predictable startup logs.
- The `MLVocab` import in `resources/ml.py` is removed along with
  `_get_ml_registries_impl`. The enum is still used elsewhere in the
  plugin (verify grep before deleting any other imports).
- `populate_by_name=True` on the new response models lets construction
  use either `schema=` or `schema_=` keywords; the alias handles the
  wire serialization. Same pattern as the existing `AssetTableRef`.
- Validation order matters: schema check → table-exists check →
  is-vocabulary / is-asset check. The early checks produce more
  specific error messages and avoid spurious `DerivaMLException`
  bubbling.
