Ever and # Storage Introspection Surface — Design

**Date:** 2026-06-11
**Status:** Approved in brainstorming; spec for implementation.
**Subproject:** `deriva-ml-mcp-plugin`
**Depends on:** deriva-ml ≥ 1.46.1 (`list_cached_bags` /
`list_cached_assets` / `delete_cached_bag` / `delete_cached_asset` /
extended `get_storage_summary`, shipped in deriva-ml PRs #286/#289/#290).

## 1. Problem statement

deriva-ml 1.46 gave the Python API full local-storage introspection
(what bags/assets are cached, per-species summaries) and targeted
deletion. None of it is reachable over MCP. Worse, the
`manage-deriva-storage` skill references `deriva://storage/summary`,
`deriva://storage/cache`, and `deriva://storage/execution-dirs` —
resources that **have never existed** in this plugin or in
deriva-mcp-core (a phantom surface). And there is no MCP-readable
orientation document explaining what the cache *is*: an agent
connected to the server has no way to learn the storage model (cache
vs working directory, the three storage species, what's safe to
delete) before acting.

The write side is already exposed (`deriva_ml_cache_dataset` warms
the server's cache), making the absence of the read/cleanup side an
asymmetry.

## 2. Scope decision (made in brainstorming)

**Resources + targeted deletes.** Four read-only data resources, one
orientation resource, three targeted cleanup tools. **No blanket
`deriva_ml_clear_cache` over MCP** — bulk cache wipes on a possibly
shared server stay a server-operator (Python API / shell) action.

## 3. The server-local caveat (design stance)

The cache and working directory live on the **MCP server process's
filesystem**, per `(hostname, catalog_id)` — typically
`~/.deriva-ml/{hostname}/{catalog_id}/`. When the server runs on the
user's machine (stdio launch) this is the user's own cache; when the
server is remote, it is an inspection of the *server's* cache. Every
resource/tool response carries `"server_local": true` and the
descriptions state the caveat plainly. We do not pretend this is the
client's disk.

The catalog-scoped URI shape resolves "whose cache" *within* the
server: each `(hostname, catalog_id)` pair has its own cache
subtree, exactly mirroring deriva-ml's on-disk layout.

## 4. Resources (added to `resources/ml.py`)

All follow the module's existing pattern: read-only, catalog-scoped
URIs, JSON payloads, error envelope via `_helpers`, `DerivaML`
obtained through `ml_context.py`.

| URI | Backing call | Payload |
|---|---|---|
| `deriva://catalog/{h}/{c}/deriva-ml/storage/summary` | `ml.get_storage_summary()` | the summary dict as-is (incl. the per-species keys `bag_count`/`bag_size_mb`/`asset_count`/`asset_size_mb`) + `server_local` |
| `deriva://catalog/{h}/{c}/deriva-ml/storage/bags` | `ml.list_cached_bags()` | `{"bags": [CachedBag.model_dump(mode="json"), …], "count": n, "server_local": true}` — most-recently-built first (order is part of the deriva-ml contract) |
| `deriva://catalog/{h}/{c}/deriva-ml/storage/assets` | `ml.list_cached_assets()` | `{"assets": [CachedAsset.model_dump(mode="json"), …], "count": n, "server_local": true}` |
| `deriva://catalog/{h}/{c}/deriva-ml/storage/execution-dirs` | `ml.list_execution_dirs()` | `{"execution_dirs": [...], "count": n, "server_local": true}` (dicts pass through; `path` values stringified) |
| `deriva://catalog/{h}/{c}/deriva-ml/storage/guide` | static text + live paths/summary | see §5 |

`model_dump(mode="json")` handles `Path` → str and `datetime` → ISO
serialization; `status` (a `StrEnum`) serializes to its string value.

Empty caches yield empty lists with `count: 0` — never errors.

## 5. The storage guide resource

`deriva://catalog/{h}/{c}/deriva-ml/storage/guide` is the
orientation document that previously existed nowhere on the MCP
side. It is **mostly static prose with a live header**, assembled per
request:

**Live header (from the catalog's `DerivaML` instance):**
- resolved `cache_dir` and `working_dir` paths for this
  `(hostname, catalog_id)`,
- current one-line summary (bag count / asset count / execution-dir
  count / total MB) from `get_storage_summary()`.

**Static body (single module-level template string):**
1. **The three storage species** — what each is, what writes it,
   where it lives:
   - *Cached dataset bags* — content-addressed
     `cache/bags/{checksum}/Dataset_{RID}/` tracked in
     `cache/index.sqlite`; written by dataset downloads and
     `deriva_ml_cache_dataset`.
   - *Cached input assets* — `cache/assets/{rid}_{md5}/`; written by
     `AssetSpec(cache=True)` / `download_asset(use_cache=True)`.
   - *Execution working directories* — one per execution; hold
     staged outputs until `commit_output_assets()` uploads them.
2. **Cache vs working directory** — purpose, sharing semantics
   (cache shared across executions; working dirs per-execution),
   configurability (`cache_dir` / `working_dir` parameters), and the
   safety rule: **cache contents are always re-downloadable; an
   execution directory whose outputs were never committed is the
   only local data that cannot be recovered from the catalog.**
3. **Index-coherence rule** — never hand-edit or delete inside
   `cache/bags/` or `index.sqlite`; the deletion tools remove index
   row and directory together.
4. **Surface directory** — which resource answers which question and
   which tool performs which cleanup (the four data resources + three
   tools by name), plus the bag-preview resource and
   `deriva_ml_bag_info` for size-before-download questions.
5. **Server-local caveat** — as §3.

The static body is the canonical storage-model text; the
`manage-deriva-storage` skill will point here instead of restating
(and instead of its current phantom URIs).

## 6. Tools (added to `tools/maintenance.py`)

All stateless (`hostname=`, `catalog_id=` like every plugin tool),
idempotent, returning the deriva-ml stats dicts inside the standard
success/error envelope, each response also carrying
`"server_local": true`.

| Tool | Backing call | Returns |
|---|---|---|
| `deriva_ml_delete_cached_bag(hostname, catalog_id, dataset_rid, version=None)` | `ml.delete_cached_bag(...)` | `{"bags_removed": n, "bytes_freed": n}` |
| `deriva_ml_delete_cached_asset(hostname, catalog_id, rid, md5=None)` | `ml.delete_cached_asset(...)` | `{"assets_removed": n, "bytes_freed": n}` |
| `deriva_ml_clean_execution_dirs(hostname, catalog_id, older_than_days=None, exclude_rids=None)` | `ml.clean_execution_dirs(...)` | `{"dirs_removed": n, "bytes_freed": n, "errors": n}` |

Tool descriptions carry the safety framing: cached bags/assets are
re-downloadable; `clean_execution_dirs` removes staged outputs that
may not have been uploaded — callers should check
`storage/execution-dirs` (and execution status) first, and the
description says so. Deleting something that isn't cached returns
zeros, not an error.

## 7. Plumbing

- `pyproject.toml`: raise the deriva-ml pin to `>= 1.46.1`.
- `plugin.py`: no changes expected — `resources/ml.py` and
  `tools/maintenance.py` registrars are already dispatched.
- Prompt updates: add one line to the `deriva_ml_getting_started`
  prompt's resource overview mentioning the `storage/*` family and
  the guide.

## 8. Testing

Follow the existing tests/ pattern (fake/mocked `DerivaML` via the
module's established conftest helpers; no live catalog):

- Resource routing: each of the five URIs resolves and returns the
  expected payload shape (`count`, `server_local`, serialized
  fields — `path` is a string, `built_at` ISO, `status` a plain
  string).
- Empty-cache behavior: empty lists, `count: 0`, no error envelope.
- Guide content: live header contains the resolved cache_dir /
  working_dir strings; body contains the three species and the
  index-coherence rule (string-presence assertions, not full-text
  golden files).
- Tools: each delegates with the right arguments (mock assertion),
  wraps the stats dict in the success envelope, and passes through
  idempotent zeros.
- Pin: a test importing the four deriva-ml APIs guards the
  version-bump requirement at CI time.

## 9. Follow-through in other repos (separate commits, same effort)

- **deriva-ml-skills / manage-deriva-storage**: replace the phantom
  `deriva://storage/*` references with the real
  `deriva://catalog/{h}/{c}/deriva-ml/storage/*` URIs and the three
  tool names; point the conceptual sections at the guide resource.
- No deriva-mcp-core changes.

## 10. Non-goals

- `deriva_ml_clear_cache` over MCP (scope decision, §2).
- Client-side cache inspection (MCP architecturally cannot see the
  client's disk).
- Cache eviction policy / quotas.
- RAG indexing of the guide (static orientation text; the existing
  prompt/resource discovery flow covers it).
