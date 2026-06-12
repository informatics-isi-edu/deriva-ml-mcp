# ADR-0001: Local storage management is out of MCP scope

Date: 2026-06-11
Status: Accepted

## Context

deriva-ml 1.46 added a local cache & storage introspection API to the
Python library: `list_cached_bags()`, `list_cached_assets()`,
`get_storage_summary()` (per-species breakdown),
`delete_cached_bag()`, `delete_cached_asset()`, an index-coherent
`clear_cache()`, and the existing `list_execution_dirs()` /
`clean_execution_dirs()`. All of these operate on the filesystem of
the machine where the **Python library** runs — the user's machine,
where datasets are downloaded, executions stage their outputs, and
`~/.deriva-ml/{hostname}/{catalog_id}/` lives.

A design exercise (2026-06-11) proposed mirroring this surface into
this plugin: `deriva://catalog/{h}/{c}/deriva-ml/storage/*` resources
plus `deriva_ml_delete_cached_bag` / `deriva_ml_delete_cached_asset`
/ `deriva_ml_clean_execution_dirs` tools, with a `server_local: true`
caveat on every payload acknowledging that MCP tools execute on the
**server's** filesystem.

Separately, the `manage-deriva-storage` skill in `deriva-ml-skills`
referenced `deriva://storage/summary`, `deriva://storage/cache`, and
`deriva://storage/execution-dirs` — resources that have never existed
in this plugin or in deriva-mcp-core.

## Decision

**No local-storage management surface in MCP — resources or tools.
The proposed design is rejected.**

The deciding fact: in the deployments this plugin targets, the MCP
server does **not** share a filesystem with the user's working
environment. The server runs remotely (or in a container); the cache
and execution working directories that matter to a user are on the
machine where their deriva-ml Python code runs. An MCP storage
surface would inspect and delete files on the *server's* disk — at
best meaningless to the user, at worst destructive to a shared
server's cache — while appearing to manage the user's own storage.
A `server_local: true` caveat does not fix a tool whose answer is
about the wrong machine; it documents the confusion.

The correct surface for storage management is the **skill layer**
(`deriva-ml-skills/manage-deriva-storage`), which runs Python
(`uv run python` / scripts) **on the user's machine** where the
filesystem is actually attached, calling the deriva-ml 1.46 API
directly.

## Consequences

- This plugin ships no `storage/*` resources and no cache/working-dir
  cleanup tools. Contributors tempted to add them should read this
  ADR first.
- `manage-deriva-storage` (deriva-ml-skills) is the canonical
  storage-management surface; its phantom `deriva://storage/*`
  references are removed in the same change that lands this ADR, and
  the skill states the local-only principle explicitly.
- The workspace-level `CLAUDE.md` carries a one-line cross-repo note
  pointing here, since the boundary involves three repos (deriva-ml
  owns the API, this plugin deliberately abstains, the skill is the
  user surface).

### Known tension (recorded, not resolved here)

Two existing MCP surfaces already touch the server's local cache:

- `deriva_ml_cache_dataset` (tools/dataset/complex.py) downloads a
  bag into the **server's** cache.
- `deriva_ml_bag_info` / the `dataset/{rid}/bag-preview` resource
  report `cache_status` / `cache_path` fields describing the
  **server's** cache.

These predate this ADR and are useful in server-local deployments
(stdio-launched servers sharing the user's machine) and harmless-ish
otherwise (a warmed server cache wastes server disk; a wrong
`cache_status` may mislead). They are out of this ADR's scope but
flagged: if the remote-server deployment model hardens further, they
deserve the same scrutiny — either removal or an explicit
server-side-cache framing.

## References

- deriva-ml cache-introspection API: deriva-ml PRs #286, #289, #290
  (v1.46.x); spec at
  `deriva-ml/docs/superpowers/specs/2026-06-11-cache-introspection-design.md`.
- The rejected design (for the record of what was considered): a
  five-resource + three-tool mirror with catalog-scoped URIs; its
  spec was withdrawn unmerged from this repo on 2026-06-11.
- `deriva-ml-skills/skills/manage-deriva-storage/SKILL.md` — the
  canonical storage-management surface.
