# Design discussion: `cache_dataset` / `denormalize_dataset` deprecation

**Status:** open question. No commits proposed in this doc.

**Origin:** 2026-05-23 audit (`docs/audit-2026-05-23.md`), Lens C
GRAY case 3b, plus a follow-up reframing after v0.5.0 removed the
8 execution-mutating tools.

**Audience:** decision-maker for the plugin's wire surface.

---

## TL;DR

By the same principle that drove the v0.5.0 execution-lifecycle
removal, **two more dataset tools fail the stateless rule**:

- `deriva_ml_cache_dataset` — materializes a bag to the **MCP
  server's** local disk, not the user's. HARD violation.
- `deriva_ml_denormalize_dataset` — materializes (bounded) rows
  server-side and returns them inline. Borderline; passes the
  bounded-resource rule today, fails on principle.

The question is whether the v0.5.0 principle should extend to bag
materialization too. **My read: yes for `cache_dataset`; keep
`denormalize_dataset` with caveats.** Detail below.

---

## What each tool does today

### `deriva_ml_cache_dataset`

Wraps `DerivaML.cache_dataset(spec)`, which downloads a bag from
the catalog to `~/.deriva-ml/bag_cache/` on the **machine running
the MCP server**.

Wire shape: client calls with `(hostname, catalog_id, dataset_rid,
version)`; server returns a `CacheDatasetResponse` carrying
`bag_info` (table list + byte totals) plus the on-disk
`bag_directory` path. Audit-flagged HARD violation in Lens C §1.5
GRAY case 3b.

Implementation: `tools/dataset/complex.py` (since the dataset/
package split).

### `deriva_ml_denormalize_dataset`

Wraps `DerivaML.get_denormalized_as_dict(spec, row_per=...)`,
which walks the FK graph of a cached bag to produce wide-flat rows
suitable for direct ML consumption. Returns rows **inline** in the
MCP response (no disk writes for the row materialization itself,
though it does call `cache_dataset` internally to ensure the bag
is local).

Wire shape: client calls with `(hostname, catalog_id, dataset_rid,
version, row_per, exclude_tables)`; server returns rows + a
`describe` block + truncation flag. 10× preflight guard fires
before materializing more than `_MAX_LIMIT × 10` rows.

Lens C §1.4 SOFT classification: "bounded today, watch the
upstream streaming behavior."

Implementation: `tools/dataset/complex.py`.

---

## Stateless-rule judgment

| Tool | What state lives where | Verdict |
|---|---|---|
| `cache_dataset` | Bytes on the MCP server's filesystem, indexed by RID + version. No user-side mirror. | **HARD violation.** The user cannot access the bytes. The `bag_directory` field in the response is a path on the server filesystem — useless to a remote client. |
| `denormalize_dataset` | Transient row materialization in server RAM (bounded by `_MAX_LIMIT × 10` preflight); transitively calls `cache_dataset` so the bag bytes ARE on server disk while denormalization runs. | **GRAY.** Passes the bounded-resource clause. Fails the no-server-side-storage clause indirectly (via the cached bag). But the OUTPUT — the rows — IS legitimately stateless. |

The asymmetric case for `denormalize_dataset` is what makes it
interesting: the **output** is the kind of thing MCP exists to
serve (structured data the LLM consumes directly). The dependence
on a server-side cache is an implementation detail of how the bag
pipeline works upstream, not an architectural commitment by the
plugin.

---

## Use cases worth weighing

### Who actually calls `cache_dataset` from the wire?

I can't think of a legitimate user-facing reason. The bag is on
the server; the user never sees the bytes. The `bag_info` data
(row counts, byte totals) is already covered by:

- `deriva_ml_bag_info` tool — same data, no materialization
- `deriva://catalog/.../ml/dataset/{rid}/bag-preview` resource —
  same data, snapshot form

If the user wants the bag locally, they run `ml.cache_dataset(...)`
in their own Python — that puts the bytes on **their** disk where
they can use them.

The only honest server-side use case I can think of: pre-warming
the server's cache for `denormalize_dataset`. If we remove
`denormalize_dataset` too, `cache_dataset` loses its last
internal use case.

### Who calls `denormalize_dataset`?

LLMs doing "give me the training data as rows so I can summarize
it" or "show me what features Image table joins to under the
default FK walk." Both legitimate. The deriva-ml-skills
`debug-bag-contents` skill is the primary consumer per the audit's
Lens B.

Without `denormalize_dataset`, the user has to:

1. Run `ml.cache_dataset(...)` locally (bag → local disk)
2. Run `ml.get_denormalized_as_dict(...)` locally (rows → memory)
3. Paste rows into the conversation OR ask the LLM to write code
   that processes them

That's more friction, but it's the same Python-only pattern that
`work-with-assets` already establishes for asset file I/O. The
LLM stays useful because it can write the Python; the user runs
it.

---

## Three options

### Option A — Remove both (full stateless purity)

Pros:
- Architecturally consistent with v0.5.0
- Eliminates the last two HARD/SOFT violations
- One clean story for skills to teach: "bag materialization is
  user-local Python; MCP gives you metadata only"

Cons:
- Loses inline denormalization-output convenience
- More friction for the "show me what's in this bag" LLM workflow
- Another wire-surface break (would be `v0.5.0`)

### Option B — Remove `cache_dataset` only; keep `denormalize_dataset`

Pros:
- Preserves the genuinely-useful inline output of denormalization
- Resolves the clear-cut HARD violation
- Denormalization has the 10× preflight guard already

Cons:
- `denormalize_dataset` transitively depends on `cache_dataset` —
  removing the tool but keeping the API call internally is fine
  in code, but the conceptual line gets fuzzy
- Skills documenting "cache first, then denormalize" become wrong
- Still a wire-surface break (`v0.5.0`)

### Option C — Keep both, document the constraint loudly

Pros:
- No wire-surface break
- Acknowledges the dev-mode use case (single-machine localhost,
  server == user)

Cons:
- Leaves the audit at "3 HARD + 2 SOFT" forever
- Sets a precedent that the stateless rule has exceptions for
  legacy tools, which weakens the rule for future tools
- A multi-tenant MCP deployment (the eventual target) silently
  breaks

---

## My recommendation: **Option B**

Remove `cache_dataset` from the wire. Keep `denormalize_dataset`
with two changes:

1. **Replace the `bag_directory` field in `denormalize_dataset`'s
   response** with a `note` that says the bag was materialized
   server-side and is not accessible to the caller. The rows are
   what's useful; the path isn't.
2. **Make explicit in the docstring** that the row materialization
   is bounded by `_MAX_LIMIT × 10` and that the call may take
   seconds-to-minutes the first time on a given bag (because of
   the internal `cache_dataset` call).

For `cache_dataset` removal, the deprecation path mirrors v0.5.0:

- Delete the tool from `tools/dataset/complex.py`
- Drop `deriva_ml_cache_dataset` from `_DATASET_TOOLS` in
  `test_plugin.py`
- Drop the `CacheDatasetResponse` / `CacheDatasetBagInfo` models
  from `_response_models.py`
- Update `_GETTING_STARTED_GUIDE` references (search-and-replace
  for `deriva_ml_cache_dataset`)
- Skill follow-up: `manage-storage`, `dataset-lifecycle`, and
  `debug-bag-contents` will need to teach the user-local Python
  pattern (`ml.cache_dataset(...)`). Add to the existing skills
  follow-up note in deriva-ml-skills.

Estimated effort: similar in shape to the v0.5.0 commit, but
much smaller — ~one file delete (or, more precisely, one tool
extraction), plus the response-model + test cleanup. ~2 hours
of focused work.

---

## What to decide

1. **Pick A, B, or C.**
2. If B: confirm the `bag_directory` field can be dropped from
   `denormalize_dataset`'s response without breaking a skill that
   relies on it. (Quick grep in deriva-ml-skills; unlikely to be
   referenced anywhere downstream of the LLM.)
3. If A or B: schedule the work as `v0.5.0`. Pair with one more
   pass to confirm no other HARD/SOFT violations sneak in (the
   audit was 2026-05-23; the surface has barely moved since).
4. If C: update the audit doc and CLAUDE.md to note the exception
   explicitly so future contributors don't have to re-derive the
   reasoning.

I'd lean strongly toward B. C is the worst because it codifies an
exception that will erode the rule. A is correct but loses the
genuinely-useful denormalize convenience for a comparatively small
purity gain.

---

## Cross-references

- `docs/audit-2026-05-23.md` § 1.4 (SOFT) and § 1.5 (GRAY)
- `CLAUDE.md` "Stateless / bounded-resource rule for MCP operations"
- v0.5.0 commit `2730a6f` (the execution-lifecycle removal that
  established the deprecation pattern)
- `docs/superpowers/notes/2026-05-23-skills-followups-from-mcp-audit.md`
  in `deriva-ml-skills` (will need an addendum if A or B lands)
