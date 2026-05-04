# MCP surface audit — 2026-05-04

Final-pass audit of deriva-ml-mcp v3.3.0 surface (49 tools, 13
resources, 2 prompts) using the `/grill-with-docs` skill in
caveman mode. Four lenses applied:

1. **ML developer perspective** — curator / catalog-evolver / model-dev
2. **Direct vs skill-mediated use** — does the tool make sense
   without a skill priming it
3. **MCP engineering + maintainability** — drift surfaces, helper
   conventions, file sizes, test coverage
4. **Pull-down candidates to deriva-ml** — operations that walk
   the deriva-ml domain model belong in the library first

Walking lens-first (one pass per lens, all artifacts visible each
time) per Q1 of the grilling.

---

## Decisions (rolling)

### L1 (ML developer) — coverage gaps

#### Q1.1: curator gaps

| Gap | Severity | Disposition |
|---|---|---|
| No `deriva_ml_create_asset_table` | medium | **Pull-down candidate.** Asset tables are deriva-ml-shaped; manual recipe currently in skill bodies. |
| No bulk-tag operation | low | Defer. `update_asset` per-RID is fine for typical curator scale; revisit if usage friction surfaces. |
| No "find untagged" filter | low | Defer. Workaround is `list_assets` + client-side filter. |
| Vocab term creation via tier-1 `add_term` | (correct by design) | Per ADR-0001 inheritance rule. |

#### Q1.2: catalog-evolver gaps

Most of this user's workflow is tier-1 territory (correct per
ADR-0001). The deriva-ml-mcp surface only owns ML-specific
structural concerns.

| Gap | Severity | Disposition |
|---|---|---|
| No "which datasets reference this table?" | medium | **Pull-down candidate.** Domain-specific (datasets are a deriva-ml abstraction; the reference graph is a deriva-ml query). Catalog-evolver about to drop a column needs this to know which datasets break. |
| No "which features reference this column?" | medium | **Pull-down candidate.** Same shape; about to drop column → which features lose a value column. |
| Asset-table shape validation | medium | Collapses into the `create_asset_table` pull-down from Q1.1. If create-method enforces the shape, validation is free. |
| No deregister for `add_dataset_element_type` | low | Defer. Narrow case, no current pain. |

#### Q1.3: model-developer gaps

| Gap | Severity | Disposition |
|---|---|---|
| No forward lineage ("find executions consuming RID X") | medium | **Pull-down candidate.** Natural complement to `lookup_lineage`. Answers the "safe to delete this dataset version?" question. Recommend separate method (different mental model + response-tree shape) rather than a `direction=` param on `lookup_lineage`. |
| No multirun-status summary | medium | **Pull-down candidate.** Counts by status across a workflow's executions. Cheap (one query w/ count aggregation). Today: `list_executions` returns full records → LLM counts client-side. |
| No execution-rank helper | (intentional) | Round 6 explicitly removed `rank_executions` from scope; standing by that decision. Manual 3-step pattern in `compare-model-runs` Phase 2A is well-served by existing tools. |
| No stale-config detector | low | Defer. Could be a `warn_if_outdated=True` param on `validate_execution_configuration`. |
| No way for MCP to inspect config files | (correct by design) | Local files; out of MCP scope. LLM reads file → passes dict to validate tools. |

#### Q1 summary — pull-down candidates from L1

| # | Candidate | Source persona | Severity |
|---|---|---|---|
| 1 | `create_asset_table` (with shape validation built in) | Curator | medium |
| 2 | `find_datasets_referencing(table_or_column)` | Catalog evolver | medium |
| 3 | `find_features_referencing(table, column)` | Catalog evolver | medium |
| 4 | `find_executions_consuming(rid)` (forward lineage) | Model dev | medium |
| 5 | Multirun-status summary | Model dev | medium |

Five candidates from L1. None blocking; all real ergonomic gaps.

### L2 (direct vs skill-mediated use)

#### Q2.1 — direct-use shape

- Zero tools mention skills in docstrings → tools are self-contained.
- Zero tools missing `Example:` blocks → project standard high.
- Floor for docstring length: 17 lines; all have Args/Returns/Raises/Example.
- Tiny nit: `deriva_ml_list_dataset_element_types` has an empty
  `Args:` block (no tool-specific params). Cosmetic; fix in
  cleanup pass.

#### Q2.2 — skills add real value (not duplication)

Skill layer earns its weight via:

1. Multi-tool orchestration recipes (e.g. `compare-model-runs`
   = 3 tools + Python aggregation; pattern can't live in a
   single tool docstring).
2. State-machine framing (e.g. `execution-lifecycle` walks
   Created → Running → Pending_Upload → Uploaded; tool
   docstrings name per-call pre/post conditions but the *arc*
   is skill-shaped).
3. Pattern-detection decision trees (e.g.
   `compare-model-runs` Phase 1 branches on metric-storage
   pattern).
4. Pitfall / project-shape framing.

**Residual duplication:** 19 of 27 tier-2 skills restate the
stateless-model boilerplate. Each instance ~1 line; aggregate
~19 lines. Already in always-on `deriva-ml-context`. Defer to
a future small slim pass.

#### Q2.3 — should tool docstrings reference skills?

**No.** Standing decision. Reasons:
- Cross-repo coupling (different release cycles).
- Non-Claude-Code clients don't have skills.
- Round 3's audit pattern: skills → tools, not tools → skills.
- Vocabulary (tools) shouldn't know about orchestration (skills).

### L3 (engineering + maintainability)

#### Q3.1 — `dataset/read.py` size

940 lines after Round 6 (+170 from `validate_*` tools). Project's
own historical split threshold: 1363 lines. **Defer split.** Not
at the threshold; splitting now would create
two-files-with-confusing-boundaries (validate tools' Pydantic
models live in deriva-ml's `aux_classes.py`, separate from any
deriva-ml-mcp file).

#### Q3.2 — `_impl` helper convention asymmetry

13 read tools have extracted `_impl` helpers (so tool/resource
share state and cannot drift). Zero mutate tools have helpers
(audit emission inline; resources are read-only anyway, so
mutate→resource path doesn't exist).

**Disposition:** Document the convention in deriva-ml-mcp's
CLAUDE.md as a contributor note. No test — the lint rule for
"extract a helper" is fuzzy (when is inlining OK?), high
false-positive risk. Pattern is well-followed; new tools
follow existing files as templates.

#### Q3.3 — test-frozenset pattern (`_DATASET_TOOLS`, `_ML_RESOURCE_URIS`, etc.)

**Keep as-is.** The frozenset is documentation; manual-update
discipline catches drift. Round 6 caught 2 missed-update bugs
this way. ~30 sec/tool burden is real but small. Auto-derived
alternative (b) would replace a clear failure mode with a more
opaque one.

### L4 (pull-down candidates, audit of existing tools)

L1 already surfaced 5 forward-looking pull-down candidates. L4
audits **existing** tool placement.

#### Q4.1 — fat-body tools have transport-shaped logic, not domain logic

10 tools have body > 50 lines after the docstring. Spot-checked
top three (`denormalize_dataset` 158, `list_feature_values` 148,
`update_dataset` 120). The bulk in each is:

- Pagination (cursor / `after_rid` / `preflight_count`) — MCP
  wire-protocol concern.
- Selector-string-to-callable translation (`"newest"` →
  `FeatureRecord.select_newest`) — MCP-side; ~5 lines.
- Mode switching (e.g. catalog-shape vs dataset-described in
  `denormalize_dataset`) — MCP convenience.

Verdict: **no missed pull-down candidates among existing fat-body
tools.** Logic is correctly placed at the transport layer.

#### Q4.2 — lineage resource hardcodes depth

The `ml/lineage/{rid}` resource always walks `depth=None,
max_executions=500`; the tool surfaces both as parameters. URI
templates can't carry per-call config cleanly — this asymmetry
is correct. The 500-execution cap protects against pathological
unbounded walks. No change.

#### Q4.3 — package-level placement

| File | Tool count | Lines | Verdict |
|---|---|---|---|
| `tools/asset.py` | 4 | 525 | OK |
| `tools/dataset/read.py` | 9 | 940 | OK (defer split per Q3.1) |
| `tools/dataset/mutate.py` | 7 | 734 | OK |
| `tools/dataset/complex.py` | 3 | 539 | OK |
| `tools/execution/read.py` | 6 | 694 | OK |
| `tools/execution/mutate.py` | 7 | 888 | OK |
| `tools/feature.py` | 6 | 756 | **Watch.** Split into `feature/{read,mutate}.py` if it grows by 2+ more tools. |
| `tools/maintenance.py` | 2 | 221 | OK |
| `tools/workflow.py` | 5 | 597 | OK |

`feature.py` is the only borderline; would benefit from a `read`/`mutate` split if it grows. Defer.

---

## Synthesis & disposition

### Pull-down candidates → deriva-ml (5, all forward-looking)

Bundled by query shape into three small focused rounds:

| Round | Tools | Estimated | Use case |
|---|---|---|---|
| **A — Asset-table creation** | `create_asset_table` w/ shape validation built-in | ~3-5 hr | Curator: replace the "tier-1 `create_table` + manual hatrac column shape" recipe currently in `work-with-assets` skill |
| **B — Schema-evolution impact** | `find_datasets_referencing(table_or_column)` + `find_features_referencing(table, column)` | ~3-5 hr | Catalog evolver: pre-change "what breaks if I drop this?" analysis. Same query shape → bundle. |
| **C — Execution introspection** | `find_executions_consuming(rid)` (forward lineage) + multirun-status summary | ~2-4 hr | Model dev: "safe to delete this dataset version?" + sweep status counts |

Each round follows the established pattern: design via `/grill-with-docs` against deriva-ml repo → implement + tests + Pydantic models → PR per the per-repo convention → bump deriva-ml minor. Then a follow-up round in deriva-ml-mcp to add the thin wrappers (~30 min per tool).

**Spawn-task chips queued** for rounds A, B, and C.

### MCP engineering — quick wins applied this round

- **QW1 ✅** — `list_dataset_element_types` empty `Args:` block fixed (replaced with explanatory comment).
- **QW2 ✅** — Helper-asymmetry note added to deriva-ml-mcp CLAUDE.md "Tool / Resource Dual-Mode Policy" section. The base policy was already there (pre-existing); this commit adds the read-vs-mutate asymmetry observation.

### MCP engineering — defer / watch

| Item | Disposition |
|---|---|
| `dataset/read.py` 940 lines | Defer split; threshold 1363 |
| `feature.py` 6 tools / 756 lines | **Watch.** Split `feature/{read,mutate}.py` if it grows by 2+ |
| ~19 skills repeating stateless boilerplate | Defer slim pass; small per-skill cost |
| Test-frozenset pattern | Keep as-is; manual-update discipline is the safety net |

### Standing decisions confirmed

- Tool docstrings do NOT reference skills (decoupling correct).
- Resources are read-only mirrors of tools via `_impl` helpers.
- Lineage resource hardcoded params correct (URI templates can't carry per-call config).
- `rank_executions` removal from Round 6 confirmed; manual 3-step pattern in `compare-model-runs` Phase 2A is well-served.

### Counts after this audit

- **Tools:** 49 (no change in count; QW1 cosmetic only — initial draft of this audit said 48, corrected to 49 during the parallel skills audit)
- **Resources:** 13 (no change)
- **Prompts:** 2 (no change; they survived Round 2's audit and remain correctly placed for cold-start orientation)
- **Pull-down candidates:** 5 forward-looking (rounds A/B/C, all queued)
- **Cross-cutting concerns:** 4 deferred (file-size watches, boilerplate slim, test-frozenset pattern, helper convention asymmetry — last one now documented)

### Quality bar achieved

- Every tool has Args / Returns / Raises / Example blocks.
- Zero tool docstrings reference skills.
- Every read tool with a sibling resource shares an `_impl` helper.
- Every PR adding a tool updates the registry-drift frozenset (caught in CI).

The deriva-ml-mcp surface is in a defensible state. The 5 forward-looking gaps are real ergonomic improvements but none are blocking.
