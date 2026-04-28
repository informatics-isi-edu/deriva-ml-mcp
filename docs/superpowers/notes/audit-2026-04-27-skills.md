# deriva-ml-skills Audit

**Date:** 2026-04-27
**Scope:** `deriva-ml-skills` (23-skill Claude Code plugin) against `deriva-ml-mcp` v3.0 tool/resource surface.

---

## Part A: Tool Coverage by Skills

### The 23 skills (one-line each)

**User-invocable (`/deriva-ml:<name>`):**

- `dataset-lifecycle` — End-to-end dataset operations: create, populate, split, version, browse, download BDBags.
- `debug-bag-contents` — Diagnose missing data in BDBag exports (FK traversal, missing tables, materialization issues).
- `execution-lifecycle` — Full execution lifecycle: pre-flight, create, start, work, commit/abort, nested, CLI.
- `troubleshoot-execution` — Recovery guide for stuck, failed, or incomplete DerivaML executions.
- `create-feature` — Feature design, creation, value insertion with provenance, selectors, browsing.
- `work-with-assets` — File asset discovery, download, upload, provenance, asset-type tagging.
- `manage-storage` — Local cache inspection, cleanup, pre-fetching, cache-vs-working-dir guidance.
- `configure-experiment` — Hydra-zen experiment project setup, config groups, DerivaModelConfig.
- `write-hydra-config` — Write and validate hydra-zen config files (DatasetSpecConfig, asset_store, builds).
- `new-model` — Scaffold a new DerivaML model function and wire it to configs and workflows.
- `model-development-workflow` — End-to-end project progression: schema → small dataset → dry run → production.
- `setup-notebook-environment` — Install Jupyter kernel, uv sync, nbstripout, Deriva auth for notebooks.
- `run-notebook` — Create, develop, and run DerivaML Jupyter notebooks with execution tracking.
- `route-run-workflows` — Router: configure, create notebooks, write configs, debug ML workflows.
- `route-project-setup` — Router: versions, environment setup, bag export troubleshooting, dev environment.
- `check-deriva-ml-versions` — Check and update the DerivaML ecosystem (lib, MCP plugin, skills plugin).
- `help` — General orientation for DerivaML, Deriva, and what Claude can help with.

**Always-on (auto-invoked):**

- `deriva-ml-context` — Plugin-wide steering: five abstractions, precedence principle, stateless model.
- `maintain-experiment-notes` — Silently append tacit decisions to `experiment-decisions.md` after any consequential action.
- `catalog-operations-workflow` — Steer toward committed Python scripts for catalog-modifying operations.
- `api-naming-conventions` — Reference for `lookup_` vs `find_` vs `list_` vs `create_` etc. method prefixes.
- `ml-data-engineering` — Get data OUT of DerivaML INTO ML pipelines: restructure, denormalize, DatasetBag.
- `generate-scripts` — Generate committed or ephemeral Python scripts for catalog operations.

**Note:** CLAUDE.md says 24 tier-2 skills in the Architecture section but the marketplace and skill directory both show 23. The discrepancy appears to be an off-by-one in the CLAUDE.md prose.

### Tool coverage matrix

**Dataset domain (17 tools)**

| Tool | Mentioned in skills |
|---|---|
| `deriva_ml_list_datasets` | api-naming-conventions, dataset-lifecycle, troubleshoot-execution |
| `deriva_ml_get_dataset` | api-naming-conventions, dataset-lifecycle, debug-bag-contents, execution-lifecycle, ml-data-engineering, model-development-workflow, troubleshoot-execution, write-hydra-config |
| `deriva_ml_list_dataset_members` | api-naming-conventions, dataset-lifecycle, debug-bag-contents, ml-data-engineering |
| `deriva_ml_list_dataset_relations` | api-naming-conventions, dataset-lifecycle |
| `deriva_ml_list_dataset_element_types` | api-naming-conventions, dataset-lifecycle |
| `deriva_ml_bag_info` | api-naming-conventions, dataset-lifecycle, debug-bag-contents, execution-lifecycle, manage-storage, ml-data-engineering, model-development-workflow, troubleshoot-execution, work-with-assets |
| `deriva_ml_get_dataset_spec` | api-naming-conventions, dataset-lifecycle, debug-bag-contents, model-development-workflow, troubleshoot-execution |
| `deriva_ml_create_dataset` | api-naming-conventions, dataset-lifecycle, deriva-ml-context, model-development-workflow, write-hydra-config |
| `deriva_ml_delete_dataset` | api-naming-conventions, dataset-lifecycle, debug-bag-contents |
| `deriva_ml_add_dataset_members` | api-naming-conventions, dataset-lifecycle, debug-bag-contents, deriva-ml-context, model-development-workflow, troubleshoot-execution |
| `deriva_ml_delete_dataset_members` | api-naming-conventions, dataset-lifecycle, debug-bag-contents |
| `deriva_ml_update_dataset` | api-naming-conventions, dataset-lifecycle |
| `deriva_ml_add_dataset_element_type` | api-naming-conventions, dataset-lifecycle, debug-bag-contents, model-development-workflow |
| `deriva_ml_increment_dataset_version` | api-naming-conventions, create-feature, dataset-lifecycle, debug-bag-contents, deriva-ml-context, model-development-workflow, troubleshoot-execution, write-hydra-config |
| `deriva_ml_cache_dataset` | deriva-ml-context, execution-lifecycle, manage-storage, model-development-workflow |
| `deriva_ml_denormalize_dataset` | api-naming-conventions, create-feature, dataset-lifecycle, debug-bag-contents, generate-scripts, ml-data-engineering, model-development-workflow |
| `deriva_ml_split_dataset` | create-feature, dataset-lifecycle, maintain-experiment-notes, ml-data-engineering, model-development-workflow, troubleshoot-execution, write-hydra-config |

**Execution domain (11 tools)**

| Tool | Mentioned in skills |
|---|---|
| `deriva_ml_list_executions` | api-naming-conventions, execution-lifecycle, troubleshoot-execution |
| `deriva_ml_get_execution` | api-naming-conventions, configure-experiment, execution-lifecycle, manage-storage, model-development-workflow, troubleshoot-execution, work-with-assets |
| `deriva_ml_find_workflow_executions` | api-naming-conventions, dataset-lifecycle, execution-lifecycle, model-development-workflow, work-with-assets |
| `deriva_ml_list_execution_children` | api-naming-conventions, configure-experiment, execution-lifecycle, model-development-workflow, troubleshoot-execution |
| `deriva_ml_list_execution_parents` | api-naming-conventions, configure-experiment, execution-lifecycle, model-development-workflow, troubleshoot-execution |
| `deriva_ml_create_execution` | api-naming-conventions, create-feature, dataset-lifecycle, deriva-ml-context, execution-lifecycle, manage-storage, model-development-workflow, troubleshoot-execution, work-with-assets |
| `deriva_ml_start_execution` | create-feature, dataset-lifecycle, deriva-ml-context, execution-lifecycle, model-development-workflow, troubleshoot-execution, work-with-assets |
| `deriva_ml_commit_execution` | create-feature, dataset-lifecycle, deriva-ml-context, execution-lifecycle, model-development-workflow, troubleshoot-execution, work-with-assets |
| `deriva_ml_update_execution` | api-naming-conventions, deriva-ml-context, execution-lifecycle, run-notebook, troubleshoot-execution |
| `deriva_ml_abort_execution` | create-feature, dataset-lifecycle, deriva-ml-context, execution-lifecycle, run-notebook, troubleshoot-execution, work-with-assets |
| `deriva_ml_create_execution_dataset` | api-naming-conventions, execution-lifecycle, troubleshoot-execution |
| `deriva_ml_add_nested_execution` | api-naming-conventions, execution-lifecycle, troubleshoot-execution |

**Feature domain (6 tools)**

| Tool | Mentioned in skills |
|---|---|
| `deriva_ml_list_features` | api-naming-conventions, create-feature, dataset-lifecycle, ml-data-engineering, troubleshoot-execution |
| `deriva_ml_get_feature` | api-naming-conventions, create-feature, model-development-workflow, troubleshoot-execution |
| `deriva_ml_list_feature_values` | api-naming-conventions, create-feature, ml-data-engineering, model-development-workflow |
| `deriva_ml_create_feature` | api-naming-conventions, create-feature, deriva-ml-context, maintain-experiment-notes, troubleshoot-execution |
| `deriva_ml_delete_feature` | api-naming-conventions, create-feature |
| `deriva_ml_add_feature_values` | api-naming-conventions, create-feature, deriva-ml-context, execution-lifecycle, model-development-workflow |

**Workflow domain (5 tools)**

| Tool | Mentioned in skills |
|---|---|
| `deriva_ml_list_workflows` | api-naming-conventions, configure-experiment, execution-lifecycle |
| `deriva_ml_get_workflow` | api-naming-conventions |
| `deriva_ml_find_workflow_by_url` | api-naming-conventions, deriva-ml-context, execution-lifecycle, troubleshoot-execution |
| `deriva_ml_create_workflow` | api-naming-conventions, create-feature, dataset-lifecycle, deriva-ml-context, execution-lifecycle, troubleshoot-execution, write-hydra-config |
| `deriva_ml_update_workflow` | api-naming-conventions, execution-lifecycle |

**Asset domain (4 tools)**

| Tool | Mentioned in skills |
|---|---|
| `deriva_ml_list_asset_tables` | api-naming-conventions, deriva-ml-context, work-with-assets |
| `deriva_ml_list_assets` | api-naming-conventions, work-with-assets |
| `deriva_ml_lookup_asset` | api-naming-conventions, dataset-lifecycle, deriva-ml-context, execution-lifecycle, model-development-workflow, work-with-assets, write-hydra-config |
| `deriva_ml_update_asset` | api-naming-conventions, deriva-ml-context, work-with-assets |

**Maintenance domain (2 tools)**

| Tool | Mentioned in skills |
|---|---|
| `deriva_ml_reindex_vocabularies` | **NONE** |
| `deriva_ml_resync_indexes` | **NONE** |

**Resources (11 resources)**

All 11 resources under `deriva://catalog/{h}/{c}/ml/...` are referenced in the skills — the execution-lifecycle and dataset-lifecycle skills both list the resource URIs explicitly. The `registries` resource is specifically called out in `execution-lifecycle` SKILL.md.

### Orphan tools (no skill mentions)

- **`deriva_ml_reindex_vocabularies`** — Zero skill mentions. This tool is needed after adding/removing vocabulary terms via `add_term`/`delete_term`. The gap matters because skills do teach adding vocabulary terms (via the generic `add_term` call) but never mention that a `derive_ml_reindex_vocabularies` call is needed afterward.
- **`deriva_ml_resync_indexes`** — Zero skill mentions. This is the documented cross-user freshness bridge (v1.4). The `CLAUDE.md` for `deriva-ml-mcp` explicitly calls this out as the manual bridge for cross-user staleness, but no skill surfaces it to the LLM.

### Orphan skills (no tool calls)

All 23 skills either reference `deriva_ml_*` tools directly, or are router/context/reference skills that intentionally contain no tool calls:

- `route-run-workflows` — Pure router, delegates to other skills. No tool calls is correct.
- `route-project-setup` — Pure router. No tool calls is correct.
- `help` — Orientation skill. No tool calls is correct.
- `setup-notebook-environment` — Environment setup; uses shell commands, not `deriva_ml_*` tools. Correct.
- `check-deriva-ml-versions` — Calls Python version check script, not MCP tools. Correct.
- `api-naming-conventions` — Reference card, no workflow. Correct.

No skills are orphaned in the problematic sense.

---

## Part B: Workflow Coverage

### Workflow 1: Create dataset from catalog data

**Status: Covered end-to-end.**

`dataset-lifecycle` SKILL.md covers: check element types (`deriva_ml_list_dataset_element_types`), add element type if missing (`deriva_ml_add_dataset_element_type`), create execution, `deriva_ml_create_dataset`, `deriva_ml_add_dataset_members` with both `member_rids` and `members_by_table` forms, then `deriva_ml_increment_dataset_version`. The references directory has detailed workflow.md and concepts.md with concrete examples. The `derive-ml-context` also mentions the key tools.

**Minor gap:** No explicit guidance on when to call `add_term` to add a new `Dataset_Type` before creating the dataset, or that `deriva_ml_reindex_vocabularies` should follow.

### Workflow 2: Register a new model architecture as a workflow

**Status: Partially covered.**

`execution-lifecycle` covers `deriva_ml_create_workflow` with dedup behavior and mentions `deriva_ml_find_workflow_by_url` for pre-check. The `new-model` and `configure-experiment` skills cover the Python-side scaffold. However, the skills do not guide the LLM on computing the Git URL and checksum locally (the "Boundary rule" in the MCP source states "the caller computes the Git URL, checksum, and version locally and passes them in — the MCP server never runs git introspection"), so there is no end-to-end example of `git rev-parse HEAD` → URL construction → `deriva_ml_create_workflow` call. `deriva_ml_update_workflow` is mentioned only in `api-naming-conventions` and `execution-lifecycle` without meaningful context.

**Missing:** Concrete recipe for computing Git checksum and URL from a local repo before registering.

### Workflow 3: Running a training execution

**Status: Covered end-to-end.**

`execution-lifecycle` covers all six phases: pre-flight (validate RIDs, check cache, stage data), create execution, start, do work (Python API for I/O), commit or abort. The three paths (MCP tools, Python API, CLI) are all documented. `troubleshoot-execution` covers recovery paths. Code examples exist in both SKILL.md and the references directory.

### Workflow 4: Defining and recording features (model evaluation outputs)

**Status: Covered end-to-end.**

`create-feature` covers: semantic-search-first discovery (`rag_search`), deciding whether a feature is needed vs a column, designing term/value/asset columns, `deriva_ml_create_feature`, `deriva_ml_get_feature` for schema inspection, `deriva_ml_add_feature_values` with the hybrid dispatch (Created vs Running state), and querying with selectors via `deriva_ml_list_feature_values`. The references directory has detailed concepts.md and feature-selectors.md.

### Workflow 5: Comparing model runs

**Status: Partially covered.**

`execution-lifecycle` and `model-development-workflow` both mention `deriva_ml_find_workflow_executions` and `deriva_ml_list_executions` with status filtering. `create-feature` covers querying feature values per execution with `selector="by_execution"`. However, there is no skill that ties these together into a comparison workflow: "find executions of workflow X, for each execution get feature values with `selector='by_execution'`, compare metrics." The pieces exist in separate skills; an LLM doing a first-time comparison would need to compose from multiple skill loads.

**Missing:** A dedicated "compare runs" section in either `create-feature` or `execution-lifecycle` that shows the end-to-end query pattern.

### Workflow 6: Caching a dataset locally for offline training

**Status: Covered end-to-end.**

`execution-lifecycle` (pre-flight section), `manage-storage`, and `model-development-workflow` all cover `deriva_ml_cache_dataset`. The `manage-storage` skill gives the full pre-flight sequence (bag_info → cache if needed → verify materialized). Cache status values (`not_cached`, `cached_metadata_only`, `cached_materialized`, `cached_incomplete`) are documented.

**Minor v3.0 gap:** `manage-storage` SKILL.md line 82 mentions `cache_path` as a top-level field: `- \`cache_path\`: where it lives on disk`. In v3.0, bag-info keys were moved under `bag_info` nested key for `cache_dataset`, but `deriva_ml_bag_info` (read tool) still returns `cache_path` top-level — this may be correct for that tool. Needs verification.

### Workflow 7: Splitting a dataset for train/test/val

**Status: Covered end-to-end.**

`dataset-lifecycle` covers `deriva_ml_split_dataset` with all major parameters: `test_size`, `val_size`, `stratify_by_column`, `dry_run`, element table auto-detection. Stratified split guidance is present. The `write-hydra-config` skill also mentions split as a tracked operation.

### Workflow 8: Recovering from a failed execution

**Status: Covered.**

`troubleshoot-execution` has a dedicated "I Need to Resume an Aborted Execution" section. It explicitly documents that `restore_execution` has no equivalent and provides the workaround: inspect prior execution → create fresh execution with same config → link via description. The SKILL.md also covers "Execution Stuck in Running" and "Files Not Uploaded" scenarios.

**Bug:** Line 129 of `troubleshoot-execution/SKILL.md` says `deriva_ml_update_execution(hostname, catalog_id, execution_rid, status="Failed", message="Manually marked as failed")`. This is wrong — `deriva_ml_update_execution` only accepts `description`, not `status` or `message`. Status changes must go through `abort_execution` or `commit_execution`. The same incorrect signature appears in `execution-lifecycle/references/workflow.md`, `run-notebook/references/workflow.md`, `execution-lifecycle/references/concepts.md`, and `api-naming-conventions/SKILL.md`.

### Workflow 9: Cross-user dataset visibility (resync_indexes)

**Status: Not covered.**

`deriva_ml_resync_indexes` is mentioned nowhere in the skills. The `deriva-ml-mcp` CLAUDE.md explicitly documents this as the v1.4 bridge for cross-user freshness (when user B needs to see user A's mutations). There is no skill that tells the LLM when or how to call it. The only related text is in the `CLAUDE.md` workspace notes, which Claude does not load in normal conversation.

This is the most significant user-facing gap: a user experiencing "I can't see the dataset my colleague just created" has no skill-level guidance to reach for `deriva_ml_resync_indexes`.

### Workflow 10: Exploring an unfamiliar catalog

**Status: Partially covered.**

`deriva-ml-context` sets up the five abstractions and provides pointers to domain skills. The `help` skill covers orientation. The `ml-data-engineering` skill covers `deriva_ml_denormalize_dataset` for wide-table exploration. The skills recommend `rag_search` as the discovery mechanism.

**Missing:** No explicit "here is how to start if you know nothing about this catalog" walkthrough. The canonical discovery order (read `deriva://catalog/{h}/{c}/ml/registries` → browse workflows → look at datasets → explore features on tables) is mentioned in individual skills but never synthesized into a first-visit recipe. The `help` skill could carry this.

### Workflow coverage summary

- 4/10 fully covered: workflows 3 (training execution), 4 (features), 6 (caching), 7 (splitting).
- 4/10 partially covered: workflows 1 (dataset creation — minor vocab gap), 2 (workflow registration — missing Git URL recipe), 5 (comparing runs — missing synthesis), 10 (catalog exploration — missing first-visit recipe).
- 2/10 not covered: workflow 8 is largely covered but has a concrete API bug; workflow 9 (cross-user visibility) has zero coverage.

---

## Part C: Skill Design Quality

### Granularity

The 23 skills are well-sized overall. A few observations:

- **`execution-lifecycle`** is large (the SKILL.md alone is ~180 lines plus 3 reference files). It could be split into "running-mcp-executions" and "running-cli-executions" but the current structure with references works reasonably well because the primary SKILL.md stays focused on decision points.
- **`route-run-workflows`** and **`route-project-setup`** are thin routers that overlap slightly. Both mention version checks, bag export troubleshooting, and environment setup. A user asking "how do I get started" might hit either one. The routing logic in descriptions could be tighter.
- **`api-naming-conventions`** is always-on and comprehensive. The risk is that it duplicates information already in domain skills (the `lookup_` vs `find_` table is useful but large for an always-on context).
- **`generate-scripts`** and **`catalog-operations-workflow`** partly overlap: both steer toward Python scripts over interactive tool use. The boundary (`generate-scripts` generates the script text; `catalog-operations-workflow` teaches when to use that pattern) is logical but may not be clear from descriptions alone.

### Discoverability (frontmatter descriptions)

Most descriptions are strong and follow the "ALWAYS use when X, triggers on: Y" pattern. Specific concerns:

- **`deriva_ml_get_workflow`** has minimal skill coverage (only `api-naming-conventions` mentions it). An LLM asked "show me the details of workflow 1-WF" has no skill that explicitly teaches when to use this tool versus reading the resource URI. This is a coverage gap, not a discoverability gap, but it manifests because no skill uses `deriva_ml_get_workflow` as its primary teaching example.
- **`troubleshoot-execution`** has `user-invocable: false` and `disable-model-invocation: true`. The description says "ALWAYS use when a DerivaML execution fails." These flags are contradictory for an "ALWAYS use" skill — if model invocation is disabled, the LLM must explicitly load it. Check that the flags are correct.
- **`debug-bag-contents`** description is appropriately scoped to export issues. However it is not in the "user-invocable" list in the CLAUDE.md Architecture section (it only lists user-invocable skills by name, and debug-bag-contents is not there). The frontmatter does not have `user-invocable: false`, so it should be user-invocable — CLAUDE.md's omission is just a documentation gap.
- **`help`** description says "ALWAYS prefer this skill for general 'what/how/why' questions about the DerivaML ecosystem before routing to more specific skills." The word "ALWAYS" combined with a broad trigger set risks over-triggering; it may fire on queries that should go directly to `execution-lifecycle` or `dataset-lifecycle`.

### Cross-references

Cross-references are strong. Domain skills consistently point to related skills:

- `execution-lifecycle` references `dataset-lifecycle`, `create-feature`, `ml-data-engineering`, `configure-experiment`, `write-hydra-config`, `run-notebook`.
- `create-feature` references `dataset-lifecycle` and `execution-lifecycle`.
- `troubleshoot-execution` correctly separates tier-2 errors from tier-1 (generic catalog) errors with explicit `/deriva:troubleshoot-deriva-errors` references.
- Tier-1 references use the `(tier-1; deriva-skills)` annotation convention as documented in CLAUDE.md.

The main gap: no skill cross-references the maintenance tools (`reindex_vocabularies`, `resync_indexes`), so the cross-reference web does not surface them even indirectly.

### Code examples

Code examples are present throughout and generally concrete. The `execution-lifecycle` skill has especially strong examples covering all three paths (MCP tools, Python API, CLI). The `create-feature` and `dataset-lifecycle` skills both include realistic tool calls with actual RID shapes.

One concern: `manage-storage/SKILL.md` line 184 shows `deriva_ml_cache_dataset(hostname=..., asset_rid="3WSE")` — this tool does not have an `asset_rid` parameter (it has `dataset_rid`). This is an incorrect example.

### Failure-mode coverage

Failure modes are well-covered for the main workflows. `troubleshoot-execution` is dedicated to failure recovery. `execution-lifecycle` has a "Critical Rules" section. `dataset-lifecycle` mentions what happens when element types are not registered. `create-feature` covers the case where a feature already exists.

The main failure-mode gap is in `work-with-assets`: the skill covers discovery, download, and upload at a high level but does not give concrete recovery guidance for common asset upload failures (timeout, permission denied on hatrac, asset already exists at path).

### v3.0 API drift risk

Three concrete v3.0-era drift issues found:

1. **`status="success"` in `deriva_ml_denormalize_dataset` example** — `ml-data-engineering/references/denormalize-guide.md` line 91 shows `"status": "success"` in the response JSON for `deriva_ml_denormalize_dataset`. This tool's response does NOT have a `status` field at all (it uses `mode` as the top-level discriminator). This is a phantom field that does not exist in the actual v3.0 wire shape.

2. **`deriva_ml_update_execution` accepts `status` and `message` (false)** — Multiple skills document the call signature `deriva_ml_update_execution(hostname, catalog_id, execution_rid, status="Failed", message="...")`. This is incorrect: the tool only accepts `description`. Files affected:
   - `troubleshoot-execution/SKILL.md:129`
   - `execution-lifecycle/references/workflow.md:38, 123`
   - `execution-lifecycle/references/concepts.md:114, 120`
   - `run-notebook/references/workflow.md:262`
   - `api-naming-conventions/SKILL.md:179`

   This is the highest-severity drift issue: an LLM following this guidance will call the tool with the wrong parameters and get an error.

3. **`cache_path` as top-level field** — `manage-storage/SKILL.md:82` lists `cache_path` as a direct response field of `deriva_ml_bag_info`. In v3.0, `cache_path` moved under the `bag_info` sub-object for `derive_ml_cache_dataset`. For `derive_ml_bag_info` (the read tool), `cache_path` may still be top-level — the source code shows `return json.dumps(info, default=str)` which passes through whatever `ml.bag_info()` returns. This should be verified against the actual `bag_info` return shape, but may be correct for the read tool.

---

## Part D: deriva-ml-context skill

**File:** `/Users/carl/GitHub/DerivaML/deriva-ml-skills/skills/deriva-ml-context/SKILL.md`

### Steering principle coverage

The skill correctly carries the load-bearing principle: "In a deriva-ml-loaded catalog you must use the deriva-ml abstractions for them — the `deriva_ml_*` MCP tools listed above and the deriva-ml Python API — NOT the raw `insert_entities` / `update_entities` / `get_entities` core tools from `deriva-mcp-core`."

It enumerates six concrete reasons why raw primitives are wrong (business logic bypass, FK validation, provenance tracking, version management, RAG re-indexing, audit emission). This is thorough and actionable.

### Structure for LLM internalization

The skill is well-structured for an always-on context:

- Opens with a concise "What is DerivaML?" paragraph.
- Presents the five abstractions in a scannable table with primary skill pointers and key MCP tools per abstraction.
- Explains the stateless model (always pass hostname= and catalog_id=).
- Delivers the steering principle with a concrete enumeration of what gets bypassed.
- Explains how to extend built-in vocabularies (the `add_term` generic approach since dedicated extender tools were removed).
- Closes with a "when to reach back to raw catalog surface" section that draws the tier-1/tier-2 boundary cleanly.

### Observations

- The `disable-model-invocation: false` flag is correct for an always-on skill.
- The trigger set in the description is appropriately broad: it matches on 'derivaml', 'deriva-ml', 'dataset', 'workflow', 'execution', etc.
- One potential issue: the table lists `deriva_ml_add_feature_value` (singular) in the Feature row, but the correct tool name is `deriva_ml_add_feature_values` (plural). This is a minor naming inconsistency but unlikely to cause runtime errors since the detailed instructions in `create-feature` use the correct plural form.
- The skill correctly notes that the 24-skill count in CLAUDE.md is inconsistent with the actual 23 skills (implied by stating "~24 skills").

Overall, `deriva-ml-context` is the most carefully written skill in the plugin. It delivers its steering purpose clearly and would be internalized effectively by an LLM.

---

## Top-line findings

1. **Two maintenance tools are completely invisible.** `deriva_ml_reindex_vocabularies` and `deriva_ml_resync_indexes` have zero skill coverage. The cross-user freshness gap (workflow 9) is the most impactful consequence: users experiencing stale RAG views from collaborator edits have no documented path to resolution. Both tools need at least a single skill mention with a trigger scenario.

2. **`deriva_ml_update_execution` is documented with a false signature across at least 5 skill files.** The skills claim it accepts `status=` and `message=` parameters; the actual tool only accepts `description=`. Any LLM following this guidance will generate a failing tool call. This is the highest-severity correctness bug in the skill set.

3. **The cross-user workflow coverage gap is total.** Workflow 9 (cross-user visibility / `resync_indexes`) is taught nowhere. The `deriva-ml-mcp` CLAUDE.md documents it as a known limitation with a manual bridge — that bridge needs a skill or at least a mention in `execution-lifecycle` or `troubleshoot-execution`.

4. **Core workflow coverage is solid.** 8 of 10 workflows have meaningful coverage (4 fully, 4 partially). The execution lifecycle, feature lifecycle, dataset operations, and caching workflows are particularly well-documented with concrete examples and cross-skill reinforcement.

5. **`deriva-ml-context` is the strongest skill and correctly carries its load-bearing role.** The steering principle (DerivaML abstractions > raw primitives), the stateless model explanation, and the vocabulary extension pattern are all present and accurate. This skill alone prevents the most common LLM mistake (using raw entity CRUD for ML domain objects).
