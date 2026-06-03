# Skills → MCP-surface migration audit (2026-06-03)

> **PRIMARY GOAL (revised):** maximize MCP capability in a **chatbot environment
> with NO access to skills**. In that setting the MCP's prompts, resources, tool
> docstrings, and the RAG corpus are the *only* guidance the model ever gets.
> See the "SKILLS-LESS CHATBOT" section at the bottom — it supersedes the
> narrower dedup framing below where the two differ.

**Question:** Are there elements of `deriva-ml-skills` that should be migrated or
duplicated into the MCP's always-on surfaces (prompts, tool docstrings, resources)?

**Method:** Inventoried the MCP's current model-visible surface (prompts, 18
resources, 44 tool docstrings, `coverage.md`), extracted the load-bearing rules
from the meta-skills (`using-deriva-mcp`, `deriva-ml-context`,
`api-naming-conventions`, `help`, `browse-erd`) and the operational skills
(dataset/execution/feature/asset/troubleshoot/bag), then **deduplicated** every
candidate against what the MCP already exposes.

**Headline:** The MCP surface is already strong. Most knowledge the skills teach
is *already present* in the prompts/docstrings — the getting-started + concepts
prompts cover the (hostname,catalog_id) rule, pagination contract, verb
conventions, five domains, resource-first rule, inheritance-with-override,
mutation chain, asset metadata/fileIO split, `rag_search` doc_type values, and
name-resolution. The dataset/asset/feature tool docstrings already carry the
version/release semantics, `asset_types` membership rule + Input/Output_File
tags, and `partition_by`/denormalize semantics. **Do not re-add these.**

A small number of genuine gaps remain — verified absent from the live surface.

---

## CONFIRMED GAPS (real migration candidates)

### G1 — `cite_url` rendering guidance is absent (recommend: PROMPT)
**Status:** The `cite_url` *field* is returned by the dataset/execution/asset
resources and tool payloads (`_helpers.py`, `tools/*/read.py`), but **no
model-visible surface tells the model to render RIDs as Markdown links to
`cite_url`**, nor explains the snapshot-pinned-vs-live distinction or the
`/id/...` resolver-vs-`/chaise/...` distinction.
**Source:** `deriva-ml-context` (cite_url rendering rule) + `browse-erd`
reference (`/id/` for stored content vs `/chaise/` for UI nav).
**Why it matters:** Without it, the model presents bare RID strings to users
(unclickable, non-citable) and, when *writing* RIDs into catalog content
(descriptions/annotations), may use a `/chaise/` UI URL that breaks on UI
changes instead of the durable `/id/` resolver form.
**Recommendation:** Add a short "PRESENTING & STORING RIDs" section to the
getting-started prompt: (a) render RIDs from resource/tool payloads as
`[RID](cite_url)`; (b) released-dataset cite_urls are snapshot-pinned, others
are live; (c) when storing an RID reference *in* the catalog, use the `/id/`
resolver form, never a `/chaise/` URL.

### G2 — feature-selector exclusivity is implicit, not explicit (recommend: TOOL DOCSTRING)
**Status:** `deriva_ml_list_feature_values` has `selector`, `selector_workflow`,
`selector_execution_rid` params; the docstring says "Requires
`selector_workflow`" / "Requires `selector_execution_rid`" but does **not**
state the three are **mutually exclusive — pass exactly one**.
**Source:** `create-feature/references/feature-selectors.md`.
**Why it matters:** A model could pass `selector="select_newest"` *and*
`selector_workflow=...` expecting an intersection; the actual behavior is
undefined/last-wins. One sentence prevents it.
**Recommendation:** Add to the `deriva_ml_list_feature_values` docstring:
"`selector`, `selector_workflow`, and `selector_execution_rid` are mutually
exclusive — supply at most one." (Custom callables remain Python-API-only, which
is already implied.)

### G3 — `ListMcpResourcesTool` returns concrete resources only, not templates (recommend: PROMPT, one line)
**Status:** The getting-started prompt lists the `deriva://catalog/.../deriva-ml/...`
resource *templates* in a table, but does not warn that
`ListMcpResourcesTool` enumerates only the ~3 static resources, NOT the
catalog-scoped templates — which must be read directly with `ReadMcpResourceTool`.
**Source:** `using-deriva-mcp` (the ListMcpResources-vs-templates note).
**Why it matters:** A model that "checks what resources exist" via
`ListMcpResourcesTool` sees 3 entries and wrongly concludes the per-RID catalog
resources don't exist — then falls back to `query_attribute` (the anti-pattern
the prompt elsewhere warns against). (Observed first-hand: this session hit
exactly this confusion.)
**Recommendation:** One line in the getting-started "read-side / resources"
section: "`ListMcpResourcesTool` lists only the static resources; the
`deriva://catalog/.../deriva-ml/...` templates above are not enumerated — construct
the URI and read it directly with `ReadMcpResourceTool`."

---

## NON-GAPS (verified already covered — do NOT migrate)

- **(hostname, catalog_id) rule, pagination contract, error envelope, verb
  conventions, five domains, the menu/tool-count** — in getting-started prompt.
- **find_\* vs list_\* naming, inheritance-with-override, built-in vocabularies,
  entity-resolution workflow, rag_search doc_type values** — in prompts.py
  (13 `find_` mentions; `catalog-schema`/`catalog-data`/`ml-docs` enumerated at
  prompts.py:449-451, 823-824).
- **Release-before-consume / dev-version-not-pinnable** — in dataset tool
  docstrings (`tools/dataset/mutate.py`, `read.py`).
- **`asset_types` membership-not-equality + auto Input_File/Output_File tags** —
  in `tools/asset.py`.
- **`partition_by` element-vs-row + multi-value denormalize row-multiplication** —
  in `tools/dataset/complex.py`.
- **create_workflow dedup on URL+checksum; caller computes git locally** — in
  `tools/workflow.py` docstring.

## CORRECT BOUNDARIES (absent by design — do NOT add)

- **`DerivaMLDirtyWorkflowError` / git-dirty checks** are absent from MCP
  `create_workflow` — and should be. The MCP tool does no git introspection
  (caller computes URL/checksum locally, per `tools/workflow.py:12-13`); the
  dirty-tree check lives in the local-Python execution path. Belongs in the
  execution-lifecycle skill, not the MCP.
- **Execution lifecycle (create/start/commit/abort), feature-value writes, asset
  file I/O, bag materialization** — deliberately removed/deferred per the
  stateless rule (`coverage.md`). The "this is local-Python, not MCP" boundary is
  already stated in the getting-started prompt's MUTATION section. Keep in skills.
- **Hydra-zen config templates, CLI/ERD launch steps, data-container ("carry
  structure") guidance, parameter-naming style** — workflow/task knowledge;
  correctly skill-only.

## PROCESS NOTE — keep the SYNC contract honest

`deriva-ml-context/SKILL.md` carries a SYNC comment: it mirrors `_CONCEPTS_GUIDE`
in `prompts.py`; "when the conceptual core changes, update both." That contract
covers the *concepts* body. The three gaps above (G1–G3) are **not** part of that
mirrored body — they're MCP-surface operational rules currently living only in
`using-deriva-mcp` / `deriva-ml-context` / `create-feature`. If G1–G3 land in the
prompt/docstrings, fold them under the same sync discipline so the skill and MCP
don't re-diverge.

---

## Recommended actions (small, additive — each its own PR)

1. **G1** → add "PRESENTING & STORING RIDs" section to the getting-started
   prompt (and mirror into `deriva-ml-context` per the sync contract).
2. **G2** → one sentence in `deriva_ml_list_feature_values` docstring.
3. **G3** → one line in the getting-started prompt's resources section.

All three are documentation-only, additive, and low-risk. They close the
highest-value model-misuse gaps without duplicating the (already strong)
existing surface.

---

# SKILLS-LESS CHATBOT — the primary lens (revised goal)

When the goal is "maximize capability in a chatbot that has no skills," the
classification changes. Skill knowledge falls into three buckets:

1. **MCP-surface rules** — needed regardless of environment. (G1–G3 above.)
2. **Workflow how-to** (the execution `with` pattern, asset file I/O, dataset
   lifecycle, hydra-zen configs) — the chatbot can't *execute* these (they're
   local Python), but it MUST be able to *explain them to the user*.
3. **Local-Python-only mutations** (create/commit/abort execution, add_features,
   asset upload) — correctly absent as tools; a chatbot can't run them anyway.

The architectural cut (MCP = observe + simple mutate; skills/local-Python =
orchestrate + generate code) is **correct and should stay**. The problem in a
skills-less chatbot is not missing tools — it is **cold trails**: the MCP
repeatedly says "see the X skill" for bucket-2 knowledge the chatbot cannot
reach, instead of routing the model to a source it CAN reach.

## The key fact that makes this cheap to fix

**The how-to corpus already exists and is already RAG-indexed.**
`deriva-ml/docs/user-guide/` contains `executions.md`, `datasets.md`,
`features.md`, `denormalization.md`, `offline.md`, `reproducibility.md`,
`sharing.md`, `hydra-zen.md`, `exploring.md`, `migration.md` — i.e. the
action-level guidance the skills teach. The entire `deriva-ml` repo root is
indexed as `doc_type="ml-docs"` (`resources/rag.py`). So the *substance* is
reachable by a skills-less chatbot via `rag_search(doc_type="ml-docs")`.

**The gap is routing, not content.** The getting-started prompt mentions
rag_search-for-how-to only once, in passing, scoped to execution work
(`prompts.py:542`). Every "see the X skill" pointer is a dead end for a chatbot;
none of them say "...or rag_search `doc_type=ml-docs` for `<topic>`."

## CHATBOT GAPS (in priority order)

### C-G1 — Position rag_search as the skills substitute (highest leverage; PROMPT)
Add an explicit section to getting-started: **"NO SKILLS? USE THE DOCS."**
State plainly: when you need how-to/workflow guidance that isn't in a tool
docstring or this prompt — how to run an execution, the `with
ml.create_execution(...) as exe:` pattern, registering/downloading asset bytes,
the dataset version lifecycle, writing a hydra-zen config, offline mode — call
`rag_search(query="<topic>", doc_type="ml-docs")`. Name the indexed user-guide
pages so the model knows what's retrievable. This single addition converts every
cold "see the skill" trail into a reachable path.

### C-G2 — Turn every "see the X skill" pointer into a skill-OR-rag_search pointer (PROMPT + tool docstrings)
At each dangling reference, append the reachable alternative. Concretely:
- `prompts.py:~554,726,958` (execution-lifecycle skill) → "...or
  `rag_search('execution lifecycle', doc_type='ml-docs')` (see
  `user-guide/executions.md`)."
- `prompts.py:~984-1003`, `tools/asset.py:~19`, `tools/execution/__init__.py:~8`
  (work-with-assets skill) → add an asset-I/O how-to pointer. **Caveat:** confirm
  an asset file-I/O page exists in the indexed corpus; `user-guide/` has no
  `assets.md`. If absent, that how-to is genuinely unreachable by a chatbot —
  either (a) add a short `user-guide/assets.md` to the deriva-ml repo (it gets
  indexed automatically), or (b) inline a brief "to upload/download asset bytes,
  the user runs this Python locally: ..." snippet in the asset tool docstrings.
- `prompts.py:~663` (`work-with-executions` skill) → this skill does not exist
  yet; drop the aspirational pointer or replace with the rag_search route.

### C-G3 — Make the local-Python boundary self-explaining, not skill-delegating (PROMPT/docstrings)
Where the MCP correctly lacks a mutation (create_execution, add_features, asset
upload), the chatbot should be able to hand the USER the local-Python pattern,
not say "use a skill." The canonical `with ml.create_execution(...) as exe:
exe.add_features(...); ... ; exe.commit_output_assets()` snippet is small and
stable. Recommend embedding the *minimal* canonical snippet (or a tight pointer
to the rag_search page that contains it) directly in the MUTATION section of
getting-started, so a skills-less chatbot can explain "here's the code to run
locally" instead of dead-ending.

### C-G4 — Capability/"what can I do here" menu is adequate but not action-oriented (optional; PROMPT)
The five-domains menu lists tools but not *tasks*. A skills-less chatbot fielding
"how do I set up a new ML project?" / "how do I run an experiment?" has no
task-level catalog (that lived in the `help` / `setup-derivaml-project` skills).
Optional enhancement: a short "COMMON TASKS → where to look" table in
getting-started mapping intents (set up project, define schema, organize data,
run experiment, troubleshoot) to the relevant tools + rag_search topics. Lower
priority than C-G1/C-G2 because it's discoverability polish, not a cold trail.

## What NOT to do (still true under the chatbot lens)
- Do NOT re-add the local-Python execution/asset-upload tools as MCP tools — the
  stateless rule is correct and a chatbot cannot run them regardless.
- Do NOT duplicate full skill workflows verbatim into prompts — they're large,
  and the RAG corpus already holds the substance. Route to it; don't inline it.
- Do NOT remove the "see the X skill" references for Claude-Code users — keep
  them AND add the rag_search alternative (serve both environments).

## Revised priority
For the skills-less-chatbot goal, **C-G1 and C-G2 are the highest-value changes
in this entire audit** — higher than G1–G3 — because they convert a structurally
incomplete experience (dead-end pointers) into a complete one using content that
already exists. C-G3 closes the "explain the local step" gap. G1–G3 remain worth
doing but are smaller polish by comparison.
