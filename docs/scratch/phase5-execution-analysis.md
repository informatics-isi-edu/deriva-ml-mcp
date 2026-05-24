# Phase 5 — Execution domain analysis (scratch)

> **STATUS: DEFERRED (2026-05-24).** The `commit_execution` MCP tool proposed in this document is **permanently deferred** in favor of a future `work-with-executions` skill in [deriva-ml-skills](https://github.com/informatics-isi-edu/deriva-ml-skills). Per the v0.5.0 stateless-MCP architectural rule (see [CLAUDE.md](../../CLAUDE.md) and [`docs/superpowers/notes/audit-2026-05-23.md`](../superpowers/notes/audit-2026-05-23.md)), execution lifecycle does not belong on the MCP wire — it composes only at the caller's local Python where the asset manifest + SQLite registry live. This document is preserved as **historical design context** showing what a hypothetical MCP-side commit tool would have needed to do (including the bug-fix call-sequence per ADR-0009 on the deriva-ml side); it is **NOT a roadmap**.

> **Status note (2026-05-24):** This analysis was originally drafted
> against deriva-ml's pre-v1.39 surface. Updated 2026-05-24 to reflect
> the unified `commit_output_assets` surface that shipped in deriva-ml
> v1.39.0 (see
> [ADR-0009](https://github.com/informatics-isi-edu/deriva-ml/blob/main/docs/adr/0009-unified-commit-output-assets.md)).
> Method names and call sequences below are current as of that release.
> Specifically: `upload_execution_outputs`, `upload_outputs`,
> `ExecutionSnapshot.upload_outputs`, and `upload_pending` are gone;
> `commit_output_assets` (on `Execution`) and `commit_pending_executions`
> (on `DerivaML`) replace them. The `retry_failed=` kwarg was removed —
> idempotency is now intrinsic to `commit_output_assets`.

Working notes for Task 5.1. Inputs:

- `deriva-ml/src/deriva_ml/core/mixins/execution.py` (~749 lines, `ExecutionMixin`).
- `deriva-ml/src/deriva_ml/execution/execution.py` (~2555 lines, `Execution` class
  — context manager, lifecycle methods, asset-upload glue).
- `deriva-ml/src/deriva_ml/execution/state_machine.py` (~600 lines — `ALLOWED_TRANSITIONS`
  table + `transition()` driver + reconciliation).
- `deriva-ml/src/deriva_ml/execution/state_store.py` — `ExecutionStatus` enum (7 values).
- `deriva-ml/src/deriva_ml/execution/execution_record.py` — catalog-bound row used by
  `lookup_execution` / `find_executions` / hierarchy queries.
- `deriva-mcp/src/deriva_mcp/tools/execution.py` (~412 lines, **9** `@mcp.tool()`
  decorators — verified by `grep -c "@mcp.tool()"` → 9).
- `deriva-mcp-core/src/deriva_mcp_core/tools/{entity,query,schema,vocabulary}.py` for
  cross-reference.

## A. The state machine (load-bearing for tool design)

`ExecutionStatus` is a `StrEnum` with **7** values:

```
Created  Running  Stopped  Failed  Pending_Upload  Uploaded  Aborted
```

`ALLOWED_TRANSITIONS` (state_machine.py L109-143) — the full graph:

```
Created  → Running                 (start work; happy path)
Created  → Aborted                 (terminal cleanup before any work)

Running  → Stopped                 (work succeeded; happy path)
Running  → Failed                  (work raised; failure path)
Running  → Pending_Upload          (HARD-CRASH RECOVERY only — process
                                   SIGKILL'd before __exit__ could fire,
                                   user calls update_status manually)
Running  → Aborted                 (mid-run cancellation)

Stopped  → Pending_Upload          (upload phase begins)
Stopped  → Aborted                 (decide not to upload after all)

Pending_Upload → Uploaded          (catalog INSERTs + asset PUTs all done)
Pending_Upload → Failed            (upload itself failed)

Failed   → Pending_Upload          (RETRY: re-attempt the failed upload)
Failed   → Aborted                 (give up after upload retry failure)

Uploaded → Pending_Upload          (ADDITIVE upload: a different lifecycle
                                   owner — typically the runner harness —
                                   adds a follow-on asset batch to an
                                   already-Uploaded execution)
```

Things to notice for tool design:

- **Created → Running and Running → Stopped are NOT exposed via top-level
  ML methods**; they live ONLY on the `Execution.execute()` context-manager
  `__enter__` / `__exit__`. The OUT-OF-CONTEXT counterparts are
  `Execution.execution_start()` / `Execution.execution_stop()`. So a
  caller who can't use `with execution.execute():` (i.e. an MCP caller)
  drives the lifecycle through these named methods. Important.
- **Stopped is NOT terminal.** Stopped means "the run finished but the
  upload phase hasn't been kicked off". From an MCP caller's POV "I'm
  done with the run" usually means Stopped + (later) Uploaded.
  Conflating them in one tool is a Q1 question.
- **Aborted is terminal-from-anywhere-pre-Uploaded.** Once you hit
  Uploaded the data is durable; Aborted is for "discard this run".
- **Re-starting a Stopped/Failed/Uploaded execution is NOT in the
  transition table** — once you've stopped, you can only upload-or-abort.
  "Restart" in the colloquial sense means "create a new execution"
  (see Q1 below).
- **`abort()` exists on `Execution`**; the old MCP tool surface had no
  abort endpoint, so this is a Phase 5 net-new lever.

## B. DerivaML execution surface — `ExecutionMixin` on `DerivaML`

Public methods (no leading underscore), in declaration order:

| Method | Returns | What it does |
|---|---|---|
| `create_execution(configuration=None, *, datasets=None, assets=None, workflow=None, description=None, dry_run=False)` | `Execution` (status=Created in registry; row INSERTed in catalog unless dry_run) | Builds an `ExecutionConfiguration` (or accepts one), downloads input datasets/assets to local cache, INSERTs the Execution row, returns the in-memory `Execution`. Online-mode only. |
| `lookup_execution(rid)` | `ExecutionRecord` (catalog-bound, mutable) | Read one execution by RID. Setters write through to the catalog (description, status). Online only. |
| `list_executions(*, status, workflow_rid, mode, since)` | `list[ExecutionSnapshot]` | Read from the LOCAL SQLite registry — no server contact, frozen snapshots, works offline. The "what's on this workspace" question. |
| `find_executions(workflow=None, workflow_type=None, status=None)` | `Iterable[ExecutionRecord]` | Catalog-side search. Online only. The "what's in the catalog" question. |
| `find_incomplete_executions()` | `list[ExecutionSnapshot]` | Sugar over `list_executions(status=[Created, Running, Stopped, Failed, Pending_Upload])`. SQLite-only. |
| `pending_summary()` | `WorkspacePendingSummary` | Workspace-wide pending-upload aggregator. SQLite-only. Used by standalone uploaders. |
| `resume_execution(rid)` | `Execution` (re-hydrated from registry) | Re-hydrate a prior Execution from the local SQLite registry. Runs JIT reconciliation. Works in both online and offline modes. |
| `gc_executions(*, older_than, status, delete_working_dir)` | `int` (count removed) | Garbage-collect SQLite registry rows (and optionally working dirs). Does NOT touch catalog. |
| `lookup_experiment(rid)` | `Experiment` | Wrap an execution in the Experiment helper for hydra-config introspection. |
| `find_experiments(workflow_rid, status)` | `Iterable[Experiment]` | Find executions that have Hydra config metadata. |
| `commit_pending_executions(*, execution_rids=None, clean_folder=False)` | `UploadReport` | Blocking commit of pending state for selected executions (or every workspace-registered execution with pending work, when `execution_rids` is omitted). The "actually flush to catalog" lever. Drives the full lifecycle bracket per execution (status, descriptions, Upload_Duration, optional folder cleanup); per-execution failures are aggregated into the returned report rather than stopping the batch. Online only. (deriva-ml v1.39+, replaces the legacy `upload_pending(retry_failed=...)`.) |

Methods on `Execution` itself (the context-manager / write-side):

| Method | Notes |
|---|---|
| `execute()` | No-op returning `self`. Used as `with exe.execute(): ...`. The lifecycle transitions actually happen in `__enter__` / `__exit__`. |
| `__enter__` | Created → Running via `transition()`. |
| `__exit__` | Running → Stopped (clean) or Running → Failed (exception). Tolerates already-terminal states (caller may have called `commit_output_assets()` inside the with-block, advancing past Running). |
| `execution_start()` | Imperative Created → Running for callers who can't use the context manager (e.g. MCP — same use case as the multirun parent execution managed by atexit). |
| `execution_stop()` | Imperative Running → Stopped. The non-context-manager counterpart. |
| `update_status(target, *, error=None)` | Generic status setter that goes through `state_machine.transition()`. Validates the requested transition. |
| `abort()` | Transition to Aborted from any non-terminal state. |
| `add_features(records)` | Stage feature values (used by Phase 3 `add_feature_values` tool). |
| `add_nested_execution(child, sequence=None)` | Add a child execution row. |
| `is_nested()` / `is_parent()` | Query parent/child status. |
| `list_input_datasets()` | Datasets specified at config time (read-only after creation). |
| `list_assets(asset_role=None)` | Catalog-bound asset list. |
| `create_dataset(...)` | Create a NEW dataset linked to this execution as provenance. |
| `add_files(...)` | Register external files in the catalog as a dataset, linked to this execution. |
| `pending_summary()` | Per-execution pending-upload snapshot. |
| `commit_output_assets(clean_folder=None, progress_callback=None)` | The one method that commits an execution's output assets to the catalog (deriva-ml v1.39+, replaces the trio `upload_execution_outputs` / `upload_outputs` / `ExecutionSnapshot.upload_outputs`). Drives the full lifecycle bracket — Stopped → Pending_Upload → Uploaded, asset description writes, Upload_Duration recording, optional working-folder cleanup. Returns an `UploadReport`. Idempotent on re-run after a success (no-op returning an empty report) or partial failure (resumes from last known-good state via `BagCatalogLoader`'s `match_by_columns` dedup). Raises on failure — failure isolation is `commit_pending_executions`'s job. |
| `download_dataset_bag(spec)` | Materialise a referenced dataset to local disk. |
| `download_asset(...)` | Download one input asset. |

`ExecutionRecord` (catalog-bound) hierarchy methods:

| Method | Notes |
|---|---|
| `list_execution_children(recurse=False)` | Yield child executions. Cycle-safe via `_visited`. |
| `list_execution_parents(recurse=False)` | Yield parent executions. Cycle-safe via `_visited`. |
| `add_nested_execution(child, sequence=None)` | Add a child relationship. |
| `update_status(status, status_detail="")` | Direct catalog write of the Status column. (NB: bypasses the `state_machine.transition()` validation — used for catalog-side updates that don't need the SQLite/registry dance.) |
| `list_input_datasets()` / `list_assets(asset_role=None)` | Same shape as `Execution`'s; reads from catalog. |

## C. Old deriva-mcp execution tools (9 confirmed)

Verified: `grep -c "@mcp.tool()" /Users/carl/.../deriva-mcp/src/deriva_mcp/tools/execution.py` → 9.

| # | Tool name | Signature (sketch) | Calls |
|---|---|---|---|
| 1 | `create_execution` | `(workflow_name, workflow_type, description="", dataset_rids=None, asset_rids=None, dry_run=False)` | `ml.create_workflow(...)` + `ExecutionConfiguration(...)` + `ml.create_execution(config, dry_run=...)`. **Stashes the result on a connection-singleton "active execution".** |
| 2 | `start_execution` | `()` (zero args!) | `_get_active_tool_execution().execution_start()` |
| 3 | `stop_execution` | `()` (zero args!) | `_get_active_tool_execution().execution_stop()` |
| 4 | `update_execution_status` | `(status: str, message: str)` (zero exec arg!) | `_get_active_tool_execution().update_status(status_enum, message)` |
| 5 | `set_execution_description` | `(execution_rid: str, description: str)` | `ml.lookup_execution(rid)` then `record.description = ...` |
| 6 | `restore_execution` | `(execution_rid: str)` | `ml.restore_execution(rid)` then stash on singleton (NB: `ml.restore_execution` was renamed to `ml.resume_execution` upstream — old tool calls a now-missing method; tracked as bit-rot in old surface). |
| 7 | `create_execution_dataset` | `(description="", dataset_types=None)` (zero exec arg!) | `_get_active_tool_execution().create_dataset(...)` |
| 8 | `add_nested_execution` | `(parent_execution_rid, child_execution_rid, sequence=None)` | `ml.lookup_execution(parent_rid).add_nested_execution(child_rid, sequence=sequence)` |
| 9 | `list_nested_executions` | `(execution_rid, recurse=False)` | `ml.lookup_execution(rid).list_nested_executions(recurse=recurse)` (NB: upstream renamed to `list_execution_children`; old tool calls a now-missing method — bit-rot). |

Notable observations:

- **Connection-singleton "active execution" is everywhere.** Tools 2-4 and 7
  take ZERO execution argument and rely on a process-global
  `conn_info.active_tool_execution`. This is the same boundary-rule violation
  Phase 1 stripped from `get_ml`. Phase 5 MUST take `execution_rid: str` as
  an explicit arg on every tool that operates on one.
- **Old tools mix workflow registration into `create_execution`.** Tool #1
  accepts `workflow_name` + `workflow_type` and registers a workflow as a
  side effect. Phase 4 already gives us `create_workflow` + `find_workflow_by_url`
  as standalone tools — `create_execution` should accept `workflow_rid` and
  let the caller compose. Same boundary discipline as Phase 4's URL-required
  `create_workflow`.
- **No tool for "list executions" or "find executions".** Big gap. The
  catalog admin lens ("show me all running executions") and the ML developer
  lens ("did my run finish?") both need this. Net-new in Phase 5.
- **No tool for "upload outputs" / "commit pending" / "abort"** — the most
  important user-facing levers per the state machine. The old tools end at
  `stop_execution`, which leaves the execution in `Stopped` status with
  feature-value rows STAGED in SQLite but NOT yet flushed to the catalog.
  This is the Phase 3 design carryover Q1 — see resolution below.
- **Two tools (#6, #9) call methods that no longer exist upstream.** They
  would error at runtime today. The old surface is bit-rotted.

## D. deriva-mcp-core overlap

Tools in core that *might* overlap with execution operations:

- `entity.py`: `get_entities`, `update_entities`, `delete_entities`. The
  Execution table is a row; could be CRUD'd this way.
- `query.py`: `query_attribute`, `count_table` — generic counts.
- `schema.py`: `create_table` etc. — irrelevant (Execution table predefined).

What IS core's domain (drop):

- **`set_execution_description`** → `update_entities("deriva-ml", "Execution",
  entities=[{"RID": rid, "Description": "..."}])`. Pure column update with
  no business logic. Mirrors Phase 2's `set_dataset_description` drop.

What we KEEP in this plugin (despite formal CRUD overlap):

- **Status transitions.** `update_entities("Execution", ..., {"Status":
  "Stopped"})` would bypass `state_machine.transition()` and corrupt the
  registry / leave SQLite stale. MUST go through Execution-domain tools.
- **Hierarchy edits / queries.** `Execution_Execution` is an internal
  association table; LLM shouldn't have to know it exists. Same logic as
  keeping `update_dataset_types` over raw insert/delete.
- **Pending-upload semantics.** `commit_pending_executions()` does asset
  PUT + catalog INSERT + status/Upload_Duration/description writes in a
  coordinated way with per-execution failure isolation. Generic CRUD
  can't replicate this. (Prior to v1.39 this was `upload_pending()`,
  which skipped the lifecycle bracket — the bug ADR-0009 fixed.)

## E. Resolutions to the 4 Phase 3 design carryover questions

### Q1: Lifecycle exposure — long-running vs batch-wrapper vs hybrid

**Resolution: Hybrid (option C), but with sharper edges than Phase 3 stated.**

**Position.** Expose three explicit lifecycle tools — `start_execution`,
`commit_execution`, `abort_execution` — that operate on a passed-in
`execution_rid`. Phase 3's `add_feature_values` keeps its current
batch-wrapper semantics: it only auto-wraps if the execution is in `Created`
or already `Running` state, and routes through a single-shot
`execute() -> add_features() -> exit` cycle.

**Rationale.**

- Pure batch-wrapper (B) breaks the genuine multi-tool pipeline use case.
  An LLM session that wants to "register a workflow → create execution →
  add features twice with different feature names → commit" cannot do that
  if every mutation tool slams the execution back to Stopped after one
  call. The Phase 3 carryover note flagged this exact pain.
- Pure long-running (A) means every mutation tool needs to inspect status,
  validate "expected Running", and either raise or auto-wrap. Confusing
  rules at every tool.
- Hybrid (C) gives the caller the choice. If you have one batch and you
  don't want lifecycle ceremony: call `add_feature_values` against a
  `Created` execution; it auto-wraps. If you have multiple batches /
  cross-tool work: call `start_execution` first, do all your mutation
  tool calls (the wrapper sees Running and skips the open/close — sees
  the auto-wrap as a no-op since it's already Running), then
  `commit_execution`. Mental model: `start_execution` "opens the door";
  `commit_execution` "closes and locks the door"; without an explicit
  open, each mutation tool opens-and-closes around itself.

**Sharper edges than the original C statement.**

- `start_execution(execution_rid)` advances Created → Running (calls
  `execution.execution_start()`).
- `commit_execution(execution_rid)` advances Running → Stopped → Pending_Upload
  → Uploaded by calling `execution.commit_output_assets()`. One MCP call,
  three transitions, ONE deriva-ml call — `commit_output_assets` drives
  the full lifecycle bracket internally (status transitions, asset
  descriptions, Upload_Duration, optional folder cleanup). Returns the
  `UploadReport`. (The pre-v1.39 design that called `execution_stop()`
  then a separate upload method was wrong — it would have inherited the
  status-stuck-Stopped bug from the legacy `upload_outputs`; see
  ADR-0009 for the bug history.)
- `abort_execution(execution_rid, reason=None)` advances any-non-terminal
  → Aborted via `execution.abort()`.
- `add_feature_values` (Phase 3, already shipped): no source change
  required for the hybrid model. The current implementation calls
  `execution.execute()` as a context manager which is a no-op if the
  execution is ALREADY Running (the `__enter__` advances Created →
  Running, but if status is Running the validate_transition raises
  `InvalidTransitionError`). So we either (a) tighten Phase 3's
  `add_feature_values` to check `if status == Created` before entering
  the context (auto-wrap mode) and `if status == Running` skip the
  enter/exit (long-running mode); or (b) defer that check to a Phase 5
  follow-up. **Recommendation: Phase 5.2 ships the lifecycle tools and
  patches `add_feature_values` to accept both states in the same task.**
  Document the contract.

**Trade-off acknowledged.** Hybrid mode means the same tool (`add_feature_values`)
behaves slightly differently in two modes. We document this explicitly and
test BOTH paths in unit + integration. The alternative (forcing one mode)
costs more user pain than the hybrid documentation cost.

### Q2: `with execution.execute():` must be inside `with deriva_call():`

**Resolution: Confirm + codify as a tool-authoring rule.**

**Position.** Phase 3's `add_feature_values` already does this correctly
(deriva_call OUTER, execute INNER). Phase 5 inherits the rule with no
changes; we add a short paragraph to the conventions section in the plan
(or an inline comment in `tools/execution.py` on each lifecycle tool that
opens an execution context).

**Rationale.** The `__exit__` of `Execution.execute()` calls
`state_machine.transition()`, which writes SQLite then PUTs the catalog
Execution row. The catalog PUT is a deriva-py call that needs the
401-on-close handling that `deriva_call()` provides. Without it, a
mid-call session expiry on the close transaction silently leaves
`sync_pending=True` and corrupt audit (the success-audit fired in our
tool wrapper before the close failed).

**Codification.**

- Add to the per-tool docstring (Audit notes section) for each Phase 5
  mutation tool that enters an execution context: "Wraps `with
  execution.execute():` inside `with deriva_call():` so the execute
  __exit__'s catalog PUT is covered by the 401-on-close handler."
- Add a one-line rule to the Phase 5 conventions block in the plan.

### Q3: Idempotency / retry safety

**Resolution: Option C (defer + naturally-idempotent design + dedicated audit
dimension), with one specific carve-out.**

**Position.**

- Phase 5 mutation tools should be designed to be naturally idempotent
  WHERE POSSIBLE without an explicit idempotency_key parameter:
  - `start_execution`: idempotent if status is already Running (no-op
    return with status field). Same for "already past Running" (clear
    error).
  - `commit_execution`: idempotent. The deriva-ml v1.39 method
    `commit_output_assets()` short-circuits if the execution is
    already `Uploaded` with no pending work (returns an empty
    `UploadReport`); on a partial-failure resume it picks up at the
    last known-good state, with `BagCatalogLoader`'s `match_by_columns`
    dedup making row inserts idempotent at the catalog. There is no
    `retry_failed=` kwarg — under the bag pipeline it was a documented
    no-op anyway and ADR-0009 removed it. The MCP tool surfaces the
    same idempotency guarantee with no additional logic of its own.
  - `abort_execution`: idempotent if status is already Aborted (no-op).
- Where natural idempotency is genuinely impossible (multi-batch flush
  half-way through), document non-idempotent behavior and rely on the
  audit trail (the `deriva_ml_<op>_failed` audit row carries
  `attempted_count` already; that's the forensic recovery hook).
- Defer `idempotency_key` parameter as a Phase 7+ feature gated on
  evidence of real-world replay-collision pain. Implementing it requires
  a server-side dedup store (audit table check by key, or a dedicated
  idempotency table) that's not in scope for Phase 5.

**Specific carve-out.** `commit_execution` returns the upstream
`UploadReport`'s per-table counts and per-row error lines INSIDE the
JSON response (not just `attempted_count`). This gives the caller
forensic visibility into what landed and what didn't, even in
non-idempotent partial-failure cases. Same shape as Q4 (formalized
richer response_fields).

**Rationale.** Phase 5 is the most complex domain; adding a brand-new
idempotency-key contract on top of the lifecycle redesign would balloon
scope. Natural idempotency where possible + UploadReport visibility
covers 80% of real cases without infrastructure debt.

### Q4: Bulk-mutation `response_fields` shape

**Resolution: Keep the current free-form mechanism, but FORMALIZE a
recommended shape for the upload-report case specifically.**

**Position.**

- Don't make `_error_envelope.response_fields` strict — it's a
  general-purpose escape hatch and tightening the schema everywhere
  would force every tool to converge on a shape it doesn't need.
- DO add a documented "richer shape" convention for tools that return
  per-record results (like `commit_execution` returning an UploadReport):
  - On the SUCCESS path: include `tables: list[{name, inserted, failed}]`,
    `assets: list[{kind, uploaded, failed}]`, and `errors: list[str]`
    (drawn straight from the UploadReport).
  - On the FAILURE path: same shape inside `response_fields`.
- The existing `attempted_count` / `failed_entry_index` from
  `add_feature_values` (Phase 3) STAYS — it's the right shape for that
  tool's per-entry failure model. The new shape is additive.

**Rationale.** A one-size-fits-all shape would trade clarity for
uniformity — `add_feature_values` doesn't have per-table breakdowns; the
UploadReport tools don't have per-entry indices. Keep both shapes,
document each in its tool's docstring under "Response notes:".

## F. Per-tool reasoning + disposition

| # | Old tool | Disposition | New tool / target | Rationale |
|---|---|---|---|---|
| 1 | `create_execution` | renamed | `create_execution` | Same intent, different signature: takes `workflow_rid` (not workflow_name + workflow_type — those are Phase 4's `create_workflow` job), takes optional `dataset_rids` / `asset_rids` for provenance, drops the connection-singleton stash. `dry_run=True` doesn't audit (Phase 2 convention). |
| 2 | `start_execution` | renamed | `start_execution` | Now takes explicit `execution_rid`. Calls `execution.execution_start()`. Rejects if status != Created with `{"error": ...}` (argument validation, no audit). |
| 3 | `stop_execution` | merged | `commit_execution` | Old tool only advanced Stopped — left the execution in a partial state (rows staged but not flushed). New `commit_execution` does the full Stopped → Pending_Upload → Uploaded sequence + returns the UploadReport. The "stop without upload" path is not a real user intent — see Q1 resolution. |
| 4 | `update_execution_status` | dropped | (none) | The status transitions are governed by the state machine; exposing a free-form "set status to X" tool is dangerous (lets the LLM pick illegal targets) and redundant once `start_execution` / `commit_execution` / `abort_execution` exist. The error-message-with-state use case is covered by `abort_execution(reason=...)`. The "I crashed; mark it Pending_Upload manually" recovery path is a Phase 7+ admin tool, not user-facing. |
| 5 | `set_execution_description` | dropped-to-core | `update_entities` | Pure column update; use `update_entities("deriva-ml", "Execution", entities=[{"RID": rid, "Description": "..."}])`. Mirrors Phase 2's `set_dataset_description` decision. |
| 6 | `restore_execution` | dropped | (none) | The "active execution" singleton is dead with the boundary-rule (see #1, #2, #3, #7). Once every mutation tool takes `execution_rid` explicitly, there's no state to restore. The caller passes the RID; if they don't have it, `list_executions` / `find_executions` / `find_workflow_executions` finds it. |
| 7 | `create_execution_dataset` | renamed | `create_execution_dataset` | Same intent. New version takes explicit `execution_rid` + `dataset_types`. Calls `Execution.create_dataset` (which goes through `Dataset.create_dataset` with `execution_rid` provenance). |
| 8 | `add_nested_execution` | kept | `add_nested_execution` | Same intent. Already takes both RIDs explicitly. |
| 9 | `list_nested_executions` | renamed | `list_execution_children` | Mirrors the upstream rename (`Execution.list_nested_executions` → `ExecutionRecord.list_execution_children`) and aligns with the dataset-hierarchy template (`list_dataset_children`). Also adds a complement `list_execution_parents` as net-new (the parent-direction read). |

## G. Net-new tools (gaps in the old surface)

Following Phase 2/3/4 pattern (each phase added net-new reads on top of write-only old surfaces), Phase 5 adds:

1. **`list_executions(hostname, catalog_id, ...)`** — paged catalog-side enumeration of executions. Mirrors `list_workflows` / `list_datasets`. Uses `ml.find_executions()` (catalog-side, returns ExecutionRecord). NB: NOT `ml.list_executions()` which is SQLite-registry-only (irrelevant to MCP server which has no persistent workspace). Filters: `workflow_rid`, `status`. Cursor pagination.
2. **`get_execution(hostname, catalog_id, execution_rid)`** — full per-execution describe by RID. Mirrors `get_workflow` / `get_dataset`. Calls `ml.lookup_execution(rid)`.
3. **`find_workflow_executions(hostname, catalog_id, workflow_rid, ...)`** — read all executions of a given workflow. Convenience over `list_executions(workflow_rid=...)` BUT keep it separate because the user intent ("find runs of THIS workflow") is distinct from "page through all executions". Uses `ml.find_executions(workflow=workflow_rid)`.
4. **`commit_execution(hostname, catalog_id, execution_rid, retry_failed=False)`** — Stopped → Pending_Upload → Uploaded. Returns UploadReport details. (See Q1, Q3, Q4.)
5. **`abort_execution(hostname, catalog_id, execution_rid, reason=None)`** — any-non-terminal → Aborted. Calls `execution.abort()`.
6. **`list_execution_parents(hostname, catalog_id, execution_rid, recurse=False)`** — complement of `list_execution_children`. Calls `ExecutionRecord.list_execution_parents(recurse=recurse)`.

## H. Granularity clustering — final tool list

Grouped by user intent. Writes are kept specific (one intent per write tool); reads are coarser.

**Reads (lookup / enumeration):**

1. **`list_executions`** — page through executions in the catalog. Filters: workflow_rid, status. (admin lens: "show me all running executions"; ML dev lens: "what runs do I have?")
2. **`get_execution`** — describe one execution by RID.
3. **`find_workflow_executions`** — list all executions of a given workflow. (ML dev lens: "find runs of this training pipeline".)
4. **`list_execution_children`** — list nested child executions (mirror of Phase 2's `list_dataset_relations` direction param? No — keep separate per upstream's split into `list_execution_children` / `list_execution_parents`).
5. **`list_execution_parents`** — list parent executions (net-new symmetric complement).

**Writes (lifecycle / structure):**

6. **`create_execution`** — register a new execution against a workflow.
7. **`start_execution`** — Created → Running. Long-running mode opener (per Q1).
8. **`commit_execution`** — Stopped → Pending_Upload → Uploaded. Long-running mode closer (per Q1). Returns UploadReport. (Conceptually merges the old `stop_execution` with the new "upload outputs" semantic.)
9. **`abort_execution`** — any-non-terminal → Aborted.
10. **`create_execution_dataset`** — create a new dataset linked to an execution as provenance.
11. **`add_nested_execution`** — add parent/child relationship.

That's **11 tools** to implement in Phase 5.2 (9 old: 4 renamed + 1 kept + 1 merged + 2 dropped + 1 dropped-to-core; 6 net-new = 11 net).

## I. Cross-domain integration points

- **Workflow domain (Phase 4).** `create_execution(workflow_rid=...)` is the consumer of `create_workflow`'s return. The full chain: `find_workflow_by_url` → `create_workflow` (if missing) → `create_execution` → `start_execution` → ... → `commit_execution`.
- **Feature domain (Phase 3).** Phase 3 DoD #2 mutation round trip: `create_workflow` → `create_execution` → `add_feature_values` → `list_feature_values`. With Q1's hybrid model, this works in two equivalent ways:
  - Auto-wrap: `create_workflow` → `create_execution` → `add_feature_values` (auto-wraps Created → Running → Stopped). NB: this leaves rows staged in SQLite but never flushed without `commit_execution`. So we must EITHER also call `commit_execution`, OR (for the Phase 3 DoD test specifically) verify directly via the ml.add_features path which does an in-process flush at execute-exit.
  - Long-running: `create_workflow` → `create_execution` → `start_execution` → `add_feature_values` (sees Running, skips wrap) → `commit_execution`.
  - **For Phase 3 DoD #2, use the long-running form** so the assertion is "value landed in catalog after commit" (deterministic). See Phase 3 DoD #2 closure plan in the report.
- **Dataset domain (Phase 2).** `create_execution(dataset_rids=[...])` records input-dataset provenance; `create_execution_dataset` produces output-dataset provenance.

## J. Out-of-scope methods (deferred)

- `ml.list_executions(...)` (SQLite-only) — workspace registry isn't accessible across MCP boundary; the server doesn't persist a workspace. Dropped silently (not added to coverage; not an old tool).
- `ml.gc_executions(...)` — workspace-registry cleanup; same boundary as above. Dropped silently.
- `ml.find_incomplete_executions()` — SQLite-only; same boundary.
- `ml.pending_summary()` — workspace-registry aggregator; same boundary.
- `Execution.add_features` — exposed via Phase 3's `add_feature_values`; not a Phase 5 tool.
- `Execution.create_dataset` — wrapped as `create_execution_dataset`.
- `Execution.add_files` — file-import is an asset/dataset domain operation. Could be exposed in a future asset domain pass; Phase 5 skips.
- `Execution.download_dataset_bag` / `download_asset` — asset-domain reads/writes; Phase 5 skips. (The original analysis listed `upload_assets` here; that method does not exist on the `Execution` class — hallucination from the pre-v1.39 draft. The legitimate upload path is `commit_output_assets`, which is the lever `commit_execution` drives.)
- `ml.lookup_experiment` / `find_experiments` — Hydra-config introspection; might warrant its own "Experiment" domain tool surface in a future phase. Phase 5 skips.
- `ml.commit_pending_executions(execution_rids=[...])` (workspace-wide, not scoped to one execution) — exposed only via the per-execution `commit_output_assets()` call inside `commit_execution`. The workspace-wide form has no useful MCP analog (server has no persistent workspace; failure-isolation across an arbitrary batch is not a meaningful MCP user intent).

## K. Open judgment calls (surface in final report)

1. **Hybrid mode for `add_feature_values`.** Resolution to Q1 says the
   tool should accept both Created (auto-wrap) and Running (no-op wrap)
   states. Phase 3 ships the auto-wrap form only. Phase 5.2 will need a
   small patch to `add_feature_values`. Surface this in the report as a
   "tightening required" follow-up.

2. **`commit_execution` returns UploadReport — non-trivial JSON shape.**
   The UploadReport has per-table counts, per-asset-kind counts, error
   lines, and timestamps. Could grow large. Recommendation: include the
   summary fields (`total_uploaded`, `total_failed`, top-N error lines)
   directly; let callers fetch full details via `get_execution`. Trade-off
   surfaced.

3. **Long-running model leaks SQLite registry state.** Phase 3 carryover Q1
   mentioned "Execution.add_features stages records to a local SQLite
   registry; they only flush to ermrest when the execution context exits."
   In a long-running session, the SQLite registry on the MCP server (which
   may be ephemeral in stdio mode!) holds the staged rows until
   `commit_execution`. If the MCP server restarts mid-session, those rows
   are lost (the workspace path is process-local). Two options:
   - (a) Document this limitation explicitly: "Long-running mode requires
     the MCP server process to remain alive between `start_execution` and
     `commit_execution`. For ephemeral stdio sessions, prefer the
     auto-wrap path (one mutation per call)."
   - (b) Pin the workspace path to a stable location keyed by hostname +
     catalog_id so a server restart can resume. Significant work; defer.
   - **Recommendation: (a)** with a docstring note on `start_execution`.
     Surface in report.

4. **Should `list_executions` and `get_execution` use the SQLite registry
   or the catalog?** Ambiguous because of the two-tier source-of-truth.
   - Catalog: source of truth for "what executions exist"; works across
     sessions; needed for the admin lens.
   - SQLite registry: source of truth for "what's staged for upload";
     workspace-local; not portable to MCP boundary.
   - **Recommendation:** All MCP read tools query the CATALOG (via
     `ml.find_executions` / `ml.lookup_execution`), never SQLite. The
     SQLite registry is invisible to the MCP server's mental model.

5. **Asset upload visibility.** Old `list_asset_executions` lives in
   `tools/schema.py`, not `execution.py` — it's an asset-domain read.
   Skip from Phase 5 coverage table; flag for an asset-domain pass.

6. **`abort_execution`'s reason.** Phase 4's audit-field-presence
   convention says optional fields are emitted unconditionally with
   `None` as no-change marker. For `abort_execution(reason=None)`, the
   audit will always include `reason` (sometimes None). This is fine —
   `reason` is a bounded admin annotation, not user-supplied free text.
   Recommendation: include `reason` in audit; do NOT include it in the
   abort row's catalog write (the upstream `error` column is for
   exception messages, not abort rationales).

## L. Phase 3 DoD #2 closure sketch (sketch for Task 5.2's integration test file)

`tests/test_integration_execution.py` (new, mirroring
test_integration_workflow.py shape):

```python
"""Live-catalog integration tests for deriva-ml-mcp execution tools.

Includes the deferred Phase 3 DoD #2 mutation round-trip:
create_workflow -> create_execution -> add_feature_values -> commit_execution
-> list_feature_values verifies values landed.
"""

# pytestmark = same skipif as workflow integration test

# Fixture: integration_execution_tools(demo_mutation_catalog) registers
# BOTH execution and feature and workflow modules against one PluginContext
# (so add_feature_values and the lifecycle tools are co-located).

async def test_execution_round_trip(...):
    """Step 1-7: registration, creation, start, commit, abort coverage."""

    # 1. create_workflow (so create_execution has a workflow_rid).
    create_wf = json.loads(await tools["create_workflow"](...))
    workflow_rid = create_wf["workflow_rid"]

    # 2. list_executions baseline (zero in mut catalog).
    list0 = json.loads(await tools["list_executions"](...))
    baseline = list0["count"]

    # 3. create_execution. Status must be "Created".
    create_exe = json.loads(await tools["create_execution"](
        hostname=..., catalog_id=..., workflow_rid=workflow_rid,
        dataset_rids=None, asset_rids=None, dry_run=False,
        description="integration test run",
    ))
    assert create_exe["status"] == "created"
    execution_rid = create_exe["execution_rid"]
    assert isinstance(execution_rid, str) and execution_rid

    # 4. get_execution. Should return Status="Created".
    get_exe = json.loads(await tools["get_execution"](
        hostname=..., catalog_id=..., execution_rid=execution_rid))
    assert get_exe["status"] == "Created"
    assert get_exe["workflow_rid"] == workflow_rid

    # 5. start_execution. Created -> Running.
    start = json.loads(await tools["start_execution"](
        hostname=..., catalog_id=..., execution_rid=execution_rid))
    assert start["status"] == "running"
    assert start["execution_rid"] == execution_rid

    # 6. commit_execution. Running -> Stopped -> Pending_Upload -> Uploaded.
    commit = json.loads(await tools["commit_execution"](
        hostname=..., catalog_id=..., execution_rid=execution_rid))
    assert commit["status"] == "uploaded"
    assert commit["execution_rid"] == execution_rid
    assert commit["report"]["total_failed"] == 0

    # 7. get_execution again. Status now "Uploaded".
    get_after = json.loads(await tools["get_execution"](
        hostname=..., catalog_id=..., execution_rid=execution_rid))
    assert get_after["status"] == "Uploaded"


async def test_phase3_dod2_feature_round_trip(...):
    """Phase 3 DoD #2: feature values land in the catalog through the
    full lifecycle.

    NB: requires create_feature support, which the demo_mutation_catalog
    needs to either (a) ship with a pre-defined feature, or (b) the
    test exercises create_feature first. Phase 5.2 will land
    create_feature in this catalog via the Phase 3 unit-tested tool
    (already shipped; this is the integration smoke).
    """
    # Setup: create_feature("Image", "QualityScore", terms=["good", "bad"]).
    # NB: Image table needs to exist in demo schema. demo_catalog
    # already creates it -- verified during Phase 3 DoD #1.

    # 1. create_workflow.
    workflow_rid = ...

    # 2. create_execution against that workflow.
    execution_rid = ...

    # 3. start_execution (long-running mode -- so we can call
    #    add_feature_values then commit_execution and see the values
    #    actually flushed).
    await tools["start_execution"](execution_rid=execution_rid, ...)

    # 4. add_feature_values with two entries.
    add = json.loads(await tools["add_feature_values"](
        hostname=..., catalog_id=...,
        table="Image", feature_name="QualityScore",
        execution_rid=execution_rid,
        entries=[
            {"Image": image_rid_1, "QualityScore": "good"},
            {"Image": image_rid_2, "QualityScore": "bad"},
        ],
    ))
    assert add["status"] == "added"
    assert add["count"] == 2

    # 5. commit_execution -- this is the moment values flush to ermrest.
    commit = json.loads(await tools["commit_execution"](
        execution_rid=execution_rid, ...))
    assert commit["status"] == "uploaded"

    # 6. list_feature_values -- the two values are now visible.
    listed = json.loads(await tools["list_feature_values"](
        hostname=..., catalog_id=...,
        table="Image", feature_name="QualityScore"))
    assert listed["count"] == 2
    rids = sorted(v["target_rid"] for v in listed["values"])
    assert rids == sorted([image_rid_1, image_rid_2])
    scores = sorted(v["QualityScore"] for v in listed["values"])
    assert scores == ["bad", "good"]
```

**Open question for the Phase 5.2 implementer.** The
`demo_mutation_catalog` fixture currently has `populate=False` and
`create_features=False`. The DoD #2 round-trip needs at minimum a target
table (`Image`) with two rows to attach feature values to. Either:
- (a) Create the rows inline in the test (insert via core's
  `insert_entities("demo-schema", "Image", ...)`).
- (b) Promote a richer fixture (`demo_populated_mutation_catalog`?) with
  pre-seeded Image rows.
- **Recommendation:** (a) — the test owns its setup, no new fixture.

## M. Workflow `update_workflow` precondition note

Phase 4's `tests/test_integration_workflow.py:268-288` documents an upstream
deriva-ml bug (`Workflow.__setattr__` Pydantic v2 gate that's permanently
False). Phase 5.2's integration test will:

- Check whether the upstream fix has landed before Phase 5.2 implementation
  starts. (Look at deriva-ml's git log for `Workflow.__setattr__`.)
- If fixed: tighten the assertion at line 286-287 to also verify
  `matching[0]["description"] == "updated description"`.
- If still broken: leave the inline comment block in place and add a
  similar pattern (inline-doc + scoped assertion) for any Phase 5
  mutation that depends on Workflow's catalog-bound setters working.
  Currently no Phase 5 tool depends on Workflow setters — `create_execution`
  takes `workflow_rid` directly and never calls `wf.description = ...` —
  so this is informational only.

## N. Final tool list summary

11 tools to implement in Phase 5.2:

| # | Tool | mutates | Calls |
|---|---|---|---|
| 1 | `list_executions` | False | `ml.find_executions(workflow=workflow_rid, status=status)` (then paginate) |
| 2 | `get_execution` | False | `ml.lookup_execution(rid)` |
| 3 | `find_workflow_executions` | False | `ml.find_executions(workflow=workflow_rid)` |
| 4 | `list_execution_children` | False | `ml.lookup_execution(rid).list_execution_children(recurse=...)` |
| 5 | `list_execution_parents` | False | `ml.lookup_execution(rid).list_execution_parents(recurse=...)` |
| 6 | `create_execution` | True | `ml.create_execution(workflow=workflow_rid, datasets=[...], assets=[...], description=..., dry_run=...)` |
| 7 | `start_execution` | True | `ml.resume_execution(rid).execution_start()` (or direct lookup_execution + transition) |
| 8 | `commit_execution` | True | `ml.resume_execution(rid).commit_output_assets()` — ONE call. (Originally specified as `.execution_stop()` then `.upload_outputs(retry_failed=...)`; that sequence was wrong — `upload_outputs` did NOT transition status to `Uploaded`, the bug ADR-0009 fixed. In v1.39 the lifecycle bracket is intrinsic to `commit_output_assets`: it auto-stops a Running execution, transitions through Pending_Upload → Uploaded, writes Upload_Duration and asset descriptions, and optionally cleans the working folder. The MCP tool does not need to compose two calls.) |
| 9 | `abort_execution` | True | `ml.resume_execution(rid).abort()` |
| 10 | `create_execution_dataset` | True | `ml.resume_execution(rid).create_dataset(dataset_types=..., description=...)` |
| 11 | `add_nested_execution` | True | `ml.lookup_execution(parent_rid).add_nested_execution(child_rid, sequence=...)` |

NB: Tools 7-10 use `ml.resume_execution(rid)` to re-hydrate the Execution
object before calling lifecycle methods. Whether the MCP server's process
maintains a workspace registry across calls is the open question (K.3).
For dry_run executions, lifecycle tools will be no-ops; surface that in
the docstring.
