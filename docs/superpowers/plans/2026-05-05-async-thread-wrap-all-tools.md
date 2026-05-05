# Async Thread-Wrap All Tool Modules — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap every synchronous deriva-ml / deriva-py call inside an `async def` tool with `await asyncio.to_thread(...)`, across the seven tool modules PR #28 didn't cover.

**Architecture:** Same pattern PR #28 established in `tools/dataset/complex.py`. Inside each `async def` tool's `with deriva_call():` block, change every direct sync call (`ml.<method>(...)`, `ds.<method>(...)`, `exe.<method>(...)`, `_pkg.get_ml(...)`, plus helpers that themselves call sync deriva-ml) to `await asyncio.to_thread(<callable>, *args, **kwargs)`. For generator-returning sync calls, drain into a list inside a nested `def _drain():` helper and thread the helper (precedent: `complex.py:325–338`). No new functions, no API changes, no tests rewritten — purely a behavioral fix that keeps the asyncio event loop responsive.

**Tech Stack:** Python 3.11+, asyncio, deriva-ml, deriva-mcp-core plugin SDK, pytest, ruff, uv.

**Spec context:** [`CLAUDE.md`](../../../CLAUDE.md) — see "Tool Implementation Rules → asyncio.to_thread" and "Development Gotchas → Sync calls in async tools" (added 2026-05-05). PR #28 is the precedent: `fix(complex): wrap sync deriva-ml calls in asyncio.to_thread`.

---

## Why this is one PR with N module-scoped commits

The change is mechanical and uniform; one PR makes the lesson visible (everything that was missed is fixed together). Module-scoped commits keep the diff reviewable and let `git bisect` find regressions if any slip through. The order below moves from smallest to largest module so reviewers can build pattern intuition before hitting the 700+ line files.

## Per-task pattern (applies to every code task below)

Each tool-module task follows the same shape:

1. **Identify all sync deriva-ml/deriva-py callsites inside `with deriva_call():` blocks of `async def` tools.** Use the grep recipe (Step A below).
2. **Add `import asyncio` to the module's imports** if not already present.
3. **Wrap each callsite** with `await asyncio.to_thread(...)`. Patterns:
   - `result = ml.find_workflows()` → `result = await asyncio.to_thread(ml.find_workflows)`
   - `result = ml.find_workflows(deleted=True)` → `result = await asyncio.to_thread(ml.find_workflows, deleted=True)`
   - `ml = _pkg.get_ml(hostname, catalog_id)` → `ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)`
   - `ds = ml.lookup_dataset(rid)` → `ds = await asyncio.to_thread(ml.lookup_dataset, rid)`
   - **Generator returns** (e.g. `ds.get_denormalized_as_dict(...)` returns iterator): drain inside a nested `def _drain():` and thread that. See `tools/dataset/complex.py:325–338` for the canonical example.
   - **Property reads** (`ds.dataset_rid`, `ds.description`, `ds.current_version`) — these are NOT sync I/O calls; leave them alone. Properties on already-resolved objects are zero-cost dict lookups.
   - **Helper functions defined in the same module** (e.g. `_list_datasets_impl(ml, ...)`): if the helper itself contains sync deriva-ml calls (it usually does), wrap the helper invocation: `payload = await asyncio.to_thread(_list_datasets_impl, ml, ...)`. Don't recursively wrap inside the helper — the helper runs entirely inside the worker thread.
4. **Run that module's tests** — both unit (`tests/test_<module>.py`) and integration (`tests/test_integration_<module>.py` if present).
5. **Commit** with the message format below.

### Step A — find sync callsites inside a module

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp
awk '/with deriva_call\(\):/,/^        return|^        except|^    @/' \
    src/deriva_ml_mcp/tools/<MODULE>.py \
  | grep -nE '^\s+(ml|ds|exe|wf|exec_obj|_pkg|f|asset|feature)\.|=\s*(ml|ds|exe|wf|exec_obj|_pkg)\.'
```

(The awk slices each `deriva_call()` block; the grep flags the sync attribute calls inside.)

### Step B — module-tests recipe

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_<module>.py -v
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_integration_<module>.py -v   # if present
```

Expected: all tests pass. The change is behavior-preserving (same calls, same returns, just routed through a thread).

### Step C — commit message format

```
fix(<module>): wrap sync deriva-ml calls in asyncio.to_thread

Mirrors PR #28's pattern for tools/<module>.py. Every sync deriva-ml /
deriva-py call inside an async def tool's deriva_call() block now
runs via await asyncio.to_thread(...), so the asyncio event loop
stays responsive while catalog work runs.

Resolves the missing coverage from PR #28 for this module.
```

---

## Task 0: Branch + verify clean baseline

**Files:** none (workspace setup).

- [ ] **Step 1: Branch off main**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp
git checkout main
git pull --ff-only origin main
git checkout -b fix/async-thread-wrap-tools
```

Expected: on a fresh branch off `main`, working tree clean.

- [ ] **Step 2: Run the full test suite to capture a green baseline**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest -x 2>&1 | tail -10
```

Expected: all tests pass (or known-flaky tests skip). If anything fails on `main`, **STOP** — that's a pre-existing regression unrelated to this work; surface it before continuing. Note the count for comparison at the end.

- [ ] **Step 3: Confirm `import asyncio` is already present in `dataset/complex.py` (the existing example)**

```bash
grep -n '^import asyncio' src/deriva_ml_mcp/tools/dataset/complex.py
```

Expected: line 1–10 of the file. If absent, the existing PR #28 work is broken — STOP.

---

## Task 1: `tools/asset.py`

**Files:**
- Modify: `src/deriva_ml_mcp/tools/asset.py` (~7 callsites, 525 LoC)
- Test: `tests/test_asset.py`, `tests/test_integration_asset.py`

- [ ] **Step 1: Read the module to understand its current shape**

```bash
grep -n 'async def\|with deriva_call\|asyncio' src/deriva_ml_mcp/tools/asset.py | head -30
```

Note: the `async def` tool definitions, the `with deriva_call():` blocks under each, and whether `import asyncio` already exists.

- [ ] **Step 2: List the callsites that need wrapping (Step A from header)**

```bash
awk '/with deriva_call\(\):/,/^        return|^        except|^    @/' \
    src/deriva_ml_mcp/tools/asset.py \
  | grep -nE '^\s+(ml|ds|exe|wf|exec_obj|_pkg|f|asset|feature)\.|=\s*(ml|ds|exe|wf|exec_obj|_pkg)\.'
```

Expected: ~7 lines. Each is a candidate.

- [ ] **Step 3: Add `import asyncio` if absent**

If `grep -n '^import asyncio' src/deriva_ml_mcp/tools/asset.py` returns no match, add it at the top of the imports block (after `from __future__ import annotations` if present, alongside other stdlib imports). Use the Edit tool against the existing import block — don't blind-add at line 1.

- [ ] **Step 4: Wrap each callsite using the patterns in the per-task header**

Apply the wrapping rules to every line found in Step 2. For each one:
- If the call is a single attribute call: `await asyncio.to_thread(target.method, *args, **kwargs)`.
- If the call returns a generator/iterator: define a nested `def _drain(): return list(...)` and use `await asyncio.to_thread(_drain)`.
- If the line is a property read (no parentheses, no I/O): leave it alone.
- If the line calls a same-module helper that itself does sync work: wrap the helper invocation.

Make these edits one Edit-tool call at a time, working top-down through the file. Don't try to batch all 7 into one Edit.

- [ ] **Step 5: Lint check**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run ruff check src/deriva_ml_mcp/tools/asset.py
```

Expected: clean (or only pre-existing lint warnings unrelated to this change).

- [ ] **Step 6: Run module tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_asset.py tests/test_integration_asset.py -v 2>&1 | tail -20
```

Expected: all tests pass. If any fail, the wrap missed something — review what changed in the failing test's call path.

- [ ] **Step 7: Commit**

```bash
git add src/deriva_ml_mcp/tools/asset.py
git commit -m "fix(asset): wrap sync deriva-ml calls in asyncio.to_thread

Mirrors PR #28's pattern for tools/asset.py. Every sync deriva-ml /
deriva-py call inside an async def tool's deriva_call() block now
runs via await asyncio.to_thread(...), so the asyncio event loop
stays responsive while catalog work runs.

Resolves the missing coverage from PR #28 for this module."
```

---

## Task 2: `tools/feature.py`

**Files:**
- Modify: `src/deriva_ml_mcp/tools/feature.py` (~11 callsites, 756 LoC)
- Test: `tests/test_feature.py`, `tests/test_integration_feature.py`

- [ ] **Step 1: List callsites**

```bash
awk '/with deriva_call\(\):/,/^        return|^        except|^    @/' \
    src/deriva_ml_mcp/tools/feature.py \
  | grep -nE '^\s+(ml|ds|exe|wf|exec_obj|_pkg|f|asset|feature)\.|=\s*(ml|ds|exe|wf|exec_obj|_pkg)\.'
```

Expected: ~11 lines.

- [ ] **Step 2: Add `import asyncio` if absent**

```bash
grep -n '^import asyncio' src/deriva_ml_mcp/tools/feature.py
```

If no match, add via Edit to the imports block.

- [ ] **Step 3: Wrap each callsite**

Apply the pattern from the per-task header.

**Watch out for `ml.feature_record_class(...)`**: it returns a Pydantic class (no I/O), so wrapping is unnecessary. The line `ImageClassification = ml.feature_record_class(...)` does not need `asyncio.to_thread` if the function only constructs and returns a class (look at the deriva-ml source if uncertain).

**Watch out for `feature_record_class().__init__(...)`** and similar — instantiating a Pydantic model is not I/O.

The actual catalog calls are typically `ml.find_features()`, `ml.lookup_feature(...)`, `ml.add_features([...])`, `ml.feature_values(...)`, `ml.delete_feature(...)`. Those need wrapping.

- [ ] **Step 4: Lint check**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run ruff check src/deriva_ml_mcp/tools/feature.py
```

- [ ] **Step 5: Run module tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_feature.py tests/test_integration_feature.py -v 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml_mcp/tools/feature.py
git commit -m "fix(feature): wrap sync deriva-ml calls in asyncio.to_thread

Mirrors PR #28's pattern for tools/feature.py. Every sync deriva-ml /
deriva-py call inside an async def tool's deriva_call() block now
runs via await asyncio.to_thread(...), so the asyncio event loop
stays responsive while catalog work runs.

Resolves the missing coverage from PR #28 for this module."
```

---

## Task 3: `tools/workflow.py`

**Files:**
- Modify: `src/deriva_ml_mcp/tools/workflow.py` (~17 callsites, 597 LoC)
- Test: `tests/test_workflow.py`, `tests/test_integration_workflow.py`

- [ ] **Step 1: List callsites**

```bash
awk '/with deriva_call\(\):/,/^        return|^        except|^    @/' \
    src/deriva_ml_mcp/tools/workflow.py \
  | grep -nE '^\s+(ml|ds|exe|wf|exec_obj|_pkg|f|asset|feature)\.|=\s*(ml|ds|exe|wf|exec_obj|_pkg)\.'
```

Expected: ~17 lines.

- [ ] **Step 2: Add `import asyncio` if absent**

- [ ] **Step 3: Wrap each callsite**

Watch out for: `ml.find_workflows()` (returns list, simple wrap), `ml.lookup_workflow(rid)` (simple wrap), `ml.lookup_workflow_by_url(url)` (simple wrap), `ml.find_workflow_executions(wf_rid)` (simple wrap), `ml.create_workflow(...)` (simple wrap, but this is the one that does the dirty-tree git check — ensure `DERIVA_ML_ALLOW_DIRTY=true` is set during testing). `wf.set_workflow_description(...)` (simple wrap). Property reads (`wf.workflow_rid`, `wf.name`, `wf.description`) need NO wrapping.

- [ ] **Step 4: Lint check**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run ruff check src/deriva_ml_mcp/tools/workflow.py
```

- [ ] **Step 5: Run module tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_workflow.py tests/test_integration_workflow.py -v 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml_mcp/tools/workflow.py
git commit -m "fix(workflow): wrap sync deriva-ml calls in asyncio.to_thread

Mirrors PR #28's pattern for tools/workflow.py. Every sync deriva-ml /
deriva-py call inside an async def tool's deriva_call() block now
runs via await asyncio.to_thread(...), so the asyncio event loop
stays responsive while catalog work runs.

Resolves the missing coverage from PR #28 for this module."
```

---

## Task 4: `tools/execution/read.py`

**Files:**
- Modify: `src/deriva_ml_mcp/tools/execution/read.py` (~13 callsites, 694 LoC)
- Test: `tests/test_execution.py`, `tests/test_integration_execution.py`

- [ ] **Step 1: List callsites**

```bash
awk '/with deriva_call\(\):/,/^        return|^        except|^    @/' \
    src/deriva_ml_mcp/tools/execution/read.py \
  | grep -nE '^\s+(ml|ds|exe|wf|exec_obj|_pkg|f|asset|feature)\.|=\s*(ml|ds|exe|wf|exec_obj|_pkg)\.'
```

Expected: ~13 lines.

- [ ] **Step 2: Add `import asyncio` if absent**

- [ ] **Step 3: Wrap each callsite**

Calls in this module: `ml.find_executions(...)` (list), `ml.lookup_execution(rid)`, `exe.find_workflow_executions(...)`, `_pkg.get_ml(...)`. Property reads (`exe.execution_rid`, `exe.status`) need NO wrapping.

- [ ] **Step 4: Lint check**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run ruff check src/deriva_ml_mcp/tools/execution/read.py
```

- [ ] **Step 5: Run module tests**

`test_execution.py` covers both read and mutate; running it before the next task gives you a checkpoint. Integration tests cover both too.

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_execution.py tests/test_integration_execution.py -v 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml_mcp/tools/execution/read.py
git commit -m "fix(execution.read): wrap sync deriva-ml calls in asyncio.to_thread

Mirrors PR #28's pattern for tools/execution/read.py. Every sync
deriva-ml / deriva-py call inside an async def tool's deriva_call()
block now runs via await asyncio.to_thread(...), so the asyncio
event loop stays responsive while catalog work runs.

Resolves the missing coverage from PR #28 for this module."
```

---

## Task 5: `tools/execution/mutate.py`

**Files:**
- Modify: `src/deriva_ml_mcp/tools/execution/mutate.py` (~23 callsites, 888 LoC)
- Test: `tests/test_execution.py`, `tests/test_integration_execution.py`

- [ ] **Step 1: List callsites**

```bash
awk '/with deriva_call\(\):/,/^        return|^        except|^    @/' \
    src/deriva_ml_mcp/tools/execution/mutate.py \
  | grep -nE '^\s+(ml|ds|exe|wf|exec_obj|_pkg|f|asset|feature)\.|=\s*(ml|ds|exe|wf|exec_obj|_pkg)\.'
```

Expected: ~23 lines.

- [ ] **Step 2: Add `import asyncio` if absent**

- [ ] **Step 3: Wrap each callsite**

Calls: `ml.create_execution(config)` (the lifecycle entry point — slow, must be wrapped), `exe.commit_execution()`, `exe.abort_execution()`, `exe.update_execution(...)`, `ml.add_nested_execution(parent, child)`, `_pkg.get_ml(...)`. ExecutionConfiguration construction is pure-Python (no wrap).

**Special case:** `ml.create_execution(config)` returns an Execution object. The CONSTRUCTION of the config is pure Python; the `create_execution` call is sync I/O. Wrap it as a single call:
`exe = await asyncio.to_thread(ml.create_execution, config)`.

- [ ] **Step 4: Lint check**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run ruff check src/deriva_ml_mcp/tools/execution/mutate.py
```

- [ ] **Step 5: Run module tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_execution.py tests/test_integration_execution.py -v 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml_mcp/tools/execution/mutate.py
git commit -m "fix(execution.mutate): wrap sync deriva-ml calls in asyncio.to_thread

Mirrors PR #28's pattern for tools/execution/mutate.py. Every sync
deriva-ml / deriva-py call inside an async def tool's deriva_call()
block now runs via await asyncio.to_thread(...), so the asyncio
event loop stays responsive while catalog work runs.

Resolves the missing coverage from PR #28 for this module."
```

---

## Task 6: `tools/dataset/read.py`

**Files:**
- Modify: `src/deriva_ml_mcp/tools/dataset/read.py` (~27 callsites, 940 LoC)
- Test: `tests/test_dataset_read.py`

- [ ] **Step 1: List callsites**

```bash
awk '/with deriva_call\(\):/,/^        return|^        except|^    @/' \
    src/deriva_ml_mcp/tools/dataset/read.py \
  | grep -nE '^\s+(ml|ds|exe|wf|exec_obj|_pkg|f|asset|feature)\.|=\s*(ml|ds|exe|wf|exec_obj|_pkg)\.'
```

Expected: ~27 lines.

- [ ] **Step 2: Add `import asyncio` if absent**

- [ ] **Step 3: Wrap each callsite**

This module is large but the calls are all read-side: `ml.find_datasets(...)`, `ml.lookup_dataset(rid)`, `ds.list_dataset_members(...)`, `ds.list_dataset_parents(...)`, `ds.list_dataset_children(...)`, `ds.dataset_history()`, `ml.list_dataset_element_types()`, `ml.bag_info(spec)`, plus same-module helpers (`_list_datasets_impl`, `_summarize_dataset`).

For helpers that already take a `ml` instance and do sync work internally, wrap the **helper invocation**, not the calls inside the helper:
```python
# Before:
payload = _list_datasets_impl(ml, after_rid=after_rid, limit=capped, ...)
# After:
payload = await asyncio.to_thread(_list_datasets_impl, ml, after_rid=after_rid, limit=capped, ...)
```

This module is also where `_summarize_dataset` (`ds.list_dataset_members()` plus aggregation) lives — wrap the function invocation similarly.

For `ds.dataset_history()` returning an iterable that's then list-comprehended (`for h in ds.dataset_history()`), drain via a `_drain()` helper:
```python
def _history_list():
    return [
        DatasetHistoryEntry(...)
        for h in ds.dataset_history()
    ]
history = await asyncio.to_thread(_history_list)
```

- [ ] **Step 4: Lint check**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run ruff check src/deriva_ml_mcp/tools/dataset/read.py
```

- [ ] **Step 5: Run module tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_dataset_read.py -v 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml_mcp/tools/dataset/read.py
git commit -m "fix(dataset.read): wrap sync deriva-ml calls in asyncio.to_thread

Mirrors PR #28's pattern for tools/dataset/read.py. Every sync
deriva-ml / deriva-py call inside an async def tool's deriva_call()
block now runs via await asyncio.to_thread(...), so the asyncio
event loop stays responsive while catalog work runs. Same-module
helpers that themselves contain sync calls are wrapped at the
call site (the helper runs entirely in the worker thread).

Resolves the missing coverage from PR #28 for this module."
```

---

## Task 7: `tools/dataset/mutate.py`

**Files:**
- Modify: `src/deriva_ml_mcp/tools/dataset/mutate.py` (~28 callsites, 734 LoC)
- Test: `tests/test_dataset_mutate.py`

- [ ] **Step 1: List callsites**

```bash
awk '/with deriva_call\(\):/,/^        return|^        except|^    @/' \
    src/deriva_ml_mcp/tools/dataset/mutate.py \
  | grep -nE '^\s+(ml|ds|exe|wf|exec_obj|_pkg|f|asset|feature)\.|=\s*(ml|ds|exe|wf|exec_obj|_pkg)\.'
```

Expected: ~28 lines.

- [ ] **Step 2: Add `import asyncio` if absent**

- [ ] **Step 3: Wrap each callsite**

Calls: `exe.create_dataset(...)`, `ds.add_dataset_members(...)`, `ds.add_dataset_type(...)`, `ds.delete_dataset_members(...)`, `ds.update_dataset(...)`, `ds.increment_version(...)`, `ml.create_execution(...)` (for the dataset-creation execution), `ml.delete_dataset(rid)`. All sync.

**Watch out for:** `ml.create_execution(config)` followed by `exe = ml.create_execution(...)`. The execution context is a regular object after construction, so subsequent `exe.<method>()` calls should each be individually wrapped.

- [ ] **Step 4: Lint check**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run ruff check src/deriva_ml_mcp/tools/dataset/mutate.py
```

- [ ] **Step 5: Run module tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_dataset_mutate.py -v 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml_mcp/tools/dataset/mutate.py
git commit -m "fix(dataset.mutate): wrap sync deriva-ml calls in asyncio.to_thread

Mirrors PR #28's pattern for tools/dataset/mutate.py. Every sync
deriva-ml / deriva-py call inside an async def tool's deriva_call()
block now runs via await asyncio.to_thread(...), so the asyncio
event loop stays responsive while catalog work runs.

Resolves the missing coverage from PR #28 for this module."
```

---

## Task 8: Verify nothing was missed

The grep recipe should now return zero hits for unwrapped sync calls inside async tools. Audit-style verification.

**Files:** none (audit only).

- [ ] **Step 1: Confirm every module has `import asyncio`**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp
for f in src/deriva_ml_mcp/tools/asset.py \
         src/deriva_ml_mcp/tools/feature.py \
         src/deriva_ml_mcp/tools/workflow.py \
         src/deriva_ml_mcp/tools/dataset/read.py \
         src/deriva_ml_mcp/tools/dataset/mutate.py \
         src/deriva_ml_mcp/tools/execution/read.py \
         src/deriva_ml_mcp/tools/execution/mutate.py ; do
  grep -q '^import asyncio' "$f" || echo "MISSING import asyncio: $f"
done
```

Expected: no output (every module has the import).

- [ ] **Step 2: Confirm no unwrapped sync deriva-ml calls remain inside async def tools**

```bash
for f in src/deriva_ml_mcp/tools/asset.py \
         src/deriva_ml_mcp/tools/feature.py \
         src/deriva_ml_mcp/tools/workflow.py \
         src/deriva_ml_mcp/tools/dataset/read.py \
         src/deriva_ml_mcp/tools/dataset/mutate.py \
         src/deriva_ml_mcp/tools/execution/read.py \
         src/deriva_ml_mcp/tools/execution/mutate.py ; do
  echo "=== $f ==="
  awk '/with deriva_call\(\):/,/^        return|^        except|^    @/' "$f" \
    | grep -nE '^\s+(ml|ds|exe|wf|exec_obj|_pkg|f|asset|feature)\.[a-z_]+\(|=\s*(ml|ds|exe|wf|exec_obj|_pkg)\.[a-z_]+\(' \
    | grep -v 'asyncio.to_thread'
done
```

Expected: each `===` heading is followed by no lines (all sync calls now go through `asyncio.to_thread`).

If any line appears, that callsite was missed — go back to the relevant task and wrap it. (False positives are possible — property accesses ending in `()` if any exist; review each finding manually.)

- [ ] **Step 3: Run the full test suite**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest 2>&1 | tail -10
```

Expected: same pass count as the baseline from Task 0 Step 2. Any new failures point at a wrap that broke a behavior — bisect via `git log --oneline main..HEAD` and `git checkout` each commit.

- [ ] **Step 4: Run ruff across the full src tree**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run ruff check src tests
```

Expected: clean (or only pre-existing warnings unrelated to this change).

---

## Task 9: End-to-end smoke against a live catalog

The unit tests pass with mocked deriva-ml. The actual symptom — silent rejection in MCP clients — is only visible against a live host. This task verifies the fix landed for real.

**Files:** none (verification).

**Pre-requisites:**
- A reachable Deriva catalog (localhost or remote). The CIFAR-10 catalog from earlier (catalog 41) is sufficient if it still exists; otherwise use any catalog with at least one Dataset.
- The MCP venv at `/Users/carl/.local/share/deriva-mcp-tool/.venv` consumes deriva-ml-mcp editable, so the changes are live without a reinstall.

- [ ] **Step 1: Restart Claude Desktop / Claude Code**

Restart the MCP host so the `deriva-mcp-core` subprocess respawns and loads the updated plugin. Editable installs DON'T require reinstall, but the running server process needs to re-import the plugin code.

- [ ] **Step 2: Smoke-test the tools that previously rejected**

In your client, request a sequence of `deriva_ml_*` tool calls that hit the previously-failing paths:

```
deriva_ml_list_datasets(hostname="localhost", catalog_id="41")
deriva_ml_list_dataset_relations(hostname="localhost", catalog_id="41", dataset_rid="<a Split RID>")
deriva_ml_list_dataset_members(hostname="localhost", catalog_id="41", dataset_rid="<a leaf RID>")
deriva_ml_get_dataset(hostname="localhost", catalog_id="41", dataset_rid="<any RID>")
deriva_ml_list_workflows(hostname="localhost", catalog_id="41")
deriva_ml_list_executions(hostname="localhost", catalog_id="41")
```

Expected: all calls return data without rejection. Before this fix, `list_dataset_relations` (and others on first use) intermittently rejected silently.

- [ ] **Step 3: Make a catalog-modifying call to exercise the mutate path**

```
deriva_ml_create_workflow(hostname="localhost", catalog_id="41",
                          name="async-test", workflow_type="<existing type>",
                          description="async wrap smoke test")
```

Expected: returns the created workflow's RID. No silent rejection.

- [ ] **Step 4: If catalog 41 doesn't exist, build it**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-sandbox
DERIVA_ML_ALLOW_DIRTY=true uv run python src/scripts/load_cifar10.py \
    --hostname localhost \
    --create-catalog cifar10 \
    --num-images 5000
```

Then redo Steps 2 and 3 against the new catalog ID.

---

## Task 10: Push and open PR

**Files:** none.

- [ ] **Step 1: Verify branch state**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp
git log --oneline main..HEAD
```

Expected: 7 commits, one per module-task (Tasks 1–7). Plus possibly Task 0/8/9 had no commits.

- [ ] **Step 2: Push**

```bash
git push -u origin fix/async-thread-wrap-tools
```

- [ ] **Step 3: Open PR**

```bash
gh pr create --base main --title "fix(tools): wrap sync deriva-ml calls in asyncio.to_thread across all modules" --body "$(cat <<'EOF'
## Summary

Wraps every synchronous deriva-ml / deriva-py call inside an \`async def\` tool with \`await asyncio.to_thread(...)\`, across the seven tool modules PR #28 didn't cover.

## Why

PR #28 (\`fix(complex): wrap sync deriva-ml calls in asyncio.to_thread\`) fixed this for \`tools/dataset/complex.py\`, but the symptom — intermittent silent tool rejection in MCP clients ("user doesn't want to proceed" without any user action) — recurs for every tool in the unfixed modules. The asyncio event loop blocks during sync deriva calls, the host's permission stream times out, pending tool calls reject.

Symptom diagnosed in deriva-ml-sandbox conversation 2026-05-05; mechanism is \`Tool permission stream closed before response received\` from Claude Desktop's main.log.

## Modules covered

- \`tools/asset.py\`
- \`tools/feature.py\`
- \`tools/workflow.py\`
- \`tools/execution/read.py\`
- \`tools/execution/mutate.py\`
- \`tools/dataset/read.py\`
- \`tools/dataset/mutate.py\`

(\`tools/maintenance.py\` doesn't use \`deriva_call()\` blocks the same way and was excluded after audit.)

## Test plan

- [x] Per-module unit + integration tests pass after each commit
- [x] Full \`uv run pytest\` baseline preserved (Task 0 vs Task 8)
- [x] End-to-end smoke against a live catalog (Task 9)
- [x] Lint clean (\`ruff check src tests\`)

## Reference

- Rule documented in \`CLAUDE.md\` → "Tool Implementation Rules" + "Development Gotchas → Sync calls in async tools" (commit 1a13947, 2026-05-05)
- Plan: \`docs/superpowers/plans/2026-05-05-async-thread-wrap-all-tools.md\`
- Precedent: PR #28 (\`fix(complex): wrap sync deriva-ml calls in asyncio.to_thread\`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

---

## Self-Review

**Spec coverage:** Every module identified by the audit has a task. Maintenance.py excluded with reason. The non-trivial cases (generators, helpers) are called out with the canonical pattern. ✓

**Placeholder scan:** No "TBD" / "TODO" / "implement later." Each step has an exact command or an exact code transformation. ✓

**Type consistency:** All task names match across the plan, PR-body bullet points match the actual modules touched, the per-task-header pattern is referenced consistently across tasks. The wrap pattern is shown once (per-task header) and not duplicated per task — readers of any single task get the link to the canonical pattern. ✓

**Edge cases addressed:**
- Generator returns: drain helper pattern (Task header + Task 6 worked example)
- Same-module helpers: wrap at call site, not recursively
- Property reads: explicitly NOT wrapped
- Pydantic class construction (`feature_record_class`): NOT wrapped (no I/O)
- Pure-Python config construction (`ExecutionConfiguration`): NOT wrapped

**Risks the plan should surface:**
- One regression I haven't fully thought through: behavior change from "blocks event loop, then completes" to "yields event loop, then completes." If any test relies on **synchronous ordering** between the tool's internal calls and other concurrent work, that could break. The existing PR #28 tests passed without this issue, so probably fine, but Task 8's full-suite run is the safety net.
- Lint: ruff sometimes flags `asyncio.to_thread(callable, *args)` differently for keyword-only vs positional args. Each module's Step 4 lint check catches this per-commit.
