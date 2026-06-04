# Read-Through RAG Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut per-connection RAG startup overhead by replacing the three per-user bulk on-connect indexers (Dataset/Workflow/Execution) with read-through (index-on-find) indexing, while keeping vocabulary always-indexed on connect.

**Architecture:** Remove only the `_make_hook` on-connect bulk-pass wiring. Add a fire-and-forget "index-on-find" helper that the dataset/workflow/execution list+get tools call after building their response, warming each returned row's per-user RAG source newest-first. Add client-side `dataset_type`/`workflow_type` filters to the dataset/workflow list helpers for a consistent structured-filter surface. The existing surgical `_reindex_*` writers, the `deriva_ml_resync_indexes` warm-button, and all vocabulary machinery stay.

**Tech Stack:** Python 3 / asyncio, deriva-ml, deriva-mcp-core plugin API, Pydantic response models, pytest, `uv` for all tooling.

**Repo:** `deriva-ml-mcp-plugin` (sibling of this worktree). All paths below are relative to that repo root. Work on branch `feat/read-through-rag-indexing` (already created; the design spec is committed there).

**Reference spec:** `docs/superpowers/specs/2026-06-04-drop-per-user-row-indexing-design.md`

**Tooling reminders:**
- Always `uv run <cmd>` — never call `pytest`/`ruff` directly.
- Run tests with `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest ...`.
- Lint/format touched files: `uv run ruff check src tests` and `uv run ruff format src tests`.
- House style: `from __future__ import annotations`, unquoted annotations, Google-style docstrings with `Example:`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/deriva_ml_mcp_plugin/resources/rag.py` | RAG indexing wiring + helpers | Remove 3 `_make_hook` on-connect registrations; add `_index_rows_on_find` fire-and-forget helper; update module docstring + `register_rag_sources` docstring |
| `src/deriva_ml_mcp_plugin/tools/dataset/read.py` | Dataset list/get tools + impls | Add `dataset_type` filter to `_list_datasets_impl`; surface `dataset_type` param on `deriva_ml_list_datasets`; call index-on-find in list + get |
| `src/deriva_ml_mcp_plugin/tools/workflow.py` | Workflow list/get tools + impls | Add `workflow_type` filter to `_list_workflows_impl`; surface param on `deriva_ml_list_workflows`; call index-on-find in list + get |
| `src/deriva_ml_mcp_plugin/tools/execution/read.py` | Execution list/get tools + impls | Call index-on-find in `deriva_ml_list_executions` + `deriva_ml_get_execution` (filters already exist) |
| `tests/test_rag.py` | RAG helper unit tests | Drop the 3 bulk-hook tests; add `_index_rows_on_find` tests |
| `tests/test_plugin.py` | Plugin wiring tests | `four_catalog_connect_hooks` → one-hook; drop per-RID-naming bulk test; fix vocab-hook index `[3]`→`[0]` |
| `tests/test_dataset_read.py` / `tests/test_workflow*.py` / `tests/test_execution*.py` | Read-tool tests | Add type-filter tests + index-on-find scheduling tests |

The index-on-find helper lives in `rag.py` (next to the `_reindex_*` writers it reuses) and is called from the read tools via lazy import — same cycle-avoidance idiom the mutation tools already use for `_reindex_dataset`.

---

## Task 1: Add the `_index_rows_on_find` fire-and-forget helper

**Files:**
- Modify: `src/deriva_ml_mcp_plugin/resources/rag.py`
- Test: `tests/test_rag.py`

This helper takes rows already in hand (from a list/get response), sorts them newest-first, and schedules a background per-RID reindex for each. It must never block or raise into the caller. It mirrors the established fire-and-forget idiom in `deriva-mcp-core/src/deriva_mcp_core/tools/catalog.py:158-186` (`create_task` + module-level task set + `add_done_callback`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_rag.py`:

```python
def test_index_rows_on_find_schedules_reindex_per_rid() -> None:
    """``_index_rows_on_find`` fires one reindex coroutine per row, newest-first."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from deriva_ml_mcp_plugin.resources import rag as rag_module

    # Rows as the list impls produce them (RID + an RCT for sort order).
    rows = [
        {"rid": "1-OLD", "rct": "2026-01-01T00:00:00+00:00"},
        {"rid": "1-NEW", "rct": "2026-06-01T00:00:00+00:00"},
    ]
    calls: list[str] = []

    async def fake_reindex(hostname, catalog_id, rid):
        calls.append(rid)
        return 1

    async def run() -> None:
        with patch.object(rag_module, "_reindex_dataset", new=AsyncMock(side_effect=fake_reindex)):
            rag_module._index_rows_on_find(
                "h", "1", rag_module._DATASET_TOKEN, rows, rct_key="rct"
            )
            # Let the scheduled background tasks run to completion.
            await asyncio.gather(*rag_module._on_find_tasks)

    asyncio.run(run())

    # Both RIDs indexed, newest-first (1-NEW before 1-OLD).
    assert calls == ["1-NEW", "1-OLD"]


def test_index_rows_on_find_swallows_reindex_errors() -> None:
    """A failing reindex never propagates out of the background task."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from deriva_ml_mcp_plugin.resources import rag as rag_module

    rows = [{"rid": "1-AAAA", "rct": "2026-01-01T00:00:00+00:00"}]

    async def run() -> None:
        boom = AsyncMock(side_effect=RuntimeError("rag boom"))
        with patch.object(rag_module, "_reindex_dataset", new=boom):
            rag_module._index_rows_on_find("h", "1", rag_module._DATASET_TOKEN, rows, rct_key="rct")
            # gather must not raise even though the reindex raised.
            await asyncio.gather(*rag_module._on_find_tasks, return_exceptions=True)

    asyncio.run(run())  # no exception escapes


def test_index_rows_on_find_no_running_loop_is_noop() -> None:
    """Called with no running event loop, the helper is a silent no-op (import-time safety)."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    # Not inside asyncio.run -> no running loop. Must not raise.
    rag_module._index_rows_on_find(
        "h", "1", rag_module._DATASET_TOKEN, [{"rid": "1-AAAA"}], rct_key="rct"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_rag.py -k index_rows_on_find -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_index_rows_on_find'`.

- [ ] **Step 3: Implement the helper**

In `src/deriva_ml_mcp_plugin/resources/rag.py`, add near the `_reindex_*` helpers (after `_reindex_execution`). First add a module-level task set next to the other module state (near the top, after the constants):

```python
# Strong references to in-flight index-on-find background tasks. Without
# this set the event loop may garbage-collect a pending task before it
# runs (asyncio holds only a weak reference). Mirrors the
# ``_connect_tasks`` pattern in deriva-mcp-core/tools/catalog.py.
_on_find_tasks: set[asyncio.Task] = set()

# Maps a per-user-trio table token to the surgical re-index coroutine
# that warms one row of that table. Used by ``_index_rows_on_find``.
_REINDEX_BY_TOKEN: dict[str, Callable[[str, str, str], Awaitable[int]]] = {}
```

Then add the helper (the `_REINDEX_BY_TOKEN` map is populated after the
`_reindex_*` functions are defined — add the three assignments directly
below `_reindex_execution`):

```python
_REINDEX_BY_TOKEN[_DATASET_TOKEN] = _reindex_dataset
_REINDEX_BY_TOKEN[_WORKFLOW_TOKEN] = _reindex_workflow
_REINDEX_BY_TOKEN[_EXECUTION_TOKEN] = _reindex_execution


def _index_rows_on_find(
    hostname: str,
    catalog_id: str,
    table_token: str,
    rows: list[dict[str, Any]],
    *,
    rct_key: str = "rct",
) -> None:
    """Warm the calling user's per-RID RAG sources for rows just read.

    Read-through (index-on-find) indexing: a list/get tool that just
    fetched ``rows`` calls this to schedule a best-effort background
    re-index of each row, so a later ``rag_search`` finds them. Rows are
    warmed **newest-first** (descending ``rct_key``) so the most recently
    created entities become searchable first.

    Fire-and-forget: schedules one background task per row and returns
    immediately. Never blocks the caller's response and never raises --
    a missing event loop (import-time / sync test) is a silent no-op, and
    each per-row reindex swallows its own errors (see ``_reindex_*``).

    Args:
        hostname: Deriva hostname.
        catalog_id: Catalog ID or alias.
        table_token: One of ``_DATASET_TOKEN`` / ``_WORKFLOW_TOKEN`` /
            ``_EXECUTION_TOKEN``. Selects the per-row reindex coroutine.
        rows: Summary-shape row dicts (must carry ``"rid"``). Rows
            without a RID are skipped.
        rct_key: Key holding each row's record-creation timestamp, used
            only to order the warm newest-first. Rows missing the key
            sort last (treated as oldest).

    Returns:
        None.

    Example:
        >>> _index_rows_on_find(  # doctest: +SKIP
        ...     "host", "1", _DATASET_TOKEN,
        ...     [{"rid": "1-AAAA", "rct": "2026-06-01T00:00:00+00:00"}],
        ... )
    """
    reindex_fn = _REINDEX_BY_TOKEN.get(table_token)
    if reindex_fn is None:
        return
    # Newest-first: rows with a later rct warm sooner. Missing/empty rct
    # sorts last (empty string < any real ISO timestamp).
    ordered = sorted(rows, key=lambda r: r.get(rct_key) or "", reverse=True)

    async def _do() -> None:
        for row in ordered:
            rid = row.get("rid")
            if not rid:
                continue
            try:
                await reindex_fn(hostname, catalog_id, rid)
            except Exception:  # noqa: BLE001 -- best-effort warm
                logger.debug(
                    "index-on-find reindex failed for %s %s in %s/%s",
                    table_token,
                    rid,
                    hostname,
                    catalog_id,
                    exc_info=True,
                )

    try:
        task = asyncio.get_running_loop().create_task(_do())
        _on_find_tasks.add(task)
        task.add_done_callback(_on_find_tasks.discard)
    except RuntimeError:
        # No running event loop (import-time, sync test) -- silent no-op.
        pass
```

Verify `Callable` and `Awaitable` are already imported in `rag.py` (they are — used by `_make_vocab_hook`'s return annotation). `Any` is imported. No new imports needed.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_rag.py -k index_rows_on_find -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + format**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run ruff check src/deriva_ml_mcp_plugin/resources/rag.py tests/test_rag.py && uv run ruff format src/deriva_ml_mcp_plugin/resources/rag.py tests/test_rag.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin
git add src/deriva_ml_mcp_plugin/resources/rag.py tests/test_rag.py
git commit -m "feat(rag): add _index_rows_on_find read-through warm helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Remove the three on-connect bulk indexers

**Files:**
- Modify: `src/deriva_ml_mcp_plugin/resources/rag.py`
- Test: `tests/test_plugin.py`, `tests/test_rag.py`

Delete only the `_make_hook` factory and its three `ctx.on_catalog_connect(...)` registrations. The `_fetch_*_rows` fetchers, serializers, `_reindex_*`, and `_resync_*` all stay (the kept `deriva_ml_resync_indexes` path uses them).

- [ ] **Step 1: Update the wiring test to expect one hook**

In `tests/test_plugin.py`, change `test_register_wires_four_catalog_connect_hooks`:

```python
def test_register_wires_one_catalog_connect_hook(ctx):
    """``register(ctx)`` wires exactly the vocab catalog-connect hook.

    Read-through indexing removed the three per-user bulk row indexers
    (Dataset/Workflow/Execution). Only the catalog-public vocab indexer
    fires on connect now. Hook identity is covered in test_rag.py; this
    pins the count so re-adding a bulk hook is caught at the plugin level.
    """
    register(ctx)
    assert len(ctx._catalog_connect_hooks) == 1
```

In the same file, update `test_vocab_hook_writes_to_vocab_prefix_not_data_prefix`: change `vocab_hook = ctx._catalog_connect_hooks[3]` to `ctx._catalog_connect_hooks[0]`.

Delete `test_data_sources_use_per_rid_naming` entirely (it asserts the bulk hooks write per-RID sources on connect; those hooks no longer exist — the per-RID naming is still covered by the `_reindex_*` tests in test_rag.py).

- [ ] **Step 2: Run to verify the new/changed tests fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_plugin.py -k "one_catalog_connect_hook or vocab_hook_writes" -v`
Expected: `test_register_wires_one_catalog_connect_hook` FAILS (`assert 4 == 1`); `vocab_hook_writes` FAILS (`IndexError`/wrong hook at `[0]`) — both because the bulk hooks are still registered.

- [ ] **Step 3: Remove the bulk-hook registrations**

In `src/deriva_ml_mcp_plugin/resources/rag.py`, in `register_rag_sources`, delete these three blocks (the Dataset/Workflow/Execution registrations), leaving the two `rag_github_source` calls and the vocab hook:

```python
    ctx.on_catalog_connect(
        _make_hook(_fetch_dataset_rows, _DATASET_TABLE, _DATASET_TOKEN, _DatasetSerializer())
    )
    ctx.on_catalog_connect(
        _make_hook(_fetch_workflow_rows, _WORKFLOW_TABLE, _WORKFLOW_TOKEN, _WorkflowSerializer())
    )
    ctx.on_catalog_connect(
        _make_hook(
            _fetch_execution_rows, _EXECUTION_TABLE, _EXECUTION_TOKEN, _ExecutionSerializer()
        )
    )
```

After deletion the only remaining `ctx.on_catalog_connect(...)` call is `ctx.on_catalog_connect(_make_vocab_hook())`.

- [ ] **Step 4: Delete the `_make_hook` factory**

In the same file, delete the entire `def _make_hook(...)` function (the per-user bulk-pass factory, roughly the block whose `hook` ignores `schema_hash`/`schema_json` and loops `_write_row_chunk` over fetched rows). Do NOT delete `_make_vocab_hook`, `_fetch_*_rows`, the serializers, `_reindex_*`, `_resync_*`, `_row_source_name`, `_write_row_chunk`, or `_coerce_for_index`.

- [ ] **Step 5: Update the module + `register_rag_sources` docstrings**

In `rag.py`, edit the module docstring: the numbered list item describing "Three per-user-per-RID `on_catalog_connect` hooks" should now describe them as **read-through (index-on-find)** writers warmed by the list/get tools, not on-connect hooks. Edit `register_rag_sources`'s docstring so the "four catalog-connect hooks" wording becomes "the GitHub doc sources and the vocabulary catalog-connect hook" (one hook). Keep all factual references to the `data:` per-RID source shape — that shape is unchanged.

- [ ] **Step 6: Drop the now-dead bulk-hook tests in test_rag.py**

Delete from `tests/test_rag.py` the tests that fired `_make_hook` on-connect bulk behavior:
`test_dataset_hook_writes_one_per_rid_source_per_row`,
`test_workflow_hook_writes_one_per_rid_source_per_row`,
`test_execution_hook_writes_one_per_rid_source_per_row`,
`test_dataset_hook_swallows_fetch_exception`,
`test_workflow_hook_isolates_per_row_write_failures`,
`test_dataset_hook_skips_rows_without_rid`,
`test_dataset_hook_partitions_per_user`.
Keep the serializer tests, `_row_source_name` test, `_write_row_chunk` test, all `_reindex_*` tests, all vocab tests, and the new `_index_rows_on_find` tests from Task 1.

- [ ] **Step 7: Run the affected tests**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_plugin.py tests/test_rag.py -v`
Expected: PASS. No reference to a deleted `_make_hook`/hook test remains.

- [ ] **Step 8: Grep for stragglers**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && grep -rn "_make_hook\b" src tests`
Expected: no matches (only `_make_vocab_hook` should exist, which won't match `_make_hook\b` followed by word boundary — verify the grep returns nothing for the bare `_make_hook`).

- [ ] **Step 9: Lint + format + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin
uv run ruff check src tests && uv run ruff format src/deriva_ml_mcp_plugin/resources/rag.py tests/test_rag.py tests/test_plugin.py
git add src/deriva_ml_mcp_plugin/resources/rag.py tests/test_rag.py tests/test_plugin.py
git commit -m "refactor(rag): drop on-connect bulk row indexers (keep vocab hook)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add `dataset_type` filter to the dataset list path

**Files:**
- Modify: `src/deriva_ml_mcp_plugin/tools/dataset/read.py`
- Test: `tests/test_dataset_read.py` (create if absent; otherwise add to the existing dataset-read test module)

`DatasetSummary.dataset_types` is a **list** — the filter is a membership test, applied to the fetched rows before pagination.

- [ ] **Step 1: Write the failing test**

First confirm the test module name: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && ls tests | grep -i dataset`. Add to the dataset-read test file (use `tests/test_dataset_read.py`; create it with the standard imports if it does not exist). The test drives `_list_datasets_impl` with a fake `ml`:

```python
def test_list_datasets_impl_filters_by_dataset_type() -> None:
    """``dataset_type=`` keeps only datasets whose type list contains it."""
    from types import SimpleNamespace

    from deriva_ml_mcp_plugin.tools.dataset.read import _list_datasets_impl

    def _ds(rid, types):
        return SimpleNamespace(
            dataset_rid=rid,
            description="d",
            dataset_types=types,
            current_version=None,
        )

    fake_ml = SimpleNamespace(
        find_datasets=lambda deleted, sort: [
            _ds("1-TRN", ["Training"]),
            _ds("1-TST", ["Testing"]),
            _ds("1-BTH", ["Training", "Validation"]),
        ]
    )

    resp = _list_datasets_impl(
        fake_ml, after_rid=None, limit=100, dataset_type="Training"
    )
    rids = [d.rid for d in resp.datasets]
    assert rids == ["1-BTH", "1-TRN"]  # RID-ascending of the two Training matches


def test_list_datasets_impl_dataset_type_none_returns_all() -> None:
    """No filter -> unchanged behavior (all datasets)."""
    from types import SimpleNamespace

    from deriva_ml_mcp_plugin.tools.dataset.read import _list_datasets_impl

    def _ds(rid, types):
        return SimpleNamespace(
            dataset_rid=rid, description="d", dataset_types=types, current_version=None
        )

    fake_ml = SimpleNamespace(
        find_datasets=lambda deleted, sort: [_ds("1-A", ["Training"]), _ds("1-B", ["Testing"])]
    )
    resp = _list_datasets_impl(fake_ml, after_rid=None, limit=100, dataset_type=None)
    assert {d.rid for d in resp.datasets} == {"1-A", "1-B"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_dataset_read.py -k dataset_type -v`
Expected: FAIL with `TypeError: _list_datasets_impl() got an unexpected keyword argument 'dataset_type'`.

- [ ] **Step 3: Implement the filter**

In `src/deriva_ml_mcp_plugin/tools/dataset/read.py`, add a `dataset_type` parameter to `_list_datasets_impl` and apply the membership filter to the fetched rows **before** `_paginate`:

```python
def _list_datasets_impl(
    ml: Any,
    *,
    after_rid: str | None,
    limit: int,
    include_deleted: bool = False,
    sort: bool = False,
    dataset_type: str | None = None,
) -> DatasetListResponse:
```

Add to the `Args:` docstring:

```
        dataset_type: Optional ``Dataset_Type`` vocabulary term. When
            set, keep only datasets whose ``dataset_types`` list contains
            this term (membership test, applied before pagination so
            ``limit`` counts filtered rows). When ``None`` (default), no
            type filtering.
```

In the body, after building `datasets` (the sorted/raw list) and before `_paginate(...)`, insert:

```python
    if dataset_type is not None:
        datasets = [d for d in datasets if dataset_type in (d.dataset_types or [])]
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_dataset_read.py -k dataset_type -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint + format + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin
uv run ruff check src/deriva_ml_mcp_plugin/tools/dataset/read.py tests/test_dataset_read.py && uv run ruff format src/deriva_ml_mcp_plugin/tools/dataset/read.py tests/test_dataset_read.py
git add src/deriva_ml_mcp_plugin/tools/dataset/read.py tests/test_dataset_read.py
git commit -m "feat(dataset): client-side dataset_type filter in _list_datasets_impl

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Surface `dataset_type` on the `deriva_ml_list_datasets` tool

**Files:**
- Modify: `src/deriva_ml_mcp_plugin/tools/dataset/read.py`
- Test: `tests/test_dataset_read.py`

- [ ] **Step 1: Write the failing test**

This test calls the registered tool through the capturing-MCP fixture. Match the existing fixture pattern in the dataset-read tests (look at how a sibling test obtains `capturing_mcp` / a `ctx`). The assertion: passing `dataset_type` forwards it into `_list_datasets_impl`.

```python
def test_list_datasets_tool_forwards_dataset_type(dataset_ctx, capturing_mcp, mock_ml) -> None:
    """``deriva_ml_list_datasets(dataset_type=...)`` forwards the filter to the impl."""
    from unittest.mock import patch

    register(dataset_ctx)  # however the existing tests register; mirror them
    seen = {}

    real_impl = _list_datasets_impl

    def spy(ml, **kwargs):
        seen.update(kwargs)
        # Return an empty page so the tool completes.
        return DatasetListResponse(datasets=[], count=0, truncated=False, next_after_rid=None)

    with patch("deriva_ml_mcp_plugin.tools.dataset.read._list_datasets_impl", side_effect=spy):
        import asyncio

        asyncio.run(
            capturing_mcp.tools["deriva_ml_list_datasets"](
                hostname="h", catalog_id="1", dataset_type="Training"
            )
        )
    assert seen.get("dataset_type") == "Training"
```

> Note for the implementer: adapt `dataset_ctx`/`capturing_mcp`/`mock_ml`/`register` to the exact fixtures already used by the other tests in this file. If the file registers tools differently, copy that setup verbatim — do not invent a new harness.

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_dataset_read.py -k forwards_dataset_type -v`
Expected: FAIL — the tool rejects the unknown `dataset_type` kwarg (`TypeError`) or never forwards it.

- [ ] **Step 3: Add the param to the tool**

In `deriva_ml_list_datasets`, add `dataset_type: str | None = None` to the signature (after `sort`), document it in `Args:`, and forward it in the `_list_datasets_impl` call:

```python
                payload = await asyncio.to_thread(
                    _list_datasets_impl,
                    ml,
                    after_rid=after_rid,
                    limit=capped,
                    include_deleted=include_deleted,
                    sort=sort,
                    dataset_type=dataset_type,
                )
```

Add to the tool docstring `Args:`:

```
            dataset_type: Optional ``Dataset_Type`` term. When set, only
                datasets tagged with this type are returned (structured
                filter -- prefer this over fuzzy ``rag_search`` when the
                user names a type). Combine with a later ``rag_search``
                for "Training datasets matching <description text>".
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_dataset_read.py -v`
Expected: PASS.

- [ ] **Step 5: Lint + format + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin
uv run ruff check src/deriva_ml_mcp_plugin/tools/dataset/read.py tests/test_dataset_read.py && uv run ruff format src/deriva_ml_mcp_plugin/tools/dataset/read.py tests/test_dataset_read.py
git add src/deriva_ml_mcp_plugin/tools/dataset/read.py tests/test_dataset_read.py
git commit -m "feat(dataset): expose dataset_type filter on deriva_ml_list_datasets

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add `workflow_type` filter to the workflow list path + tool

**Files:**
- Modify: `src/deriva_ml_mcp_plugin/tools/workflow.py`
- Test: `tests/test_workflow.py` (use the existing workflow test module)

`WorkflowSummary.workflow_type` is also a **list** — same membership-test pattern as Task 3, done in one task for impl + tool since the workflow file holds both.

- [ ] **Step 1: Write the failing tests**

Confirm the module: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && ls tests | grep -i workflow`. Add:

```python
def test_list_workflows_impl_filters_by_workflow_type() -> None:
    """``workflow_type=`` keeps only workflows whose type list contains it."""
    from types import SimpleNamespace

    from deriva_ml_mcp_plugin.tools.workflow import _list_workflows_impl

    def _wf(rid, types):
        return SimpleNamespace(
            workflow_rid=rid, name="w", url="u", checksum="c", version="1",
            workflow_type=types, description="d",
        )

    fake_ml = SimpleNamespace(
        find_workflows=lambda sort: [
            _wf("1-TRN", ["Model_Training"]),
            _wf("1-INF", ["Inference"]),
        ]
    )
    resp = _list_workflows_impl(
        fake_ml, after_rid=None, limit=100, workflow_type="Model_Training"
    )
    assert [w.rid for w in resp.workflows] == ["1-TRN"]
```

> If `_summarize_workflow` calls `_resolve_workflow_rid(wf)` and that needs more than `workflow_rid` on the fake, set `workflow_rid` directly on the `SimpleNamespace` (as above) so the resolver's happy path returns it. If the resolver still fails on a bare namespace, patch `_resolve_workflow_rid` to return the row's `workflow_rid` for this test.

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_workflow.py -k workflow_type -v`
Expected: FAIL with `TypeError: _list_workflows_impl() got an unexpected keyword argument 'workflow_type'`.

- [ ] **Step 3: Implement the impl filter**

In `src/deriva_ml_mcp_plugin/tools/workflow.py`, add `workflow_type: str | None = None` to `_list_workflows_impl`, document it, and filter after the rows are fetched/sorted and before `_paginate`:

```python
    if workflow_type is not None:
        workflows = [w for w in workflows if workflow_type in (w.workflow_type or [])]
```

(Use whatever the local variable holding the post-sort list is named — mirror the dataset impl. If the impl summarizes inside the paginate call, filter on the raw objects' `.workflow_type` before summarizing.)

- [ ] **Step 4: Add the param to the tool**

In `deriva_ml_list_workflows`, add `workflow_type: str | None = None` to the signature, document it in `Args:` (mirror the dataset wording — "structured filter, prefer over fuzzy rag_search"), and forward it into the `_list_workflows_impl` call.

- [ ] **Step 5: Add the tool-forwarding test**

Mirror Task 4 Step 1's spy test for `deriva_ml_list_workflows` + `workflow_type`, using the workflow test file's fixtures.

- [ ] **Step 6: Run to verify all pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_workflow.py -k workflow_type -v`
Expected: PASS.

- [ ] **Step 7: Lint + format + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin
uv run ruff check src/deriva_ml_mcp_plugin/tools/workflow.py tests/test_workflow.py && uv run ruff format src/deriva_ml_mcp_plugin/tools/workflow.py tests/test_workflow.py
git add src/deriva_ml_mcp_plugin/tools/workflow.py tests/test_workflow.py
git commit -m "feat(workflow): client-side workflow_type filter on list path + tool

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Wire index-on-find into the dataset read tools

**Files:**
- Modify: `src/deriva_ml_mcp_plugin/tools/dataset/read.py`
- Test: `tests/test_dataset_read.py`

After `deriva_ml_list_datasets` builds `payload` and after `deriva_ml_get_dataset` builds its detail, schedule the read-through warm. Use the lazy-import idiom (avoid the rag↔tools cycle). The warm uses the rows already in `payload`.

- [ ] **Step 1: Write the failing test**

```python
def test_list_datasets_tool_schedules_index_on_find(dataset_ctx, capturing_mcp, mock_ml) -> None:
    """``deriva_ml_list_datasets`` warms the returned rows via _index_rows_on_find."""
    from unittest.mock import patch

    register(dataset_ctx)  # mirror existing registration
    captured = {}

    def fake_warm(hostname, catalog_id, token, rows, **kwargs):
        captured["token"] = token
        captured["rids"] = [r.get("rid") for r in rows]

    with patch(
        "deriva_ml_mcp_plugin.resources.rag._index_rows_on_find", side_effect=fake_warm
    ):
        import asyncio

        # mock_ml.find_datasets should yield at least one dataset with a RID.
        asyncio.run(
            capturing_mcp.tools["deriva_ml_list_datasets"](hostname="h", catalog_id="1")
        )

    from deriva_ml_mcp_plugin.resources.rag import _DATASET_TOKEN

    assert captured.get("token") == _DATASET_TOKEN
    assert captured.get("rids")  # at least one RID warmed
```

> Adapt `mock_ml` so `find_datasets` returns datasets with RIDs (mirror what the existing list-datasets happy-path test uses). The warm receives **dicts** (`payload.datasets` model-dumped) — see Step 3 for how the tool produces them.

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_dataset_read.py -k schedules_index_on_find -v`
Expected: FAIL — `_index_rows_on_find` is never called (`captured` empty).

- [ ] **Step 3: Call the warm in the list tool**

In `deriva_ml_list_datasets`, after `payload` is built and **before** `return payload.model_dump_json(...)` (and only on the non-preflight path), add:

```python
                # Read-through indexing: warm the per-user RAG sources for
                # the rows we just returned so a later rag_search finds
                # them (newest-first). Fire-and-forget; never blocks or
                # fails the read. Lazy import avoids the rag<->tools cycle.
                try:
                    from deriva_ml_mcp_plugin.resources.rag import (
                        _DATASET_TOKEN,
                        _index_rows_on_find,
                    )

                    _index_rows_on_find(
                        hostname,
                        catalog_id,
                        _DATASET_TOKEN,
                        [d.model_dump(mode="json") for d in payload.datasets],
                    )
                except Exception:  # noqa: BLE001 -- warm is best-effort
                    logger.debug("index-on-find scheduling failed for list_datasets", exc_info=True)
            return payload.model_dump_json(by_alias=True)
```

`dataset/read.py` does **not** define a module logger (verified). Add it once near the top of the file, after the imports:

```python
import logging

logger = logging.getLogger(__name__)
```

(If `import logging` already exists, add only the `logger = ...` line. `workflow.py` already defines a module-level `logger`, so Task 7 needs no logger addition. `execution/read.py` does **not** have a module-level `logger` — Task 8 adds one, see its Step 3 note.)

> `DatasetSummary` carries no `rct` field, so the newest-first sort falls back to "no rct → all equal → stable order." That is acceptable: the list is already returned RID-ascending (or RCT-desc when `sort=True`), and warm order within a single page is a minor optimization. The rct-based ordering matters most for the resync/bulk path; for a single page the in-hand order is fine. (No extra work — the helper tolerates missing `rct`.)

- [ ] **Step 4: Wire the get tool**

In `deriva_ml_get_dataset`, after the detail payload is built and before returning, add the same warm for the single row:

```python
                try:
                    from deriva_ml_mcp_plugin.resources.rag import (
                        _DATASET_TOKEN,
                        _index_rows_on_find,
                    )

                    _index_rows_on_find(
                        hostname, catalog_id, _DATASET_TOKEN, [{"rid": dataset_rid}]
                    )
                except Exception:  # noqa: BLE001 -- warm is best-effort
                    logger.debug("index-on-find scheduling failed for get_dataset", exc_info=True)
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_dataset_read.py -v`
Expected: PASS.

- [ ] **Step 6: Lint + format + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin
uv run ruff check src/deriva_ml_mcp_plugin/tools/dataset/read.py tests/test_dataset_read.py && uv run ruff format src/deriva_ml_mcp_plugin/tools/dataset/read.py tests/test_dataset_read.py
git add src/deriva_ml_mcp_plugin/tools/dataset/read.py tests/test_dataset_read.py
git commit -m "feat(dataset): index-on-find warm in list/get dataset tools

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Wire index-on-find into the workflow read tools

**Files:**
- Modify: `src/deriva_ml_mcp_plugin/tools/workflow.py`
- Test: `tests/test_workflow.py`

- [ ] **Step 1: Write the failing test**

Mirror Task 6 Step 1 for `deriva_ml_list_workflows`, asserting `_index_rows_on_find` is called with `_WORKFLOW_TOKEN` and the returned RIDs.

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_workflow.py -k schedules_index_on_find -v`
Expected: FAIL.

- [ ] **Step 3: Call the warm in `deriva_ml_list_workflows`**

After `payload` is built, before returning:

```python
                try:
                    from deriva_ml_mcp_plugin.resources.rag import (
                        _WORKFLOW_TOKEN,
                        _index_rows_on_find,
                    )

                    _index_rows_on_find(
                        hostname,
                        catalog_id,
                        _WORKFLOW_TOKEN,
                        [w.model_dump(mode="json") for w in payload.workflows],
                    )
                except Exception:  # noqa: BLE001 -- warm is best-effort
                    logger.debug("index-on-find scheduling failed for list_workflows", exc_info=True)
```

- [ ] **Step 4: Wire the workflow get tool (if present)**

Find the single-workflow read tool: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && grep -n "deriva_ml_get_workflow\|_get_workflow_impl" src/deriva_ml_mcp_plugin/tools/workflow.py`. If a `deriva_ml_get_workflow` tool exists, add the single-row warm (mirror Task 6 Step 4 with `_WORKFLOW_TOKEN` and `[{"rid": workflow_rid}]`). If no get-workflow tool exists, skip this step (note it in the commit).

- [ ] **Step 5: Run to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_workflow.py -v`
Expected: PASS.

- [ ] **Step 6: Lint + format + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin
uv run ruff check src/deriva_ml_mcp_plugin/tools/workflow.py tests/test_workflow.py && uv run ruff format src/deriva_ml_mcp_plugin/tools/workflow.py tests/test_workflow.py
git add src/deriva_ml_mcp_plugin/tools/workflow.py tests/test_workflow.py
git commit -m "feat(workflow): index-on-find warm in list/get workflow tools

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Wire index-on-find into the execution read tools

**Files:**
- Modify: `src/deriva_ml_mcp_plugin/tools/execution/read.py`
- Test: `tests/test_execution.py` (the existing execution test module)

Note: `ExecutionSummary` carries **no** creation-timestamp field (its fields are `rid`, `workflow_rid`, etc. — verified in `_response_models.py`). So the warm cannot sort executions newest-first from the summary; it warms them in the page order the tool already returns (which is RCT-desc when the caller passes `sort=True`, RID-asc otherwise). Pass no `rct_key` override — the helper's missing-key fallback ("all equal → stable order") is exactly right here. Newest-first warm ordering is realized at the page level via `sort=True`, not inside the warm.

- [ ] **Step 1: Write the failing test**

Mirror Task 6 Step 1 for `deriva_ml_list_executions`, asserting `_index_rows_on_find` is called with `_EXECUTION_TOKEN` and the returned RIDs. Also add one for `deriva_ml_get_execution` asserting a single-row warm. Use the fixtures already in `tests/test_execution.py`.

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_execution.py -k schedules_index_on_find -v`
Expected: FAIL.

- [ ] **Step 3: Call the warm in `deriva_ml_list_executions`**

`execution/read.py` has no module-level `logger` (only a function-local `import logging` at ~line 648). Add a module logger near the top after the imports:

```python
import logging

logger = logging.getLogger(__name__)
```

(If `import logging` is already at module scope, add only the `logger = ...` line.)

Then, after `payload` is built, before returning:

```python
                try:
                    from deriva_ml_mcp_plugin.resources.rag import (
                        _EXECUTION_TOKEN,
                        _index_rows_on_find,
                    )

                    _index_rows_on_find(
                        hostname,
                        catalog_id,
                        _EXECUTION_TOKEN,
                        [e.model_dump(mode="json") for e in payload.executions],
                        # No rct_key: ExecutionSummary has no timestamp field;
                        # warm in the tool's returned page order.
                    )
                except Exception:  # noqa: BLE001 -- warm is best-effort
                    logger.debug("index-on-find scheduling failed for list_executions", exc_info=True)
```

- [ ] **Step 4: Call the warm in `deriva_ml_get_execution`**

After the single execution is fetched/summarized, before returning, add the single-row warm (`[{"rid": execution_rid}]`, `_EXECUTION_TOKEN`).

- [ ] **Step 5: Run to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_execution.py -v`
Expected: PASS.

- [ ] **Step 6: Lint + format + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin
uv run ruff check src/deriva_ml_mcp_plugin/tools/execution/read.py tests/test_execution.py && uv run ruff format src/deriva_ml_mcp_plugin/tools/execution/read.py tests/test_execution.py
git add src/deriva_ml_mcp_plugin/tools/execution/read.py tests/test_execution.py
git commit -m "feat(execution): index-on-find warm in list/get execution tools

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Full suite + lint gate

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest -q`
Expected: all pass. If anything in `test_integration*.py` asserts that a Dataset/Workflow/Execution row is a `rag_search` hit *purely from on-connect bulk indexing*, update it to first call the corresponding `deriva_ml_list_*` tool (which now warms the index) before asserting the search hit — this matches the read-through model. Do NOT weaken assertions that exercise the kept `deriva_ml_resync_indexes` path.

- [ ] **Step 2: Lint + format the whole tree**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run ruff check src tests && uv run ruff format --check src tests`
Expected: clean. Fix any residue and re-run.

- [ ] **Step 3: Grep for dangling references to removed wiring**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && grep -rn "four_catalog_connect\|four catalog-connect\|_catalog_connect_hooks\[3\]\|_make_hook\b" src tests docs`
Expected: no matches except intentional history in the design spec. Fix any stragglers in code/tests.

- [ ] **Step 4: Commit any cleanup**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin
git add -A
git commit -m "test: align integration tests with read-through indexing

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" || echo "nothing to commit"
```

---

## Task 10: Update CLAUDE.md + module docs to describe read-through indexing

**Files:**
- Modify: `src/deriva_ml_mcp_plugin/resources/rag.py` (module docstring — if not already fully updated in Task 2)
- Modify: `CLAUDE.md` (repo root — the RAG/indexing description, if present)

- [ ] **Step 1: Check CLAUDE.md for indexing claims**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && grep -n "on_catalog_connect\|on connect\|bulk\|per-user\|index\|rag" CLAUDE.md`
Identify any sentence stating the plugin bulk-indexes Dataset/Workflow/Execution rows on connect.

- [ ] **Step 2: Update the wording**

Edit those sentences to state: schema (core) + vocabulary index on connect; Dataset/Workflow/Execution rows index **read-through** (on create/mutate and on list/get/find), warmed newest-first; `deriva_ml_resync_indexes` remains the manual warm-everything button. Keep it to a few sentences — match CLAUDE.md's existing density.

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin
git add CLAUDE.md src/deriva_ml_mcp_plugin/resources/rag.py
git commit -m "docs: describe read-through RAG indexing model

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Follow-on (separate repo, NOT part of this plan)

`deriva-skills` — update `skills/semantic-awareness/SKILL.md` and `references/find-before-you-create.md` to (1) route structured finds (type/status/RID/URL) to deterministic `find_*`/`list_*` instead of `rag_search`; (2) describe the warm-before-rank ordering (list/find warms the index, then `rag_search` ranks); (3) note vocab terms are always warm while data rows warm on first browse; (4) route workflow dedup to `find_workflow_by_url`. Tracked as its own change in the `deriva-skills` repo.

---

## Self-Review Notes

- **Spec coverage:** drop bulk hooks (Task 2 ✓), keep vocab on connect (untouched ✓), index-on-find helper + newest-first (Task 1 ✓), wire list+get for all three entities (Tasks 6/7/8 ✓), `dataset_type`/`workflow_type` filters (Tasks 3/4/5 ✓), keep `deriva_ml_resync_indexes` + machinery (Task 2 keeps `_fetch_*_rows`/`_resync_*` ✓), keep serializers/`_reindex_*` (Task 2 ✓), docs (Task 10 ✓), skill follow-on flagged out-of-scope ✓.
- **Type consistency:** `_index_rows_on_find(hostname, catalog_id, table_token, rows, *, rct_key)` used identically in Tasks 1/6/7/8; token constants `_DATASET_TOKEN`/`_WORKFLOW_TOKEN`/`_EXECUTION_TOKEN` referenced consistently; `_REINDEX_BY_TOKEN` maps them to the existing `_reindex_*` coroutines.
- **Known adaptation points (flagged inline, not placeholders):** exact test-fixture names (`dataset_ctx`/`capturing_mcp`/`mock_ml`) and the execution summary's RCT field name must be read from the existing test files / source at implementation time — each such step says so explicitly and gives the grep to find it.
