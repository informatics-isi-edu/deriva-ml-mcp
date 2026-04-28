# Test Coverage Audit — deriva-ml-mcp

_Audited: 2026-04-27 against commit HEAD (v3.0.0)._

## Part A: Quantitative Coverage

### Overall

- **287 tests collected** (14 deselected — integration suite gated by `@pytest.mark.integration`)
- **287 pass / 0 fail / 0 skip**
- **Overall line coverage: 95%** (1763 / 1848 statements)
- **Time: 3.57 s**

### Per-module coverage

| Module | Coverage | Uncovered lines |
|---|---|---|
| `__init__.py` | 100% | — |
| `_helpers.py` | 86% | 200, 320-325 |
| `_response_models.py` | 100% | — |
| `_version.py` | 0% | 3-24 (auto-generated; excluded by design) |
| `ml_context.py` | 100% | — |
| `plugin.py` | 100% | — |
| `prompts.py` | 100% | — |
| `resources/__init__.py` | 100% | — |
| `resources/ml.py` | 95% | 320-321, 348-349, 376-377 |
| `resources/rag.py` | 94% | 182, 311-313, 318-320, 325-333, 513-520, 539-546, 902 |
| `tools/__init__.py` | 100% | — |
| `tools/asset.py` | 100% | — |
| `tools/dataset/__init__.py` | 100% | — |
| `tools/dataset/complex.py` | 97% | 317, 521-522 |
| `tools/dataset/mutate.py` | 94% | 216-217, 322-323, 402-403, 550-551, 714-715 |
| `tools/execution.py` | 91% | 252-253, 272-273, 529-530, 767-768, 862-863, 922-923, 978-979, 1070-1071, 1155-1156, 1246-1247, 1319-1320 |
| `tools/feature.py` | 96% | 360, 363, 372, 379-380 |
| `tools/maintenance.py` | 100% | — |
| `tools/workflow.py` | 96% | 410-411, 514-515 |

**Modules with <80% coverage:** `_helpers.py` (86%) and `_version.py` (0%, intentionally excluded). No module other than the auto-generated version file is below 80%.

**Uncovered line analysis:**

- `_helpers.py:200` — the `KeyError` branch in `_read_rid` when a dict member has neither `"RID"` nor the configured `rid_key`. Never hit directly; the helper is exercised through domain tests but only via object-path (not dict-path with a missing key).
- `_helpers.py:320-325` — the fallback branches in `_row_rid_for`'s inner `extract()` closure: the `"RID" in row` branch (line 320-321) and the `.endswith(".RID")` loop (lines 322-325). Only the `preferred` branch is exercised in practice.
- `resources/ml.py:320-321, 348-349, 376-377` — error-envelope catch blocks for the `ml/asset-tables`, `ml/asset/{rid}`, and `ml/registries` resources. Happy paths are tested; no error-path tests for these three resources (see Part B).
- `resources/rag.py:311-313, 318-320, 325-333` — the `_fetch_dataset_rows`, `_fetch_workflow_rows`, and `_fetch_execution_rows` helpers. These are called via the `on_catalog_connect` hooks; the hook tests mock the store interaction rather than invoking these functions through the hook, so the function bodies go untested. Lines 513-520 and 539-546 are the exception-swallowing branches in `_reindex_workflow` and `_reindex_execution` (only the dataset exception swallow is tested in `test_rag.py`). Line 902 is the `continue` branch in `_write_vocab_chunks` when a serializer returns `None` for a term — `test_write_vocab_chunks_empty_terms_drains_only` does not hit the serializer-returns-None branch inside the loop.
- `tools/dataset/mutate.py:216-217, 322-323, 402-403, 550-551, 714-715` — all are `_reindex_dataset` exception-swallowing catch blocks (the `except Exception: logger...; return 0` pattern). Only `create_dataset`'s reindex failure test exists; the analogous branches for `add_dataset_members`, `delete_dataset_members`, `update_dataset`, and `increment_dataset_version` are not exercised.
- `tools/execution.py` uncovered pairs — most are `_reindex_execution` exception-swallowing blocks (`862-863, 922-923, 978-979, 1070-1071, 1155-1156, 1319-1320`). Lines `252-253` and `272-273` are the exception-swallow branches in `_get_execution_detail_impl` for `list_assets` failures (input and output asset enumeration). Lines `529-530` are the `preflight_count=True` branch inside `find_workflow_executions` — this tool has no preflight test.
- `tools/workflow.py:410-411, 514-515` — exception-swallowing blocks for `_reindex_workflow` in `create_workflow` and `update_workflow`. Only `create_workflow`'s reindex is tested; `update_workflow`'s reindex failure block is untested.

---

## Part B: Qualitative Coverage

### Happy path vs. error path

Balance is excellent across all five domains. Every tool in the dataset, workflow, execution, feature, and asset modules has at minimum one success test and one error test. Read-only tools (`list_*`, `get_*`) confirm `mock_audit.call_count == 0` on failure, and mutating tools confirm `_success_calls(mock_audit, "deriva_ml_<op>_failed")` is non-empty. No tool is left with a happy-path test only.

The one partial gap: `find_workflow_executions` in `tools/execution.py` has success and error tests but no `preflight_count=True` test, leaving line 529-530 uncovered. The equivalent branch in `list_executions`, `list_datasets`, `list_workflows`, and `list_features` is covered.

### Wire-shape assertions

Wire-shape discipline is very strong. Every test that calls a tool does:

```python
payload = json.loads(result)
assert payload["status"] == "created"
assert payload["workflow_rid"] == "1-NEW"
```

No test inspects internal Python state instead of the JSON wire. The `extra="forbid"` Pydantic models provide an additional safety net: if a helper builds the wrong shape, `model_dump_json` would raise a `ValidationError` and the test would fail at the `json.loads` step or see `{"error": ...}` in the payload.

### Audit assertions

All `mutates=True` tools have dual-patch audit coverage via `make_patch_audit(module)` or `_patch_audit()` context managers. Tests assert both the positive event (`_success_calls(mock_audit, "deriva_ml_<op>")`) and the negative event (`_success_calls(mock_audit, "deriva_ml_<op>_failed")`). Spot checks:

- `test_create_workflow_success_emits_audit` (workflow.py:227): asserts `kwargs["workflow_rid"]`, `kwargs["status"]`, and confirms `"description" not in kwargs` (which verifies cardinality of the audit payload, not just presence).
- `test_create_execution_failure_emits_failed_audit` (execution.py:377): asserts `error_type` in failed audit kwargs.

`mutates=False` tools (maintenance, read-only tools) are tested with `mock_audit.assert_not_called()` to confirm no audit fires.

### Dual-mode (tool + resource) coverage

Per the plugin's dual-mode policy, every domain ships both a tool and a resource. Coverage:

| Domain | Tool tests | Resource tests |
|---|---|---|
| Dataset | 72 tests (read.py, mutate.py, complex.py split) | `test_ml_datasets_*`, `test_ml_dataset_detail_*`, `test_ml_dataset_members_*` |
| Workflow | 18 tests | `test_ml_workflows_*`, `test_ml_workflow_detail_*` |
| Execution | 40 tests | `test_ml_executions_*`, `test_ml_execution_detail_*` |
| Feature | 25 tests | `test_ml_features_for_table_*` |
| Asset | 17 tests | `test_ml_asset_tables_smoke`, `test_ml_asset_detail_*` |

Each resource has at least one happy-path test. Error-path coverage is partial: the dataset, workflow, execution, and feature resources have error-path tests, but the three asset/registry resources (`ml/asset-tables`, `ml/asset/{rid}`, `ml/registries`) are missing error-path tests. This leaves their exception-envelope catch blocks (lines 320-321, 348-349, 376-377 in `resources/ml.py`) uncovered.

### RAG re-index pattern coverage

The pattern is well tested for the `create_*` branch of each domain that participates in RAG:

- `test_create_dataset_triggers_surgical_reindex` — asserts `_reindex_dataset` awaited with correct `(host, cat, rid)`.
- `test_create_workflow_triggers_surgical_reindex` — same for workflow.
- `test_create_execution_triggers_surgical_reindex` — same for execution.
- `test_create_execution_dry_run_skips_reindex` — confirms no reindex on dry-run.
- `test_create_execution_dataset_triggers_both_reindexes` — two separate asserts for dataset and execution reindex.
- `test_create_dataset_reindex_failure_does_not_fail_tool` — confirms the exception-swallow policy.

**Gaps:** `update_workflow` calls `_reindex_workflow` (lines 511-513 in `workflow.py`) but there is no test asserting it. Multiple `_reindex_dataset` exception-swallow branches for `add_dataset_members`, `delete_dataset_members`, `update_dataset`, and `increment_dataset_version` are untested (mutate.py lines 322-323, 402-403, 550-551, 714-715). The `_reindex_workflow` and `_reindex_execution` exception-swallow blocks are also untested (workflow.py 410-411; execution.py 862-863, etc.). Feature tools do not call any reindex function, which is correct — features are not in the per-user RAG index.

### Pagination edge cases

Coverage is thorough for the primary paginated tools:

- **Small page / cursor advancement:** tested for `list_datasets`, `list_workflows`, `list_executions`, `list_dataset_members` — each has a two-page cursor walk test.
- **Cap to `_MAX_LIMIT`:** `test_list_datasets_pagination_caps_limit_and_advances_cursor` explicitly passes 2500 rows and verifies cap to 1000.
- **`preflight_count=True` branch:** tested for `list_datasets`, `list_workflows`, `list_executions`, `list_features`, `list_feature_values`. The `preflight_count` assertions are minimal (`"action_required" in out`) — they confirm key presence but not the value of `action_required`. A stronger assertion would pin the human-readable message text.
- **`direction="both"` cursor incoherence warning:** `test_list_dataset_relations_after_rid_with_both_emits_warning` exists and asserts `"warning" in out`.
- **`after_rid` respects single-direction:** `test_list_dataset_relations_after_rid_respected_in_single_direction` exists.

**Gap:** `find_workflow_executions` has no `preflight_count=True` test (lines 528-530 in `execution.py` uncovered). `list_execution_children` and `list_execution_parents` have no cursor advancement tests — only success and error paths.

### v3.0 wire-break coverage (11 wire breaks)

Per the v3.0 migration map in CLAUDE.md, the 11 breaks are:

| Wire break | Test assertion |
|---|---|
| `add_dataset_members`: `status="added"` | `test_add_dataset_members_success`: `assert out["status"] == "added"` ✓ |
| `delete_dataset_members`: `status="removed"` | `test_delete_dataset_members_success`: `assert out["status"] == "removed"` ✓ |
| `add_dataset_element_type`: `status="created"` | `test_add_dataset_element_type_success`: `assert out == {"status": "created", ...}` ✓ |
| `increment_dataset_version`: `status="incremented"` | test asserts `assert out["status"] == "incremented"` ✓ |
| `cache_dataset`: `status="cached"` | `test_cache_dataset_success` asserts `out["status"] == "cached"` ✓ |
| `cache_dataset`: bag-info nested under `bag_info` | `assert out["bag_info"]["cache_status"] == "cached_materialized"` ✓ |
| `split_dataset`: `status="split"` | asserts `out["status"] == "split"` ✓ |
| `start_execution` no-op: `status="already_running"` (no `note`) | `test_start_execution_idempotent_when_already_running`: `assert payload["status"] == "already_running"` and `assert "note" not in payload` ✓ |
| `abort_execution` no-op: `status="already_aborted"` | `test_abort_execution_idempotent_when_already_aborted`: `assert payload["status"] == "already_aborted"` ✓ |
| `abort_execution`: `reason` always present | `assert payload["reason"] is None` ✓ |
| `update_dataset` description-only: `dataset_types/added/removed` always present (null) | `test_update_dataset_description_only`: `assert out["dataset_types"] is None; assert out["added"] is None; assert out["removed"] is None` ✓ |

**All 11 v3.0 wire breaks are explicitly tested by value-level assertions.** This is the strongest part of the test suite.

One marginal gap: `create_execution_dataset`'s v3.0 contract says `dataset_types` is always present (null when omitted by caller). The only success test (`test_create_execution_dataset_success_emits_audit`) passes `dataset_types=["Training"]` explicitly; there is no test passing `dataset_types=None` and asserting `payload["dataset_types"] is None`.

### Mocking discipline

Mocking is well-structured. Centralized fixtures in `conftest.py` (`ctx`, `mock_ml`, `capturing_mcp`) are used consistently. Each domain module adds a local `<domain>_ctx` fixture that registers its tools under `mock_ml`.

**Duplication:** `_make_dataset_mock`, `_make_workflow_mock`, `_make_execution_record_mock`, and `_make_feature_mock` factories are defined once in each of `test_dataset.py`, `test_resources.py`, `test_execution.py`, and `test_workflow.py`. The resource test file re-defines all four locally (e.g., `test_resources.py:60-130`). This is not incorrect — the shapes differ slightly between tool and resource tests — but there is non-trivial duplication. A shared `tests/_factories.py` would DRY these up.

Direct `MagicMock()` construction is used inside test functions and local factory helpers (e.g., `test_resources.py:60`, `test_feature.py:20`). This is appropriate: the factories are lean and specific to their context. No test constructs a god-MagicMock that preconfigures 15+ attributes.

### Async test harness

`pyproject.toml` sets `asyncio_mode = "auto"`. All async test functions (`async def test_*`) run automatically without `@pytest.mark.asyncio` decorators. The harness is `pytest-asyncio 1.3.0`. No inconsistency found — there are no sync wrapper functions calling `asyncio.run()` or mixing anyio with asyncio. This is applied uniformly across all 12 test files.

### Test naming + organization

Test names follow the pattern `test_<tool_name>_<scenario>` consistently. Finding the tests for a given tool is straightforward: `grep "def test_" tests/test_workflow.py` returns names like `test_create_workflow_success_emits_audit`, `test_create_workflow_dedup_exists`, etc.

Tests within each file are organized in sections delimited by `# -----------` comments matching the tool name. This is uniform across all domain test files.

No orphan tests were found. All tool names referenced in test functions correspond to tools registered in `test_plugin.py::test_all_registered_tools_exact`.

---

## Part C: Test Design Quality

### Test interdependence

Tests are independent. Each `async def test_*` function receives fresh fixture instances (all fixtures are function-scoped except the integration session fixtures). Running `uv run pytest tests/test_workflow.py` alone yields 18 pass, confirming no hidden state from other modules leaks in. The `_set_plugin_context(None)` teardown in `conftest.py`'s `ctx` fixture (line 33) cleans up the contextvar between tests.

### Fixture quality

Fixtures are focused. `conftest.py` provides exactly three function-scoped fixtures (`capturing_mcp`, `ctx`, `mock_ml`) plus two session-scoped integration fixtures. Each domain file adds one `<domain>_ctx` fixture that does a single job: patch `get_ml`, import the module, call `register(ctx)`, yield.

The `_CapturingMCP` class in `tests/_helpers.py` is a clean stand-in that only stores decoratee references — no side effects, no state bleed between tests.

### Test brittleness

Low brittleness overall. Tests assert on behavior (JSON payload shape, method call counts/arguments) rather than log message text. Exception: error-path tests do string-match the exception message (`assert "kaboom" in payload["error"]`), which is slightly brittle to rewording — but this is a minor concern.

Two patterns that could become brittle:

1. `test_plugin.py::test_all_registered_tools_exact` uses an exact frozenset equality check. This is intentionally strict (it catches both missing and extra tools), so brittleness here is by design and desirable.
2. `test_prompts.py::test_prompts_are_ascii` could give a misleading failure if the `prompts.py` template strings ever intentionally include Unicode (e.g., arrow characters in formatting). The 100-char floor test is similarly a weak proxy for "prompt text exists."

### Assertion specificity

Assertion specificity is high in tool tests. The standard pattern:

```python
payload = json.loads(result)
assert payload["status"] == "created"
assert payload["workflow_rid"] == "1-NEW"
```

is used throughout and tests exact values.

Weaker assertions exist in a small number of places:

- **Preflight `action_required`:** `assert "action_required" in out` (7 occurrences across test files) confirms key presence but not the message content. This matters because a broken preflight that returned an empty string would pass.
- **Error message substring:** `assert "kaboom" in payload["error"]` is a substring match, not an exact comparison. Acceptable for readability, but a future refactor that changes error wrapping could obscure the original message.
- **`test_ml_asset_tables_smoke`:** named "smoke" — confirms `count == 2` and the set of names, but does not verify the per-table `schema` field shape.

---

## Top-line findings

- **Coverage is genuinely strong at 95% overall.** All 287 unit tests pass, all 11 v3.0 wire-break contracts are verified by value-level assertions, and every mutating tool has both success-path and failure-path audit assertions. This is a well-maintained test suite.

- **Three resources are missing error-path tests.** `ml/asset-tables`, `ml/asset/{rid}`, and `ml/registries` in `resources/ml.py` have no exception-path tests, leaving their catch blocks (lines 320-321, 348-349, 376-377) uncovered. Each needs a single test that makes `get_ml` raise and asserts `"error" in payload`.

- **Reindex coverage is incomplete for update operations.** `update_workflow` calls `_reindex_workflow` (workflow.py:511-513) but has no test asserting this. Multiple mutating dataset tools (`add_dataset_members`, `delete_dataset_members`, `update_dataset`, `increment_dataset_version`) have no tests for their reindex exception-swallow branches. These are not critical bugs but are observable gaps given the thorough create-path reindex tests.

- **`find_workflow_executions` is missing a `preflight_count=True` test.** It's the only paginated tool without this coverage (lines 528-530 in execution.py uncovered). A two-line test modeled on `test_list_executions_preflight` would close it.

- **Mock factory duplication across test files.** `_make_dataset_mock`, `_make_workflow_mock`, and `_make_execution_record_mock` are redefined independently in `test_resources.py`, `test_execution.py`, and `test_workflow.py`. Negligible runtime impact, but a shape change to the mock API requires updates in multiple places. Consolidating into `tests/_factories.py` (parallel to `tests/_helpers.py`) would improve maintainability.
