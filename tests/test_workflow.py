"""Unit tests for workflow domain tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from deriva_ml.core.exceptions import DerivaMLException

from tests.conftest import _success_calls, make_patch_audit

# Dual-patch context manager for the workflow module. See
# ``make_patch_audit`` in ``tests/conftest.py`` for the canonical
# explanation of why both bind sites are patched together.
_patch_workflow_audit = make_patch_audit("workflow")


def _make_workflow_mock(
    rid: str = "1-WF",
    name: str = "MyPipeline",
    workflow_type: list[str] | None = None,
    url: str = "https://github.com/example/repo",
    checksum: str = "abc123",
    version: str = "1.0.0",
    description: str = "",
) -> MagicMock:
    """Build a Workflow-shaped MagicMock.

    The real Workflow class exposes ``rid``, ``name``, ``workflow_type``
    (list[str]), ``url``, ``checksum``, ``version``, and ``description``.
    """
    wf = MagicMock()
    wf.rid = rid
    wf.name = name
    wf.workflow_type = workflow_type or ["Model_Training"]
    wf.url = url
    wf.checksum = checksum
    wf.version = version
    wf.description = description
    return wf


@pytest.fixture()
def workflow_ctx(ctx, mock_ml):
    """Register workflow tools with mock_ml as the DerivaML stand-in.

    Patches at the use-site (`deriva_ml_mcp.tools.workflow.get_ml`) and
    imports the tool module *inside* the patch block so registration sees
    the mock.
    """
    with patch("deriva_ml_mcp.tools.workflow.get_ml", return_value=mock_ml):
        from deriva_ml_mcp.tools import workflow as workflow_module

        workflow_module.register(ctx)
        yield ctx


# ---------------------------------------------------------------------------
# list_workflows
# ---------------------------------------------------------------------------


async def test_list_workflows_success(workflow_ctx, capturing_mcp, mock_ml):
    mock_ml.find_workflows.return_value = [
        _make_workflow_mock(rid="1-AAA", name="A"),
        _make_workflow_mock(rid="1-BBB", name="B"),
    ]
    result = await capturing_mcp.tools["deriva_ml_list_workflows"](hostname="h", catalog_id="1")
    payload = json.loads(result)
    assert payload["count"] == 2
    assert payload["workflows"][0]["rid"] == "1-AAA"
    assert payload["workflows"][1]["rid"] == "1-BBB"
    assert payload["truncated"] is False
    assert payload["next_after_rid"] is None


async def test_list_workflows_preflight(workflow_ctx, capturing_mcp, mock_ml):
    mock_ml.find_workflows.return_value = [_make_workflow_mock(rid=f"1-{i:03d}") for i in range(7)]
    result = await capturing_mcp.tools["deriva_ml_list_workflows"](
        hostname="h", catalog_id="1", preflight_count=True
    )
    payload = json.loads(result)
    assert payload["total_count"] == 7
    assert payload["entities_fetched"] is False
    assert "action_required" in payload


async def test_list_workflows_pagination_cap_and_cursor(workflow_ctx, capturing_mcp, mock_ml):
    mock_ml.find_workflows.return_value = [_make_workflow_mock(rid=f"1-{i:03d}") for i in range(5)]
    # Limit=2: first page yields 1-000, 1-001.
    page1 = json.loads(
        await capturing_mcp.tools["deriva_ml_list_workflows"](hostname="h", catalog_id="1", limit=2)
    )
    assert [w["rid"] for w in page1["workflows"]] == ["1-000", "1-001"]
    assert page1["truncated"] is True
    assert page1["next_after_rid"] == "1-001"

    # Use cursor to fetch the next page.
    page2 = json.loads(
        await capturing_mcp.tools["deriva_ml_list_workflows"](
            hostname="h", catalog_id="1", limit=2, after_rid="1-001"
        )
    )
    assert [w["rid"] for w in page2["workflows"]] == ["1-002", "1-003"]
    assert page2["truncated"] is True
    assert page2["next_after_rid"] == "1-003"


async def test_list_workflows_error_path(workflow_ctx, capturing_mcp, mock_ml):
    mock_ml.find_workflows.side_effect = RuntimeError("kaboom")
    with _patch_workflow_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_list_workflows"](hostname="h", catalog_id="1")
    payload = json.loads(result)
    assert "error" in payload
    assert "kaboom" in payload["error"]
    # Read-only tool: no audit row on failure.
    assert mock_audit.call_count == 0


async def test_list_workflows_sort_forwards_to_deriva_ml(workflow_ctx, capturing_mcp, mock_ml):
    """sort=True forwards sort=True to deriva_ml.find_workflows and skips post-fetch RID sort."""
    wf_a = _make_workflow_mock(rid="1-WF-NEWEST")
    wf_b = _make_workflow_mock(rid="1-WF-OLDER")
    mock_ml.find_workflows.return_value = [wf_a, wf_b]

    result = await capturing_mcp.tools["deriva_ml_list_workflows"](
        hostname="h", catalog_id="1", sort=True
    )
    payload = json.loads(result)

    mock_ml.find_workflows.assert_called_once()
    assert mock_ml.find_workflows.call_args.kwargs.get("sort") is True
    rids = [w["rid"] for w in payload["workflows"]]
    assert rids == ["1-WF-NEWEST", "1-WF-OLDER"]


async def test_list_workflows_sort_default_preserves_rid_sort(workflow_ctx, capturing_mcp, mock_ml):
    """sort=False (default) calls find_workflows with sort=None and re-sorts by RID asc."""
    wf_z = _make_workflow_mock(rid="1-Z")
    wf_a = _make_workflow_mock(rid="1-A")
    mock_ml.find_workflows.return_value = [wf_z, wf_a]

    result = await capturing_mcp.tools["deriva_ml_list_workflows"](hostname="h", catalog_id="1")
    payload = json.loads(result)

    assert mock_ml.find_workflows.call_args.kwargs.get("sort") is None
    rids = [w["rid"] for w in payload["workflows"]]
    assert rids == ["1-A", "1-Z"]


# ---------------------------------------------------------------------------
# get_workflow
# ---------------------------------------------------------------------------


async def test_get_workflow_success(workflow_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_workflow.return_value = _make_workflow_mock(
        rid="1-WF", name="MyPipeline", description="trains models"
    )
    result = await capturing_mcp.tools["deriva_ml_get_workflow"](
        hostname="h", catalog_id="1", workflow_rid="1-WF"
    )
    payload = json.loads(result)
    assert payload["rid"] == "1-WF"
    assert payload["name"] == "MyPipeline"
    assert payload["description"] == "trains models"
    assert payload["workflow_type"] == ["Model_Training"]
    mock_ml.lookup_workflow.assert_called_once_with("1-WF")


async def test_get_workflow_error_path(workflow_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_workflow.side_effect = RuntimeError("not found")
    with _patch_workflow_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_get_workflow"](
            hostname="h", catalog_id="1", workflow_rid="missing"
        )
    payload = json.loads(result)
    assert "error" in payload
    assert "not found" in payload["error"]
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# find_workflow_by_url
# ---------------------------------------------------------------------------


async def test_find_workflow_by_url_success(workflow_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_workflow_by_url.return_value = _make_workflow_mock(
        rid="1-WF", url="https://github.com/example/repo/blob/abc/main.py"
    )
    result = await capturing_mcp.tools["deriva_ml_find_workflow_by_url"](
        hostname="h",
        catalog_id="1",
        url_or_checksum="https://github.com/example/repo/blob/abc/main.py",
    )
    payload = json.loads(result)
    assert payload["rid"] == "1-WF"
    assert payload["url"] == "https://github.com/example/repo/blob/abc/main.py"
    mock_ml.lookup_workflow_by_url.assert_called_once_with(
        "https://github.com/example/repo/blob/abc/main.py"
    )


async def test_find_workflow_by_url_not_found(workflow_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_workflow_by_url.side_effect = DerivaMLException("not found")
    with _patch_workflow_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_find_workflow_by_url"](
            hostname="h", catalog_id="1", url_or_checksum="nope"
        )
    payload = json.loads(result)
    assert "error" in payload
    assert "not found" in payload["error"]
    # Read-only tool: no audit row on failure.
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# create_workflow
# ---------------------------------------------------------------------------


async def test_create_workflow_success_emits_audit(workflow_ctx, capturing_mcp, mock_ml):
    # No existing workflow at this URL.
    mock_ml.lookup_workflow_by_url.side_effect = DerivaMLException("no match")
    # Vocabulary check passes; the plugin now does explicit lookup_term
    # for each workflow_type (previously hidden inside ml.create_workflow).
    mock_ml.lookup_term.return_value = MagicMock()
    mock_ml._add_workflow.return_value = "1-NEW"
    with _patch_workflow_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_create_workflow"](
            hostname="h",
            catalog_id="1",
            name="MyPipeline",
            workflow_type="Model_Training",
            url="https://github.com/example/repo/blob/abc/main.py",
            checksum="abc123",
            version="1.0.0",
            description="trains things",
        )
    payload = json.loads(result)
    assert payload["status"] == "created"
    assert payload["workflow_rid"] == "1-NEW"
    assert payload["name"] == "MyPipeline"
    assert payload["workflow_type"] == ["Model_Training"]
    assert payload["url"] == "https://github.com/example/repo/blob/abc/main.py"
    assert payload["checksum"] == "abc123"
    assert payload["version"] == "1.0.0"
    # ml.create_workflow is no longer called -- the plugin constructs
    # Workflow(...) directly with url/checksum/version at construction
    # time, then hands off to _add_workflow. Vocabulary validation
    # happens via lookup_term BEFORE construction.
    mock_ml.create_workflow.assert_not_called()
    mock_ml.lookup_term.assert_called_once()
    # _add_workflow receives the constructed Workflow with url/checksum
    # /version already set (the v1.36.5+ requirement: setup_url_checksum
    # validator short-circuits at ``if not self.url``).
    mock_ml._add_workflow.assert_called_once()
    wf_arg = mock_ml._add_workflow.call_args.args[0]
    assert wf_arg.url == "https://github.com/example/repo/blob/abc/main.py"
    assert wf_arg.checksum == "abc123"
    assert wf_arg.version == "1.0.0"
    assert wf_arg.name == "MyPipeline"
    # Audit emitted with bounded identifiers; description NOT included.
    success = _success_calls(mock_audit, "deriva_ml_create_workflow")
    assert len(success) == 1
    kwargs = success[0].kwargs
    assert kwargs["workflow_rid"] == "1-NEW"
    assert kwargs["status"] == "created"
    assert kwargs["workflow_type"] == ["Model_Training"]
    assert "description" not in kwargs
    assert "url" not in kwargs


async def test_create_workflow_dedup_exists(workflow_ctx, capturing_mcp, mock_ml):
    """No checksum supplied -> dedup pre-check falls back to URL."""
    existing = _make_workflow_mock(rid="1-OLD")
    mock_ml.lookup_workflow_by_url.return_value = existing
    url = "https://github.com/example/repo/blob/abc/main.py"
    with _patch_workflow_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_create_workflow"](
            hostname="h",
            catalog_id="1",
            name="MyPipeline",
            workflow_type=["Model_Training"],
            url=url,
        )
    payload = json.loads(result)
    assert payload["status"] == "exists"
    assert payload["workflow_rid"] == "1-OLD"
    # With checksum=None, the lookup falls back to URL (matching
    # _add_workflow's `checksum or url` dedup key).
    mock_ml.lookup_workflow_by_url.assert_called_once_with(url)
    # No create / no _add_workflow call.
    mock_ml.create_workflow.assert_not_called()
    mock_ml._add_workflow.assert_not_called()
    # Audit fires with status=exists.
    success = _success_calls(mock_audit, "deriva_ml_create_workflow")
    assert len(success) == 1
    assert success[0].kwargs["status"] == "exists"
    assert success[0].kwargs["workflow_rid"] == "1-OLD"


async def test_create_workflow_dedup_uses_checksum_when_provided(
    workflow_ctx, capturing_mcp, mock_ml
):
    """Dedup pre-check must look up on `checksum or url` (matching
    `_add_workflow`'s internal dedup logic), not URL alone. If a caller
    passes a checksum that matches an existing workflow registered at a
    different URL (e.g. file moved or commit rebased), the pre-check
    must still find it — otherwise the tool reports status='created'
    while `_add_workflow` silently dedups and returns the existing RID.
    Regression test for the checksum-vs-URL mismatch caught in review.
    """
    existing = _make_workflow_mock(rid="1-OLD-CHECKSUM-MATCH")
    mock_ml.lookup_workflow_by_url.return_value = existing
    with _patch_workflow_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_create_workflow"](
            hostname="h",
            catalog_id="1",
            name="MyPipeline",
            workflow_type=["Model_Training"],
            url="https://github.com/example/repo/blob/NEW/main.py",
            checksum="sha256:cafebabe",
        )
    payload = json.loads(result)
    assert payload["status"] == "exists"
    assert payload["workflow_rid"] == "1-OLD-CHECKSUM-MATCH"
    # CRITICAL: lookup_workflow_by_url was called with the checksum
    # (which takes precedence per `checksum or url`), NOT the URL.
    mock_ml.lookup_workflow_by_url.assert_called_once_with("sha256:cafebabe")
    # No create / no _add_workflow call.
    mock_ml.create_workflow.assert_not_called()
    mock_ml._add_workflow.assert_not_called()
    # Audit fires with status=exists.
    success = _success_calls(mock_audit, "deriva_ml_create_workflow")
    assert len(success) == 1
    assert success[0].kwargs["status"] == "exists"


async def test_create_workflow_failure_emits_failed_audit(workflow_ctx, capturing_mcp, mock_ml):
    # Dedup pre-check passes (no existing); vocabulary lookup itself raises.
    # Previously the vocab check lived inside ml.create_workflow; now the
    # plugin calls ml.lookup_term(MLVocab.workflow_type, wt) explicitly
    # for each type BEFORE constructing the Workflow.
    mock_ml.lookup_workflow_by_url.side_effect = DerivaMLException("no match")
    mock_ml.lookup_term.side_effect = RuntimeError("vocab term unknown")
    with _patch_workflow_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_create_workflow"](
            hostname="h",
            catalog_id="1",
            name="MyPipeline",
            workflow_type=["Bogus_Type"],
            url="https://github.com/example/repo/blob/abc/main.py",
        )
    payload = json.loads(result)
    assert "error" in payload
    assert "vocab term unknown" in payload["error"]
    # _add_workflow must NOT be reached when vocab validation fails.
    mock_ml._add_workflow.assert_not_called()
    failed = _success_calls(mock_audit, "deriva_ml_create_workflow_failed")
    assert len(failed) == 1
    kwargs = failed[0].kwargs
    assert kwargs["name"] == "MyPipeline"
    assert kwargs["workflow_type"] == ["Bogus_Type"]
    assert kwargs["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# update_workflow
# ---------------------------------------------------------------------------


async def test_update_workflow_description_only_emits_audit(workflow_ctx, capturing_mcp, mock_ml):
    wf = _make_workflow_mock(rid="1-WF")
    mock_ml.lookup_workflow.return_value = wf
    with _patch_workflow_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_update_workflow"](
            hostname="h",
            catalog_id="1",
            workflow_rid="1-WF",
            description="new desc",
        )
    payload = json.loads(result)
    assert payload["status"] == "updated"
    assert payload["workflow_rid"] == "1-WF"
    assert payload["updated_fields"] == ["description"]
    # Setter assignment fires the catalog-bound write hook on the mock.
    assert wf.description == "new desc"
    success = _success_calls(mock_audit, "deriva_ml_update_workflow")
    assert len(success) == 1
    kwargs = success[0].kwargs
    assert kwargs["workflow_rid"] == "1-WF"
    assert kwargs["updated_fields"] == ["description"]
    assert "description" not in kwargs
    # workflow_type=None passed through (no change).
    assert kwargs["workflow_type"] is None


async def test_update_workflow_workflow_type_only_emits_audit(workflow_ctx, capturing_mcp, mock_ml):
    wf = _make_workflow_mock(rid="1-WF")
    mock_ml.lookup_workflow.return_value = wf
    with _patch_workflow_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_update_workflow"](
            hostname="h",
            catalog_id="1",
            workflow_rid="1-WF",
            workflow_type=["Embedding"],
        )
    payload = json.loads(result)
    assert payload["status"] == "updated"
    assert payload["workflow_rid"] == "1-WF"
    assert payload["updated_fields"] == ["workflow_type"]
    assert wf.workflow_type == ["Embedding"]
    success = _success_calls(mock_audit, "deriva_ml_update_workflow")
    assert len(success) == 1
    assert success[0].kwargs["workflow_type"] == ["Embedding"]
    assert success[0].kwargs["updated_fields"] == ["workflow_type"]


async def test_update_workflow_validation_both_none(workflow_ctx, capturing_mcp, mock_ml):
    with _patch_workflow_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_update_workflow"](
            hostname="h", catalog_id="1", workflow_rid="1-WF"
        )
    payload = json.loads(result)
    assert "error" in payload
    assert "at least one" in payload["error"]
    # No catalog work and no audit on validation failure.
    mock_ml.lookup_workflow.assert_not_called()
    assert mock_audit.call_count == 0


async def test_update_workflow_validation_empty_list(workflow_ctx, capturing_mcp, mock_ml):
    with _patch_workflow_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_update_workflow"](
            hostname="h",
            catalog_id="1",
            workflow_rid="1-WF",
            workflow_type=[],
        )
    payload = json.loads(result)
    assert "error" in payload
    assert "cannot be empty" in payload["error"]
    mock_ml.lookup_workflow.assert_not_called()
    assert mock_audit.call_count == 0


async def test_update_workflow_failure_emits_failed_audit(workflow_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_workflow.side_effect = RuntimeError("workflow not found")
    with _patch_workflow_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_update_workflow"](
            hostname="h",
            catalog_id="1",
            workflow_rid="1-MISSING",
            description="new desc",
        )
    payload = json.loads(result)
    assert "error" in payload
    assert "workflow not found" in payload["error"]
    failed = _success_calls(mock_audit, "deriva_ml_update_workflow_failed")
    assert len(failed) == 1
    kwargs = failed[0].kwargs
    assert kwargs["workflow_rid"] == "1-MISSING"
    assert kwargs["updated_fields"] == []
    assert kwargs["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# v1.3: surgical RAG re-index wired into mutating workflow tools
# ---------------------------------------------------------------------------


async def test_create_workflow_triggers_surgical_reindex(workflow_ctx, capturing_mcp, mock_ml):
    """``deriva_ml_create_workflow`` calls ``_reindex_workflow`` with the new RID."""
    from unittest.mock import AsyncMock

    mock_ml.lookup_workflow_by_url.side_effect = DerivaMLException("no match")
    # ml.create_workflow is no longer called; the plugin constructs
    # Workflow(...) directly under v1.36.5+. lookup_term replaces the
    # vocab check that used to live inside create_workflow.
    mock_ml.lookup_term.return_value = MagicMock()
    mock_ml._add_workflow.return_value = "1-NEW"
    fake_reindex = AsyncMock(return_value=1)
    with (
        patch("deriva_ml_mcp.resources.rag._reindex_workflow", new=fake_reindex),
        _patch_workflow_audit(),
    ):
        await capturing_mcp.tools["deriva_ml_create_workflow"](
            hostname="h",
            catalog_id="1",
            name="P",
            workflow_type="Model_Training",
            url="https://github.com/x/r/blob/abc/main.py",
        )
    fake_reindex.assert_awaited_once_with("h", "1", "1-NEW")
