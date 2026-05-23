"""Unit tests for execution domain tools."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from deriva_ml.execution.execution import ExecutionStatus

from tests._helpers import make_patch_audit

# Dual-patch context manager for the execution module. See
# ``make_patch_audit`` in ``tests/_helpers.py`` for the canonical
# explanation of why both bind sites are patched together.
_patch_execution_audit = make_patch_audit("execution")


def _make_execution_record_mock(
    rid: str = "1-EXEC",
    workflow_rid: str = "1-WF",
    status: ExecutionStatus = ExecutionStatus.Created,
    description: str = "",
    start_time: datetime | None = None,
    stop_time: datetime | None = None,
    duration: str | None = None,
) -> MagicMock:
    """Build an ``ExecutionRecord``-shaped MagicMock.

    The real ``ExecutionRecord`` exposes ``execution_rid``, ``workflow``
    (with ``.rid``), ``workflow_rid``, ``status``, ``description``,
    ``start_time``, ``stop_time``, and ``duration``.
    """
    record = MagicMock()
    record.execution_rid = rid
    record.workflow_rid = workflow_rid
    workflow = MagicMock()
    workflow.rid = workflow_rid
    record.workflow = workflow
    record.status = status
    record.description = description
    record.start_time = start_time
    record.stop_time = stop_time
    record.duration = duration
    return record


def _make_execution_mock(
    execution_rid: str = "1-EXEC",
    status: ExecutionStatus = ExecutionStatus.Created,
) -> MagicMock:
    """Build an ``Execution``-shaped MagicMock.

    The real ``Execution`` exposes ``execution_rid`` (str) and a
    ``status`` property (ExecutionStatus, read-through from SQLite).
    """
    execution = MagicMock()
    execution.execution_rid = execution_rid
    execution.status = status
    return execution


@pytest.fixture()
def execution_ctx(ctx, mock_ml):
    """Register execution tools with mock_ml as the DerivaML stand-in.

    Patches at the use-site (``deriva_ml_mcp.tools.execution.get_ml``) and
    imports the tool module *inside* the patch block so registration sees
    the mock.
    """
    with patch("deriva_ml_mcp.tools.execution.get_ml", return_value=mock_ml):
        from deriva_ml_mcp.tools import execution as execution_module

        execution_module.register(ctx)
        yield ctx


# ---------------------------------------------------------------------------
# list_executions
# ---------------------------------------------------------------------------


async def test_list_executions_success(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.find_executions.return_value = [
        _make_execution_record_mock(rid="1-AAA", workflow_rid="1-WF"),
        _make_execution_record_mock(rid="1-BBB", workflow_rid="1-WF"),
    ]
    result = await capturing_mcp.tools["deriva_ml_list_executions"](hostname="h", catalog_id="1")
    payload = json.loads(result)
    assert payload["count"] == 2
    assert payload["executions"][0]["rid"] == "1-AAA"
    assert payload["executions"][1]["rid"] == "1-BBB"
    assert payload["truncated"] is False
    assert payload["next_after_rid"] is None


async def test_list_executions_preflight(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.find_executions.return_value = [
        _make_execution_record_mock(rid=f"1-{i:03d}") for i in range(5)
    ]
    result = await capturing_mcp.tools["deriva_ml_list_executions"](
        hostname="h", catalog_id="1", preflight_count=True
    )
    payload = json.loads(result)
    assert payload["total_count"] == 5
    assert payload["entities_fetched"] is False
    assert "action_required" in payload


async def test_list_executions_pagination_cap_and_cursor(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.find_executions.return_value = [
        _make_execution_record_mock(rid=f"1-{i:03d}") for i in range(5)
    ]
    page1 = json.loads(
        await capturing_mcp.tools["deriva_ml_list_executions"](
            hostname="h", catalog_id="1", limit=2
        )
    )
    assert [e["rid"] for e in page1["executions"]] == ["1-000", "1-001"]
    assert page1["truncated"] is True
    assert page1["next_after_rid"] == "1-001"

    page2 = json.loads(
        await capturing_mcp.tools["deriva_ml_list_executions"](
            hostname="h", catalog_id="1", limit=2, after_rid="1-001"
        )
    )
    assert [e["rid"] for e in page2["executions"]] == ["1-002", "1-003"]


async def test_list_executions_error_path(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.find_executions.side_effect = RuntimeError("kaboom")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_list_executions"](
            hostname="h", catalog_id="1"
        )
    payload = json.loads(result)
    assert "error" in payload
    assert "kaboom" in payload["error"]
    # Read-only tool: no audit row on failure.
    assert mock_audit.call_count == 0


async def test_list_executions_status_filter_passes_enum(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.find_executions.return_value = []
    await capturing_mcp.tools["deriva_ml_list_executions"](
        hostname="h", catalog_id="1", status="Running"
    )
    # The tool is responsible for converting str -> ExecutionStatus.
    args, kwargs = mock_ml.find_executions.call_args
    assert kwargs["status"] == ExecutionStatus.Running


async def test_list_executions_workflow_type_filter_forwarded(
    execution_ctx, capturing_mcp, mock_ml
):
    """workflow_type forwards verbatim to find_executions, including in preflight.

    Pins the cross-workflow type-filter path enabled in deriva-ml's
    find_executions(workflow_type=...) signature. Without the
    parameter being threaded through, "show me every Training
    execution" forces the LLM to enumerate workflows first.
    """
    mock_ml.find_executions.return_value = []
    # Non-preflight: workflow_type travels through _list_executions_impl.
    await capturing_mcp.tools["deriva_ml_list_executions"](
        hostname="h", catalog_id="1", workflow_type="Model_Training"
    )
    _args, kwargs = mock_ml.find_executions.call_args
    assert kwargs["workflow_type"] == "Model_Training"
    # Preflight: workflow_type also travels through the count closure.
    mock_ml.find_executions.reset_mock()
    await capturing_mcp.tools["deriva_ml_list_executions"](
        hostname="h",
        catalog_id="1",
        workflow_type="Inference",
        preflight_count=True,
    )
    _args, kwargs = mock_ml.find_executions.call_args
    assert kwargs["workflow_type"] == "Inference"


# ---------------------------------------------------------------------------
# get_execution
# ---------------------------------------------------------------------------


async def test_get_execution_success(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_execution.return_value = _make_execution_record_mock(
        rid="1-EXEC",
        workflow_rid="1-WF",
        status=ExecutionStatus.Stopped,
        description="some run",
    )
    result = await capturing_mcp.tools["deriva_ml_get_execution"](
        hostname="h", catalog_id="1", execution_rid="1-EXEC"
    )
    payload = json.loads(result)
    assert payload["rid"] == "1-EXEC"
    assert payload["workflow_rid"] == "1-WF"
    assert payload["status"] == "Stopped"
    assert payload["description"] == "some run"
    mock_ml.lookup_execution.assert_called_once_with("1-EXEC")


async def test_get_execution_error_path(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_execution.side_effect = RuntimeError("not found")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_get_execution"](
            hostname="h", catalog_id="1", execution_rid="missing"
        )
    payload = json.loads(result)
    assert "error" in payload
    assert "not found" in payload["error"]
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# find_workflow_executions
# ---------------------------------------------------------------------------


async def test_find_workflow_executions_success(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.find_executions.return_value = [
        _make_execution_record_mock(rid="1-AAA", workflow_rid="1-WF"),
        _make_execution_record_mock(rid="1-BBB", workflow_rid="1-WF"),
    ]
    result = await capturing_mcp.tools["deriva_ml_find_workflow_executions"](
        hostname="h", catalog_id="1", workflow_rid="1-WF"
    )
    payload = json.loads(result)
    assert payload["count"] == 2
    assert all(e["workflow_rid"] == "1-WF" for e in payload["executions"])
    args, kwargs = mock_ml.find_executions.call_args
    assert kwargs["workflow"] == "1-WF"


async def test_find_workflow_executions_error_path(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.find_executions.side_effect = RuntimeError("kaboom")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_find_workflow_executions"](
            hostname="h", catalog_id="1", workflow_rid="1-WF"
        )
    payload = json.loads(result)
    assert "error" in payload
    assert mock_audit.call_count == 0


async def test_list_executions_sort_forwards_to_deriva_ml(execution_ctx, capturing_mcp, mock_ml):
    """sort=True forwards sort=True to deriva_ml.find_executions and skips post-fetch RID sort."""
    # Mock find_executions to return records in RCT-desc-ish order
    # (the mock simulates what deriva-ml returns when sort=True).
    record_a = _make_execution_record_mock(rid="1-EXEC-NEWEST")
    record_b = _make_execution_record_mock(rid="1-EXEC-OLDER")
    mock_ml.find_executions.return_value = [record_a, record_b]

    result = await capturing_mcp.tools["deriva_ml_list_executions"](
        hostname="h", catalog_id="1", sort=True
    )
    payload = json.loads(result)

    # Confirm the deriva-ml call received sort=True
    mock_ml.find_executions.assert_called_once()
    assert mock_ml.find_executions.call_args.kwargs.get("sort") is True

    # Confirm the wire response preserved the RCT-desc order (NOT
    # re-sorted by RID).
    rids = [e["rid"] for e in payload["executions"]]
    assert rids == ["1-EXEC-NEWEST", "1-EXEC-OLDER"]


async def test_list_executions_sort_default_preserves_rid_sort(
    execution_ctx, capturing_mcp, mock_ml
):
    """sort=False (default) calls find_executions with sort=None and re-sorts by RID asc."""
    # Records intentionally NOT in RID order from the mock; the
    # wrapper should re-sort them ascending.
    record_z = _make_execution_record_mock(rid="1-Z")
    record_a = _make_execution_record_mock(rid="1-A")
    mock_ml.find_executions.return_value = [record_z, record_a]

    result = await capturing_mcp.tools["deriva_ml_list_executions"](hostname="h", catalog_id="1")
    payload = json.loads(result)

    # find_executions called with sort=None (the False -> None mapping)
    assert mock_ml.find_executions.call_args.kwargs.get("sort") is None
    # Wire response is RID-ascending
    rids = [e["rid"] for e in payload["executions"]]
    assert rids == ["1-A", "1-Z"]


# ---------------------------------------------------------------------------
# list_execution_children
# ---------------------------------------------------------------------------


async def test_list_execution_children_success(execution_ctx, capturing_mcp, mock_ml):
    parent = _make_execution_record_mock(rid="1-PARENT")
    parent.list_execution_children.return_value = [
        _make_execution_record_mock(rid="1-CHILD-A"),
        _make_execution_record_mock(rid="1-CHILD-B"),
    ]
    mock_ml.lookup_execution.return_value = parent

    result = await capturing_mcp.tools["deriva_ml_list_execution_children"](
        hostname="h", catalog_id="1", execution_rid="1-PARENT"
    )
    payload = json.loads(result)
    assert payload["parent_rid"] == "1-PARENT"
    assert payload["count"] == 2
    assert payload["recurse"] is False
    assert [c["rid"] for c in payload["children"]] == ["1-CHILD-A", "1-CHILD-B"]
    parent.list_execution_children.assert_called_once_with(recurse=False)


async def test_list_execution_children_recurse_true(execution_ctx, capturing_mcp, mock_ml):
    parent = _make_execution_record_mock(rid="1-PARENT")
    parent.list_execution_children.return_value = []
    mock_ml.lookup_execution.return_value = parent
    result = await capturing_mcp.tools["deriva_ml_list_execution_children"](
        hostname="h", catalog_id="1", execution_rid="1-PARENT", recurse=True
    )
    payload = json.loads(result)
    assert payload["recurse"] is True
    parent.list_execution_children.assert_called_once_with(recurse=True)


async def test_list_execution_children_error_path(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_execution.side_effect = RuntimeError("missing")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_list_execution_children"](
            hostname="h", catalog_id="1", execution_rid="1-PARENT"
        )
    payload = json.loads(result)
    assert "error" in payload
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# list_execution_parents
# ---------------------------------------------------------------------------


async def test_list_execution_parents_success(execution_ctx, capturing_mcp, mock_ml):
    child = _make_execution_record_mock(rid="1-CHILD")
    child.list_execution_parents.return_value = [
        _make_execution_record_mock(rid="1-PARENT-A"),
    ]
    mock_ml.lookup_execution.return_value = child

    result = await capturing_mcp.tools["deriva_ml_list_execution_parents"](
        hostname="h", catalog_id="1", execution_rid="1-CHILD"
    )
    payload = json.loads(result)
    assert payload["child_rid"] == "1-CHILD"
    assert payload["count"] == 1
    assert payload["parents"][0]["rid"] == "1-PARENT-A"
    child.list_execution_parents.assert_called_once_with(recurse=False)


async def test_list_execution_parents_error_path(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_execution.side_effect = RuntimeError("missing")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_list_execution_parents"](
            hostname="h", catalog_id="1", execution_rid="1-CHILD"
        )
    payload = json.loads(result)
    assert "error" in payload
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# get_lineage (data-flow provenance traversal)
# ---------------------------------------------------------------------------


async def test_get_lineage_success(execution_ctx, capturing_mcp, mock_ml):
    """The tool returns the LineageResult Pydantic model serialized to
    JSON. Underlying call is ``ml.lookup_lineage(rid, depth=...,
    max_executions=...)``; tool surfaces those parameters explicitly."""
    from deriva_ml.execution.lineage import (
        ExecutionSummary,
        LineageNode,
        LineageResult,
        RootDescriptor,
    )

    payload = LineageResult(
        root=RootDescriptor(
            rid="2-PRED",
            type="Asset",
            description="predictions.csv",
            producing_execution=ExecutionSummary(
                rid="1-EXEC",
                description="Train ResNet-50",
                workflow=None,
                status="Completed",
            ),
        ),
        lineage=LineageNode(
            execution=ExecutionSummary(
                rid="1-EXEC",
                description="Train ResNet-50",
                workflow=None,
                status="Completed",
            ),
            consumed_datasets=[],
            consumed_assets=[],
            parents=[],
        ),
        executions_visited=1,
        walked_complete=True,
        cycle_detected=False,
        depth_capped=False,
    )
    mock_ml.lookup_lineage.return_value = payload

    result = await capturing_mcp.tools["deriva_ml_get_lineage"](
        hostname="h", catalog_id="1", rid="2-PRED", depth=2, max_executions=100
    )
    out = json.loads(result)
    assert out["root"]["rid"] == "2-PRED"
    assert out["walked_complete"] is True
    # Tool surfaces depth + max_executions to underlying call.
    mock_ml.lookup_lineage.assert_called_once_with(
        "2-PRED", depth=2, max_executions=100
    )


async def test_get_lineage_defaults_unbounded(execution_ctx, capturing_mcp, mock_ml):
    """When depth and max_executions are omitted by the caller, the
    tool's signature defaults (depth=None, max_executions=500) reach
    the underlying method unchanged."""
    from deriva_ml.execution.lineage import (
        ExecutionSummary,
        LineageNode,
        LineageResult,
        RootDescriptor,
    )

    mock_ml.lookup_lineage.return_value = LineageResult(
        root=RootDescriptor(
            rid="1-EXEC", type="Execution",
            producing_execution=ExecutionSummary(
                rid="1-EXEC", description="root execution", workflow=None,
                status="Completed",
            ),
        ),
        lineage=LineageNode(
            execution=ExecutionSummary(
                rid="1-EXEC", description="root execution", workflow=None,
                status="Completed",
            ),
            consumed_datasets=[], consumed_assets=[], parents=[],
        ),
        executions_visited=1, walked_complete=True, cycle_detected=False,
        depth_capped=False,
    )

    await capturing_mcp.tools["deriva_ml_get_lineage"](
        hostname="h", catalog_id="1", rid="1-EXEC"
    )
    mock_ml.lookup_lineage.assert_called_once_with(
        "1-EXEC", depth=None, max_executions=500
    )


async def test_get_lineage_error_path_workflow_rid(execution_ctx, capturing_mcp, mock_ml):
    """A Workflow RID is not lineage-shaped; ``lookup_lineage`` raises
    and the tool wraps as ``{"error": ...}`` without emitting an audit
    row (read tool, silent on failure)."""
    mock_ml.lookup_lineage.side_effect = RuntimeError(
        "Workflow RIDs are not lineage-shaped"
    )
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_get_lineage"](
            hostname="h", catalog_id="1", rid="1-WF"
        )
    payload = json.loads(result)
    assert payload == {"error": "Workflow RIDs are not lineage-shaped"}
    assert mock_audit.call_count == 0


