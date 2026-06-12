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

    Patches at the use-site (``deriva_ml_mcp_plugin.tools.execution.get_ml``) and
    imports the tool module *inside* the patch block so registration sees
    the mock.
    """
    with patch("deriva_ml_mcp_plugin.tools.execution.get_ml", return_value=mock_ml):
        from deriva_ml_mcp_plugin.tools import execution as execution_module

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


async def test_list_executions_tool_schedules_index_on_find(execution_ctx, capturing_mcp, mock_ml):
    """``deriva_ml_list_executions`` warms the returned rows via _index_rows_on_find."""
    mock_ml.find_executions.return_value = [
        _make_execution_record_mock(rid="1-AAA", workflow_rid="1-WF"),
        _make_execution_record_mock(rid="1-BBB", workflow_rid="1-WF"),
    ]

    captured: dict = {}

    def fake_warm(hostname, catalog_id, token, rows, **kwargs):
        captured["token"] = token
        captured["rids"] = [r.get("rid") for r in rows]

    # The tool does a lazy ``from deriva_ml_mcp_plugin.resources.rag import
    # _index_rows_on_find`` at call time, so patch the name in its
    # defining module (the lazy import binds to ``rag._index_rows_on_find``).
    with patch(
        "deriva_ml_mcp_plugin.resources.rag._index_rows_on_find",
        side_effect=fake_warm,
    ):
        await capturing_mcp.tools["deriva_ml_list_executions"](hostname="h", catalog_id="1")

    from deriva_ml_mcp_plugin.resources.rag import _EXECUTION_TOKEN

    assert captured.get("token") == _EXECUTION_TOKEN
    assert set(captured.get("rids") or []) == {"1-AAA", "1-BBB"}


async def test_list_executions_preflight_does_not_index_on_find(
    execution_ctx, capturing_mcp, mock_ml
):
    """The preflight (count-only) path returns before building rows, so it must
    NOT schedule a read-through warm."""
    mock_ml.find_executions.return_value = [
        _make_execution_record_mock(rid=f"1-{i:03d}") for i in range(3)
    ]

    called = False

    def fake_warm(*args, **kwargs):
        nonlocal called
        called = True

    with patch(
        "deriva_ml_mcp_plugin.resources.rag._index_rows_on_find",
        side_effect=fake_warm,
    ):
        await capturing_mcp.tools["deriva_ml_list_executions"](
            hostname="h", catalog_id="1", preflight_count=True
        )

    assert called is False


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


async def test_get_execution_tool_schedules_index_on_find(execution_ctx, capturing_mcp, mock_ml):
    """``deriva_ml_get_execution`` warms the single returned row via _index_rows_on_find."""
    mock_ml.lookup_execution.return_value = _make_execution_record_mock(
        rid="1-EXEC", workflow_rid="1-WF"
    )

    captured: dict = {}

    def fake_warm(hostname, catalog_id, token, rows, **kwargs):
        captured["token"] = token
        captured["rids"] = [r.get("rid") for r in rows]

    # Lazy import in the tool binds to ``rag._index_rows_on_find`` at call
    # time, so patch the name in its defining module.
    with patch(
        "deriva_ml_mcp_plugin.resources.rag._index_rows_on_find",
        side_effect=fake_warm,
    ):
        await capturing_mcp.tools["deriva_ml_get_execution"](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )

    from deriva_ml_mcp_plugin.resources.rag import _EXECUTION_TOKEN

    assert captured.get("token") == _EXECUTION_TOKEN
    assert captured.get("rids") == ["1-EXEC"]


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
    mock_ml.lookup_lineage.assert_called_once_with("2-PRED", depth=2, max_executions=100)


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
            rid="1-EXEC",
            type="Execution",
            producing_execution=ExecutionSummary(
                rid="1-EXEC",
                description="root execution",
                workflow=None,
                status="Completed",
            ),
        ),
        lineage=LineageNode(
            execution=ExecutionSummary(
                rid="1-EXEC",
                description="root execution",
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

    await capturing_mcp.tools["deriva_ml_get_lineage"](hostname="h", catalog_id="1", rid="1-EXEC")
    mock_ml.lookup_lineage.assert_called_once_with("1-EXEC", depth=None, max_executions=500)


async def test_get_lineage_error_path_workflow_rid(execution_ctx, capturing_mcp, mock_ml):
    """A Workflow RID is not lineage-shaped; ``lookup_lineage`` raises
    and the tool wraps as ``{"error": ...}`` without emitting an audit
    row (read tool, silent on failure)."""
    mock_ml.lookup_lineage.side_effect = RuntimeError("Workflow RIDs are not lineage-shaped")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_get_lineage"](
            hostname="h", catalog_id="1", rid="1-WF"
        )
    payload = json.loads(result)
    assert payload == {
        "error": "Workflow RIDs are not lineage-shaped",
        "error_type": "RuntimeError",
    }
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# find_experiments (P2 hygiene sweep)
# ---------------------------------------------------------------------------


async def test_find_experiments_happy_path(execution_ctx, capturing_mcp, mock_ml):
    """``deriva_ml_find_experiments`` returns execution summaries for
    executions that have Hydra config assets.

    The tool calls ``ml.find_experiments()`` (catalog scan for
    ``*-config.yaml`` in Execution_Metadata), then resolves each
    ``Experiment.execution_rid`` via ``ml.lookup_execution`` to build
    the summary. The response shape is the same as
    ``deriva_ml_list_executions``.
    """
    exp_a = MagicMock()
    exp_a.execution_rid = "1-EXPA"
    exp_b = MagicMock()
    exp_b.execution_rid = "1-EXPB"

    mock_ml.find_experiments.return_value = [exp_a, exp_b]
    mock_ml.lookup_execution.side_effect = lambda rid: _make_execution_record_mock(
        rid=rid, status=ExecutionStatus.Uploaded
    )

    result = await capturing_mcp.tools["deriva_ml_find_experiments"](hostname="h", catalog_id="1")
    out = json.loads(result)

    assert out["count"] == 2
    assert out["truncated"] is False
    rids = [e["rid"] for e in out["executions"]]
    assert "1-EXPA" in rids
    assert "1-EXPB" in rids
    assert all(e["status"] == "Uploaded" for e in out["executions"])
    mock_ml.find_experiments.assert_called_once_with(workflow_rid=None, status=None)


async def test_find_experiments_preflight_count(execution_ctx, capturing_mcp, mock_ml):
    """``preflight_count=True`` returns ``{"total_count": N, ...}``."""
    exp = MagicMock()
    exp.execution_rid = "1-EXPA"
    mock_ml.find_experiments.return_value = [exp]

    result = await capturing_mcp.tools["deriva_ml_find_experiments"](
        hostname="h", catalog_id="1", preflight_count=True
    )
    out = json.loads(result)
    assert out["total_count"] == 1
    assert out["entities_fetched"] is False
    assert "action_required" in out


async def test_find_experiments_error_path(execution_ctx, capturing_mcp, mock_ml):
    """``ml.find_experiments`` raising wraps as ``{"error": ...}`` without
    an audit row (read-only tool, silent on failure)."""
    mock_ml.find_experiments.side_effect = RuntimeError("catalog unreachable")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_find_experiments"](
            hostname="h", catalog_id="1"
        )
    out = json.loads(result)
    assert "error" in out
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# 2026-05-26 projection audit: workflow_rid recovery (e2e finding developer/02)
# ---------------------------------------------------------------------------
#
# Background: deriva-ml #226 renamed the Workflow field ``rid`` ->
# ``workflow_rid`` but its construction sites in find_workflows /
# lookup_workflow still pass the OLD kwarg name. Because
# ``VALIDATION_CONFIG`` doesn't set extra="forbid", that value is
# silently dropped, so every Workflow returned by either API has
# ``workflow_rid=None``, and the ExecutionRecord's workflow_rid
# property (which reads through self._workflow.workflow_rid) also
# returns None on every execution. Surfaces as
# ``deriva_ml_get_execution -> {"workflow_rid": null}`` for every
# execution in the catalog (e2e finding developer/02, 2026-05-26).
#
# The MCP layer plugs this with a defensive recovery in
# ``_resolve_workflow_rid``: when the fast attribute read returns
# None but the record is bound to a live catalog, fetch the Workflow
# FK directly via ``ml._retrieve_rid(execution_rid)``. These tests
# exercise both the fast path (record exposes workflow_rid) and the
# recovery path (record's workflow_rid is None -> ermrest lookup).


def test_resolve_workflow_rid_fast_path_returns_attribute():
    """When ``record.workflow_rid`` is populated, return it without I/O."""
    from deriva_ml_mcp_plugin.tools.execution.read import _resolve_workflow_rid

    rec = MagicMock()
    rec.workflow_rid = "1-WF"
    # _ml_instance set but should NOT be consulted on the fast path.
    rec._ml_instance = MagicMock()
    rec._ml_instance._retrieve_rid.side_effect = AssertionError("recovery path called")
    assert _resolve_workflow_rid(rec) == "1-WF"
    rec._ml_instance._retrieve_rid.assert_not_called()


def test_resolve_workflow_rid_recovery_fetches_from_ermrest():
    """When ``record.workflow_rid`` is None, recover via _retrieve_rid."""
    from deriva_ml_mcp_plugin.tools.execution.read import _resolve_workflow_rid

    rec = MagicMock()
    rec.workflow_rid = None
    rec.execution_rid = "1-EXEC"
    rec._ml_instance = MagicMock()
    rec._ml_instance._retrieve_rid.return_value = {
        "RID": "1-EXEC",
        "Workflow": "1-WF-RECOVERED",
        "Status": "Uploaded",
    }
    assert _resolve_workflow_rid(rec) == "1-WF-RECOVERED"
    rec._ml_instance._retrieve_rid.assert_called_once_with("1-EXEC")


def test_resolve_workflow_rid_recovery_returns_none_without_ml_instance():
    """Mock-shaped records that don't expose ``_ml_instance`` return None safely."""
    from deriva_ml_mcp_plugin.tools.execution.read import _resolve_workflow_rid

    rec = MagicMock()
    rec.workflow_rid = None
    rec.execution_rid = "1-EXEC"
    # Spec restricts attribute access to a closed list -- no _ml_instance.
    del rec._ml_instance
    assert _resolve_workflow_rid(rec) is None


def test_resolve_workflow_rid_recovery_returns_none_on_fetch_failure():
    """If ``_retrieve_rid`` raises, recovery returns None (no propagation)."""
    from deriva_ml_mcp_plugin.tools.execution.read import _resolve_workflow_rid

    rec = MagicMock()
    rec.workflow_rid = None
    rec.execution_rid = "1-EXEC"
    rec._ml_instance = MagicMock()
    rec._ml_instance._retrieve_rid.side_effect = RuntimeError("ermrest unreachable")
    assert _resolve_workflow_rid(rec) is None


def test_resolve_workflow_rid_recovery_returns_none_when_workflow_fk_missing():
    """``Execution.Workflow == None`` (legitimate) returns None, not raises."""
    from deriva_ml_mcp_plugin.tools.execution.read import _resolve_workflow_rid

    rec = MagicMock()
    rec.workflow_rid = None
    rec.execution_rid = "1-EXEC"
    rec._ml_instance = MagicMock()
    rec._ml_instance._retrieve_rid.return_value = {"RID": "1-EXEC", "Workflow": None}
    assert _resolve_workflow_rid(rec) is None


async def test_get_execution_recovers_workflow_rid_via_ermrest(
    execution_ctx, capturing_mcp, mock_ml
):
    """End-to-end: a deriva-ml bug-shaped record still gets a real workflow_rid."""
    # Simulate the deriva-ml #226 regression: ExecutionRecord with
    # workflow_rid=None but _ml_instance present and the underlying
    # ermrest row carrying the real Workflow FK.
    record = _make_execution_record_mock(rid="1-EXEC", workflow_rid=None)
    record._ml_instance = mock_ml
    mock_ml._retrieve_rid.return_value = {
        "RID": "1-EXEC",
        "Workflow": "1-WF-FROM-ERMREST",
        "Status": "Uploaded",
    }
    mock_ml.lookup_execution.return_value = record
    result = await capturing_mcp.tools["deriva_ml_get_execution"](
        hostname="h", catalog_id="1", execution_rid="1-EXEC"
    )
    payload = json.loads(result)
    assert payload["rid"] == "1-EXEC"
    assert payload["workflow_rid"] == "1-WF-FROM-ERMREST"


async def test_list_execution_children_recovers_workflow_rid(execution_ctx, capturing_mcp, mock_ml):
    """Recovery applies to every child in list_execution_children."""
    parent = _make_execution_record_mock(rid="1-PARENT", workflow_rid="1-WF")
    parent._ml_instance = mock_ml
    child_a = _make_execution_record_mock(rid="1-CHILD-A", workflow_rid=None)
    child_a._ml_instance = mock_ml
    child_b = _make_execution_record_mock(rid="1-CHILD-B", workflow_rid=None)
    child_b._ml_instance = mock_ml
    parent.list_execution_children.return_value = [child_a, child_b]
    mock_ml.lookup_execution.return_value = parent
    mock_ml._retrieve_rid.side_effect = lambda rid: {
        "1-CHILD-A": {"RID": "1-CHILD-A", "Workflow": "1-WF-A"},
        "1-CHILD-B": {"RID": "1-CHILD-B", "Workflow": "1-WF-B"},
    }[rid]
    result = await capturing_mcp.tools["deriva_ml_list_execution_children"](
        hostname="h", catalog_id="1", execution_rid="1-PARENT"
    )
    payload = json.loads(result)
    assert payload["count"] == 2
    rids = {(c["rid"], c["workflow_rid"]) for c in payload["children"]}
    assert rids == {("1-CHILD-A", "1-WF-A"), ("1-CHILD-B", "1-WF-B")}


async def test_get_execution_preserves_workflow_rid_when_present(
    execution_ctx, capturing_mcp, mock_ml
):
    """Sanity check: when workflow_rid IS populated, recovery is not invoked."""
    record = _make_execution_record_mock(rid="1-EXEC", workflow_rid="1-WF-DIRECT")
    record._ml_instance = mock_ml
    mock_ml._retrieve_rid.side_effect = AssertionError(
        "recovery path called when fast path should have returned"
    )
    mock_ml.lookup_execution.return_value = record
    result = await capturing_mcp.tools["deriva_ml_get_execution"](
        hostname="h", catalog_id="1", execution_rid="1-EXEC"
    )
    payload = json.loads(result)
    assert payload["workflow_rid"] == "1-WF-DIRECT"
