"""Unit tests for execution domain tools."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from deriva_ml.execution.execution import ExecutionStatus

from tests._helpers import _success_calls, make_patch_audit

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
# create_execution
# ---------------------------------------------------------------------------


async def test_create_execution_success_emits_audit(execution_ctx, capturing_mcp, mock_ml):
    new_exec = _make_execution_mock(execution_rid="1-NEW")
    mock_ml.create_execution.return_value = new_exec
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_create_execution"](
            hostname="h",
            catalog_id="1",
            workflow_rid="1-WF",
            description="run",
        )
    payload = json.loads(result)
    assert payload["status"] == "created"
    assert payload["execution_rid"] == "1-NEW"
    assert payload["workflow_rid"] == "1-WF"
    assert payload["dataset_count"] == 0
    assert payload["asset_count"] == 0
    assert payload["dry_run"] is False

    success = _success_calls(mock_audit, "deriva_ml_create_execution")
    assert success
    assert success[0].kwargs["workflow_rid"] == "1-WF"
    assert success[0].kwargs["execution_rid"] == "1-NEW"
    assert success[0].kwargs["dataset_count"] == 0
    assert success[0].kwargs["asset_count"] == 0


async def test_create_execution_dry_run_skips_audit(execution_ctx, capturing_mcp, mock_ml):
    new_exec = _make_execution_mock(execution_rid="DRY-RUN-RID")
    mock_ml.create_execution.return_value = new_exec
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_create_execution"](
            hostname="h",
            catalog_id="1",
            workflow_rid="1-WF",
            description="run",
            dry_run=True,
        )
    payload = json.loads(result)
    assert payload["status"] == "created"
    assert payload["dry_run"] is True
    # Dry-run skips audit per Q3 / Phase 3 convention.
    assert mock_audit.call_count == 0
    args, kwargs = mock_ml.create_execution.call_args
    assert kwargs["dry_run"] is True


async def test_create_execution_with_datasets_and_assets(execution_ctx, capturing_mcp, mock_ml):
    new_exec = _make_execution_mock(execution_rid="1-NEW")
    mock_ml.create_execution.return_value = new_exec
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_create_execution"](
            hostname="h",
            catalog_id="1",
            workflow_rid="1-WF",
            dataset_rids=["4HM@1.0.0", "5XY@2.0.0"],
            asset_rids=["WGTS"],
        )
    payload = json.loads(result)
    assert payload["dataset_count"] == 2
    assert payload["asset_count"] == 1

    args, kwargs = mock_ml.create_execution.call_args
    # Strings pass through; upstream coerces them.
    assert kwargs["datasets"] == ["4HM@1.0.0", "5XY@2.0.0"]
    assert kwargs["assets"] == ["WGTS"]

    success = _success_calls(mock_audit, "deriva_ml_create_execution")
    assert success[0].kwargs["dataset_count"] == 2
    assert success[0].kwargs["asset_count"] == 1


async def test_create_execution_failure_emits_failed_audit(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.create_execution.side_effect = RuntimeError("workflow missing")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_create_execution"](
            hostname="h", catalog_id="1", workflow_rid="1-WF"
        )
    payload = json.loads(result)
    assert "error" in payload
    failed = _success_calls(mock_audit, "deriva_ml_create_execution_failed")
    assert failed
    assert failed[0].kwargs["workflow_rid"] == "1-WF"


# ---------------------------------------------------------------------------
# start_execution
# ---------------------------------------------------------------------------


async def test_start_execution_success_emits_audit(execution_ctx, capturing_mcp, mock_ml):
    execution = _make_execution_mock(execution_rid="1-EXEC", status=ExecutionStatus.Created)
    mock_ml.resume_execution.return_value = execution
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_start_execution"](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    payload = json.loads(result)
    assert payload["status"] == "running"
    assert payload["execution_rid"] == "1-EXEC"
    execution.execution_start.assert_called_once()
    success = _success_calls(mock_audit, "deriva_ml_start_execution")
    assert success
    assert success[0].kwargs["execution_rid"] == "1-EXEC"


async def test_start_execution_idempotent_when_already_running(
    execution_ctx, capturing_mcp, mock_ml
):
    execution = _make_execution_mock(execution_rid="1-EXEC", status=ExecutionStatus.Running)
    mock_ml.resume_execution.return_value = execution
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_start_execution"](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    payload = json.loads(result)
    assert payload["status"] == "running"
    assert payload["note"] == "already running"
    execution.execution_start.assert_not_called()
    # Idempotent no-op: no audit row.
    assert mock_audit.call_count == 0


async def test_start_execution_rejects_terminal_state(execution_ctx, capturing_mcp, mock_ml):
    execution = _make_execution_mock(execution_rid="1-EXEC", status=ExecutionStatus.Uploaded)
    mock_ml.resume_execution.return_value = execution
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_start_execution"](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    payload = json.loads(result)
    assert "error" in payload
    assert "Uploaded" in payload["error"]
    execution.execution_start.assert_not_called()
    # Arg-validation-style error: no audit row.
    assert mock_audit.call_count == 0


async def test_start_execution_failure_emits_failed_audit(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.resume_execution.side_effect = RuntimeError("missing")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_start_execution"](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    payload = json.loads(result)
    assert "error" in payload
    failed = _success_calls(mock_audit, "deriva_ml_start_execution_failed")
    assert failed


# ---------------------------------------------------------------------------
# commit_execution
# ---------------------------------------------------------------------------


def _make_uploaded_dict(per_table: dict[str, int] | None = None):
    """Build the dict shape returned by ``Execution.upload_execution_outputs``.

    Real return type is ``dict[str, list[AssetFilePath]]`` mapping
    ``"{schema}/{table}"`` to uploaded asset paths. For test purposes
    we substitute opaque integer-count tuples — the tool only reads
    ``len(value)``.
    """
    return {table: [object()] * count for table, count in (per_table or {}).items()}


def _attach_pending_features(execution: MagicMock, count: int) -> None:
    """Wire ``execution._manifest_store.list_pending_feature_records`` to return ``count`` rows.

    ``commit_execution`` snapshots the pending feature-record count
    before calling ``upload_execution_outputs`` so its synthetic
    ``total_uploaded`` includes feature inserts (which the asset-only
    return value of ``upload_execution_outputs`` doesn't expose).
    """
    execution._manifest_store.list_pending_feature_records.return_value = [object()] * count


async def test_commit_execution_success_emits_audit(execution_ctx, capturing_mcp, mock_ml):
    execution = _make_execution_mock(execution_rid="1-EXEC", status=ExecutionStatus.Running)
    _attach_pending_features(execution, count=3)
    execution.upload_execution_outputs.return_value = _make_uploaded_dict(
        {"deriva-ml/Execution_Metadata": 2}
    )
    mock_ml.resume_execution.return_value = execution
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_commit_execution"](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    payload = json.loads(result)
    assert payload["status"] == "uploaded"
    assert payload["execution_rid"] == "1-EXEC"
    # 2 asset uploads + 3 feature flushes.
    assert payload["report"]["total_uploaded"] == 5
    assert payload["report"]["total_failed"] == 0
    assert payload["report"]["per_table"] == {"deriva-ml/Execution_Metadata": 2}
    execution.execution_stop.assert_called_once()
    execution.upload_execution_outputs.assert_called_once_with()
    success = _success_calls(mock_audit, "deriva_ml_commit_execution")
    assert success
    assert success[0].kwargs["total_uploaded"] == 5
    assert success[0].kwargs["total_failed"] == 0
    assert success[0].kwargs["retry_failed"] is False


async def test_commit_execution_retry_failed_true(execution_ctx, capturing_mcp, mock_ml):
    # retry_failed is propagated to the audit row but does not (yet)
    # alter the upload_execution_outputs call shape — that surface
    # has its own per-asset retry policy via max_retries / retry_delay,
    # not a retry_failed switch. We keep the public arg for forward
    # compatibility with the lease engine path.
    execution = _make_execution_mock(execution_rid="1-EXEC", status=ExecutionStatus.Pending_Upload)
    _attach_pending_features(execution, count=0)
    execution.upload_execution_outputs.return_value = _make_uploaded_dict({})
    mock_ml.resume_execution.return_value = execution
    with _patch_execution_audit() as mock_audit:
        await capturing_mcp.tools["deriva_ml_commit_execution"](
            hostname="h", catalog_id="1", execution_rid="1-EXEC", retry_failed=True
        )
    # Already past Running, no execution_stop call.
    execution.execution_stop.assert_not_called()
    execution.upload_execution_outputs.assert_called_once_with()
    success = _success_calls(mock_audit, "deriva_ml_commit_execution")
    assert success[0].kwargs["retry_failed"] is True


async def test_commit_execution_additive_upload_from_uploaded(
    execution_ctx, capturing_mcp, mock_ml
):
    # Per deriva-ml 3d21f55, an Uploaded execution can have additional
    # outputs registered after its initial commit. commit_execution must
    # accept Uploaded as a valid prior state -- upload_execution_outputs
    # cycles Uploaded -> Pending_Upload -> Uploaded internally when there
    # are new pending entries, or no-ops when there aren't. The MCP tool
    # must not reject this path with a state-machine error.
    execution = _make_execution_mock(execution_rid="1-EXEC", status=ExecutionStatus.Uploaded)
    _attach_pending_features(execution, count=0)
    execution.upload_execution_outputs.return_value = _make_uploaded_dict({})
    mock_ml.resume_execution.return_value = execution
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_commit_execution"](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    payload = json.loads(result)
    assert "error" not in payload
    # Already past Running, no execution_stop call (mirrors Pending_Upload path).
    execution.execution_stop.assert_not_called()
    execution.upload_execution_outputs.assert_called_once_with()
    success = _success_calls(mock_audit, "deriva_ml_commit_execution")
    assert success, "additive-upload path must still emit success audit"


async def test_commit_execution_failure_emits_failed_audit(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.resume_execution.side_effect = RuntimeError("offline")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_commit_execution"](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    payload = json.loads(result)
    assert "error" in payload
    failed = _success_calls(mock_audit, "deriva_ml_commit_execution_failed")
    assert failed


# ---------------------------------------------------------------------------
# abort_execution
# ---------------------------------------------------------------------------


async def test_abort_execution_success_emits_audit(execution_ctx, capturing_mcp, mock_ml):
    execution = _make_execution_mock(execution_rid="1-EXEC", status=ExecutionStatus.Running)
    mock_ml.resume_execution.return_value = execution
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_abort_execution"](
            hostname="h", catalog_id="1", execution_rid="1-EXEC", reason="cancel by user"
        )
    payload = json.loads(result)
    assert payload["status"] == "aborted"
    assert payload["execution_rid"] == "1-EXEC"
    execution.abort.assert_called_once()
    success = _success_calls(mock_audit, "deriva_ml_abort_execution")
    assert success
    assert success[0].kwargs["execution_rid"] == "1-EXEC"
    assert success[0].kwargs["reason"] == "cancel by user"


async def test_abort_execution_idempotent_when_already_aborted(
    execution_ctx, capturing_mcp, mock_ml
):
    execution = _make_execution_mock(execution_rid="1-EXEC", status=ExecutionStatus.Aborted)
    mock_ml.resume_execution.return_value = execution
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_abort_execution"](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    payload = json.loads(result)
    assert payload["status"] == "aborted"
    assert payload["note"] == "already aborted"
    execution.abort.assert_not_called()
    assert mock_audit.call_count == 0


async def test_abort_execution_failure_emits_failed_audit(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.resume_execution.side_effect = RuntimeError("missing")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_abort_execution"](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    payload = json.loads(result)
    assert "error" in payload
    failed = _success_calls(mock_audit, "deriva_ml_abort_execution_failed")
    assert failed


# ---------------------------------------------------------------------------
# create_execution_dataset
# ---------------------------------------------------------------------------


async def test_create_execution_dataset_success_emits_audit(execution_ctx, capturing_mcp, mock_ml):
    execution = _make_execution_mock(execution_rid="1-EXEC", status=ExecutionStatus.Stopped)
    new_dataset = MagicMock()
    new_dataset.rid = "1-NEW-DS"
    execution.create_dataset.return_value = new_dataset
    mock_ml.resume_execution.return_value = execution

    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_create_execution_dataset"](
            hostname="h",
            catalog_id="1",
            execution_rid="1-EXEC",
            description="output bag",
            dataset_types=["Training"],
        )
    payload = json.loads(result)
    assert payload["status"] == "created"
    assert payload["dataset_rid"] == "1-NEW-DS"
    assert payload["execution_rid"] == "1-EXEC"
    assert payload["dataset_types"] == ["Training"]
    success = _success_calls(mock_audit, "deriva_ml_create_execution_dataset")
    assert success
    assert success[0].kwargs["dataset_rid"] == "1-NEW-DS"
    assert success[0].kwargs["execution_rid"] == "1-EXEC"


async def test_create_execution_dataset_failure_emits_failed_audit(
    execution_ctx, capturing_mcp, mock_ml
):
    mock_ml.resume_execution.side_effect = RuntimeError("missing")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_create_execution_dataset"](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    payload = json.loads(result)
    assert "error" in payload
    failed = _success_calls(mock_audit, "deriva_ml_create_execution_dataset_failed")
    assert failed


# ---------------------------------------------------------------------------
# add_nested_execution
# ---------------------------------------------------------------------------


async def test_add_nested_execution_success_emits_audit(execution_ctx, capturing_mcp, mock_ml):
    parent = _make_execution_mock(execution_rid="1-PARENT", status=ExecutionStatus.Running)
    mock_ml.resume_execution.return_value = parent
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_add_nested_execution"](
            hostname="h",
            catalog_id="1",
            parent_execution_rid="1-PARENT",
            child_execution_rid="1-CHILD",
            sequence=2,
        )
    payload = json.loads(result)
    assert payload["status"] == "added"
    assert payload["parent_rid"] == "1-PARENT"
    assert payload["child_rid"] == "1-CHILD"
    assert payload["sequence"] == 2
    parent.add_nested_execution.assert_called_once_with("1-CHILD", sequence=2)
    success = _success_calls(mock_audit, "deriva_ml_add_nested_execution")
    assert success
    assert success[0].kwargs["parent_rid"] == "1-PARENT"
    assert success[0].kwargs["child_rid"] == "1-CHILD"
    assert success[0].kwargs["sequence"] == 2


async def test_add_nested_execution_failure_emits_failed_audit(
    execution_ctx, capturing_mcp, mock_ml
):
    mock_ml.resume_execution.side_effect = RuntimeError("missing")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_add_nested_execution"](
            hostname="h",
            catalog_id="1",
            parent_execution_rid="1-PARENT",
            child_execution_rid="1-CHILD",
        )
    payload = json.loads(result)
    assert "error" in payload
    failed = _success_calls(mock_audit, "deriva_ml_add_nested_execution_failed")
    assert failed
