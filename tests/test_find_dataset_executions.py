"""Unit tests for the dataset-scoped execution query surface.

Covers the two tools that expose the library's
``ml.find_executions(dataset=..., dataset_role=...)``:

- ``deriva_ml_find_dataset_executions`` -- the dataset-centric intent
  tool (mirrors ``deriva_ml_find_workflow_executions``).
- ``deriva_ml_list_executions(dataset=...)`` -- the same query reachable
  from the general browse tool.

All tests drive the ``mock_ml`` MagicMock; none touch a live catalog.
The library-side ``find_executions(dataset=, dataset_role=)`` is NOT in
the installed deriva-ml package, but a MagicMock accepts the kwargs and
returns whatever the test configures, so the impl can be exercised
end-to-end against the mock.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from deriva_ml.execution.execution import ExecutionStatus

from tests._helpers import make_patch_audit

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
    """Build an ``ExecutionRecord``-shaped MagicMock (same shape as test_execution.py)."""
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


@pytest.fixture()
def execution_ctx(ctx, mock_ml):
    """Register execution tools with mock_ml as the DerivaML stand-in."""
    with patch("deriva_ml_mcp_plugin.tools.execution.get_ml", return_value=mock_ml):
        from deriva_ml_mcp_plugin.tools import execution as execution_module

        execution_module.register(ctx)
        yield ctx


def _role_partitioned_side_effect(*, input_rids: list[str], output_rids: list[str]):
    """Return a side_effect for ``find_executions`` that partitions by ``dataset_role``.

    The impl is expected to query input and output edges separately
    (one ``find_executions`` call per role) when it needs per-row role
    tagging. This helper returns the per-role record lists so a test
    can assert the resulting role tags.
    """

    def _side_effect(*args, **kwargs):
        role = kwargs.get("dataset_role")
        if role == "input":
            return [_make_execution_record_mock(rid=r) for r in input_rids]
        if role == "output":
            return [_make_execution_record_mock(rid=r) for r in output_rids]
        return []

    return _side_effect


# ---------------------------------------------------------------------------
# deriva_ml_find_dataset_executions
# ---------------------------------------------------------------------------


async def test_find_dataset_executions_any_role_tags_rows(execution_ctx, capturing_mcp, mock_ml):
    """dataset_role="any" returns rows tagged with their input/output role."""
    mock_ml.find_executions.side_effect = _role_partitioned_side_effect(
        input_rids=["1-IN"], output_rids=["1-OUT"]
    )
    result = await capturing_mcp.tools["deriva_ml_find_dataset_executions"](
        hostname="h", catalog_id="1", dataset_rid="1-DSET"
    )
    payload = json.loads(result)
    assert payload["count"] == 2
    rows = {e["rid"]: e for e in payload["executions"]}
    assert rows["1-IN"]["dataset_role"] == "input"
    assert rows["1-OUT"]["dataset_role"] == "output"
    # Every row carries the dataset_role + dataset_version keys.
    for e in payload["executions"]:
        assert "dataset_role" in e
        assert "dataset_version" in e


async def test_find_dataset_executions_input_role_only(execution_ctx, capturing_mcp, mock_ml):
    """dataset_role="input" queries the input edge and tags rows 'input'."""
    mock_ml.find_executions.return_value = [_make_execution_record_mock(rid="1-IN")]
    result = await capturing_mcp.tools["deriva_ml_find_dataset_executions"](
        hostname="h", catalog_id="1", dataset_rid="1-DSET", dataset_role="input"
    )
    payload = json.loads(result)
    assert payload["count"] == 1
    assert payload["executions"][0]["dataset_role"] == "input"
    # The bare RID is forwarded as the dataset arg.
    _args, kwargs = mock_ml.find_executions.call_args
    assert kwargs["dataset"] == "1-DSET"
    assert kwargs["dataset_role"] == "input"


async def test_find_dataset_executions_version_builds_datasetspec(
    execution_ctx, capturing_mcp, mock_ml
):
    """A scalar dataset_version constructs a DatasetSpec(rid, version) for the impl."""
    from deriva_ml.dataset.aux_classes import DatasetSpec

    mock_ml.find_executions.return_value = [_make_execution_record_mock(rid="1-IN")]
    result = await capturing_mcp.tools["deriva_ml_find_dataset_executions"](
        hostname="h",
        catalog_id="1",
        dataset_rid="1-DSET",
        dataset_role="input",
        dataset_version="1.2.0",
    )
    payload = json.loads(result)
    # The dataset arg passed to the library is a DatasetSpec pinned to the version.
    _args, kwargs = mock_ml.find_executions.call_args
    spec = kwargs["dataset"]
    assert isinstance(spec, DatasetSpec)
    assert spec.rid == "1-DSET"
    assert str(spec.version) == "1.2.0"
    # The version surfaces on each row.
    assert payload["executions"][0]["dataset_version"] == "1.2.0"


async def test_find_dataset_executions_preflight(execution_ctx, capturing_mcp, mock_ml):
    """preflight_count=True honors the dataset filter and returns a total count."""
    mock_ml.find_executions.side_effect = _role_partitioned_side_effect(
        input_rids=["1-IN-A", "1-IN-B"], output_rids=["1-OUT"]
    )
    result = await capturing_mcp.tools["deriva_ml_find_dataset_executions"](
        hostname="h", catalog_id="1", dataset_rid="1-DSET", preflight_count=True
    )
    payload = json.loads(result)
    assert payload["total_count"] == 3
    assert payload["entities_fetched"] is False
    assert "action_required" in payload


async def test_find_dataset_executions_error_path(execution_ctx, capturing_mcp, mock_ml):
    """A failure wraps as {"error": ...} without an audit row (read-only tool)."""
    mock_ml.find_executions.side_effect = RuntimeError("kaboom")
    with _patch_execution_audit() as mock_audit:
        result = await capturing_mcp.tools["deriva_ml_find_dataset_executions"](
            hostname="h", catalog_id="1", dataset_rid="1-DSET"
        )
    payload = json.loads(result)
    assert "error" in payload
    assert "kaboom" in payload["error"]
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# deriva_ml_list_executions(dataset=...) -- shape parity + regression guard
# ---------------------------------------------------------------------------


async def test_list_executions_without_dataset_has_no_dataset_keys(
    execution_ctx, capturing_mcp, mock_ml
):
    """Regression guard: no dataset filter -> rows are plain ExecutionSummary.

    The existing wire shape must be byte-identical to before this
    feature: no ``dataset_role`` / ``dataset_version`` keys appear.
    """
    mock_ml.find_executions.return_value = [
        _make_execution_record_mock(rid="1-AAA"),
        _make_execution_record_mock(rid="1-BBB"),
    ]
    result = await capturing_mcp.tools["deriva_ml_list_executions"](hostname="h", catalog_id="1")
    payload = json.loads(result)
    assert payload["count"] == 2
    for e in payload["executions"]:
        assert "dataset_role" not in e
        assert "dataset_version" not in e


async def test_list_executions_with_dataset_returns_dataset_rows(
    execution_ctx, capturing_mcp, mock_ml
):
    """With a dataset filter, rows are DatasetExecutionSummary-shaped."""
    mock_ml.find_executions.side_effect = _role_partitioned_side_effect(
        input_rids=["1-IN"], output_rids=["1-OUT"]
    )
    result = await capturing_mcp.tools["deriva_ml_list_executions"](
        hostname="h", catalog_id="1", dataset="1-DSET"
    )
    payload = json.loads(result)
    assert payload["count"] == 2
    rows = {e["rid"]: e for e in payload["executions"]}
    assert rows["1-IN"]["dataset_role"] == "input"
    assert rows["1-OUT"]["dataset_role"] == "output"


async def test_list_executions_dataset_version_builds_datasetspec(
    execution_ctx, capturing_mcp, mock_ml
):
    """list_executions with dataset_version builds a DatasetSpec passed to the library."""
    from deriva_ml.dataset.aux_classes import DatasetSpec

    mock_ml.find_executions.return_value = [_make_execution_record_mock(rid="1-IN")]
    await capturing_mcp.tools["deriva_ml_list_executions"](
        hostname="h",
        catalog_id="1",
        dataset="1-DSET",
        dataset_role="input",
        dataset_version="2.0.0",
    )
    _args, kwargs = mock_ml.find_executions.call_args
    spec = kwargs["dataset"]
    assert isinstance(spec, DatasetSpec)
    assert spec.rid == "1-DSET"
    assert str(spec.version) == "2.0.0"


async def test_list_executions_dataset_preflight(execution_ctx, capturing_mcp, mock_ml):
    """list_executions preflight count honors the dataset filter."""
    mock_ml.find_executions.side_effect = _role_partitioned_side_effect(
        input_rids=["1-IN"], output_rids=["1-OUT-A", "1-OUT-B"]
    )
    result = await capturing_mcp.tools["deriva_ml_list_executions"](
        hostname="h", catalog_id="1", dataset="1-DSET", preflight_count=True
    )
    payload = json.loads(result)
    assert payload["total_count"] == 3
    assert payload["entities_fetched"] is False


# ---------------------------------------------------------------------------
# Dual-role pagination hazard: same execution RID on BOTH edges
# ---------------------------------------------------------------------------


async def test_find_dataset_executions_dual_role_survives_page_boundary(
    execution_ctx, capturing_mcp, mock_ml
):
    """A dual-role execution's two rows must both survive pagination.

    Scenario: execution ``1-DUAL`` both CONSUMED and PRODUCED the same
    dataset RID, so it appears once on the input edge and once on the
    output edge -- two rows sharing ``rid="1-DUAL"`` but differing in
    ``dataset_role``. With a composite cursor keyed on
    ``(rid, dataset_role)`` they occupy distinct cursor positions; a
    naive ``rid``-only cursor would skip the second same-rid row when
    the pair straddles a page boundary (silent row loss).

    Layout (composite-key ascending order, ``\\x00`` sorts first):
        1-AAAA / input
        1-DUAL / input
        1-DUAL / output
        1-ZZZZ / output
    With ``limit=2`` the dual-role pair straddles the page-1/page-2
    boundary, exercising exactly the hazard.
    """
    mock_ml.find_executions.side_effect = _role_partitioned_side_effect(
        input_rids=["1-AAAA", "1-DUAL"], output_rids=["1-DUAL", "1-ZZZZ"]
    )

    # Page 1.
    page1 = json.loads(
        await capturing_mcp.tools["deriva_ml_find_dataset_executions"](
            hostname="h", catalog_id="1", dataset_rid="1-DSET", limit=2
        )
    )
    assert page1["count"] == 2
    assert page1["truncated"] is True
    assert page1["next_after_rid"] is not None

    # Page 2 (follow the cursor exactly as a client would).
    page2 = json.loads(
        await capturing_mcp.tools["deriva_ml_find_dataset_executions"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-DSET",
            limit=2,
            after_rid=page1["next_after_rid"],
        )
    )

    # Collect (rid, role) pairs across BOTH pages.
    seen = [(e["rid"], e["dataset_role"]) for e in page1["executions"] + page2["executions"]]

    # No drop: both 1-DUAL rows present, one per role.
    assert ("1-DUAL", "input") in seen
    assert ("1-DUAL", "output") in seen
    # No duplicate: exactly four rows, all distinct (rid, role).
    assert len(seen) == 4
    assert len(set(seen)) == 4
    # The other two edge rows are present too.
    assert ("1-AAAA", "input") in seen
    assert ("1-ZZZZ", "output") in seen


# ---------------------------------------------------------------------------
# Serialization: DatasetExecutionSummary subclass fields survive the wire
# ---------------------------------------------------------------------------


def test_dataset_execution_summary_serializes_subclass_fields():
    """DatasetExecutionSummary keeps dataset_role/version on model_dump_json."""
    from deriva_ml_mcp_plugin._response_models import (
        DatasetExecutionSummary,
        ExecutionListResponse,
    )

    row = DatasetExecutionSummary(
        rid="1-EXEC",
        workflow_rid="1-WF",
        status="Completed",
        description="run",
        start_time=None,
        stop_time=None,
        duration=None,
        dataset_role="input",
        dataset_version="1.0.0",
    )
    resp = ExecutionListResponse(executions=[row], count=1, truncated=False, next_after_rid=None)
    payload = json.loads(resp.model_dump_json(by_alias=True))
    e = payload["executions"][0]
    assert e["dataset_role"] == "input"
    assert e["dataset_version"] == "1.0.0"
