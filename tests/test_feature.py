"""Unit tests for feature domain tools."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import _success_calls


@contextmanager
def _patch_feature_audit():
    """Dual-patch ``audit_event`` for both the success site and the helpers
    failure site so tests can capture either with a single mock.

    Why two patches: Python's ``from X import name`` binds ``name`` in the
    importing module's namespace at import time, so a single-patch facade
    isn't possible without changing every call site to use attribute
    lookup. See ``_patch_audit`` in tests/test_dataset.py for the
    canonical explanation.
    """
    with (
        patch("deriva_ml_mcp.tools.feature.audit_event") as mock_audit,
        patch("deriva_ml_mcp._helpers.audit_event", new=mock_audit),
    ):
        yield mock_audit


def _make_column_mock(name: str, type_name: str = "text", nullok: bool = True, default=None):
    """Build a Column-shaped MagicMock for term/asset/value columns."""
    col = MagicMock()
    col.name = name
    col.nullok = nullok
    col.default = default
    col.type = MagicMock()
    col.type.typename = type_name
    return col


def _make_feature_mock(
    feature_name: str = "Quality",
    target_table_name: str = "Image",
    feature_table_name: str = "Execution_Image_Quality",
    term_columns: list | None = None,
    asset_columns: list | None = None,
    value_columns: list | None = None,
    feature_table_comment: str = "",
) -> MagicMock:
    """Build a Feature-shaped MagicMock with the attributes the tools read.

    The real Feature class exposes ``feature_name``, ``target_table``,
    ``feature_table``, ``term_columns``, ``asset_columns``, and
    ``value_columns``. ``target_table`` and ``feature_table`` are Table
    objects with a ``.name`` attribute.
    """
    f = MagicMock()
    f.feature_name = feature_name
    f.target_table = MagicMock()
    f.target_table.name = target_table_name
    f.feature_table = MagicMock()
    f.feature_table.name = feature_table_name
    f.feature_table.comment = feature_table_comment
    f.term_columns = term_columns or []
    f.asset_columns = asset_columns or []
    f.value_columns = value_columns or []
    return f


@pytest.fixture()
def feature_ctx(ctx, mock_ml):
    """Register feature tools with mock_ml as the DerivaML stand-in.

    Patches at the use-site (`deriva_ml_mcp.tools.feature.get_ml`) and
    imports the tool module *inside* the patch block so registration sees
    the mock.
    """
    with patch("deriva_ml_mcp.tools.feature.get_ml", return_value=mock_ml):
        from deriva_ml_mcp.tools import feature as feature_module

        feature_module.register(ctx)
        yield ctx


# ---------------------------------------------------------------------------
# list_features
# ---------------------------------------------------------------------------


async def test_list_features_success(feature_ctx, capturing_mcp, mock_ml):
    """Returns one entry per Feature with column names extracted."""
    f1 = _make_feature_mock(
        feature_name="Quality",
        target_table_name="Image",
        feature_table_name="Execution_Image_Quality",
        term_columns=[_make_column_mock("Quality_Type")],
        value_columns=[_make_column_mock("score", type_name="float4")],
    )
    f2 = _make_feature_mock(
        feature_name="Box",
        target_table_name="Image",
        feature_table_name="Execution_Image_Box",
        asset_columns=[_make_column_mock("Box_Asset")],
    )
    mock_ml.find_features.return_value = [f1, f2]

    out = json.loads(await capturing_mcp.tools["list_features"](hostname="h", catalog_id="1"))

    assert out["count"] == 2
    assert out["truncated"] is False
    feature_names = {f["feature_name"] for f in out["features"]}
    assert feature_names == {"Quality", "Box"}
    quality = next(f for f in out["features"] if f["feature_name"] == "Quality")
    assert quality["target_table"] == "Image"
    assert quality["feature_table"] == "Execution_Image_Quality"
    assert quality["term_columns"] == ["Quality_Type"]
    assert quality["value_columns"] == ["score"]
    mock_ml.find_features.assert_called_once_with(table=None)


async def test_list_features_filters_by_table(feature_ctx, capturing_mcp, mock_ml):
    """``table=`` is forwarded to find_features."""
    mock_ml.find_features.return_value = []
    await capturing_mcp.tools["list_features"](hostname="h", catalog_id="1", table="Image")
    mock_ml.find_features.assert_called_once_with(table="Image")


async def test_list_features_preflight(feature_ctx, capturing_mcp, mock_ml):
    mock_ml.find_features.return_value = [
        _make_feature_mock(feature_name=f"F{i}", feature_table_name=f"FT{i:04d}") for i in range(7)
    ]
    out = json.loads(
        await capturing_mcp.tools["list_features"](
            hostname="h", catalog_id="1", preflight_count=True
        )
    )
    assert out["total_count"] == 7
    assert out["entities_fetched"] is False
    assert "action_required" in out


async def test_list_features_error_path(feature_ctx, capturing_mcp, mock_ml):
    mock_ml.find_features.side_effect = RuntimeError("schema down")
    with _patch_feature_audit() as mock_audit:
        out = json.loads(await capturing_mcp.tools["list_features"](hostname="h", catalog_id="1"))
    assert out == {"error": "schema down"}
    # Read tool: no audit on failure.
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# get_feature
# ---------------------------------------------------------------------------


async def test_get_feature_success(feature_ctx, capturing_mcp, mock_ml):
    """Returns full schema with term/asset/value column metadata."""
    term_col = _make_column_mock("Quality_Type", nullok=False)
    asset_col = _make_column_mock("Box_Asset", nullok=True)
    value_col = _make_column_mock("score", type_name="float4", nullok=True)
    f = _make_feature_mock(
        feature_name="Quality",
        target_table_name="Image",
        feature_table_name="Execution_Image_Quality",
        feature_table_comment="quality scores",
        term_columns=[term_col],
        asset_columns=[asset_col],
        value_columns=[value_col],
    )
    mock_ml.lookup_feature.return_value = f

    out = json.loads(
        await capturing_mcp.tools["get_feature"](
            hostname="h", catalog_id="1", table="Image", feature_name="Quality"
        )
    )
    assert out["feature_name"] == "Quality"
    assert out["target_table"] == "Image"
    assert out["feature_table"] == "Execution_Image_Quality"
    assert out["comment"] == "quality scores"
    assert out["term_columns"] == [{"name": "Quality_Type", "nullok": False}]
    assert out["asset_columns"] == [{"name": "Box_Asset", "nullok": True}]
    assert out["value_columns"] == [
        {"name": "score", "type": "float4", "nullok": True, "default": None}
    ]
    mock_ml.lookup_feature.assert_called_once_with("Image", "Quality")


async def test_get_feature_error_path(feature_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_feature.side_effect = RuntimeError("no such feature")
    with _patch_feature_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["get_feature"](
                hostname="h", catalog_id="1", table="Image", feature_name="Missing"
            )
        )
    assert out == {"error": "no such feature"}
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# list_feature_values
# ---------------------------------------------------------------------------


def _make_record_mock(target_rid: str, feature_name: str = "Quality", **fields) -> MagicMock:
    """Build a FeatureRecord-shaped MagicMock supporting ``.model_dump()``."""
    rec = MagicMock()
    rec.RID = target_rid
    payload = {"RID": target_rid, "Feature_Name": feature_name, **fields}
    rec.model_dump.return_value = payload
    return rec


async def test_list_feature_values_success_no_selector(feature_ctx, capturing_mcp, mock_ml):
    """Returns model_dump'd records, paged by RID."""
    recs = [_make_record_mock(f"1-{i:04d}") for i in range(3)]
    mock_ml.feature_values.return_value = iter(recs)

    out = json.loads(
        await capturing_mcp.tools["list_feature_values"](
            hostname="h", catalog_id="1", table="Image", feature_name="Quality"
        )
    )
    assert out["count"] == 3
    assert out["truncated"] is False
    assert [r["RID"] for r in out["records"]] == ["1-0000", "1-0001", "1-0002"]
    # selector=None passed through.
    mock_ml.feature_values.assert_called_once_with("Image", "Quality", selector=None)


async def test_list_feature_values_selector_newest(feature_ctx, capturing_mcp, mock_ml):
    """selector='newest' resolves to FeatureRecord.select_newest staticmethod."""
    mock_ml.feature_values.return_value = iter([_make_record_mock("1-AAAA")])
    await capturing_mcp.tools["list_feature_values"](
        hostname="h",
        catalog_id="1",
        table="Image",
        feature_name="Quality",
        selector="newest",
    )
    selector_arg = mock_ml.feature_values.call_args.kwargs["selector"]
    # select_newest is a staticmethod, so the selector_arg should be the
    # function itself (not None).
    from deriva_ml.feature import FeatureRecord

    assert selector_arg is FeatureRecord.select_newest


@pytest.mark.parametrize(
    "selector_name,attr_name",
    [
        ("first", "select_first"),
        ("latest", "select_latest"),
        ("majority_vote", "select_majority_vote"),
    ],
)
async def test_list_feature_values_other_simple_selectors(
    selector_name, attr_name, feature_ctx, capturing_mcp, mock_ml
):
    """Each simple selector resolves to the matching FeatureRecord factory.

    Exercises the dispatch branches that test_list_feature_values_selector_newest
    doesn't cover. ``first``/``latest`` are staticmethods (passed by reference);
    ``majority_vote`` is a classmethod factory (called eagerly to produce a
    callable). The tool's selector dispatch should handle both shapes.
    """
    mock_ml.feature_values.return_value = iter([_make_record_mock("1-AAAA")])
    await capturing_mcp.tools["list_feature_values"](
        hostname="h",
        catalog_id="1",
        table="Image",
        feature_name="Quality",
        selector=selector_name,
    )
    selector_arg = mock_ml.feature_values.call_args.kwargs["selector"]
    # The selector_arg should be a callable produced by the matching factory
    # (either the staticmethod itself or the result of calling the
    # classmethod factory). Either way, it must not be None.
    assert selector_arg is not None
    assert callable(selector_arg)


async def test_list_feature_values_selector_by_workflow_requires_arg(
    feature_ctx, capturing_mcp, mock_ml
):
    """Validation: selector='by_workflow' without selector_workflow returns error."""
    out = json.loads(
        await capturing_mcp.tools["list_feature_values"](
            hostname="h",
            catalog_id="1",
            table="Image",
            feature_name="Quality",
            selector="by_workflow",
        )
    )
    assert "error" in out
    assert "selector_workflow" in out["error"]
    # No upstream call was made.
    mock_ml.feature_values.assert_not_called()


async def test_list_feature_values_selector_by_execution(feature_ctx, capturing_mcp, mock_ml):
    """selector='by_execution' calls the FeatureRecord factory and forwards it."""
    mock_ml.feature_values.return_value = iter([])
    await capturing_mcp.tools["list_feature_values"](
        hostname="h",
        catalog_id="1",
        table="Image",
        feature_name="Quality",
        selector="by_execution",
        selector_execution_rid="EXEC-1",
    )
    selector_arg = mock_ml.feature_values.call_args.kwargs["selector"]
    assert callable(selector_arg)


async def test_list_feature_values_dataset_scope(feature_ctx, capturing_mcp, mock_ml):
    """When dataset_rid is set, look up the dataset and call its feature_values."""
    ds = MagicMock()
    ds.feature_values.return_value = iter([_make_record_mock("1-AAAA")])
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["list_feature_values"](
            hostname="h",
            catalog_id="1",
            table="Image",
            feature_name="Quality",
            dataset_rid="DS-1",
        )
    )
    assert out["count"] == 1
    mock_ml.lookup_dataset.assert_called_once_with("DS-1")
    ds.feature_values.assert_called_once_with("Image", "Quality", selector=None)
    # Top-level catalog feature_values not used in dataset mode.
    mock_ml.feature_values.assert_not_called()


async def test_list_feature_values_preflight(feature_ctx, capturing_mcp, mock_ml):
    mock_ml.feature_values.return_value = iter([_make_record_mock(f"1-{i:04d}") for i in range(4)])
    out = json.loads(
        await capturing_mcp.tools["list_feature_values"](
            hostname="h",
            catalog_id="1",
            table="Image",
            feature_name="Quality",
            preflight_count=True,
        )
    )
    assert out["total_count"] == 4
    assert out["entities_fetched"] is False
    assert "action_required" in out


async def test_list_feature_values_error_path(feature_ctx, capturing_mcp, mock_ml):
    mock_ml.feature_values.side_effect = RuntimeError("ermrest down")
    with _patch_feature_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["list_feature_values"](
                hostname="h", catalog_id="1", table="Image", feature_name="Quality"
            )
        )
    assert out == {"error": "ermrest down"}
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# create_feature
# ---------------------------------------------------------------------------


async def test_create_feature_success(feature_ctx, capturing_mcp, mock_ml):
    """Happy path: returns the new feature schema and emits success audit."""
    new_feature = _make_feature_mock(
        feature_name="Quality",
        target_table_name="Image",
        feature_table_name="Execution_Image_Quality",
        term_columns=[_make_column_mock("Quality_Type")],
        value_columns=[_make_column_mock("score", type_name="float4")],
    )
    mock_ml.lookup_feature.return_value = new_feature

    with _patch_feature_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["create_feature"](
                hostname="h",
                catalog_id="1",
                target_table="Image",
                feature_name="Quality",
                terms=["Quality_Type"],
                metadata=["score"],
                comment="image quality",
            )
        )

    assert out["status"] == "created"
    assert out["feature_name"] == "Quality"
    assert out["target_table"] == "Image"
    assert out["feature_table"] == "Execution_Image_Quality"
    assert out["term_columns"] == ["Quality_Type"]
    assert out["value_columns"] == ["score"]

    mock_ml.create_feature.assert_called_once()
    success = _success_calls(mock_audit, "deriva_ml_create_feature")
    assert success, "expected deriva_ml_create_feature audit event"
    assert success[0].kwargs["target_table"] == "Image"
    assert success[0].kwargs["feature_name"] == "Quality"
    assert success[0].kwargs["n_term_cols"] == 1
    assert success[0].kwargs["n_asset_cols"] == 0
    # No `comment` in audit fields (free text).
    assert "comment" not in success[0].kwargs


async def test_create_feature_failure_emits_failed_audit(feature_ctx, capturing_mcp, mock_ml):
    mock_ml.create_feature.side_effect = RuntimeError("invalid term table")
    with _patch_feature_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["create_feature"](
                hostname="h",
                catalog_id="1",
                target_table="Image",
                feature_name="Quality",
                terms=["BadTerm"],
            )
        )
    assert out == {"error": "invalid term table"}

    failed = _success_calls(mock_audit, "deriva_ml_create_feature_failed")
    assert failed, "expected deriva_ml_create_feature_failed audit event"
    assert failed[0].kwargs["error_type"] == "RuntimeError"
    assert failed[0].kwargs["target_table"] == "Image"
    assert failed[0].kwargs["feature_name"] == "Quality"
    assert not _success_calls(mock_audit, "deriva_ml_create_feature")


# ---------------------------------------------------------------------------
# delete_feature
# ---------------------------------------------------------------------------


async def test_delete_feature_success(feature_ctx, capturing_mcp, mock_ml):
    """Happy path: returns deleted status and emits success audit."""
    mock_ml.delete_feature.return_value = True

    with _patch_feature_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["delete_feature"](
                hostname="h", catalog_id="1", table="Image", feature_name="Quality"
            )
        )

    assert out == {"status": "deleted", "feature_name": "Quality", "table": "Image"}
    mock_ml.delete_feature.assert_called_once_with("Image", "Quality")
    success = _success_calls(mock_audit, "deriva_ml_delete_feature")
    assert success
    assert success[0].kwargs["target_table"] == "Image"
    assert success[0].kwargs["feature_name"] == "Quality"


async def test_delete_feature_not_found_no_audit(feature_ctx, capturing_mcp, mock_ml):
    """When delete_feature returns False, no audit event fires."""
    mock_ml.delete_feature.return_value = False

    with _patch_feature_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["delete_feature"](
                hostname="h", catalog_id="1", table="Image", feature_name="Missing"
            )
        )

    assert out == {"status": "not_found", "feature_name": "Missing", "table": "Image"}
    # No state changed -> no audit row.
    assert mock_audit.call_count == 0


async def test_delete_feature_failure_emits_failed_audit(feature_ctx, capturing_mcp, mock_ml):
    mock_ml.delete_feature.side_effect = RuntimeError("permission denied")
    with _patch_feature_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["delete_feature"](
                hostname="h", catalog_id="1", table="Image", feature_name="Quality"
            )
        )
    assert out == {"error": "permission denied"}
    failed = _success_calls(mock_audit, "deriva_ml_delete_feature_failed")
    assert failed
    assert failed[0].kwargs["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# add_feature_values
# ---------------------------------------------------------------------------


async def test_add_feature_values_success(feature_ctx, capturing_mcp, mock_ml):
    """Happy path: builds records, runs them through execution.execute(), audits."""
    feature = _make_feature_mock()
    record_class = MagicMock()
    record_class.side_effect = lambda **kw: MagicMock(model_dump=lambda: kw)
    feature.feature_record_class.return_value = record_class
    mock_ml.lookup_feature.return_value = feature

    execution = MagicMock()
    execution.add_features.return_value = 2
    # execute() returns a context manager (the execution itself).
    execution.execute.return_value = execution
    execution.__enter__ = MagicMock(return_value=execution)
    execution.__exit__ = MagicMock(return_value=False)
    mock_ml.resume_execution.return_value = execution

    with _patch_feature_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["add_feature_values"](
                hostname="h",
                catalog_id="1",
                table="Image",
                feature_name="Quality",
                execution_rid="EXEC-1",
                entries=[
                    {"Image": "1-AAAA", "Quality_Type": "good"},
                    {"Image": "1-BBBB", "Quality_Type": "bad"},
                ],
            )
        )

    assert out == {
        "status": "added",
        "feature_name": "Quality",
        "execution_rid": "EXEC-1",
        "count": 2,
    }
    assert record_class.call_count == 2
    execution.add_features.assert_called_once()

    success = _success_calls(mock_audit, "deriva_ml_add_feature_values")
    assert success
    assert success[0].kwargs["target_table"] == "Image"
    assert success[0].kwargs["feature_name"] == "Quality"
    assert success[0].kwargs["execution_rid"] == "EXEC-1"
    assert success[0].kwargs["added_count"] == 2


async def test_add_feature_values_empty_entries_validation_error(
    feature_ctx, capturing_mcp, mock_ml
):
    """Empty entries -> arg-validation error, no audit, no upstream call."""
    with _patch_feature_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["add_feature_values"](
                hostname="h",
                catalog_id="1",
                table="Image",
                feature_name="Quality",
                execution_rid="EXEC-1",
                entries=[],
            )
        )
    assert "error" in out
    assert "entries" in out["error"]
    # Response-shape parity: attempted_count is present here too (=0)
    # so callers can read it unconditionally regardless of which
    # failure path triggered.
    assert out["attempted_count"] == 0
    assert mock_audit.call_count == 0
    mock_ml.lookup_feature.assert_not_called()


async def test_add_feature_values_record_construction_fails(feature_ctx, capturing_mcp, mock_ml):
    """A bad entry dict -> wrapped error with which entry failed."""
    feature = _make_feature_mock()
    record_class = MagicMock(side_effect=TypeError("missing required field"))
    feature.feature_record_class.return_value = record_class
    mock_ml.lookup_feature.return_value = feature

    with _patch_feature_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["add_feature_values"](
                hostname="h",
                catalog_id="1",
                table="Image",
                feature_name="Quality",
                execution_rid="EXEC-1",
                entries=[{"Image": "1-AAAA"}],
            )
        )
    assert "error" in out
    assert out["failed_entry_index"] == 0
    assert out["attempted_count"] == 1
    failed = _success_calls(mock_audit, "deriva_ml_add_feature_values_failed")
    assert failed


async def test_add_feature_values_upstream_failure(feature_ctx, capturing_mcp, mock_ml):
    """An upstream RuntimeError propagates through _error_envelope with audit."""
    feature = _make_feature_mock()
    record_class = MagicMock(side_effect=lambda **kw: MagicMock())
    feature.feature_record_class.return_value = record_class
    mock_ml.lookup_feature.return_value = feature
    mock_ml.resume_execution.side_effect = RuntimeError("execution missing")

    with _patch_feature_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["add_feature_values"](
                hostname="h",
                catalog_id="1",
                table="Image",
                feature_name="Quality",
                execution_rid="EXEC-1",
                entries=[{"Image": "1-AAAA"}],
            )
        )
    assert out["error"] == "execution missing"
    # response_fields always includes attempted_count for partial-state visibility.
    assert out["attempted_count"] == 1
    # No failed_entry_index since record construction succeeded.
    assert "failed_entry_index" not in out
    failed = _success_calls(mock_audit, "deriva_ml_add_feature_values_failed")
    assert failed
    assert failed[0].kwargs["execution_rid"] == "EXEC-1"
