"""Unit tests for feature domain tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import _success_calls, make_patch_audit

# Dual-patch context manager for the feature module. See
# ``make_patch_audit`` in ``tests/conftest.py`` for the canonical
# explanation of why both bind sites are patched together.
_patch_feature_audit = make_patch_audit("feature")


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

    Patches at the use-site (`deriva_ml_mcp_plugin.tools.feature.get_ml`) and
    imports the tool module *inside* the patch block so registration sees
    the mock.
    """
    with patch("deriva_ml_mcp_plugin.tools.feature.get_ml", return_value=mock_ml):
        from deriva_ml_mcp_plugin.tools import feature as feature_module

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

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_features"](hostname="h", catalog_id="1")
    )

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
    await capturing_mcp.tools["deriva_ml_list_features"](
        hostname="h", catalog_id="1", table="Image"
    )
    mock_ml.find_features.assert_called_once_with(table="Image")


async def test_list_features_preflight(feature_ctx, capturing_mcp, mock_ml):
    mock_ml.find_features.return_value = [
        _make_feature_mock(feature_name=f"F{i}", feature_table_name=f"FT{i:04d}") for i in range(7)
    ]
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_features"](
            hostname="h", catalog_id="1", preflight_count=True
        )
    )
    assert out["total_count"] == 7
    assert out["entities_fetched"] is False
    assert "action_required" in out


async def test_list_features_error_path(feature_ctx, capturing_mcp, mock_ml):
    mock_ml.find_features.side_effect = RuntimeError("schema down")
    with _patch_feature_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_list_features"](hostname="h", catalog_id="1")
        )
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
        await capturing_mcp.tools["deriva_ml_get_feature"](
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
            await capturing_mcp.tools["deriva_ml_get_feature"](
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
        await capturing_mcp.tools["deriva_ml_list_feature_values"](
            hostname="h", catalog_id="1", table="Image", feature_name="Quality"
        )
    )
    assert out["count"] == 3
    assert out["truncated"] is False
    assert [r["RID"] for r in out["records"]] == ["1-0000", "1-0001", "1-0002"]
    # selector=None and new kwargs passed through.
    mock_ml.feature_values.assert_called_once_with(
        "Image", "Quality", selector=None, materialize_limit=50_000, execution_rids=None
    )


async def test_list_feature_values_selector_newest(feature_ctx, capturing_mcp, mock_ml):
    """selector='newest' resolves to FeatureRecord.select_newest staticmethod."""
    mock_ml.feature_values.return_value = iter([_make_record_mock("1-AAAA")])
    await capturing_mcp.tools["deriva_ml_list_feature_values"](
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
    await capturing_mcp.tools["deriva_ml_list_feature_values"](
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
        await capturing_mcp.tools["deriva_ml_list_feature_values"](
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
    await capturing_mcp.tools["deriva_ml_list_feature_values"](
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
        await capturing_mcp.tools["deriva_ml_list_feature_values"](
            hostname="h",
            catalog_id="1",
            table="Image",
            feature_name="Quality",
            dataset_rid="DS-1",
        )
    )
    assert out["count"] == 1
    mock_ml.lookup_dataset.assert_called_once_with("DS-1")
    ds.feature_values.assert_called_once_with(
        "Image", "Quality", selector=None, materialize_limit=50_000, execution_rids=None
    )
    # Top-level catalog feature_values not used in dataset mode.
    mock_ml.feature_values.assert_not_called()


async def test_list_feature_values_preflight(feature_ctx, capturing_mcp, mock_ml):
    mock_ml.feature_values.return_value = iter([_make_record_mock(f"1-{i:04d}") for i in range(4)])
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_feature_values"](
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
            await capturing_mcp.tools["deriva_ml_list_feature_values"](
                hostname="h", catalog_id="1", table="Image", feature_name="Quality"
            )
        )
    assert out == {"error": "ermrest down"}
    assert mock_audit.call_count == 0


async def test_list_feature_values_forwards_execution_rids(feature_ctx, capturing_mcp, mock_ml):
    """execution_rids= is forwarded to deriva-ml's feature_values(execution_rids=...)."""
    mock_ml.feature_values.return_value = iter([_make_record_mock("1-FV-A")])

    await capturing_mcp.tools["deriva_ml_list_feature_values"](
        hostname="h",
        catalog_id="1",
        table="Image",
        feature_name="Quality",
        execution_rids=["1-EXEC-A", "1-EXEC-B"],
    )

    mock_ml.feature_values.assert_called_once()
    assert mock_ml.feature_values.call_args.kwargs.get("execution_rids") == [
        "1-EXEC-A",
        "1-EXEC-B",
    ]


async def test_list_feature_values_default_max_results_passed(feature_ctx, capturing_mcp, mock_ml):
    """The default max_results=50000 is forwarded as materialize_limit=50000."""
    mock_ml.feature_values.return_value = iter([_make_record_mock("1-FV-A")])

    await capturing_mcp.tools["deriva_ml_list_feature_values"](
        hostname="h", catalog_id="1", table="Image", feature_name="Quality"
    )

    mock_ml.feature_values.assert_called_once()
    assert mock_ml.feature_values.call_args.kwargs.get("materialize_limit") == 50_000


async def test_list_feature_values_records_carry_rid(feature_ctx, capturing_mcp, mock_ml):
    """Each returned record carries a non-null ``RID``.

    Regression for the curator/02 finding from the 2026-05-26 e2e run
    (``deriva-ml-model-template-e2e/findings/curator/02-feature-values-
    next-after-rid-empty-string.md``): prior to the deriva-ml fix that
    added ``RID`` to ``FeatureRecord``, every returned record had
    ``RID: null`` because the projection dropped the system column.
    This test pins that records arrive with a non-null RID.
    """
    recs = [_make_record_mock(f"1-{i:04d}", Image_Class="cat") for i in range(3)]
    mock_ml.feature_values.return_value = iter(recs)

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_feature_values"](
            hostname="h", catalog_id="1", table="Image", feature_name="Quality"
        )
    )
    assert out["count"] == 3
    for record in out["records"]:
        assert record["RID"] is not None, f"record is missing RID: {record}"
        assert isinstance(record["RID"], str)


async def test_list_feature_values_cursor_advances(feature_ctx, capturing_mcp, mock_ml):
    """Pagination cursor advances — page 1 and page 2 are disjoint.

    Regression for the curator/02 finding from the 2026-05-26 e2e run
    (``deriva-ml-model-template-e2e/findings/curator/02-feature-values-
    next-after-rid-empty-string.md``): before the fix, ``next_after_rid``
    came back as ``""`` because the records had no ``RID`` attribute, so
    the cursor never advanced and the caller looped on page 1 forever.
    This test pins the contract: when ``truncated`` is True the
    ``next_after_rid`` is a real RID, and passing it as ``after_rid``
    on the next call returns the next page disjoint from the first.
    """
    # 10 records, page size 4 -> 3 pages of (4, 4, 2).
    all_recs = [_make_record_mock(f"1-{i:04d}") for i in range(10)]
    mock_ml.feature_values.return_value = iter(all_recs)

    page1 = json.loads(
        await capturing_mcp.tools["deriva_ml_list_feature_values"](
            hostname="h", catalog_id="1", table="Image", feature_name="Quality", limit=4
        )
    )
    assert page1["count"] == 4
    assert page1["truncated"] is True
    assert page1["next_after_rid"] is not None
    assert page1["next_after_rid"] != ""
    assert page1["next_after_rid"] == page1["records"][-1]["RID"]

    # Reset the iterator for the second call (the mock returns a fresh iter).
    mock_ml.feature_values.return_value = iter(all_recs)
    page2 = json.loads(
        await capturing_mcp.tools["deriva_ml_list_feature_values"](
            hostname="h",
            catalog_id="1",
            table="Image",
            feature_name="Quality",
            limit=4,
            after_rid=page1["next_after_rid"],
        )
    )
    assert page2["count"] == 4
    page1_rids = {r["RID"] for r in page1["records"]}
    page2_rids = {r["RID"] for r in page2["records"]}
    assert page1_rids.isdisjoint(page2_rids), (
        f"page 1 and page 2 overlap: {page1_rids & page2_rids}"
    )


async def test_list_feature_values_next_after_rid_is_none_when_not_truncated(
    feature_ctx, capturing_mcp, mock_ml
):
    """``next_after_rid`` is ``None`` (not ``""``) when the page is not truncated.

    Companion to ``test_list_feature_values_cursor_advances`` -- pins the
    other half of the curator/02 finding: when there's no next page, the
    cursor field must be JSON ``null`` so callers can use the standard
    "while next_after_rid is not None" pagination idiom.
    """
    recs = [_make_record_mock(f"1-{i:04d}") for i in range(3)]
    mock_ml.feature_values.return_value = iter(recs)

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_feature_values"](
            hostname="h", catalog_id="1", table="Image", feature_name="Quality", limit=100
        )
    )
    assert out["count"] == 3
    assert out["truncated"] is False
    assert out["next_after_rid"] is None


async def test_list_feature_values_enriches_records_when_upstream_missing_rid(
    feature_ctx, capturing_mcp, mock_ml
):
    """Records arriving with ``RID=None`` are enriched via a second pathBuilder query.

    Reproduces the curator/02 finding scenario directly: the upstream
    ``ml.feature_values()`` returns records with ``RID=None`` because
    deriva-ml's ``FeatureRecord`` projection drops the system column.
    The MCP layer notices, queries the feature table again to map
    ``(Execution, target_rid, RCT) -> RID``, and attaches the row's
    RID to each record before pagination.
    """

    # Build records that look like what upstream ml.feature_values()
    # produces today: RID=None on the mock, the (Execution, Image, RCT)
    # triple exposed via attribute access for the enrichment lookup.
    def _no_rid_record(execution: str, image: str, rct: str) -> MagicMock:
        rec = MagicMock()
        rec.RID = None
        rec.Execution = execution
        rec.Image = image
        rec.RCT = rct
        rec.model_dump.return_value = {
            "RID": None,
            "Feature_Name": "Image_Classification",
            "Execution": execution,
            "Image": image,
            "RCT": rct,
            "Image_Class": "cat",
        }
        return rec

    upstream_recs = [
        _no_rid_record("EXEC-1", "IMG-1", "2026-01-01T00:00:01Z"),
        _no_rid_record("EXEC-1", "IMG-2", "2026-01-01T00:00:02Z"),
    ]
    mock_ml.feature_values.return_value = iter(upstream_recs)

    # Stand up the pathBuilder shape the enricher walks: ml.pathBuilder()
    # -> .schemas[<schema>].tables[<feature_table>].entities().fetch()
    # returns dicts with RID + Execution + Image + RCT.
    feature_obj = _make_feature_mock(
        feature_name="Image_Classification",
        target_table_name="Image",
        feature_table_name="Execution_Image_Image_Classification",
    )
    feature_obj.feature_table.schema = MagicMock()
    feature_obj.feature_table.schema.name = "deriva-ml"
    mock_ml.lookup_feature.return_value = feature_obj

    raw_rows = [
        {
            "RID": "FV-1",
            "Execution": "EXEC-1",
            "Image": "IMG-1",
            "RCT": "2026-01-01T00:00:01Z",
        },
        {
            "RID": "FV-2",
            "Execution": "EXEC-1",
            "Image": "IMG-2",
            "RCT": "2026-01-01T00:00:02Z",
        },
    ]
    pb = MagicMock()
    pb.schemas.__getitem__.return_value.tables.__getitem__.return_value.entities.return_value.fetch.return_value = iter(
        raw_rows
    )
    mock_ml.pathBuilder.return_value = pb

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_feature_values"](
            hostname="h",
            catalog_id="1",
            table="Image",
            feature_name="Image_Classification",
        )
    )
    assert out["count"] == 2
    rids_returned = {r["RID"] for r in out["records"]}
    assert rids_returned == {"FV-1", "FV-2"}, (
        f"expected enrichment to attach FV-1/FV-2, got {rids_returned}"
    )
    # Cursor field is None when not truncated.
    assert out["truncated"] is False
    assert out["next_after_rid"] is None


async def test_list_feature_values_drops_records_without_rid(feature_ctx, capturing_mcp, mock_ml):
    """Records arriving without a ``RID`` attribute are dropped.

    Defensive: in old deriva-ml versions (pre-1.40) ``FeatureRecord``
    didn't carry ``RID``. The MCP tool now refuses to surface such
    records to the wire because the caller cannot correlate them with
    the catalog and the cursor would degenerate to ``""``. Once every
    pinned deriva-ml is fixed this branch becomes unreachable -- the
    test stays as a guardrail.
    """
    good = _make_record_mock("1-AAAA")
    # A record whose RID attribute is None — the upstream bug shape.
    bad = MagicMock()
    bad.RID = None
    bad.model_dump.return_value = {"RID": None, "Feature_Name": "Quality"}
    mock_ml.feature_values.return_value = iter([good, bad])

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_feature_values"](
            hostname="h", catalog_id="1", table="Image", feature_name="Quality"
        )
    )
    assert out["count"] == 1
    assert out["records"][0]["RID"] == "1-AAAA"


async def test_list_feature_values_max_results_translates_to_error_envelope(
    feature_ctx, capturing_mcp, mock_ml
):
    """When deriva-ml raises DerivaMLMaterializeLimitExceeded, wire returns {"error": ...}."""
    from deriva_ml import DerivaMLMaterializeLimitExceeded

    mock_ml.feature_values.side_effect = DerivaMLMaterializeLimitExceeded(
        actual_count=12_345,
        limit=100,
    )
    result = await capturing_mcp.tools["deriva_ml_list_feature_values"](
        hostname="h",
        catalog_id="1",
        table="Image",
        feature_name="Quality",
        max_results=100,
    )
    payload = json.loads(result)
    assert "error" in payload
    # Message contains the cap and the actual count (verify both show up)
    assert "100" in payload["error"]
    assert "12345" in payload["error"]


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
            await capturing_mcp.tools["deriva_ml_create_feature"](
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
            await capturing_mcp.tools["deriva_ml_create_feature"](
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
            await capturing_mcp.tools["deriva_ml_delete_feature"](
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
            await capturing_mcp.tools["deriva_ml_delete_feature"](
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
            await capturing_mcp.tools["deriva_ml_delete_feature"](
                hostname="h", catalog_id="1", table="Image", feature_name="Quality"
            )
        )
    assert out == {"error": "permission denied"}
    failed = _success_calls(mock_audit, "deriva_ml_delete_feature_failed")
    assert failed
    assert failed[0].kwargs["error_type"] == "RuntimeError"
