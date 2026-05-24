"""Unit tests for asset domain tools (v1.2)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from tests.conftest import _success_calls, make_patch_audit

# Dual-patch context manager for the asset module. See
# ``make_patch_audit`` in ``tests/_helpers.py`` for the canonical
# explanation of why both bind sites are patched together.
_patch_asset_audit = make_patch_audit("asset")


def _make_asset_mock(
    asset_rid: str = "1-AAAA",
    filename: str = "scan.png",
    length: int = 12345,
    md5: str = "abc",
    url: str = "/hatrac/scan.png",
    description: str = "MRI scan",
    asset_table: str = "Image",
    asset_types: list[str] | None = None,
    executions: list | None = None,
) -> MagicMock:
    """Build an Asset-shaped MagicMock."""
    a = MagicMock()
    a.asset_rid = asset_rid
    a.filename = filename
    a.length = length
    a.md5 = md5
    a.url = url
    a.description = description
    a.asset_table = asset_table
    a.asset_types = list(asset_types) if asset_types else []
    a.list_executions.return_value = list(executions) if executions else []
    return a


def _make_execution_record_mock(
    rid: str = "1-EXEC", asset_role: str | None = "Output"
) -> MagicMock:
    rec = MagicMock()
    rec.execution_rid = rid
    rec.asset_role = asset_role
    return rec


@pytest.fixture()
def asset_ctx(ctx, mock_ml):
    """Register asset tools with mock_ml as the DerivaML stand-in.

    Patches at the use-site (``deriva_ml_mcp.tools.asset.get_ml``) and
    imports the tool module *inside* the patch block so registration
    sees the mock.
    """
    with patch("deriva_ml_mcp.tools.asset.get_ml", return_value=mock_ml):
        from deriva_ml_mcp.tools import asset as asset_module

        asset_module.register(ctx)
        yield ctx


# ---------------------------------------------------------------------------
# list_assets
#
# Note: ``deriva_ml_list_asset_tables`` was retired in v3.4 along with
# the ``ml/asset-tables`` resource. Asset table discovery now happens
# through the schema-scoped ``ml/assets/{schema}`` resource (tested in
# ``test_resources.py``).
# ---------------------------------------------------------------------------


async def test_list_assets_success(asset_ctx, capturing_mcp, mock_ml):
    """Each Asset renders into the summary shape, sorted by RID."""
    mock_ml.list_assets.return_value = [
        _make_asset_mock("1-BBBB", filename="b.png"),
        _make_asset_mock("1-AAAA", filename="a.png"),
    ]
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_assets"](
            hostname="h", catalog_id="1", asset_table="Image"
        )
    )
    assert out["count"] == 2
    # Sorted ascending by RID.
    assert [a["rid"] for a in out["assets"]] == ["1-AAAA", "1-BBBB"]
    assert out["assets"][0]["filename"] == "a.png"
    assert out["truncated"] is False
    assert out["next_after_rid"] is None
    mock_ml.list_assets.assert_called_with("Image")


async def test_list_assets_pagination_truncates(asset_ctx, capturing_mcp, mock_ml):
    """When the page is full, truncated=True and next_after_rid is the last RID."""
    mock_ml.list_assets.return_value = [_make_asset_mock(f"1-{i:04d}") for i in range(5)]
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_assets"](
            hostname="h", catalog_id="1", asset_table="Image", limit=2
        )
    )
    assert out["count"] == 2
    assert out["truncated"] is True
    assert out["next_after_rid"] == "1-0001"


async def test_list_assets_preflight_count(asset_ctx, capturing_mcp, mock_ml):
    """preflight_count returns total without rendering rows."""
    mock_ml.list_assets.return_value = [_make_asset_mock(f"1-{i:04d}") for i in range(7)]
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_assets"](
            hostname="h", catalog_id="1", asset_table="Image", preflight_count=True
        )
    )
    assert out["total_count"] == 7
    assert out["entities_fetched"] is False


async def test_list_assets_failure_returns_error_envelope(asset_ctx, capturing_mcp, mock_ml):
    mock_ml.list_assets.side_effect = RuntimeError("not an asset table")
    with _patch_asset_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_list_assets"](
                hostname="h", catalog_id="1", asset_table="Image"
            )
        )
    assert out == {"error": "not an asset table"}
    # Read-only: no audit on failure.
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# find_assets (cross-table query surface)
#
# Wraps DerivaML.find_assets(asset_table=None, asset_type=None). At
# least one identifier is required at the MCP layer -- unfiltered scan
# of every asset in every table is intentionally not exposed (use
# deriva_ml_list_assets per table when you want the full contents of
# one table).
# ---------------------------------------------------------------------------


async def test_find_assets_by_type_returns_matching_rows(asset_ctx, capturing_mcp, mock_ml):
    """Happy path: asset_type filter passed through; results render
    via the same _summarize_asset path as list_assets (sorted ascending
    by RID).
    """
    mock_ml.find_assets.return_value = iter(
        [
            _make_asset_mock("1-BBBB", asset_table="Trained_Model", asset_types=["Model_File"]),
            _make_asset_mock("1-AAAA", asset_table="Trained_Model", asset_types=["Model_File"]),
        ]
    )
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_find_assets"](
            hostname="h", catalog_id="1", asset_type="Model_File"
        )
    )
    assert out["count"] == 2
    assert [a["rid"] for a in out["assets"]] == ["1-AAAA", "1-BBBB"]
    assert {a["asset_table"] for a in out["assets"]} == {"Trained_Model"}
    assert out["truncated"] is False
    assert out["next_after_rid"] is None
    mock_ml.find_assets.assert_called_with(asset_table=None, asset_type="Model_File")


async def test_find_assets_cross_table_returns_rows_from_multiple_tables(
    asset_ctx, capturing_mcp, mock_ml
):
    """Cross-table semantics: when the same asset_type is present in
    two different asset tables, find_assets returns rows from both --
    the per-row asset_table field tells the caller which table each
    row came from.
    """
    mock_ml.find_assets.return_value = iter(
        [
            _make_asset_mock("1-AAAA", asset_table="Trained_Model", asset_types=["Model_File"]),
            _make_asset_mock("1-CCCC", asset_table="Model_Variant", asset_types=["Model_File"]),
            _make_asset_mock("1-BBBB", asset_table="Trained_Model", asset_types=["Model_File"]),
        ]
    )
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_find_assets"](
            hostname="h", catalog_id="1", asset_type="Model_File"
        )
    )
    assert out["count"] == 3
    # Mixed tables surface in the per-row asset_table field.
    tables_by_rid = {a["rid"]: a["asset_table"] for a in out["assets"]}
    assert tables_by_rid == {
        "1-AAAA": "Trained_Model",
        "1-BBBB": "Trained_Model",
        "1-CCCC": "Model_Variant",
    }


async def test_find_assets_empty_match_returns_empty_list_not_error(
    asset_ctx, capturing_mcp, mock_ml
):
    """No matches => count=0, empty assets list, no error envelope."""
    mock_ml.find_assets.return_value = iter([])
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_find_assets"](
            hostname="h", catalog_id="1", asset_type="Missing_Type"
        )
    )
    assert out["count"] == 0
    assert out["assets"] == []
    assert out["truncated"] is False
    assert out["next_after_rid"] is None
    assert "error" not in out


async def test_find_assets_scoped_to_one_table(asset_ctx, capturing_mcp, mock_ml):
    """asset_table without asset_type narrows the scan to one table."""
    mock_ml.find_assets.return_value = iter(
        [_make_asset_mock("1-AAAA", asset_table="Image")]
    )
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_find_assets"](
            hostname="h", catalog_id="1", asset_table="Image"
        )
    )
    assert out["count"] == 1
    assert out["assets"][0]["asset_table"] == "Image"
    mock_ml.find_assets.assert_called_with(asset_table="Image", asset_type=None)


async def test_find_assets_both_scoping_kwargs_are_forwarded(asset_ctx, capturing_mcp, mock_ml):
    """asset_type + asset_table both honored -- the Python find_assets
    signature accepts both simultaneously."""
    mock_ml.find_assets.return_value = iter(
        [_make_asset_mock("1-AAAA", asset_table="Trained_Model", asset_types=["Model_File"])]
    )
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_find_assets"](
            hostname="h",
            catalog_id="1",
            asset_type="Model_File",
            asset_table="Trained_Model",
        )
    )
    assert out["count"] == 1
    mock_ml.find_assets.assert_called_with(
        asset_table="Trained_Model", asset_type="Model_File"
    )


async def test_find_assets_validation_requires_one_identifier(
    asset_ctx, capturing_mcp, mock_ml
):
    """Neither asset_type nor asset_table => validation error envelope,
    no catalog call. Unfiltered cross-catalog scans are intentionally
    not exposed."""
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_find_assets"](
            hostname="h", catalog_id="1"
        )
    )
    assert "error" in out
    assert "at least one" in out["error"]
    mock_ml.find_assets.assert_not_called()


async def test_find_assets_pagination_truncates(asset_ctx, capturing_mcp, mock_ml):
    """Standard cursor pagination: page filled => truncated=True,
    next_after_rid = last RID of page."""
    mock_ml.find_assets.return_value = iter(
        [_make_asset_mock(f"1-{i:04d}", asset_table="Image", asset_types=["X"]) for i in range(5)]
    )
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_find_assets"](
            hostname="h", catalog_id="1", asset_type="X", limit=2
        )
    )
    assert out["count"] == 2
    assert out["truncated"] is True
    assert out["next_after_rid"] == "1-0001"


async def test_find_assets_preflight_count(asset_ctx, capturing_mcp, mock_ml):
    """preflight_count returns total without rendering rows."""
    mock_ml.find_assets.return_value = iter(
        [_make_asset_mock(f"1-{i:04d}", asset_table="Image", asset_types=["X"]) for i in range(7)]
    )
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_find_assets"](
            hostname="h", catalog_id="1", asset_type="X", preflight_count=True
        )
    )
    assert out["total_count"] == 7
    assert out["entities_fetched"] is False
    assert "asset_type" in out["action_required"]


async def test_find_assets_failure_returns_error_envelope(asset_ctx, capturing_mcp, mock_ml):
    """Read-only: a deriva-ml failure surfaces as {"error": ...} with
    no audit emission."""
    mock_ml.find_assets.side_effect = RuntimeError("Model_File: term not found")
    with _patch_asset_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_find_assets"](
                hostname="h", catalog_id="1", asset_type="Model_File"
            )
        )
    assert out == {"error": "Model_File: term not found"}
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# lookup_asset
# ---------------------------------------------------------------------------


async def test_lookup_asset_success_bundles_executions(asset_ctx, capturing_mcp, mock_ml):
    """Detail payload has the full asset summary plus the executions list."""
    asset = _make_asset_mock(
        asset_rid="1-AAAA",
        asset_types=["Training_Data"],
        executions=[
            _make_execution_record_mock("1-EXEC1", asset_role="Output"),
            _make_execution_record_mock("1-EXEC2", asset_role="Input"),
        ],
    )
    mock_ml.lookup_asset.return_value = asset

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_lookup_asset"](
            hostname="h", catalog_id="1", asset_rid="1-AAAA"
        )
    )
    assert out["rid"] == "1-AAAA"
    assert out["filename"] == "scan.png"
    assert out["url"] == "/hatrac/scan.png"
    assert out["description"] == "MRI scan"
    assert out["asset_table"] == "Image"
    assert out["asset_types"] == ["Training_Data"]
    assert out["executions"] == [
        {"rid": "1-EXEC1", "asset_role": "Output"},
        {"rid": "1-EXEC2", "asset_role": "Input"},
    ]


async def test_lookup_asset_no_executions(asset_ctx, capturing_mcp, mock_ml):
    """Asset with no executions returns an empty list (not missing key)."""
    asset = _make_asset_mock(asset_rid="1-AAAA", executions=[])
    mock_ml.lookup_asset.return_value = asset

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_lookup_asset"](
            hostname="h", catalog_id="1", asset_rid="1-AAAA"
        )
    )
    assert out["executions"] == []


async def test_lookup_asset_executions_call_failure_surfaces_error(
    asset_ctx, capturing_mcp, mock_ml
):
    """Issue #41 regression: a list_executions failure now surfaces in
    ``executions_error`` rather than being silently swallowed into ``[]``.

    Prior behaviour: blanket ``except Exception`` flipped the list to
    ``[]`` and gave no way for the caller to distinguish "no
    executions" from "lookup failed". Today the asset detail still
    serializes (best-effort on the rest of the payload), but the
    failure is reported.
    """
    asset = _make_asset_mock(asset_rid="1-AAAA")
    asset.list_executions.side_effect = RuntimeError("association table missing")
    mock_ml.lookup_asset.return_value = asset

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_lookup_asset"](
            hostname="h", catalog_id="1", asset_rid="1-AAAA"
        )
    )
    # Best-effort: the asset detail still came through.
    assert out["rid"] == "1-AAAA"
    assert out["executions"] == []
    # New in #41: the error is reported, not swallowed.
    assert out["executions_error"] == "RuntimeError: association table missing"


async def test_lookup_asset_no_execution_association_returns_clean_empty(
    asset_ctx, capturing_mcp, mock_ml
):
    """Issue #41: the legitimate "asset table has no Execution association"
    case (raised by ``DerivaModel.find_association`` as the typed
    ``NoAssociationException``) keeps the clean empty-list shape --
    no spurious error message. This is the only case we silence;
    everything else surfaces in ``executions_error``.
    """
    from deriva_ml.core.exceptions import NoAssociationException

    asset = _make_asset_mock(asset_rid="1-AAAA")
    asset.list_executions.side_effect = NoAssociationException("Image", "Execution")
    mock_ml.lookup_asset.return_value = asset

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_lookup_asset"](
            hostname="h", catalog_id="1", asset_rid="1-AAAA"
        )
    )
    assert out["rid"] == "1-AAAA"
    assert out["executions"] == []
    assert out["executions_error"] is None


async def test_lookup_asset_generic_no_association_message_is_not_silenced(
    asset_ctx, capturing_mcp, mock_ml
):
    """Regression: a bare ``DerivaMLException`` carrying the literal
    message ``"No association tables found ... Execution"`` is **not**
    silently swallowed -- it surfaces in ``executions_error``.

    This proves the helper no longer depends on the exception's message
    text to decide whether the error was the legitimate "no association"
    case. Only the typed ``NoAssociationException`` subclass triggers
    the clean-empty path now.
    """
    from deriva_ml.core.exceptions import DerivaMLException

    asset = _make_asset_mock(asset_rid="1-AAAA")
    asset.list_executions.side_effect = DerivaMLException(
        "No association tables found between Image and Execution."
    )
    mock_ml.lookup_asset.return_value = asset

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_lookup_asset"](
            hostname="h", catalog_id="1", asset_rid="1-AAAA"
        )
    )
    assert out["rid"] == "1-AAAA"
    assert out["executions"] == []
    assert out["executions_error"] is not None
    assert "DerivaMLException" in out["executions_error"]
    assert "No association tables found" in out["executions_error"]


async def test_lookup_asset_b18_fresh_then_failed_lookup(asset_ctx, capturing_mcp, mock_ml):
    """Issue #41 (B18) scenario regression: an MCP request inserts a
    fresh association, a second MCP request looks up the asset, and an
    error inside the executions lookup is now surfaced rather than
    masquerading as "no executions". Two-call sequence -- the original
    bug's pattern -- proves the second call's failure is no longer
    invisible.
    """
    # First call: lookup succeeds with no executions yet.
    asset_v1 = _make_asset_mock(asset_rid="1-AAAA", executions=[])
    mock_ml.lookup_asset.return_value = asset_v1
    out1 = json.loads(
        await capturing_mcp.tools["deriva_ml_lookup_asset"](
            hostname="h", catalog_id="1", asset_rid="1-AAAA"
        )
    )
    assert out1["executions"] == []
    assert out1["executions_error"] is None

    # Now an association is "inserted" (mock state change) and the
    # second lookup hits a transient ermrest error -- the original B18
    # symptom.
    asset_v2 = _make_asset_mock(asset_rid="1-AAAA")
    asset_v2.list_executions.side_effect = RuntimeError("ermrest 503 Service Unavailable")
    mock_ml.lookup_asset.return_value = asset_v2
    out2 = json.loads(
        await capturing_mcp.tools["deriva_ml_lookup_asset"](
            hostname="h", catalog_id="1", asset_rid="1-AAAA"
        )
    )
    # Pre-fix behaviour: out2["executions"] == [] and no signal of why.
    # Post-fix behaviour: out2["executions"] == [] AND the caller is
    # told the lookup failed.
    assert out2["executions"] == []
    assert out2["executions_error"] == "RuntimeError: ermrest 503 Service Unavailable"


async def test_lookup_asset_failure_returns_error_envelope(asset_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_asset.side_effect = RuntimeError("not an asset")
    with _patch_asset_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_lookup_asset"](
                hostname="h", catalog_id="1", asset_rid="1-XXXX"
            )
        )
    assert out == {"error": "not an asset"}
    # Read-only: no audit on failure.
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# update_asset
# ---------------------------------------------------------------------------


def _wire_description_setter_mock(asset) -> PropertyMock:
    """Replace the asset's plain ``description`` attribute with a PropertyMock.

    v0.5.0: the plugin now uses deriva-ml v1.38.0's write-through
    ``Asset.description`` setter instead of the legacy pathBuilder
    workaround. Tests need PropertyMock to detect that the setter
    fires (plain attribute assignment on a MagicMock would be silent).

    PropertyMock is attached to the TYPE, not the instance, so the
    caller must ``del type(asset).description`` after the test to keep
    the descriptor from leaking to sibling tests sharing a MagicMock
    class.
    """
    description_prop = PropertyMock(return_value="initial")
    type(asset).description = description_prop
    return description_prop


async def test_update_asset_types_set_diff_adds_and_removes(asset_ctx, capturing_mcp, mock_ml):
    """Set-style diff: terms in the new list but not current => add; opposite => remove."""
    asset = _make_asset_mock(asset_rid="1-AAAA", asset_types=["A", "C"])
    mock_ml.lookup_asset.return_value = asset

    with _patch_asset_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_asset"](
                hostname="h",
                catalog_id="1",
                asset_rid="1-AAAA",
                asset_types=["A", "B"],
            )
        )
    assert out["status"] == "updated"
    assert out["updated_fields"] == ["asset_types"]
    # B is added; C is removed; A is left alone (was already present).
    asset.add_asset_type.assert_called_once_with("B")
    asset.remove_asset_type.assert_called_once_with("C")
    success = _success_calls(mock_audit, "deriva_ml_update_asset")
    assert success
    assert success[0].kwargs["added"] == ["B"]
    assert success[0].kwargs["removed"] == ["C"]


async def test_update_asset_types_no_diff_skips_calls(asset_ctx, capturing_mcp, mock_ml):
    """If the desired list matches current, no add/remove calls fire."""
    asset = _make_asset_mock(asset_rid="1-AAAA", asset_types=["X", "Y"])
    mock_ml.lookup_asset.return_value = asset
    with patch("deriva_ml_mcp.tools.asset.audit_event"):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_asset"](
                hostname="h",
                catalog_id="1",
                asset_rid="1-AAAA",
                asset_types=["Y", "X"],
            )
        )
    assert out["status"] == "updated"
    asset.add_asset_type.assert_not_called()
    asset.remove_asset_type.assert_not_called()


async def test_update_asset_description_only(asset_ctx, capturing_mcp, mock_ml):
    """description-only edit invokes the v1.38.0 write-through setter; types untouched."""
    asset = _make_asset_mock(
        asset_rid="1-AAAA", asset_table="Image", asset_types=["X"], description="old"
    )
    mock_ml.lookup_asset.return_value = asset
    description_prop = _wire_description_setter_mock(asset)

    try:
        with _patch_asset_audit() as mock_audit:
            out = json.loads(
                await capturing_mcp.tools["deriva_ml_update_asset"](
                    hostname="h",
                    catalog_id="1",
                    asset_rid="1-AAAA",
                    description="new desc",
                )
            )
        assert out["status"] == "updated"
        assert out["updated_fields"] == ["description"]
        # The v1.38.0 write-through setter was invoked with the new value.
        description_prop.assert_any_call("new desc")
        # Type APIs were not touched.
        asset.add_asset_type.assert_not_called()
        asset.remove_asset_type.assert_not_called()
        success = _success_calls(mock_audit, "deriva_ml_update_asset")
        assert success
        assert success[0].kwargs["updated_fields"] == ["description"]
    finally:
        del type(asset).description


async def test_update_asset_types_and_description_together(asset_ctx, capturing_mcp, mock_ml):
    """Passing both kwargs edits both; updated_fields lists both."""
    asset = _make_asset_mock(asset_rid="1-AAAA", asset_types=["A"])
    mock_ml.lookup_asset.return_value = asset
    description_prop = _wire_description_setter_mock(asset)

    try:
        with patch("deriva_ml_mcp.tools.asset.audit_event"):
            out = json.loads(
                await capturing_mcp.tools["deriva_ml_update_asset"](
                    hostname="h",
                    catalog_id="1",
                    asset_rid="1-AAAA",
                    asset_types=["A", "B"],
                    description="brand new",
                )
            )
        assert out["status"] == "updated"
        assert sorted(out["updated_fields"]) == ["asset_types", "description"]
        # Diff-add ran (B is new); description setter also fired.
        asset.add_asset_type.assert_called_once_with("B")
        description_prop.assert_any_call("brand new")
    finally:
        del type(asset).description


async def test_update_asset_validation_both_none(asset_ctx, capturing_mcp, mock_ml):
    """update_asset() with neither field returns the validation error envelope."""
    with _patch_asset_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_asset"](
                hostname="h", catalog_id="1", asset_rid="1-AAAA"
            )
        )
    assert "error" in out
    assert "at least one" in out["error"]
    # No catalog work and no audit on validation failure.
    mock_ml.lookup_asset.assert_not_called()
    assert mock_audit.call_count == 0


async def test_update_asset_failure_emits_failed_audit(asset_ctx, capturing_mcp, mock_ml):
    """A mid-update failure emits the _failed audit event with partial state."""
    mock_ml.lookup_asset.side_effect = RuntimeError("asset not found")
    with _patch_asset_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_asset"](
                hostname="h",
                catalog_id="1",
                asset_rid="1-MISSING",
                asset_types=["X"],
            )
        )
    assert out["error"] == "asset not found"
    assert out["asset_rid"] == "1-MISSING"
    assert out["added_done"] == []
    assert out["removed_done"] == []
    failed = _success_calls(mock_audit, "deriva_ml_update_asset_failed")
    assert len(failed) == 1
    assert failed[0].kwargs["asset_rid"] == "1-MISSING"
    assert failed[0].kwargs["updated_fields"] == []


async def test_update_asset_partial_remove_failure_surfaces_progress(
    asset_ctx, capturing_mcp, mock_ml
):
    """If a remove mid-loop fails, response + audit surface the partial state."""
    asset = _make_asset_mock(asset_rid="1-AAAA", asset_types=["A", "B"])
    mock_ml.lookup_asset.return_value = asset
    # add succeeds; first remove succeeds (A); second remove fails (B).
    call_count = {"n": 0}

    def remove_side_effect(term):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("term B not found")

    asset.remove_asset_type.side_effect = remove_side_effect

    with _patch_asset_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_asset"](
                hostname="h",
                catalog_id="1",
                asset_rid="1-AAAA",
                asset_types=["NewTag"],
            )
        )
    assert out["error"] == "term B not found"
    # NewTag was added; A was removed; B failed -- partial state visible.
    assert out["added_done"] == ["NewTag"]
    assert out["removed_done"] == ["A"]
    failed = _success_calls(mock_audit, "deriva_ml_update_asset_failed")
    assert failed
    assert failed[0].kwargs["added_done"] == ["NewTag"]
    assert failed[0].kwargs["removed_done"] == ["A"]
