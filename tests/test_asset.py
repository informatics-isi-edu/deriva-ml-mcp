"""Unit tests for asset domain tools (v1.2)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

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


async def test_lookup_asset_executions_call_failure_falls_back_to_empty_list(
    asset_ctx, capturing_mcp, mock_ml
):
    """If list_executions raises, the detail payload still serializes (best-effort)."""
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


def _wire_pathbuilder_for_description(
    mock_ml, asset_table_name: str = "Image", schema_name: str = "demo-schema"
) -> MagicMock:
    """Configure mock_ml so the description pathBuilder write call records."""
    asset_table_obj = MagicMock()
    asset_table_obj.name = asset_table_name
    asset_table_obj.schema.name = schema_name
    mock_ml.model.name_to_table.return_value = asset_table_obj
    update_mock = MagicMock()
    asset_path = MagicMock()
    asset_path.update = update_mock
    schema_obj = MagicMock()
    schema_obj.tables = {asset_table_name: asset_path}
    mock_ml.pathBuilder.return_value.schemas = {schema_name: schema_obj}
    return update_mock


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
    """description-only edit writes the row via pathBuilder; types untouched."""
    asset = _make_asset_mock(
        asset_rid="1-AAAA", asset_table="Image", asset_types=["X"], description="old"
    )
    mock_ml.lookup_asset.return_value = asset
    update_mock = _wire_pathbuilder_for_description(mock_ml)

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
    # The pathBuilder write happened with the right shape.
    update_mock.assert_called_once_with([{"RID": "1-AAAA", "Description": "new desc"}])
    # In-memory mirror updated.
    assert asset.description == "new desc"
    # Type APIs were not touched.
    asset.add_asset_type.assert_not_called()
    asset.remove_asset_type.assert_not_called()
    success = _success_calls(mock_audit, "deriva_ml_update_asset")
    assert success
    assert success[0].kwargs["updated_fields"] == ["description"]


async def test_update_asset_types_and_description_together(asset_ctx, capturing_mcp, mock_ml):
    """Passing both kwargs edits both; updated_fields lists both."""
    asset = _make_asset_mock(asset_rid="1-AAAA", asset_types=["A"])
    mock_ml.lookup_asset.return_value = asset
    update_mock = _wire_pathbuilder_for_description(mock_ml)

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
    # Diff-add ran (B is new); description write also ran.
    asset.add_asset_type.assert_called_once_with("B")
    update_mock.assert_called_once_with([{"RID": "1-AAAA", "Description": "brand new"}])


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
