"""Unit tests for dataset domain tools (Batches 1, 2 & 3: read, mutate, complex)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import _success_calls, make_patch_audit

# Dual-patch context manager for the dataset module. See
# ``make_patch_audit`` in ``tests/conftest.py`` for the canonical
# explanation of why both bind sites are patched together.
_patch_audit = make_patch_audit("dataset")


def _make_dataset_mock(
    rid: str,
    description: str = "",
    dataset_types: list[str] | None = None,
    current_version: str = "0.1.0",
    chaise_url: str = "https://example.org/chaise",
) -> MagicMock:
    """Build a MagicMock that quacks like a Dataset object."""
    ds = MagicMock()
    ds.dataset_rid = rid
    ds.description = description
    ds.dataset_types = dataset_types or []
    ds.current_version = current_version
    ds.get_chaise_url.return_value = chaise_url
    return ds


@pytest.fixture()
def dataset_ctx(ctx, mock_ml):
    """Register dataset tools with mock_ml as the DerivaML stand-in.

    Patches at the use-site (`deriva_ml_mcp.tools.dataset.get_ml`) and imports
    the tool module *inside* the patch block so registration sees the mock.
    """
    with patch("deriva_ml_mcp.tools.dataset.get_ml", return_value=mock_ml):
        from deriva_ml_mcp.tools import dataset as dataset_module

        dataset_module.register(ctx)
        yield ctx


# ---------------------------------------------------------------------------
# list_datasets
# ---------------------------------------------------------------------------


async def test_list_datasets_success(dataset_ctx, capturing_mcp, mock_ml):
    """Page mode returns a serialized list with truncated/next_after_rid."""
    mock_ml.find_datasets.return_value = [
        _make_dataset_mock("1-AAAA", "first", ["Training"], "1.0.0"),
        _make_dataset_mock("1-BBBB", "second", ["Testing"], "1.1.0"),
    ]
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_datasets"](hostname="h", catalog_id="1")
    )

    assert out["count"] == 2
    assert out["truncated"] is False
    assert out["next_after_rid"] is None
    assert {d["rid"] for d in out["datasets"]} == {"1-AAAA", "1-BBBB"}
    mock_ml.find_datasets.assert_called_once_with(deleted=False)


async def test_list_datasets_preflight_returns_count_only(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.find_datasets.return_value = [_make_dataset_mock(f"1-{i:04d}") for i in range(5)]
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_datasets"](
            hostname="h", catalog_id="1", preflight_count=True
        )
    )
    assert out["total_count"] == 5
    assert out["entities_fetched"] is False
    assert "action_required" in out


async def test_list_datasets_pagination_caps_limit_and_advances_cursor(
    dataset_ctx, capturing_mcp, mock_ml
):
    """limit > 1000 caps to 1000; after_rid skips already-seen rows."""
    mock_ml.find_datasets.return_value = [_make_dataset_mock(f"1-{i:04d}") for i in range(2500)]

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_datasets"](
            hostname="h", catalog_id="1", limit=2000
        )
    )
    assert out["count"] == 1000  # capped
    assert out["truncated"] is True
    assert out["next_after_rid"] == "1-0999"

    # Advance cursor.
    out2 = json.loads(
        await capturing_mcp.tools["deriva_ml_list_datasets"](
            hostname="h", catalog_id="1", after_rid="1-0999", limit=5
        )
    )
    assert [d["rid"] for d in out2["datasets"]] == [
        "1-1000",
        "1-1001",
        "1-1002",
        "1-1003",
        "1-1004",
    ]
    assert out2["truncated"] is True


async def test_list_datasets_error_path(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.find_datasets.side_effect = RuntimeError("boom")
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_datasets"](hostname="h", catalog_id="1")
    )
    assert out == {"error": "boom"}


# ---------------------------------------------------------------------------
# get_dataset
# ---------------------------------------------------------------------------


async def test_get_dataset_success(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", "desc", ["Training"], "1.0.0")
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_get_dataset"](
            hostname="h", catalog_id="1", dataset_rid="1-AAAA"
        )
    )
    assert out["rid"] == "1-AAAA"
    assert out["description"] == "desc"
    assert out["dataset_types"] == ["Training"]
    assert out["current_version"] == "1.0.0"
    assert out["chaise_url"] == "https://example.org/chaise"
    assert "history" not in out
    mock_ml.lookup_dataset.assert_called_once_with("1-AAAA")


async def test_get_dataset_with_history(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    history_entry = MagicMock()
    history_entry.dataset_version = "1.0.0"
    history_entry.snapshot = "snap1"
    history_entry.description = "v1"
    history_entry.execution_rid = "EXEC-1"
    ds.dataset_history.return_value = [history_entry]
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_get_dataset"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            include_history=True,
        )
    )
    assert out["history"] == [
        {
            "version": "1.0.0",
            "snapshot": "snap1",
            "description": "v1",
            "execution_rid": "EXEC-1",
        }
    ]


async def test_get_dataset_not_found(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_dataset.side_effect = RuntimeError("Dataset not found")
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_get_dataset"](
            hostname="h", catalog_id="1", dataset_rid="missing"
        )
    )
    assert out == {"error": "Dataset not found"}


# ---------------------------------------------------------------------------
# list_dataset_members
# ---------------------------------------------------------------------------


async def test_list_dataset_members_summary_mode(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.list_dataset_members.return_value = {
        "Image": [{"RID": "i1"}, {"RID": "i2"}, {"RID": "i3"}],
        "Subject": [{"RID": "s1"}],
    }
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_dataset_members"](
            hostname="h", catalog_id="1", dataset_rid="1-AAAA"
        )
    )
    assert out["summary"] == {"Image": 3, "Subject": 1}
    assert out["total"] == 4
    assert sorted(out["tables"]) == ["Image", "Subject"]


async def test_list_dataset_members_page_mode(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.list_dataset_members.return_value = {
        "Image": [{"RID": f"i-{i:03d}", "name": f"img{i}"} for i in range(5)],
    }
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_dataset_members"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            element_table="Image",
            limit=2,
        )
    )
    assert out["element_table"] == "Image"
    assert out["returned_count"] == 2
    assert out["truncated"] is True
    assert out["next_after_rid"] == "i-001"
    assert [r["RID"] for r in out["rows"]] == ["i-000", "i-001"]


async def test_list_dataset_members_preflight(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.list_dataset_members.return_value = {
        "Image": [{"RID": f"i-{i:03d}"} for i in range(7)],
    }
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_dataset_members"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            element_table="Image",
            preflight_count=True,
        )
    )
    assert out["element_table"] == "Image"
    assert out["total_count"] == 7
    assert out["entities_fetched"] is False


async def test_list_dataset_members_unknown_element_table(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.list_dataset_members.return_value = {"Image": []}
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_dataset_members"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            element_table="NotARealTable",
        )
    )
    assert "error" in out
    assert "NotARealTable" in out["error"]


async def test_list_dataset_members_error_path(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_dataset.side_effect = RuntimeError("nope")
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_dataset_members"](
            hostname="h", catalog_id="1", dataset_rid="1-AAAA"
        )
    )
    assert out == {"error": "nope"}


# ---------------------------------------------------------------------------
# list_dataset_relations
# ---------------------------------------------------------------------------


async def test_list_dataset_relations_both(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    parents = [_make_dataset_mock("1-PARENT", "p")]
    children = [
        _make_dataset_mock("1-CH1", "c1"),
        _make_dataset_mock("1-CH2", "c2"),
    ]
    ds.list_dataset_parents.return_value = parents
    ds.list_dataset_children.return_value = children
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_dataset_relations"](
            hostname="h", catalog_id="1", dataset_rid="1-AAAA"
        )
    )
    assert {p["rid"] for p in out["parents"]} == {"1-PARENT"}
    assert {c["rid"] for c in out["children"]} == {"1-CH1", "1-CH2"}


async def test_list_dataset_relations_parents_only(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.list_dataset_parents.return_value = [_make_dataset_mock("1-PARENT")]
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_dataset_relations"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            direction="parents",
        )
    )
    assert "parents" in out
    assert "children" not in out
    ds.list_dataset_children.assert_not_called()


async def test_list_dataset_relations_error(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_dataset.side_effect = RuntimeError("relations boom")
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_dataset_relations"](
            hostname="h", catalog_id="1", dataset_rid="x"
        )
    )
    assert out == {"error": "relations boom"}


async def test_list_dataset_relations_after_rid_with_both_emits_warning(
    dataset_ctx, capturing_mcp, mock_ml
):
    """direction='both' with after_rid is incoherent (parents/children RIDs
    are disjoint). The tool must ignore the cursor and surface a warning."""
    ds = _make_dataset_mock("1-AAAA")
    # Both sides return rows that would all be filtered if after_rid actually
    # applied — proves the cursor was ignored.
    ds.list_dataset_parents.return_value = [
        _make_dataset_mock("1-PAR-001"),
        _make_dataset_mock("1-PAR-002"),
    ]
    ds.list_dataset_children.return_value = [
        _make_dataset_mock("1-CHI-001"),
        _make_dataset_mock("1-CHI-002"),
    ]
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_dataset_relations"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            direction="both",
            after_rid="1-ZZZ-999",
        )
    )

    assert "warning" in out
    assert "after_rid was ignored" in out["warning"]
    # Cursor was ignored → all rows returned on both sides.
    assert len(out["parents"]) == 2
    assert len(out["children"]) == 2


async def test_list_dataset_relations_after_rid_respected_in_single_direction(
    dataset_ctx, capturing_mcp, mock_ml
):
    """direction='parents' (or 'children') uses after_rid normally — no warning."""
    ds = _make_dataset_mock("1-AAAA")
    ds.list_dataset_parents.return_value = [
        _make_dataset_mock("1-PAR-001"),
        _make_dataset_mock("1-PAR-002"),
        _make_dataset_mock("1-PAR-003"),
    ]
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_dataset_relations"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            direction="parents",
            after_rid="1-PAR-001",
        )
    )

    assert "warning" not in out
    # Cursor advanced past 1-PAR-001 → only 1-PAR-002 and 1-PAR-003 remain.
    assert [p["rid"] for p in out["parents"]] == ["1-PAR-002", "1-PAR-003"]


async def test_list_datasets_truncated_on_exact_limit_match(dataset_ctx, capturing_mcp, mock_ml):
    """Convention (matches deriva-mcp-core's get_entities): when the page
    returns exactly `limit` rows, truncated=True even if no more rows exist.
    Callers must call again to confirm there's no next page."""
    # Exactly 5 datasets, limit=5 → page is full → truncated=True per convention.
    mock_ml.find_datasets.return_value = [_make_dataset_mock(f"1-{i:04d}") for i in range(5)]

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_datasets"](hostname="h", catalog_id="1", limit=5)
    )

    assert out["count"] == 5
    assert out["truncated"] is True  # Even though len(items) == limit exactly.
    assert out["next_after_rid"] == "1-0004"


# ---------------------------------------------------------------------------
# list_dataset_element_types
# ---------------------------------------------------------------------------


async def test_list_dataset_element_types_success(dataset_ctx, capturing_mcp, mock_ml):
    table_a = MagicMock()
    table_a.name = "Image"
    table_a.schema.name = "domain"
    table_b = MagicMock()
    table_b.name = "Subject"
    table_b.schema.name = "domain"
    mock_ml.list_dataset_element_types.return_value = [table_a, table_b]

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_dataset_element_types"](
            hostname="h", catalog_id="1"
        )
    )
    assert out["count"] == 2
    assert out["element_types"] == [
        {"name": "Image", "schema": "domain"},
        {"name": "Subject", "schema": "domain"},
    ]


async def test_list_dataset_element_types_error(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.list_dataset_element_types.side_effect = RuntimeError("bad")
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_list_dataset_element_types"](
            hostname="h", catalog_id="1"
        )
    )
    assert out == {"error": "bad"}


# ---------------------------------------------------------------------------
# bag_info
# ---------------------------------------------------------------------------


async def test_bag_info_success(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.bag_info.return_value = {
        "tables": {"Image": {"row_count": 100, "is_asset": True, "asset_bytes": 12345}},
        "total_rows": 100,
        "total_asset_bytes": 12345,
        "total_asset_size": "12 KB",
        "cache_status": "not_cached",
        "cache_path": None,
    }

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_bag_info"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            version="1.0.0",
        )
    )
    assert out["total_rows"] == 100
    assert out["cache_status"] == "not_cached"
    # Verify spec construction.
    spec_arg = mock_ml.bag_info.call_args.args[0]
    assert spec_arg.rid == "1-AAAA"
    assert str(spec_arg.version) == "1.0.0"
    assert spec_arg.exclude_tables is None


async def test_bag_info_with_exclude_tables(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.bag_info.return_value = {"tables": {}, "total_rows": 0}

    await capturing_mcp.tools["deriva_ml_bag_info"](
        hostname="h",
        catalog_id="1",
        dataset_rid="1-AAAA",
        version="1.0.0",
        exclude_tables=["Big_Asset"],
    )

    spec_arg = mock_ml.bag_info.call_args.args[0]
    assert spec_arg.exclude_tables == {"Big_Asset"}


async def test_bag_info_error(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.bag_info.side_effect = RuntimeError("cant compute")
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_bag_info"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            version="1.0.0",
        )
    )
    assert out == {"error": "cant compute"}


# ---------------------------------------------------------------------------
# get_dataset_spec
# ---------------------------------------------------------------------------


async def test_get_dataset_spec_with_explicit_version(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", "desc", ["Training"], "2.0.0")
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_get_dataset_spec"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            version="1.5.0",
        )
    )
    assert out["spec"] == 'DatasetSpecConfig(rid="1-AAAA", version="1.5.0")'
    assert out["version"] == "1.5.0"
    assert out["dataset_rid"] == "1-AAAA"
    assert out["description"] == "desc"
    assert out["dataset_types"] == ["Training"]
    assert out["warning"] is None


async def test_get_dataset_spec_falls_back_to_current_version(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", "desc", [], "2.0.0")
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_get_dataset_spec"](
            hostname="h", catalog_id="1", dataset_rid="1-AAAA"
        )
    )
    assert out["version"] == "2.0.0"
    assert out["spec"] == 'DatasetSpecConfig(rid="1-AAAA", version="2.0.0")'
    assert out["warning"] is not None
    assert "current version" in out["warning"]


async def test_get_dataset_spec_error(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_dataset.side_effect = RuntimeError("missing")
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_get_dataset_spec"](
            hostname="h", catalog_id="1", dataset_rid="1-AAAA"
        )
    )
    assert out == {"error": "missing"}


# ===========================================================================
# Batch 2: mutation tools
# ===========================================================================


# ---------------------------------------------------------------------------
# create_dataset
# ---------------------------------------------------------------------------


async def test_create_dataset_success(dataset_ctx, capturing_mcp, mock_ml):
    """Happy path: returns the new dataset summary and emits success audit."""
    new_ds = _make_dataset_mock("1-NEW", "desc", ["Training"], "1.0.0")
    with (
        patch("deriva_ml_mcp.tools.dataset.Dataset") as mock_dataset_cls,
        _patch_audit() as mock_audit,
    ):
        mock_dataset_cls.create_dataset.return_value = new_ds
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_create_dataset"](
                hostname="h",
                catalog_id="1",
                execution_rid="EXEC-1",
                dataset_types=["Training"],
                description="desc",
            )
        )
    assert out["status"] == "created"
    assert out["rid"] == "1-NEW"
    assert out["description"] == "desc"
    assert out["dataset_types"] == ["Training"]
    assert out["current_version"] == "1.0.0"
    assert out["execution_rid"] == "EXEC-1"

    success = _success_calls(mock_audit, "deriva_ml_create_dataset")
    assert success, "expected deriva_ml_create_dataset audit event"
    assert success[0].kwargs["execution_rid"] == "EXEC-1"
    assert success[0].kwargs["dataset_rid"] == "1-NEW"
    assert success[0].kwargs["dataset_types"] == ["Training"]


async def test_create_dataset_parses_explicit_version(dataset_ctx, capturing_mcp, mock_ml):
    """A version string is parsed into a DatasetVersion before being passed in."""
    new_ds = _make_dataset_mock("1-NEW", current_version="2.5.7")
    with (
        patch("deriva_ml_mcp.tools.dataset.Dataset") as mock_dataset_cls,
        patch("deriva_ml_mcp.tools.dataset.audit_event"),
    ):
        mock_dataset_cls.create_dataset.return_value = new_ds
        await capturing_mcp.tools["deriva_ml_create_dataset"](
            hostname="h",
            catalog_id="1",
            execution_rid="EXEC-1",
            version="2.5.7",
        )

    passed_version = mock_dataset_cls.create_dataset.call_args.kwargs["version"]
    # DatasetVersion subclasses semver.Version; verify the components.
    assert (passed_version.major, passed_version.minor, passed_version.patch) == (2, 5, 7)


async def test_create_dataset_failure_emits_failed_audit(dataset_ctx, capturing_mcp, mock_ml):
    """Error path: audit fires `_failed`, response has {error: ...}."""
    with (
        patch("deriva_ml_mcp.tools.dataset.Dataset") as mock_dataset_cls,
        _patch_audit() as mock_audit,
    ):
        mock_dataset_cls.create_dataset.side_effect = RuntimeError("invalid term")
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_create_dataset"](
                hostname="h",
                catalog_id="1",
                execution_rid="EXEC-1",
                dataset_types=["BadTerm"],
            )
        )
    assert out == {"error": "invalid term"}

    failed = _success_calls(mock_audit, "deriva_ml_create_dataset_failed")
    assert failed, "expected deriva_ml_create_dataset_failed audit event"
    assert failed[0].kwargs["error_type"] == "RuntimeError"
    assert failed[0].kwargs["execution_rid"] == "EXEC-1"
    assert failed[0].kwargs["dataset_types"] == ["BadTerm"]
    # No success event was emitted.
    assert not _success_calls(mock_audit, "deriva_ml_create_dataset")


# ---------------------------------------------------------------------------
# delete_dataset
# ---------------------------------------------------------------------------


async def test_delete_dataset_success(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_delete_dataset"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA"
            )
        )
    assert out == {"status": "deleted", "dataset_rid": "1-AAAA", "recursive": False}
    mock_ml.delete_dataset.assert_called_once_with(ds, recurse=False)

    success = _success_calls(mock_audit, "deriva_ml_delete_dataset")
    assert success
    assert success[0].kwargs["dataset_rid"] == "1-AAAA"
    assert success[0].kwargs["recursive"] is False


async def test_delete_dataset_recursive(dataset_ctx, capturing_mcp, mock_ml):
    """recurse=True is forwarded to ml.delete_dataset and recorded in audit."""
    ds = _make_dataset_mock("1-AAAA")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_delete_dataset"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA", recurse=True
            )
        )
    assert out["recursive"] is True
    mock_ml.delete_dataset.assert_called_once_with(ds, recurse=True)
    success = _success_calls(mock_audit, "deriva_ml_delete_dataset")
    assert success[0].kwargs["recursive"] is True


async def test_delete_dataset_error_path(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_dataset.side_effect = RuntimeError("not found")
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_delete_dataset"](
                hostname="h", catalog_id="1", dataset_rid="missing"
            )
        )
    assert out == {"error": "not found"}
    failed = _success_calls(mock_audit, "deriva_ml_delete_dataset_failed")
    assert failed
    assert failed[0].kwargs["error_type"] == "RuntimeError"
    assert failed[0].kwargs["dataset_rid"] == "missing"


# ---------------------------------------------------------------------------
# add_dataset_members
# ---------------------------------------------------------------------------


async def test_add_dataset_members_with_list(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", current_version="1.1.0")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_add_dataset_members"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                member_rids=["r1", "r2", "r3"],
                description="adding three",
                execution_rid="EXEC-1",
            )
        )
    assert out["status"] == "success"
    assert out["added_count"] == 3
    assert out["dataset_rid"] == "1-AAAA"
    assert out["new_version"] == "1.1.0"
    ds.add_dataset_members.assert_called_once_with(
        members=["r1", "r2", "r3"], description="adding three", execution_rid="EXEC-1"
    )
    success = _success_calls(mock_audit, "deriva_ml_add_dataset_members")
    assert success
    assert success[0].kwargs["added_count"] == 3
    assert success[0].kwargs["new_version"] == "1.1.0"


async def test_add_dataset_members_with_dict(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", current_version="1.2.0")
    mock_ml.lookup_dataset.return_value = ds
    with patch("deriva_ml_mcp.tools.dataset.audit_event"):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_add_dataset_members"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                members_by_table={"Image": ["i1", "i2"], "Subject": ["s1"]},
            )
        )
    assert out["added_count"] == 3
    forwarded = ds.add_dataset_members.call_args.kwargs["members"]
    assert forwarded == {"Image": ["i1", "i2"], "Subject": ["s1"]}


async def test_add_dataset_members_validates_inputs(dataset_ctx, capturing_mcp, mock_ml):
    """Both args missing -> early ValueError-style {error: ...}, no audit call."""
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_add_dataset_members"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA"
            )
        )
    assert "error" in out
    assert "exactly one" in out["error"]
    assert mock_audit.call_count == 0
    mock_ml.lookup_dataset.assert_not_called()


async def test_add_dataset_members_both_args_rejected(dataset_ctx, capturing_mcp, mock_ml):
    """Supplying both member_rids and members_by_table is rejected."""
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_add_dataset_members"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            member_rids=["r1"],
            members_by_table={"Image": ["i1"]},
        )
    )
    assert "error" in out


async def test_add_dataset_members_error_path(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.add_dataset_members.side_effect = RuntimeError("cycle detected")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_add_dataset_members"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                member_rids=["r1"],
            )
        )
    assert out == {"error": "cycle detected"}
    failed = _success_calls(mock_audit, "deriva_ml_add_dataset_members_failed")
    assert failed
    assert failed[0].kwargs["attempted_count"] == 1


# ---------------------------------------------------------------------------
# delete_dataset_members
# ---------------------------------------------------------------------------


async def test_delete_dataset_members_success(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", current_version="1.3.0")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_delete_dataset_members"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                member_rids=["r1", "r2"],
                description="trimming",
                execution_rid="EXEC-2",
            )
        )
    assert out["status"] == "success"
    assert out["removed_count"] == 2
    assert out["new_version"] == "1.3.0"
    ds.delete_dataset_members.assert_called_once_with(
        members=["r1", "r2"], description="trimming", execution_rid="EXEC-2"
    )
    success = _success_calls(mock_audit, "deriva_ml_delete_dataset_members")
    assert success
    assert success[0].kwargs["removed_count"] == 2


async def test_delete_dataset_members_error(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.delete_dataset_members.side_effect = RuntimeError("rid not in dataset")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_delete_dataset_members"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                member_rids=["r1", "r2", "r3"],
            )
        )
    assert out == {"error": "rid not in dataset"}
    failed = _success_calls(mock_audit, "deriva_ml_delete_dataset_members_failed")
    assert failed
    assert failed[0].kwargs["attempted_count"] == 3


# ---------------------------------------------------------------------------
# update_dataset (v1.2 -- renamed from update_dataset_types and widened to
# take optional `dataset_types` (set-style diff) + optional `description`.)
# ---------------------------------------------------------------------------


async def test_update_dataset_types_set_diff_adds_and_removes(dataset_ctx, capturing_mcp, mock_ml):
    """`dataset_types` is set-style: diff against current types, add+remove."""
    ds = _make_dataset_mock("1-AAAA", dataset_types=["Training", "Stale"], current_version="1.4.0")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                dataset_types=["Training", "Validation"],
            )
        )
    assert out["status"] == "updated"
    assert out["updated_fields"] == ["dataset_types"]
    assert out["added"] == ["Validation"]
    assert out["removed"] == ["Stale"]
    assert out["new_version"] == "1.4.0"
    # add_dataset_types is called with the diff-add list (not the desired
    # final list) -- Training is already present and not re-added.
    ds.add_dataset_types.assert_called_once_with(["Validation"])
    ds.remove_dataset_type.assert_called_once_with("Stale")
    success = _success_calls(mock_audit, "deriva_ml_update_dataset")
    assert success
    assert success[0].kwargs["added"] == ["Validation"]
    assert success[0].kwargs["removed"] == ["Stale"]


async def test_update_dataset_types_no_diff_skips_calls(dataset_ctx, capturing_mcp, mock_ml):
    """If desired == current, neither add_dataset_types nor remove fire."""
    ds = _make_dataset_mock("1-AAAA", dataset_types=["Training"])
    mock_ml.lookup_dataset.return_value = ds
    with patch("deriva_ml_mcp.tools.dataset.audit_event"):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                dataset_types=["Training"],
            )
        )
    assert out["status"] == "updated"
    assert out["added"] == []
    assert out["removed"] == []
    ds.add_dataset_types.assert_not_called()
    ds.remove_dataset_type.assert_not_called()


async def test_update_dataset_remove_only_loops_per_term(dataset_ctx, capturing_mcp, mock_ml):
    """Diff-remove of multiple terms loops through remove_dataset_type."""
    ds = _make_dataset_mock("1-AAAA", dataset_types=["Keep", "A", "B", "C"])
    mock_ml.lookup_dataset.return_value = ds
    with patch("deriva_ml_mcp.tools.dataset.audit_event"):
        await capturing_mcp.tools["deriva_ml_update_dataset"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            dataset_types=["Keep"],
        )
    # add_dataset_types should not be called when no diff-add.
    ds.add_dataset_types.assert_not_called()
    assert ds.remove_dataset_type.call_count == 3
    # Removes happen in sorted order (the diff is computed via set
    # arithmetic and then sorted for determinism).
    assert [c.args[0] for c in ds.remove_dataset_type.call_args_list] == ["A", "B", "C"]


async def test_update_dataset_validation_both_none(dataset_ctx, capturing_mcp, mock_ml):
    """update_dataset() with no fields returns the validation error envelope."""
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA"
            )
        )
    assert "error" in out
    assert "at least one" in out["error"]
    # No catalog work and no audit on validation failure.
    mock_ml.lookup_dataset.assert_not_called()
    assert mock_audit.call_count == 0


async def test_update_dataset_types_error(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.add_dataset_types.side_effect = RuntimeError("unknown term")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                dataset_types=["BadTerm"],
            )
        )
    # Error response now includes partial-state fields so the LLM can
    # see which sub-operations completed before the failure.
    assert out["error"] == "unknown term"
    assert out["dataset_rid"] == "1-AAAA"
    assert out["added_done"] == []  # add_dataset_types raised; nothing committed
    assert out["removed_done"] == []
    assert out["added_requested"] == ["BadTerm"]
    failed = _success_calls(mock_audit, "deriva_ml_update_dataset_failed")
    assert failed
    assert failed[0].kwargs["added"] == ["BadTerm"]
    assert failed[0].kwargs["added_done"] == []
    assert failed[0].kwargs["removed_done"] == []


async def test_update_dataset_types_partial_remove_failure_surfaces_progress(
    dataset_ctx, capturing_mcp, mock_ml
):
    """If a remove mid-way through the loop fails, the error response and
    audit must surface which terms were already removed."""
    # Current types include the two we're going to remove (A, B) plus
    # one we're keeping ("Keep") -- since dataset_types is set-style,
    # passing ["Keep", "NewTag"] diffs to add=[NewTag], remove=[A, B].
    ds = _make_dataset_mock("1-AAAA", dataset_types=["Keep", "A", "B"])
    # add_dataset_types succeeds; first remove succeeds; second remove fails.
    call_count = {"n": 0}

    def remove_side_effect(term):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("term B not found")

    ds.remove_dataset_type.side_effect = remove_side_effect
    mock_ml.lookup_dataset.return_value = ds

    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                dataset_types=["Keep", "NewTag"],
            )
        )

    # Error response shows partial state: NewTag added, A removed, B failed.
    assert out["error"] == "term B not found"
    assert out["added_done"] == ["NewTag"]
    assert out["removed_done"] == ["A"]
    assert out["added_requested"] == ["NewTag"]
    assert out["removed_requested"] == ["A", "B"]

    # Audit also captures partial state.
    failed = _success_calls(mock_audit, "deriva_ml_update_dataset_failed")
    assert failed
    assert failed[0].kwargs["added_done"] == ["NewTag"]
    assert failed[0].kwargs["removed_done"] == ["A"]


async def test_update_dataset_description_only(dataset_ctx, capturing_mcp, mock_ml):
    """description-only update writes via pathBuilder; types untouched."""
    ds = _make_dataset_mock("1-AAAA", dataset_types=["Training"], current_version="2.0.0")
    mock_ml.lookup_dataset.return_value = ds
    # The implementation reads ml._dataset_table for the schema/name pair
    # and then writes through pb.schemas[X].tables[Y].update(...). Mock
    # those out so the patched pathBuilder records the update call.
    table_mock = MagicMock()
    table_mock.schema.name = "deriva-ml"
    table_mock.name = "Dataset"
    mock_ml._dataset_table = table_mock
    update_mock = MagicMock()
    mock_ml.pathBuilder.return_value.schemas = {
        "deriva-ml": MagicMock(tables={"Dataset": MagicMock(update=update_mock)})
    }

    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                description="new desc",
            )
        )
    assert out["status"] == "updated"
    assert out["updated_fields"] == ["description"]
    # types fields are NOT present when only description was edited.
    assert "dataset_types" not in out
    assert "added" not in out
    # pathBuilder write happened with the right shape.
    update_mock.assert_called_once_with([{"RID": "1-AAAA", "Description": "new desc"}])
    # In-memory mirror updated too.
    assert ds.description == "new desc"
    # No type-mutation API was touched.
    ds.add_dataset_types.assert_not_called()
    ds.remove_dataset_type.assert_not_called()
    success = _success_calls(mock_audit, "deriva_ml_update_dataset")
    assert success
    assert success[0].kwargs["updated_fields"] == ["description"]


async def test_update_dataset_types_and_description_together(dataset_ctx, capturing_mcp, mock_ml):
    """Passing both fields edits both; updated_fields lists both."""
    ds = _make_dataset_mock("1-AAAA", dataset_types=["Training"], current_version="1.0.0")
    mock_ml.lookup_dataset.return_value = ds
    table_mock = MagicMock()
    table_mock.schema.name = "deriva-ml"
    table_mock.name = "Dataset"
    mock_ml._dataset_table = table_mock
    update_mock = MagicMock()
    mock_ml.pathBuilder.return_value.schemas = {
        "deriva-ml": MagicMock(tables={"Dataset": MagicMock(update=update_mock)})
    }

    with patch("deriva_ml_mcp.tools.dataset.audit_event"):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                dataset_types=["Training", "Validation"],
                description="combined edit",
            )
        )
    assert out["status"] == "updated"
    assert sorted(out["updated_fields"]) == ["dataset_types", "description"]
    # Diff add ran for the new term; description write also ran.
    ds.add_dataset_types.assert_called_once_with(["Validation"])
    update_mock.assert_called_once_with([{"RID": "1-AAAA", "Description": "combined edit"}])


# ---------------------------------------------------------------------------
# add_dataset_element_type
# ---------------------------------------------------------------------------


async def test_add_dataset_element_type_success(dataset_ctx, capturing_mcp, mock_ml):
    assoc = MagicMock()
    assoc.name = "Dataset_Image"
    mock_ml.add_dataset_element_type.return_value = assoc
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_add_dataset_element_type"](
                hostname="h", catalog_id="1", table_name="Image"
            )
        )
    assert out == {
        "status": "success",
        "table_name": "Image",
        "association_table": "Dataset_Image",
    }
    mock_ml.add_dataset_element_type.assert_called_once_with("Image")
    success = _success_calls(mock_audit, "deriva_ml_add_dataset_element_type")
    assert success
    assert success[0].kwargs["table_name"] == "Image"
    assert success[0].kwargs["association_table"] == "Dataset_Image"


async def test_add_dataset_element_type_error(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.add_dataset_element_type.side_effect = RuntimeError("not a domain table")
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_add_dataset_element_type"](
                hostname="h", catalog_id="1", table_name="Execution"
            )
        )
    assert out == {"error": "not a domain table"}
    failed = _success_calls(mock_audit, "deriva_ml_add_dataset_element_type_failed")
    assert failed
    assert failed[0].kwargs["table_name"] == "Execution"
    assert failed[0].kwargs["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# increment_dataset_version
# ---------------------------------------------------------------------------


async def test_increment_dataset_version_success(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", current_version="1.2.0")
    new_ver = MagicMock()
    new_ver.__str__ = lambda self: "1.3.0"
    ds.increment_dataset_version.return_value = new_ver
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_increment_dataset_version"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                component="minor",
                description="rebuilt members",
                execution_rid="EXEC-3",
            )
        )
    assert out["status"] == "success"
    assert out["dataset_rid"] == "1-AAAA"
    assert out["previous_version"] == "1.2.0"
    assert out["new_version"] == "1.3.0"
    assert out["component"] == "minor"

    # Check that VersionPart.minor was passed (not the raw string).
    from deriva_ml.dataset.aux_classes import VersionPart

    forwarded = ds.increment_dataset_version.call_args.kwargs
    assert forwarded["component"] == VersionPart.minor
    assert forwarded["description"] == "rebuilt members"
    assert forwarded["execution_rid"] == "EXEC-3"

    success = _success_calls(mock_audit, "deriva_ml_increment_dataset_version")
    assert success
    assert success[0].kwargs["component"] == "minor"
    assert success[0].kwargs["previous_version"] == "1.2.0"
    assert success[0].kwargs["new_version"] == "1.3.0"


async def test_increment_dataset_version_major(dataset_ctx, capturing_mcp, mock_ml):
    """component='major' is forwarded as VersionPart.major."""
    ds = _make_dataset_mock("1-AAAA", current_version="1.2.3")
    new_ver = MagicMock()
    new_ver.__str__ = lambda self: "2.0.0"
    ds.increment_dataset_version.return_value = new_ver
    mock_ml.lookup_dataset.return_value = ds
    with patch("deriva_ml_mcp.tools.dataset.audit_event"):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_increment_dataset_version"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA", component="major"
            )
        )
    assert out["new_version"] == "2.0.0"
    from deriva_ml.dataset.aux_classes import VersionPart

    assert ds.increment_dataset_version.call_args.kwargs["component"] == VersionPart.major


async def test_increment_dataset_version_error(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.increment_dataset_version.side_effect = RuntimeError("version conflict")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_increment_dataset_version"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                component="patch",
            )
        )
    assert out == {"error": "version conflict"}
    failed = _success_calls(mock_audit, "deriva_ml_increment_dataset_version_failed")
    assert failed
    assert failed[0].kwargs["component"] == "patch"


# ===========================================================================
# Batch 3: complex tools (cache_dataset, denormalize_dataset, split_dataset)
# ===========================================================================


# ---------------------------------------------------------------------------
# cache_dataset
# ---------------------------------------------------------------------------


def _bag_info_payload() -> dict:
    """Return a bag_info-shaped dict for cache_dataset returns."""
    return {
        "tables": {"Image": {"row_count": 10, "is_asset": True, "asset_bytes": 1024}},
        "total_rows": 10,
        "total_asset_bytes": 1024,
        "total_asset_size": "1 KB",
        "cache_status": "cached_materialized",
        "cache_path": "/tmp/cache/1-AAAA",
    }


async def test_cache_dataset_explicit_version(dataset_ctx, capturing_mcp, mock_ml):
    """version supplied explicitly: lookup_dataset is NOT called for fallback."""
    mock_ml.cache_dataset.return_value = _bag_info_payload()
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_cache_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                version="1.2.3",
            )
        )
    assert out["status"] == "success"
    assert out["dataset_rid"] == "1-AAAA"
    assert out["version"] == "1.2.3"
    assert out["materialize"] is True
    assert out["cache_status"] == "cached_materialized"
    assert out["cache_path"] == "/tmp/cache/1-AAAA"

    # spec carried the explicit version through.
    (spec_arg,) = mock_ml.cache_dataset.call_args.args
    assert spec_arg.rid == "1-AAAA"
    assert str(spec_arg.version) == "1.2.3"
    assert spec_arg.exclude_tables is None
    assert mock_ml.cache_dataset.call_args.kwargs["materialize"] is True

    # Fallback path NOT exercised.
    mock_ml.lookup_dataset.assert_not_called()

    success = _success_calls(mock_audit, "deriva_ml_cache_dataset")
    assert success
    assert success[0].kwargs["dataset_rid"] == "1-AAAA"
    assert success[0].kwargs["version"] == "1.2.3"
    assert success[0].kwargs["materialize"] is True


async def test_cache_dataset_falls_back_to_current_version(dataset_ctx, capturing_mcp, mock_ml):
    """version=None pulls current_version off the looked-up dataset."""
    ds = _make_dataset_mock("1-AAAA", current_version="2.0.0")
    mock_ml.lookup_dataset.return_value = ds
    mock_ml.cache_dataset.return_value = _bag_info_payload()
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_cache_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                materialize=False,
            )
        )
    assert out["version"] == "2.0.0"
    assert out["materialize"] is False
    (spec_arg,) = mock_ml.cache_dataset.call_args.args
    assert str(spec_arg.version) == "2.0.0"
    assert mock_ml.cache_dataset.call_args.kwargs["materialize"] is False

    success = _success_calls(mock_audit, "deriva_ml_cache_dataset")
    assert success[0].kwargs["version"] == "2.0.0"
    assert success[0].kwargs["materialize"] is False


async def test_cache_dataset_exclude_tables_converted_to_set(dataset_ctx, capturing_mcp, mock_ml):
    """exclude_tables converts list -> set for DatasetSpec, AND propagates
    into the success audit so operators can see which tables were skipped
    (M-8 fix from Batch 3 review)."""
    mock_ml.cache_dataset.return_value = _bag_info_payload()
    with _patch_audit() as mock_audit:
        await capturing_mcp.tools["deriva_ml_cache_dataset"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            version="1.0.0",
            exclude_tables=["Big_Asset", "Other"],
        )
    (spec_arg,) = mock_ml.cache_dataset.call_args.args
    assert spec_arg.exclude_tables == {"Big_Asset", "Other"}
    # Audit captures the excluded tables for operator visibility.
    success = _success_calls(mock_audit, "deriva_ml_cache_dataset")
    assert success
    assert success[0].kwargs["exclude_tables"] == ["Big_Asset", "Other"]


async def test_cache_dataset_error_path(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.cache_dataset.side_effect = RuntimeError("download failed")
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_cache_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                version="1.0.0",
            )
        )
    assert out == {"error": "download failed"}
    failed = _success_calls(mock_audit, "deriva_ml_cache_dataset_failed")
    assert failed
    assert failed[0].kwargs["dataset_rid"] == "1-AAAA"
    assert failed[0].kwargs["materialize"] is True


# ---------------------------------------------------------------------------
# denormalize_dataset
# ---------------------------------------------------------------------------


def _describe_payload(total_rows: int = 5) -> dict:
    """Return a Denormalizer.describe() shaped dict."""
    return {
        "row_per": "Image",
        "row_per_source": "auto-inferred",
        "row_per_candidates": ["Image"],
        "columns": [["Image.RID", "text"], ["Subject.Name", "text"]],
        "include_tables": ["Image", "Subject"],
        "via": [],
        "join_path": ["Image", "Subject"],
        "transparent_intermediates": [],
        "ambiguities": [],
        "estimated_row_count": {
            "in_scope_row_per_rows": total_rows,
            "orphan_rows": 0,
            "total": total_rows,
        },
        "anchors": {"total": total_rows, "by_type": {"Image": total_rows}},
        "source": "catalog",
    }


async def test_denormalize_dataset_validates_include_tables(dataset_ctx, capturing_mcp, mock_ml):
    """Empty include_tables short-circuits to a validation error (no audit)."""
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_denormalize_dataset"](
            hostname="h",
            catalog_id="1",
            include_tables=[],
        )
    )
    assert "error" in out
    assert "include_tables" in out["error"]
    mock_ml.estimate_denormalized_size.assert_not_called()


async def test_denormalize_dataset_catalog_shape(dataset_ctx, capturing_mcp, mock_ml):
    """dataset_rid=None hits estimate_denormalized_size and tags catalog_shape."""
    mock_ml.estimate_denormalized_size.return_value = {
        "columns": [["Image.RID", "text"]],
        "join_path": ["Image"],
        "tables": {"Image": {"row_count": 100, "is_asset": True, "asset_bytes": 1024}},
        "total_rows": 100,
        "total_asset_bytes": 1024,
        "total_asset_size": "1 KB",
    }
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_denormalize_dataset"](
            hostname="h",
            catalog_id="1",
            include_tables=["Image"],
        )
    )
    assert out["mode"] == "catalog_shape"
    assert out["include_tables"] == ["Image"]
    assert out["total_rows"] == 100
    mock_ml.estimate_denormalized_size.assert_called_once_with(["Image"])


async def test_denormalize_dataset_dataset_shape_only(dataset_ctx, capturing_mcp, mock_ml):
    """dataset_rid set + limit=0 returns shape only."""
    ds = _make_dataset_mock("1-AAAA")
    ds.describe_denormalized.return_value = _describe_payload()
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_denormalize_dataset"](
            hostname="h",
            catalog_id="1",
            include_tables=["Image", "Subject"],
            dataset_rid="1-AAAA",
        )
    )
    assert out["mode"] == "dataset_shape"
    assert out["dataset_rid"] == "1-AAAA"
    assert out["row_per"] == "Image"
    assert "rows" not in out
    ds.get_denormalized_as_dict.assert_not_called()


async def test_denormalize_dataset_with_rows(dataset_ctx, capturing_mcp, mock_ml):
    """limit>0 drains the generator, sorts, and paginates by Image.RID."""
    ds = _make_dataset_mock("1-AAAA")
    ds.describe_denormalized.return_value = _describe_payload(total_rows=3)

    sample_rows = [
        {"Image.RID": "1-IMG2", "Subject.Name": "B"},
        {"Image.RID": "1-IMG1", "Subject.Name": "A"},
        {"Image.RID": "1-IMG3", "Subject.Name": "C"},
    ]
    ds.get_denormalized_as_dict.return_value = iter(sample_rows)
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_denormalize_dataset"](
            hostname="h",
            catalog_id="1",
            include_tables=["Image", "Subject"],
            dataset_rid="1-AAAA",
            limit=2,
        )
    )
    assert out["mode"] == "dataset_rows"
    assert out["returned_count"] == 2
    assert out["truncated"] is True
    assert out["next_after_rid"] == "1-IMG2"
    assert [r["Image.RID"] for r in out["rows"]] == ["1-IMG1", "1-IMG2"]


async def test_denormalize_dataset_after_rid_advances_cursor(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.describe_denormalized.return_value = _describe_payload(total_rows=3)
    sample_rows = [
        {"Image.RID": "1-IMG1"},
        {"Image.RID": "1-IMG2"},
        {"Image.RID": "1-IMG3"},
    ]
    ds.get_denormalized_as_dict.return_value = iter(sample_rows)
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_denormalize_dataset"](
            hostname="h",
            catalog_id="1",
            include_tables=["Image"],
            dataset_rid="1-AAAA",
            limit=10,
            after_rid="1-IMG1",
        )
    )
    assert [r["Image.RID"] for r in out["rows"]] == ["1-IMG2", "1-IMG3"]
    assert out["truncated"] is False
    assert out["next_after_rid"] is None


async def test_denormalize_dataset_preflight_returns_estimated_count(
    dataset_ctx, capturing_mcp, mock_ml
):
    ds = _make_dataset_mock("1-AAAA")
    ds.describe_denormalized.return_value = _describe_payload(total_rows=42)
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_denormalize_dataset"](
            hostname="h",
            catalog_id="1",
            include_tables=["Image"],
            dataset_rid="1-AAAA",
            preflight_count=True,
        )
    )
    assert out["mode"] == "dataset_preflight"
    assert out["total_count"] == 42
    assert out["entities_fetched"] is False
    ds.get_denormalized_as_dict.assert_not_called()


async def test_denormalize_dataset_error_path(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.estimate_denormalized_size.side_effect = RuntimeError("schema error")
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_denormalize_dataset"](
            hostname="h",
            catalog_id="1",
            include_tables=["Image"],
        )
    )
    assert out == {"error": "schema error"}


async def test_denormalize_dataset_refuses_oversize_fetch_without_preflight(
    dataset_ctx, capturing_mcp, mock_ml
):
    """If the estimated row count is > 10x the requested limit, the tool
    refuses to drain the generator and routes the caller through preflight
    instead. Prevents accidental OOM on huge denormalized joins (I-1 fix
    from Batch 3 review)."""
    ds = _make_dataset_mock("1-AAAA")
    # Estimate is 100,000 rows, caller asked for limit=10 — refuse.
    ds.describe_denormalized.return_value = _describe_payload(total_rows=100_000)
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_denormalize_dataset"](
            hostname="h",
            catalog_id="1",
            include_tables=["Image"],
            dataset_rid="1-AAAA",
            limit=10,
        )
    )

    assert out["mode"] == "dataset_preflight_required"
    assert out["estimated_row_count"] == 100_000
    assert out["requested_limit"] == 10
    assert out["entities_fetched"] is False
    assert "preflight_count=True" in out["action_required"]
    # Critically: the generator was NOT drained.
    ds.get_denormalized_as_dict.assert_not_called()


async def test_denormalize_dataset_islice_bounds_generator_consumption(
    dataset_ctx, capturing_mcp, mock_ml
):
    """The implementation uses itertools.islice(gen, capped+1) so a huge
    generator is never fully materialized. Verify by checking that the
    rows returned match what islice would yield, not a sort over the
    full generator output."""
    ds = _make_dataset_mock("1-AAAA")
    # Estimated count is small enough that the oversize gate is not
    # triggered (50 < 10 * 5 = 50 → strictly greater is the gate, so
    # 49 passes through). Use 49 to slip under the threshold for limit=5.
    ds.describe_denormalized.return_value = _describe_payload(total_rows=49)
    # Build a generator that would be expensive to fully materialize:
    # yield 1000 rows in RID order. islice(gen, 6) takes only the first 6.
    consumed = {"n": 0}

    def gen():
        for i in range(1000):
            consumed["n"] += 1
            yield {"Image.RID": f"1-IMG{i:04d}"}

    ds.get_denormalized_as_dict.return_value = gen()
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.tools["deriva_ml_denormalize_dataset"](
            hostname="h",
            catalog_id="1",
            include_tables=["Image"],
            dataset_rid="1-AAAA",
            limit=5,
        )
    )

    # Returned page is a real page of 5; islice consumed at most 6.
    assert out["returned_count"] == 5
    assert consumed["n"] <= 6, f"generator consumed {consumed['n']} times (expected ≤ 6)"


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------


def _split_result_payload(*, with_validation: bool = False, dry_run: bool = False) -> dict:
    """Build a SplitResult.model_dump()-shaped dict."""
    payload: dict = {
        "source": "1-AAAA",
        "split": {"rid": "1-SPLT", "version": "1.0.0", "count": 100},
        "training": {"rid": "1-TRN", "version": "1.0.0", "count": 80},
        "testing": {"rid": "1-TST", "version": "1.0.0", "count": 20},
        "validation": None,
        "strategy": "random",
        "element_table": "Image",
        "seed": 42,
        "dry_run": dry_run,
    }
    if with_validation:
        payload["validation"] = {"rid": "1-VAL", "version": "1.0.0", "count": 10}
        payload["training"]["count"] = 70
    return payload


async def test_split_dataset_basic_success(dataset_ctx, capturing_mcp, mock_ml):
    payload = _split_result_payload()
    result_obj = MagicMock()
    result_obj.model_dump.return_value = payload
    with (
        patch("deriva_ml_mcp.tools.dataset._split_dataset", return_value=result_obj) as mock_split,
        _patch_audit() as mock_audit,
    ):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_split_dataset"](
                hostname="h",
                catalog_id="1",
                source_dataset_rid="1-AAAA",
            )
        )
    assert out["status"] == "success"
    assert out["source"] == "1-AAAA"
    assert out["training"]["rid"] == "1-TRN"
    assert out["testing"]["rid"] == "1-TST"
    assert out["validation"] is None
    assert out["strategy"] == "random"

    args, kwargs = mock_split.call_args
    assert args[0] is mock_ml
    assert args[1] == "1-AAAA"
    assert kwargs["test_size"] == 0.2
    assert kwargs["seed"] == 42
    assert kwargs["dry_run"] is False
    # split_description default is empty string and should be passed through.
    assert kwargs["split_description"] == ""

    success = _success_calls(mock_audit, "deriva_ml_split_dataset")
    assert success
    audit_kwargs = success[0].kwargs
    assert audit_kwargs["source_dataset_rid"] == "1-AAAA"
    assert audit_kwargs["strategy"] == "random"
    assert audit_kwargs["element_table"] == "Image"
    assert audit_kwargs["training_count"] == 80
    assert audit_kwargs["testing_count"] == 20
    assert audit_kwargs["validation_count"] is None
    # dry_run is intentionally NOT in the audit payload — non-dry-run runs
    # are the only ones that audit at all (the I-3 fix from Batch 3 review:
    # dry_run skips auditing entirely), so the field is implicit.
    assert "dry_run" not in audit_kwargs
    # No free-text description in the audit payload.
    assert "split_description" not in audit_kwargs


async def test_split_dataset_dry_run(dataset_ctx, capturing_mcp, mock_ml):
    """dry_run=True must NOT emit a success audit (I-3 fix from review).

    Audit logs are reserved for actual state changes; a dry_run computes
    the split assignment without creating any catalog datasets, so there
    is nothing to audit. The response still carries dry_run=True so the
    caller has full visibility."""
    payload = _split_result_payload(dry_run=True)
    result_obj = MagicMock()
    result_obj.model_dump.return_value = payload
    with (
        patch("deriva_ml_mcp.tools.dataset._split_dataset", return_value=result_obj) as mock_split,
        _patch_audit() as mock_audit,
    ):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_split_dataset"](
                hostname="h",
                catalog_id="1",
                source_dataset_rid="1-AAAA",
                dry_run=True,
            )
        )
    assert out["dry_run"] is True
    assert mock_split.call_args.kwargs["dry_run"] is True
    # NO success audit for dry_run.
    success = _success_calls(mock_audit, "deriva_ml_split_dataset")
    assert success == []
    # And no failure audit either (the run succeeded).
    failed = _success_calls(mock_audit, "deriva_ml_split_dataset_failed")
    assert failed == []


async def test_split_dataset_stratified_three_way(dataset_ctx, capturing_mcp, mock_ml):
    payload = _split_result_payload(with_validation=True)
    payload["strategy"] = "stratified"
    result_obj = MagicMock()
    result_obj.model_dump.return_value = payload
    with (
        patch("deriva_ml_mcp.tools.dataset._split_dataset", return_value=result_obj) as mock_split,
        _patch_audit() as mock_audit,
    ):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_split_dataset"](
                hostname="h",
                catalog_id="1",
                source_dataset_rid="1-AAAA",
                test_size=0.2,
                val_size=0.1,
                stratify_by_column="Image_Classification.Image_Class",
                stratify_missing="drop",
                include_tables=["Image", "Image_Classification"],
                training_types=["Labeled"],
            )
        )
    assert out["validation"]["rid"] == "1-VAL"
    assert out["validation"]["count"] == 10
    assert out["strategy"] == "stratified"

    kwargs = mock_split.call_args.kwargs
    assert kwargs["stratify_by_column"] == "Image_Classification.Image_Class"
    assert kwargs["stratify_missing"] == "drop"
    assert kwargs["include_tables"] == ["Image", "Image_Classification"]
    assert kwargs["val_size"] == 0.1
    assert kwargs["training_types"] == ["Labeled"]

    success = _success_calls(mock_audit, "deriva_ml_split_dataset")
    audit_kwargs = success[0].kwargs
    assert audit_kwargs["strategy"] == "stratified"
    assert audit_kwargs["training_count"] == 70
    assert audit_kwargs["validation_count"] == 10


async def test_split_dataset_error_path(dataset_ctx, capturing_mcp, mock_ml):
    with (
        patch(
            "deriva_ml_mcp.tools.dataset._split_dataset",
            side_effect=ValueError("invalid stratify column"),
        ),
        _patch_audit() as mock_audit,
    ):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_split_dataset"](
                hostname="h",
                catalog_id="1",
                source_dataset_rid="1-AAAA",
                stratify_by_column="bogus",
            )
        )
    assert out == {"error": "invalid stratify column"}
    failed = _success_calls(mock_audit, "deriva_ml_split_dataset_failed")
    assert failed
    assert failed[0].kwargs["source_dataset_rid"] == "1-AAAA"
    assert failed[0].kwargs["dry_run"] is False
