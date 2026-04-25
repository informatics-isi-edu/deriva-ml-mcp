"""Unit tests for dataset domain tools (Batch 1: read-only)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


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


def test_list_datasets_success(dataset_ctx, capturing_mcp, mock_ml):
    """Page mode returns a serialized list with truncated/next_after_rid."""
    mock_ml.find_datasets.return_value = [
        _make_dataset_mock("1-AAAA", "first", ["Training"], "1.0.0"),
        _make_dataset_mock("1-BBBB", "second", ["Testing"], "1.1.0"),
    ]
    out = json.loads(_run(capturing_mcp.tools["list_datasets"](hostname="h", catalog_id="1")))

    assert out["count"] == 2
    assert out["truncated"] is False
    assert out["next_after_rid"] is None
    assert {d["rid"] for d in out["datasets"]} == {"1-AAAA", "1-BBBB"}
    mock_ml.find_datasets.assert_called_once_with(deleted=False)


def test_list_datasets_preflight_returns_count_only(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.find_datasets.return_value = [_make_dataset_mock(f"1-{i:04d}") for i in range(5)]
    out = json.loads(
        _run(
            capturing_mcp.tools["list_datasets"](hostname="h", catalog_id="1", preflight_count=True)
        )
    )
    assert out["total_count"] == 5
    assert out["entities_fetched"] is False
    assert "action_required" in out


def test_list_datasets_pagination_caps_limit_and_advances_cursor(
    dataset_ctx, capturing_mcp, mock_ml
):
    """limit > 1000 caps to 1000; after_rid skips already-seen rows."""
    mock_ml.find_datasets.return_value = [_make_dataset_mock(f"1-{i:04d}") for i in range(2500)]

    out = json.loads(
        _run(capturing_mcp.tools["list_datasets"](hostname="h", catalog_id="1", limit=2000))
    )
    assert out["count"] == 1000  # capped
    assert out["truncated"] is True
    assert out["next_after_rid"] == "1-0999"

    # Advance cursor.
    out2 = json.loads(
        _run(
            capturing_mcp.tools["list_datasets"](
                hostname="h", catalog_id="1", after_rid="1-0999", limit=5
            )
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


def test_list_datasets_error_path(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.find_datasets.side_effect = RuntimeError("boom")
    out = json.loads(_run(capturing_mcp.tools["list_datasets"](hostname="h", catalog_id="1")))
    assert out == {"error": "boom"}


# ---------------------------------------------------------------------------
# get_dataset
# ---------------------------------------------------------------------------


def test_get_dataset_success(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", "desc", ["Training"], "1.0.0")
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        _run(capturing_mcp.tools["get_dataset"](hostname="h", catalog_id="1", dataset_rid="1-AAAA"))
    )
    assert out["rid"] == "1-AAAA"
    assert out["description"] == "desc"
    assert out["dataset_types"] == ["Training"]
    assert out["current_version"] == "1.0.0"
    assert out["chaise_url"] == "https://example.org/chaise"
    assert "history" not in out
    mock_ml.lookup_dataset.assert_called_once_with("1-AAAA")


def test_get_dataset_with_history(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    history_entry = MagicMock()
    history_entry.dataset_version = "1.0.0"
    history_entry.snapshot = "snap1"
    history_entry.description = "v1"
    history_entry.execution_rid = "EXEC-1"
    ds.dataset_history.return_value = [history_entry]
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        _run(
            capturing_mcp.tools["get_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                include_history=True,
            )
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


def test_get_dataset_not_found(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_dataset.side_effect = RuntimeError("Dataset not found")
    out = json.loads(
        _run(
            capturing_mcp.tools["get_dataset"](hostname="h", catalog_id="1", dataset_rid="missing")
        )
    )
    assert out == {"error": "Dataset not found"}


# ---------------------------------------------------------------------------
# list_dataset_members
# ---------------------------------------------------------------------------


def test_list_dataset_members_summary_mode(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.list_dataset_members.return_value = {
        "Image": [{"RID": "i1"}, {"RID": "i2"}, {"RID": "i3"}],
        "Subject": [{"RID": "s1"}],
    }
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        _run(
            capturing_mcp.tools["list_dataset_members"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA"
            )
        )
    )
    assert out["summary"] == {"Image": 3, "Subject": 1}
    assert out["total"] == 4
    assert sorted(out["tables"]) == ["Image", "Subject"]


def test_list_dataset_members_page_mode(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.list_dataset_members.return_value = {
        "Image": [{"RID": f"i-{i:03d}", "name": f"img{i}"} for i in range(5)],
    }
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        _run(
            capturing_mcp.tools["list_dataset_members"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                element_table="Image",
                limit=2,
            )
        )
    )
    assert out["element_table"] == "Image"
    assert out["returned_count"] == 2
    assert out["truncated"] is True
    assert out["next_after_rid"] == "i-001"
    assert [r["RID"] for r in out["rows"]] == ["i-000", "i-001"]


def test_list_dataset_members_preflight(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.list_dataset_members.return_value = {
        "Image": [{"RID": f"i-{i:03d}"} for i in range(7)],
    }
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        _run(
            capturing_mcp.tools["list_dataset_members"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                element_table="Image",
                preflight_count=True,
            )
        )
    )
    assert out["element_table"] == "Image"
    assert out["total_count"] == 7
    assert out["entities_fetched"] is False


def test_list_dataset_members_unknown_element_table(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.list_dataset_members.return_value = {"Image": []}
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        _run(
            capturing_mcp.tools["list_dataset_members"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                element_table="NotARealTable",
            )
        )
    )
    assert "error" in out
    assert "NotARealTable" in out["error"]


def test_list_dataset_members_error_path(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_dataset.side_effect = RuntimeError("nope")
    out = json.loads(
        _run(
            capturing_mcp.tools["list_dataset_members"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA"
            )
        )
    )
    assert out == {"error": "nope"}


# ---------------------------------------------------------------------------
# list_dataset_relations
# ---------------------------------------------------------------------------


def test_list_dataset_relations_both(dataset_ctx, capturing_mcp, mock_ml):
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
        _run(
            capturing_mcp.tools["list_dataset_relations"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA"
            )
        )
    )
    assert {p["rid"] for p in out["parents"]} == {"1-PARENT"}
    assert {c["rid"] for c in out["children"]} == {"1-CH1", "1-CH2"}


def test_list_dataset_relations_parents_only(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.list_dataset_parents.return_value = [_make_dataset_mock("1-PARENT")]
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        _run(
            capturing_mcp.tools["list_dataset_relations"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                direction="parents",
            )
        )
    )
    assert "parents" in out
    assert "children" not in out
    ds.list_dataset_children.assert_not_called()


def test_list_dataset_relations_error(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_dataset.side_effect = RuntimeError("relations boom")
    out = json.loads(
        _run(
            capturing_mcp.tools["list_dataset_relations"](
                hostname="h", catalog_id="1", dataset_rid="x"
            )
        )
    )
    assert out == {"error": "relations boom"}


# ---------------------------------------------------------------------------
# list_dataset_element_types
# ---------------------------------------------------------------------------


def test_list_dataset_element_types_success(dataset_ctx, capturing_mcp, mock_ml):
    table_a = MagicMock()
    table_a.name = "Image"
    table_a.schema.name = "domain"
    table_b = MagicMock()
    table_b.name = "Subject"
    table_b.schema.name = "domain"
    mock_ml.list_dataset_element_types.return_value = [table_a, table_b]

    out = json.loads(
        _run(capturing_mcp.tools["list_dataset_element_types"](hostname="h", catalog_id="1"))
    )
    assert out["count"] == 2
    assert out["element_types"] == [
        {"name": "Image", "schema": "domain"},
        {"name": "Subject", "schema": "domain"},
    ]


def test_list_dataset_element_types_error(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.list_dataset_element_types.side_effect = RuntimeError("bad")
    out = json.loads(
        _run(capturing_mcp.tools["list_dataset_element_types"](hostname="h", catalog_id="1"))
    )
    assert out == {"error": "bad"}


# ---------------------------------------------------------------------------
# bag_info
# ---------------------------------------------------------------------------


def test_bag_info_success(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.bag_info.return_value = {
        "tables": {"Image": {"row_count": 100, "is_asset": True, "asset_bytes": 12345}},
        "total_rows": 100,
        "total_asset_bytes": 12345,
        "total_asset_size": "12 KB",
        "cache_status": "not_cached",
        "cache_path": None,
    }

    out = json.loads(
        _run(
            capturing_mcp.tools["bag_info"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                version="1.0.0",
            )
        )
    )
    assert out["total_rows"] == 100
    assert out["cache_status"] == "not_cached"
    # Verify spec construction.
    spec_arg = mock_ml.bag_info.call_args.args[0]
    assert spec_arg.rid == "1-AAAA"
    assert str(spec_arg.version) == "1.0.0"
    assert spec_arg.exclude_tables is None


def test_bag_info_with_exclude_tables(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.bag_info.return_value = {"tables": {}, "total_rows": 0}

    _run(
        capturing_mcp.tools["bag_info"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            version="1.0.0",
            exclude_tables=["Big_Asset"],
        )
    )
    spec_arg = mock_ml.bag_info.call_args.args[0]
    assert spec_arg.exclude_tables == {"Big_Asset"}


def test_bag_info_error(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.bag_info.side_effect = RuntimeError("cant compute")
    out = json.loads(
        _run(
            capturing_mcp.tools["bag_info"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                version="1.0.0",
            )
        )
    )
    assert out == {"error": "cant compute"}


# ---------------------------------------------------------------------------
# get_dataset_spec
# ---------------------------------------------------------------------------


def test_get_dataset_spec_with_explicit_version(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", "desc", ["Training"], "2.0.0")
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        _run(
            capturing_mcp.tools["get_dataset_spec"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                version="1.5.0",
            )
        )
    )
    assert out["spec"] == 'DatasetSpecConfig(rid="1-AAAA", version="1.5.0")'
    assert out["version"] == "1.5.0"
    assert out["dataset_rid"] == "1-AAAA"
    assert out["description"] == "desc"
    assert out["dataset_types"] == ["Training"]
    assert out["warning"] is None


def test_get_dataset_spec_falls_back_to_current_version(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", "desc", [], "2.0.0")
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        _run(
            capturing_mcp.tools["get_dataset_spec"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA"
            )
        )
    )
    assert out["version"] == "2.0.0"
    assert out["spec"] == 'DatasetSpecConfig(rid="1-AAAA", version="2.0.0")'
    assert out["warning"] is not None
    assert "current version" in out["warning"]


def test_get_dataset_spec_error(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_dataset.side_effect = RuntimeError("missing")
    out = json.loads(
        _run(
            capturing_mcp.tools["get_dataset_spec"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA"
            )
        )
    )
    assert out == {"error": "missing"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Synchronously drive a coroutine to completion for unit tests."""
    import asyncio

    return asyncio.run(coro)
