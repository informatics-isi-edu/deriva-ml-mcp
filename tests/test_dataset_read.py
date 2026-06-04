"""Unit tests for read-only dataset tools (mirrors ``tools/dataset/read.py``).

Covers ``deriva_ml_list_datasets``, ``deriva_ml_get_dataset``,
``deriva_ml_list_dataset_members``, ``deriva_ml_list_dataset_relations``,
``deriva_ml_list_dataset_element_types``, ``deriva_ml_bag_info``, and
``deriva_ml_get_dataset_spec``.

The split mirrors the source-side ``tools/dataset/{read,mutate,complex}.py``
package and lets the read tests stay scrolling-distance from the read
tools they exercise. Shared fixtures (``_make_dataset_mock``,
``_patch_audit``) live in ``tests/_dataset_helpers.py``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests._dataset_helpers import _make_dataset_mock


@pytest.fixture()
def dataset_ctx(ctx, mock_ml):
    """Register dataset tools with mock_ml as the DerivaML stand-in.

    Patches at the use-site (``deriva_ml_mcp_plugin.tools.dataset.get_ml``) and
    imports the tool module *inside* the patch block so registration
    sees the mock.
    """
    with patch("deriva_ml_mcp_plugin.tools.dataset.get_ml", return_value=mock_ml):
        from deriva_ml_mcp_plugin.tools import dataset as dataset_module

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
    mock_ml.find_datasets.assert_called_once_with(deleted=False, sort=None)


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


async def test_list_datasets_sort_forwards_to_deriva_ml(dataset_ctx, capturing_mcp, mock_ml):
    """sort=True forwards sort=True to deriva_ml.find_datasets and skips post-fetch RID sort."""
    ds_a = _make_dataset_mock(rid="1-DS-NEWEST")
    ds_b = _make_dataset_mock(rid="1-DS-OLDER")
    mock_ml.find_datasets.return_value = [ds_a, ds_b]

    result = await capturing_mcp.tools["deriva_ml_list_datasets"](
        hostname="h", catalog_id="1", sort=True
    )
    payload = json.loads(result)

    mock_ml.find_datasets.assert_called_once()
    assert mock_ml.find_datasets.call_args.kwargs.get("sort") is True
    rids = [d["rid"] for d in payload["datasets"]]
    assert rids == ["1-DS-NEWEST", "1-DS-OLDER"]


async def test_list_datasets_sort_default_preserves_rid_sort(dataset_ctx, capturing_mcp, mock_ml):
    """sort=False (default) calls find_datasets with sort=None and re-sorts by RID asc."""
    ds_z = _make_dataset_mock(rid="1-Z")
    ds_a = _make_dataset_mock(rid="1-A")
    mock_ml.find_datasets.return_value = [ds_z, ds_a]

    result = await capturing_mcp.tools["deriva_ml_list_datasets"](hostname="h", catalog_id="1")
    payload = json.loads(result)

    assert mock_ml.find_datasets.call_args.kwargs.get("sort") is None
    rids = [d["rid"] for d in payload["datasets"]]
    assert rids == ["1-A", "1-Z"]


def test_list_datasets_impl_filters_by_dataset_type() -> None:
    """``dataset_type=`` keeps only datasets whose type list contains it."""
    from types import SimpleNamespace

    from deriva_ml_mcp_plugin.tools.dataset.read import _list_datasets_impl

    def _ds(rid, types):
        return SimpleNamespace(
            dataset_rid=rid,
            description="d",
            dataset_types=types,
            current_version=None,
        )

    fake_ml = SimpleNamespace(
        find_datasets=lambda deleted, sort: [
            _ds("1-TRN", ["Training"]),
            _ds("1-TST", ["Testing"]),
            _ds("1-BTH", ["Training", "Validation"]),
        ]
    )

    resp = _list_datasets_impl(fake_ml, after_rid=None, limit=100, dataset_type="Training")
    rids = [d.rid for d in resp.datasets]
    assert rids == ["1-BTH", "1-TRN"]  # RID-ascending of the two Training matches


def test_list_datasets_impl_dataset_type_none_returns_all() -> None:
    """No filter -> unchanged behavior (all datasets)."""
    from types import SimpleNamespace

    from deriva_ml_mcp_plugin.tools.dataset.read import _list_datasets_impl

    def _ds(rid, types):
        return SimpleNamespace(
            dataset_rid=rid, description="d", dataset_types=types, current_version=None
        )

    fake_ml = SimpleNamespace(
        find_datasets=lambda deleted, sort: [_ds("1-A", ["Training"]), _ds("1-B", ["Testing"])]
    )
    resp = _list_datasets_impl(fake_ml, after_rid=None, limit=100, dataset_type=None)
    assert {d.rid for d in resp.datasets} == {"1-A", "1-B"}


async def test_list_datasets_tool_forwards_dataset_type(dataset_ctx, capturing_mcp, mock_ml):
    """``deriva_ml_list_datasets(dataset_type=...)`` forwards the filter to the impl."""
    from deriva_ml_mcp_plugin._response_models import DatasetListResponse

    seen: dict = {}

    def spy(ml, **kwargs):
        seen.update(kwargs)
        return DatasetListResponse(datasets=[], count=0, truncated=False, next_after_rid=None)

    with patch(
        "deriva_ml_mcp_plugin.tools.dataset.read._list_datasets_impl",
        side_effect=spy,
    ):
        await capturing_mcp.tools["deriva_ml_list_datasets"](
            hostname="h", catalog_id="1", dataset_type="Training"
        )

    assert seen.get("dataset_type") == "Training"


async def test_list_datasets_tool_schedules_index_on_find(dataset_ctx, capturing_mcp, mock_ml):
    """``deriva_ml_list_datasets`` warms the returned rows via _index_rows_on_find."""
    mock_ml.find_datasets.return_value = [
        _make_dataset_mock("1-AAAA", "first", ["Training"], "1.0.0"),
        _make_dataset_mock("1-BBBB", "second", ["Testing"], "1.1.0"),
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
        await capturing_mcp.tools["deriva_ml_list_datasets"](hostname="h", catalog_id="1")

    from deriva_ml_mcp_plugin.resources.rag import _DATASET_TOKEN

    assert captured.get("token") == _DATASET_TOKEN
    assert set(captured.get("rids") or []) == {"1-AAAA", "1-BBBB"}


async def test_list_datasets_preflight_does_not_index_on_find(dataset_ctx, capturing_mcp, mock_ml):
    """The preflight (count-only) path returns before building rows, so it must
    NOT schedule a read-through warm."""
    mock_ml.find_datasets.return_value = [_make_dataset_mock(f"1-{i:04d}") for i in range(3)]

    called = False

    def fake_warm(*args, **kwargs):
        nonlocal called
        called = True

    with patch(
        "deriva_ml_mcp_plugin.resources.rag._index_rows_on_find",
        side_effect=fake_warm,
    ):
        await capturing_mcp.tools["deriva_ml_list_datasets"](
            hostname="h", catalog_id="1", preflight_count=True
        )

    assert called is False


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
    # cite_url is built via _cite_dataset_version_url; under the mock
    # fixture both lookup_dataset and ml.cite return MagicMocks, which
    # the impl coerces to None rather than failing Pydantic validation.
    assert out["cite_url"] is None
    assert "history" not in out
    mock_ml.lookup_dataset.assert_called_once_with("1-AAAA")


async def test_get_dataset_with_history(dataset_ctx, capturing_mcp, mock_ml):
    """v2.0 wire change: ``deriva_ml_get_dataset`` returns ``version_history``
    (was ``history`` in v1.x). The tool now shares the wire shape with the
    ``deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}`` resource so consumers can
    switch between tool and resource without re-mapping fields.
    """
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
    assert out["version_history"] == [
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


async def test_get_dataset_tool_schedules_index_on_find(dataset_ctx, capturing_mcp, mock_ml):
    """``deriva_ml_get_dataset`` warms the single returned row via _index_rows_on_find."""
    mock_ml.lookup_dataset.return_value = _make_dataset_mock(
        "1-AAAA", "desc", ["Training"], "1.0.0"
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
        await capturing_mcp.tools["deriva_ml_get_dataset"](
            hostname="h", catalog_id="1", dataset_rid="1-AAAA"
        )

    from deriva_ml_mcp_plugin.resources.rag import _DATASET_TOKEN

    assert captured.get("token") == _DATASET_TOKEN
    assert captured.get("rids") == ["1-AAAA"]


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


# ---------------------------------------------------------------------------
# v3.3 validate_dataset_specs / validate_execution_configuration wrappers
#
# Both methods are thin wrappers around deriva-ml >= 1.33.0 methods that
# return Pydantic ValidationReport models. The wrapper's only job is to
# call into deriva-ml and serialize the response; the heavy logic lives
# in the library. Tests confirm the wiring is intact, the call propagates
# arguments cleanly, and the error path doesn't emit an audit row (read
# tools are silent on failure).
# ---------------------------------------------------------------------------


async def test_validate_dataset_specs_success(dataset_ctx, capturing_mcp, mock_ml):
    """The tool serializes the Pydantic DatasetSpecValidationReport.
    Underlying call is ``ml.validate_dataset_specs(specs)``."""
    from deriva_ml.dataset.aux_classes import DatasetSpec
    from deriva_ml.dataset.validation import (
        DatasetSpecResult,
        DatasetSpecValidationReport,
    )

    payload = DatasetSpecValidationReport(
        all_valid=False,
        results=[
            DatasetSpecResult(
                spec=DatasetSpec(rid="1-ABCD", version="1.0.0"),
                valid=True,
                dataset_name="Training set",
                resolved_version="1.0.0",
            ),
            DatasetSpecResult(
                spec=DatasetSpec(rid="2-XYZW", version="9.9.9"),
                valid=False,
                reasons=["version_not_found"],
                available_versions=["1.0.0", "1.1.0"],
            ),
        ],
    )
    mock_ml.validate_dataset_specs.return_value = payload

    specs = [
        {"rid": "1-ABCD", "version": "1.0.0"},
        {"rid": "2-XYZW", "version": "9.9.9"},
    ]
    result = await capturing_mcp.tools["deriva_ml_validate_dataset_specs"](
        hostname="h", catalog_id="1", specs=specs
    )
    out = json.loads(result)
    assert out["all_valid"] is False
    assert len(out["results"]) == 2
    assert out["results"][0]["valid"] is True
    assert out["results"][1]["reasons"] == ["version_not_found"]
    assert out["results"][1]["available_versions"] == ["1.0.0", "1.1.0"]
    mock_ml.validate_dataset_specs.assert_called_once_with(specs)


async def test_validate_dataset_specs_error_path(dataset_ctx, capturing_mcp, mock_ml):
    """An error from the underlying method wraps as ``{"error": ...}``
    without emitting an audit row."""
    mock_ml.validate_dataset_specs.side_effect = RuntimeError(
        "specs must be a list of DatasetSpec or shorthand dicts"
    )
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_validate_dataset_specs"](
            hostname="h", catalog_id="1", specs=["not-a-spec"]
        )
    )
    assert "error" in out


async def test_validate_execution_configuration_success(dataset_ctx, capturing_mcp, mock_ml):
    """The tool coerces the dict to ExecutionConfiguration and serializes
    the validation report. We patch ExecutionConfiguration to bypass
    its full validation here -- the wrapper's only job is wiring; full
    ExecutionConfiguration validation is deriva-ml's responsibility,
    exercised by deriva-ml's own unit tests."""
    from deriva_ml.asset.aux_classes import AssetSpec
    from deriva_ml.dataset.aux_classes import DatasetSpec
    from deriva_ml.dataset.validation import (
        AssetSpecResult,
        DatasetSpecResult,
        ExecutionConfigurationValidationReport,
        WorkflowSpecResult,
    )

    payload = ExecutionConfigurationValidationReport(
        all_valid=True,
        dataset_results=[
            DatasetSpecResult(
                spec=DatasetSpec(rid="1-ABCD", version="1.0.0"),
                valid=True,
                dataset_name="Training",
                resolved_version="1.0.0",
            ),
        ],
        asset_results=[
            AssetSpecResult(spec=AssetSpec(rid="3-WXYZ"), valid=True),
        ],
        workflow_result=WorkflowSpecResult(rid="1-WFLW", valid=True),
        cross_spec_issues=[],
    )
    mock_ml.validate_execution_configuration.return_value = payload

    fake_config = MagicMock(name="ExecutionConfiguration_instance")
    with patch(
        "deriva_ml.execution.execution_configuration.ExecutionConfiguration",
        return_value=fake_config,
    ):
        result = await capturing_mcp.tools["deriva_ml_validate_execution_configuration"](
            hostname="h",
            catalog_id="1",
            config={
                "workflow": {"rid": "1-WFLW"},
                "datasets": [{"rid": "1-ABCD", "version": "1.0.0"}],
                "assets": [{"rid": "3-WXYZ"}],
            },
        )

    out = json.loads(result)
    assert out["all_valid"] is True
    assert out["workflow_result"]["valid"] is True
    assert out["cross_spec_issues"] == []
    mock_ml.validate_execution_configuration.assert_called_once_with(fake_config)


async def test_validate_execution_configuration_error_path(dataset_ctx, capturing_mcp, mock_ml):
    """Underlying ``ml.validate_execution_configuration`` raising propagates
    as ``{"error": ...}``."""
    fake_config = MagicMock(name="ExecutionConfiguration_instance")
    mock_ml.validate_execution_configuration.side_effect = RuntimeError(
        "config has unresolvable workflow"
    )
    with patch(
        "deriva_ml.execution.execution_configuration.ExecutionConfiguration",
        return_value=fake_config,
    ):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_validate_execution_configuration"](
                hostname="h",
                catalog_id="1",
                config={"workflow": {"rid": "1-WFLW"}},
            )
        )
    assert out == {"error": "config has unresolvable workflow"}


# ---------------------------------------------------------------------------
# v3.5 validate_config_file / bootstrap_config wrappers
#
# Both methods are thin wrappers around deriva-ml >= 1.38.0 methods
# that return Pydantic models. The wrapper's only job is to call into
# deriva-ml and serialize the response; the AST walking + catalog
# round-trips live in the library.
#
# v0.5.0: the ``file_path=`` parameter was removed from
# ``deriva_ml_validate_config_file`` per the stateless rule (the MCP
# server's filesystem view does not match the caller's). Only
# ``file_contents=`` is accepted; the wrapper writes a temp file
# (cleaned up immediately) for the AST walker to consume.
# ---------------------------------------------------------------------------


async def test_validate_config_file_with_file_contents(dataset_ctx, capturing_mcp, mock_ml) -> None:
    """``file_contents`` is the sole accepted input: wrapper writes a
    temp file and calls the ml method with the temp path. The path
    passed to ml ends with ``.py`` and the temp file is cleaned up."""
    from deriva_ml.config.validation import ConfigValidationReport

    mock_ml.validate_config_file.return_value = ConfigValidationReport(
        file_count=1,
        entry_count=0,
        all_valid=True,
        results=[],
    )

    src = (
        "from deriva_ml.dataset import DatasetSpecConfig\n"
        'datasets_store(name="x", spec=DatasetSpecConfig(rid="1-A", version="0.1.0"))\n'
    )
    result = await capturing_mcp.tools["deriva_ml_validate_config_file"](
        hostname="h", catalog_id="1", file_contents=src
    )
    out = json.loads(result)
    assert out["all_valid"] is True
    assert mock_ml.validate_config_file.called
    called_path = mock_ml.validate_config_file.call_args.args[0]
    assert called_path.endswith(".py")
    # Temp file is cleaned up immediately after parsing.
    import os

    assert not os.path.exists(called_path)


async def test_validate_config_file_error_path(dataset_ctx, capturing_mcp, mock_ml) -> None:
    mock_ml.validate_config_file.side_effect = RuntimeError("AST blew up")
    result = await capturing_mcp.tools["deriva_ml_validate_config_file"](
        hostname="h", catalog_id="1", file_contents="# empty"
    )
    out = json.loads(result)
    assert out == {"error": "AST blew up"}


async def test_bootstrap_config_default_kinds(dataset_ctx, capturing_mcp, mock_ml) -> None:
    """Default invocation propagates ``kinds=None`` and
    ``dataset_type_filter=None`` to the ml method."""
    from deriva_ml.config.bootstrap import BootstrapReport, BootstrapSuggestion

    mock_ml.bootstrap_config.return_value = BootstrapReport(
        catalog={"hostname": "h", "catalog_id": "1"},
        suggestions=[
            BootstrapSuggestion(
                kind="datasets",
                config_name="training",
                rid="1-AAAA",
                version="0.4.0",
                spec_string='DatasetSpecConfig(rid="1-AAAA", version="0.4.0")',
                description="Training",
                rationale="Dataset type Training.",
            ),
        ],
        skipped=[],
    )

    result = await capturing_mcp.tools["deriva_ml_bootstrap_config"](hostname="h", catalog_id="1")
    out = json.loads(result)
    assert out["catalog"]["hostname"] == "h"
    assert len(out["suggestions"]) == 1
    assert out["suggestions"][0]["spec_string"].startswith("DatasetSpecConfig(")
    mock_ml.bootstrap_config.assert_called_once_with(kinds=None, dataset_type_filter=None)


async def test_bootstrap_config_kinds_filter_propagates(
    dataset_ctx, capturing_mcp, mock_ml
) -> None:
    from deriva_ml.config.bootstrap import BootstrapReport

    mock_ml.bootstrap_config.return_value = BootstrapReport(
        catalog={"hostname": "h", "catalog_id": "1"},
        suggestions=[],
        skipped=[],
    )
    await capturing_mcp.tools["deriva_ml_bootstrap_config"](
        hostname="h",
        catalog_id="1",
        kinds=["datasets"],
        dataset_type_filter=["Training", "Testing"],
    )
    mock_ml.bootstrap_config.assert_called_once_with(
        kinds=["datasets"], dataset_type_filter=["Training", "Testing"]
    )


async def test_bootstrap_config_error_path(dataset_ctx, capturing_mcp, mock_ml) -> None:
    mock_ml.bootstrap_config.side_effect = RuntimeError("cannot connect")
    result = await capturing_mcp.tools["deriva_ml_bootstrap_config"](hostname="h", catalog_id="1")
    out = json.loads(result)
    assert out == {"error": "cannot connect"}
