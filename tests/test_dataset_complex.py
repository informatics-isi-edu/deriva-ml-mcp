"""Unit tests for complex dataset tools (mirrors ``tools/dataset/complex.py``).

Covers ``deriva_ml_cache_dataset`` and ``deriva_ml_denormalize_dataset``
-- the two dataset tools whose bodies deserve their own file space
(each is substantial enough that grouping them with the simple-mutate
tools made the source file uncomfortably long; same rationale applies
to keeping their tests in their own file).

The split mirrors the source-side ``tools/dataset/{read,mutate,complex}.py``
package. Shared fixtures (``_make_dataset_mock``, ``_patch_audit``)
live in ``tests/_dataset_helpers.py``.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tests._dataset_helpers import _make_dataset_mock


@pytest.fixture()
def dataset_ctx(ctx, mock_ml):
    """Register dataset tools with mock_ml as the DerivaML stand-in.

    Patches at the use-site (``deriva_ml_mcp.tools.dataset.get_ml``) and
    imports the tool module *inside* the patch block so registration
    sees the mock.
    """
    with patch("deriva_ml_mcp.tools.dataset.get_ml", return_value=mock_ml):
        from deriva_ml_mcp.tools import dataset as dataset_module

        dataset_module.register(ctx)
        yield ctx

# NOTE: cache_dataset tests were removed in v5.0.0 along with the
# deriva_ml_cache_dataset tool. The denormalize_dataset tests below
# transitively exercise cache_dataset via the library call inside
# get_denormalized_as_dict, but the wire-level tool is gone.
# See docs/audit-2026-05-23.md and the design note at
# docs/superpowers/notes/2026-05-23-cache-denormalize-deprecation-design.md.



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

