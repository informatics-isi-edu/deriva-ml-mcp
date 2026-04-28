"""Unit tests for complex dataset tools (mirrors ``tools/dataset/complex.py``).

Covers ``deriva_ml_cache_dataset``, ``deriva_ml_denormalize_dataset``,
and ``deriva_ml_split_dataset`` -- the three dataset tools whose bodies
deserve their own file space (each is substantial enough that grouping
them with the simple-mutate tools made the source file uncomfortably
long; same rationale applies to keeping their tests in their own
file).

The split mirrors the source-side ``tools/dataset/{read,mutate,complex}.py``
package. Shared fixtures (``_make_dataset_mock``, ``_patch_audit``)
live in ``tests/_dataset_helpers.py``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests._dataset_helpers import _make_dataset_mock, _patch_audit
from tests._helpers import _success_calls


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
    assert out["status"] == "cached"
    assert out["dataset_rid"] == "1-AAAA"
    assert out["version"] == "1.2.3"
    assert out["materialize"] is True
    assert out["bag_info"]["cache_status"] == "cached_materialized"
    assert out["bag_info"]["cache_path"] == "/tmp/cache/1-AAAA"

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
    assert out["status"] == "split"
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
