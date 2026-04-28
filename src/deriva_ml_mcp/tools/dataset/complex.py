"""Complex dataset tools.

This submodule houses the three substantial dataset tools that warrant
their own breathing room:

- ``deriva_ml_cache_dataset`` (~90 lines): warm the local DerivaML
  bag cache (``mutates=True`` because it touches the local filesystem,
  but no catalog state is changed).
- ``deriva_ml_denormalize_dataset`` (~220 lines, the biggest tool in
  the dataset domain): the catalog-shape vs dataset-described describe
  branch with a paged row preview and the bounded-materialization
  guard (the ``preflight_required`` short-circuit when estimated rows
  vastly exceed the requested limit).
- ``deriva_ml_split_dataset`` (~140 lines): sklearn-style train / test
  / (val) split that creates a parent split dataset and 2-3 child
  datasets in the catalog (``mutates=True``), with a ``dry_run`` path
  that intentionally skips the audit event.

Audit event lookup goes through ``_pkg.audit_event(...)`` (attribute
lookup on the parent package) for the same one-patch-site reason as
``mutate.py`` -- see that module's header note for the rationale.
"""

from __future__ import annotations

import itertools
import json
import logging
from typing import TYPE_CHECKING, Any

from deriva_mcp_core import deriva_call
from deriva_ml.dataset.aux_classes import DatasetSpec

logger = logging.getLogger(__name__)

# See the header note in ``mutate.py`` for why ``audit_event``,
# ``get_ml``, and ``_split_dataset`` are accessed via attribute lookup
# on the parent package (``_pkg.<name>``) rather than direct
# ``from ... import``: a single ``patch("deriva_ml_mcp.tools.dataset.<name>")``
# must redirect every call across read / mutate / complex submodules.
import deriva_ml_mcp.tools.dataset as _pkg  # noqa: E402  (intentional cycle)
from deriva_ml_mcp._helpers import (
    _MAX_LIMIT,
    _error_envelope,
    _paginate,
    _row_rid_for,
)
from deriva_ml_mcp._response_models import (
    CacheDatasetBagInfo,
    CacheDatasetResponse,
    PreflightCountResponse,
    SplitDatasetResponse,
)

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def register(ctx: PluginContext) -> None:
    """Register the three complex dataset tools with the plugin context.

    Args:
        ctx: PluginContext supplied by deriva-mcp-core at startup.

    Returns:
        None.

    Example:
        >>> from deriva_mcp_core.plugin.api import PluginContext
        >>> # ctx provided by the framework
        >>> register(ctx)  # doctest: +SKIP
    """

    @ctx.tool(mutates=True)
    async def deriva_ml_cache_dataset(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        version: str | None = None,
        materialize: bool = True,
        exclude_tables: list[str] | None = None,
    ) -> str:
        """Warm the local cache for a dataset bag before running an experiment.

        Downloads the bag's metadata (and assets, if ``materialize=True``)
        to the local DerivaML cache. Subsequent calls hit the cache without
        re-downloading. ``mutates=True`` because it touches the local
        filesystem — but no catalog state is mutated.

        Args:
            dataset_rid: RID of the dataset to cache.
            version: Specific version (semver string). If None, looks up the
                dataset's current_version automatically.
            materialize: If True (default), download asset files. If False,
                only fetch the bag metadata.
            exclude_tables: Tables to omit from the bag (e.g. large blob
                tables you don't need).

        Returns:
            JSON string ``{"status": "cached", "dataset_rid",
            "version", "materialize", "bag_info": {...upstream
            DerivaML keys...}}``. v3.0 changes: status renamed from
            ``"success"`` to ``"cached"``; bag-info keys nested under
            ``bag_info`` instead of spread top-level.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.cache_dataset``.

        Example:
            ``{"status": "cached", "dataset_rid": "1-AAAA", "version":
            "1.0.0", "materialize": true, "bag_info": {"cache_status":
            "cached_materialized", "cache_path": "/path/to/bag", ...}}``.
        """
        try:
            with deriva_call():
                ml = _pkg.get_ml(hostname, catalog_id)
                if version is None:
                    ds = ml.lookup_dataset(dataset_rid)
                    used_version = (
                        str(ds.current_version) if ds.current_version is not None else None
                    )
                else:
                    used_version = version
                spec = DatasetSpec(
                    rid=dataset_rid,
                    version=used_version,
                    exclude_tables=set(exclude_tables) if exclude_tables else None,
                )
                info = ml.cache_dataset(spec, materialize=materialize)
            _pkg.audit_event(
                "deriva_ml_cache_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                version=used_version,
                materialize=materialize,
                exclude_tables=exclude_tables or [],
            )
            return CacheDatasetResponse(
                status="cached",
                dataset_rid=dataset_rid,
                version=used_version,
                materialize=materialize,
                bag_info=CacheDatasetBagInfo(**info),
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="cache_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                materialize=materialize,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_denormalize_dataset(
        hostname: str,
        catalog_id: str,
        include_tables: list[str],
        dataset_rid: str | None = None,
        version: str | None = None,
        row_per: str | None = None,
        via: list[str] | None = None,
        limit: int = 0,
        after_rid: str | None = None,
        preflight_count: bool = False,
    ) -> str:
        """Preview a wide-table view (catalog-wide or dataset-scoped).

        TWO MODES:

        - Catalog-shape (``dataset_rid=None``): returns a size estimate for
          the denormalized join across ``include_tables``. No rows. Use to
          scope a download before committing.
        - Dataset-described (``dataset_rid`` set): describes the
          denormalized view for one dataset. With ``limit > 0``, also
          returns up to ``limit`` rows (cursor-paged via ``after_rid`` /
          ``preflight_count``). With ``limit == 0`` (default), returns
          shape only.

        PAGINATION (dataset-row mode): When the row count is unknown, call
        with ``preflight_count=True`` first to get the planner's estimated
        row count from the describe plan. Present that to the user, choose
        a ``limit``, then call again with ``preflight_count=False``. Use
        ``after_rid`` (the RID of the last row from the previous page) to
        advance the cursor.

        Note: ``get_denormalized_as_dict`` materializes the full join
        before paging — for very large datasets, prefer downloading the
        bag and querying it locally rather than paging through this tool.

        Args:
            include_tables: Tables to include in the denormalized join.
                REQUIRED in both modes.
            dataset_rid: If set, scope to one dataset. If None, catalog-wide
                shape mode.
            version: Optional dataset version (dataset mode only).
            row_per: Anchor table for the join (advanced; usually None).
            via: FK path hints (advanced; usually None).
            limit: If 0 (default), shape only. If > 0, also return up to
                ``limit`` rows (capped at 1000). Dataset mode only.
            after_rid: Cursor for paging row results. Dataset mode only.
            preflight_count: If True, return only the planner's row-count
                estimate. Dataset mode only.

        Returns:
            Catalog-shape: ``{"mode": "catalog_shape",
            "include_tables", "columns", "join_path", "tables",
            "total_rows", "total_asset_bytes", "total_asset_size"}``.
            Dataset shape only: ``{"mode": "dataset_shape", "dataset_rid",
            "version", "columns", "join_path", "row_per",
            "estimated_row_count", ...}``. Dataset with rows: same as shape
            plus ``"rows", "returned_count", "truncated", "next_after_rid"``.
            Dataset preflight: ``{"mode": "dataset_preflight", "dataset_rid",
            "total_count", "entities_fetched": False, "action_required"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.estimate_denormalized_size`` or
                ``deriva_ml.dataset.dataset.Dataset.describe_denormalized``
                / ``get_denormalized_as_dict``.

        Example:
            Catalog-shape: ``{"mode": "catalog_shape", "include_tables":
            ["Image"], "columns": [["Image.RID", "text"]], "join_path":
            ["Image"], "total_rows": 1000, ...}``.
        """
        if not include_tables:
            return json.dumps({"error": "include_tables is required and must be non-empty"})
        try:
            if dataset_rid is None:
                with deriva_call():
                    ml = _pkg.get_ml(hostname, catalog_id)
                    estimate = ml.estimate_denormalized_size(include_tables)
                return json.dumps(
                    {
                        "mode": "catalog_shape",
                        "include_tables": include_tables,
                        **estimate,
                    },
                    default=str,
                )

            # Dataset mode.
            capped = min(max(limit, 0), _MAX_LIMIT)
            with deriva_call():
                ml = _pkg.get_ml(hostname, catalog_id)
                ds = ml.lookup_dataset(dataset_rid)
                desc = ds.describe_denormalized(
                    include_tables,
                    row_per=row_per,
                    via=via,
                    version=version,
                )
                # Resolve effective row_per from desc when caller didn't supply
                # one, so cursor pagination uses the unambiguous
                # f"{row_per}.RID" column (M-1 fix from the Batch 3 review).
                row_per_effective: str | None = row_per or desc.get("row_per")
                # Bounded materialization: if the caller asked for rows
                # without preflighting, refuse to drain the generator
                # when the estimated row count is wildly larger than the
                # requested limit. Forces the caller back through
                # preflight_count=True before fetching pages of an
                # otherwise-OOM-able join (I-1 fix from the Batch 3
                # review). Threshold: 10x the requested limit.
                rows: list[dict[str, Any]] | None
                if capped > 0 and not preflight_count:
                    estimated = desc.get("estimated_row_count", {}).get("total")
                    if estimated is not None and estimated > 10 * capped:
                        return json.dumps(
                            {
                                "mode": "dataset_preflight_required",
                                "dataset_rid": dataset_rid,
                                "estimated_row_count": estimated,
                                "requested_limit": capped,
                                "entities_fetched": False,
                                "action_required": (
                                    f"Estimated {estimated} rows is more than "
                                    f"10x the requested limit ({capped}). "
                                    "Call again with preflight_count=True to "
                                    "confirm the count, then choose a larger "
                                    "limit or accept the cost before retrying."
                                ),
                            },
                            default=str,
                        )
                    # itertools.islice puts a hard upper bound on
                    # generator materialization; we read at most
                    # (capped + 1) rows just to learn whether a next
                    # page exists. Note: this means after_rid filters
                    # on the FIRST capped+1 yielded rows, not the
                    # global sorted set. For a sorted-by-RID generator
                    # this is correct; for unsorted output the caller
                    # gets a sliced page that may not match a strict
                    # global ordering. DerivaML's denormalizer yields
                    # in row_per order which is RID-stable in practice.
                    rows = list(
                        itertools.islice(
                            ds.get_denormalized_as_dict(
                                include_tables,
                                row_per=row_per,
                                via=via,
                                version=version,
                            ),
                            capped + 1,
                        )
                    )
                else:
                    rows = None

            if preflight_count:
                total = desc.get("estimated_row_count", {}).get("total")
                action: str
                if total is None:
                    action = (
                        "describe_denormalized returned no row-count estimate; "
                        "cannot suggest a safe limit. Try a small limit (e.g. 10) "
                        "to start."
                    )
                else:
                    action = (
                        f"Estimated {total} rows for the denormalized view. "
                        "Choose a limit and call again with preflight_count=False."
                    )
                return PreflightCountResponse(
                    total_count=total,
                    action_required=action,
                    mode="dataset_preflight",
                    dataset_rid=dataset_rid,
                ).model_dump_json(by_alias=True)

            if rows is None:
                return json.dumps(
                    {
                        "mode": "dataset_shape",
                        "dataset_rid": dataset_rid,
                        "version": version,
                        **desc,
                    },
                    default=str,
                )

            # Sort + paginate rows by their RID-bearing column.
            row_rid = _row_rid_for(row_per_effective)
            sorted_rows = sorted(rows, key=row_rid)
            page, truncated, next_after = _paginate(
                sorted_rows, after_rid=after_rid, limit=capped, key=row_rid
            )
            return json.dumps(
                {
                    "mode": "dataset_rows",
                    "dataset_rid": dataset_rid,
                    "version": version,
                    **desc,
                    "rows": page,
                    "returned_count": len(page),
                    "truncated": truncated,
                    "next_after_rid": next_after,
                },
                default=str,
            )
        except Exception as exc:
            # Read-only tool: log+return without an audit row (I-2 fix).
            return _error_envelope(
                exc,
                operation="denormalize_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_split_dataset(
        hostname: str,
        catalog_id: str,
        source_dataset_rid: str,
        test_size: float = 0.2,
        train_size: float | None = None,
        val_size: float | None = None,
        seed: int = 42,
        shuffle: bool = True,
        stratify_by_column: str | None = None,
        stratify_missing: str = "error",
        element_table: str | None = None,
        include_tables: list[str] | None = None,
        training_types: list[str] | None = None,
        testing_types: list[str] | None = None,
        validation_types: list[str] | None = None,
        split_description: str = "",
        workflow_type: str = "Dataset_Split",
        dry_run: bool = False,
    ) -> str:
        """Split a dataset into train/test/(validation) child datasets in the catalog.

        sklearn-style split semantics. Creates a parent "split" dataset and
        2-3 child datasets (training, testing, optionally validation) in
        the catalog, all linked to the source via ``Dataset_Dataset``
        relations with full provenance.

        DerivaML's ``selection_fn`` (a Python callable) cannot cross the
        MCP boundary; this tool exposes only random and stratified
        strategies. For custom selection, use the Python API directly.

        Args:
            source_dataset_rid: RID of the source dataset to split.
            test_size: Float (0-1) fraction of data for testing. Default 0.2.
            train_size: Optional float (0-1) fraction for training. If None,
                complement of ``test_size`` (and ``val_size``).
            val_size: Optional float (0-1) fraction for validation. If None,
                no validation split is created (two-way split).
            seed: Random seed for reproducibility.
            shuffle: Whether to shuffle before splitting. Ignored when
                using stratified selection.
            stratify_by_column: Column name for stratified splitting (dot
                notation, e.g. ``Image_Classification.Image_Class``).
            stratify_missing: Policy for null values in the stratify
                column: ``"error"``, ``"drop"``, or ``"include"``.
            element_table: Name of the element table to split. If None,
                auto-detected from the source dataset's members.
            include_tables: Tables to include when denormalizing for the
                selection function. Required when ``stratify_by_column``
                is set.
            training_types: Additional dataset types for the training set
                beyond ``"Training"`` (e.g., ``["Labeled"]``).
            testing_types: Additional dataset types for the testing set
                beyond ``"Testing"``.
            validation_types: Additional dataset types for the validation
                set beyond ``"Validation"``. Ignored when ``val_size`` is None.
            split_description: Description for the parent Split dataset.
            workflow_type: Workflow type vocabulary term. Default
                ``"Dataset_Split"``.
            dry_run: If True, compute the split assignment without creating
                any catalog datasets. Use to validate strategy choice and
                partition sizes before committing.

        Returns:
            JSON string ``{"status": "split", "source", "split", "training",
            "testing", "validation", "strategy", "element_table", "seed",
            "dry_run"}``. Each partition is ``{"rid", "version", "count"}``.
            ``validation`` is null for two-way splits.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.dataset.split.split_dataset`` (e.g. invalid
                strategy, stratify column missing).

        Example:
            ``{"status": "split", "source": "1-AAAA", "split": {"rid":
            "1-SPLT", "version": "1.0.0", "count": 100}, "training":
            {"rid": "1-TRN", "version": "1.0.0", "count": 80}, "testing":
            {"rid": "1-TST", "version": "1.0.0", "count": 20},
            "validation": null, "strategy": "random", "element_table":
            "Image", "seed": 42, "dry_run": false}``.
        """
        try:
            with deriva_call():
                ml = _pkg.get_ml(hostname, catalog_id)
                result = _pkg._split_dataset(
                    ml,
                    source_dataset_rid,
                    test_size=test_size,
                    train_size=train_size,
                    val_size=val_size,
                    shuffle=shuffle,
                    seed=seed,
                    stratify_by_column=stratify_by_column,
                    stratify_missing=stratify_missing,
                    split_description=split_description,
                    training_types=training_types,
                    testing_types=testing_types,
                    validation_types=validation_types,
                    element_table=element_table,
                    include_tables=include_tables,
                    workflow_type=workflow_type,
                    dry_run=dry_run,
                )
                payload = result.model_dump()

            # dry_run doesn't mutate catalog state, so no audit row.
            # Convention: audit logs are for state changes only. The
            # response carries dry_run=True so callers see the mode.
            # (I-3 fix from Batch 3 review.)
            if not dry_run:
                training = payload.get("training") or {}
                testing = payload.get("testing") or {}
                validation = payload.get("validation") or {}
                _pkg.audit_event(
                    "deriva_ml_split_dataset",
                    hostname=hostname,
                    catalog_id=catalog_id,
                    source_dataset_rid=source_dataset_rid,
                    strategy=payload.get("strategy"),
                    element_table=payload.get("element_table"),
                    seed=seed,
                    training_count=training.get("count") if training else None,
                    testing_count=testing.get("count") if testing else None,
                    validation_count=validation.get("count") if validation else None,
                )
                # v1.3 surgical re-index: split creates a parent + 2-3
                # child datasets and links them via Dataset_Dataset
                # relations. Refresh the source dataset (relations
                # changed), the parent split, and each child so all
                # affected per-RID sources get fresh chunks. Per-RID
                # try/except so one failed re-index doesn't cancel the
                # rest -- defense in depth (the helper itself swallows,
                # but if the lazy import fails or the iteration raises,
                # we still want the other RIDs refreshed).
                from deriva_ml_mcp.resources.rag import _reindex_dataset

                affected_rids: list[str] = [source_dataset_rid]
                split_meta = payload.get("split") or {}
                for entry in (split_meta, training, testing, validation):
                    rid = entry.get("rid") if entry else None
                    if rid:
                        affected_rids.append(rid)
                for rid in affected_rids:
                    try:
                        await _reindex_dataset(hostname, catalog_id, rid)
                    except Exception:  # noqa: BLE001 -- best-effort, per-RID
                        logger.exception(
                            "re-index failed for dataset %s after split_dataset (source=%s)",
                            rid,
                            source_dataset_rid,
                        )
            # v3.0: status renamed from "success" to "split". Construct
            # via unpacking from the helper's payload (same fields).
            return SplitDatasetResponse(status="split", **payload).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="split_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                source_dataset_rid=source_dataset_rid,
                seed=seed,
                dry_run=dry_run,
            )
