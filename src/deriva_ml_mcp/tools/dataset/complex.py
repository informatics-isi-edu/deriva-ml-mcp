"""Complex dataset tools.

This submodule houses the substantial dataset tools that warrant
their own breathing room. As of v5.0.0 there is one:

- ``deriva_ml_denormalize_dataset`` (~220 lines, the biggest tool in
  the dataset domain): the catalog-shape vs dataset-described describe
  branch with a paged row preview and the bounded-materialization
  guard (the ``preflight_required`` short-circuit when estimated rows
  vastly exceed the requested limit).

v5.0.0 retired ``deriva_ml_cache_dataset`` per the stateless rule
(see ``CLAUDE.md``, ``docs/audit-2026-05-23.md``, and the design
note ``docs/superpowers/notes/2026-05-23-cache-denormalize-deprecation-design.md``).
The tool materialized a bag to the MCP server's local disk, where
the bytes were inaccessible to remote callers; the ``bag_directory``
field in the response described a server-side path that meant
nothing to the user. Bag materialization is now a user-local Python
operation: callers run ``ml.cache_dataset(spec)`` in their own
environment. The ``bag-preview`` resource and ``bag_info`` tool
remain for cost-sizing before download.

``deriva_ml_denormalize_dataset`` kept its place because the
OUTPUT it serves -- inline denormalized rows bounded by the 10x
preflight guard -- is exactly the kind of bounded, stateless data
MCP exists to serve. It still uses ``cache_dataset`` internally
(the bag has to be local for the FK walk), but that's an
implementation detail; the rows the caller sees do not depend on
server-side state.

Audit event lookup goes through ``_pkg.audit_event(...)`` (attribute
lookup on the parent package) for the same one-patch-site reason as
``mutate.py`` -- see that module's header note for the rationale.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from typing import TYPE_CHECKING, Any

from deriva_mcp_core import deriva_call

logger = logging.getLogger(__name__)

# See the header note in ``mutate.py`` for why ``audit_event`` and
# ``get_ml`` are accessed via attribute lookup on the parent package
# (``_pkg.<name>``) rather than direct ``from ... import``: a single
# ``patch("deriva_ml_mcp.tools.dataset.<name>")`` must redirect every
# call across read / mutate / complex submodules.
import deriva_ml_mcp.tools.dataset as _pkg  # noqa: E402  (intentional cycle)
from deriva_ml_mcp._helpers import (
    _MAX_LIMIT,
    _error_envelope,
    _paginate,
    _row_rid_for,
)
from deriva_ml_mcp._response_models import (
    PreflightCountResponse,
)

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def register(ctx: PluginContext) -> None:
    """Register the complex dataset tools with the plugin context.

    v5.0.0: one tool (``deriva_ml_denormalize_dataset``). The
    ``deriva_ml_cache_dataset`` tool was removed per the stateless rule
    (see this module's docstring + ``docs/audit-2026-05-23.md``).

    Args:
        ctx: PluginContext supplied by deriva-mcp-core at startup.

    Returns:
        None.

    Example:
        >>> from deriva_mcp_core.plugin.api import PluginContext
        >>> # ctx provided by the framework
        >>> register(ctx)  # doctest: +SKIP
    """

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
                    # Run the synchronous deriva-ml call in a thread pool
                    # so the event loop stays responsive —
                    # estimate_denormalized_size walks the catalog graph
                    # and can be slow on wide joins. See deriva-mcp-core
                    # plugin-authoring-guide.md §"Synchronous work in
                    # threads".
                    ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                    estimate = await asyncio.to_thread(
                        ml.estimate_denormalized_size, include_tables
                    )
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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive — describe_denormalized
                # and the generator drain below can take minutes on a
                # full join materialization. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in
                # threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                ds = await asyncio.to_thread(ml.lookup_dataset, dataset_rid)
                desc = await asyncio.to_thread(
                    ds.describe_denormalized,
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
                    # Drain the generator inside the thread pool —
                    # get_denormalized_as_dict materializes the join
                    # row-by-row over the wire, so islice() must run in
                    # the worker thread, not the event loop.
                    def _drain_rows() -> list[dict[str, Any]]:
                        return list(
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

                    rows = await asyncio.to_thread(_drain_rows)
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
