"""Dataset domain tools for deriva-ml-mcp.

Read-only operations on datasets: list, lookup, browse members,
walk relations, inspect bag size, generate spec configs.

Mutation tools (create, delete, add/remove members, update types, etc.)
land in subsequent batches.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from deriva_mcp_core import deriva_call
from deriva_ml.dataset.aux_classes import DatasetSpec

from deriva_ml_mcp.ml_context import get_ml

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


_MAX_LIMIT = 1000


def _summarize_dataset(ds: Any) -> dict[str, Any]:
    """Render a Dataset object into the JSON-friendly summary used by list endpoints.

    Args:
        ds: A ``deriva_ml.dataset.dataset.Dataset`` instance.

    Returns:
        Dict with ``rid``, ``description``, ``dataset_types``, ``current_version``.
    """
    current = ds.current_version
    return {
        "rid": ds.dataset_rid,
        "description": ds.description,
        "dataset_types": list(ds.dataset_types) if ds.dataset_types else [],
        "current_version": str(current) if current is not None else None,
    }


def _paginate(
    items: list[Any],
    *,
    after_rid: str | None,
    limit: int,
    rid_key: str = "rid",
) -> tuple[list[Any], bool, str | None]:
    """Apply cursor pagination to a sorted list keyed by RID.

    Args:
        items: Source list, sorted by the RID key in ascending order.
        after_rid: Skip items with RID <= after_rid.
        limit: Page size (already capped by caller).
        rid_key: Attribute or dict key used to read the RID from each item.

    Returns:
        Tuple of (page, truncated, next_after_rid).

    Note:
        ``truncated`` is True whenever the page returned exactly ``limit``
        rows, even if there happen to be no more rows. This may produce a
        false positive in the edge case where ``len(items) == limit``
        exactly, but the convention matches deriva-mcp-core's
        ``get_entities`` so callers can use one pagination idiom across
        both layers.
    """
    if after_rid is not None:
        items = [it for it in items if _read_rid(it, rid_key) > after_rid]
    page = items[:limit]
    truncated = len(page) == limit
    next_after_rid = _read_rid(page[-1], rid_key) if page and truncated else None
    return page, truncated, next_after_rid


def _read_rid(item: Any, rid_key: str) -> str:
    """Read a RID from either a dict or an object."""
    if isinstance(item, dict):
        # Dict members may use "RID" (catalog convention) or the configured key.
        for candidate in ("RID", rid_key):
            if candidate in item:
                return str(item[candidate])
        raise KeyError(f"no RID-like key in member dict (looked for RID, {rid_key})")
    return str(getattr(item, rid_key))


def register(ctx: PluginContext) -> None:
    """Register all read-only dataset tools with the plugin context.

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
    async def list_datasets(
        hostname: str,
        catalog_id: str,
        include_deleted: bool = False,
        limit: int = 100,
        after_rid: str | None = None,
        preflight_count: bool = False,
    ) -> str:
        """Browse all datasets in the catalog with optional pagination.

        PAGINATION: When the dataset count is unknown, call with
        ``preflight_count=True`` first to get just the total count. Present
        that to the user, choose a limit, then call again with
        ``preflight_count=False``. Use ``after_rid`` (the RID of the last row
        from the previous page) to advance the cursor.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            include_deleted: Include soft-deleted datasets if True.
            limit: Max datasets per page (default 100, max 1000).
            after_rid: RID of last row from previous page to advance cursor.
            preflight_count: If True, return only total count.

        Returns:
            JSON string. Preflight:
            ``{"total_count": N, "entities_fetched": False, "action_required": "..."}``.
            Page: ``{"datasets": [{"rid", "description", "dataset_types",
            "current_version"}, ...], "count": N, "truncated": bool,
            "next_after_rid": <last-rid> | null}``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated from
                ``deriva_ml.DerivaML.find_datasets``.

        Example:
            ``{"datasets": [{"rid": "1-AAAA", "description": "...",
            "dataset_types": ["Training"], "current_version": "1.0.0"}, ...],
            "count": 100, "truncated": true, "next_after_rid": "1-CDEF"}``
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                datasets = sorted(
                    ml.find_datasets(deleted=include_deleted),
                    key=lambda d: d.dataset_rid,
                )

            if preflight_count:
                total = len(datasets)
                return json.dumps(
                    {
                        "total_count": total,
                        "entities_fetched": False,
                        "action_required": (
                            f"Found {total} datasets. Choose a limit and call "
                            "again with preflight_count=False."
                        ),
                    }
                )

            capped = min(max(limit, 0), _MAX_LIMIT)
            page, truncated, next_after = _paginate(
                datasets, after_rid=after_rid, limit=capped, rid_key="dataset_rid"
            )
            return json.dumps(
                {
                    "datasets": [_summarize_dataset(d) for d in page],
                    "count": len(page),
                    "truncated": truncated,
                    "next_after_rid": next_after,
                }
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @ctx.tool(mutates=False)
    async def get_dataset(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        include_history: bool = False,
    ) -> str:
        """Read one dataset's full summary, optionally with version history.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            dataset_rid: The RID of the dataset to retrieve.
            include_history: If True, include the dataset's version history.

        Returns:
            JSON string with the full dataset summary:
            ``{"rid", "description", "dataset_types", "current_version",
            "chaise_url", "history": [...] | omitted}``.

        Raises:
            RuntimeError: If the dataset RID doesn't exist, propagated from
                ``deriva_ml.DerivaML.lookup_dataset`` (returned as
                ``{"error": ...}``).

        Example:
            ``{"rid": "1-AAAA", "description": "Training set",
            "dataset_types": ["Training"], "current_version": "1.0.0",
            "chaise_url": "https://example.org/chaise/..."}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                ds = ml.lookup_dataset(dataset_rid)
                summary = _summarize_dataset(ds)
                summary["chaise_url"] = ds.get_chaise_url()
                if include_history:
                    summary["history"] = [
                        {
                            "version": str(h.dataset_version),
                            "snapshot": h.snapshot,
                            "description": h.description,
                            "execution_rid": h.execution_rid,
                        }
                        for h in ds.dataset_history()
                    ]
            return json.dumps(summary)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @ctx.tool(mutates=False)
    async def list_dataset_members(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        element_table: str | None = None,
        limit: int = 100,
        after_rid: str | None = None,
        preflight_count: bool = False,
        recurse: bool = False,
        version: str | None = None,
    ) -> str:
        """List a dataset's members, grouped by element type.

        TWO MODES:

        - Summary mode (``element_table=None``): returns per-table counts, no
          rows. Bounded output. Use this to discover what's in a dataset.
        - Page mode (``element_table`` set): returns paged rows from one
          table. Standard cursor pagination via
          ``limit`` / ``after_rid`` / ``preflight_count``.

        Workflow: call with ``element_table=None`` first to see the shape;
        then call again with ``element_table`` set to drill into one table.

        ``recurse=True`` and ``version`` apply to BOTH modes — summary counts
        reflect the recursive/versioned member set just like page rows do.
        When ``recurse=True``, ``summary["Image"]`` includes images from
        nested child datasets, matching what a subsequent page-mode call
        with the same ``recurse`` value will return.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            dataset_rid: The RID of the dataset to inspect.
            element_table: Table name to drill into; ``None`` for summary.
            limit: Max rows per page (default 100, max 1000). Page mode only.
            after_rid: RID of last row from previous page. Page mode only.
            preflight_count: If True, return only total count for the table.
            recurse: If True, include members from nested child datasets.
            version: Optional dataset version to query.

        Returns:
            JSON string. Summary: ``{"summary": {<tname>: count, ...},
            "total": N, "tables": [...]}``. Page (preflight):
            ``{"element_table", "total_count", "entities_fetched": False,
            "action_required": "..."}``. Page (rows):
            ``{"element_table", "rows": [...], "returned_count": N,
            "truncated": bool, "next_after_rid": <last-rid> | null}``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated from
                ``deriva_ml.dataset.dataset.Dataset.list_dataset_members``.

        Example:
            Summary: ``{"summary": {"Image": 50, "Subject": 10}, "total": 60,
            "tables": ["Image", "Subject"]}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                ds = ml.lookup_dataset(dataset_rid)
                members = ds.list_dataset_members(recurse=recurse, version=version)

            if element_table is None:
                summary = {tname: len(rows) for tname, rows in members.items()}
                return json.dumps(
                    {
                        "summary": summary,
                        "total": sum(summary.values()),
                        "tables": list(members.keys()),
                    }
                )

            if element_table not in members:
                return json.dumps(
                    {
                        "error": (
                            f"element_table '{element_table}' not in dataset "
                            f"members. Available: {list(members.keys())}"
                        )
                    }
                )

            rows = sorted(members[element_table], key=lambda r: r.get("RID", ""))

            if preflight_count:
                return json.dumps(
                    {
                        "element_table": element_table,
                        "total_count": len(rows),
                        "entities_fetched": False,
                        "action_required": (
                            f"{len(rows)} rows in '{element_table}'. Choose a "
                            "limit and call again with preflight_count=False."
                        ),
                    }
                )

            capped = min(max(limit, 0), _MAX_LIMIT)
            page, truncated, next_after = _paginate(
                rows, after_rid=after_rid, limit=capped, rid_key="RID"
            )
            return json.dumps(
                {
                    "element_table": element_table,
                    "rows": page,
                    "returned_count": len(page),
                    "truncated": truncated,
                    "next_after_rid": next_after,
                }
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @ctx.tool(mutates=False)
    async def list_dataset_relations(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        direction: Literal["parents", "children", "both"] = "both",
        recurse: bool = False,
        limit: int = 100,
        after_rid: str | None = None,
        version: str | None = None,
    ) -> str:
        """Walk a dataset's nesting hierarchy in either direction.

        PAGINATION: ``after_rid`` only applies when ``direction`` is
        ``"parents"`` or ``"children"``. When ``direction="both"``, parents
        and children come from disjoint RID-spaces and a single cursor is
        incoherent — ``after_rid`` is ignored and a ``warning`` field is
        included in the response. To page both sides, call once per
        direction with that side's own ``after_rid``.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            dataset_rid: The RID of the dataset whose relations to list.
            direction: Which side(s) to walk: ``"parents"``, ``"children"``,
                or ``"both"``.
            recurse: If True, also include indirect ancestors/descendants.
            limit: Max relations per side (default 100, max 1000).
            after_rid: RID cursor; skip relations with RID <= after_rid.
                Ignored when ``direction="both"`` (see above).
            version: Optional dataset version to query.

        Returns:
            JSON string with the requested keys present:
            ``{"parents": [<dataset summary>], "children": [<dataset summary>],
            "parents_truncated": bool, "children_truncated": bool,
            "warning": str | omitted}``.
            ``parents`` / ``parents_truncated`` are omitted when
            ``direction="children"``; vice versa for ``children``.
            ``warning`` appears only when ``after_rid`` was supplied with
            ``direction="both"``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated from
                ``deriva_ml.dataset.dataset.Dataset.list_dataset_parents`` /
                ``list_dataset_children``.

        Example:
            ``{"parents": [{"rid": "1-PAR", ...}], "children": []}``.
        """
        try:
            capped = min(max(limit, 0), _MAX_LIMIT)
            # When walking both sides, the single cursor is incoherent
            # because parents and children RIDs aren't synchronized. Drop
            # the cursor and warn rather than silently mis-paginating.
            effective_after_rid: str | None = after_rid
            warning: str | None = None
            if direction == "both" and after_rid is not None:
                effective_after_rid = None
                warning = (
                    "after_rid was ignored because direction='both'. "
                    "To page both sides, call once per direction with "
                    "that side's own after_rid."
                )
            result: dict[str, Any] = {}
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                ds = ml.lookup_dataset(dataset_rid)

                if direction in ("parents", "both"):
                    parents = sorted(
                        ds.list_dataset_parents(recurse=recurse, version=version),
                        key=lambda d: d.dataset_rid,
                    )
                    page, truncated, _ = _paginate(
                        parents,
                        after_rid=effective_after_rid,
                        limit=capped,
                        rid_key="dataset_rid",
                    )
                    result["parents"] = [_summarize_dataset(p) for p in page]
                    result["parents_truncated"] = truncated

                if direction in ("children", "both"):
                    children = sorted(
                        ds.list_dataset_children(recurse=recurse, version=version),
                        key=lambda d: d.dataset_rid,
                    )
                    page, truncated, _ = _paginate(
                        children,
                        after_rid=effective_after_rid,
                        limit=capped,
                        rid_key="dataset_rid",
                    )
                    result["children"] = [_summarize_dataset(c) for c in page]
                    result["children_truncated"] = truncated

            if warning is not None:
                result["warning"] = warning

            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @ctx.tool(mutates=False)
    async def list_dataset_element_types(
        hostname: str,
        catalog_id: str,
    ) -> str:
        """List all tables registered as valid dataset member types.

        No pagination — element types are bounded (typically 1-20 per catalog).

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.

        Returns:
            JSON string ``{"element_types": [{"name", "schema"}, ...],
            "count": N}``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated from
                ``deriva_ml.DerivaML.list_dataset_element_types``.

        Example:
            ``{"element_types": [{"name": "Image", "schema": "domain"}],
            "count": 1}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                tables = list(ml.list_dataset_element_types())
            return json.dumps(
                {
                    "element_types": [{"name": t.name, "schema": t.schema.name} for t in tables],
                    "count": len(tables),
                }
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @ctx.tool(mutates=False)
    async def bag_info(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        version: str,
        exclude_tables: list[str] | None = None,
    ) -> str:
        """Describe a dataset bag's size and cache status without downloading.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            dataset_rid: The RID of the dataset to inspect.
            version: The exact version (e.g. ``"1.0.0"``). Required — use
                ``get_dataset`` to find the ``current_version`` if needed.
            exclude_tables: Tables to omit from the bag (e.g. large blob
                tables).

        Returns:
            JSON string ``{"tables": {<name>: {"row_count", "is_asset",
            "asset_bytes"}}, "total_rows", "total_asset_bytes",
            "total_asset_size", "cache_status", "cache_path"}``. Fields are
            shaped by ``deriva_ml.DerivaML.bag_info``; non-JSON-serializable
            values (e.g. ``Path``) are stringified.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated from
                ``deriva_ml.DerivaML.bag_info``.

        Example:
            ``{"tables": {"Image": {"row_count": 100, "is_asset": true,
            "asset_bytes": 12345}}, "total_rows": 100, "total_asset_bytes":
            12345, "total_asset_size": "12 KB", "cache_status": "not_cached",
            "cache_path": null}``.
        """
        try:
            spec = DatasetSpec(
                rid=dataset_rid,
                version=version,
                exclude_tables=set(exclude_tables) if exclude_tables else None,
            )
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                info = ml.bag_info(spec)
            return json.dumps(info, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @ctx.tool(mutates=False)
    async def get_dataset_spec(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        version: str | None = None,
    ) -> str:
        """Generate a ``DatasetSpecConfig(...)`` snippet for a hydra-zen config.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            dataset_rid: The RID of the dataset.
            version: Specific version to pin. If omitted, falls back to the
                dataset's current version and emits a warning recommending an
                explicit pin for reproducibility.

        Returns:
            JSON string ``{"spec": "DatasetSpecConfig(rid=\\"...\\",
            version=\\"...\\")", "dataset_rid", "version", "description",
            "dataset_types", "warning": str | null}``. ``warning`` is set when
            ``version`` was None and the current version was substituted.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated from
                ``deriva_ml.DerivaML.lookup_dataset`` when the RID is invalid.

        Example:
            ``{"spec": "DatasetSpecConfig(rid=\\"1-AAAA\\", version=\\"1.0.0\\")",
            "dataset_rid": "1-AAAA", "version": "1.0.0", "description": "...",
            "dataset_types": ["Training"], "warning": null}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                ds = ml.lookup_dataset(dataset_rid)

            warning: str | None
            if version is not None:
                used_version = version
                warning = None
            else:
                used_version = (
                    str(ds.current_version) if ds.current_version is not None else "0.1.0"
                )
                warning = (
                    f"version not specified; using current version "
                    f"{used_version}. For reproducibility, pin to an explicit "
                    "version in your config."
                )

            spec = f'DatasetSpecConfig(rid="{dataset_rid}", version="{used_version}")'
            return json.dumps(
                {
                    "spec": spec,
                    "dataset_rid": dataset_rid,
                    "version": used_version,
                    "description": ds.description,
                    "dataset_types": list(ds.dataset_types) if ds.dataset_types else [],
                    "warning": warning,
                }
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})
