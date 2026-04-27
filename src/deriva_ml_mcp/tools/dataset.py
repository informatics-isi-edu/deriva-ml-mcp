"""Dataset domain tools for deriva-ml-mcp.

Read-only tools: list, lookup, browse members, walk relations, inspect
bag size, list element types, generate spec configs.

Mutation tools: create, delete, add/remove members, update types,
register element types, increment version.

Complex tools: cache_dataset (warm local cache), denormalize_dataset
(catalog-shape vs dataset-described view, with paged row preview),
split_dataset (sklearn-style train/test/val split with provenance).

Every tool wraps DERIVA I/O in ``with deriva_call():`` and routes errors
through ``_error_envelope`` (mutation tools also emit success/failure
audit events; reads only log on failure).
"""

from __future__ import annotations

import itertools
import json
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from deriva_mcp_core import deriva_call
from deriva_mcp_core.telemetry import audit_event
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion, VersionPart
from deriva_ml.dataset.dataset import Dataset
from deriva_ml.dataset.split import split_dataset as _split_dataset

# Note: `audit_event` is imported directly above (success-path audits
# fire from THIS module's namespace), and ALSO indirectly via
# `_error_envelope` in `_helpers` (failure-path audits fire from the
# helpers module's namespace). Tests that need to capture both must
# patch BOTH `deriva_ml_mcp.tools.dataset.audit_event` and
# `deriva_ml_mcp._helpers.audit_event` to the same mock — see
# `make_patch_audit("dataset")` in `tests/conftest.py` (the canonical
# factory). Python's import-binding semantics make a single-patch
# facade impossible without changing the call shape (e.g.
# `_helpers.audit_event(...)` at every success site, which is uglier
# than the dual patch).
from deriva_ml_mcp._helpers import (
    _MAX_LIMIT,
    _error_envelope,
    _paginate,
    _read_rid,
    _row_rid_for,
    _table_to_dict,
)
from deriva_ml_mcp.ml_context import get_ml

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


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


def _list_datasets_impl(
    ml: Any,
    *,
    after_rid: str | None,
    limit: int,
    include_deleted: bool = False,
) -> dict[str, Any]:
    """Fetch + paginate datasets. Pure helper -- shared by tool and resource.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        after_rid: Cursor for cursor pagination.
        limit: Max datasets per page (already capped by caller).
        include_deleted: Forward to ``find_datasets(deleted=...)``.

    Returns:
        Dict ``{"datasets": [...], "count", "truncated", "next_after_rid"}``.

    Example:
        >>> from unittest.mock import MagicMock
        >>> ml = MagicMock()
        >>> ml.find_datasets.return_value = []
        >>> _list_datasets_impl(ml, after_rid=None, limit=100)
        {'datasets': [], 'count': 0, 'truncated': False, 'next_after_rid': None}
    """
    datasets = sorted(
        ml.find_datasets(deleted=include_deleted),
        key=lambda d: d.dataset_rid,
    )
    page, truncated, next_after = _paginate(
        datasets,
        after_rid=after_rid,
        limit=limit,
        key=partial(_read_rid, rid_key="dataset_rid"),
    )
    return {
        "datasets": [_summarize_dataset(d) for d in page],
        "count": len(page),
        "truncated": truncated,
        "next_after_rid": next_after,
    }


def _get_dataset_detail_impl(ml: Any, dataset_rid: str) -> dict[str, Any]:
    """Build the dataset detail payload (summary + chaise URL + version history).

    Used by the ``deriva://catalog/{h}/{c}/ml/dataset/{rid}`` resource.
    The shape mirrors ``deriva_ml_get_dataset(include_history=True)`` but always
    includes ``version_history`` (renamed from the tool's ``history``
    key per the resource design in coverage.md).

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        dataset_rid: The RID of the dataset to look up.

    Returns:
        Dict with ``rid``, ``description``, ``dataset_types``,
        ``current_version``, ``chaise_url``, ``version_history``.

    Example:
        >>> # Illustrative -- exercises a live ml in real use.
    """
    ds = ml.lookup_dataset(dataset_rid)
    payload = _summarize_dataset(ds)
    payload["chaise_url"] = ds.get_chaise_url()
    payload["version_history"] = [
        {
            "version": str(h.dataset_version),
            "snapshot": h.snapshot,
            "description": h.description,
            "execution_rid": h.execution_rid,
        }
        for h in ds.dataset_history()
    ]
    return payload


def _list_dataset_members_summary_impl(ml: Any, dataset_rid: str) -> dict[str, Any]:
    """Build the dataset members summary (table -> count map + total).

    Resource-only convenience: returns the per-table counts plus a
    flattened ``members`` list of ``{table, rid}`` dicts capped at
    ``_MAX_LIMIT`` rows for the bundled summary view. The ``truncated``
    flag is True when the flattened list was capped; callers should
    use the ``deriva_ml_list_dataset_members`` tool with pagination to drill in.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        dataset_rid: The RID of the dataset to inspect.

    Returns:
        Dict ``{"dataset_rid", "summary": {<table>: count, ...},
        "total_count", "members": [{"table", "rid"}, ...],
        "truncated": bool, "tables": [...]}``.
    """
    ds = ml.lookup_dataset(dataset_rid)
    members_by_table = ds.list_dataset_members()
    summary = {tname: len(rows) for tname, rows in members_by_table.items()}
    total = sum(summary.values())
    flattened: list[dict[str, Any]] = []
    # When total > _MAX_LIMIT, this loop stops mid-way through whichever
    # table happens to be iterated last (dict iteration order = insertion
    # order). Per-table counts in `summary` remain accurate; only the
    # `members` flattening is truncated. Callers needing the full member
    # list should use the paginated `list_dataset_members` tool instead.
    for tname, rows in members_by_table.items():
        for row in rows:
            if len(flattened) >= _MAX_LIMIT:
                break
            flattened.append({"table": tname, "rid": row.get("RID", "")})
        if len(flattened) >= _MAX_LIMIT:
            break
    return {
        "dataset_rid": dataset_rid,
        "summary": summary,
        "total_count": total,
        "members": flattened,
        "truncated": total > len(flattened),
        "tables": list(members_by_table.keys()),
    }


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
    async def deriva_ml_list_datasets(
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
                if preflight_count:
                    total = len(list(ml.find_datasets(deleted=include_deleted)))
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
                payload = _list_datasets_impl(
                    ml,
                    after_rid=after_rid,
                    limit=capped,
                    include_deleted=include_deleted,
                )
            return json.dumps(payload)
        except Exception as exc:
            # Read-only tool: log+return without an audit row (I-2 fix).
            return _error_envelope(
                exc,
                operation="list_datasets",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_get_dataset(
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
            # Read-only tool: log+return without an audit row (I-2 fix).
            return _error_envelope(
                exc,
                operation="get_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_list_dataset_members(
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
                rows,
                after_rid=after_rid,
                limit=capped,
                key=partial(_read_rid, rid_key="RID"),
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
            # Read-only tool: log+return without an audit row (I-2 fix).
            return _error_envelope(
                exc,
                operation="list_dataset_members",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_list_dataset_relations(
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
                        key=partial(_read_rid, rid_key="dataset_rid"),
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
                        key=partial(_read_rid, rid_key="dataset_rid"),
                    )
                    result["children"] = [_summarize_dataset(c) for c in page]
                    result["children_truncated"] = truncated

            if warning is not None:
                result["warning"] = warning

            return json.dumps(result)
        except Exception as exc:
            # Read-only tool: log+return without an audit row (I-2 fix).
            return _error_envelope(
                exc,
                operation="list_dataset_relations",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_list_dataset_element_types(
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
                    "element_types": [_table_to_dict(t) for t in tables],
                    "count": len(tables),
                }
            )
        except Exception as exc:
            # Read-only tool: log+return without an audit row (I-2 fix).
            return _error_envelope(
                exc,
                operation="list_dataset_element_types",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_bag_info(
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
                ``deriva_ml_get_dataset`` to find the ``current_version`` if needed.
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
            # Read-only tool: log+return without an audit row (I-2 fix).
            return _error_envelope(
                exc,
                operation="bag_info",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_get_dataset_spec(
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
            # Read-only tool: log+return without an audit row (I-2 fix).
            return _error_envelope(
                exc,
                operation="get_dataset_spec",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    # ------------------------------------------------------------------
    # Mutation tools (Batch 2). Each emits audit_event on success and
    # routes failures through _error_envelope.
    # ------------------------------------------------------------------

    @ctx.tool(mutates=True)
    async def deriva_ml_create_dataset(
        hostname: str,
        catalog_id: str,
        execution_rid: str,
        dataset_types: list[str] | None = None,
        description: str = "",
        version: str | None = None,
    ) -> str:
        """Register a new dataset under an execution for provenance tracking.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            execution_rid: RID of the parent Execution. Required -- the new
                dataset's provenance is rooted in this execution.
            dataset_types: Optional list of dataset-type term names to tag
                the new dataset with.
            description: Free-text description of the dataset.
            version: Optional initial semver string (e.g. ``"1.0.0"``).
                If None, DerivaML defaults to ``0.1.0``.

        Returns:
            JSON string ``{"status": "created", "rid", "description",
            "dataset_types", "current_version", "execution_rid"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.dataset.dataset.Dataset.create_dataset`` (e.g.
                invalid execution_rid, unknown dataset_type term).

        Example:
            ``{"status": "created", "rid": "1-NEW", "description": "...",
            "dataset_types": ["Training"], "current_version": "0.1.0",
            "execution_rid": "1-EXEC"}``.
        """
        try:
            parsed_version = DatasetVersion.parse(version) if version else None
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                new_ds = Dataset.create_dataset(
                    ml,
                    execution_rid=execution_rid,
                    dataset_types=dataset_types,
                    description=description,
                    version=parsed_version,
                )
                summary = _summarize_dataset(new_ds)
            audit_event(
                "deriva_ml_create_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
                dataset_rid=new_ds.dataset_rid,
                dataset_types=dataset_types or [],
                version=summary["current_version"],
            )
            return json.dumps({"status": "created", **summary, "execution_rid": execution_rid})
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="create_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
                dataset_types=dataset_types or [],
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_delete_dataset(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        recurse: bool = False,
    ) -> str:
        """Soft-delete a dataset (sets ``Deleted=True`` on the catalog row).

        With ``recurse=True``, also soft-deletes the dataset's nested
        children. Without ``recurse``, a dataset that has parents (i.e. is
        nested in another dataset) cannot be deleted; the upstream raises.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            dataset_rid: The RID of the dataset to delete.
            recurse: If True, also soft-delete nested child datasets.

        Returns:
            JSON string ``{"status": "deleted", "dataset_rid", "recursive"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.delete_dataset`` (e.g. dataset not
                found, dataset is nested and ``recurse=False``).

        Example:
            ``{"status": "deleted", "dataset_rid": "1-AAAA",
            "recursive": false}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                ds = ml.lookup_dataset(dataset_rid)
                ml.delete_dataset(ds, recurse=recurse)
            audit_event(
                "deriva_ml_delete_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                recursive=recurse,
            )
            return json.dumps(
                {
                    "status": "deleted",
                    "dataset_rid": dataset_rid,
                    "recursive": recurse,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="delete_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                recursive=recurse,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_add_dataset_members(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        member_rids: list[str] | None = None,
        members_by_table: dict[str, list[str]] | None = None,
        description: str = "",
        execution_rid: str | None = None,
    ) -> str:
        """Add records (or whole datasets) as members of a dataset.

        Provide EITHER ``member_rids`` (a mixed-table list -- each RID is
        resolved to its table) OR ``members_by_table`` (a table-keyed dict
        -- faster, no resolution needed). Exactly one of the two must be
        non-empty.

        Adding members automatically increments the dataset's minor
        version. Member tables must already be registered as dataset
        element types (see ``deriva_ml_add_dataset_element_type``).

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            dataset_rid: The RID of the dataset to add members to.
            member_rids: Mixed-table list of RIDs to add.
            members_by_table: Map of table name to list of RIDs.
            description: Free-text description recorded with the version
                bump.
            execution_rid: Optional execution to attribute the change to.

        Returns:
            JSON string ``{"status": "success", "added_count",
            "dataset_rid", "new_version"}``.

        Raises:
            ValueError: If neither or both of ``member_rids``/
                ``members_by_table`` are supplied.
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.dataset.dataset.Dataset.add_dataset_members``
                (e.g. table not a registered element type, would create a
                cycle, duplicate members).

        Example:
            ``{"status": "success", "added_count": 3, "dataset_rid":
            "1-AAAA", "new_version": "1.2.0"}``.
        """
        has_list = bool(member_rids)
        has_dict = bool(members_by_table)
        if has_list == has_dict:
            return json.dumps(
                {
                    "error": (
                        "Provide exactly one of member_rids or "
                        "members_by_table (and it must be non-empty)."
                    )
                }
            )
        members: list[str] | dict[str, list[str]]
        if has_list:
            members = list(member_rids or [])
            added_count = len(members)
        else:
            members = dict(members_by_table or {})
            added_count = sum(len(v) for v in members.values())
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                ds = ml.lookup_dataset(dataset_rid)
                ds.add_dataset_members(
                    members=members,
                    description=description,
                    execution_rid=execution_rid,
                )
                new_version = str(ds.current_version) if ds.current_version is not None else None
            audit_event(
                "deriva_ml_add_dataset_members",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                added_count=added_count,
                new_version=new_version,
            )
            return json.dumps(
                {
                    "status": "success",
                    "added_count": added_count,
                    "dataset_rid": dataset_rid,
                    "new_version": new_version,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="add_dataset_members",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                attempted_count=added_count,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_delete_dataset_members(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        member_rids: list[str],
        description: str = "",
        execution_rid: str | None = None,
    ) -> str:
        """Remove records from a dataset.

        The records themselves are NOT deleted -- only the association
        rows linking them to the dataset are removed. Removing members
        increments the dataset's minor version.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            dataset_rid: The RID of the dataset to remove members from.
            member_rids: List of member RIDs to remove.
            description: Free-text description recorded with the version
                bump.
            execution_rid: Optional execution to attribute the change to.

        Returns:
            JSON string ``{"status": "success", "removed_count",
            "dataset_rid", "new_version"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.dataset.dataset.Dataset.delete_dataset_members``
                (e.g. RID not part of dataset).

        Example:
            ``{"status": "success", "removed_count": 2, "dataset_rid":
            "1-AAAA", "new_version": "1.3.0"}``.
        """
        removed_count = len(member_rids)
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                ds = ml.lookup_dataset(dataset_rid)
                ds.delete_dataset_members(
                    members=member_rids,
                    description=description,
                    execution_rid=execution_rid,
                )
                new_version = str(ds.current_version) if ds.current_version is not None else None
            audit_event(
                "deriva_ml_delete_dataset_members",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                removed_count=removed_count,
                new_version=new_version,
            )
            return json.dumps(
                {
                    "status": "success",
                    "removed_count": removed_count,
                    "dataset_rid": dataset_rid,
                    "new_version": new_version,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="delete_dataset_members",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                attempted_count=removed_count,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_update_dataset_types(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        add: list[str] | None = None,
        remove: list[str] | None = None,
    ) -> str:
        """Add and/or remove dataset-type tags on a dataset in one call.

        Either or both of ``add`` and ``remove`` may be empty; an empty
        call is a no-op (still returns success with the unchanged type
        list). Adds happen before removes.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            dataset_rid: The RID of the dataset whose types to update.
            add: Dataset_Type term names to add.
            remove: Dataset_Type term names to remove.

        Returns:
            JSON string ``{"status": "updated", "dataset_rid",
            "dataset_types", "added", "removed", "new_version"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.dataset.dataset.Dataset.add_dataset_types`` or
                ``remove_dataset_type`` (e.g. unknown vocabulary term).

        Example:
            ``{"status": "updated", "dataset_rid": "1-AAAA",
            "dataset_types": ["Training", "Validation"],
            "added": ["Validation"], "removed": [], "new_version":
            "1.4.0"}``.
        """
        adds = list(add or [])
        removes = list(remove or [])
        # Track partial progress for the failure path: an `add_dataset_types`
        # call is atomic-per-call (all-or-none for the list), but `remove`
        # is a per-term loop that can fail mid-way. The LLM caller needs
        # visibility into which removes succeeded before failure so it can
        # reason about partial state.
        adds_done: list[str] = []
        removes_done: list[str] = []
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                ds = ml.lookup_dataset(dataset_rid)
                if adds:
                    ds.add_dataset_types(adds)
                    adds_done = list(adds)
                for term in removes:
                    ds.remove_dataset_type(term)
                    removes_done.append(term)
                updated_types = list(ds.dataset_types) if ds.dataset_types else []
                new_version = str(ds.current_version) if ds.current_version is not None else None
            audit_event(
                "deriva_ml_update_dataset_types",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                added=adds,
                removed=removes,
                new_version=new_version,
            )
            return json.dumps(
                {
                    "status": "updated",
                    "dataset_rid": dataset_rid,
                    "dataset_types": updated_types,
                    "added": adds,
                    "removed": removes,
                    "new_version": new_version,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="update_dataset_types",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                added=adds,
                removed=removes,
                added_done=adds_done,
                removed_done=removes_done,
                response_fields={
                    "dataset_rid": dataset_rid,
                    "added_done": adds_done,
                    "removed_done": removes_done,
                    "added_requested": adds,
                    "removed_requested": removes,
                },
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_add_dataset_element_type(
        hostname: str,
        catalog_id: str,
        table_name: str,
    ) -> str:
        """Register a domain table as a valid dataset member type.

        Creates the association table linking the dataset table to
        ``table_name``. The caller must have catalog admin privileges.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            table_name: Name of the domain table to register.

        Returns:
            JSON string ``{"status": "success", "table_name",
            "association_table"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.add_dataset_element_type`` (e.g.
                unknown table, table is system/ML and not eligible).

        Example:
            ``{"status": "success", "table_name": "Image",
            "association_table": "Dataset_Image"}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                assoc_table = ml.add_dataset_element_type(table_name)
                association_name = assoc_table.name
            audit_event(
                "deriva_ml_add_dataset_element_type",
                hostname=hostname,
                catalog_id=catalog_id,
                table_name=table_name,
                association_table=association_name,
            )
            return json.dumps(
                {
                    "status": "success",
                    "table_name": table_name,
                    "association_table": association_name,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="add_dataset_element_type",
                hostname=hostname,
                catalog_id=catalog_id,
                table_name=table_name,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_increment_dataset_version(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        component: Literal["major", "minor", "patch"] = "minor",
        description: str = "",
        execution_rid: str | None = None,
    ) -> str:
        """Bump a dataset's semver version after upstream changes.

        Use this to record a new version after metadata/data updates that
        the regular member/type APIs don't already version (e.g. after an
        out-of-band catalog edit). Bumping a higher-order component resets
        lower-order components to zero.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            dataset_rid: The RID of the dataset to bump.
            component: Which semver component to increment.
            description: Free-text description recorded with the bump.
            execution_rid: Optional execution to attribute the bump to.

        Returns:
            JSON string ``{"status": "success", "dataset_rid",
            "previous_version", "new_version", "component"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.dataset.dataset.Dataset.increment_dataset_version``.

        Example:
            ``{"status": "success", "dataset_rid": "1-AAAA",
            "previous_version": "1.2.0", "new_version": "1.3.0",
            "component": "minor"}``.
        """
        try:
            version_part = VersionPart(component)
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                ds = ml.lookup_dataset(dataset_rid)
                previous_version = (
                    str(ds.current_version) if ds.current_version is not None else None
                )
                new_version = ds.increment_dataset_version(
                    component=version_part,
                    description=description,
                    execution_rid=execution_rid,
                )
                new_version_str = str(new_version)
            audit_event(
                "deriva_ml_increment_dataset_version",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                component=component,
                previous_version=previous_version,
                new_version=new_version_str,
            )
            return json.dumps(
                {
                    "status": "success",
                    "dataset_rid": dataset_rid,
                    "previous_version": previous_version,
                    "new_version": new_version_str,
                    "component": component,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="increment_dataset_version",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                component=component,
            )

    # Complex tools (Batch 3). cache_dataset is local-FS-only (mutates=True
    # because it touches the cache filesystem; no catalog state is changed).
    # denormalize_dataset is read-only. split_dataset creates child datasets
    # in the catalog (mutates=True).

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
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            dataset_rid: RID of the dataset to cache.
            version: Specific version (semver string). If None, looks up the
                dataset's current_version automatically.
            materialize: If True (default), download asset files. If False,
                only fetch the bag metadata.
            exclude_tables: Tables to omit from the bag (e.g. large blob
                tables you don't need).

        Returns:
            JSON string ``{"status": "success", "dataset_rid", "version",
            "materialize", "tables", "total_rows", "total_asset_bytes",
            "total_asset_size", "cache_status", "cache_path"}``. Pass-through
            of DerivaML's bag_info-shaped return.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.cache_dataset``.

        Example:
            ``{"status": "success", "dataset_rid": "1-AAAA", "version":
            "1.0.0", "materialize": true, "cache_status":
            "cached_materialized", "cache_path": "/path/to/bag", ...}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
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
            audit_event(
                "deriva_ml_cache_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                version=used_version,
                materialize=materialize,
                exclude_tables=exclude_tables or [],
            )
            return json.dumps(
                {
                    "status": "success",
                    "dataset_rid": dataset_rid,
                    "version": used_version,
                    "materialize": materialize,
                    **info,
                },
                default=str,
            )
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
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
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
            JSON string. Catalog-shape: ``{"mode": "catalog_shape",
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
                    ml = get_ml(hostname, catalog_id)
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
                ml = get_ml(hostname, catalog_id)
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
                return json.dumps(
                    {
                        "mode": "dataset_preflight",
                        "dataset_rid": dataset_rid,
                        "total_count": total,
                        "entities_fetched": False,
                        "action_required": action,
                    },
                    default=str,
                )

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
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
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
            JSON string ``{"status": "success", "source", "split", "training",
            "testing", "validation", "strategy", "element_table", "seed",
            "dry_run"}``. Each partition is ``{"rid", "version", "count"}``.
            ``validation`` is null for two-way splits.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.dataset.split.split_dataset`` (e.g. invalid
                strategy, stratify column missing).

        Example:
            ``{"status": "success", "source": "1-AAAA", "split": {"rid":
            "1-SPLT", "version": "1.0.0", "count": 100}, "training":
            {"rid": "1-TRN", "version": "1.0.0", "count": 80}, "testing":
            {"rid": "1-TST", "version": "1.0.0", "count": 20},
            "validation": null, "strategy": "random", "element_table":
            "Image", "seed": 42, "dry_run": false}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                result = _split_dataset(
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
                audit_event(
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
            return json.dumps({"status": "success", **payload}, default=str)
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
