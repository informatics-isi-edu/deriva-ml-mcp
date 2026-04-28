"""Read-only dataset tools and shared dataset helpers.

This submodule houses the 7 read tools (``deriva_ml_list_datasets``,
``deriva_ml_get_dataset``, ``deriva_ml_list_dataset_members``,
``deriva_ml_list_dataset_relations``, ``deriva_ml_list_dataset_element_types``,
``deriva_ml_bag_info``, ``deriva_ml_get_dataset_spec``) plus the four
helpers (``_summarize_dataset``, ``_list_datasets_impl``,
``_get_dataset_detail_impl``, ``_list_dataset_members_summary_impl``)
that the read tools and the ``resources/ml.py`` / ``resources/rag.py``
modules consume to keep tool / resource shapes in sync.

The helpers live here (not in a sibling ``_helpers.py``) because they
are dataset-specific and the resource modules already import them via
``deriva_ml_mcp.tools.dataset`` -- the package ``__init__.py`` re-exports
them so that import path keeps resolving identically after the split.

Every tool wraps DERIVA I/O in ``with deriva_call():`` and routes errors
through ``_error_envelope`` (reads only log on failure, no audit row).
"""

from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from deriva_mcp_core import deriva_call
from deriva_ml.dataset.aux_classes import DatasetSpec

# ``get_ml`` is accessed inside tool bodies via attribute lookup on the
# parent package (``_pkg.get_ml``) so a single
# ``patch("deriva_ml_mcp.tools.dataset.get_ml")`` (used by the
# ``dataset_ctx`` fixture in tests/test_dataset.py) redirects every
# call across read / mutate / complex submodules. A direct
# ``from deriva_ml_mcp.ml_context import get_ml`` here would create a
# per-submodule binding the patch can't reach.
import deriva_ml_mcp.tools.dataset as _pkg  # noqa: E402  (intentional cycle)
from deriva_ml_mcp._helpers import (
    _MAX_LIMIT,
    _error_envelope,
    _paginate,
    _read_rid,
    _table_to_dict,
)
from deriva_ml_mcp._response_models import (
    DatasetDetail,
    DatasetListResponse,
    DatasetMemberRef,
    DatasetMembersSummaryResponse,
    DatasetRelationsResponse,
    DatasetSummary,
    DatasetVersionEntry,
    PreflightCountResponse,
)

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def _summarize_dataset(ds: Any) -> DatasetSummary:
    """Render a Dataset object into the validated summary used by list endpoints.

    Args:
        ds: A ``deriva_ml.dataset.dataset.Dataset`` instance.

    Returns:
        ``DatasetSummary`` Pydantic instance -- see
        ``deriva_ml_mcp._response_models``.

    Note:
        v2.2 sweep: this helper now returns Pydantic. Consumers that
        previously dict-accessed must use attribute access or call
        ``.model_dump()`` to get a plain dict back.
    """
    current = ds.current_version
    return DatasetSummary(
        rid=ds.dataset_rid,
        description=ds.description,
        dataset_types=list(ds.dataset_types) if ds.dataset_types else [],
        current_version=str(current) if current is not None else None,
    )


def _list_datasets_impl(
    ml: Any,
    *,
    after_rid: str | None,
    limit: int,
    include_deleted: bool = False,
) -> DatasetListResponse:
    """Fetch + paginate datasets. Pure helper -- shared by tool and resource.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        after_rid: Cursor for cursor pagination.
        limit: Max datasets per page (already capped by caller).
        include_deleted: Forward to ``find_datasets(deleted=...)``.

    Returns:
        ``DatasetListResponse`` -- see ``deriva_ml_mcp._response_models``.
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
    return DatasetListResponse(
        datasets=[_summarize_dataset(d) for d in page],
        count=len(page),
        truncated=truncated,
        next_after_rid=next_after,
    )


def _get_dataset_detail_impl(ml: Any, dataset_rid: str) -> DatasetDetail:
    """Build the dataset detail payload (summary + chaise URL + version history).

    Used by the ``deriva://catalog/{h}/{c}/ml/dataset/{rid}`` resource.
    The shape mirrors ``deriva_ml_get_dataset(include_history=True)``
    but always includes ``version_history``.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        dataset_rid: The RID of the dataset to look up.

    Returns:
        ``DatasetDetail`` -- see ``deriva_ml_mcp._response_models``.
        ``version_history`` is a list of deriva-ml ``DatasetHistory``
        Pydantic models (no re-declaration of the version-row shape on
        our side).
    """
    ds = ml.lookup_dataset(dataset_rid)
    summary = _summarize_dataset(ds)
    version_history = [
        DatasetVersionEntry(
            version=str(h.dataset_version),
            snapshot=h.snapshot,
            description=h.description,
            execution_rid=h.execution_rid,
        )
        for h in ds.dataset_history()
    ]
    return DatasetDetail(
        **summary.model_dump(),
        chaise_url=ds.get_chaise_url(),
        version_history=version_history,
    )


def _list_dataset_members_summary_impl(ml: Any, dataset_rid: str) -> DatasetMembersSummaryResponse:
    """Build the dataset members summary (table -> count map + total).

    Resource-only convenience: returns the per-table counts plus a
    flattened ``members`` list of ``{table, rid}`` dicts capped at
    ``_MAX_LIMIT`` rows for the bundled summary view. The ``truncated``
    flag is True when the flattened list was capped; callers should
    use the ``deriva_ml_list_dataset_members`` tool with pagination to
    drill in.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        dataset_rid: The RID of the dataset to inspect.

    Returns:
        ``DatasetMembersSummaryResponse`` -- see
        ``deriva_ml_mcp._response_models``.
    """
    ds = ml.lookup_dataset(dataset_rid)
    members_by_table = ds.list_dataset_members()
    summary = {tname: len(rows) for tname, rows in members_by_table.items()}
    total = sum(summary.values())
    flattened: list[DatasetMemberRef] = []
    # When total > _MAX_LIMIT, this loop stops mid-way through whichever
    # table happens to be iterated last (dict iteration order = insertion
    # order). Per-table counts in `summary` remain accurate; only the
    # `members` flattening is truncated. Callers needing the full member
    # list should use the paginated `list_dataset_members` tool instead.
    for tname, rows in members_by_table.items():
        for row in rows:
            if len(flattened) >= _MAX_LIMIT:
                break
            flattened.append(DatasetMemberRef(table=tname, rid=row.get("RID", "")))
        if len(flattened) >= _MAX_LIMIT:
            break
    return DatasetMembersSummaryResponse(
        dataset_rid=dataset_rid,
        summary=summary,
        total_count=total,
        members=flattened,
        truncated=total > len(flattened),
        tables=list(members_by_table.keys()),
    )


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

        See ``deriva_ml_getting_started`` (PAGINATION CONTRACT) for the two-step pagination flow.

        Args:
            include_deleted: Include soft-deleted datasets if True.
            limit: Max datasets per page (default 100, max 1000).
            after_rid: RID of last row from previous page to advance cursor.
            preflight_count: If True, return only total count.

        Returns:
            Preflight:
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
                ml = _pkg.get_ml(hostname, catalog_id)
                if preflight_count:
                    total = len(list(ml.find_datasets(deleted=include_deleted)))
                    return PreflightCountResponse(
                        total_count=total,
                        action_required=(
                            f"Found {total} datasets. Choose a limit and call "
                            "again with preflight_count=False."
                        ),
                    ).model_dump_json(by_alias=True)

                capped = min(max(limit, 0), _MAX_LIMIT)
                payload = _list_datasets_impl(
                    ml,
                    after_rid=after_rid,
                    limit=capped,
                    include_deleted=include_deleted,
                )
            return payload.model_dump_json(by_alias=True)
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
            dataset_rid: The RID of the dataset to retrieve.
            include_history: If True, populate the ``version_history``
                field with the full dataset version history. When
                False, ``version_history`` is present but empty.

        Returns:
            ``DatasetDetail`` JSON shape -- see
            ``deriva_ml_mcp._response_models``.

        Note:
            v2.0 wire change vs v1.x: the field key is
            ``version_history`` (was ``history`` in v1.x). This unifies
            the tool's wire shape with the resource's wire shape (the
            ``deriva://catalog/{h}/{c}/ml/dataset/{rid}`` resource has
            always used ``version_history``). The two surfaces now
            return identical wire shapes; consumers can switch freely
            between tool and resource without re-mapping field names.
            The field is always present (was conditionally omitted in
            v1.x); when ``include_history=False`` it's an empty list.

        Raises:
            RuntimeError: If the dataset RID doesn't exist, propagated from
                ``deriva_ml.DerivaML.lookup_dataset`` (returned as
                ``{"error": ...}``).

        Example:
            ``{"rid": "1-AAAA", "description": "Training set",
            "dataset_types": ["Training"], "current_version": "1.0.0",
            "chaise_url": "https://example.org/chaise/...",
            "version_history": []}``.
        """
        try:
            with deriva_call():
                ml = _pkg.get_ml(hostname, catalog_id)
                if include_history:
                    payload = _get_dataset_detail_impl(ml, dataset_rid)
                else:
                    # Skip the dataset_history() call entirely when not
                    # requested -- it can be expensive on long-lived
                    # datasets. Construct a DatasetDetail with an empty
                    # version_history list to preserve the unified wire
                    # shape.
                    ds = ml.lookup_dataset(dataset_rid)
                    summary = _summarize_dataset(ds)
                    payload = DatasetDetail(
                        **summary.model_dump(),
                        chaise_url=ds.get_chaise_url(),
                        version_history=[],
                    )
            return payload.model_dump_json(by_alias=True)
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
            dataset_rid: The RID of the dataset to inspect.
            element_table: Table name to drill into; ``None`` for summary.
            limit: Max rows per page (default 100, max 1000). Page mode only.
            after_rid: RID of last row from previous page. Page mode only.
            preflight_count: If True, return only total count for the table.
            recurse: If True, include members from nested child datasets.
            version: Optional dataset version to query.

        Returns:
            Summary: ``{"summary": {<tname>: count, ...},
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
                ml = _pkg.get_ml(hostname, catalog_id)
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
                return PreflightCountResponse(
                    total_count=len(rows),
                    action_required=(
                        f"{len(rows)} rows in '{element_table}'. Choose a "
                        "limit and call again with preflight_count=False."
                    ),
                    element_table=element_table,
                ).model_dump_json(by_alias=True)

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

        See ``deriva_ml_getting_started`` (PAGINATION CONTRACT) for the two-step pagination flow.

        Args:
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
            kwargs: dict[str, Any] = {}
            with deriva_call():
                ml = _pkg.get_ml(hostname, catalog_id)
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
                    kwargs["parents"] = [_summarize_dataset(p) for p in page]
                    kwargs["parents_truncated"] = truncated

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
                    kwargs["children"] = [_summarize_dataset(c) for c in page]
                    kwargs["children_truncated"] = truncated

            if warning is not None:
                kwargs["warning"] = warning

            # ``exclude_none=True`` preserves the v1.x wire shape where the
            # opposite-direction fields are *omitted* (not serialized as
            # ``null``). When ``direction="parents"`` the response has no
            # ``children`` / ``children_truncated`` keys at all.
            return DatasetRelationsResponse(**kwargs).model_dump_json(
                by_alias=True, exclude_none=True
            )
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
                ml = _pkg.get_ml(hostname, catalog_id)
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
                ml = _pkg.get_ml(hostname, catalog_id)
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
                ml = _pkg.get_ml(hostname, catalog_id)
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
