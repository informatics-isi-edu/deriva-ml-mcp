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

import asyncio
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
    _cite_dataset_version_url,
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
    sort: bool = False,
) -> DatasetListResponse:
    """Fetch + paginate datasets. Pure helper -- shared by tool and resource.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        after_rid: Cursor for cursor pagination.
        limit: Max datasets per page (already capped by caller).
        include_deleted: Forward to ``find_datasets(deleted=...)``.
        sort: If True, results are ordered newest-first by record
            creation time (RCT desc) -- forwarded to
            ``deriva_ml.DerivaML.find_datasets(sort=True)``. If False
            (default), results are RID-ascending for stable cursor
            pagination. Note that under ``sort=True`` the ``after_rid``
            cursor still works ("skip up to this RID in the RCT-sorted
            result"), but pagination through very large sorted result
            sets is bounded by the internal fetch cap.

    Returns:
        ``DatasetListResponse`` -- see ``deriva_ml_mcp._response_models``.
    """
    raw = list(
        ml.find_datasets(
            deleted=include_deleted,
            sort=True if sort else None,
        )
    )
    # Keep stable RID-ascending order for the default path; under
    # sort=True we honor the catalog-side RCT-desc ordering and skip
    # the post-fetch sort (sorting by RID would clobber the RCT order).
    if sort:
        datasets = raw
    else:
        datasets = sorted(raw, key=lambda d: d.dataset_rid)
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

    Used by the ``deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}`` resource.
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
    cite = _cite_dataset_version_url(
        ml, dataset_rid, str(ds.current_version), ds=ds
    )
    return DatasetDetail(
        **summary.model_dump(),
        cite_url=cite if isinstance(cite, str) else None,
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
        sort: bool = False,
    ) -> str:
        """Browse all datasets in the catalog with optional pagination.

        [Cold-start orientation -- WORKAROUND pending deriva-mcp-core's
        plugin-contributed server-instructions hook. When that lands,
        remove this whole bracketed block (this docstring and any other
        tool that grew one); the same content will be advertised once at
        server-init time via the ``instructions=`` field instead of being
        duplicated on every client schema fetch. See CLAUDE.md
        "Cross-Repo Sync" section for the upstream change.]

        DerivaML's five core abstractions:
          Dataset    -- versioned collection of catalog rows (typed via
                        ``Dataset_Type`` vocabulary; has an element-type
                        spec, may include assets).
          Workflow   -- versioned reference to code (URL + git commit);
                        content-addressed, typed via ``Workflow_Type``.
          Execution  -- one run of a Workflow against input Datasets,
                        producing output Datasets / Features / Assets.
          Feature    -- typed per-row annotation produced by an
                        Execution (e.g. a per-image label, a per-execution
                        scalar metric).
          Asset      -- file uploaded to hatrac with provenance link to
                        the producing Execution.

        Provenance principle: every Dataset / Feature / Asset row points
        at the Execution that produced it. Create the Execution BEFORE
        writing outputs. Traverse with ``deriva_ml_get_lineage``.

        Vocabularies: extend with ``add_term(schema="deriva-ml",
        table="<Vocab>", ...)`` -- do NOT create new vocabulary tables
        for ML domain concepts.

        Full guide prompts (fetch via MCP ``prompts/get``):
          ``deriva_ml_concepts`` -- the conceptual frame above in full.
          ``deriva_ml_getting_started`` -- the (hostname, catalog_id)
            call rule, pagination contract, resource-vs-tool discovery,
            and workflow-to-execution-to-outputs chain.

        PAGINATION CONTRACT (two-step):
          1. Call with ``preflight_count=True`` -- returns ``total_count``,
             no rows.
          2. If ``total_count > limit``, call again with
             ``after_rid=<last RID>`` from the prior page to advance the
             cursor. Repeat until ``truncated`` is False.

        [End cold-start block.]

        Args:
            include_deleted: Include soft-deleted datasets if True.
            limit: Max datasets per page (default 100, max 1000).
            after_rid: RID of last row from previous page to advance cursor.
            preflight_count: If True, return only total count.
            sort: If True, return results newest-first by record
                creation time. Recommended for "show me the most
                recent datasets" queries. Default False preserves the
                stable RID-ascending order used for cursor pagination.

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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``find_datasets`` and
                # the ``_list_datasets_impl`` helper both do catalog
                # round-trips. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in
                # threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                if preflight_count:

                    def _count() -> int:
                        return len(list(ml.find_datasets(deleted=include_deleted)))

                    total = await asyncio.to_thread(_count)
                    return PreflightCountResponse(
                        total_count=total,
                        action_required=(
                            f"Found {total} datasets. Choose a limit and call "
                            "again with preflight_count=False."
                        ),
                    ).model_dump_json(by_alias=True)

                capped = min(max(limit, 0), _MAX_LIMIT)
                payload = await asyncio.to_thread(
                    _list_datasets_impl,
                    ml,
                    after_rid=after_rid,
                    limit=capped,
                    include_deleted=include_deleted,
                    sort=sort,
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
            ``deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}`` resource has
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
            "cite_url": "https://example.org/id/1/1-AAAA@350-XXXX-YYYY",
            "version_history": []}``.
        """
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``lookup_dataset`` and
                # ``_get_dataset_detail_impl`` (which drains
                # ``ds.dataset_history()``) both do catalog round-trips.
                # See deriva-mcp-core plugin-authoring-guide.md
                # §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                if include_history:
                    payload = await asyncio.to_thread(
                        _get_dataset_detail_impl, ml, dataset_rid
                    )
                else:
                    # Skip the dataset_history() call entirely when not
                    # requested -- it can be expensive on long-lived
                    # datasets. Construct a DatasetDetail with an empty
                    # version_history list to preserve the unified wire
                    # shape.
                    ds = await asyncio.to_thread(ml.lookup_dataset, dataset_rid)
                    summary = _summarize_dataset(ds)
                    cite = _cite_dataset_version_url(
                        ml, dataset_rid, str(ds.current_version), ds=ds
                    )
                    payload = DatasetDetail(
                        **summary.model_dump(),
                        cite_url=cite if isinstance(cite, str) else None,
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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``list_dataset_members``
                # walks the dataset's element tables and can fetch many
                # rows. See deriva-mcp-core plugin-authoring-guide.md
                # §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                ds = await asyncio.to_thread(ml.lookup_dataset, dataset_rid)
                members = await asyncio.to_thread(
                    ds.list_dataset_members, recurse=recurse, version=version
                )

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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``list_dataset_parents``
                # and ``list_dataset_children`` walk the dataset hierarchy
                # over the wire. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in
                # threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                ds = await asyncio.to_thread(ml.lookup_dataset, dataset_rid)

                if direction in ("parents", "both"):
                    parents_raw = await asyncio.to_thread(
                        ds.list_dataset_parents, recurse=recurse, version=version
                    )
                    parents = sorted(parents_raw, key=lambda d: d.dataset_rid)
                    page, truncated, _ = _paginate(
                        parents,
                        after_rid=effective_after_rid,
                        limit=capped,
                        key=partial(_read_rid, rid_key="dataset_rid"),
                    )
                    kwargs["parents"] = [_summarize_dataset(p) for p in page]
                    kwargs["parents_truncated"] = truncated

                if direction in ("children", "both"):
                    children_raw = await asyncio.to_thread(
                        ds.list_dataset_children, recurse=recurse, version=version
                    )
                    children = sorted(children_raw, key=lambda d: d.dataset_rid)
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

        No tool-specific arguments beyond the standard
        ``hostname`` / ``catalog_id`` pair. No pagination — element
        types are bounded (typically 1-20 per catalog).

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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``list_dataset_element_types``
                # is a metadata round-trip. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in
                # threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)

                def _drain_element_types() -> list[Any]:
                    return list(ml.list_dataset_element_types())

                tables = await asyncio.to_thread(_drain_element_types)
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
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``_bag_info_impl``
                # inspects the bag manifest and may touch the local
                # cache. See deriva-mcp-core plugin-authoring-guide.md
                # §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                info = await asyncio.to_thread(
                    _bag_info_impl,
                    ml,
                    dataset_rid,
                    version,
                    exclude_tables=exclude_tables,
                )
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

        Same payload as the
        ``deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}/spec`` resource --
        the two share an internal helper so the payloads cannot drift.
        The resource form omits the ``version`` parameter (always uses
        current version with a warning); callers needing to pin a
        specific version use the tool.

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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``_get_dataset_spec_impl``
                # invokes ``ml.lookup_dataset`` which does a catalog
                # round-trip. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in
                # threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(
                    _get_dataset_spec_impl, ml, dataset_rid, version
                )
            return json.dumps(payload)
        except Exception as exc:
            # Read-only tool: log+return without an audit row (I-2 fix).
            return _error_envelope(
                exc,
                operation="get_dataset_spec",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_validate_dataset_specs(
        hostname: str,
        catalog_id: str,
        specs: list[dict[str, Any] | str],
    ) -> str:
        """Validate a list of ``(RID, version)`` dataset specs against the catalog.

        Cheap metadata-only pre-flight: confirms each spec's RID exists,
        points at a Dataset, and the named version exists. Returns
        per-spec results so the caller can present a fix-this-list
        rather than "config invalid, good luck."

        Use case: editing ``src/configs/datasets.py`` and want to
        confirm specific RID+version pairs resolve before saving.
        For full pre-flight including assets and workflow, use
        ``deriva_ml_validate_execution_configuration``. For a heavier
        check that exercises the actual bag-download path (catches
        FK-traversal timeouts etc.), run an Execution with
        ``dry_run=True`` -- but expect minutes-to-hours of cost (see
        deriva-ml ADR-0002).

        Args:
            specs: List of dataset specs. Each item is either a
                ``DatasetSpec`` shorthand (a bare RID string) or a
                dict with ``rid`` and ``version`` keys (matching
                ``DatasetSpec`` constructor kwargs). Coerced to
                ``DatasetSpec`` before validation.

        Returns:
            JSON string of the ``DatasetSpecValidationReport``:
            ``{"all_valid": bool, "results": [...]}``. Each result
            carries ``{rid, version, valid, reason?, available_versions?,
            dataset_name?, resolved_version?, warning?}``. Failure
            ``reason`` is one of: ``rid_not_found``, ``not_a_dataset``,
            ``version_not_found``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, e.g. on a
                Pydantic validation error if ``specs`` items can't be
                coerced to ``DatasetSpec``.

        Example:
            ``{"all_valid": false, "results": [{"rid": "1-ABCD",
            "version": "1.0.0", "valid": true, "dataset_name":
            "Training set"}, {"rid": "2-XYZW", "version": "9.9.9",
            "valid": false, "reason": "version_not_found",
            "available_versions": ["1.0.0", "1.1.0"]}]}``.
        """
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``validate_dataset_specs``
                # walks each spec and issues a metadata round-trip per
                # RID. See deriva-mcp-core plugin-authoring-guide.md
                # §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(ml.validate_dataset_specs, specs)
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="validate_dataset_specs",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_validate_execution_configuration(
        hostname: str,
        catalog_id: str,
        config: dict[str, Any],
    ) -> str:
        """Validate a full ``ExecutionConfiguration`` against the catalog.

        The cheap, metadata-only pre-flight gate before
        ``deriva-ml-run``. Walks the config's ``datasets`` and
        ``assets`` lists, validates the workflow, and reports per-spec
        results plus cross-spec issues (duplicate RIDs across specs,
        version conflicts on the same dataset, role conflicts on the
        same asset).

        **Why this exists separately from ``dry_run=True``.** Setting
        ``dry_run=True`` on an Execution does validate the config -- by
        actually downloading every dataset bag and materializing every
        asset, which can take minutes-to-hours and several GB of
        bandwidth. ``validate_execution_configuration`` is the cheap
        alternative for fast iteration: O(N) metadata round-trips,
        ~hundreds of milliseconds for typical configs, no
        ``Execution`` row created. See deriva-ml ADR-0002 for the full
        rationale.

        Args:
            config: An ``ExecutionConfiguration`` shorthand dict with
                keys matching the ``ExecutionConfiguration`` constructor:
                ``workflow`` (RID string or dict), ``datasets`` (list
                of ``DatasetSpec`` shorthands), ``assets`` (list of
                ``AssetSpec`` shorthands), and the other config fields.
                Coerced to ``ExecutionConfiguration`` before validation.

        Returns:
            JSON string of the ``ExecutionConfigurationValidationReport``:
            ``{"all_valid": bool, "dataset_results": [...],
            "asset_results": [...], "workflow_result": {...} | null,
            "cross_spec_issues": [...]}``. Each per-spec result has
            its own structured failure reason; cross-spec issues use
            reasons ``duplicate_rid``, ``version_conflict``, or
            ``role_conflict``. ``workflow_result`` is null when the
            config has no workflow (partial config; the user is
            still building it).

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, e.g. on a
                Pydantic validation error if ``config`` cannot be
                coerced to ``ExecutionConfiguration``.

        Example:
            ``{"all_valid": false, "dataset_results": [...],
            "asset_results": [...], "workflow_result": {"rid":
            "1-WFLW", "valid": true}, "cross_spec_issues":
            [{"reason": "duplicate_rid", "rid": "1-DUPL",
            "spec_lists": ["datasets", "datasets"]}]}``.
        """
        try:
            from deriva_ml.execution.execution_configuration import ExecutionConfiguration

            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``validate_execution_configuration``
                # walks the full config (workflow, datasets, assets) and
                # issues per-spec metadata round-trips. See
                # deriva-mcp-core plugin-authoring-guide.md §"Synchronous
                # work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                # MCP wire always delivers a dict; coerce to the
                # Pydantic class. Pydantic's ValidationError on bad
                # input bubbles up through the except below.
                exec_config = ExecutionConfiguration(**config)
                payload = await asyncio.to_thread(
                    ml.validate_execution_configuration, exec_config
                )
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="validate_execution_configuration",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_validate_config_file(
        hostname: str,
        catalog_id: str,
        file_contents: str,
    ) -> str:
        """Validate every spec constructor in a hydra-zen config file against the catalog.

        Parses the file via AST (no execution) and validates each
        ``DatasetSpecConfig`` / ``AssetSpecConfig`` / ``Workflow`` /
        ``DerivaMLConfig`` call. AST-only -- the file is never
        executed -- so the surface is safe to point at arbitrary
        user code in ``src/configs/``.

        Use case: a one-shot pre-flight gate before running an
        experiment, OR after a release lands, to confirm every RID
        in the config file still resolves and pins a released
        version.

        v0.5.0: the ``file_path=`` parameter was removed. The MCP
        server's filesystem view does not match the caller's, so a
        path read on the server side was always going to be wrong for
        any deployment other than single-machine localhost. Pass the
        file contents as a string instead; the server writes a temp
        file for the AST walker and cleans up on exit.

        Args:
            file_contents: The file as a string. The validator writes
                the contents to a temp file (cleaned up immediately)
                so the AST walker (which takes a path) can consume it.

        Returns:
            JSON string of a ``ConfigValidationReport``:
            ``{"file_count", "entry_count", "all_valid", "results":
            [...], "parse_errors": [...]}``. Each result carries the
            parsed entry (file, line, kind, rid, version), a valid
            flag, a reasons list, and helpful detail like
            ``available_versions`` for ``version_not_found``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}`` for I/O errors
                writing the temp file.

        Example:
            ``{"file_count": 1, "entry_count": 4, "all_valid": false,
            "results": [{"entry": {"file": "src/configs/datasets.py",
            "line": 14, "entry_kind": "DatasetSpecConfig", "rid":
            "2-B4C8", "version": "9.9.9", ...}, "valid": false,
            "reasons": ["version_not_found"], "available_versions":
            ["0.4.0", "0.3.0"]}, ...], "parse_errors": []}``.
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                # Write file_contents to a temp file so the AST walker
                # (which takes a path) can consume it. The temp file is
                # cleaned up immediately after parsing.
                import os
                import tempfile
                with tempfile.NamedTemporaryFile(
                    suffix=".py", mode="w", delete=False
                ) as tmp:
                    tmp.write(file_contents)
                    tmp_path = tmp.name
                try:
                    payload = await asyncio.to_thread(
                        ml.validate_config_file, tmp_path
                    )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="validate_config_file",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_bootstrap_config(
        hostname: str,
        catalog_id: str,
        kinds: list[str] | None = None,
        dataset_type_filter: list[str] | None = None,
    ) -> str:
        """Suggest config entries by reading the catalog.

        Walks the catalog and produces structured per-kind suggestions
        for a fresh project's ``src/configs/``. Pure read; does NOT
        write files. Each suggestion includes a ready-to-paste
        ``spec_string`` (a ``DatasetSpecConfig(...)`` / etc. line) and
        a ``rationale`` the calling skill shows the user when
        offering the entry.

        Three use cases:

        - **Fresh project.** Empty ``configs/`` -- get a base set of
          suggestions to seed every group.
        - **Catalog clone.** Repoint configs at a new catalog id;
          bootstrap regenerates the spec_strings against the new RIDs.
        - **Incremental update.** ``kinds=["datasets"]`` to refresh
          just the datasets group after a release without touching
          assets / workflow.

        Args:
            kinds: Which config groups to suggest entries for.
                Default (None) means all four: ``deriva_ml``,
                ``datasets``, ``assets``, ``workflow``. Skipping
                ``experiments`` / ``multiruns`` / ``model_config`` is
                intentional -- those are project code, not catalog
                state.
            dataset_type_filter: When suggesting datasets, restrict
                to these ``Dataset_Type`` terms. Default (None) uses
                ``["Training", "Testing", "Validation", "Complete",
                "Labeled"]``. Pass ``[]`` (empty list) to include
                every dataset_type.

        Returns:
            JSON string of a ``BootstrapReport``:
            ``{"catalog": {...}, "suggestions": [...], "skipped":
            [...]}``. Each suggestion carries ``{kind, config_name,
            rid, version?, spec_string, description?, rationale}``.
            Each skipped entry carries ``{kind, rid, reason}``.

        Example:
            ``{"catalog": {"hostname": "data.example.org",
            "catalog_id": "1"}, "suggestions": [{"kind": "datasets",
            "config_name": "cifar_10_training", "rid": "2-B4C8",
            "version": "0.4.0", "spec_string":
            "DatasetSpecConfig(rid=\\\"2-B4C8\\\", version=\\\"0.4.0\\\")",
            "description": "CIFAR-10 training images", "rationale":
            "Dataset type Training; latest released version 0.4.0."}],
            "skipped": []}``.
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(
                    partial(
                        ml.bootstrap_config,
                        kinds=kinds,
                        dataset_type_filter=dataset_type_filter,
                    )
                )
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="bootstrap_config",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )


def _bag_info_impl(
    ml: Any,
    dataset_rid: str,
    version: str | None,
    *,
    exclude_tables: list[str] | None = None,
) -> dict[str, Any]:
    """Shared helper: tool ``deriva_ml_bag_info`` and resource
    ``deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}/bag-preview`` both call
    this so their payloads cannot drift.

    Args:
        ml: The ``DerivaML`` instance bound to the catalog.
        dataset_rid: Dataset RID to inspect.
        version: Explicit version to pin, or None to use the dataset's
            current version (a ``warning`` is added to the payload when
            the substitution happens; same convention as
            ``_get_dataset_spec_impl``).
        exclude_tables: Optional tables to omit from the bag preview
            (e.g. large blob tables the caller doesn't intend to
            download). Resource form omits this; tool form forwards
            verbatim.

    Returns:
        The dict produced by ``deriva_ml.DerivaML.bag_info(spec)``,
        with non-JSON-serializable values (e.g. ``Path``) stringified
        at the wire by callers. Adds a ``warning`` key when ``version``
        was None and current-version fallback fired; the warning is
        None otherwise.

    Example:
        >>> _bag_info_impl(ml, "1-AAAA", None)  # doctest: +SKIP
    """
    warning: str | None
    if version is not None:
        used_version = version
        warning = None
    else:
        ds = ml.lookup_dataset(dataset_rid)
        used_version = (
            str(ds.current_version) if ds.current_version is not None else "0.1.0"
        )
        warning = (
            f"version not specified; using current version "
            f"{used_version}. For reproducibility, pin to an explicit "
            "version when downloading."
        )

    spec = DatasetSpec(
        rid=dataset_rid,
        version=used_version,
        exclude_tables=set(exclude_tables) if exclude_tables else None,
    )
    info = ml.bag_info(spec)
    # ``ml.bag_info`` returns a plain dict; add the warning + the
    # resolved version so callers can see what was actually inspected.
    info["dataset_rid"] = dataset_rid
    info["version"] = used_version
    info["warning"] = warning
    return info


def _get_dataset_spec_impl(
    ml: Any, dataset_rid: str, version: str | None
) -> dict[str, Any]:
    """Shared helper: tool ``deriva_ml_get_dataset_spec`` and resource
    ``deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}/spec`` both call this so
    their payloads cannot drift.

    Args:
        ml: The ``DerivaML`` instance bound to the catalog.
        dataset_rid: Dataset RID.
        version: Explicit version to pin, or None to use the
            dataset's current version (a warning is emitted in the
            payload when the substitution happens).

    Returns:
        ``{"spec": "DatasetSpecConfig(rid=...)", "dataset_rid",
        "version", "description", "dataset_types", "warning": str |
        None}``.

    Example:
        >>> _get_dataset_spec_impl(ml, "1-AAAA", None)  # doctest: +SKIP
    """
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
    return {
        "spec": spec,
        "dataset_rid": dataset_rid,
        "version": used_version,
        "description": ds.description,
        "dataset_types": list(ds.dataset_types) if ds.dataset_types else [],
        "warning": warning,
    }
