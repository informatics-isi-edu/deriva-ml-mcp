"""Asset domain tools for deriva-ml-mcp.

Read tools: ``deriva_ml_list_asset_tables``, ``deriva_ml_list_assets``,
``deriva_ml_lookup_asset``.
Mutation tools: ``deriva_ml_update_asset``.

Scope. This module is **catalog-state only**: browse asset tables, list
the rows in one, look up bundled per-asset detail, and curate the
metadata (asset_type tags + Description) on an existing asset row.

File I/O is deliberately out of scope -- registering a new asset from
a local file or downloading asset bytes back to a local path requires
filesystem access the MCP server does not (and should not) have. Those
flows live in the ``deriva-skills`` ``work-with-assets`` skill, which
generates Python the user runs locally; once the file is staged, this
module's tools cover the metadata-curation half of the round trip.

Every tool wraps DERIVA I/O in ``with deriva_call():`` and routes
errors through ``_error_envelope`` (mutation tools also emit
success/failure audit events; reads only log on failure).

Dataset/Workflow precedent. The shapes here mirror the existing
read+update-by-RID surfaces:

- ``list_asset_tables`` returns ``{name, schema}`` dicts via
  ``_table_to_dict`` (same shape used by every other "list tables"
  endpoint in the plugin).
- ``list_assets`` paginates by Asset RID using the standard
  ``_paginate`` + ``_read_rid(rid_key="asset_rid")`` cursor pattern.
- ``lookup_asset`` returns a bundled detail (the same payload the
  ``deriva://catalog/{h}/{c}/ml/asset/{rid}`` resource serves) so
  resource and tool stay byte-identical via the shared
  ``_get_asset_detail_impl`` helper.
- ``update_asset`` follows the curation pattern shared with
  ``update_dataset`` / ``update_workflow`` / ``update_execution``: a
  single per-entity update tool whose kwargs are all optional and at
  least one of which must be non-None. ``asset_types`` is set-style
  (compute the diff against current types and call add/remove on
  each); ``description`` is a free-text overwrite written through
  pathBuilder (the deriva-ml ``Asset`` class does NOT expose a
  catalog-write setter for ``description`` -- only ``Workflow`` and
  ``ExecutionRecord`` do, so we write the catalog row directly here
  via the asset table's pathBuilder).
"""

from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING, Any

from deriva_mcp_core import deriva_call
from deriva_mcp_core.telemetry import audit_event

# Note on testing audit_event: see ``make_patch_audit("asset")`` in
# ``tests/_helpers.py``. Single-patch facade is impossible due to
# Python's ``from X import name`` import binding semantics -- tests must
# patch BOTH ``deriva_ml_mcp.tools.asset.audit_event`` (this module's
# success-path emission) and ``deriva_ml_mcp._helpers.audit_event`` (the
# failure-path emission inside ``_error_envelope``).
from deriva_ml_mcp._helpers import (
    _MAX_LIMIT,
    _error_envelope,
    _paginate,
    _read_rid,
    _set_row_description,
    _table_to_dict,
)
from deriva_ml_mcp._response_models import (
    AssetDetail,
    AssetExecutionRef,
    AssetListResponse,
    AssetSummary,
    AssetTableRef,
    AssetTablesResponse,
)
from deriva_ml_mcp.ml_context import get_ml

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


# ---------------------------------------------------------------------------
# Helpers (also consumed by ``resources/ml.py`` for the asset-tables and
# asset-detail resources, per the Tool/Resource Dual-Mode Policy in
# ``CLAUDE.md``).
# ---------------------------------------------------------------------------


def _summarize_asset(asset: Any) -> dict[str, Any]:
    """Render an Asset object into the JSON-friendly summary used by list endpoints.

    Args:
        asset: A ``deriva_ml.asset.asset.Asset`` instance.

    Returns:
        Dict matching the ``AssetSummary`` Pydantic model shape. Kept
        dict-returning to preserve PR 1 scope -- consumers that build
        list responses construct the model from these dicts.

    Example:
        >>> from unittest.mock import MagicMock
        >>> a = MagicMock()
        >>> a.asset_rid = "1-AAAA"
        >>> a.filename = "image.png"
        >>> a.length = 12345
        >>> a.md5 = "abc"
        >>> a.asset_table = "Image"
        >>> a.asset_types = ["Training_Data"]
        >>> _summarize_asset(a)["rid"]
        '1-AAAA'
    """
    return {
        "rid": asset.asset_rid,
        "filename": asset.filename,
        "length": asset.length,
        "md5": asset.md5,
        "asset_table": asset.asset_table,
        "asset_types": list(asset.asset_types) if asset.asset_types else [],
    }


def _list_asset_tables_impl(ml: Any) -> AssetTablesResponse:
    """Fetch the catalog's asset tables. Pure helper -- shared by tool and resource.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.

    Returns:
        ``AssetTablesResponse`` -- see ``deriva_ml_mcp._response_models``.
        No pagination -- catalogs typically have a handful of asset
        tables and ``Table`` objects don't carry a stable RID for the
        cursor protocol.
    """
    tables = list(ml.list_asset_tables())
    rendered = [_table_to_dict(t) for t in tables]
    return AssetTablesResponse(
        asset_tables=[AssetTableRef.model_validate(r) for r in rendered],
        count=len(rendered),
    )


def _get_asset_detail_impl(ml: Any, asset_rid: str) -> AssetDetail:
    """Build the bundled asset detail payload. Pure helper -- shared by tool and resource.

    Bundles the per-asset summary (rid/filename/length/md5/url/
    description/asset_types/asset_table) plus a list of associated
    executions via ``Asset.list_executions()``. Each execution entry is
    ``{"rid": <execution_rid>, "asset_role": "Input"|"Output"|None}`` so
    callers can split provenance roles client-side without a follow-up
    tool call. The role lookup is best-effort: when the deriva-ml
    ``ExecutionRecord`` doesn't carry a per-asset role attribute (the
    common case -- ``list_asset_executions`` returns plain
    ``ExecutionRecord``s), the role surfaces as ``None``.

    If ``Asset.list_executions`` raises (e.g. malformed catalog
    metadata), the executions list comes back empty rather than aborting
    the whole detail payload -- mirrors the
    ``_get_execution_detail_impl`` best-effort pattern in
    ``tools/execution.py``.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        asset_rid: The RID of the asset to look up.

    Returns:
        ``AssetDetail`` -- see ``deriva_ml_mcp._response_models``.
    """
    asset = ml.lookup_asset(asset_rid)
    executions: list[AssetExecutionRef] = []
    try:
        for record in asset.list_executions():
            executions.append(
                AssetExecutionRef(
                    rid=getattr(record, "execution_rid", None),
                    asset_role=getattr(record, "asset_role", None),
                )
            )
    except Exception:  # noqa: BLE001 -- executions list is best-effort
        executions = []
    return AssetDetail(
        rid=asset.asset_rid,
        filename=asset.filename,
        length=asset.length,
        md5=asset.md5,
        url=asset.url,
        description=asset.description,
        asset_table=asset.asset_table,
        asset_types=list(asset.asset_types) if asset.asset_types else [],
        executions=executions,
    )


def _write_asset_description(ml: Any, asset: Any, description: str) -> None:
    """Write a new Description value to an asset row in the catalog.

    Thin wrapper around the shared ``_set_row_description`` helper:
    resolves the asset's ``Table`` object from its ``asset_table``
    name, delegates the catalog write, then mirrors the value into
    the in-memory ``asset.description`` attribute.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        asset: The Asset object to update (used for its ``asset_rid``,
            ``asset_table``, and as the in-memory mirror target).
        description: The new description value to write.
    """
    asset_table_obj = ml.model.name_to_table(asset.asset_table)
    _set_row_description(ml, asset_table_obj, asset.asset_rid, description)
    # Mirror the value into the in-memory Asset object only after the
    # catalog write returns successfully -- a write that raises must
    # not poison the in-memory copy.
    asset.description = description


def register(ctx: PluginContext) -> None:
    """Register all asset domain tools with the plugin context.

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
    async def deriva_ml_list_asset_tables(
        hostname: str,
        catalog_id: str,
    ) -> str:
        """List all asset tables in the catalog.

        An asset table is a file-backed catalog table (e.g. ``Image``,
        ``Trained_Model``, ``Execution_Metadata``). The result is the
        full set -- catalogs typically have only a handful of asset
        tables, so no pagination is offered.

        Args:

        Returns:
            JSON string ``{"asset_tables": [{"name", "schema"}, ...],
            "count": N}``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated
                from ``deriva_ml.DerivaML.list_asset_tables``.

        Example:
            ``{"asset_tables": [{"name": "Image", "schema":
            "demo-schema"}, {"name": "Execution_Metadata", "schema":
            "deriva-ml"}], "count": 2}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                payload = _list_asset_tables_impl(ml)
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="list_asset_tables",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_list_assets(
        hostname: str,
        catalog_id: str,
        asset_table: str,
        limit: int = 100,
        after_rid: str | None = None,
        preflight_count: bool = False,
    ) -> str:
        """List the rows in one asset table, with cursor pagination.

        See ``deriva_ml_getting_started`` (PAGINATION CONTRACT) for the two-step pagination flow.

        Args:
            asset_table: Name of the asset table to list (e.g.
                ``"Image"``, ``"Trained_Model"``).
            limit: Max assets per page (default 100, max 1000).
            after_rid: RID of last row from previous page to advance
                cursor.
            preflight_count: If True, return only the total count.

        Returns:
            Page: ``{"assets": [{"rid", "filename",
            "length", "md5", "asset_table", "asset_types"}, ...],
            "count", "truncated", "next_after_rid"}``. Preflight:
            ``{"total_count", "entities_fetched": False,
            "action_required"}``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated
                from ``deriva_ml.DerivaML.list_assets`` (e.g. unknown
                asset table or non-asset table).

        Example:
            ``{"assets": [{"rid": "1-AAAA", "filename": "scan.png",
            "length": 12345, "md5": "abc", "asset_table": "Image",
            "asset_types": ["Training_Data"]}], "count": 1,
            "truncated": false, "next_after_rid": null}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                if preflight_count:
                    total = len(list(ml.list_assets(asset_table)))
                    return json.dumps(
                        {
                            "total_count": total,
                            "entities_fetched": False,
                            "action_required": (
                                f"Found {total} assets in {asset_table}. "
                                "Choose a limit and call again with "
                                "preflight_count=False."
                            ),
                        }
                    )
                assets = sorted(
                    ml.list_assets(asset_table),
                    key=lambda a: getattr(a, "asset_rid", "") or "",
                )
                capped = min(max(limit, 0), _MAX_LIMIT)
                page, truncated, next_after = _paginate(
                    assets,
                    after_rid=after_rid,
                    limit=capped,
                    key=partial(_read_rid, rid_key="asset_rid"),
                )
            return AssetListResponse(
                assets=[AssetSummary(**_summarize_asset(a)) for a in page],
                count=len(page),
                truncated=truncated,
                next_after_rid=next_after,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="list_assets",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_lookup_asset(
        hostname: str,
        catalog_id: str,
        asset_rid: str,
    ) -> str:
        """Read the full bundled detail for one asset by RID.

        Returns the per-asset summary plus the list of executions
        associated with the asset (with role where the underlying API
        exposes it). Same shape as the
        ``deriva://catalog/{h}/{c}/ml/asset/{rid}`` resource -- the two
        share an internal helper so the payloads cannot drift.

        Args:
            asset_rid: The RID of the asset to look up.

        Returns:
            JSON string ``{"rid", "filename", "length", "md5", "url",
            "description", "asset_table", "asset_types", "executions":
            [{"rid", "asset_role"}, ...]}``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated
                from ``deriva_ml.DerivaML.lookup_asset`` (e.g. the RID
                is not an asset, or no such RID).

        Example:
            ``{"rid": "1-AAAA", "filename": "scan.png", "length":
            12345, "md5": "abc", "url": "/hatrac/...",
            "description": "MRI scan", "asset_table": "Image",
            "asset_types": ["Training_Data"], "executions": [{"rid":
            "1-EXEC", "asset_role": "Output"}]}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                payload = _get_asset_detail_impl(ml, asset_rid)
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="lookup_asset",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    # ------------------------------------------------------------------
    # Mutation tools. Each emits audit_event on success and routes
    # failures through _error_envelope.
    # ------------------------------------------------------------------

    @ctx.tool(mutates=True)
    async def deriva_ml_update_asset(
        hostname: str,
        catalog_id: str,
        asset_rid: str,
        asset_types: list[str] | None = None,
        description: str | None = None,
    ) -> str:
        """Update an asset's metadata (asset_type tags and/or description).

        Curation tool: pass only the fields you want to change. Both
        arguments are optional; at least one must be non-None.

        For ``asset_types``: set-style. Pass the desired final list of
        Asset_Type term names; the tool fetches the current types and
        computes the diff (add the terms in the new list that aren't
        in the current set; remove the terms in the current set that
        aren't in the new list). The two sets of mutations go through
        ``Asset.add_asset_type`` / ``Asset.remove_asset_type`` (one
        catalog round-trip per term).

        For ``description``: free-form text overwrite of the asset
        row's ``Description`` column, written through pathBuilder
        (deriva-ml's ``Asset`` class doesn't expose a catalog-write
        setter for description -- only ``Workflow`` and
        ``ExecutionRecord`` do).

        Args:
            asset_rid: The RID of the asset to update.
            asset_types: Desired final list of Asset_Type term names.
                ``None`` leaves types unchanged.
            description: New description text. ``None`` leaves the
                description unchanged.

        Returns:
            JSON string ``{"status": "updated", "asset_rid",
            "updated_fields": [...]}``. ``updated_fields`` is the list
            of field names actually written.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.lookup_asset``,
                ``Asset.add_asset_type`` / ``remove_asset_type``, or
                the Description pathBuilder write (e.g. unknown
                Asset_Type term, missing asset RID, read-only catalog).

        Example:
            ``{"status": "updated", "asset_rid": "1-AAAA",
            "updated_fields": ["asset_types", "description"]}``.
        """
        # Argument validation -- return errors directly without audit.
        if asset_types is None and description is None:
            return json.dumps(
                {"error": "at least one of asset_types or description must be provided"}
            )

        # Track partial progress for the failure path. asset_types is a
        # per-term loop (each add/remove is its own catalog round-trip),
        # so an LLM caller benefits from seeing what landed before the
        # failure -- mirrors update_dataset's partial-progress
        # response shape (the same diff-then-loop pattern).
        added_done: list[str] = []
        removed_done: list[str] = []
        updated_fields: list[str] = []
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                asset = ml.lookup_asset(asset_rid)

                if asset_types is not None:
                    desired = set(asset_types)
                    current = set(asset.asset_types or [])
                    to_add = sorted(desired - current)
                    to_remove = sorted(current - desired)
                    for term in to_add:
                        asset.add_asset_type(term)
                        added_done.append(term)
                    for term in to_remove:
                        asset.remove_asset_type(term)
                        removed_done.append(term)
                    updated_fields.append("asset_types")

                if description is not None:
                    _write_asset_description(ml, asset, description)
                    updated_fields.append("description")

            audit_event(
                "deriva_ml_update_asset",
                hostname=hostname,
                catalog_id=catalog_id,
                asset_rid=asset_rid,
                updated_fields=updated_fields,
                added=added_done,
                removed=removed_done,
            )
            return json.dumps(
                {
                    "status": "updated",
                    "asset_rid": asset_rid,
                    "updated_fields": updated_fields,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="update_asset",
                hostname=hostname,
                catalog_id=catalog_id,
                asset_rid=asset_rid,
                updated_fields=updated_fields,
                added_done=added_done,
                removed_done=removed_done,
                response_fields={
                    "asset_rid": asset_rid,
                    "added_done": added_done,
                    "removed_done": removed_done,
                    "updated_fields": updated_fields,
                },
            )
