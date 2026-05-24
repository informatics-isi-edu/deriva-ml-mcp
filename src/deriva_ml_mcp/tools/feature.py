"""Feature domain tools for deriva-ml-mcp.

Read tools: ``deriva_ml_list_features``, ``deriva_ml_get_feature``, ``deriva_ml_list_feature_values``.
Mutation tools: ``deriva_ml_create_feature``, ``deriva_ml_delete_feature``.

v0.5.0 removed ``deriva_ml_add_feature_values`` -- writing feature
VALUES requires the per-process SQLite manifest store that's owned by
the local Python execution context (see CLAUDE.md "Stateless /
bounded-resource rule"). Feature DEFINITIONS (schema creation /
deletion) stay on MCP because they're pure catalog-state operations.

Every tool wraps DERIVA I/O in ``with deriva_call():`` and routes errors
through ``_error_envelope`` (mutation tools also emit success/failure
audit events; reads only log on failure).

Pagination note: features have no first-class RID of their own
(unlike datasets/executions). ``deriva_ml_list_features`` paginates by
``feature_table.name`` instead — the cursor (``after_rid``) is the
feature_table name from the previous page.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Literal

from deriva_mcp_core import deriva_call
from deriva_mcp_core.telemetry import audit_event
from deriva_ml import DerivaMLMaterializeLimitExceeded
from deriva_ml.feature import FeatureRecord

# Note on testing audit_event: see `make_patch_audit("feature")` in
# `tests/conftest.py` (the canonical factory; bound per-file as
# `_patch_feature_audit = make_patch_audit("feature")`). Single-patch
# facade is impossible due to Python's `from X import name` import
# binding semantics — tests must patch BOTH
# `deriva_ml_mcp.tools.feature.audit_event` (this module's success-path
# emission) and `deriva_ml_mcp._helpers.audit_event` (the failure-path
# emission inside `_error_envelope`).
from deriva_ml_mcp._helpers import (
    _MAX_LIMIT,
    _error_envelope,
    _paginate,
)
from deriva_ml_mcp._response_models import (
    CreateFeatureResponse,
    DeleteFeatureResponse,
    FeatureListResponse,
    FeatureSummary,
    FeatureValueListResponse,
    FeatureValueRecord,
    PreflightCountResponse,
)
from deriva_ml_mcp.ml_context import get_ml

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def _summarize_feature(feature: Any) -> FeatureSummary:
    """Render a Feature object into the validated summary used by list endpoints.

    Args:
        feature: A ``deriva_ml.feature.Feature`` instance.

    Returns:
        ``FeatureSummary`` Pydantic instance -- see
        ``deriva_ml_mcp._response_models``.

    Note:
        v2.2 sweep: this helper now returns Pydantic. Consumers that
        previously dict-accessed must use attribute access or call
        ``.model_dump()`` to get a plain dict back.
    """
    return FeatureSummary(
        feature_name=feature.feature_name,
        target_table=feature.target_table.name,
        feature_table=feature.feature_table.name,
        term_columns=[c.name for c in feature.term_columns],
        asset_columns=[c.name for c in feature.asset_columns],
        value_columns=[c.name for c in feature.value_columns],
    )


def _list_features_impl(
    ml: Any,
    *,
    table: str | None,
    after_rid: str | None,
    limit: int,
) -> FeatureListResponse:
    """Fetch + paginate features. Pure helper -- shared by tool and resource.

    Pagination key is ``feature_table.name`` (features have no first-class
    RID); ``after_rid`` is the last feature_table name from the previous
    page.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        table: Optional target-table filter (passes to ``find_features``).
        after_rid: Cursor (last ``feature_table`` name from prior page).
        limit: Max features per page (already capped by caller).

    Returns:
        ``FeatureListResponse`` -- see ``deriva_ml_mcp._response_models``.
    """
    features = sorted(
        ml.find_features(table=table),
        key=lambda f: f.feature_table.name,
    )
    page, truncated, next_after = _paginate(
        features,
        after_rid=after_rid,
        limit=limit,
        key=lambda f: f.feature_table.name,
    )
    return FeatureListResponse(
        features=[_summarize_feature(f) for f in page],
        count=len(page),
        truncated=truncated,
        next_after_rid=next_after,
    )


def register(ctx: PluginContext) -> None:
    """Register all feature domain tools with the plugin context.

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
    async def deriva_ml_list_features(
        hostname: str,
        catalog_id: str,
        table: str | None = None,
        limit: int = 100,
        after_rid: str | None = None,
        preflight_count: bool = False,
    ) -> str:
        """Discover features defined on a table (or across the catalog).

        See ``deriva_ml_getting_started`` (PAGINATION CONTRACT) for the two-step pagination flow.

        Args:
            table: If set, filter to features on this target table. If
                None, return all features in the catalog.
            limit: Max features per page (default 100, max 1000).
            after_rid: Last ``feature_table`` name from the previous
                page; rows with ``feature_table <= after_rid`` are
                skipped.
            preflight_count: If True, return only the total count.

        Returns:
            Page: ``{"features": [{"feature_name",
            "target_table", "feature_table", "term_columns",
            "asset_columns", "value_columns"}, ...], "count",
            "truncated", "next_after_rid"}``. Preflight:
            ``{"total_count", "entities_fetched": False,
            "action_required"}``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated from
                ``deriva_ml.DerivaML.find_features``.

        Example:
            ``{"features": [{"feature_name": "Quality", "target_table":
            "Image", "feature_table": "Execution_Image_Quality",
            "term_columns": ["Quality_Type"], "asset_columns": [],
            "value_columns": []}], "count": 1, "truncated": false,
            "next_after_rid": null}``
        """
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``ml.find_features``
                # returns a generator that materializes over the wire, so
                # the drain (list/sorted) must happen inside the worker
                # thread, not on the event loop. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
                if preflight_count:

                    def _count_features() -> int:
                        return len(list(ml.find_features(table=table)))

                    total = await asyncio.to_thread(_count_features)
                    return PreflightCountResponse(
                        total_count=total,
                        action_required=(
                            f"Found {total} features. Choose a limit and "
                            "call again with preflight_count=False."
                        ),
                    ).model_dump_json(by_alias=True)

                capped = min(max(limit, 0), _MAX_LIMIT)
                payload = await asyncio.to_thread(
                    _list_features_impl,
                    ml,
                    table=table,
                    after_rid=after_rid,
                    limit=capped,
                )
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="list_features",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_get_feature(
        hostname: str,
        catalog_id: str,
        table: str,
        feature_name: str,
    ) -> str:
        """Read the full schema of one feature for building value records.

        Args:
            table: Target table the feature is defined on.
            feature_name: Name of the feature to inspect.

        Returns:
            ``{"feature_name", "target_table",
            "feature_table", "comment", "term_columns": [{"name",
            "nullok"}, ...], "asset_columns": [{"name", "nullok"}, ...],
            "value_columns": [{"name", "type", "nullok", "default"},
            ...]}``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated from
                ``deriva_ml.DerivaML.lookup_feature`` (e.g. unknown
                feature).

        Example:
            ``{"feature_name": "Quality", "target_table": "Image",
            "feature_table": "Execution_Image_Quality", "comment": "...",
            "term_columns": [{"name": "Quality_Type", "nullok": false}],
            "asset_columns": [], "value_columns": [{"name": "score",
            "type": "float4", "nullok": true, "default": null}]}``.
        """
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
                feature = await asyncio.to_thread(ml.lookup_feature, table, feature_name)
                payload = {
                    "feature_name": feature.feature_name,
                    "target_table": feature.target_table.name,
                    "feature_table": feature.feature_table.name,
                    "comment": getattr(feature.feature_table, "comment", "") or "",
                    "term_columns": [
                        {"name": c.name, "nullok": c.nullok} for c in feature.term_columns
                    ],
                    "asset_columns": [
                        {"name": c.name, "nullok": c.nullok} for c in feature.asset_columns
                    ],
                    "value_columns": [
                        {
                            "name": c.name,
                            "type": c.type.typename,
                            "nullok": c.nullok,
                            "default": c.default,
                        }
                        for c in feature.value_columns
                    ],
                }
            return json.dumps(payload)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="get_feature",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_list_feature_values(
        hostname: str,
        catalog_id: str,
        table: str,
        feature_name: str,
        selector: Literal[
            "none",
            "newest",
            "first",
            "latest",
            "majority_vote",
            "by_workflow",
            "by_execution",
        ] = "none",
        selector_workflow: str | None = None,
        selector_execution_rid: str | None = None,
        dataset_rid: str | None = None,
        limit: int = 100,
        after_rid: str | None = None,
        preflight_count: bool = False,
        execution_rids: list[str] | None = None,
        max_results: int = 50_000,
    ) -> str:
        """Query feature values for a (table, feature_name).

        Optionally collapses values via a selector and/or scopes the
        query to a single dataset's members.

        Selectors:

        - ``none`` -- return all matching records (multiple per target).
        - ``newest`` / ``latest`` / ``first`` -- time-based collapse via
          the corresponding ``FeatureRecord.select_*`` staticmethod.
        - ``majority_vote`` -- collapse by most common value.
        - ``by_workflow`` -- filter to records from one workflow.
          Requires ``selector_workflow``.
        - ``by_execution`` -- filter to records from one execution.
          Requires ``selector_execution_rid``.

        See ``deriva_ml_getting_started`` (PAGINATION CONTRACT) for the two-step pagination flow.

        Args:
            table: Target table the feature is defined on.
            feature_name: Name of the feature to query.
            selector: Selector strategy. See above for semantics.
            selector_workflow: Workflow name (required for
                ``selector="by_workflow"``).
            selector_execution_rid: Execution RID (required for
                ``selector="by_execution"``).
            dataset_rid: If set, scope the query to one dataset's
                members.
            limit: Max records per page (default 100, max 1000).
            after_rid: Last RID from previous page.
            preflight_count: If True, return only the materialized count.
            execution_rids: Optional filter -- when set, only feature
                rows whose ``Execution`` value is in this list are
                materialized. Forwarded server-side to deriva-ml's
                ``feature_values(execution_rids=...)``. Lets compare-
                runs workflows fetch values across N executions in a
                single catalog round-trip rather than N sequential
                queries. Empty list short-circuits to no rows.
            max_results: Cap on rows materialized into memory before
                pagination (default 50000). Returns
                ``{"error": "..."}`` if the result set exceeds the cap;
                the error message names the cap and suggests passing
                ``execution_rids=`` to narrow. Increase if you have a
                catalog where 50K is too tight; decrease if the LLM
                runs in a memory-constrained environment.

        Returns:
            Page: ``{"records": [<model_dump>, ...], "count",
            "truncated", "next_after_rid"}``. Preflight:
            ``{"total_count", "entities_fetched": False,
            "action_required"}``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated from
                ``feature_values``.
            DerivaMLMaterializeLimitExceeded: When the result set
                exceeds ``max_results``. Translated to
                ``{"error": "..."}`` on the wire.

        Example:
            ``{"records": [{"RID": "1-AAAA", "Quality_Type": "good", ...}],
            "count": 1, "truncated": false, "next_after_rid": null}``.
        """
        # Build the selector callable from the literal arg. Validation
        # for required follow-on args happens in this same block.
        selector_fn: Any
        if selector == "none":
            selector_fn = None
        elif selector == "newest":
            selector_fn = FeatureRecord.select_newest
        elif selector == "first":
            selector_fn = FeatureRecord.select_first
        elif selector == "latest":
            selector_fn = FeatureRecord.select_latest
        elif selector == "majority_vote":
            selector_fn = FeatureRecord.select_majority_vote()
        elif selector == "by_workflow":
            if not selector_workflow:
                return json.dumps(
                    {"error": ("selector='by_workflow' requires selector_workflow to be set.")}
                )
            # Container is set after we have ml below; build a lazy
            # placeholder and resolve inside the deriva_call block.
            selector_fn = "__by_workflow__"
        elif selector == "by_execution":
            if not selector_execution_rid:
                return json.dumps(
                    {
                        "error": (
                            "selector='by_execution' requires selector_execution_rid to be set."
                        )
                    }
                )
            selector_fn = FeatureRecord.select_by_execution(selector_execution_rid)
        else:
            return json.dumps({"error": f"unknown selector '{selector}'"})

        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``feature_values``
                # returns a generator that materializes over the wire, so
                # the drain (list) must happen inside the worker thread,
                # not on the event loop. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
                # by_workflow needs a container; build it now that we have ml.
                if selector_fn == "__by_workflow__":
                    container = (
                        await asyncio.to_thread(ml.lookup_dataset, dataset_rid)
                        if dataset_rid
                        else ml
                    )
                    selector_fn = FeatureRecord.select_by_workflow(
                        selector_workflow, container=container
                    )

                if dataset_rid is not None:
                    ds = await asyncio.to_thread(ml.lookup_dataset, dataset_rid)

                    def _drain_ds_feature_values() -> list[Any]:
                        return list(
                            ds.feature_values(
                                table,
                                feature_name,
                                selector=selector_fn,
                                materialize_limit=max_results,
                                execution_rids=execution_rids,
                            )
                        )

                    records = await asyncio.to_thread(_drain_ds_feature_values)
                else:

                    def _drain_ml_feature_values() -> list[Any]:
                        return list(
                            ml.feature_values(
                                table,
                                feature_name,
                                selector=selector_fn,
                                materialize_limit=max_results,
                                execution_rids=execution_rids,
                            )
                        )

                    records = await asyncio.to_thread(_drain_ml_feature_values)

            if preflight_count:
                total = len(records)
                return PreflightCountResponse(
                    total_count=total,
                    action_required=(
                        f"Found {total} feature records. Choose a "
                        "limit and call again with "
                        "preflight_count=False."
                    ),
                ).model_dump_json(by_alias=True)

            # Sort by RID for stable pagination. Records are pydantic
            # FeatureRecord instances; their RID lives at the .RID
            # attribute (catalog convention).
            sorted_records = sorted(records, key=lambda r: getattr(r, "RID", "") or "")
            capped = min(max(limit, 0), _MAX_LIMIT)
            page, truncated, next_after = _paginate(
                sorted_records,
                after_rid=after_rid,
                limit=capped,
                key=lambda r: getattr(r, "RID", "") or "",
            )
            # Validate each record into FeatureValueRecord. The model
            # declares the four stable provenance fields (RID, Execution,
            # Feature_Name, RCT) and allows the feature's dynamic
            # value / term / asset columns through under extra="allow".
            # r.model_dump() preserves the same dict shape we previously
            # emitted via json.dumps; constructing the model just
            # type-checks the four stable fields and pins the wire
            # contract for downstream consumers.
            return FeatureValueListResponse(
                records=[FeatureValueRecord(**r.model_dump()) for r in page],
                count=len(page),
                truncated=truncated,
                next_after_rid=next_after,
            ).model_dump_json(by_alias=True)
        except DerivaMLMaterializeLimitExceeded as exc:
            # Translate the deriva-ml exception to a clear error envelope.
            # The exception's __str__ already names the cap and suggests
            # passing execution_rids=...; we just route it through the
            # standard _error_envelope so audit/logging are uniform.
            return _error_envelope(
                exc,
                operation="list_feature_values",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="list_feature_values",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    # ------------------------------------------------------------------
    # Mutation tools. Each emits audit_event on success and routes
    # failures through _error_envelope.
    # ------------------------------------------------------------------

    @ctx.tool(mutates=True)
    async def deriva_ml_create_feature(
        hostname: str,
        catalog_id: str,
        target_table: str,
        feature_name: str,
        terms: list[str] | None = None,
        assets: list[str] | None = None,
        metadata: list[str | dict] | None = None,
        optional: list[str] | None = None,
        comment: str = "",
    ) -> str:
        """Register a new feature schema on a domain table.

        Args:
            target_table: Domain table the feature attaches to.
            feature_name: Unique name within the target table.
            terms: Vocabulary tables whose terms can be feature values.
            assets: Asset tables that can be referenced.
            metadata: Extra metadata columns/tables/keys.
            optional: Column names that may be omitted at insert.
            comment: Free-text description of the feature.

        Returns:
            JSON string ``{"status": "created", "feature_name",
            "target_table", "feature_table", "term_columns",
            "asset_columns", "value_columns"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.create_feature`` (e.g. invalid term
                or asset table).

        Example:
            ``{"status": "created", "feature_name": "Quality",
            "target_table": "Image", "feature_table":
            "Execution_Image_Quality", "term_columns": ["Quality_Type"],
            "asset_columns": [], "value_columns": []}``.
        """
        terms_list = list(terms or [])
        assets_list = list(assets or [])
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
                await asyncio.to_thread(
                    ml.create_feature,
                    target_table,
                    feature_name,
                    terms=terms_list,
                    assets=assets_list,
                    metadata=list(metadata) if metadata else None,
                    optional=list(optional) if optional else None,
                    comment=comment,
                )
                # create_feature returns the FeatureRecord subclass, not
                # the Feature schema. Re-look up the Feature for the
                # response shape.
                feature = await asyncio.to_thread(
                    ml.lookup_feature, target_table, feature_name
                )
                summary = _summarize_feature(feature)
            audit_event(
                "deriva_ml_create_feature",
                hostname=hostname,
                catalog_id=catalog_id,
                target_table=target_table,
                feature_name=feature_name,
                n_term_cols=len(terms_list),
                n_asset_cols=len(assets_list),
                n_value_cols=len(summary.value_columns),
            )
            return CreateFeatureResponse(
                status="created",
                **summary.model_dump(),
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="create_feature",
                hostname=hostname,
                catalog_id=catalog_id,
                target_table=target_table,
                feature_name=feature_name,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_delete_feature(
        hostname: str,
        catalog_id: str,
        table: str,
        feature_name: str,
    ) -> str:
        """Remove a feature definition and all its values.

        Args:
            table: Target table the feature is defined on.
            feature_name: Name of the feature to delete.

        Returns:
            JSON string ``{"status": "deleted" | "not_found",
            "feature_name", "table"}``. Audit fires only when the
            feature actually existed and was deleted.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.delete_feature`` (e.g. permission
                denied).

        Example:
            ``{"status": "deleted", "feature_name": "Quality",
            "table": "Image"}``.
        """
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
                was_deleted = await asyncio.to_thread(ml.delete_feature, table, feature_name)
            if was_deleted:
                audit_event(
                    "deriva_ml_delete_feature",
                    hostname=hostname,
                    catalog_id=catalog_id,
                    target_table=table,
                    feature_name=feature_name,
                )
                status = "deleted"
            else:
                # No state changed -- skip audit per convention.
                status = "not_found"
            return DeleteFeatureResponse(
                status=status,  # "deleted" or "not_found"
                feature_name=feature_name,
                table=table,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="delete_feature",
                hostname=hostname,
                catalog_id=catalog_id,
                target_table=table,
                feature_name=feature_name,
            )

    # NOTE: deriva_ml_add_feature_values was removed in v0.5.0 per the
    # stateless / bounded-resource rule (CLAUDE.md,
    # docs/audit-2026-05-23.md). Feature staging writes to a per-process
    # SQLite manifest store; a server-side stage cannot pair with a
    # user-side commit, so the entire execution lifecycle (create /
    # start / commit / abort / update / add_feature_values /
    # create_execution_dataset / add_nested_execution) is now owned by
    # the caller's local Python. Recover from git history if you need
    # the prior implementation; the audit doc has the architectural
    # rationale.
