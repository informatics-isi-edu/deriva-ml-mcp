"""Feature domain tools for deriva-ml-mcp.

Read tools: ``list_features``, ``get_feature``, ``list_feature_values``.
Mutation tools: ``create_feature``, ``delete_feature``,
``add_feature_values``.

Every tool wraps DERIVA I/O in ``with deriva_call():`` and routes errors
through ``_error_envelope`` (mutation tools also emit success/failure
audit events; reads only log on failure).

Pagination note: features have no first-class RID of their own
(unlike datasets/executions). ``list_features`` paginates by
``feature_table.name`` instead — the cursor (``after_rid``) is the
feature_table name from the previous page.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from deriva_mcp_core import deriva_call
from deriva_mcp_core.telemetry import audit_event
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
from deriva_ml_mcp.ml_context import get_ml

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def _summarize_feature(feature: Any) -> dict[str, Any]:
    """Render a Feature object into the JSON-friendly shape used by list endpoints.

    Args:
        feature: A ``deriva_ml.feature.Feature`` instance.

    Returns:
        Dict with ``feature_name``, ``target_table``, ``feature_table``,
        ``term_columns`` (names), ``asset_columns`` (names), and
        ``value_columns`` (names).
    """
    return {
        "feature_name": feature.feature_name,
        "target_table": feature.target_table.name,
        "feature_table": feature.feature_table.name,
        "term_columns": [c.name for c in feature.term_columns],
        "asset_columns": [c.name for c in feature.asset_columns],
        "value_columns": [c.name for c in feature.value_columns],
    }


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
    async def list_features(
        hostname: str,
        catalog_id: str,
        table: str | None = None,
        limit: int = 100,
        after_rid: str | None = None,
        preflight_count: bool = False,
    ) -> str:
        """Discover features defined on a table (or across the catalog).

        PAGINATION: When the count is unknown, call with
        ``preflight_count=True`` first. Then choose a limit and call
        again with ``preflight_count=False``. Use ``after_rid`` (the
        ``feature_table`` name from the previous page) to advance.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            table: If set, filter to features on this target table. If
                None, return all features in the catalog.
            limit: Max features per page (default 100, max 1000).
            after_rid: Last ``feature_table`` name from the previous
                page; rows with ``feature_table <= after_rid`` are
                skipped.
            preflight_count: If True, return only the total count.

        Returns:
            JSON string. Page: ``{"features": [{"feature_name",
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
                ml = get_ml(hostname, catalog_id)
                features = sorted(
                    ml.find_features(table=table),
                    key=lambda f: f.feature_table.name,
                )

            if preflight_count:
                total = len(features)
                return json.dumps(
                    {
                        "total_count": total,
                        "entities_fetched": False,
                        "action_required": (
                            f"Found {total} features. Choose a limit and "
                            "call again with preflight_count=False."
                        ),
                    }
                )

            capped = min(max(limit, 0), _MAX_LIMIT)
            page, truncated, next_after = _paginate(
                features,
                after_rid=after_rid,
                limit=capped,
                key=lambda f: f.feature_table.name,
            )
            return json.dumps(
                {
                    "features": [_summarize_feature(f) for f in page],
                    "count": len(page),
                    "truncated": truncated,
                    "next_after_rid": next_after,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="list_features",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def get_feature(
        hostname: str,
        catalog_id: str,
        table: str,
        feature_name: str,
    ) -> str:
        """Read the full schema of one feature for building value records.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            table: Target table the feature is defined on.
            feature_name: Name of the feature to inspect.

        Returns:
            JSON string. ``{"feature_name", "target_table",
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
                ml = get_ml(hostname, catalog_id)
                feature = ml.lookup_feature(table, feature_name)
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
    async def list_feature_values(
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

        PAGINATION: When the count is unknown, call with
        ``preflight_count=True`` first. Then choose a limit and call
        again with ``preflight_count=False``. Use ``after_rid`` (the
        target RID from the previous page) to advance.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
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

        Returns:
            JSON string. Page: ``{"records": [<model_dump>, ...], "count",
            "truncated", "next_after_rid"}``. Preflight:
            ``{"total_count", "entities_fetched": False,
            "action_required"}``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated from
                ``feature_values``.

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
                ml = get_ml(hostname, catalog_id)
                # by_workflow needs a container; build it now that we have ml.
                if selector_fn == "__by_workflow__":
                    container = ml.lookup_dataset(dataset_rid) if dataset_rid else ml
                    selector_fn = FeatureRecord.select_by_workflow(
                        selector_workflow, container=container
                    )

                if dataset_rid is not None:
                    ds = ml.lookup_dataset(dataset_rid)
                    records = list(ds.feature_values(table, feature_name, selector=selector_fn))
                else:
                    records = list(ml.feature_values(table, feature_name, selector=selector_fn))

            if preflight_count:
                total = len(records)
                return json.dumps(
                    {
                        "total_count": total,
                        "entities_fetched": False,
                        "action_required": (
                            f"Found {total} feature records. Choose a "
                            "limit and call again with "
                            "preflight_count=False."
                        ),
                    }
                )

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
            return json.dumps(
                {
                    "records": [r.model_dump() for r in page],
                    "count": len(page),
                    "truncated": truncated,
                    "next_after_rid": next_after,
                },
                default=str,
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
    async def create_feature(
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
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
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
                ml = get_ml(hostname, catalog_id)
                ml.create_feature(
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
                feature = ml.lookup_feature(target_table, feature_name)
                summary = _summarize_feature(feature)
            audit_event(
                "deriva_ml_create_feature",
                hostname=hostname,
                catalog_id=catalog_id,
                target_table=target_table,
                feature_name=feature_name,
                n_term_cols=len(terms_list),
                n_asset_cols=len(assets_list),
                n_value_cols=len(summary["value_columns"]),
            )
            return json.dumps({"status": "created", **summary})
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
    async def delete_feature(
        hostname: str,
        catalog_id: str,
        table: str,
        feature_name: str,
    ) -> str:
        """Remove a feature definition and all its values.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
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
                ml = get_ml(hostname, catalog_id)
                was_deleted = ml.delete_feature(table, feature_name)
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
            return json.dumps(
                {
                    "status": status,
                    "feature_name": feature_name,
                    "table": table,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="delete_feature",
                hostname=hostname,
                catalog_id=catalog_id,
                target_table=table,
                feature_name=feature_name,
            )

    @ctx.tool(mutates=True)
    async def add_feature_values(
        hostname: str,
        catalog_id: str,
        table: str,
        feature_name: str,
        execution_rid: str,
        entries: list[dict[str, Any]],
    ) -> str:
        """Insert label / score / asset values for a batch of target records.

        Each entry dict contains the target table's column key (e.g.
        ``"Image": "1-AAAA"``) plus the feature's term/asset/value
        columns. The execution attributes provenance to a specific
        execution -- use the execution domain tools to create one if
        needed.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            table: Target table the feature is defined on.
            feature_name: Name of the feature to write values for.
            execution_rid: RID of the parent execution.
            entries: Non-empty list of per-target value dicts. Each dict
                is built into a ``FeatureRecord`` via the feature's
                Pydantic model class.

        Returns:
            JSON string ``{"status": "added", "feature_name",
            "execution_rid", "count": <added>}``.

        Raises:
            ValueError: Wrapped via ``_error_envelope`` if ``entries`` is
                empty.
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.feature.Feature.feature_record_class()``
                construction or ``Execution.add_features``.

        Example:
            ``{"status": "added", "feature_name": "Quality",
            "execution_rid": "EXEC-1", "count": 2}``.
        """
        if not entries:
            # Include attempted_count for response-shape parity with the
            # other failure paths (per-record build failure and upstream
            # failure both surface attempted_count via _error_envelope's
            # response_fields). Lets callers read the field unconditionally.
            return json.dumps({"error": "entries must be a non-empty list.", "attempted_count": 0})

        # Track which entry index failed so the LLM can pinpoint the
        # offending row in the bulk request.
        failed_index: int | None = None
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                feature = ml.lookup_feature(table, feature_name)
                feature_class = feature.feature_record_class()

                records = []
                for i, entry in enumerate(entries):
                    failed_index = i
                    records.append(feature_class(**entry))
                failed_index = None

                execution = ml.resume_execution(execution_rid)
                with execution.execute():
                    added = execution.add_features(records)
            audit_event(
                "deriva_ml_add_feature_values",
                hostname=hostname,
                catalog_id=catalog_id,
                target_table=table,
                feature_name=feature_name,
                execution_rid=execution_rid,
                added_count=added,
            )
            return json.dumps(
                {
                    "status": "added",
                    "feature_name": feature_name,
                    "execution_rid": execution_rid,
                    "count": added,
                }
            )
        except Exception as exc:
            response_fields: dict[str, Any] = {"attempted_count": len(entries)}
            if failed_index is not None:
                response_fields["failed_entry_index"] = failed_index
            return _error_envelope(
                exc,
                operation="add_feature_values",
                hostname=hostname,
                catalog_id=catalog_id,
                target_table=table,
                feature_name=feature_name,
                execution_rid=execution_rid,
                response_fields=response_fields,
            )
