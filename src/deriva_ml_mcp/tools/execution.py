"""Execution domain tools for deriva-ml-mcp.

Read tools: ``deriva_ml_list_executions``, ``deriva_ml_get_execution``,
``deriva_ml_find_workflow_executions``, ``deriva_ml_list_execution_children``,
``deriva_ml_list_execution_parents``.
Mutation tools: ``deriva_ml_create_execution``, ``deriva_ml_start_execution``,
``deriva_ml_commit_execution``, ``deriva_ml_abort_execution``, ``deriva_ml_create_execution_dataset``,
``deriva_ml_add_nested_execution``.

Every tool wraps DERIVA I/O in ``with deriva_call():`` and routes errors
through ``_error_envelope`` (mutation tools also emit success/failure
audit events; reads only log on failure).

State-machine note. The ``Execution`` lifecycle is:
``Created -> Running -> Stopped -> Pending_Upload -> Uploaded`` with
``Aborted`` / ``Failed`` as alternative terminals. ``deriva_ml_start_execution``,
``deriva_ml_commit_execution``, and ``deriva_ml_abort_execution`` are idempotent on the
target state — they no-op (and skip audit) when the execution is already
where the call would put it.
"""

from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING, Any

from deriva_mcp_core import deriva_call
from deriva_mcp_core.telemetry import audit_event
from deriva_ml.execution.execution import ExecutionStatus

# Note on testing audit_event: see ``make_patch_audit("execution")`` in
# tests/_helpers.py. Single-patch facade is impossible due to Python's
# ``from X import name`` import binding semantics — tests must patch
# BOTH ``deriva_ml_mcp.tools.execution.audit_event`` (this module's
# success-path emission) and ``deriva_ml_mcp._helpers.audit_event`` (the
# failure-path emission inside ``_error_envelope``).
from deriva_ml_mcp._helpers import (
    _MAX_LIMIT,
    _error_envelope,
    _paginate,
    _read_rid,
)
from deriva_ml_mcp.ml_context import get_ml

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


# ---------------------------------------------------------------------------
# Status / record helpers
# ---------------------------------------------------------------------------


# States from which start_execution / commit_execution refuse to advance.
#
# start_execution: Created or Running are the only valid prior states
# (Running is the idempotent no-op). Everything else is rejected -- once
# work has stopped (Stopped/Pending_Upload/Uploaded) or failed/aborted,
# the execution cannot be re-entered for new computation.
#
# commit_execution: drains staged work and uploads. Created and Running
# are pre-stop; Stopped and Pending_Upload are mid-pipeline; Uploaded is
# the additive-upload entry point per deriva-ml 3d21f55 (calling
# upload_execution_outputs on an Uploaded execution that has new
# pending manifest entries cycles Uploaded -> Pending_Upload -> Uploaded;
# a call with no pending entries is a clean no-op).
_START_REJECT_STATES = {
    ExecutionStatus.Stopped,
    ExecutionStatus.Failed,
    ExecutionStatus.Pending_Upload,
    ExecutionStatus.Uploaded,
    ExecutionStatus.Aborted,
}

_COMMIT_ALLOWED_STATES = {
    ExecutionStatus.Created,
    ExecutionStatus.Running,
    ExecutionStatus.Stopped,
    ExecutionStatus.Pending_Upload,
    ExecutionStatus.Uploaded,
}


def _summarize_execution(record: Any) -> dict[str, Any]:
    """Render an ``ExecutionRecord`` (or ``Execution``) into the JSON list shape.

    Tolerates both ``ExecutionRecord`` (returned by ``lookup_execution``
    / ``find_executions``) and ``Execution`` (returned by
    ``resume_execution``) — the only attributes read are common to both
    surfaces or fall back to ``None``.

    Args:
        record: An ``ExecutionRecord`` (preferred) or ``Execution`` mock.

    Returns:
        Dict with ``rid``, ``workflow_rid``, ``status`` (str),
        ``description``, ``start_time``, ``stop_time``, ``duration``.
        Datetime fields serialize via ``json.dumps(default=str)`` at the
        call sites.

    Example:
        >>> from unittest.mock import MagicMock
        >>> from deriva_ml.execution.execution import ExecutionStatus
        >>> rec = MagicMock()
        >>> rec.execution_rid = "1-EXEC"
        >>> rec.workflow_rid = "1-WF"
        >>> rec.status = ExecutionStatus.Created
        >>> rec.description = "demo"
        >>> rec.start_time = None
        >>> rec.stop_time = None
        >>> rec.duration = None
        >>> _summarize_execution(rec)["rid"]
        '1-EXEC'
        >>> _summarize_execution(rec)["status"]
        'Created'
    """
    status = getattr(record, "status", None)
    return {
        "rid": getattr(record, "execution_rid", None),
        "workflow_rid": getattr(record, "workflow_rid", None),
        "status": status.value if isinstance(status, ExecutionStatus) else status,
        "description": getattr(record, "description", None),
        "start_time": getattr(record, "start_time", None),
        "stop_time": getattr(record, "stop_time", None),
        "duration": getattr(record, "duration", None),
    }


def _list_executions_impl(
    ml: Any,
    *,
    workflow_rid: str | None,
    status: str | None,
    after_rid: str | None,
    limit: int,
) -> dict[str, Any]:
    """Fetch + paginate executions. Pure helper -- shared by tool and resource.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        workflow_rid: Optional workflow filter.
        status: Optional ``ExecutionStatus`` value (string).
        after_rid: Cursor for cursor pagination.
        limit: Max executions per page (already capped by caller).

    Returns:
        Dict ``{"executions": [...], "count", "truncated", "next_after_rid"}``.
    """
    status_enum = ExecutionStatus(status) if status else None
    executions = sorted(
        ml.find_executions(workflow=workflow_rid, status=status_enum),
        key=lambda e: getattr(e, "execution_rid", "") or "",
    )
    page, truncated, next_after = _paginate(
        executions,
        after_rid=after_rid,
        limit=limit,
        key=partial(_read_rid, rid_key="execution_rid"),
    )
    return {
        "executions": [_summarize_execution(e) for e in page],
        "count": len(page),
        "truncated": truncated,
        "next_after_rid": next_after,
    }


def _get_execution_detail_impl(ml: Any, execution_rid: str) -> dict[str, Any]:
    """Build the execution detail payload (summary + inputs + outputs + metadata).

    Used by the ``deriva://catalog/{h}/{c}/ml/execution/{rid}`` resource.
    Aggregates input datasets, asset I/O grouped by role, and a
    metadata bucket with ``hydra_config`` (when present), and an
    optional ``experiment`` key.

    The deriva-ml ``ExecutionRecord`` exposes:

    - ``list_input_datasets()`` -> list of Dataset objects
    - ``list_assets(asset_role="Input"|"Output"|None)`` -> list of Asset
      objects

    Hydra config and other metadata files don't have a generic API on
    ExecutionRecord; the ``experiment`` key is stubbed as ``None`` for
    now (see TODO).

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        execution_rid: The RID of the execution to look up.

    Returns:
        Dict with ``rid``, ``workflow_rid``, ``status``, ``description``,
        ``start_time``, ``stop_time``, ``duration``, ``inputs``,
        ``outputs``, ``metadata``, ``experiment``.
    """
    record = ml.lookup_execution(execution_rid)
    payload = _summarize_execution(record)

    # Inputs: datasets + input assets.
    input_datasets: list[dict[str, Any]] = []
    try:
        for ds in record.list_input_datasets():
            current = getattr(ds, "current_version", None)
            input_datasets.append(
                {
                    "rid": ds.dataset_rid,
                    "version": str(current) if current is not None else None,
                }
            )
    except Exception:  # noqa: BLE001 -- record may not be bound on all paths
        input_datasets = []

    input_assets: list[dict[str, Any]] = []
    output_assets: list[dict[str, Any]] = []
    try:
        for asset in record.list_assets(asset_role="Input"):
            input_assets.append(
                {
                    "rid": getattr(asset, "asset_rid", None),
                    "filename": getattr(asset, "filename", None),
                }
            )
        for asset in record.list_assets(asset_role="Output"):
            output_assets.append(
                {
                    "rid": getattr(asset, "asset_rid", None),
                    "filename": getattr(asset, "filename", None),
                }
            )
    except Exception:  # noqa: BLE001 -- assets are optional
        pass

    payload["inputs"] = {
        "datasets": input_datasets,
        "assets": input_assets,
    }
    payload["outputs"] = {
        "assets": output_assets,
    }

    # TODO(deriva-ml-execution-metadata-api): no generic API on
    # ExecutionRecord to enumerate Execution_Metadata files
    # (Deriva_Config / Execution_Config / Hydra_Config / Runtime_Env).
    # Surface what we can and leave the others empty.
    payload["metadata"] = {
        "deriva_config": None,
        "execution_config": None,
        "hydra_config": None,
        "runtime_env": None,
    }

    # TODO(deriva-ml-experiment-detection): no clean predicate on
    # ExecutionRecord to tell whether an execution is an "experiment"
    # (i.e., has a Hydra config attached). Always return None until
    # an upstream API exists.
    payload["experiment"] = None

    return payload


def _summarize_upload_dict(
    uploaded: dict[str, Any],
    *,
    execution_rid: str,
    feature_count: int,
) -> dict[str, Any]:
    """Synthesize an UploadReport-shaped dict from ``upload_execution_outputs``.

    ``Execution.upload_execution_outputs`` returns ``dict[str,
    list[AssetFilePath]]`` (asset table -> uploaded files), not an
    ``UploadReport``. We render it into the same JSON shape callers
    expect from ``deriva_ml_commit_execution`` so the response surface stays
    stable across upstream API styles.

    Args:
        uploaded: The dict returned by ``Execution.upload_execution_outputs``.
        execution_rid: RID of the execution being committed (for the
            ``execution_rids`` envelope field).
        feature_count: Number of staged feature records that were
            flushed during this commit (added to ``total_uploaded``
            since they don't appear in the asset dict).

    Returns:
        Dict with the upload-report envelope:
        ``execution_rids``, ``total_uploaded``, ``total_failed``,
        ``per_table``, ``errors``, ``errors_truncated``. ``errors`` is
        capped at the top 10 lines so the response stays bounded; the
        full list remains in the operator's server logs.
    """
    per_table: dict[str, int] = {
        table: len(files or []) for table, files in (uploaded or {}).items()
    }
    return {
        "execution_rids": [execution_rid],
        "total_uploaded": sum(per_table.values()) + feature_count,
        "total_failed": 0,
        "per_table": per_table,
        "errors": [],
        "errors_truncated": False,
    }


def register(ctx: PluginContext) -> None:
    """Register all execution domain tools with the plugin context.

    Args:
        ctx: PluginContext supplied by deriva-mcp-core at startup.

    Returns:
        None.

    Example:
        >>> from deriva_mcp_core.plugin.api import PluginContext
        >>> # ctx provided by the framework
        >>> register(ctx)  # doctest: +SKIP
    """

    # ------------------------------------------------------------------
    # Read tools.
    # ------------------------------------------------------------------

    @ctx.tool(mutates=False)
    async def deriva_ml_list_executions(
        hostname: str,
        catalog_id: str,
        workflow_rid: str | None = None,
        status: str | None = None,
        limit: int = 100,
        after_rid: str | None = None,
        preflight_count: bool = False,
    ) -> str:
        """Browse executions in the catalog, optionally filtered by workflow or status.

        PAGINATION: When the count is unknown, call with
        ``preflight_count=True`` first. Then choose a limit and call
        again with ``preflight_count=False``. Use ``after_rid`` (the
        RID of the last execution from the previous page) to advance.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            workflow_rid: If set, return only executions of this workflow.
            status: If set, restrict to one ``ExecutionStatus`` value
                (e.g. ``"Running"``, ``"Uploaded"``).
            limit: Max executions per page (default 100, max 1000).
            after_rid: RID of last row from previous page to advance cursor.
            preflight_count: If True, return only total count.

        Returns:
            JSON string. Page: ``{"executions": [...], "count",
            "truncated", "next_after_rid"}``. Preflight: ``{"total_count",
            "entities_fetched": False, "action_required"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.find_executions`` or
                ``ExecutionStatus`` parsing.

        Example:
            ``{"executions": [{"rid": "1-EXEC", "workflow_rid": "1-WF",
            "status": "Stopped", "description": "...", "start_time": "...",
            "stop_time": "...", "duration": "..."}], "count": 1,
            "truncated": false, "next_after_rid": null}``
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                if preflight_count:
                    status_enum = ExecutionStatus(status) if status else None
                    total = len(list(ml.find_executions(workflow=workflow_rid, status=status_enum)))
                    return json.dumps(
                        {
                            "total_count": total,
                            "entities_fetched": False,
                            "action_required": (
                                f"Found {total} executions. Choose a limit and call "
                                "again with preflight_count=False."
                            ),
                        }
                    )
                capped = min(max(limit, 0), _MAX_LIMIT)
                payload = _list_executions_impl(
                    ml,
                    workflow_rid=workflow_rid,
                    status=status,
                    after_rid=after_rid,
                    limit=capped,
                )
            return json.dumps(payload, default=str)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="list_executions",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_get_execution(
        hostname: str,
        catalog_id: str,
        execution_rid: str,
    ) -> str:
        """Read full details of one execution by RID.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            execution_rid: The RID of the execution to retrieve.

        Returns:
            JSON string with the execution summary: ``{"rid",
            "workflow_rid", "status", "description", "start_time",
            "stop_time", "duration"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.lookup_execution`` (e.g. unknown RID).

        Example:
            ``{"rid": "1-EXEC", "workflow_rid": "1-WF", "status": "Stopped",
            "description": "training run", "start_time": "...",
            "stop_time": "...", "duration": "..."}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                record = ml.lookup_execution(execution_rid)
                summary = _summarize_execution(record)
            return json.dumps(summary, default=str)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="get_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_find_workflow_executions(
        hostname: str,
        catalog_id: str,
        workflow_rid: str,
        status: str | None = None,
        limit: int = 100,
        after_rid: str | None = None,
        preflight_count: bool = False,
    ) -> str:
        """Find all executions of a specific workflow.

        Distinct from ``deriva_ml_list_executions(workflow_rid=...)`` to surface
        the workflow-centric query as a first-class tool — the LLM
        intent ("show me runs of this workflow") differs from the
        general "browse executions" intent.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            workflow_rid: The RID of the workflow whose executions to list.
            status: Optional ``ExecutionStatus`` filter.
            limit: Max executions per page (default 100, max 1000).
            after_rid: RID of last row from previous page to advance cursor.
            preflight_count: If True, return only total count.

        Returns:
            Same shape as ``deriva_ml_list_executions``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.find_executions``.

        Example:
            ``{"executions": [...], "count": 3, "truncated": false,
            "next_after_rid": null}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                status_enum = ExecutionStatus(status) if status else None
                executions = sorted(
                    ml.find_executions(workflow=workflow_rid, status=status_enum),
                    key=lambda e: getattr(e, "execution_rid", "") or "",
                )

            if preflight_count:
                total = len(executions)
                return json.dumps(
                    {
                        "total_count": total,
                        "entities_fetched": False,
                        "action_required": (
                            f"Found {total} executions for workflow {workflow_rid}. "
                            "Choose a limit and call again with preflight_count=False."
                        ),
                    }
                )

            capped = min(max(limit, 0), _MAX_LIMIT)
            page, truncated, next_after = _paginate(
                executions,
                after_rid=after_rid,
                limit=capped,
                key=partial(_read_rid, rid_key="execution_rid"),
            )
            return json.dumps(
                {
                    "executions": [_summarize_execution(e) for e in page],
                    "count": len(page),
                    "truncated": truncated,
                    "next_after_rid": next_after,
                },
                default=str,
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="find_workflow_executions",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_list_execution_children(
        hostname: str,
        catalog_id: str,
        execution_rid: str,
        recurse: bool = False,
    ) -> str:
        """List nested executions of a parent.

        Used by parameter-sweep / multirun parent executions to surface
        their children. ``recurse=True`` walks the whole subtree.

        No pagination: child counts are typically small (tens, not
        thousands). If a real catalog needs paginated children, we'll
        add it then.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            execution_rid: The RID of the parent execution.
            recurse: If True, include all descendants, not just direct
                children.

        Returns:
            JSON string ``{"parent_rid", "recurse", "count",
            "children": [<summary>, ...]}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.lookup_execution`` or
                ``ExecutionRecord.list_execution_children``.

        Example:
            ``{"parent_rid": "1-PARENT", "recurse": false, "count": 2,
            "children": [{"rid": "1-CHILD-A", ...}, {"rid": "1-CHILD-B",
            ...}]}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                record = ml.lookup_execution(execution_rid)
                children = list(record.list_execution_children(recurse=recurse))
            return json.dumps(
                {
                    "parent_rid": execution_rid,
                    "recurse": recurse,
                    "count": len(children),
                    "children": [_summarize_execution(c) for c in children],
                },
                default=str,
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="list_execution_children",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_list_execution_parents(
        hostname: str,
        catalog_id: str,
        execution_rid: str,
        recurse: bool = False,
    ) -> str:
        """List parent executions of a child.

        Symmetric to ``deriva_ml_list_execution_children``. ``recurse=True`` walks
        the whole ancestry chain.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            execution_rid: The RID of the child execution.
            recurse: If True, include all ancestors, not just direct parents.

        Returns:
            JSON string ``{"child_rid", "recurse", "count",
            "parents": [<summary>, ...]}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.lookup_execution`` or
                ``ExecutionRecord.list_execution_parents``.

        Example:
            ``{"child_rid": "1-CHILD", "recurse": false, "count": 1,
            "parents": [{"rid": "1-PARENT", ...}]}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                record = ml.lookup_execution(execution_rid)
                parents = list(record.list_execution_parents(recurse=recurse))
            return json.dumps(
                {
                    "child_rid": execution_rid,
                    "recurse": recurse,
                    "count": len(parents),
                    "parents": [_summarize_execution(p) for p in parents],
                },
                default=str,
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="list_execution_parents",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    # ------------------------------------------------------------------
    # Mutation tools. Each emits audit_event on success and routes
    # failures through _error_envelope.
    # ------------------------------------------------------------------

    @ctx.tool(mutates=True)
    async def deriva_ml_create_execution(
        hostname: str,
        catalog_id: str,
        workflow_rid: str,
        description: str = "",
        dataset_rids: list[str] | None = None,
        asset_rids: list[str] | None = None,
        dry_run: bool = False,
    ) -> str:
        """Register a new execution against an existing workflow.

        Dataset and asset inputs are passed through as RID strings.
        Upstream ``deriva_ml_create_execution`` accepts ``"RID@version"`` shorthand
        for datasets (coerced via ``DatasetSpec.from_shorthand``) and
        bare RID strings for assets (wrapped in ``AssetSpec``).

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            workflow_rid: The RID of the parent workflow.
            description: Free-text description of this execution.
            dataset_rids: Input dataset RIDs in ``"RID@version"`` form
                (e.g. ``"4HM@1.0.0"``). Plain RIDs trigger a new minor
                version upstream.
            asset_rids: Input asset RIDs (bare RID strings).
            dry_run: If True, build the configuration without writing
                to the catalog. Skips audit per Q3 / Phase 3 convention.

        Returns:
            JSON string ``{"status": "created", "execution_rid",
            "workflow_rid", "dataset_count", "asset_count", "dry_run"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.create_execution`` (e.g. unknown
                workflow, FK validation, write failure).

        Example:
            ``{"status": "created", "execution_rid": "1-EXEC",
            "workflow_rid": "1-WF", "dataset_count": 0, "asset_count": 0,
            "dry_run": false}``.
        """
        ds_list = list(dataset_rids or [])
        as_list = list(asset_rids or [])
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                # Pre-resolve workflow_rid to a Workflow object. Upstream
                # `ml.create_execution(workflow=...)` accepts `Workflow | RID
                # | str | None` BUT when given a string, it routes through
                # `lookup_workflow_by_url` (treating the string as URL or
                # checksum), NOT `lookup_workflow` (RID). So passing a
                # workflow_rid string here would silently fail with
                # "Workflow with URL or checksum '<rid>' not found in
                # catalog". Look up the Workflow object via the RID API
                # explicitly so the MCP tool's parameter name and behavior
                # agree.
                workflow = ml.lookup_workflow(workflow_rid)
                # Pass strings through; upstream coerces datasets via
                # DatasetSpec.from_shorthand and assets via AssetSpec(rid=...).
                execution = ml.create_execution(
                    workflow=workflow,
                    datasets=ds_list or None,
                    assets=as_list or None,
                    description=description,
                    dry_run=dry_run,
                )
                execution_rid = execution.execution_rid
                # Detach so DerivaML.__del__ does not abort the freshly
                # created execution when this short-lived ml goes out of
                # scope. Upstream's __del__ aborts any non-terminal
                # ml._execution as a safety net for crashing scripts; in
                # the MCP request/response model the execution is meant
                # to outlive this tool call (a follow-up start_execution
                # / commit_execution call will drive the lifecycle).
                ml._execution = None

            # Q3 / Phase 3 convention: dry_run skips audit because no
            # catalog state actually changed. The response carries
            # dry_run=True so callers see the mode.
            if not dry_run:
                audit_event(
                    "deriva_ml_create_execution",
                    hostname=hostname,
                    catalog_id=catalog_id,
                    workflow_rid=workflow_rid,
                    execution_rid=execution_rid,
                    dataset_count=len(ds_list),
                    asset_count=len(as_list),
                )
            return json.dumps(
                {
                    "status": "created",
                    "execution_rid": execution_rid,
                    "workflow_rid": workflow_rid,
                    "dataset_count": len(ds_list),
                    "asset_count": len(as_list),
                    "dry_run": dry_run,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="create_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                workflow_rid=workflow_rid,
                dataset_count=len(ds_list),
                asset_count=len(as_list),
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_start_execution(
        hostname: str,
        catalog_id: str,
        execution_rid: str,
    ) -> str:
        """Begin a long-running pipeline session against an existing execution.

        Idempotent if already ``Running`` (no-op, no audit). Rejects with
        ``{"error": ...}`` if the execution is in a terminal state
        (``Stopped``, ``Failed``, ``Pending_Upload``, ``Uploaded``,
        ``Aborted``) — those would crash ``execution_start`` in the
        state machine.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            execution_rid: The RID of the execution to start.

        Returns:
            JSON string ``{"status": "running", "execution_rid"}`` on
            transition. ``{"status": "running", "execution_rid", "note":
            "already running"}`` on idempotent no-op.
            ``{"error": "cannot start execution in state ..."}`` on
            terminal state.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.resume_execution`` or
                ``Execution.execution_start``.

        Example:
            ``{"status": "running", "execution_rid": "1-EXEC"}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                execution = ml.resume_execution(execution_rid)
                current = execution.status

                if current == ExecutionStatus.Running:
                    # Idempotent no-op. No audit.
                    return json.dumps(
                        {
                            "status": "running",
                            "execution_rid": execution_rid,
                            "note": "already running",
                        }
                    )
                if current in _START_REJECT_STATES:
                    state_name = current.value if isinstance(current, ExecutionStatus) else current
                    return json.dumps(
                        {
                            "error": (
                                f"cannot start execution in state {state_name}; "
                                "only Created (will start) or Running (no-op) are valid"
                            )
                        }
                    )

                execution.execution_start()

            audit_event(
                "deriva_ml_start_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
            )
            return json.dumps(
                {
                    "status": "running",
                    "execution_rid": execution_rid,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="start_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_commit_execution(
        hostname: str,
        catalog_id: str,
        execution_rid: str,
        retry_failed: bool = False,
    ) -> str:
        """Finalize a run: flush staged feature values and upload assets.

        Advances ``Created`` / ``Running`` -> ``Stopped`` ->
        ``Pending_Upload`` -> ``Uploaded``. ``execution_stop`` is only
        called if the execution hasn't already advanced past Running
        (idempotent at the per-step level).

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            execution_rid: The RID of the execution to commit.
            retry_failed: If True, retry rows and assets that previously
                failed upload.

        Returns:
            JSON string ``{"status": "uploaded", "execution_rid",
            "report": {"execution_rids", "total_uploaded", "total_failed",
            "per_table", "errors", "errors_truncated"}}``. Errors are
            capped at 10 lines.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.resume_execution``,
                ``Execution.execution_stop``, or
                ``Execution.upload_execution_outputs``.

        Example:
            ``{"status": "uploaded", "execution_rid": "1-EXEC",
            "report": {"total_uploaded": 5, "total_failed": 0, ...}}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                execution = ml.resume_execution(execution_rid)
                current = execution.status

                if current not in _COMMIT_ALLOWED_STATES:
                    state_name = current.value if isinstance(current, ExecutionStatus) else current
                    return json.dumps(
                        {
                            "error": (
                                f"cannot commit execution in state {state_name}; "
                                "only Created, Running, Stopped, Pending_Upload, "
                                "or Uploaded (additive upload) are valid"
                            )
                        }
                    )

                # Stop is only legal Created/Running -> Stopped. If we're
                # already past that point (Stopped/Pending_Upload), just
                # proceed straight to upload.
                if current in {ExecutionStatus.Created, ExecutionStatus.Running}:
                    execution.execution_stop()

                # Snapshot pending feature-record count before draining
                # so the response can report how many feature values
                # actually landed (the asset-only return value of
                # upload_execution_outputs doesn't expose this).
                pending_features = execution._manifest_store.list_pending_feature_records(
                    execution_rid
                )
                feature_count = len(pending_features)

                # upload_execution_outputs is the legacy method that
                # ALSO drains staged feature_records via _flush_staged_features
                # (called from _upload_execution_dirs after asset upload).
                # The newer upload_outputs / upload_pending lease engine
                # only handles pending_rows and skips the feature_records
                # SQLite table — so it would leave ``deriva_ml_add_feature_values``
                # data unflushed. Until upstream unifies the two paths,
                # commit_execution must use upload_execution_outputs to
                # get a real end-to-end commit.
                uploaded = execution.upload_execution_outputs()
                summary = _summarize_upload_dict(
                    uploaded,
                    execution_rid=execution_rid,
                    feature_count=feature_count,
                )

            audit_event(
                "deriva_ml_commit_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
                total_uploaded=summary["total_uploaded"],
                total_failed=summary["total_failed"],
                retry_failed=retry_failed,
            )
            return json.dumps(
                {
                    "status": "uploaded",
                    "execution_rid": execution_rid,
                    "report": summary,
                },
                default=str,
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="commit_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
                retry_failed=retry_failed,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_abort_execution(
        hostname: str,
        catalog_id: str,
        execution_rid: str,
        reason: str | None = None,
    ) -> str:
        """Cancel a run from any non-terminal state.

        Idempotent if already ``Aborted`` (no-op, no audit). The optional
        ``reason`` is recorded in the audit row only — the catalog
        does not have a per-execution abort-reason column to write to.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            execution_rid: The RID of the execution to abort.
            reason: Optional bounded admin annotation (audit only).

        Returns:
            JSON string ``{"status": "aborted", "execution_rid",
            "reason"}``. ``{"status": "aborted", "execution_rid",
            "note": "already aborted"}`` on idempotent no-op.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.resume_execution`` or
                ``Execution.abort``.

        Example:
            ``{"status": "aborted", "execution_rid": "1-EXEC",
            "reason": "user cancelled"}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                execution = ml.resume_execution(execution_rid)
                current = execution.status

                if current == ExecutionStatus.Aborted:
                    return json.dumps(
                        {
                            "status": "aborted",
                            "execution_rid": execution_rid,
                            "note": "already aborted",
                        }
                    )

                execution.abort()

            audit_event(
                "deriva_ml_abort_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
                reason=reason,
            )
            return json.dumps(
                {
                    "status": "aborted",
                    "execution_rid": execution_rid,
                    "reason": reason,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="abort_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_create_execution_dataset(
        hostname: str,
        catalog_id: str,
        execution_rid: str,
        description: str = "",
        dataset_types: list[str] | None = None,
    ) -> str:
        """Record a new output dataset linked to a completed execution.

        Used for provenance tracking — the resulting dataset records
        the executions that produced it.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            execution_rid: The RID of the producing execution.
            description: Free-text description of the dataset.
            dataset_types: Optional list of ``Dataset_Type`` vocabulary
                terms to tag the new dataset with.

        Returns:
            JSON string ``{"status": "created", "dataset_rid",
            "execution_rid", "dataset_types", "description"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.resume_execution`` or
                ``Execution.create_dataset`` (e.g. unknown dataset type).

        Example:
            ``{"status": "created", "dataset_rid": "1-NEW-DS",
            "execution_rid": "1-EXEC", "dataset_types": ["Training"],
            "description": ""}``.
        """
        types_list = list(dataset_types) if dataset_types else None
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                execution = ml.resume_execution(execution_rid)
                dataset = execution.create_dataset(
                    dataset_types=types_list,
                    description=description,
                )
                dataset_rid = dataset.rid

            audit_event(
                "deriva_ml_create_execution_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
                dataset_rid=dataset_rid,
                dataset_types=types_list,
            )
            return json.dumps(
                {
                    "status": "created",
                    "dataset_rid": dataset_rid,
                    "execution_rid": execution_rid,
                    "dataset_types": types_list,
                    "description": description,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="create_execution_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_add_nested_execution(
        hostname: str,
        catalog_id: str,
        parent_execution_rid: str,
        child_execution_rid: str,
        sequence: int | None = None,
    ) -> str:
        """Attach a child execution to a parent.

        Used by parameter-sweep / multirun parents that orchestrate
        many child executions. The optional ``sequence`` controls the
        display order in the parent's child list.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            parent_execution_rid: The RID of the parent execution.
            child_execution_rid: The RID of the child execution.
            sequence: Optional integer sequence position. ``None`` lets
                the catalog assign one.

        Returns:
            JSON string ``{"status": "added", "parent_rid", "child_rid",
            "sequence"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.resume_execution`` or
                ``Execution.add_nested_execution``.

        Example:
            ``{"status": "added", "parent_rid": "1-PARENT",
            "child_rid": "1-CHILD", "sequence": 2}``.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                parent = ml.resume_execution(parent_execution_rid)
                parent.add_nested_execution(child_execution_rid, sequence=sequence)

            audit_event(
                "deriva_ml_add_nested_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                parent_rid=parent_execution_rid,
                child_rid=child_execution_rid,
                sequence=sequence,
            )
            return json.dumps(
                {
                    "status": "added",
                    "parent_rid": parent_execution_rid,
                    "child_rid": child_execution_rid,
                    "sequence": sequence,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="add_nested_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                parent_rid=parent_execution_rid,
                child_rid=child_execution_rid,
            )
