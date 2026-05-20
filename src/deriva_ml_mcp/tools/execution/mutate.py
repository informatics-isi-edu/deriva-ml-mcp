"""Mutating execution tools.

This submodule houses the 7 mutating execution tools:

- ``deriva_ml_create_execution`` -- register a new execution against a workflow.
- ``deriva_ml_start_execution`` -- transition Created -> Running (idempotent).
- ``deriva_ml_commit_execution`` -- drain staged work and upload assets.
- ``deriva_ml_update_execution`` -- description-only metadata curation.
- ``deriva_ml_abort_execution`` -- cancel a non-terminal execution (idempotent).
- ``deriva_ml_create_execution_dataset`` -- record a new output dataset.
- ``deriva_ml_add_nested_execution`` -- attach a child execution to a parent.

Each wraps DERIVA I/O in ``with deriva_call():``, emits
``audit_event(...)`` on success, and routes failures through
``_error_envelope`` (which fires the ``deriva_ml_<op>_failed`` audit row).

Audit event lookup goes through ``_pkg.audit_event(...)`` (attribute
lookup on the ``deriva_ml_mcp.tools.execution`` package) so the
package's single ``audit_event`` binding (set in ``__init__.py``) is the
canonical patch site for ``test_execution.py``. Same rationale as
``tools/dataset/mutate.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from deriva_mcp_core import deriva_call
from deriva_ml.execution.execution import ExecutionStatus

# Note on patchable names (``audit_event``, ``get_ml``):
# tests patch ``deriva_ml_mcp.tools.execution.{audit_event, get_ml}``
# as a single canonical site. We access them via the package binding
# (``_pkg.<name>``) rather than ``from ... import <name>`` so a single
# ``patch(...)`` redirects every call across read / mutate in one
# shot. The failure-path audit emission inside ``_error_envelope``
# requires a SECOND patch on ``deriva_ml_mcp._helpers.audit_event`` --
# see ``make_patch_audit("execution")`` in ``tests/_helpers.py``.
import deriva_ml_mcp.tools.execution as _pkg  # noqa: E402  (intentional cycle)
from deriva_ml_mcp._helpers import _error_envelope
from deriva_ml_mcp._response_models import (
    AbortExecutionResponse,
    AddNestedExecutionResponse,
    CommitExecutionReport,
    CommitExecutionResponse,
    CreateExecutionDatasetResponse,
    CreateExecutionResponse,
    StartExecutionResponse,
    UpdateExecutionResponse,
)

if TYPE_CHECKING:
    from typing import Any

    from deriva_mcp_core.plugin.api import PluginContext


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State-machine constants
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register(ctx: PluginContext) -> None:
    """Register the mutating execution tools with the plugin context.

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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``ml.create_execution``
                # is the lifecycle entry point and can be slow (catalog
                # writes + ExecutionConfiguration build). See
                # deriva-mcp-core plugin-authoring-guide.md §"Synchronous
                # work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
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
                workflow = await asyncio.to_thread(ml.lookup_workflow, workflow_rid)
                # Pass strings through; upstream coerces datasets via
                # DatasetSpec.from_shorthand and assets via AssetSpec(rid=...).
                execution = await asyncio.to_thread(
                    ml.create_execution,
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
                _pkg.audit_event(
                    "deriva_ml_create_execution",
                    hostname=hostname,
                    catalog_id=catalog_id,
                    workflow_rid=workflow_rid,
                    execution_rid=execution_rid,
                    dataset_count=len(ds_list),
                    asset_count=len(as_list),
                )
                # v1.3 surgical re-index. Best-effort -- catalog mutation
                # already succeeded; a re-index hiccup must not propagate.
                try:
                    from deriva_ml_mcp.resources.rag import _reindex_execution

                    await _reindex_execution(hostname, catalog_id, execution_rid)
                except Exception:  # noqa: BLE001 -- best-effort cache refresh
                    logger.exception(
                        "re-index failed for execution %s after create_execution",
                        execution_rid,
                    )
            return CreateExecutionResponse(
                status="created",
                execution_rid=execution_rid,
                workflow_rid=workflow_rid,
                dataset_count=len(ds_list),
                asset_count=len(as_list),
                dry_run=dry_run,
            ).model_dump_json(by_alias=True)
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

        Advances the execution from ``Created`` to ``Running``. Required
        before any feature or output write that goes through the
        ``Running`` path.

        **State machine context.** Executions move through:

            Created --start_execution--> Running --commit_execution--> Pending_Upload --> Uploaded
                \\                              \\
                 \\--abort_execution-->          \\--abort_execution--> (Aborted)
                  (Aborted)                                          OR (Failed)

        ``deriva_ml_start_execution`` is the gate INTO the Running phase.
        It accepts only ``Created`` (will advance) or ``Running`` (no-op).

        Idempotent if already ``Running`` (no-op, no audit). Rejects with
        ``{"error": ...}`` if the execution is in any of the
        ``_START_REJECT_STATES`` -- ``Stopped``, ``Pending_Upload``,
        ``Uploaded``, ``Failed``, ``Aborted``. Stopped and Pending_Upload
        are past the algorithmic phase; Failed / Uploaded / Aborted are
        terminal. Each would crash ``execution_start`` in the upstream
        state machine.

        **Do NOT call ``update_record`` to flip the status manually.** The
        state machine is enforced by the lifecycle tools (and by upstream
        ``deriva_ml.Execution`` itself). A direct ``Status`` update bypasses
        the side effects ``deriva_ml_commit_execution`` runs (notably
        upload-outputs) and can leave the execution in an inconsistent
        state.

        Args:
            execution_rid: The RID of the execution to start.

        Returns:
            JSON string ``{"status": "running", "execution_rid"}`` on
            transition. ``{"status": "already_running", "execution_rid"}``
            on idempotent no-op (v3.0 -- replaced the optional ``note``
            field with a discriminator status value).
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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``execution.status`` is
                # a pure Pydantic property read on the already-fetched
                # record and stays on the event loop. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                execution = await asyncio.to_thread(ml.resume_execution, execution_rid)
                current = execution.status

                if current == ExecutionStatus.Running:
                    # Idempotent no-op. No audit. v3.0: status discriminator
                    # carries the no-op signal (was an optional ``note``
                    # field in v2.x).
                    return StartExecutionResponse(
                        status="already_running",
                        execution_rid=execution_rid,
                    ).model_dump_json(by_alias=True)
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

                await asyncio.to_thread(execution.execution_start)

            _pkg.audit_event(
                "deriva_ml_start_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
            )
            # v1.3 surgical re-index: status field changed in the chunk.
            try:
                from deriva_ml_mcp.resources.rag import _reindex_execution

                await _reindex_execution(hostname, catalog_id, execution_rid)
            except Exception:  # noqa: BLE001 -- best-effort cache refresh
                logger.exception(
                    "re-index failed for execution %s after start_execution",
                    execution_rid,
                )
            return StartExecutionResponse(
                status="running",
                execution_rid=execution_rid,
            ).model_dump_json(by_alias=True)
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

        **REQUIRED to make staged outputs visible.** Feature values, output
        datasets, and assets written during a Running execution are STAGED
        -- they only become visible to downstream queries once
        ``deriva_ml_commit_execution`` drains them and transitions the
        execution to ``Uploaded``. **A forgotten commit leaves outputs
        invisible.** The single most common "I added the feature value
        but the catalog doesn't show it" failure mode is a missing commit.

        Advances ``Created`` / ``Running`` -> ``Stopped`` ->
        ``Pending_Upload`` -> ``Uploaded``. ``execution_stop`` is only
        called if the execution hasn't already advanced past Running
        (idempotent at the per-step level).

        **Accepts states in ``_COMMIT_ALLOWED_STATES``**: ``Created``,
        ``Running``, ``Stopped``, ``Pending_Upload``, ``Uploaded``.
        Pending_Upload is included because commit's whole purpose is to
        drain it. Uploaded is the additive-upload entry point: calling
        ``deriva_ml_commit_execution`` on an Uploaded execution that has
        new pending entries cycles Uploaded -> Pending_Upload -> Uploaded;
        with no pending entries it is a clean no-op. Rejects only the
        terminal failure states (Failed, Aborted) which can't be
        meaningfully drained.

        **Do NOT call ``update_record`` to flip the status manually** --
        that bypasses the upload-outputs side effect this tool runs and
        leaves the execution in an inconsistent state.

        Args:
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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive — commit drains staged
                # feature rows, runs ``upload_execution_outputs`` (which
                # can do multi-minute network work), and updates catalog
                # state. ``execution.status`` is a pure Pydantic property
                # read and stays on the event loop. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                execution = await asyncio.to_thread(ml.resume_execution, execution_rid)
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
                    await asyncio.to_thread(execution.execution_stop)

                # Snapshot pending feature-record count before draining
                # so the response can report how many feature values
                # actually landed (the asset-only return value of
                # upload_execution_outputs doesn't expose this). The
                # manifest store is a local SQLite engine — sync I/O
                # that still blocks the event loop, so wrap it.
                pending_features = await asyncio.to_thread(
                    execution._manifest_store.list_pending_feature_records,
                    execution_rid,
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
                uploaded = await asyncio.to_thread(execution.upload_execution_outputs)
                summary = _summarize_upload_dict(
                    uploaded,
                    execution_rid=execution_rid,
                    feature_count=feature_count,
                )

            _pkg.audit_event(
                "deriva_ml_commit_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
                total_uploaded=summary["total_uploaded"],
                total_failed=summary["total_failed"],
                retry_failed=retry_failed,
            )
            # v1.3 surgical re-index: status + stop_time + the three
            # *_duration fields changed.
            try:
                from deriva_ml_mcp.resources.rag import _reindex_execution

                await _reindex_execution(hostname, catalog_id, execution_rid)
            except Exception:  # noqa: BLE001 -- best-effort cache refresh
                logger.exception(
                    "re-index failed for execution %s after commit_execution",
                    execution_rid,
                )
            return CommitExecutionResponse(
                status="uploaded",
                execution_rid=execution_rid,
                report=CommitExecutionReport(**summary),
            ).model_dump_json(by_alias=True)
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
    async def deriva_ml_update_execution(
        hostname: str,
        catalog_id: str,
        execution_rid: str,
        description: str | None = None,
    ) -> str:
        """Update an execution's description.

        Description-only by design. The Execution table has no
        ``Execution_Type`` field (no curation symmetry to add there),
        and ``Status`` edits remain forbidden -- they are state-machine
        territory driven by ``deriva_ml_start_execution`` /
        ``deriva_ml_commit_execution`` / ``deriva_ml_abort_execution``,
        not freely editable. Free-form status writes were deliberately
        rejected in v1.0 (see the ``update_execution_status`` row in
        ``docs/coverage.md``).

        ``description`` is a free-text overwrite of the execution row's
        ``Description`` column. Setting it on the catalog-bound
        ``ExecutionRecord`` writes through to the catalog directly via
        the deriva-ml setter hook.

        Args:
            execution_rid: The RID of the execution to update.
            description: New description text. Required (must be
                non-None); passing ``None`` returns an error envelope
                because there is nothing else to update.

        Returns:
            JSON string ``{"status": "updated", "execution_rid",
            "updated_fields": ["description"]}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.lookup_execution`` or the
                ``ExecutionRecord.description`` catalog-write hook
                (e.g. unknown RID, read-only catalog snapshot).

        Example:
            ``{"status": "updated", "execution_rid": "1-EXEC",
            "updated_fields": ["description"]}``.
        """
        if description is None:
            return json.dumps(
                {
                    "error": (
                        "description must be provided; no other fields are editable on Execution "
                        "(status changes go through start/commit/abort)"
                    )
                }
            )

        updated_fields: list[str] = []
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``record.description =
                # ...`` writes back to the catalog via the
                # ExecutionRecord ``description.setter`` hook (see
                # deriva_ml/execution/execution_record.py:
                # _update_description_in_catalog), so it is synchronous
                # I/O and must run in the worker thread. See
                # deriva-mcp-core plugin-authoring-guide.md §"Synchronous
                # work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                record = await asyncio.to_thread(ml.lookup_execution, execution_rid)

                def _apply_description() -> None:
                    record.description = description

                await asyncio.to_thread(_apply_description)
                updated_fields.append("description")
            _pkg.audit_event(
                "deriva_ml_update_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
                updated_fields=updated_fields,
            )
            # v1.3 surgical re-index: description field changed in the chunk.
            try:
                from deriva_ml_mcp.resources.rag import _reindex_execution

                await _reindex_execution(hostname, catalog_id, execution_rid)
            except Exception:  # noqa: BLE001 -- best-effort cache refresh
                logger.exception(
                    "re-index failed for execution %s after update_execution",
                    execution_rid,
                )
            return UpdateExecutionResponse(
                status="updated",
                execution_rid=execution_rid,
                updated_fields=updated_fields,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="update_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
                updated_fields=updated_fields,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_abort_execution(
        hostname: str,
        catalog_id: str,
        execution_rid: str,
        reason: str | None = None,
    ) -> str:
        """Cancel a run from any non-terminal state.

        **Escape hatch.** Use when a run cannot continue and you want the
        provenance row to record that fact -- the failure is deliberate
        rather than a crash. Terminates a non-terminal execution with
        optional ``reason`` text.

        Distinguish from natural completion:
        - Use ``deriva_ml_commit_execution`` when the run completed and
          you want to flush staged outputs and finalize provenance.
        - Use ``deriva_ml_abort_execution`` when the run cannot continue
          and you want to record the cancellation. **Aborting destroys
          the staged outputs.** Use commit if there's salvageable work.

        Idempotent if already ``Aborted`` (no-op, no audit). The optional
        ``reason`` is recorded in the audit row only — the catalog
        does not have a per-execution abort-reason column to write to.

        Args:
            execution_rid: The RID of the execution to abort.
            reason: Optional bounded admin annotation (audit only).

        Returns:
            JSON string ``{"status": "aborted", "execution_rid",
            "reason"}``. ``{"status": "already_aborted", "execution_rid",
            "reason": null}`` on idempotent no-op (v3.0 -- replaced the
            optional ``note`` field with a discriminator status value).
            ``reason`` is always present; ``null`` when no reason was
            given.

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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``execution.status`` is
                # a pure Pydantic property read on the already-fetched
                # record and stays on the event loop. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                execution = await asyncio.to_thread(ml.resume_execution, execution_rid)
                current = execution.status

                if current == ExecutionStatus.Aborted:
                    # v3.0: status discriminator carries the no-op signal
                    # (was an optional ``note`` field in v2.x). ``reason``
                    # is None here -- we didn't actually record an abort
                    # reason this call.
                    return AbortExecutionResponse(
                        status="already_aborted",
                        execution_rid=execution_rid,
                        reason=None,
                    ).model_dump_json(by_alias=True)

                await asyncio.to_thread(execution.abort)

            _pkg.audit_event(
                "deriva_ml_abort_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
                reason=reason,
            )
            # v1.3 surgical re-index: status field changed (now Aborted).
            try:
                from deriva_ml_mcp.resources.rag import _reindex_execution

                await _reindex_execution(hostname, catalog_id, execution_rid)
            except Exception:  # noqa: BLE001 -- best-effort cache refresh
                logger.exception(
                    "re-index failed for execution %s after abort_execution",
                    execution_rid,
                )
            return AbortExecutionResponse(
                status="aborted",
                execution_rid=execution_rid,
                reason=reason,
            ).model_dump_json(by_alias=True)
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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``dataset.rid`` is a
                # pure Pydantic attribute read on the already-created
                # dataset and stays on the event loop. See
                # deriva-mcp-core plugin-authoring-guide.md §"Synchronous
                # work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                execution = await asyncio.to_thread(ml.resume_execution, execution_rid)
                dataset = await asyncio.to_thread(
                    execution.create_dataset,
                    dataset_types=types_list,
                    description=description,
                )
                dataset_rid = dataset.rid

            _pkg.audit_event(
                "deriva_ml_create_execution_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
                dataset_rid=dataset_rid,
                dataset_types=types_list,
            )
            # v1.3 surgical re-index: a new dataset row landed AND the
            # parent execution's outputs view changed; refresh both.
            # Per-target try/except so if dataset re-index fails, the
            # execution re-index still runs (and vice versa) -- defense
            # in depth against the helper's own try/except being
            # bypassed by an unexpected outer raise.
            from deriva_ml_mcp.resources.rag import (
                _reindex_dataset,
                _reindex_execution,
            )

            try:
                await _reindex_dataset(hostname, catalog_id, dataset_rid)
            except Exception:  # noqa: BLE001 -- best-effort, per-target
                logger.exception(
                    "re-index failed for dataset %s after create_execution_dataset",
                    dataset_rid,
                )
            try:
                await _reindex_execution(hostname, catalog_id, execution_rid)
            except Exception:  # noqa: BLE001 -- best-effort, per-target
                logger.exception(
                    "re-index failed for execution %s after create_execution_dataset",
                    execution_rid,
                )
            return CreateExecutionDatasetResponse(
                status="created",
                dataset_rid=dataset_rid,
                execution_rid=execution_rid,
                dataset_types=types_list,
                description=description,
            ).model_dump_json(by_alias=True)
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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                parent = await asyncio.to_thread(ml.resume_execution, parent_execution_rid)
                await asyncio.to_thread(
                    parent.add_nested_execution,
                    child_execution_rid,
                    sequence=sequence,
                )

            _pkg.audit_event(
                "deriva_ml_add_nested_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                parent_rid=parent_execution_rid,
                child_rid=child_execution_rid,
                sequence=sequence,
            )
            # v1.3 surgical re-index: parent's children view changed.
            try:
                from deriva_ml_mcp.resources.rag import _reindex_execution

                await _reindex_execution(hostname, catalog_id, parent_execution_rid)
            except Exception:  # noqa: BLE001 -- best-effort cache refresh
                logger.exception(
                    "re-index failed for parent execution %s after add_nested_execution",
                    parent_execution_rid,
                )
            return AddNestedExecutionResponse(
                status="added",
                parent_rid=parent_execution_rid,
                child_rid=child_execution_rid,
                sequence=sequence,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="add_nested_execution",
                hostname=hostname,
                catalog_id=catalog_id,
                parent_rid=parent_execution_rid,
                child_rid=child_execution_rid,
            )
