"""Workflow domain tools for deriva-ml-mcp.

Read tools: ``deriva_ml_list_workflows``, ``deriva_ml_get_workflow``, ``deriva_ml_find_workflow_by_url``.
Mutation tools: ``deriva_ml_create_workflow``, ``deriva_ml_update_workflow``.

Every tool wraps DERIVA I/O in ``with deriva_call():`` and routes errors
through ``_error_envelope`` (mutation tools also emit success/failure
audit events; reads only log on failure).

Workflow registration is the bridge between an external Git checkout and
the catalog. The boundary rule: the caller (deriva-skills / a notebook /
hydra-zen runner) computes the URL/checksum/version locally and passes
them in. We never run server-side git introspection on the MCP host.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from deriva_mcp_core import deriva_call
from deriva_mcp_core.telemetry import audit_event
from deriva_ml.core.enums import MLVocab
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.execution.workflow import Workflow

logger = logging.getLogger(__name__)

# Note on testing audit_event: see ``make_patch_audit("workflow")`` in
# tests/conftest.py. Single-patch facade is impossible due to Python's
# ``from X import name`` import binding semantics — tests must patch
# BOTH ``deriva_ml_mcp.tools.workflow.audit_event`` (this module's
# success-path emission) and ``deriva_ml_mcp._helpers.audit_event`` (the
# failure-path emission inside ``_error_envelope``).
from deriva_ml_mcp._helpers import (
    _MAX_LIMIT,
    _error_envelope,
    _paginate,
)
from deriva_ml_mcp._response_models import (
    CreateWorkflowResponse,
    PreflightCountResponse,
    UpdateWorkflowResponse,
    WorkflowListResponse,
    WorkflowSummary,
)
from deriva_ml_mcp.ml_context import get_ml

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def _resolve_workflow_rid(wf: Any) -> str | None:
    """Resolve the RID of a Workflow object, with a defensive recovery path.

    Same root cause as ``tools.execution.read._resolve_workflow_rid``:
    deriva-ml #226 renamed the field from ``rid`` to ``workflow_rid``
    but its two construction sites (``find_workflows``,
    ``lookup_workflow``) still pass the OLD kwarg name. Pydantic's
    permissive config silently drops the value, leaving
    ``workflow.workflow_rid == None`` on every Workflow returned by
    either API.

    Fast path: if ``wf.workflow_rid`` is populated, return it. Recovery
    path: if the workflow is bound to a catalog (``_ml_instance``)
    and carries either ``checksum`` or ``url``, call
    ``ml._find_workflow_rid_by_url(checksum_or_url)`` to look the RID
    up. That helper is server-side filtered (single round trip with
    one row in the response) so the recovery cost is bounded.

    Args:
        wf: A ``deriva_ml.execution.workflow.Workflow`` instance, or a
            Mock-shaped stand-in.

    Returns:
        The Workflow RID, or None if recovery wasn't possible.

    Example:
        >>> from unittest.mock import MagicMock
        >>> wf = MagicMock()
        >>> wf.workflow_rid = "1-WF"
        >>> _resolve_workflow_rid(wf)
        '1-WF'
    """
    rid = getattr(wf, "workflow_rid", None)
    if rid:
        return rid
    ml = getattr(wf, "_ml_instance", None)
    if ml is None:
        return None
    lookup_key = getattr(wf, "checksum", None) or getattr(wf, "url", None)
    if not lookup_key:
        return None
    try:
        return ml._find_workflow_rid_by_url(lookup_key)
    except Exception:  # noqa: BLE001 -- defensive recovery, return None on any failure
        return None


def _summarize_workflow(wf: Any) -> WorkflowSummary:
    """Render a Workflow object into the validated summary used by list endpoints.

    Args:
        wf: A ``deriva_ml.execution.workflow.Workflow`` instance.

    Returns:
        ``WorkflowSummary`` Pydantic instance -- see
        ``deriva_ml_mcp._response_models``. Construction validates
        the field types; bad data from ``deriva-ml`` raises
        ``ValidationError`` here rather than surfacing as malformed
        JSON downstream.

    Note:
        v2.2 sweep: this helper now returns Pydantic. Consumers that
        previously dict-accessed (``summary["rid"]``) must use
        attribute access (``summary.rid``) or call ``.model_dump()``
        to get a plain dict back.

    NOTE(2026-05-26 audit): ``Workflow.rid`` was renamed to
    ``Workflow.workflow_rid`` in deriva-ml #226 but the deriva-ml
    construction sites still pass ``rid=`` and the field is silently
    dropped. We read through ``_resolve_workflow_rid`` so the
    Workflow's RID is recovered (via checksum/url lookup) when the
    upstream property returns None. Once deriva-ml is patched the
    recovery path becomes dead code.
    """
    return WorkflowSummary(
        rid=_resolve_workflow_rid(wf),
        name=wf.name,
        url=wf.url,
        checksum=wf.checksum,
        version=wf.version,
        workflow_type=list(wf.workflow_type) if wf.workflow_type else [],
        description=wf.description,
    )


def _list_workflows_impl(
    ml: Any,
    *,
    after_rid: str | None,
    limit: int,
    sort: bool = False,
) -> WorkflowListResponse:
    """Fetch + paginate workflows. Pure helper -- shared by tool and resource.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        after_rid: Cursor for pagination.
        limit: Max workflows per page (already capped by caller).
        sort: If True, results are ordered newest-first by record
            creation time (RCT desc) -- forwarded to
            ``deriva_ml.DerivaML.find_workflows(sort=True)``. If False
            (default), results are RID-ascending for stable cursor
            pagination. Note that under ``sort=True`` the ``after_rid``
            cursor still works ("skip up to this RID in the RCT-sorted
            result"), but pagination through very large sorted result
            sets is bounded by the internal fetch cap.

    Returns:
        ``WorkflowListResponse`` -- see ``deriva_ml_mcp._response_models``.
    """
    raw = list(ml.find_workflows(sort=True if sort else None))
    # Keep stable RID-ascending order for the default path; under
    # sort=True we honor the catalog-side RCT-desc ordering and skip
    # the post-fetch sort (sorting by RID would clobber the RCT order).
    #
    # Both the sort key and the pagination cursor read through
    # ``_resolve_workflow_rid`` so the deriva-ml #226 regression (every
    # Workflow returns workflow_rid=None) does not collapse the sort
    # to "all-empty" or break cursor pagination. The recovery branch
    # in the resolver is a server-side filter (single round trip per
    # workflow); a fix in deriva-ml will skip it entirely.
    if sort:
        workflows = raw
    else:
        workflows = sorted(raw, key=lambda w: _resolve_workflow_rid(w) or "")
    page, truncated, next_after = _paginate(
        workflows,
        after_rid=after_rid,
        limit=limit,
        key=lambda w: _resolve_workflow_rid(w) or "",
    )
    return WorkflowListResponse(
        workflows=[_summarize_workflow(w) for w in page],
        count=len(page),
        truncated=truncated,
        next_after_rid=next_after,
    )


def _get_workflow_impl(ml: Any, workflow_rid: str) -> WorkflowSummary:
    """Read one workflow's full summary.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        workflow_rid: The RID of the workflow to look up.

    Returns:
        ``WorkflowSummary`` -- see ``deriva_ml_mcp._response_models``.
        The ``WorkflowDetail`` wrapper type was retired in the P2
        hygiene sweep because it was shape-identical to
        ``WorkflowSummary`` and had no detail-only fields.
        Resources that previously returned ``WorkflowDetail`` now
        return ``WorkflowSummary`` directly (same wire shape).
    """
    wf = ml.lookup_workflow(workflow_rid)
    return _summarize_workflow(wf)


def register(ctx: PluginContext) -> None:
    """Register all workflow domain tools with the plugin context.

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
    async def deriva_ml_list_workflows(
        hostname: str,
        catalog_id: str,
        limit: int = 100,
        after_rid: str | None = None,
        preflight_count: bool = False,
        sort: bool = False,
    ) -> str:
        """Browse all workflows registered in the catalog.

        See ``deriva_ml_getting_started`` (PAGINATION CONTRACT) for the two-step pagination flow.

        Args:
            limit: Max workflows per page (default 100, max 1000).
            after_rid: RID of last row from previous page to advance cursor.
            preflight_count: If True, return only total count.
            sort: If True, return results newest-first by record
                creation time. Recommended for "show me the most
                recent workflows" queries. Default False preserves the
                stable RID-ascending order used for cursor pagination.

        Returns:
            Preflight:
            ``{"total_count": N, "entities_fetched": False, "action_required": "..."}``.
            Page: ``{"workflows": [{"rid", "name", "url", "checksum",
            "version", "workflow_type", "description"}, ...], "count": N,
            "truncated": bool, "next_after_rid": <last-rid> | null}``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated from
                ``deriva_ml.DerivaML.find_workflows``.

        Example:
            ``{"workflows": [{"rid": "1-WF", "name": "MyPipeline",
            "url": "https://github.com/example/repo", "checksum": "abc123",
            "version": "1.0.0", "workflow_type": ["Model_Training"],
            "description": ""}], "count": 1, "truncated": false,
            "next_after_rid": null}``
        """
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``ml.find_workflows``
                # returns a generator that materializes over the wire, so
                # the drain (list/sorted) must happen inside the worker
                # thread, not on the event loop. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
                if preflight_count:

                    def _count_workflows() -> int:
                        return len(list(ml.find_workflows()))

                    total = await asyncio.to_thread(_count_workflows)
                    return PreflightCountResponse(
                        total_count=total,
                        action_required=(
                            f"Found {total} workflows. Choose a limit and call "
                            "again with preflight_count=False."
                        ),
                    ).model_dump_json(by_alias=True)

                capped = min(max(limit, 0), _MAX_LIMIT)
                payload = await asyncio.to_thread(
                    _list_workflows_impl,
                    ml,
                    after_rid=after_rid,
                    limit=capped,
                    sort=sort,
                )
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:
            # Read-only tool: log+return without an audit row.
            return _error_envelope(
                exc,
                operation="list_workflows",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_get_workflow(
        hostname: str,
        catalog_id: str,
        workflow_rid: str,
    ) -> str:
        """Read full details of one workflow by RID.

        Args:
            workflow_rid: The RID of the workflow to retrieve.

        Returns:
            JSON string with the workflow summary:
            ``{"rid", "name", "url", "checksum", "version",
            "workflow_type", "description"}``.

        Raises:
            RuntimeError: If the workflow RID doesn't exist, propagated
                from ``deriva_ml.DerivaML.lookup_workflow`` (returned as
                ``{"error": ...}``).

        Example:
            ``{"rid": "1-WF", "name": "MyPipeline",
            "url": "https://github.com/example/repo", "checksum": "abc123",
            "version": "1.0.0", "workflow_type": ["Model_Training"],
            "description": ""}``.
        """
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
                summary = await asyncio.to_thread(_get_workflow_impl, ml, workflow_rid)
            return summary.model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="get_workflow",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_find_workflow_by_url(
        hostname: str,
        catalog_id: str,
        url_or_checksum: str,
    ) -> str:
        """Look up a workflow by Git URL or checksum.

        Critical for hydra-zen configs that carry only the URL. The URL
        should be a GitHub URL pointing to a specific commit, or the Git
        object hash (checksum) of the workflow file.

        **Use this tool only for read-only intent.** If your goal is to
        ensure a workflow exists -- creating it if missing -- call
        ``deriva_ml_create_workflow`` directly. That tool is idempotent on
        ``(url, checksum)``: the second call returns the existing RID
        with ``status="exists"`` rather than creating a duplicate.
        Preflight-then-create with this tool wastes a round-trip on the
        dedup ``create_workflow`` already performs internally.

        Args:
            url_or_checksum: GitHub URL with commit hash, or Git object
                hash (checksum) of the workflow file.

        Returns:
            JSON string with the workflow summary:
            ``{"rid", "name", "url", "checksum", "version",
            "workflow_type", "description"}``.

        Raises:
            RuntimeError: If no workflow with the given URL or checksum
                exists, propagated from
                ``deriva_ml.DerivaML.lookup_workflow_by_url`` (returned
                as ``{"error": ...}``).

        Example:
            ``{"rid": "1-WF", "name": "MyPipeline",
            "url": "https://github.com/example/repo/blob/abc/main.py",
            "checksum": "abc123", "version": "1.0.0",
            "workflow_type": ["Model_Training"], "description": ""}``.
        """
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
                wf = await asyncio.to_thread(ml.lookup_workflow_by_url, url_or_checksum)
                summary = _summarize_workflow(wf)
            return summary.model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="find_workflow_by_url",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    # ------------------------------------------------------------------
    # Mutation tools. Each emits audit_event on success and routes
    # failures through _error_envelope.
    # ------------------------------------------------------------------

    @ctx.tool(mutates=True)
    async def deriva_ml_create_workflow(
        hostname: str,
        catalog_id: str,
        name: str,
        workflow_type: str | list[str],
        url: str,
        checksum: str | None = None,
        version: str | None = None,
        description: str = "",
    ) -> str:
        """Register a new workflow (script + Git URL + workflow_type tags).

        URL is required. Workflows are deduplicated: if a workflow with the
        same URL or checksum already exists, returns its RID with
        ``status="exists"`` instead of creating a duplicate.

        **This tool is idempotent. Do not preflight with
        ``deriva_ml_find_workflow_by_url``.**

        Calling ``deriva_ml_create_workflow(url=X, checksum=Y, ...)`` twice
        with the same ``(X, Y)`` returns the SAME RID both times. The first
        call carries ``status="created"``; the second carries
        ``status="exists"``. Both responses point to the same
        ``workflow_rid`` -- the LLM can unconditionally use the returned
        RID without branching on ``status``.

        The wrong pattern (do not do this; it wastes a round-trip on the
        dedup the server already performs)::

            # ANTI-PATTERN -- do not do this
            existing = deriva_ml_find_workflow_by_url(url=X)
            if existing is None:
                deriva_ml_create_workflow(url=X, checksum=Y, ...)
            else:
                workflow_rid = existing["workflow_rid"]

        The right pattern::

            # CORRECT -- one round trip, idempotent
            result = deriva_ml_create_workflow(url=X, checksum=Y,
                                               name="MyPipeline",
                                               workflow_type="Model_Training")
            workflow_rid = result["workflow_rid"]  # works whether new or existing

        ``deriva_ml_find_workflow_by_url`` is the right tool only when you
        do NOT intend to create the workflow if it is missing -- e.g. you
        are answering "is this workflow already registered?" or you are
        linking against an existing workflow whose presence you want to
        verify before doing other work. If your intent is "make sure this
        workflow exists, creating it if not", go straight to
        ``deriva_ml_create_workflow``.

        Boundary rule: the caller (deriva-skills / notebook / hydra-zen
        runner) computes the Git URL, checksum, and version locally and
        passes them in. The MCP server never runs git introspection.

        Args:
            name: Human-readable name of the workflow.
            workflow_type: Type(s) tag — single string or list. Each
                must be a term in the ``Workflow_Type`` vocabulary.
            url: Git URL pinned to a specific commit (REQUIRED).
            checksum: Optional Git object hash of the workflow file.
                Used as a secondary dedup key.
            version: Optional semver string for the workflow release.
            description: Free-text description of what the workflow does.

        Returns:
            JSON string ``{"status": "created" | "exists", "workflow_rid",
            "name", "workflow_type", "url", "checksum", "version"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.create_workflow`` /
                ``_add_workflow`` (e.g. unknown workflow_type term, or
                catalog write failure).

        Example:
            ``{"status": "created", "workflow_rid": "1-WF",
            "name": "MyPipeline", "workflow_type": ["Model_Training"],
            "url": "https://github.com/example/repo/blob/abc/main.py",
            "checksum": "abc123", "version": "1.0.0"}``.
        """
        normalized_types = (
            [workflow_type] if isinstance(workflow_type, str) else list(workflow_type)
        )
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
                # Pre-check exists solely to set status="exists" for the
                # audit + return value. _add_workflow already dedups
                # internally on `checksum or url` and returns the existing
                # RID either way (deriva_ml/core/mixins/workflow.py
                # _add_workflow source) — so without this pre-check the
                # tool would always report status="created" even when no
                # row was inserted. Match _add_workflow's lookup key
                # exactly (checksum-first, url-fallback) so the pre-check
                # finds the same dedup hits the upstream code does.
                # TOCTOU note: a concurrent insert between this lookup
                # and _add_workflow would result in status="created"
                # being reported for what _add_workflow then dedups —
                # acceptable for serialized MCP execution.
                lookup_key = checksum or url
                rid: str
                status: str
                try:
                    existing = await asyncio.to_thread(ml.lookup_workflow_by_url, lookup_key)
                except DerivaMLException:
                    existing = None
                if existing is not None:
                    # NOTE(2026-05-26 audit): ``Workflow.rid`` -> ``workflow_rid``
                    # in deriva-ml #226. Route through the defensive
                    # resolver so the rename regression (workflow_rid
                    # populated as None on every Workflow returned by
                    # lookup_workflow_by_url) doesn't surface here as
                    # a null ``workflow_rid`` in the response.
                    rid = _resolve_workflow_rid(existing)
                    status = "exists"
                else:
                    # Validate every workflow_type term against the
                    # Workflow_Type vocabulary BEFORE constructing the
                    # Workflow. Previously this lived inside
                    # ``ml.create_workflow(name, workflow_type, description)``,
                    # which we no longer call -- see the note below.
                    for wt in normalized_types:
                        await asyncio.to_thread(ml.lookup_term, MLVocab.workflow_type, wt)
                    # Construct ``Workflow`` directly with url / checksum
                    # / version set at construction time so the model
                    # validator (``setup_url_checksum``) short-circuits
                    # at ``if not self.url`` and skips the running-script
                    # git introspection. The pre-1.36.5 path constructed
                    # via ``ml.create_workflow`` and then set the three
                    # attributes post-hoc; under v1.36.5+ the validator
                    # is strict and tries to derive url/checksum from
                    # the running script's git context at construction
                    # time -- which blows up inside the MCP server
                    # (no script path; no git repo around the worker).
                    # Boundary rule (per the tool docstring): the
                    # caller computes url/checksum/version locally and
                    # passes them in. The MCP server never runs git
                    # introspection -- direct construction enforces it.
                    wf = Workflow(
                        name=name,
                        workflow_type=workflow_type,
                        description=description,
                        url=url,
                        checksum=checksum,
                        version=version,
                    )
                    # _add_workflow is the canonical write path even
                    # though the leading underscore looks private — all
                    # workflow inserts in deriva_ml itself go through it.
                    # If deriva_ml 2.x renames it, our pre-flight smoke
                    # test will catch the mismatch via the import.
                    rid = await asyncio.to_thread(ml._add_workflow, wf)
                    status = "created"
            audit_event(
                "deriva_ml_create_workflow",
                hostname=hostname,
                catalog_id=catalog_id,
                workflow_rid=rid,
                name=name,
                workflow_type=normalized_types,
                status=status,
            )
            # v1.3 surgical re-index. Best-effort -- catalog mutation
            # already succeeded; a re-index hiccup must not propagate
            # to the tool's success path.
            try:
                from deriva_ml_mcp.resources.rag import _reindex_workflow

                await _reindex_workflow(hostname, catalog_id, rid)
            except Exception:  # noqa: BLE001 -- best-effort cache refresh
                logger.exception(
                    "re-index failed for workflow %s after create_workflow",
                    rid,
                )
            return CreateWorkflowResponse(
                status=status,
                workflow_rid=rid,
                name=name,
                workflow_type=normalized_types,
                url=url,
                checksum=checksum,
                version=version,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="create_workflow",
                hostname=hostname,
                catalog_id=catalog_id,
                name=name,
                workflow_type=normalized_types,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_update_workflow(
        hostname: str,
        catalog_id: str,
        workflow_rid: str,
        description: str | None = None,
        workflow_type: list[str] | None = None,
    ) -> str:
        """Update a workflow's description and/or workflow_type tags.

        Pass ``None`` to leave a field unchanged. At least one of
        ``description`` or ``workflow_type`` must be non-None. The
        underlying ``Workflow`` object is catalog-bound, so setting
        ``wf.description`` / ``wf.workflow_type`` writes back to the
        catalog directly via the Workflow's __setattr__ hook.

        Args:
            workflow_rid: The RID of the workflow to update.
            description: New description text. None leaves unchanged.
            workflow_type: New list of workflow_type tag terms. None
                leaves unchanged. Empty list is rejected (callers should
                pass None for "no change", not ``[]``).

        Returns:
            JSON string ``{"status": "updated", "workflow_rid",
            "updated_fields": [...]}``. ``updated_fields`` is the list of
            field names that were actually written.

        Audit notes:
            The audit row always carries a ``workflow_type`` field
            (``None`` when the caller didn't pass one) for searchability
            — operators querying audits by workflow_type tag find every
            mutation that touched the field, including the no-change
            calls. ``description`` is NEVER in the audit (free text).

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.lookup_workflow`` or the Workflow
                __setattr__ catalog-write hook (e.g. unknown
                workflow_type term, read-only catalog snapshot).

        Example:
            ``{"status": "updated", "workflow_rid": "1-WF",
            "updated_fields": ["description", "workflow_type"]}``.
        """
        # Argument validation — return errors directly without audit.
        if description is None and workflow_type is None:
            return json.dumps(
                {"error": ("at least one of description or workflow_type must be provided")}
            )
        if workflow_type is not None and len(workflow_type) == 0:
            return json.dumps(
                {"error": ("workflow_type list cannot be empty; pass None to leave unchanged")}
            )

        updated_fields: list[str] = []
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``wf.description`` and
                # ``wf.workflow_type`` setattrs write back to the catalog
                # via the Workflow ``__setattr__`` hook (see
                # deriva_ml/execution/workflow.py), so they're synchronous
                # I/O and must run in the worker thread. See
                # deriva-mcp-core plugin-authoring-guide.md §"Synchronous
                # work in threads".
                ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
                wf = await asyncio.to_thread(ml.lookup_workflow, workflow_rid)

                def _apply_updates() -> None:
                    if description is not None:
                        wf.description = description
                    if workflow_type is not None:
                        wf.workflow_type = workflow_type

                await asyncio.to_thread(_apply_updates)
                if description is not None:
                    updated_fields.append("description")
                if workflow_type is not None:
                    updated_fields.append("workflow_type")
            audit_event(
                "deriva_ml_update_workflow",
                hostname=hostname,
                catalog_id=catalog_id,
                workflow_rid=workflow_rid,
                updated_fields=updated_fields,
                workflow_type=workflow_type,
            )
            # v1.3 surgical re-index: description and/or workflow_type
            # tags appear in the chunk; refresh after the catalog write.
            try:
                from deriva_ml_mcp.resources.rag import _reindex_workflow

                await _reindex_workflow(hostname, catalog_id, workflow_rid)
            except Exception:  # noqa: BLE001 -- best-effort cache refresh
                logger.exception(
                    "re-index failed for workflow %s after update_workflow",
                    workflow_rid,
                )
            return UpdateWorkflowResponse(
                status="updated",
                workflow_rid=workflow_rid,
                updated_fields=updated_fields,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="update_workflow",
                hostname=hostname,
                catalog_id=catalog_id,
                workflow_rid=workflow_rid,
                updated_fields=updated_fields,
            )
