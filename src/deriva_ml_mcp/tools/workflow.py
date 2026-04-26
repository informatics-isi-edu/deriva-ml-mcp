"""Workflow domain tools for deriva-ml-mcp.

Read tools: ``list_workflows``, ``get_workflow``, ``find_workflow_by_url``.
Mutation tools: ``create_workflow``, ``update_workflow``.

Every tool wraps DERIVA I/O in ``with deriva_call():`` and routes errors
through ``_error_envelope`` (mutation tools also emit success/failure
audit events; reads only log on failure).

Workflow registration is the bridge between an external Git checkout and
the catalog. The boundary rule: the caller (deriva-skills / a notebook /
hydra-zen runner) computes the URL/checksum/version locally and passes
them in. We never run server-side git introspection on the MCP host.
"""

from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING, Any

from deriva_mcp_core import deriva_call
from deriva_mcp_core.telemetry import audit_event
from deriva_ml.core.exceptions import DerivaMLException

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
    _read_rid,
)
from deriva_ml_mcp.ml_context import get_ml

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def _summarize_workflow(wf: Any) -> dict[str, Any]:
    """Render a Workflow object into the JSON-friendly summary used by list endpoints.

    Args:
        wf: A ``deriva_ml.execution.workflow.Workflow`` instance.

    Returns:
        Dict with ``rid``, ``name``, ``url``, ``checksum``, ``version``,
        ``workflow_type`` (always a list), and ``description``.

    Example:
        >>> from unittest.mock import MagicMock
        >>> wf = MagicMock()
        >>> wf.rid = "1-WF"
        >>> wf.name = "MyPipeline"
        >>> wf.url = "https://github.com/example/repo"
        >>> wf.checksum = "abc123"
        >>> wf.version = "1.0.0"
        >>> wf.workflow_type = ["Model_Training"]
        >>> wf.description = "trains things"
        >>> _summarize_workflow(wf)["rid"]
        '1-WF'
    """
    return {
        "rid": wf.rid,
        "name": wf.name,
        "url": wf.url,
        "checksum": wf.checksum,
        "version": wf.version,
        "workflow_type": list(wf.workflow_type) if wf.workflow_type else [],
        "description": wf.description,
    }


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
    async def list_workflows(
        hostname: str,
        catalog_id: str,
        limit: int = 100,
        after_rid: str | None = None,
        preflight_count: bool = False,
    ) -> str:
        """Browse all workflows registered in the catalog.

        PAGINATION: When the count is unknown, call with
        ``preflight_count=True`` first. Then choose a limit and call
        again with ``preflight_count=False``. Use ``after_rid`` (the
        RID of the last workflow from the previous page) to advance
        the cursor.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            limit: Max workflows per page (default 100, max 1000).
            after_rid: RID of last row from previous page to advance cursor.
            preflight_count: If True, return only total count.

        Returns:
            JSON string. Preflight:
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
                ml = get_ml(hostname, catalog_id)
                workflows = sorted(ml.find_workflows(), key=lambda w: w.rid)

            if preflight_count:
                total = len(workflows)
                return json.dumps(
                    {
                        "total_count": total,
                        "entities_fetched": False,
                        "action_required": (
                            f"Found {total} workflows. Choose a limit and call "
                            "again with preflight_count=False."
                        ),
                    }
                )

            capped = min(max(limit, 0), _MAX_LIMIT)
            page, truncated, next_after = _paginate(
                workflows,
                after_rid=after_rid,
                limit=capped,
                key=partial(_read_rid, rid_key="rid"),
            )
            return json.dumps(
                {
                    "workflows": [_summarize_workflow(w) for w in page],
                    "count": len(page),
                    "truncated": truncated,
                    "next_after_rid": next_after,
                }
            )
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
    async def get_workflow(
        hostname: str,
        catalog_id: str,
        workflow_rid: str,
    ) -> str:
        """Read full details of one workflow by RID.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
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
                ml = get_ml(hostname, catalog_id)
                wf = ml.lookup_workflow(workflow_rid)
                summary = _summarize_workflow(wf)
            return json.dumps(summary)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="get_workflow",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def find_workflow_by_url(
        hostname: str,
        catalog_id: str,
        url_or_checksum: str,
    ) -> str:
        """Look up a workflow by Git URL or checksum.

        Critical for hydra-zen configs that carry only the URL. The URL
        should be a GitHub URL pointing to a specific commit, or the Git
        object hash (checksum) of the workflow file.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
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
                ml = get_ml(hostname, catalog_id)
                wf = ml.lookup_workflow_by_url(url_or_checksum)
                summary = _summarize_workflow(wf)
            return json.dumps(summary)
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
    async def create_workflow(
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

        Boundary rule: the caller (deriva-skills / notebook / hydra-zen
        runner) computes the Git URL, checksum, and version locally and
        passes them in. The MCP server never runs git introspection.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
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
                ml = get_ml(hostname, catalog_id)
                # Pre-check: dedup on URL/checksum. lookup_workflow_by_url
                # raises DerivaMLException when nothing matches; we treat
                # that as "not found" and proceed to create.
                rid: str
                status: str
                try:
                    existing = ml.lookup_workflow_by_url(url)
                except DerivaMLException:
                    existing = None
                if existing is not None:
                    rid = existing.rid
                    status = "exists"
                else:
                    wf = ml.create_workflow(
                        name=name,
                        workflow_type=workflow_type,
                        description=description,
                    )
                    wf.url = url
                    wf.checksum = checksum
                    wf.version = version
                    rid = ml._add_workflow(wf)
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
            return json.dumps(
                {
                    "status": status,
                    "workflow_rid": rid,
                    "name": name,
                    "workflow_type": normalized_types,
                    "url": url,
                    "checksum": checksum,
                    "version": version,
                }
            )
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
    async def update_workflow(
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
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            workflow_rid: The RID of the workflow to update.
            description: New description text. None leaves unchanged.
            workflow_type: New list of workflow_type tag terms. None
                leaves unchanged. Empty list is rejected (callers should
                pass None for "no change", not ``[]``).

        Returns:
            JSON string ``{"status": "updated", "workflow_rid",
            "updated_fields": [...]}``. ``updated_fields`` is the list of
            field names that were actually written.

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
                ml = get_ml(hostname, catalog_id)
                wf = ml.lookup_workflow(workflow_rid)
                if description is not None:
                    wf.description = description
                    updated_fields.append("description")
                if workflow_type is not None:
                    wf.workflow_type = workflow_type
                    updated_fields.append("workflow_type")
            audit_event(
                "deriva_ml_update_workflow",
                hostname=hostname,
                catalog_id=catalog_id,
                workflow_rid=workflow_rid,
                updated_fields=updated_fields,
                workflow_type=workflow_type,
            )
            return json.dumps(
                {
                    "status": "updated",
                    "workflow_rid": workflow_rid,
                    "updated_fields": updated_fields,
                }
            )
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="update_workflow",
                hostname=hostname,
                catalog_id=catalog_id,
                workflow_rid=workflow_rid,
                updated_fields=updated_fields,
            )
