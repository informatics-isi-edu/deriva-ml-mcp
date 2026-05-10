"""Read-only execution tools and shared execution helpers.

This submodule houses the 5 read tools (``deriva_ml_list_executions``,
``deriva_ml_get_execution``, ``deriva_ml_find_workflow_executions``,
``deriva_ml_list_execution_children``, ``deriva_ml_list_execution_parents``)
plus the three helpers (``_summarize_execution``,
``_list_executions_impl``, ``_get_execution_detail_impl``) that the
read tools and the ``resources/ml.py`` / ``resources/rag.py`` modules
consume to keep tool / resource shapes in sync.

The helpers live here (not in a sibling ``_helpers.py``) because they
are execution-specific and the resource modules already import them via
``deriva_ml_mcp.tools.execution`` -- the package ``__init__.py`` re-exports
them so that import path keeps resolving identically after the split.

Every tool wraps DERIVA I/O in ``with deriva_call():`` and routes errors
through ``_error_envelope`` (reads only log on failure, no audit row).
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING, Any

from deriva_mcp_core import deriva_call
from deriva_ml.execution.execution import ExecutionStatus

# ``get_ml`` is accessed inside tool bodies via attribute lookup on the
# parent package (``_pkg.get_ml``) so a single
# ``patch("deriva_ml_mcp.tools.execution.get_ml")`` (used by the
# ``execution_ctx`` fixture in tests/test_execution.py) redirects every
# call across read / mutate submodules. A direct
# ``from deriva_ml_mcp.ml_context import get_ml`` here would create a
# per-submodule binding the patch can't reach.
import deriva_ml_mcp.tools.execution as _pkg  # noqa: E402  (intentional cycle)
from deriva_ml_mcp._helpers import (
    _MAX_LIMIT,
    _cite_dataset_version_url,
    _cite_url,
    _error_envelope,
    _paginate,
    _read_rid,
)
from deriva_ml_mcp._response_models import (
    ExecutionAssetRef,
    ExecutionChildrenResponse,
    ExecutionDetail,
    ExecutionExperiment,
    ExecutionInputDatasetRef,
    ExecutionInputs,
    ExecutionListResponse,
    ExecutionOutputs,
    ExecutionParentsResponse,
    ExecutionSummary,
    PreflightCountResponse,
)

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def _summarize_execution(record: Any) -> ExecutionSummary:
    """Render an ``ExecutionRecord`` (or ``Execution``) into the validated summary.

    Tolerates both ``ExecutionRecord`` (returned by ``lookup_execution``
    / ``find_executions``) and ``Execution`` (returned by
    ``resume_execution``) — the only attributes read are common to both
    surfaces or fall back to ``None``.

    Args:
        record: An ``ExecutionRecord`` (preferred) or ``Execution`` mock.

    Returns:
        ``ExecutionSummary`` Pydantic instance -- see
        ``deriva_ml_mcp._response_models``.

    Note:
        v2.2 sweep: this helper now returns Pydantic. Consumers that
        previously dict-accessed must use attribute access or call
        ``.model_dump()`` to get a plain dict back.

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
        >>> _summarize_execution(rec).rid
        '1-EXEC'
        >>> _summarize_execution(rec).status
        'Created'
    """
    status = getattr(record, "status", None)
    return ExecutionSummary(
        rid=getattr(record, "execution_rid", None),
        workflow_rid=getattr(record, "workflow_rid", None),
        status=status.value if isinstance(status, ExecutionStatus) else status,
        description=getattr(record, "description", None),
        start_time=getattr(record, "start_time", None),
        stop_time=getattr(record, "stop_time", None),
        duration=getattr(record, "duration", None),
    )


def _list_executions_impl(
    ml: Any,
    *,
    workflow_rid: str | None,
    status: str | None,
    after_rid: str | None,
    limit: int,
    sort: bool = False,
) -> ExecutionListResponse:
    """Fetch + paginate executions. Pure helper -- shared by tool and resource.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        workflow_rid: Optional workflow filter.
        status: Optional ``ExecutionStatus`` value (string).
        after_rid: Cursor for cursor pagination.
        limit: Max executions per page (already capped by caller).
        sort: If True, results are ordered newest-first by record
            creation time (RCT desc) -- forwarded to
            ``deriva_ml.DerivaML.find_executions(sort=True)``. If False
            (default), results are RID-ascending for stable cursor
            pagination. Note that under ``sort=True`` the ``after_rid``
            cursor still works ("skip up to this RID in the RCT-sorted
            result"), but pagination through very large sorted result
            sets is bounded by the internal fetch cap.

    Returns:
        ``ExecutionListResponse`` -- see ``deriva_ml_mcp._response_models``.
    """
    status_enum = ExecutionStatus(status) if status else None
    raw = list(
        ml.find_executions(
            workflow=workflow_rid,
            status=status_enum,
            sort=True if sort else None,
        )
    )
    # Keep stable RID-ascending order for the default path; under
    # sort=True we honor the catalog-side RCT-desc ordering and skip
    # the post-fetch sort (sorting by RID would clobber the RCT order).
    if sort:
        executions = raw
    else:
        executions = sorted(
            raw,
            key=lambda e: getattr(e, "execution_rid", "") or "",
        )
    page, truncated, next_after = _paginate(
        executions,
        after_rid=after_rid,
        limit=limit,
        key=partial(_read_rid, rid_key="execution_rid"),
    )
    return ExecutionListResponse(
        executions=[_summarize_execution(e) for e in page],
        count=len(page),
        truncated=truncated,
        next_after_rid=next_after,
    )


def _get_execution_detail_impl(ml: Any, execution_rid: str) -> ExecutionDetail:
    """Build the execution detail payload (summary + inputs + outputs + experiment).

    Used by the ``deriva://catalog/{h}/{c}/ml/execution/{rid}`` resource.
    Aggregates input datasets, asset I/O grouped by role, and an
    optional ``experiment`` key for executions that are Hydra-driven
    experiments.

    The deriva-ml ``ExecutionRecord`` exposes:

    - ``list_input_datasets()`` -> list of Dataset objects
    - ``list_assets(asset_role="Input"|"Output"|None)`` -> list of Asset
      objects

    The ``metadata`` key is omitted entirely until deriva-ml provides a
    generic enumerator for ``Execution_Metadata`` files (Hydra config,
    Deriva config, etc. are stored as Asset rows joined through
    ``Execution_Metadata`` -- not addressable through ``list_assets``'s
    ``asset_role`` filter, which only handles Input/Output). The
    ``experiment`` key is omitted when the execution has no
    ``Experiment`` row (the common case); when present it surfaces the
    cheap accessor fields (``name`` / ``config_choices`` /
    ``model_config``) but NOT the full hydra_config dict (potentially
    large -- callers wanting it should fetch the metadata asset).

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        execution_rid: The RID of the execution to look up.

    Returns:
        ``ExecutionDetail`` Pydantic model -- see
        ``deriva_ml_mcp._response_models``. Wire shape matches v1.x
        verbatim: ``{...summary fields..., "inputs": {"datasets":
        [...], "assets": [...]}, "outputs": {"assets": [...]},
        "experiment": {...} | null}``.

        v2.0 wire change vs v1.x: the ``experiment`` key is now
        always present (set to ``null`` when the execution is not a
        Hydra-driven experiment). v1.x omitted the key entirely. This
        is a small breaking change but the typed contract makes
        introspection easier.
    """
    record = ml.lookup_execution(execution_rid)
    summary = _summarize_execution(record)

    # Inputs: datasets + input assets.
    input_datasets: list[ExecutionInputDatasetRef] = []
    try:
        ds_iter = record.list_input_datasets()
    except Exception:  # noqa: BLE001 -- record may not be bound on all paths
        ds_iter = []
    for ds in ds_iter:
        try:
            current = getattr(ds, "current_version", None)
            version_str = str(current) if current is not None else None
            cite = _cite_dataset_version_url(ml, ds.dataset_rid, version_str)
            input_datasets.append(
                ExecutionInputDatasetRef(
                    rid=ds.dataset_rid,
                    version=version_str,
                    # Pydantic rejects non-string cite values (e.g. when
                    # ml.cite is mocked and returns a MagicMock); coerce
                    # defensively rather than dropping the whole entry.
                    cite_url=cite if isinstance(cite, str) else None,
                )
            )
        except Exception:  # noqa: BLE001 -- per-row presentation field
            continue

    # ``record.list_assets(asset_role=...)`` already walks every
    # ``*_Execution`` association table -- including
    # ``Execution_Asset_Execution`` AND ``Execution_Metadata_Execution``
    # -- so a single call returns both kinds of outputs mixed together.
    # We categorize them by the asset row's ``asset_table`` attribute:
    # ``"Execution_Metadata"`` rows go into ``outputs.metadata`` (Hydra
    # configs, env snapshots, uv.lock, etc.); everything else goes into
    # ``outputs.assets`` (the run's real products: model weights,
    # prediction CSVs, training logs, plots).
    #
    # Inputs do not need this split today -- the caller-facing
    # ``inputs.assets`` represents user-supplied input files and these
    # are conventionally not ``Execution_Metadata`` rows. If that
    # assumption ever breaks, mirror the categorization on the input
    # branch.
    input_assets: list[ExecutionAssetRef] = []
    output_assets: list[ExecutionAssetRef] = []
    output_metadata: list[ExecutionAssetRef] = []

    def _ref(asset: Any) -> ExecutionAssetRef:
        """Build a ref from a deriva-ml ``Asset`` object.

        Uses ``getattr`` with safe defaults so a thinly-mocked or
        older-API asset (missing description / asset_types) still
        produces a well-formed ref rather than raising.
        """
        rid = getattr(asset, "asset_rid", None)
        cite = _cite_url(ml, rid) if rid else None
        return ExecutionAssetRef(
            rid=rid,
            filename=getattr(asset, "filename", None),
            description=getattr(asset, "description", None),
            asset_types=list(getattr(asset, "asset_types", []) or []),
            asset_table=getattr(asset, "asset_table", None),
            # Pydantic rejects non-string cite values (e.g. when
            # ml.cite is mocked and returns a MagicMock); coerce
            # defensively rather than dropping the whole entry.
            cite_url=cite if isinstance(cite, str) else None,
        )

    try:
        in_iter = list(record.list_assets(asset_role="Input"))
    except Exception:  # noqa: BLE001 -- assets are optional
        in_iter = []
    try:
        out_iter = list(record.list_assets(asset_role="Output"))
    except Exception:  # noqa: BLE001 -- assets are optional
        out_iter = []

    for asset in in_iter:
        try:
            input_assets.append(_ref(asset))
        except Exception:  # noqa: BLE001 -- per-row presentation field
            continue
    for asset in out_iter:
        try:
            ref = _ref(asset)
        except Exception:  # noqa: BLE001 -- per-row presentation field
            continue
        if ref.asset_table == "Execution_Metadata":
            output_metadata.append(ref)
        else:
            output_assets.append(ref)

    # Experiment: try lookup_experiment(execution_rid). The deriva-ml
    # API raises if the execution has no Experiment row; treat that as
    # "not an experiment" and set experiment=None. When present,
    # surface the cheap fields (name + config_choices + model_config)
    # but NOT the full hydra_config payload -- it can be 10-100 KB and
    # a caller wanting it should fetch the metadata asset directly.
    experiment: ExecutionExperiment | None = None
    try:
        exp = ml.lookup_experiment(execution_rid)
        experiment = ExecutionExperiment.model_validate(
            {
                "name": getattr(exp, "name", None),
                "config_choices": getattr(exp, "config_choices", {}) or {},
                # The wire key is "model_config" (alias) -- pass the dict
                # under the wire-key form to take advantage of model_validate's
                # alias-aware construction.
                "model_config": getattr(exp, "model_config", {}) or {},
            }
        )
    except Exception:  # noqa: BLE001 -- absent experiment is the common case
        experiment = None

    return ExecutionDetail(
        **summary.model_dump(),
        inputs=ExecutionInputs(datasets=input_datasets, assets=input_assets),
        outputs=ExecutionOutputs(assets=output_assets, metadata=output_metadata),
        experiment=experiment,
    )


def register(ctx: PluginContext) -> None:
    """Register the read-only execution tools with the plugin context.

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
    async def deriva_ml_list_executions(
        hostname: str,
        catalog_id: str,
        workflow_rid: str | None = None,
        status: str | None = None,
        limit: int = 100,
        after_rid: str | None = None,
        preflight_count: bool = False,
        sort: bool = False,
    ) -> str:
        """Browse executions in the catalog, optionally filtered by workflow or status.

        See ``deriva_ml_getting_started`` (PAGINATION CONTRACT) for the two-step pagination flow.

        Args:
            workflow_rid: If set, return only executions of this workflow.
            status: If set, restrict to one ``ExecutionStatus`` value
                (e.g. ``"Running"``, ``"Uploaded"``).
            limit: Max executions per page (default 100, max 1000).
            after_rid: RID of last row from previous page to advance cursor.
            preflight_count: If True, return only total count.
            sort: If True, return results newest-first by record
                creation time. Recommended for "show me the most
                recent runs" queries. Default False preserves the
                stable RID-ascending order used for cursor pagination.

        Returns:
            Page: ``{"executions": [...], "count",
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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``ml.find_executions``
                # returns a generator that materializes over the wire, so
                # the drain (list) must happen inside the worker thread,
                # not on the event loop. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                if preflight_count:
                    status_enum = ExecutionStatus(status) if status else None

                    def _count_executions() -> int:
                        return len(list(ml.find_executions(workflow=workflow_rid, status=status_enum)))

                    total = await asyncio.to_thread(_count_executions)
                    return PreflightCountResponse(
                        total_count=total,
                        action_required=(
                            f"Found {total} executions. Choose a limit and call "
                            "again with preflight_count=False."
                        ),
                    ).model_dump_json(by_alias=True)
                capped = min(max(limit, 0), _MAX_LIMIT)
                payload = await asyncio.to_thread(
                    _list_executions_impl,
                    ml,
                    workflow_rid=workflow_rid,
                    status=status,
                    after_rid=after_rid,
                    limit=capped,
                    sort=sort,
                )
            return payload.model_dump_json(by_alias=True)
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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``_summarize_execution``
                # is a pure Pydantic build over already-fetched attrs and
                # stays on the event loop. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                record = await asyncio.to_thread(ml.lookup_execution, execution_rid)
                summary = _summarize_execution(record)
            return summary.model_dump_json(by_alias=True)
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
        sort: bool = False,
    ) -> str:
        """Find all executions of a specific workflow.

        Distinct from ``deriva_ml_list_executions(workflow_rid=...)`` to surface
        the workflow-centric query as a first-class tool — the LLM
        intent ("show me runs of this workflow") differs from the
        general "browse executions" intent.

        Args:
            workflow_rid: The RID of the workflow whose executions to list.
            status: Optional ``ExecutionStatus`` filter.
            limit: Max executions per page (default 100, max 1000).
            after_rid: RID of last row from previous page to advance cursor.
            preflight_count: If True, return only total count.
            sort: If True, return results newest-first by record
                creation time. Recommended for "show me the most
                recent runs" queries. Default False preserves the
                stable RID-ascending order used for cursor pagination.

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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``ml.find_executions``
                # returns a generator that materializes over the wire, so
                # the drain (list) must happen inside the worker thread,
                # not on the event loop. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                if preflight_count:
                    status_enum = ExecutionStatus(status) if status else None

                    def _count_executions() -> int:
                        return len(list(ml.find_executions(workflow=workflow_rid, status=status_enum)))

                    total = await asyncio.to_thread(_count_executions)
                    return PreflightCountResponse(
                        total_count=total,
                        action_required=(
                            f"Found {total} executions for workflow {workflow_rid}. "
                            "Choose a limit and call again with preflight_count=False."
                        ),
                    ).model_dump_json(by_alias=True)

                capped = min(max(limit, 0), _MAX_LIMIT)
                payload = await asyncio.to_thread(
                    _list_executions_impl,
                    ml,
                    workflow_rid=workflow_rid,
                    status=status,
                    after_rid=after_rid,
                    limit=capped,
                    sort=sort,
                )
            # v2.3 reuses ExecutionListResponse here -- the
            # find_workflow_executions response is shape-identical to
            # _list_executions_impl's response (filtered by workflow).
            return payload.model_dump_json(by_alias=True)
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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``list_execution_children``
                # returns a generator that materializes over the wire, so
                # the drain must happen inside the worker thread. See
                # deriva-mcp-core plugin-authoring-guide.md §"Synchronous
                # work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                record = await asyncio.to_thread(ml.lookup_execution, execution_rid)

                def _drain_children() -> list[Any]:
                    return list(record.list_execution_children(recurse=recurse))

                children = await asyncio.to_thread(_drain_children)
            return ExecutionChildrenResponse(
                parent_rid=execution_rid,
                recurse=recurse,
                count=len(children),
                children=[_summarize_execution(c) for c in children],
            ).model_dump_json(by_alias=True)
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
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``list_execution_parents``
                # returns a generator that materializes over the wire, so
                # the drain must happen inside the worker thread. See
                # deriva-mcp-core plugin-authoring-guide.md §"Synchronous
                # work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                record = await asyncio.to_thread(ml.lookup_execution, execution_rid)

                def _drain_parents() -> list[Any]:
                    return list(record.list_execution_parents(recurse=recurse))

                parents = await asyncio.to_thread(_drain_parents)
            return ExecutionParentsResponse(
                child_rid=execution_rid,
                recurse=recurse,
                count=len(parents),
                parents=[_summarize_execution(p) for p in parents],
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="list_execution_parents",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.tool(mutates=False)
    async def deriva_ml_get_lineage(
        hostname: str,
        catalog_id: str,
        rid: str,
        depth: int | None = None,
        max_executions: int = 500,
    ) -> str:
        """Walk the data-flow provenance chain behind an artifact.

        Given a Dataset, Asset, Feature value, or Execution RID,
        returns a tree of producing executions and their consumed
        inputs back to the natural root of every branch. Replaces
        what would otherwise be 5-15 round-trips through typed read
        methods with one call.

        The walk follows **data-flow parents only**: for each
        execution node, the parents are the producing executions of
        its consumed datasets and assets. This tool does NOT walk
        ``Execution_Execution`` (orchestration links) -- that's a
        different question (use ``deriva_ml_list_execution_parents``
        / ``deriva_ml_list_execution_children`` for the orchestration
        view). See deriva-ml ADR-0001 for the rationale.

        Same shape as the
        ``deriva://catalog/{h}/{c}/ml/lineage/{rid}`` resource -- the
        two share an internal helper so the payloads cannot drift.

        Args:
            rid: RID of any Dataset, Asset, Feature value, or Execution.
                Workflow RIDs are not lineage-shaped and produce a
                clear error.
            depth: Number of parent levels to walk from the immediate
                producing execution. ``None`` (default) walks to the
                root. ``0`` returns only the immediate producing
                execution node. ``N>0`` walks ``N`` levels up.
            max_executions: Defensive cap on distinct executions the
                walk will expand. Default 500. If exceeded,
                ``walked_complete`` is set to False and the partial
                tree is returned.

        Returns:
            JSON string of the LineageResult: ``{"root": {...}, "lineage":
            {... tree of producing executions ...}, "executions_visited":
            int, "walked_complete": bool, "cycle_detected": bool,
            "depth_capped": bool}``. ``lineage`` is ``None`` when the
            artifact has no producing-execution link (manually-inserted
            data with no provenance link).

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated
                from ``deriva_ml.DerivaML.lookup_lineage`` (RID not
                found, RID points at a Workflow, RID's table cannot
                be classified).

        Example:
            ``{"root": {"rid": "2-PRED1", "type": "Asset", ...},
            "lineage": {"execution": {...}, "consumed_datasets": [...],
            "consumed_assets": [...], "parents": [...]},
            "executions_visited": 3, "walked_complete": true,
            "cycle_detected": false, "depth_capped": false}``.
        """
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``_get_lineage_impl``
                # wraps ``ml.lookup_lineage`` which performs catalog I/O.
                # See deriva-mcp-core plugin-authoring-guide.md
                # §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(
                    _get_lineage_impl, ml, rid, depth, max_executions
                )
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="get_lineage",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )


def _get_lineage_impl(ml: Any, rid: str, depth: int | None, max_executions: int) -> Any:
    """Shared helper: tool ``deriva_ml_get_lineage`` and resource
    ``deriva://catalog/{h}/{c}/ml/lineage/{rid}`` both call this so
    their payloads cannot drift.

    Wraps ``ml.lookup_lineage(rid, depth=..., max_executions=...)``
    and returns the ``LineageResult`` Pydantic model unchanged.

    Args:
        ml: The ``DerivaML`` instance bound to the catalog.
        rid: Artifact RID.
        depth: Parent-walk depth cap (``None`` = unbounded).
        max_executions: Defensive cap on distinct executions visited.

    Returns:
        ``deriva_ml.execution.lineage.LineageResult``.

    Example:
        >>> _get_lineage_impl(ml, "2-PRED1", None, 500)  # doctest: +SKIP
    """
    return ml.lookup_lineage(rid, depth=depth, max_executions=max_executions)
