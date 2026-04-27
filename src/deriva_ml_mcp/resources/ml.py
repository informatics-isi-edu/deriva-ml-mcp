"""ML-domain MCP resources for deriva-ml-mcp.

Resources are read-only URI-addressable views into ML-domain catalog
state. The MCP client fetches them via ``read_resource(uri)`` without
a tool call. They are suitable for stable, reference-like data where
the caller already knows the URI shape.

Resources differ from tools in three respects:

1. **No mutation.** Resources are implicitly read-only. There is no
   ``mutates=`` kwarg and no audit emission on success or failure.
2. **No query string.** MCP resources are pure path-templated URIs;
   they cannot accept pagination cursors or filters. Each list-style
   resource is bounded by ``_MAX_LIMIT`` (1000 rows) and the response
   includes ``truncated`` so the LLM knows to switch to the equivalent
   paginated tool when the cap is hit.
3. **Convenience aggregation.** Detail resources may bundle data that
   would otherwise be several tool calls into one payload (see
   ``ml/dataset/{rid}`` and ``ml/execution/{rid}``).

Resources registered:

    deriva://catalog/{hostname}/{catalog_id}/ml/datasets
    deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}
    deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/members
    deriva://catalog/{hostname}/{catalog_id}/ml/workflows
    deriva://catalog/{hostname}/{catalog_id}/ml/workflow/{workflow_rid}
    deriva://catalog/{hostname}/{catalog_id}/ml/executions
    deriva://catalog/{hostname}/{catalog_id}/ml/execution/{execution_rid}
    deriva://catalog/{hostname}/{catalog_id}/ml/features/{table_name}
    deriva://catalog/{hostname}/{catalog_id}/ml/asset-tables
    deriva://catalog/{hostname}/{catalog_id}/ml/asset/{asset_rid}
    deriva://catalog/{hostname}/{catalog_id}/ml/registries
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from deriva_mcp_core import deriva_call
from deriva_ml.core.enums import MLVocab

from deriva_ml_mcp._helpers import _MAX_LIMIT, _error_envelope
from deriva_ml_mcp.ml_context import get_ml
from deriva_ml_mcp.tools.asset import (
    _get_asset_detail_impl,
    _list_asset_tables_impl,
)
from deriva_ml_mcp.tools.dataset import (
    _get_dataset_detail_impl,
    _list_dataset_members_summary_impl,
    _list_datasets_impl,
)
from deriva_ml_mcp.tools.execution import (
    _get_execution_detail_impl,
    _list_executions_impl,
)
from deriva_ml_mcp.tools.feature import _list_features_impl
from deriva_ml_mcp.tools.workflow import _get_workflow_impl, _list_workflows_impl

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def _vocab_terms(ml: Any, vocab_name: str) -> list[dict[str, Any]]:
    """Read all terms from one vocabulary table as compact dicts.

    Returns ``name`` + ``rid`` only, by design. The full description and
    synonym lists are available via core's ``get_term`` tool or via
    ``rag_search`` if the LLM needs them; bundling them into the
    registries snapshot would make this resource ~12 KB per fetch
    (4 vocabularies x ~20 terms x ~150-byte descriptions) for a payload
    that the LLM reads on every "what types are available" question.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        vocab_name: Vocabulary table name (e.g. ``"Dataset_Type"``).

    Returns:
        List of dicts ``[{"name", "rid"}, ...]``. Empty list if the
        vocab table is missing or has no terms; suppresses upstream
        errors so a missing vocab does not nuke the whole registries
        snapshot.
    """
    try:
        terms = ml.list_vocabulary_terms(vocab_name)
    except Exception:  # noqa: BLE001 -- registry is best-effort
        return []
    return [
        {
            "name": getattr(t, "name", None),
            "rid": getattr(t, "rid", None),
        }
        for t in terms
    ]


def register(ctx: PluginContext) -> None:
    """Register all ML-domain resource handlers with the plugin context.

    Args:
        ctx: PluginContext supplied by deriva-mcp-core at startup.

    Returns:
        None.

    Example:
        >>> from deriva_mcp_core.plugin.api import PluginContext
        >>> # ctx provided by the framework
        >>> register(ctx)  # doctest: +SKIP
    """

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/ml/datasets")
    async def ml_datasets(hostname: str, catalog_id: str) -> str:
        """Snapshot of all datasets in the catalog (up to 1000 rows).

        Returns ``{"datasets": [...], "count", "truncated",
        "next_after_rid"}``. When ``truncated`` is True, switch to the
        ``deriva_ml_list_datasets`` tool for full cursor pagination.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                payload = _list_datasets_impl(
                    ml,
                    after_rid=None,
                    limit=_MAX_LIMIT,
                )
            return json.dumps(payload)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_datasets",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}")
    async def ml_dataset_detail(hostname: str, catalog_id: str, dataset_rid: str) -> str:
        """Detail payload for one dataset: summary + chaise URL + version_history.

        Absorbs the old ``deriva://dataset/{rid}/versions`` resource: full
        version history is included as the ``version_history`` key. The
        history is bounded per dataset so no pagination is needed.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                payload = _get_dataset_detail_impl(ml, dataset_rid)
            return json.dumps(payload)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_dataset_detail",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/members")
    async def ml_dataset_members(hostname: str, catalog_id: str, dataset_rid: str) -> str:
        """Members of a dataset grouped by table, with a flattened sample list.

        Returns per-table counts, the full table list, and a flattened
        ``members`` list of ``{table, rid}`` capped at 1000 rows. When
        ``truncated`` is True (the catalog has more than 1000 members),
        use the ``deriva_ml_list_dataset_members`` tool with pagination.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                payload = _list_dataset_members_summary_impl(ml, dataset_rid)
            return json.dumps(payload)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_dataset_members",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/ml/workflows")
    async def ml_workflows(hostname: str, catalog_id: str) -> str:
        """Snapshot of all workflows in the catalog (up to 1000 rows).

        When ``truncated`` is True, use the ``deriva_ml_list_workflows`` tool for
        cursor-paginated access.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                payload = _list_workflows_impl(ml, after_rid=None, limit=_MAX_LIMIT)
            return json.dumps(payload)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_workflows",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/ml/workflow/{workflow_rid}")
    async def ml_workflow_detail(hostname: str, catalog_id: str, workflow_rid: str) -> str:
        """Detail payload for one workflow: name, type, url, checksum, version, etc."""
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                payload = _get_workflow_impl(ml, workflow_rid)
            return json.dumps(payload)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_workflow_detail",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/ml/executions")
    async def ml_executions(hostname: str, catalog_id: str) -> str:
        """Snapshot of all executions in the catalog (up to 1000 rows).

        When ``truncated`` is True, use the ``deriva_ml_list_executions`` tool for
        cursor-paginated access (it also supports filtering by workflow
        and status).
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                payload = _list_executions_impl(
                    ml,
                    workflow_rid=None,
                    status=None,
                    after_rid=None,
                    limit=_MAX_LIMIT,
                )
            return json.dumps(payload, default=str)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_executions",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/ml/execution/{execution_rid}")
    async def ml_execution_detail(hostname: str, catalog_id: str, execution_rid: str) -> str:
        """Detail payload for one execution.

        Absorbs the old ``execution/{rid}/inputs``, ``/outputs``,
        ``/metadata`` and ``experiment/{rid}`` URIs. The payload bundles
        the execution summary plus ``inputs`` and ``outputs``. An
        ``experiment`` key is added when the execution is a Hydra-driven
        experiment (carries name + config_choices + model_config; the
        full hydra_config dict is NOT surfaced -- it can be 10-100 KB,
        and callers wanting it should fetch the metadata asset directly).
        The ``metadata`` key is omitted entirely until deriva-ml provides
        a generic enumerator for ``Execution_Metadata`` files (TODO in
        ``_get_execution_detail_impl``).
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                payload = _get_execution_detail_impl(ml, execution_rid)
            return json.dumps(payload, default=str)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_execution_detail",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/ml/features/{table_name}")
    async def ml_features_for_table(hostname: str, catalog_id: str, table_name: str) -> str:
        """Features defined on the given target table (up to 1000 rows).

        When ``truncated`` is True, use the ``deriva_ml_list_features`` tool with
        pagination.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                payload = _list_features_impl(
                    ml,
                    table=table_name,
                    after_rid=None,
                    limit=_MAX_LIMIT,
                )
            return json.dumps(payload)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_features_for_table",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/ml/asset-tables")
    async def ml_asset_tables(hostname: str, catalog_id: str) -> str:
        """Snapshot of all asset tables in the catalog.

        Catalogs typically have only a handful of asset tables, so the
        full set is returned (no pagination, no ``truncated`` flag).
        For paginated row-level browsing of one asset table use the
        ``deriva_ml_list_assets`` tool.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                payload = _list_asset_tables_impl(ml)
            return json.dumps(payload)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_asset_tables",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/ml/asset/{asset_rid}")
    async def ml_asset_detail(hostname: str, catalog_id: str, asset_rid: str) -> str:
        """Detail payload for one asset: summary + bundled executions.

        Same shape as ``deriva_ml_lookup_asset`` -- the resource and
        tool share an internal helper so the two can never drift.

        Note on the ``executions`` list: each entry has
        ``{rid, asset_role}``, but ``asset_role`` is best-effort and
        commonly ``None`` (deriva-ml's ``ExecutionRecord`` doesn't
        carry the per-asset role attribute by default). A ``None``
        role is not an error -- it means the relationship exists but
        the role wasn't surfaced through the upstream API.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                payload = _get_asset_detail_impl(ml, asset_rid)
            return json.dumps(payload)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_asset_detail",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/ml/registries")
    async def ml_registries(hostname: str, catalog_id: str) -> str:
        """Bundled snapshot of the four catalog ML vocabularies.

        Returns ``{"dataset_types", "workflow_types", "asset_types",
        "execution_statuses"}``, each a list of vocabulary terms.
        Missing or empty vocabularies surface as ``[]`` (best-effort);
        no exception aborts the snapshot.
        """
        try:
            with deriva_call():
                ml = get_ml(hostname, catalog_id)
                payload = {
                    "dataset_types": _vocab_terms(ml, MLVocab.dataset_type.value),
                    "workflow_types": _vocab_terms(ml, MLVocab.workflow_type.value),
                    "asset_types": _vocab_terms(ml, MLVocab.asset_type.value),
                    "execution_statuses": _vocab_terms(ml, MLVocab.execution_status.value),
                }
            return json.dumps(payload)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_registries",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )
