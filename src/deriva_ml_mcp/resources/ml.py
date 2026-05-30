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

    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/datasets
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/dataset/{dataset_rid}
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/dataset/{dataset_rid}/spec
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/dataset/{dataset_rid}/members
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/workflows
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/workflow/{workflow_rid}
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/executions
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/execution/{execution_rid}
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/lineage/{rid}
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/features/{table_name}
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/asset/{asset_rid}
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/assets/{schema}
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/assets/{schema}/{asset_table}
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/vocabularies/{schema}
    deriva://catalog/{hostname}/{catalog_id}/deriva-ml/vocabularies/{schema}/{vocab_name}
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING, Any

from deriva_mcp_core import deriva_call

import deriva_ml_mcp.resources.ml as _pkg  # noqa: E402  (intentional self-import)
from deriva_ml_mcp._helpers import (
    _MAX_LIMIT,
    _error_envelope,
    _paginate,
    _read_rid,
)
from deriva_ml_mcp._response_models import (
    AssetTableContentsResponse,
    AssetTableNameRef,
    AssetTablesInSchemaResponse,
    VocabularyTablesResponse,
    VocabularyTableSummary,
    VocabularyTermDetail,
    VocabularyTermsResponse,
)

# ``get_ml`` is accessed inside resource bodies via attribute lookup on
# the module (``_pkg.get_ml``) so tests patching
# ``deriva_ml_mcp.resources.ml.get_ml`` rebind the module attribute,
# and ``_pkg.get_ml`` (resolved at call time) sees the patched value.
# A direct ``from deriva_ml_mcp.ml_context import get_ml`` here would
# capture the unpatched reference at module-import time and bypass the
# patch -- same rationale as ``tools/dataset/read.py``.
from deriva_ml_mcp.ml_context import get_ml  # noqa: F401  (re-exported for test patching)
from deriva_ml_mcp.tools.asset import (
    _get_asset_detail_impl,
    _summarize_asset,
)
from deriva_ml_mcp.tools.dataset import (
    _bag_info_impl,
    _get_dataset_detail_impl,
    _get_dataset_spec_impl,
    _list_dataset_members_summary_impl,
    _list_datasets_impl,
)
from deriva_ml_mcp.tools.execution import (
    _get_execution_detail_impl,
    _get_lineage_impl,
    _list_executions_impl,
)
from deriva_ml_mcp.tools.feature import _list_features_impl
from deriva_ml_mcp.tools.workflow import _get_workflow_impl, _list_workflows_impl

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def _resolve_schema_table(ml: Any, schema: str, table_name: str) -> Any | None:
    """Look up ``schema.table_name`` in the catalog model, honoring the URI's schema.

    Unlike ``ml.model.name_to_table(name)`` which silently searches
    domain schemas, then ML schema, then WWW (priority order), this
    helper honors the URI's schema literally. Two same-named tables
    in different schemas are disambiguated by the caller.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        schema: Schema name from the URI segment.
        table_name: Table name from the URI segment.

    Returns:
        The ``deriva.core.ermrest_model.Table`` object, or ``None`` if
        the schema or the table within it is missing. The caller
        distinguishes the two cases via a separate check on
        ``ml.model.schemas``.
    """
    schema_obj = ml.model.schemas.get(schema)
    if schema_obj is None:
        return None
    return schema_obj.tables.get(table_name)


def _list_schema_vocabularies_impl(ml: Any, schema: str) -> VocabularyTablesResponse:
    """Walk one schema and return its vocabulary tables with term counts.

    For each vocab table, ``term_count`` is computed via
    ``len(ml.list_vocabulary_terms(table))``. A per-vocab fetch failure
    surfaces as ``term_count = None`` so one bad vocab does not abort
    the whole snapshot -- mirrors the pre-v3.4 ``_vocab_terms``
    best-effort pattern.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        schema: The schema name to scan.

    Returns:
        :class:`VocabularyTablesResponse` with one entry per vocab.
        Empty list when the schema exists but has no vocabularies.

    Raises:
        KeyError: If the schema does not exist in the catalog model.
            The wrapper translates this to an error envelope.
    """
    schema_obj = ml.model.schemas[schema]
    entries: list[VocabularyTableSummary] = []
    for table in schema_obj.tables.values():
        if not table.is_vocabulary():
            continue
        try:
            term_count: int | None = len(ml.list_vocabulary_terms(table))
        except Exception:  # noqa: BLE001 -- best-effort per-vocab count
            term_count = None
        entries.append(
            VocabularyTableSummary(
                name=table.name,
                rid=getattr(table, "rid", "") or "",
                term_count=term_count,
            )
        )
    return VocabularyTablesResponse(
        schema=schema,
        count=len(entries),
        vocabularies=entries,
    )


def _get_vocabulary_terms_impl(
    ml: Any,
    schema: str,
    vocab_name: str,
) -> VocabularyTermsResponse:
    """Read all terms from one vocabulary table.

    Honors the URI's schema literally (no fallback search). Validates
    that the table exists and is a vocabulary; the wrapper surfaces
    the failures as error envelopes with specific messages.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        schema: Schema containing the vocabulary table.
        vocab_name: Name of the vocabulary table.

    Returns:
        :class:`VocabularyTermsResponse` with all six fields per term
        (``name``, ``rid``, ``description``, ``synonyms``, ``id``,
        ``uri``).

    Raises:
        KeyError: If the schema or table does not exist.
        ValueError: If the table exists but is not a vocabulary.
    """
    table = _resolve_schema_table(ml, schema, vocab_name)
    if table is None:
        if schema not in ml.model.schemas:
            raise KeyError(f"schema not found: {schema}")
        raise KeyError(f"vocabulary not found: {schema}.{vocab_name}")
    if not table.is_vocabulary():
        raise ValueError(f"table {schema}.{vocab_name} is not a vocabulary")
    raw_terms = ml.list_vocabulary_terms(table)
    terms = [
        VocabularyTermDetail(
            name=t.name,
            rid=t.rid,
            description=t.description,
            synonyms=list(t.synonyms or []),
            id=getattr(t, "id", None),
            uri=getattr(t, "uri", None),
        )
        for t in raw_terms
    ]
    return VocabularyTermsResponse(
        schema=schema,
        vocabulary=vocab_name,
        count=len(terms),
        terms=terms,
    )


def _list_schema_asset_tables_impl(ml: Any, schema: str) -> AssetTablesInSchemaResponse:
    """Walk one schema and return its asset tables.

    Asset rows are not enumerated -- catalog asset tables can carry
    millions of rows, and counting them would require materializing
    every row through ``ml.list_assets``. The
    ``ml/assets/{schema}/{asset_table}`` resource carries
    ``truncated`` / ``next_after_rid`` so callers learn "this table is
    huge" the moment they fetch the snapshot.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        schema: The schema name to scan.

    Returns:
        :class:`AssetTablesInSchemaResponse` with one entry per asset
        table. Empty list when the schema exists but has no asset
        tables.

    Raises:
        KeyError: If the schema does not exist in the catalog model.
            The wrapper translates this to an error envelope.
    """
    schema_obj = ml.model.schemas[schema]
    entries: list[AssetTableNameRef] = []
    for table in schema_obj.tables.values():
        if not table.is_asset():
            continue
        entries.append(
            AssetTableNameRef(
                name=table.name,
                rid=getattr(table, "rid", "") or "",
            )
        )
    return AssetTablesInSchemaResponse(
        schema=schema,
        count=len(entries),
        asset_tables=entries,
    )


def _get_asset_table_contents_impl(
    ml: Any,
    schema: str,
    asset_table: str,
) -> AssetTableContentsResponse:
    """Read one page of asset rows from one asset table.

    Honors the URI's schema literally. Validates that the table exists
    and is an asset table. Pagination is bounded by ``_MAX_LIMIT``;
    ``truncated`` / ``next_after_rid`` are populated so callers can
    fall back to the paginated ``deriva_ml_list_assets`` tool when the
    table exceeds the cap.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        schema: Schema containing the asset table.
        asset_table: Name of the asset table.

    Returns:
        :class:`AssetTableContentsResponse` with the bounded page of
        ``AssetSummary`` rows (same per-row shape as
        ``deriva_ml_list_assets``).

    Raises:
        KeyError: If the schema or table does not exist.
        ValueError: If the table exists but is not an asset table.
    """
    table = _resolve_schema_table(ml, schema, asset_table)
    if table is None:
        if schema not in ml.model.schemas:
            raise KeyError(f"schema not found: {schema}")
        raise KeyError(f"asset table not found: {schema}.{asset_table}")
    if not table.is_asset():
        raise ValueError(f"table {schema}.{asset_table} is not an asset table")
    assets = sorted(
        ml.list_assets(table),
        key=lambda a: getattr(a, "asset_rid", "") or "",
    )
    page, truncated, next_after = _paginate(
        assets,
        after_rid=None,
        limit=_MAX_LIMIT,
        key=partial(_read_rid, rid_key="asset_rid"),
    )
    return AssetTableContentsResponse(
        schema=schema,
        asset_table=asset_table,
        assets=[_summarize_asset(a) for a in page],
        count=len(page),
        truncated=truncated,
        next_after_rid=next_after,
    )


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

    # ------------------------------------------------------------------
    # Cold-start orientation: same content as the
    # ``deriva_ml_getting_started`` and ``deriva_ml_concepts`` PROMPTS,
    # exposed as static RESOURCES so resource-walking clients (the
    # failure mode in the 2026-05-13 e2e test, Section A of the
    # 2026-05-16 findings doc) discover the orientation text without
    # going through the prompt subsystem. Two channels, one source of
    # truth: both handlers return the same constants used by the
    # prompts in ``prompts.py``. Findings ref: C1c.
    # ------------------------------------------------------------------

    # Late import: ``prompts.py`` imports nothing heavy at module scope,
    # but keeping the import inside ``register`` mirrors the per-resource
    # import discipline used by the catalog handlers below (no module-
    # level side effects at plugin load time).
    from deriva_ml_mcp.prompts import _CONCEPTS_GUIDE, _GETTING_STARTED_GUIDE

    @ctx.resource("deriva://deriva-ml/getting-started")
    async def deriva_ml_getting_started_resource() -> str:
        """Cold-start orientation guide for the deriva-ml-mcp plugin.

        Same content as the ``deriva_ml_getting_started`` MCP prompt --
        exposed here as a resource so clients that walk resources (and
        skip prompts) still see the (hostname, catalog_id) rule, the
        pagination contract, the resource-vs-tool decision, and the
        discovery-via-resources orientation. Read this before using any
        ``deriva_ml_*`` tool or ``deriva://catalog/.../deriva-ml/...`` resource.
        """
        return _GETTING_STARTED_GUIDE

    @ctx.resource("deriva://deriva-ml/concepts")
    async def deriva_ml_concepts_resource() -> str:
        """Conceptual frame for the DerivaML domain.

        Same content as the ``deriva_ml_concepts`` MCP prompt -- the
        five core abstractions (Dataset, Workflow, Execution, Feature,
        Asset), the provenance principle, and the vocabulary-extension
        pattern. Read this BEFORE ``deriva://deriva-ml/getting-started``
        if you do not already have a DerivaML mental model.
        """
        return _CONCEPTS_GUIDE

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/deriva-ml/datasets")
    async def ml_datasets(hostname: str, catalog_id: str) -> str:
        """Snapshot of all datasets in the catalog (up to 1000 rows).

        Returns ``{"datasets": [...], "count", "truncated",
        "next_after_rid"}``. When ``truncated`` is True, switch to the
        ``deriva_ml_list_datasets`` tool for full cursor pagination.
        """
        try:
            with deriva_call():
                # Sync deriva-py I/O runs in a worker thread so the
                # asyncio event loop is not blocked. See
                # ``tests/test_async_thread_wrap.py`` for the structural
                # rule that enforces this pattern.
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(
                    _list_datasets_impl,
                    ml,
                    after_rid=None,
                    limit=_MAX_LIMIT,
                )
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_datasets",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/deriva-ml/dataset/{dataset_rid}")
    async def ml_dataset_detail(hostname: str, catalog_id: str, dataset_rid: str) -> str:
        """Detail payload for one dataset: summary + chaise URL + version_history.

        Absorbs the old ``deriva://dataset/{rid}/versions`` resource: full
        version history is included as the ``version_history`` key. The
        history is bounded per dataset so no pagination is needed.
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(_get_dataset_detail_impl, ml, dataset_rid)
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_dataset_detail",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/deriva-ml/dataset/{dataset_rid}/spec")
    async def ml_dataset_spec(hostname: str, catalog_id: str, dataset_rid: str) -> str:
        """``DatasetSpecConfig(...)`` snippet for one dataset (current version).

        Same payload as ``deriva_ml_get_dataset_spec`` -- the resource
        and tool share an internal helper so the two cannot drift.
        Resource form omits the ``version`` parameter; the snippet
        always uses the dataset's current version (with a ``warning``
        recommending an explicit pin for reproducibility). Callers
        needing to pin a specific version use the tool.
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(_get_dataset_spec_impl, ml, dataset_rid, None)
            import json as _json

            return _json.dumps(payload)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_dataset_spec",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource(
        "deriva://catalog/{hostname}/{catalog_id}/deriva-ml/dataset/{dataset_rid}/bag-preview"
    )
    async def ml_dataset_bag_preview(hostname: str, catalog_id: str, dataset_rid: str) -> str:
        """Bag-content preview for one dataset (current version, no exclusions).

        Shares the ``_bag_info_impl`` helper with ``deriva_ml_bag_info``
        for the catalog-derived row counts, asset byte totals, and per-
        table breakdown -- the load-bearing "what's in this bag, what
        will it cost to download" data. The resource form omits the
        ``version`` and ``exclude_tables`` parameters (always current
        version, no exclusions) and includes a ``warning`` recommending
        an explicit version pin for reproducibility. Use the tool
        (``deriva_ml_bag_info``) when you need to pin a version or skip
        large blob tables.

        v0.5.0: the ``cache_status`` and ``cache_path`` fields are
        stripped from the resource response. Those fields describe the
        MCP SERVER's local cache, not the caller's -- a resource read
        cannot legitimately surface server-side cache state to a client
        that expects to see its own. Callers who want their local cache
        state should check it locally via ``ml.cache_status(...)``.

        Use to decide whether to download a bag locally (via
        ``ml.cache_dataset(spec)`` in user-local Python -- the MCP
        wire tool ``deriva_ml_cache_dataset`` was retired in v0.5.0
        per the stateless rule; bag materialization belongs in the
        caller's environment). The per-table row counts and asset
        byte totals are enough to decide cost without materializing
        the bag anywhere.

        Closes a documented-but-missing URI flagged in the 2026-05-13
        e2e findings (referenced by the debug-bag-contents skill).
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                info = await asyncio.to_thread(_bag_info_impl, ml, dataset_rid, None)
            # Strip server-side cache state per the stateless rule.
            # The helper populates these for the tool path (where the
            # caller IS the server, single-machine dev); the resource
            # path serves remote clients whose cache state lives
            # elsewhere. Drop unconditionally; the keys may or may not
            # be present in the underlying ml.bag_info dict depending
            # on the deriva-ml version.
            info.pop("cache_status", None)
            info.pop("cache_path", None)
            import json as _json

            return _json.dumps(info, default=str)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_dataset_bag_preview",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource(
        "deriva://catalog/{hostname}/{catalog_id}/deriva-ml/dataset/{dataset_rid}/members"
    )
    async def ml_dataset_members(hostname: str, catalog_id: str, dataset_rid: str) -> str:
        """Members of a dataset grouped by table, with a flattened sample list.

        Returns per-table counts, the full table list, and a flattened
        ``members`` list of ``{table, rid}`` capped at 1000 rows. When
        ``truncated`` is True (the catalog has more than 1000 members),
        use the ``deriva_ml_list_dataset_members`` tool with pagination.
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(
                    _list_dataset_members_summary_impl, ml, dataset_rid
                )
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_dataset_members",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/deriva-ml/workflows")
    async def ml_workflows(hostname: str, catalog_id: str) -> str:
        """Snapshot of all workflows in the catalog (up to 1000 rows).

        When ``truncated`` is True, use the ``deriva_ml_list_workflows`` tool for
        cursor-paginated access.
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(
                    _list_workflows_impl, ml, after_rid=None, limit=_MAX_LIMIT
                )
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_workflows",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/deriva-ml/workflow/{workflow_rid}")
    async def ml_workflow_detail(hostname: str, catalog_id: str, workflow_rid: str) -> str:
        """Detail payload for one workflow: name, type, url, checksum, version, etc."""
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(_get_workflow_impl, ml, workflow_rid)
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_workflow_detail",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/deriva-ml/executions")
    async def ml_executions(hostname: str, catalog_id: str) -> str:
        """Snapshot of all executions in the catalog (up to 1000 rows).

        When ``truncated`` is True, use the ``deriva_ml_list_executions`` tool for
        cursor-paginated access (it also supports filtering by workflow
        and status).
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(
                    _list_executions_impl,
                    ml,
                    workflow_rid=None,
                    workflow_type=None,
                    status=None,
                    after_rid=None,
                    limit=_MAX_LIMIT,
                )
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_executions",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/deriva-ml/execution/{execution_rid}")
    async def ml_execution_detail(hostname: str, catalog_id: str, execution_rid: str) -> str:
        """Detail payload for one execution.

        Absorbs the old ``execution/{rid}/inputs``, ``/outputs``,
        ``/metadata`` and ``experiment/{rid}`` URIs into a single
        bundle: the execution summary plus ``inputs`` and ``outputs``.

        Outputs are split into ``outputs.assets`` (real
        ``Execution_Asset`` files like model weights and prediction
        CSVs) and ``outputs.metadata`` (``Execution_Metadata`` files
        like ``configuration.json``, ``hydra-*.yaml``, ``uv.lock``, and
        the environment snapshot). Both lists are always present even
        when empty. An ``experiment`` key is added when the execution
        is a Hydra-driven experiment (carries name + config_choices +
        model_config; the full hydra_config dict is NOT surfaced -- it
        can be 10-100 KB, and callers wanting it should fetch the
        metadata asset directly).
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(_get_execution_detail_impl, ml, execution_rid)
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_execution_detail",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/deriva-ml/lineage/{rid}")
    async def ml_lineage(hostname: str, catalog_id: str, rid: str) -> str:
        """Provenance chain for any artifact (Dataset, Asset, Feature
        value, or Execution).

        Same shape as the ``deriva_ml_get_lineage`` tool -- the resource
        and tool share an internal helper so the two cannot drift.
        Walks data-flow parents only; orchestration links
        (``Execution_Execution``) are NOT followed (that's a separate
        question, served by ``deriva_ml_list_execution_parents`` /
        ``deriva_ml_list_execution_children``). See deriva-ml ADR-0001.

        Resource form has no ``depth`` / ``max_executions`` overrides;
        callers needing those should use the tool. The resource always
        walks unbounded with the default ``max_executions=500`` cap.
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(
                    _get_lineage_impl, ml, rid, depth=None, max_executions=500
                )
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_lineage",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/deriva-ml/features/{table_name}")
    async def ml_features_for_table(hostname: str, catalog_id: str, table_name: str) -> str:
        """Features defined on the given target table (up to 1000 rows).

        When ``truncated`` is True, use the ``deriva_ml_list_features`` tool with
        pagination.
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(
                    _list_features_impl,
                    ml,
                    table=table_name,
                    after_rid=None,
                    limit=_MAX_LIMIT,
                )
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_features_for_table",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/deriva-ml/asset/{asset_rid}")
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
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(_get_asset_detail_impl, ml, asset_rid)
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_asset_detail",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/deriva-ml/assets/{schema}")
    async def ml_assets_in_schema(hostname: str, catalog_id: str, schema: str) -> str:
        """Asset tables in one schema.

        Returns ``{schema, count, asset_tables: [{name, rid}, ...]}``.
        Schema is honored literally (no fallback search). An existing
        schema with no asset tables returns an empty list. A
        non-existent schema returns an error envelope.

        For row-level access, use
        ``ml/assets/{schema}/{asset_table}`` (snapshot, capped at
        1000 rows) or ``deriva_ml_list_assets`` (paginated tool).
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                # schema lookup on ml.model is an in-memory dict access
                # populated during get_ml's schema fetch; safe in async.
                if schema not in ml.model.schemas:
                    return _error_envelope(
                        KeyError(f"schema not found: {schema}"),
                        operation="resource_ml_assets_in_schema",
                        hostname=hostname,
                        catalog_id=catalog_id,
                        audit=False,
                    )
                payload = await asyncio.to_thread(_list_schema_asset_tables_impl, ml, schema)
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_assets_in_schema",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource(
        "deriva://catalog/{hostname}/{catalog_id}/deriva-ml/assets/{schema}/{asset_table}"
    )
    async def ml_asset_table_contents(
        hostname: str, catalog_id: str, schema: str, asset_table: str
    ) -> str:
        """Contents of one asset table (snapshot capped at 1000 rows).

        Returns ``{schema, asset_table, count, truncated,
        next_after_rid, assets: [AssetSummary, ...]}``. Per-row shape
        is identical to ``deriva_ml_list_assets``. When ``truncated``
        is True the asset table exceeds the cap; switch to the
        paginated tool.

        Returns an error envelope when the schema or table doesn't
        exist, or when the named table exists but is not an asset
        table.
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(
                    _get_asset_table_contents_impl, ml, schema, asset_table
                )
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_asset_table_contents",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource("deriva://catalog/{hostname}/{catalog_id}/deriva-ml/vocabularies/{schema}")
    async def ml_vocabularies_in_schema(hostname: str, catalog_id: str, schema: str) -> str:
        """Vocabulary tables in one schema.

        Returns ``{schema, count, vocabularies: [{name, rid,
        term_count}, ...]}``. ``term_count`` is best-effort: a
        per-vocab fetch failure surfaces as ``null`` rather than
        aborting the snapshot. An existing schema with no vocabularies
        returns an empty list. A non-existent schema returns an error
        envelope.

        For full term contents, use
        ``ml/vocabularies/{schema}/{vocab_name}``.
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                if schema not in ml.model.schemas:
                    return _error_envelope(
                        KeyError(f"schema not found: {schema}"),
                        operation="resource_ml_vocabularies_in_schema",
                        hostname=hostname,
                        catalog_id=catalog_id,
                        audit=False,
                    )
                payload = await asyncio.to_thread(_list_schema_vocabularies_impl, ml, schema)
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_vocabularies_in_schema",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )

    @ctx.resource(
        "deriva://catalog/{hostname}/{catalog_id}/deriva-ml/vocabularies/{schema}/{vocab_name}"
    )
    async def ml_vocabulary_terms(
        hostname: str, catalog_id: str, schema: str, vocab_name: str
    ) -> str:
        """All terms in one vocabulary table.

        Returns ``{schema, vocabulary, count, terms: [{name, rid,
        description, synonyms, id, uri}, ...]}``. Surfaces all six
        fields from deriva-ml's ``VocabularyTerm`` (including CURIE
        and URI). Bounded by ``_MAX_LIMIT``; vocabularies almost
        always fit under the cap.

        Returns an error envelope when the schema or vocabulary
        doesn't exist, or when the named table exists but is not a
        vocabulary.
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                payload = await asyncio.to_thread(
                    _get_vocabulary_terms_impl, ml, schema, vocab_name
                )
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            return _error_envelope(
                exc,
                operation="resource_ml_vocabulary_terms",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )
