"""RID description tool: resolve a bare RID and route to the right surface.

``deriva_ml_describe_rid`` is "step 0" of the ordered query strategy in
``deriva_ml_getting_started``: given a RID of unknown type, resolve it
to its (schema, table) via ERMrest's catalog-wide RID resolution
(``DerivaML.resolve_rid``), classify it into the DerivaML abstractions,
and name the typed tool / resource to call next. This replaces the
try-``get_dataset``-then-``get_execution`` guessing an LLM otherwise
does with a bare RID.

RID resolution itself is a GENERIC Deriva primitive (an ERMrest
feature surfaced by deriva-py's ``ErmrestCatalog.resolve_rid``), so the
plain resolve belongs in deriva-mcp-core per the boundary rules --
tracked upstream as a core issue. This tool is the ML-domain value-add
on top: the classification into the five abstractions and the
next-tool routing. When core ships its generic ``resolve_rid``, this
tool keeps its classification layer and can delegate the resolve.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from deriva_mcp_core import deriva_call

from deriva_ml_mcp_plugin._helpers import _error_envelope
from deriva_ml_mcp_plugin._response_models import DescribeRidResponse
from deriva_ml_mcp_plugin.ml_context import get_ml

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def _classify(ml, table) -> tuple[str, str, str | None]:
    """Classify a resolved table into a DerivaML kind + routing suggestion.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance (consulted for
            ``ml_schema`` and the ``model.is_*`` classifiers).
        table: The ermrest ``Table`` object from ``resolve_rid``.

    Returns:
        ``(kind, suggestion, resource_uri_suffix)`` where the suffix is
        the ``deriva-ml/...`` resource path tail for kinds that have a
        per-RID resource (``None`` otherwise; the caller prepends the
        catalog prefix and appends the RID).
    """
    tname = table.name
    sname = table.schema.name
    if sname == ml.ml_schema and tname == "Dataset":
        return (
            "dataset",
            "Call deriva_ml_get_dataset(rid) for summary + version history; "
            "deriva_ml_list_dataset_members(rid) for contents; "
            "deriva_ml_get_lineage(rid) for provenance.",
            "dataset",
        )
    if sname == ml.ml_schema and tname == "Workflow":
        return (
            "workflow",
            "Call deriva_ml_get_workflow(rid) for detail; "
            "deriva_ml_find_workflow_executions(rid) for its runs.",
            "workflow",
        )
    if sname == ml.ml_schema and tname == "Execution":
        return (
            "execution",
            "Call deriva_ml_get_execution(rid) for status/detail, then chain "
            "deriva_ml_get_lineage(rid) for consumed/produced artifacts.",
            "execution",
        )
    if ml.model.is_vocabulary(table):
        return (
            "vocabulary_term",
            f"This RID is a term in the {sname}.{tname} vocabulary. Use core's "
            "lookup_term / the vocabularies resource for term detail.",
            None,
        )
    if ml.model.is_asset(table):
        return (
            "asset",
            "Call deriva_ml_lookup_asset(rid) for metadata + download URL; "
            "deriva_ml_get_lineage(rid) for the producing execution.",
            "asset",
        )
    if ml.model.is_association(table):
        return (
            "association",
            f"This RID is a link row in {sname}.{tname}. If it is a feature-values "
            "table, query it via deriva_ml_list_feature_values (selectors pick the "
            "annotation layer); otherwise deriva_ml_get_lineage on the linked "
            "entity RIDs.",
            None,
        )
    return (
        "entity",
        f"This RID is a domain row in {sname}.{tname}. deriva_ml_get_lineage(rid) "
        "walks its provenance if it is dataset/execution-linked; generic catalog "
        "tools (get_entities / query_attribute) read the row itself.",
        None,
    )


def register(ctx: PluginContext) -> None:
    """Register the RID description tool with the plugin context.

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
    async def deriva_ml_describe_rid(
        hostname: str,
        catalog_id: str,
        rid: str,
    ) -> str:
        """Describe a RID: which table it lives in, what kind of thing it is, what to call next.

        STEP 0 of the ordered query strategy (see
        ``deriva_ml_getting_started``): when the user hands you a bare
        RID and you don't know what it identifies, call this FIRST --
        do not guess by trying deriva_ml_get_dataset /
        deriva_ml_get_execution in turn, and do not go schema-spelunking.
        One call resolves the RID catalog-wide (ERMrest RID resolution),
        classifies it (dataset / workflow / execution / vocabulary_term /
        asset / association / entity), and names the right next tool.

        Args:
            rid: The RID to describe (e.g. ``"1-ABCD"``). Any table,
                any schema.

        Returns:
            JSON string: ``{"rid", "schema", "table", "kind",
            "suggestion", "resource_uri"}``. ``resource_uri`` is the
            matching ``deriva://catalog/.../deriva-ml/<kind>/{rid}``
            resource for dataset / workflow / execution / asset RIDs,
            else ``null``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}`` -- propagated
                from ``DerivaML.resolve_rid`` when the RID does not
                exist in the catalog.

        Example:
            ``{"rid": "1-ABCD", "schema": "deriva-ml",
            "table": "Execution", "kind": "execution",
            "suggestion": "Call deriva_ml_get_execution(rid)...",
            "resource_uri":
            "deriva://catalog/h/1/deriva-ml/execution/1-ABCD"}``.
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
                # resolve_rid + the model classifiers are synchronous
                # deriva-py/deriva-ml catalog I/O; run the whole
                # resolve-and-classify in a worker thread.

                def _resolve_and_classify():
                    result = ml.resolve_rid(rid)
                    return result.table, _classify(ml, result.table)

                table, (kind, suggestion, uri_kind) = await asyncio.to_thread(_resolve_and_classify)
            resource_uri = (
                f"deriva://catalog/{hostname}/{catalog_id}/deriva-ml/{uri_kind}/{rid}"
                if uri_kind
                else None
            )
            return DescribeRidResponse(
                rid=rid,
                schema_=table.schema.name,
                table=table.name,
                kind=kind,
                suggestion=suggestion,
                resource_uri=resource_uri,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="describe_rid",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,  # pure read
            )
