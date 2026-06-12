"""Vocabulary domain tools for deriva-ml-mcp-plugin.

Mutation tools: ``deriva_ml_create_vocabulary``.

This module is the deriva-ml-aware counterpart to the generic
``create_vocabulary`` in deriva-mcp-core. The generic tool produces
a working vocabulary table on any catalog, but doesn't know about
the deriva-ml project name (used as the curie prefix) or the
domain schema (used as the default placement). The tool here
delegates to ``DerivaML.create_vocabulary``, which picks both up
automatically and also triggers a navbar update so the new vocab
shows up in Chaise without a separate ``apply_catalog_annotations``
call.

Prefer this tool over deriva-mcp-core's ``create_vocabulary`` on
any catalog that has the deriva-ml schema installed. Both produce
tables of the same physical shape (the columns and pseudo-types
are identical -- ``VocabularyTableDef`` underlies both); the
difference is in defaults and side effects.

Use ``add_term`` from deriva-mcp-core to populate terms after
creation -- there is no deriva-ml-specific add_term variant.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from deriva_mcp_core import deriva_call
from deriva_mcp_core.plugin.api import fire_schema_change
from deriva_mcp_core.telemetry import audit_event

from deriva_ml_mcp_plugin._helpers import _error_envelope
from deriva_ml_mcp_plugin._response_models import CreateVocabularyResponse
from deriva_ml_mcp_plugin.ml_context import get_ml

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def register(ctx: PluginContext) -> None:
    """Register the vocabulary domain tools with the plugin context.

    Currently registers a single tool, ``deriva_ml_create_vocabulary``.
    See the module docstring for the rationale and the relationship
    to deriva-mcp-core's generic ``create_vocabulary``.

    Args:
        ctx: PluginContext supplied by deriva-mcp-core at startup.

    Returns:
        None.

    Example:
        >>> from deriva_mcp_core.plugin.api import PluginContext
        >>> # ctx provided by the framework; not constructed by user code
        >>> register(ctx)  # doctest: +SKIP
    """

    @ctx.tool(mutates=True)
    async def deriva_ml_create_vocabulary(
        hostname: str,
        catalog_id: str,
        vocab_name: str,
        comment: str = "",
        schema: str | None = None,
        update_navbar: bool = True,
    ) -> str:
        """Create a controlled vocabulary table via deriva-ml.

        Delegates to ``DerivaML.create_vocabulary``, which builds the
        canonical DERIVA vocabulary shape (Name, ID, URI, Description,
        Synonyms with ``ermrest_curie`` / ``ermrest_uri`` pseudo-types
        and the standard key set). The curie template is scoped to
        the deriva-ml project name, so terms carry stable identifiers
        of the form ``{project}:{RID}``. By default the navigation
        bar is also refreshed so the new vocabulary becomes visible
        in Chaise immediately.

        Use this tool in preference to the generic
        ``create_vocabulary`` from deriva-mcp-core on any catalog
        that has the deriva-ml schema installed. Both produce
        physically-identical tables, but this one applies the
        deriva-ml defaults and side effects.

        Populate terms with deriva-mcp-core's ``add_term`` -- the
        same tool works for both ml-created and generically-created
        vocabularies because the column shape matches.

        Args:
            vocab_name: Name for the new vocabulary table. Must be
                a valid SQL identifier and unique within the target
                schema.
            comment: Optional human-readable description.
            schema: Schema name to create the table in. ``None``
                (default) means the deriva-ml domain schema.
            update_navbar: If ``True`` (default), refresh the
                catalog's navigation bar so the new vocab shows up
                in Chaise immediately. Set to ``False`` during batch
                table creation, then apply catalog annotations once at
                the end via the Python API
                (``ml.apply_catalog_annotations()`` -- deliberately not
                exposed as an MCP tool).

        Returns:
            JSON string ``{"status": "created", "schema",
            "vocab_name", "columns": [<col_name>, ...]}``. The
            column list reflects what ``VocabularyTableDef``
            produced (typically: Name, ID, URI, Description,
            Synonyms, plus the system columns).

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated
                from ``deriva_ml.DerivaML.create_vocabulary`` (e.g.
                duplicate vocab_name, invalid SQL identifier,
                catalog access failure).

        Example:
            ``{"status": "created", "schema": "my_project",
            "vocab_name": "Tissue_Type", "columns": ["RID", "RCT",
            "RMT", "RCB", "RMB", "Name", "ID", "URI", "Description",
            "Synonyms"]}``.
        """
        try:
            with deriva_call():
                # Run the synchronous deriva-ml call in a thread
                # pool so the event loop stays responsive. See
                # deriva-mcp-core plugin-authoring-guide.md
                # §"Synchronous work in threads".
                ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
                vocab_table = await asyncio.to_thread(
                    ml.create_vocabulary,
                    vocab_name,
                    comment=comment,
                    schema=schema,
                    update_navbar=update_navbar,
                )
                resolved_schema = vocab_table.schema.name
                columns = [c.name for c in vocab_table.columns]
            # Notify framework so plugins watching for schema mutations
            # (RAG index, denormalizer, etc.) can refresh. Mirrors what
            # deriva-mcp-core's generic create_vocabulary does.
            fire_schema_change(hostname, catalog_id)
            audit_event(
                "deriva_ml_create_vocabulary",
                hostname=hostname,
                catalog_id=catalog_id,
                schema=resolved_schema,
                vocab_name=vocab_name,
                update_navbar=update_navbar,
            )
            return CreateVocabularyResponse(
                status="created",
                schema=resolved_schema,
                vocab_name=vocab_name,
                columns=columns,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="create_vocabulary",
                hostname=hostname,
                catalog_id=catalog_id,
                vocab_name=vocab_name,
            )
