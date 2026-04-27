"""Vocabulary management tools for deriva-ml-mcp.

Currently exposes one tool:

- ``deriva_ml_reindex_vocabularies`` -- force re-index of vocabulary
  tables in the RAG vector store. Use after adding/removing terms via
  core's ``add_term`` / ``delete_term`` tools (which don't fire any
  framework lifecycle hook -- tracked upstream as
  ``deriva-mcp-core#3``).

Why this is a tool, not a resource: it has a side effect (vector
store mutation), even though catalog state is unchanged. ``mutates=False``
because it doesn't change *catalog* state -- the audit log doesn't
care about cache refreshes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from deriva_mcp_core import deriva_call

from deriva_ml_mcp._helpers import _error_envelope

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def register(ctx: PluginContext) -> None:
    """Register vocabulary management tools with the plugin context.

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
    async def deriva_ml_reindex_vocabularies(
        hostname: str,
        catalog_id: str,
        vocab: str | None = None,
    ) -> str:
        """Force re-index of vocabulary tables in the RAG vector store.

        Use after adding/removing terms via core's ``add_term`` /
        ``delete_term`` tools (which don't fire any framework lifecycle
        hook -- tracked upstream as deriva-mcp-core#3). The
        ``deriva_ml_getting_started`` prompt's discovery section
        explains when to call this.

        ``mutates=False`` because the call does not change catalog
        state -- the audit log treats cache refreshes as reads, so no
        audit row is emitted on success or failure.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            vocab: Optional vocab qname (``"schema.table"``). If
                ``None``, re-indexes all vocabularies in the catalog.

        Returns:
            JSON string. Success: ``{"reindexed": {qname: term_count, ...}}``.
            Failure: ``{"error": str}``.

        Raises:
            RuntimeError: Wrapped as ``{"error": ...}``, propagated
                from ``ml.find_vocabularies`` /
                ``ml.list_vocabulary_terms`` if catalog access itself
                fails. Per-vocab failures during a successful pass are
                swallowed and logged; they do not appear in the
                response.

        Example:
            Re-index all vocabs after adding a Tissue_Type term::

                deriva_ml_reindex_vocabularies(hostname="myhost", catalog_id="1")

            Re-index only the Dataset_Type vocab::

                deriva_ml_reindex_vocabularies(
                    hostname="myhost",
                    catalog_id="1",
                    vocab="deriva-ml.Dataset_Type",
                )
        """
        try:
            with deriva_call():
                # Imported lazily so module import time stays cheap and
                # so test patches of ``rag._index_vocabularies`` reach
                # this call site cleanly.
                from deriva_ml_mcp.resources.rag import _index_vocabularies

                indexed = await _index_vocabularies(hostname, catalog_id, only_vocab=vocab)
            return json.dumps({"reindexed": indexed})
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="reindex_vocabularies",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,  # not a catalog mutation
            )
