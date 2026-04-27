"""RAG cache management tools for deriva-ml-mcp.

Both tools in this module are **manual cache-refresh** verbs --
``mutates=False``, no audit emission, no catalog-state change. They
exist because RAG indexes can fall out of sync with catalog state in
two scenarios the framework's lifecycle hooks don't cover:

- ``deriva_ml_reindex_vocabularies`` (v1.1) -- after adding/removing
  terms via core's ``add_term`` / ``delete_term`` (which don't fire
  any framework lifecycle hook; tracked upstream as
  ``deriva-mcp-core#3``).

- ``deriva_ml_resync_indexes`` (v1.4) -- to refresh the calling
  user's per-user-trio (Dataset / Workflow / Execution) sources
  after **another user** (or a non-MCP client like Chaise) has
  mutated catalog state the calling user can see. v1.3's surgical
  re-index covers OWN mutations; this tool is the bridge for
  cross-user freshness. See #9 for the design discussion (Option 4
  in the original issue body).

(The module name ``vocabulary`` predates the v1.4 addition and is
now misleading; rename to ``maintenance`` is a v1.x cleanup.)
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

    @ctx.tool(mutates=False)
    async def deriva_ml_resync_indexes(
        hostname: str,
        catalog_id: str,
        target: str | None = None,
    ) -> str:
        """Refresh the calling user's per-user RAG sources for this catalog.

        v1.3's surgical per-RID re-indexing handles freshness for the
        calling user's OWN mutations. This tool is the bridge for
        **cross-user freshness**: when another user (or a non-MCP
        client like Chaise) has mutated catalog state the calling user
        can see, those changes don't appear in the calling user's
        per-user RAG sources until the user explicitly resyncs.

        Two modes:

        - ``target=None`` (default): refresh ALL of the calling user's
          per-user sources for this catalog (datasets + workflows +
          executions). The bigger hammer; LLM-friendly default for
          "I think my view is stale, refresh everything."

        - ``target="<table>:<rid>"``: refresh just one source.
          ``<table>`` is one of ``dataset`` / ``workflow`` /
          ``execution``; ``<rid>`` is the row's RID. For when the
          caller knows exactly which row is stale.

        ``mutates=False`` because the call does not change catalog
        state -- it only refreshes the RAG vector store. No audit
        emission on success or failure (cache refresh, not catalog
        mutation).

        Best-effort throughout: per-RID failures during a resync are
        logged and swallowed; the response reports the count of
        sources successfully refreshed per table. A partial success
        (e.g. dataset 5/6 refreshed because one raised) is still a
        success-shape response.

        WHEN TO USE THIS TOOL. ``deriva_ml_getting_started``'s
        cross-user freshness section explains the trigger conditions.
        Common cases: collaborative curation (another user just
        released a dataset visible to you); cross-client edits (a
        colleague just renamed an execution via Chaise); or a generic
        "I'm not sure my RAG view is current" hedge before a
        consequential ``rag_search``.

        Args:
            hostname: The Deriva server hostname.
            catalog_id: The catalog ID as a string.
            target: Optional ``"<table>:<rid>"`` selector. ``None``
                (default) refreshes all of the calling user's
                per-user sources.

        Returns:
            JSON string. Success:
            ``{"resynced": {"dataset": N, "workflow": M, "execution": K}}``
            -- counts of sources successfully refreshed per table.
            For ``target="<table>:<rid>"`` mode, only the targeted
            table's count is non-zero. Failure: ``{"error": str}``.

        Raises:
            ValueError: If ``target`` is malformed (not
                ``"<table>:<rid>"`` shape, or ``<table>`` not in
                ``dataset`` / ``workflow`` / ``execution``). Wrapped
                as ``{"error": ...}``.

        Example:
            Refresh everything (most common pattern)::

                deriva_ml_resync_indexes(hostname="myhost", catalog_id="1")

            Surgical refresh of one dataset::

                deriva_ml_resync_indexes(
                    hostname="myhost",
                    catalog_id="1",
                    target="dataset:1-AAAA",
                )
        """
        try:
            with deriva_call():
                # Lazy import per the established pattern -- keeps
                # module import time cheap and lets tests patch
                # rag._resync_user_sources at the use-site.
                from deriva_ml_mcp.resources.rag import _resync_user_sources

                counts = await _resync_user_sources(hostname, catalog_id, target=target)
            return json.dumps({"resynced": counts})
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="resync_indexes",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,  # not a catalog mutation
            )
