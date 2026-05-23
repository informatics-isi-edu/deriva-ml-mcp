"""Plugin entry point for deriva-ml-mcp.

The ``register`` function is exposed via the ``deriva_mcp.plugins``
entry-point group in ``pyproject.toml``. ``deriva-mcp-core`` calls it
once per server startup with a :class:`PluginContext` instance.

Example:
    Manual invocation (for testing only - normally invoked by core's
    plugin loader)::

        from deriva_mcp_core.plugin.api import PluginContext
        from deriva_ml_mcp.plugin import register

        ctx = PluginContext(some_mcp_server)
        register(ctx)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deriva_ml_mcp import prompts as _ml_prompts
from deriva_ml_mcp.resources import ml as _ml_resources
from deriva_ml_mcp.resources import rag as _ml_rag
from deriva_ml_mcp.tools import asset as _asset
from deriva_ml_mcp.tools import dataset as _dataset
from deriva_ml_mcp.tools import execution as _execution
from deriva_ml_mcp.tools import feature as _feature
from deriva_ml_mcp.tools import maintenance as _maintenance
from deriva_ml_mcp.tools import vocabulary as _vocabulary
from deriva_ml_mcp.tools import workflow as _workflow

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def register(ctx: PluginContext) -> None:
    """Register all deriva-ml-mcp tools and resources with the given context.

    Phase 5 ships dataset, feature, workflow, and execution domain tools.
    Phase 6 added MCP resources and per-user RAG indexing (one GitHub
    docs source plus three on_catalog_connect hooks for Dataset,
    Workflow, and Execution rows). v1.0 polish added 3 MCP prompts
    (``deriva_ml_getting_started``, ``deriva_ml_execution_lifecycle``,
    ``deriva_ml_workflow_dedup``) so an LLM connecting cold has an
    anchor for the ML domain.

    v1.1 added vocabulary RAG indexing: a fourth on_catalog_connect
    hook bulk-indexes all discovered vocabulary tables under a
    catalog-public ``vocab:`` source, and a one-tool vocabulary domain
    (``deriva_ml_reindex_vocabularies``) lets callers force a refresh
    after using core's ``add_term`` / ``delete_term`` (which don't
    fire any framework lifecycle hook -- tracked upstream as
    deriva-mcp-core#3).

    v1.2 added the asset domain (catalog-state operations -- file
    I/O lives in the deriva-skills ``work-with-assets`` skill) plus
    curation symmetry: every catalog-mutable typed entity (Dataset,
    Workflow, Asset) has one ``deriva_ml_update_<entity>(rid,
    *fields)`` tool that mutates only the kwargs provided.
    v0.5.0 removed the execution-mutation surface entirely per the
    stateless rule -- executions originate in the caller's local
    Python (see docs/audit-2026-05-23.md). The pre-v1.2
    ``deriva_ml_update_dataset_types`` tool was renamed (and
    widened) to ``deriva_ml_update_dataset``.

    Args:
        ctx: PluginContext supplied by deriva-mcp-core at startup.

    Returns:
        None.

    Example:
        >>> from deriva_mcp_core.plugin.api import PluginContext
        >>> # ctx provided by the framework; not constructed by user code
        >>> register(ctx)  # doctest: +SKIP
    """
    _dataset.register(ctx)
    _feature.register(ctx)
    _workflow.register(ctx)
    _execution.register(ctx)
    _asset.register(ctx)
    _vocabulary.register(ctx)
    _maintenance.register(ctx)
    _ml_resources.register(ctx)
    _ml_rag.register_rag_sources(ctx)
    _ml_prompts.register(ctx)
