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

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def register(ctx: PluginContext) -> None:
    """Register all deriva-ml-mcp tools and resources with the given context.

    Per-domain registrars are added in subsequent phases of the migration
    (dataset, feature, workflow, execution, resources). During phase 0
    this function is a no-op so the plugin can be loaded and smoke-tested
    in isolation.

    Args:
        ctx: PluginContext supplied by deriva-mcp-core at startup.

    Returns:
        None.

    Example:
        >>> from deriva_mcp_core.plugin.api import PluginContext
        >>> # ctx provided by the framework; not constructed by user code
        >>> register(ctx)  # doctest: +SKIP
    """
    return None
