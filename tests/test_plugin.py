"""Smoke tests for the deriva-ml-mcp plugin entry point."""

from __future__ import annotations

from importlib import metadata

from deriva_ml_mcp.plugin import register

# Dataset tools registered by ``tools.dataset.register`` (Batches 1-3).
_DATASET_TOOLS = frozenset(
    {
        # Batch 1: read tools
        "list_datasets",
        "get_dataset",
        "list_dataset_members",
        "list_dataset_relations",
        "list_dataset_element_types",
        "bag_info",
        "get_dataset_spec",
        # Batch 2: mutation tools
        "create_dataset",
        "delete_dataset",
        "add_dataset_members",
        "delete_dataset_members",
        "update_dataset_types",
        "add_dataset_element_type",
        "increment_dataset_version",
        # Batch 3: complex tools
        "cache_dataset",
        "denormalize_dataset",
        "split_dataset",
    }
)


def test_register_runs_without_error(ctx):
    """``register(ctx)`` must succeed end-to-end."""
    register(ctx)


def test_entry_point_resolves_to_register():
    """The 'deriva-ml' entry point must resolve to our register function."""
    eps = metadata.entry_points(group="deriva_mcp.plugins")
    matching = [ep for ep in eps if ep.name == "deriva-ml"]
    assert matching, "entry point 'deriva-ml' not declared"
    assert matching[0].load() is register


def test_dataset_tools_registered(ctx, capturing_mcp):
    """All 17 Phase 2 dataset tools must be registered by ``register(ctx)``."""
    register(ctx)
    assert _DATASET_TOOLS.issubset(capturing_mcp.tools.keys()), (
        f"missing tools: {_DATASET_TOOLS - capturing_mcp.tools.keys()}"
    )


def test_all_registered_tools_have_explicit_mutates(ctx, capturing_mcp):
    """Every registered tool must declare ``mutates`` explicitly.

    Catches the common bug where a tool author forgets ``mutates=`` on the
    ``@ctx.tool(...)`` decorator. ``deriva-mcp-core``'s ``PluginContext.tool``
    already raises ``TypeError`` at registration time when ``mutates`` is
    omitted, but it strips the kwarg before forwarding to FastMCP, so the
    capturing fixture never sees it. We wrap ``ctx.tool`` here to record the
    ``mutates`` value alongside each tool's name.
    """
    seen_mutates: dict[str, bool] = {}
    original_tool = ctx.tool

    def capturing_tool(*args, mutates, **kwargs):
        decorator = original_tool(*args, mutates=mutates, **kwargs)

        def wrapper(fn):
            seen_mutates[fn.__name__] = mutates
            return decorator(fn)

        return wrapper

    ctx.tool = capturing_tool  # type: ignore[method-assign]
    try:
        register(ctx)
    finally:
        ctx.tool = original_tool  # type: ignore[method-assign]

    missing = [name for name in capturing_mcp.tools if name not in seen_mutates]
    assert not missing, f"tools missing explicit mutates=: {missing}"
    assert all(isinstance(v, bool) for v in seen_mutates.values()), (
        f"mutates= must be bool; got: {seen_mutates}"
    )
