"""Smoke tests for the deriva-ml-mcp plugin entry point."""

from __future__ import annotations

from importlib import metadata

from deriva_ml_mcp.plugin import register

# Dataset tools registered by ``tools.dataset.register``. Grouped by
# semantic role, not implementation history (the "Batch 1/2/3"
# rollout was internal scaffolding).
_DATASET_TOOLS = frozenset(
    {
        # Read-only tools
        "list_datasets",
        "get_dataset",
        "list_dataset_members",
        "list_dataset_relations",
        "list_dataset_element_types",
        "bag_info",
        "get_dataset_spec",
        # Mutation tools
        "create_dataset",
        "delete_dataset",
        "add_dataset_members",
        "delete_dataset_members",
        "update_dataset_types",
        "add_dataset_element_type",
        "increment_dataset_version",
        # Complex / local-FS tools
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
    """All 17 Phase 2 dataset tools must be registered by ``register(ctx)``,
    AND no unexpected tools sneak in. The exact-equality check forces a
    test/coverage.md update whenever a tool is added to ``dataset.py``;
    when Phase 3 lands and feature/workflow/execution tools start
    registering through the same entry point, this becomes
    ``_DATASET_TOOLS.issubset(...)`` and a parallel check ensures the
    UNEXPECTED set is empty per-phase rather than overall."""
    register(ctx)
    actual = set(capturing_mcp.tools.keys())
    missing = _DATASET_TOOLS - actual
    unexpected = actual - _DATASET_TOOLS
    assert not missing, f"missing tools: {sorted(missing)}"
    assert not unexpected, (
        f"unregistered new tools (update _DATASET_TOOLS and coverage.md): {sorted(unexpected)}"
    )


def test_all_registered_tools_have_explicit_mutates(ctx, capturing_mcp):
    """Every registered tool must go through ``ctx.tool(mutates=...)``.

    Note on what this catches vs. what core catches:

    - ``deriva-mcp-core``'s ``PluginContext.tool`` already raises
      ``TypeError`` at registration time if ``mutates=`` is omitted
      from a ``@ctx.tool(...)`` call. So the "I forgot mutates="
      bug is caught by ``test_register_runs_without_error`` (the
      whole register call would fail).
    - This test catches a different bug: a tool author who BYPASSES
      ``ctx.tool`` entirely and uses ``@some_mcp.tool()`` directly,
      which would skip core's mutates validation. Such a tool
      registers successfully but never appears in our wrapped-call
      record below, so the ``missing`` check flags it.

    Implementation: we wrap ``ctx.tool`` for the duration of
    ``register(ctx)`` and record the ``mutates`` value alongside each
    tool's name. ``PluginContext.tool`` strips ``mutates`` before
    forwarding to FastMCP, so the capturing fixture's ``tool_kwargs``
    dict never sees the value — hence the wrapper.
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
