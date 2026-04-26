"""Smoke tests for the deriva-ml-mcp plugin entry point."""

from __future__ import annotations

from importlib import metadata

from deriva_ml_mcp.plugin import register

# Per-domain frozensets of registered tools. Each new domain phase
# (workflow, execution) adds one of these and OR-unions it
# into ``_ALL_REGISTERED_TOOLS`` below. The exact-equality check in
# ``test_all_registered_tools_exact`` then catches both:
#
#   - tools added without an entry in their per-domain frozenset, AND
#   - tools registered without an updated ``coverage.md`` row.
#
# Update protocol when adding a new domain:
#   1. Add ``_WORKFLOW_TOOLS = frozenset({...})`` (etc.) here.
#   2. Add it to the ``_ALL_REGISTERED_TOOLS`` union below.
#   3. ``test_all_registered_tools_exact`` keeps its shape; no rename.

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

_FEATURE_TOOLS = frozenset(
    {
        # Read-only tools
        "list_features",
        "get_feature",
        "list_feature_values",
        # Mutation tools
        "create_feature",
        "delete_feature",
        "add_feature_values",
    }
)

_WORKFLOW_TOOLS = frozenset(
    {
        # Read-only tools
        "list_workflows",
        "get_workflow",
        "find_workflow_by_url",
        # Mutation tools
        "create_workflow",
        "update_workflow",
    }
)

# Union of all per-domain tool sets. Phase 4 registers dataset + feature
# + workflow; Phase 5+ adds ``| _EXECUTION_TOOLS``, etc.
_ALL_REGISTERED_TOOLS = _DATASET_TOOLS | _FEATURE_TOOLS | _WORKFLOW_TOOLS


def test_register_runs_without_error(ctx):
    """``register(ctx)`` must succeed end-to-end."""
    register(ctx)


def test_entry_point_resolves_to_register():
    """The 'deriva-ml' entry point must resolve to our register function."""
    eps = metadata.entry_points(group="deriva_mcp.plugins")
    matching = [ep for ep in eps if ep.name == "deriva-ml"]
    assert matching, "entry point 'deriva-ml' not declared"
    assert matching[0].load() is register


def test_all_registered_tools_exact(ctx, capturing_mcp):
    """The set of registered tools must equal ``_ALL_REGISTERED_TOOLS`` exactly.

    Exact-equality (rather than ``issubset`` + ``not extra``) catches
    BOTH directions in one assertion:

    - missing: a tool listed in a per-domain frozenset but not yet
      registered (regression).
    - unexpected: a tool registered without being added to its
      per-domain frozenset (forces a ``coverage.md`` and frozenset
      update on every new tool).

    When Phase 3+ adds a new domain, define ``_FEATURE_TOOLS`` (etc.)
    and OR-it into ``_ALL_REGISTERED_TOOLS`` above; this test keeps
    its shape.
    """
    register(ctx)
    actual = frozenset(capturing_mcp.tools.keys())
    assert actual == _ALL_REGISTERED_TOOLS, (
        f"missing: {sorted(_ALL_REGISTERED_TOOLS - actual)}; "
        f"unexpected (update per-domain frozenset and coverage.md): "
        f"{sorted(actual - _ALL_REGISTERED_TOOLS)}"
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
