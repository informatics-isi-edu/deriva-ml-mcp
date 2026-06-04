"""Unit tests for the deriva-ml MCP prompts.

Prompts are pure data -- no DERIVA I/O, no catalog access. The tests
exercise registration shape (count, names) and content invariants
(non-empty, ASCII-only). Per-tool behavior tests live with their
respective modules.

History: shipped four prompts at v1.0. v3.x removed two:

  - ``deriva_ml_workflow_dedup`` -- per-tool LLM-trap warning;
    content moved to the ``deriva_ml_create_workflow`` and
    ``deriva_ml_find_workflow_by_url`` tool docstrings.
  - ``deriva_ml_execution_lifecycle`` -- per-tool warnings moved to
    the four lifecycle tool docstrings (``deriva_ml_start_execution``,
    ``deriva_ml_commit_execution``, ``deriva_ml_abort_execution``,
    ``deriva_ml_add_feature_values``); cross-cutting state-machine
    depth covered by RAG-indexed ``user-guide/executions.md``.

The remaining two (``deriva_ml_concepts``, ``deriva_ml_getting_started``)
serve cold-start orientation for non-Claude-Code clients. Long-term
plan: migrate to the FastMCP ``instructions=`` field once a
deriva-mcp-core API for plugin-contributed instructions exists.
"""

from __future__ import annotations

from deriva_ml_mcp_plugin import prompts

_EXPECTED_PROMPT_NAMES = frozenset(
    {
        "deriva_ml_concepts",
        "deriva_ml_getting_started",
        "deriva_ml_primer",
    }
)


def test_register_runs_without_error(ctx):
    """``prompts.register(ctx)`` must succeed end-to-end."""
    prompts.register(ctx)


def test_three_prompts_registered(ctx, capturing_mcp):
    """Exactly three prompts land in the capturing MCP after register."""
    prompts.register(ctx)
    assert len(capturing_mcp.prompts) == 3


def test_primer_prompt_returns_primer_body(ctx, capturing_mcp):
    """The deriva_ml_primer prompt returns the rendered primer."""
    prompts.register(ctx)
    fn = capturing_mcp.prompts["deriva_ml_primer"]
    assert fn() == prompts._render_primer()


def test_primer_tool_registered_read_only(ctx, capturing_mcp):
    """deriva_ml_primer is registered as a read-only tool.

    ``PluginContext.tool`` consumes ``mutates=`` as a named parameter and
    strips it before forwarding to the underlying MCP, so it never lands in
    the capturing fixture's ``tool_kwargs``. To assert the value we wrap
    ``ctx.tool`` for the duration of ``register(ctx)`` and record the
    ``mutates`` flag per tool -- the same pattern as
    ``test_plugin.py::test_all_registered_tools_have_explicit_mutates``.
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
        prompts.register(ctx)
    finally:
        ctx.tool = original_tool  # type: ignore[method-assign]

    assert "deriva_ml_primer" in capturing_mcp.tools
    assert seen_mutates["deriva_ml_primer"] is False


def test_primer_tool_returns_primer_body(ctx, capturing_mcp):
    """The deriva_ml_primer tool returns the rendered primer regardless of args."""
    prompts.register(ctx)
    fn = capturing_mcp.tools["deriva_ml_primer"]
    assert fn() == prompts._render_primer()
    assert fn(hostname="h", catalog_id="1") == prompts._render_primer()


def test_prompt_names_exact(ctx, capturing_mcp):
    """The two prompts must be registered under the documented names.

    Exact-equality (rather than ``issubset``) catches both directions:
    a missing prompt and an unexpected one. The plugin-level test in
    ``test_plugin.py`` uses ``>=`` instead because ``register(ctx)``
    there also registers tools and resources whose names should not
    leak into the prompt name space (a sanity check the bare-prompts
    test cannot make).
    """
    prompts.register(ctx)
    assert frozenset(capturing_mcp.prompts.keys()) == _EXPECTED_PROMPT_NAMES


def test_each_prompt_returns_nonempty_string(ctx, capturing_mcp):
    """Every prompt callable must return a substantial guide string.

    The 100-char floor is a sanity bound -- the shortest remaining
    guide is well over 1500 chars. A regression that accidentally
    empties a guide constant would drop the result to near-zero
    length and fail this check.
    """
    prompts.register(ctx)
    for name, fn in capturing_mcp.prompts.items():
        result = fn()
        assert isinstance(result, str), f"{name} did not return a string"
        assert len(result) > 100, f"{name} returned a near-empty body ({len(result)} chars)"


def test_guide_manifest_shape():
    """_GUIDE_MANIFEST is a list of (name, source, summary) triples."""
    manifest = prompts._GUIDE_MANIFEST
    assert isinstance(manifest, list)
    assert len(manifest) >= 4
    for entry in manifest:
        name, source, summary = entry  # exactly three fields
        assert isinstance(name, str) and name
        assert source in {"deriva-ml", "core"}
        assert isinstance(summary, str) and summary


def test_guide_manifest_names_core_tier1_guides():
    """The four deriva-mcp-core tier-1 guides are named with source 'core'."""
    core_names = {n for (n, src, _) in prompts._GUIDE_MANIFEST if src == "core"}
    assert core_names == {
        "query_guide",
        "entity_guide",
        "annotation_guide",
        "catalog_guide",
    }


def test_prompts_are_ascii(ctx, capturing_mcp):
    """Every prompt body must be plain ASCII (workspace convention).

    Mirrors the deriva-mcp-core convention: no en-dashes, smart quotes,
    or other Unicode in code or built-in prompt strings. Unicode is
    fine in .md files and test data, but built-in prompt text ships
    inside the plugin package and should stay portable.
    """
    prompts.register(ctx)
    for name, fn in capturing_mcp.prompts.items():
        result = fn()
        try:
            result.encode("ascii")
        except UnicodeEncodeError as exc:  # pragma: no cover -- regression assertion path
            raise AssertionError(f"{name} contains non-ASCII characters: {exc}") from exc


def test_render_primer_contains_both_guide_bodies():
    """The primer inlines the concepts and getting-started guide bodies."""
    body = prompts._render_primer()
    # A distinctive phrase from each guide must be present verbatim.
    assert "DERIVA-ML GETTING STARTED" in body
    assert "five core abstractions" in body or "Dataset" in body


def test_render_primer_lists_all_guide_names():
    """Every guide name from the manifest appears in the primer text."""
    body = prompts._render_primer()
    for name, _, _ in prompts._GUIDE_MANIFEST:
        assert name in body, f"{name} missing from primer"


def test_render_primer_is_ascii():
    """The primer body must be plain ASCII (workspace convention)."""
    prompts._render_primer().encode("ascii")  # raises UnicodeEncodeError on failure


def test_render_primer_has_three_blocks():
    """The primer has a mandatory-core header, a manifest header, and a closing directive."""
    body = prompts._render_primer()
    assert "DERIVA-ML AGENT GUIDELINES" in body  # block 1 header
    assert "ON-DEMAND GUIDES" in body  # block 2 header
    assert "get_guide" in body  # block 3 references on-demand fetch


def test_get_guide_returns_plugin_guide_body(ctx, capturing_mcp):
    """get_guide returns the full body for a plugin-owned guide."""
    prompts.register(ctx)
    get_guide = capturing_mcp.tools["get_guide"]
    assert get_guide(name="deriva_ml_concepts") == prompts._CONCEPTS_GUIDE
    assert get_guide(name="deriva_ml_getting_started") == prompts._GETTING_STARTED_GUIDE


def test_get_guide_redirects_core_guide(ctx, capturing_mcp):
    """get_guide returns a slash-command redirect for a core guide name."""
    prompts.register(ctx)
    get_guide = capturing_mcp.tools["get_guide"]
    result = get_guide(name="query_guide")
    assert "query_guide" in result
    assert "/<server>:" in result or "slash-command" in result


def test_get_guide_unknown_name_errors(ctx, capturing_mcp):
    """get_guide returns a structured error for an unknown name."""
    import json as _json

    prompts.register(ctx)
    get_guide = capturing_mcp.tools["get_guide"]
    result = get_guide(name="does_not_exist")
    payload = _json.loads(result)
    assert "error" in payload
    # The error lists the valid names so the agent can recover.
    assert "deriva_ml_concepts" in payload["error"]


def test_get_guide_registered_read_only(ctx, capturing_mcp):
    """get_guide is a read-only tool."""
    prompts.register(ctx)
    assert "get_guide" in capturing_mcp.tools
