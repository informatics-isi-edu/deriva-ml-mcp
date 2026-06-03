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
    }
)


def test_register_runs_without_error(ctx):
    """``prompts.register(ctx)`` must succeed end-to-end."""
    prompts.register(ctx)


def test_two_prompts_registered(ctx, capturing_mcp):
    """Exactly two prompts land in the capturing MCP after register."""
    prompts.register(ctx)
    assert len(capturing_mcp.prompts) == 2


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


def test_render_primer_lists_all_core_guide_names():
    """Every core guide name from the manifest appears in the primer text."""
    body = prompts._render_primer()
    for name, source, _ in prompts._GUIDE_MANIFEST:
        assert name in body, f"{name} missing from primer"


def test_render_primer_is_ascii():
    """The primer body must be plain ASCII (workspace convention)."""
    prompts._render_primer().encode("ascii")  # raises UnicodeEncodeError on failure


def test_render_primer_has_three_blocks():
    """The primer has a mandatory-core header, a manifest header, and a closing directive."""
    body = prompts._render_primer()
    assert "DERIVA-ML AGENT GUIDELINES" in body  # block 1 header
    assert "ON-DEMAND GUIDES" in body            # block 2 header
    assert "get_guide" in body                   # block 3 references on-demand fetch
