"""Unit tests for the three deriva-ml MCP prompts.

Prompts are pure data -- no DERIVA I/O, no catalog access. The tests
exercise registration shape (count, names) and content invariants
(non-empty, ASCII-only). Per-tool behavior tests live with their
respective modules.
"""

from __future__ import annotations

from deriva_ml_mcp import prompts

_EXPECTED_PROMPT_NAMES = frozenset(
    {
        "deriva_ml_getting_started",
        "deriva_ml_execution_lifecycle",
        "deriva_ml_workflow_dedup",
    }
)


def test_register_runs_without_error(ctx):
    """``prompts.register(ctx)`` must succeed end-to-end."""
    prompts.register(ctx)


def test_three_prompts_registered(ctx, capturing_mcp):
    """Exactly three prompts land in the capturing MCP after register."""
    prompts.register(ctx)
    assert len(capturing_mcp.prompts) == 3


def test_prompt_names_exact(ctx, capturing_mcp):
    """The three prompts must be registered under the documented names.

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

    The 100-char floor is a sanity bound -- the shortest guide
    (workflow_dedup) is well over 1500 chars. A regression that
    accidentally empties a guide constant would drop the result to
    near-zero length and fail this check.
    """
    prompts.register(ctx)
    for name, fn in capturing_mcp.prompts.items():
        result = fn()
        assert isinstance(result, str), f"{name} did not return a string"
        assert len(result) > 100, f"{name} returned a near-empty body ({len(result)} chars)"


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
