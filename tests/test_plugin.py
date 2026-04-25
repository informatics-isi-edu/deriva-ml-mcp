"""Smoke tests for the deriva-ml-mcp plugin entry point."""

from __future__ import annotations

from importlib import metadata

from deriva_ml_mcp.plugin import register


def test_register_runs_without_error(ctx):
    """``register(ctx)`` must succeed even when no domain modules exist yet."""
    register(ctx)


def test_entry_point_resolves_to_register():
    """The 'deriva-ml' entry point must resolve to our register function."""
    eps = metadata.entry_points(group="deriva_mcp.plugins")
    matching = [ep for ep in eps if ep.name == "deriva-ml"]
    assert matching, "entry point 'deriva-ml' not declared"
    assert matching[0].load() is register


def test_register_registers_no_tools_in_phase_0(ctx, capturing_mcp):
    """Phase 0: register() is a no-op. Asserting this catches accidental
    additions before the corresponding phase lands."""
    register(ctx)
    assert capturing_mcp.tools == {}
    assert capturing_mcp.resources == {}
