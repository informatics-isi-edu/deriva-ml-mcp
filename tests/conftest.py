"""Shared pytest fixtures for deriva-ml-mcp tests.

Mirrors the unit-test fixture pattern from
``deriva-mcp-core/tests/test_tools.py`` (``_CapturingMCP``) and the
integration-test patterns from ``deriva-mcp/tests/conftest.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from deriva_mcp_core.plugin.api import PluginContext, _set_plugin_context


class _CapturingMCP:
    """Minimal FastMCP stand-in that stores registered tools and resources.

    Used by unit tests to invoke registered tools directly without spinning
    up a real MCP server. Mirrors the fixture in
    ``deriva-mcp-core/tests/test_tools.py``.
    """

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.tool_kwargs: dict[str, dict[str, Any]] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(self, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            self.tool_kwargs[fn.__name__] = kwargs
            return fn

        return decorator

    def resource(self, uri, *args, **kwargs):
        def decorator(fn):
            self.resources[uri] = fn
            return fn

        return decorator

    def prompt(self, name=None, *args, **kwargs):
        def decorator(fn):
            self.prompts[name or fn.__name__] = fn
            return fn

        return decorator


@pytest.fixture()
def capturing_mcp():
    """Bare _CapturingMCP for tests that don't need a PluginContext wrapper."""
    return _CapturingMCP()


@pytest.fixture()
def ctx(capturing_mcp):
    """PluginContext wrapping a fresh _CapturingMCP, with the contextvar set."""
    plugin_ctx = PluginContext(capturing_mcp)
    _set_plugin_context(plugin_ctx)
    yield plugin_ctx
    _set_plugin_context(None)  # type: ignore[arg-type]


@pytest.fixture()
def mock_ml():
    """A MagicMock standing in for a DerivaML instance.

    Per-test setup configures return values for the methods being exercised.
    """
    return MagicMock()
