"""Shared pytest fixtures for deriva-ml-mcp tests.

Mirrors the unit-test fixture pattern from
``deriva-mcp-core/tests/test_tools.py`` (``_CapturingMCP``) and the
integration-test patterns from ``deriva-mcp/tests/conftest.py``.

Integration-test fixtures (``deriva_host``, ``demo_catalog``) and the
``_server_reachable()`` helper live here so multiple integration test
files can share them. Each integration file still defines its own
``pytestmark`` (the ``integration`` marker + ``skipif`` gate) — see
``test_integration.py`` and ``test_integration_feature.py``.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from deriva_mcp_core.plugin.api import PluginContext, _set_plugin_context


def _success_calls(mock_audit: Any, event_name: str) -> list:
    """Return audit_event call records whose first positional arg matches.

    Generic over any ``audit_event`` MagicMock — works for dataset,
    feature, workflow, execution, and any future domain that emits
    ``deriva_ml_<op>`` / ``deriva_ml_<op>_failed`` audit events.

    Args:
        mock_audit: A ``MagicMock`` standing in for
            ``deriva_mcp_core.telemetry.audit_event``.
        event_name: The exact event name (positional arg 0) to filter
            for, e.g. ``"deriva_ml_create_dataset"`` or
            ``"deriva_ml_create_dataset_failed"``.

    Returns:
        List of ``unittest.mock.call`` records where ``call.args[0] ==
        event_name``. Empty list if no matching calls were made.

    Example:
        >>> from unittest.mock import MagicMock
        >>> m = MagicMock()
        >>> m("deriva_ml_create_dataset", dataset_rid="1-AAAA")
        >>> calls = _success_calls(m, "deriva_ml_create_dataset")
        >>> len(calls)
        1
        >>> calls[0].kwargs["dataset_rid"]
        '1-AAAA'
    """
    return [c for c in mock_audit.call_args_list if c.args and c.args[0] == event_name]


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


# ---------------------------------------------------------------------------
# Integration-test helpers and session fixtures
# ---------------------------------------------------------------------------


def _server_reachable() -> bool:
    """Quick TCP probe to ``${DERIVA_HOST:-localhost}:443``.

    Used by integration test modules in their ``pytestmark`` ``skipif``
    gate. Lives here so multiple integration test files can import the
    same helper rather than defining it independently.

    Returns:
        True if a TCP connect succeeded within 2 seconds, False otherwise.

    Example:
        >>> # In an integration test file:
        >>> from tests.conftest import _server_reachable
        >>> import pytest
        >>> pytestmark = [
        ...     pytest.mark.integration,
        ...     pytest.mark.skipif(not _server_reachable(), reason="..."),
        ... ]
    """
    host = os.environ.get("DERIVA_HOST", "localhost")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, 443))
        sock.close()
        return result == 0
    except Exception:
        return False


@pytest.fixture(scope="session")
def deriva_host() -> str:
    """Hostname of the Deriva server under test.

    Returns:
        The value of ``$DERIVA_HOST`` if set, otherwise ``"localhost"``.
    """
    return os.environ.get("DERIVA_HOST", "localhost")


@pytest.fixture(scope="session")
def demo_catalog(deriva_host: str) -> Iterator[tuple[str, str]]:
    """Spin up a demo catalog for the test session and tear it down after.

    Session-scoped so multiple integration tests can share the same
    empty schema-shaped catalog. Cleanup runs deterministically at
    session end via ``destroy_demo_catalog`` rather than relying on
    ``create_demo_catalog``'s atexit hook.

    Args:
        deriva_host: Hostname injected by the ``deriva_host`` fixture.

    Yields:
        ``(hostname, catalog_id)`` tuple. ``catalog_id`` is stringified
        because tool signatures take it as ``str``.

    Example:
        >>> def test_smoke(demo_catalog):  # doctest: +SKIP
        ...     host, catalog_id = demo_catalog
        ...     # ... call tools with host + catalog_id ...
    """
    from deriva_ml.demo_catalog import create_demo_catalog, destroy_demo_catalog

    catalog = create_demo_catalog(
        deriva_host,
        domain_schema="demo-schema",
        project_name="ml-mcp-int-test",
        populate=False,
        create_features=False,
        create_datasets=False,
        on_exit_delete=False,
    )
    try:
        yield deriva_host, str(catalog.catalog_id)
    finally:
        destroy_demo_catalog(catalog)
