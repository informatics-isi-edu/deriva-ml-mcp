"""Shared pytest fixtures for deriva-ml-mcp tests.

Module-level utilities (``_success_calls``, ``make_patch_audit``,
``_CapturingMCP``, ``_server_reachable``) live in ``tests/_helpers.py``
and are imported here so they're discoverable from the conventional
location AND from explicit consumer imports (``from tests._helpers
import ...``).

Integration-test fixtures (``deriva_host``, ``demo_catalog``,
``demo_mutation_catalog``) live here so multiple integration test
files can share them. Each integration file still defines its own
``pytestmark`` (the ``integration`` marker + ``skipif`` gate) — see
``test_integration.py``, ``test_integration_feature.py``, and
``test_integration_workflow.py``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from deriva_mcp_core.plugin.api import PluginContext, _set_plugin_context

# Re-export module-level utilities so consumers can import either via
# ``from tests.conftest import _success_calls`` (existing call sites)
# or via ``from tests._helpers import _success_calls`` (preferred for
# new code).
from tests._helpers import (  # noqa: F401
    _CapturingMCP,
    _server_reachable,
    _success_calls,
    make_patch_audit,
)


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
# Integration-test session fixtures
# ---------------------------------------------------------------------------


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


@pytest.fixture(scope="session")
def demo_mutation_catalog(deriva_host: str) -> Iterator[tuple[str, str]]:
    """Dedicated demo catalog for write-side integration tests.

    Distinct from ``demo_catalog`` so mutation tests (workflow create,
    execution create, etc.) cannot pollute the read-only fixture.
    Read-side tests assert empty-catalog invariants (e.g. ``count == 0``)
    that would break if a mutation row were created in the same catalog
    earlier in the session.

    Same shape as ``demo_catalog`` (empty schema, no populated rows
    or datasets) — only the identity differs. Session-scoped, so the
    ~10s catalog provisioning cost is paid once for the whole mutation
    test surface.

    Args:
        deriva_host: Hostname injected by the ``deriva_host`` fixture.

    Yields:
        ``(hostname, catalog_id)`` tuple. ``catalog_id`` is stringified
        because tool signatures take it as ``str``.

    Example:
        >>> def test_workflow_round_trip(demo_mutation_catalog):  # doctest: +SKIP
        ...     host, catalog_id = demo_mutation_catalog
        ...     # ... call create_workflow, update_workflow, etc. ...
    """
    from deriva_ml.demo_catalog import create_demo_catalog, destroy_demo_catalog

    catalog = create_demo_catalog(
        deriva_host,
        domain_schema="demo-schema",
        project_name="ml-mcp-int-mut",
        populate=False,
        create_features=False,
        create_datasets=False,
        on_exit_delete=False,
    )
    try:
        yield deriva_host, str(catalog.catalog_id)
    finally:
        destroy_demo_catalog(catalog)
