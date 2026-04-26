"""Live-catalog integration tests for deriva-ml-mcp feature tools.

Same shape as ``tests/test_integration.py`` — gated by the
``integration`` pytest marker and a ``skipif`` that probes
``${DERIVA_HOST:-localhost}:443``. The catalog session fixture
(``demo_catalog``) and the ``_server_reachable`` helper live in
``tests/conftest.py`` so this file and ``test_integration.py`` share
them.

Scope of v1 (per Task 3.4 plan): a read-side round-trip only —
``list_features`` (page mode + preflight) is exercised against an
empty schema-shaped catalog. Write-side tools (``create_feature``,
``delete_feature``, ``add_feature_values``) require the execution
scaffolding that lands in Phase 5; their unit-test mocks already
cover the envelope/audit shape, and a follow-up integration test can
add end-to-end write coverage once Phase 5 ships.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pytest
from deriva.core import get_credential

# `set_current_credential` is private-by-convention in deriva_mcp_core
# (the docstring says "Not intended for use in tool or resource handlers"),
# but tests aren't tool handlers. We import it here to seed the per-request
# contextvar that ml_context.get_ml() reads — mirroring what stdio-mode
# server startup does in deriva_mcp_core/server.py. Same rationale as in
# test_integration.py; see that file's import comment for details.
from deriva_mcp_core.context import set_current_credential
from deriva_mcp_core.plugin.api import PluginContext, _set_plugin_context

from tests.conftest import _CapturingMCP, _server_reachable

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _server_reachable(),
        reason=(f"No Deriva server reachable on {os.environ.get('DERIVA_HOST', 'localhost')}:443"),
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def integration_feature_tools(demo_catalog: tuple[str, str]) -> Iterator[_CapturingMCP]:
    """Register feature tools against a fresh ``_CapturingMCP`` and seed the
    per-request credential contextvar so ``get_ml(...)`` resolves correctly.

    Mirrors ``integration_dataset_tools`` in ``test_integration.py``: same
    PluginContext / contextvar pattern, swapping in the ``feature`` module
    instead of ``dataset``.
    """
    hostname, _ = demo_catalog
    credential = get_credential(hostname)
    set_current_credential(credential)

    capturing = _CapturingMCP()
    plugin_ctx = PluginContext(capturing)
    _set_plugin_context(plugin_ctx)
    try:
        from deriva_ml_mcp.tools import feature as feature_module

        feature_module.register(plugin_ctx)
        yield capturing
    finally:
        _set_plugin_context(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------


async def test_feature_read_round_trip(
    demo_catalog: tuple[str, str],
    integration_feature_tools: _CapturingMCP,
) -> None:
    """Exercise read-only feature tools against a real (empty) catalog.

    We deliberately scope to the read side: ``create_feature`` /
    ``delete_feature`` / ``add_feature_values`` need an existing
    ``Execution`` (and a started execution context for value writes),
    which depends on the execution-domain scaffolding scheduled for
    Phase 5. The read-side tools alone validate that:

    - The plugin's tool registration works against a real PluginContext.
    - ``ml_context.get_ml`` builds a working DerivaML using the
      credential contextvar set by the fixture.
    - ``list_features``' JSON envelope matches the documented shape
      for an empty catalog (zero features, both paginated and preflight
      modes).
    """
    hostname, catalog_id = demo_catalog
    tools = integration_feature_tools.tools

    # Page mode against an empty schema: zero features, no pagination.
    page_out = json.loads(await tools["list_features"](hostname=hostname, catalog_id=catalog_id))
    assert "features" in page_out
    assert "count" in page_out
    assert "truncated" in page_out
    assert "next_after_rid" in page_out
    assert isinstance(page_out["features"], list)
    assert page_out["count"] == len(page_out["features"])
    assert page_out["count"] == 0
    assert page_out["truncated"] is False
    assert page_out["next_after_rid"] is None

    # Preflight mode: no entities fetched, total_count present and zero.
    preflight_out = json.loads(
        await tools["list_features"](hostname=hostname, catalog_id=catalog_id, preflight_count=True)
    )
    assert preflight_out["entities_fetched"] is False
    assert isinstance(preflight_out["total_count"], int)
    assert preflight_out["total_count"] == 0
    assert "action_required" in preflight_out
