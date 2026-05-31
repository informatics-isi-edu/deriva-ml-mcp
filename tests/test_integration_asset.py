"""Live-catalog integration tests for deriva-ml-mcp-plugin asset tools.

Same shape as ``tests/test_integration_workflow.py`` — gated by the
``integration`` pytest marker and a ``skipif`` that probes
``${DERIVA_HOST:-localhost}:443``. The shared ``_server_reachable``
helper and the ``deriva_host`` session fixture live in
``tests/conftest.py``.

Scope: the three asset tools (``deriva_ml_list_assets``,
``deriva_ml_lookup_asset``, ``deriva_ml_update_asset``) exercised
end-to-end against a live demo catalog. The bare ``demo_catalog``
fixture is sufficient: it ships with the ``Image`` asset table
(created by ``create_domain_schema``) plus the standard
``Execution_Metadata`` / ``Execution_Asset`` tables in the
``deriva-ml`` schema. Asset table discovery is exercised through
the ``ml/assets/{schema}`` resource in
``tests/test_integration_resources.py``.

Coverage limits: the bare demo catalog has no asset rows (asset
upload requires a full execution lifecycle — exercised in
``test_integration_execution.py``). This file therefore covers:

- ``list_assets`` (empty + preflight) -- empty-table behavior.
- ``lookup_asset`` -- error envelope for a nonexistent RID.
- ``update_asset`` -- argument-validation envelope (pre-catalog) and
  error envelope when the RID does not resolve.

Write-side success-path coverage for ``update_asset`` (real asset
mutations) is not exercised here -- that requires a populated
catalog with ``Execution_Metadata`` rows from a prior commit, which
is out of scope for an isolated asset integration suite. Unit tests
in ``tests/test_asset.py`` cover the success path with mocks.
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
# the other integration test modules.
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
def integration_asset_tools(
    demo_catalog: tuple[str, str],
) -> Iterator[_CapturingMCP]:
    """Register asset tools against a fresh ``_CapturingMCP`` and seed the
    per-request credential contextvar so ``get_ml(...)`` resolves correctly.

    Uses the read-only ``demo_catalog`` fixture (not
    ``demo_mutation_catalog``): asset enumeration tools are read-only, the
    ``update_asset`` calls in this file all hit the failure path, and the
    bare demo catalog has no asset rows whose state we could pollute.
    Sharing the read-only catalog with ``test_integration.py`` and
    ``test_integration_resources.py`` keeps the per-session catalog count
    minimal.
    """
    hostname, _ = demo_catalog
    credential = get_credential(hostname)
    set_current_credential(credential)

    capturing = _CapturingMCP()
    plugin_ctx = PluginContext(capturing)
    _set_plugin_context(plugin_ctx)
    try:
        from deriva_ml_mcp_plugin.tools import asset as asset_module

        asset_module.register(plugin_ctx)
        yield capturing
    finally:
        _set_plugin_context(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Read-side tools
# ---------------------------------------------------------------------------


async def test_list_assets_empty_image_table(
    demo_catalog: tuple[str, str],
    integration_asset_tools: _CapturingMCP,
) -> None:
    """``list_assets`` on an empty asset table returns the documented page shape.

    The bare demo catalog has the ``Image`` asset table but no rows.
    The page envelope must still carry the four pagination fields with
    the documented empty-page values.
    """
    hostname, catalog_id = demo_catalog
    tools = integration_asset_tools.tools

    out = json.loads(
        await tools["deriva_ml_list_assets"](
            hostname=hostname,
            catalog_id=catalog_id,
            asset_table="Image",
        )
    )
    assert out.get("error") is None, f"list_assets returned error: {out}"
    assert out["assets"] == []
    assert out["count"] == 0
    assert out["truncated"] is False
    assert out["next_after_rid"] is None


async def test_list_assets_preflight_count(
    demo_catalog: tuple[str, str],
    integration_asset_tools: _CapturingMCP,
) -> None:
    """``list_assets(preflight_count=True)`` returns the preflight envelope.

    Empty table -> ``total_count == 0`` and the documented
    ``action_required`` hint string.
    """
    hostname, catalog_id = demo_catalog
    tools = integration_asset_tools.tools

    out = json.loads(
        await tools["deriva_ml_list_assets"](
            hostname=hostname,
            catalog_id=catalog_id,
            asset_table="Image",
            preflight_count=True,
        )
    )
    assert out.get("error") is None, f"preflight returned error: {out}"
    assert out["total_count"] == 0
    assert out["entities_fetched"] is False
    assert "preflight_count=False" in out["action_required"]


async def test_list_assets_unknown_table_returns_error(
    demo_catalog: tuple[str, str],
    integration_asset_tools: _CapturingMCP,
) -> None:
    """An unknown ``asset_table`` argument surfaces as an error envelope.

    The pre-catalog signature accepts any string; the failure happens
    inside ``ml.list_assets`` and is caught by ``_error_envelope``.
    """
    hostname, catalog_id = demo_catalog
    tools = integration_asset_tools.tools

    out = json.loads(
        await tools["deriva_ml_list_assets"](
            hostname=hostname,
            catalog_id=catalog_id,
            asset_table="NoSuchAssetTable",
        )
    )
    assert "error" in out


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


async def test_lookup_asset_unknown_rid_returns_error(
    demo_catalog: tuple[str, str],
    integration_asset_tools: _CapturingMCP,
) -> None:
    """``lookup_asset`` against a nonexistent RID surfaces as an error envelope.

    The demo catalog has no asset rows, so any RID resolves to a
    deriva-ml ``DerivaMLNotFoundError`` (or similar) which the tool
    wraps via ``_error_envelope``. ``audit=False`` for read tools, so
    no audit assertion is needed here.
    """
    hostname, catalog_id = demo_catalog
    tools = integration_asset_tools.tools

    out = json.loads(
        await tools["deriva_ml_lookup_asset"](
            hostname=hostname,
            catalog_id=catalog_id,
            asset_rid="9-NOPE",
        )
    )
    assert "error" in out


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


async def test_update_asset_validation_envelope(
    demo_catalog: tuple[str, str],
    integration_asset_tools: _CapturingMCP,
) -> None:
    """``update_asset`` with neither ``asset_types`` nor ``description``
    returns the documented validation error before touching the catalog.

    The error path here does not emit an audit event (the validation
    branch returns before the audit-bearing ``with deriva_call():``
    body), which mirrors the behavior pinned by unit tests in
    ``tests/test_asset.py``.
    """
    hostname, catalog_id = demo_catalog
    tools = integration_asset_tools.tools

    out = json.loads(
        await tools["deriva_ml_update_asset"](
            hostname=hostname,
            catalog_id=catalog_id,
            asset_rid="9-NOPE",
        )
    )
    assert "error" in out
    assert "asset_types or description" in out["error"]


async def test_update_asset_unknown_rid_returns_error(
    demo_catalog: tuple[str, str],
    integration_asset_tools: _CapturingMCP,
) -> None:
    """``update_asset`` against a nonexistent RID surfaces as an error envelope.

    Validates that the failure path through ``_error_envelope`` works
    against a live catalog (the ``ml.lookup_asset`` call inside the
    tool body is what raises). Unit tests in ``tests/test_asset.py``
    pin the audit-event side effect; this test only verifies the wire
    shape against a real catalog.
    """
    hostname, catalog_id = demo_catalog
    tools = integration_asset_tools.tools

    out = json.loads(
        await tools["deriva_ml_update_asset"](
            hostname=hostname,
            catalog_id=catalog_id,
            asset_rid="9-NOPE",
            description="updated",
        )
    )
    assert "error" in out
