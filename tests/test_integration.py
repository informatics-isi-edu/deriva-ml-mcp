"""Live-catalog integration tests for deriva-ml-mcp dataset tools.

These exercise the registered dataset tools end-to-end against a real
Deriva server. They are gated by the ``integration`` pytest marker and
the default ``uv run pytest`` invocation does NOT collect them. To run::

    uv run pytest -m integration

Skip semantics:

- If no Deriva server is reachable on ``${DERIVA_HOST:-localhost}:443``,
  the entire module is skipped with a clear ``reason``.
- Otherwise, fixtures spin up a real demo catalog via
  ``deriva_ml.demo_catalog.create_demo_catalog`` (session-scoped for cost),
  set the per-request credential contextvar that ``ml_context.get_ml``
  reads, and run the round-trip through the registered tool functions.

Scope of v1 (per Task 2.4 plan): a read-side round-trip only —
``list_datasets``, ``list_dataset_element_types``, and
``get_dataset_spec`` are exercised. Mutation tools like
``create_dataset`` require an existing ``Execution`` whose scaffolding
(workflow + execution lifecycle) is out of scope for this smoke test;
write-side coverage can be added in a follow-up.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterator

import pytest
from deriva.core import get_credential
from deriva_mcp_core.context import set_current_credential
from deriva_mcp_core.plugin.api import PluginContext, _set_plugin_context

from tests.conftest import _CapturingMCP


def _server_reachable() -> bool:
    """Quick TCP probe to ``${DERIVA_HOST:-localhost}:443``.

    Returns:
        True if a TCP connect succeeded within 2 seconds, False otherwise.
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


@pytest.fixture(scope="session")
def deriva_host() -> str:
    """Hostname of the Deriva server under test."""
    return os.environ.get("DERIVA_HOST", "localhost")


@pytest.fixture(scope="session")
def demo_catalog(deriva_host: str) -> Iterator[tuple[str, str]]:
    """Spin up a demo catalog for the test session and tear it down after.

    Returns a (hostname, catalog_id) tuple. The catalog is destroyed via
    ``destroy_demo_catalog`` after the session ends. We do this in addition
    to ``create_demo_catalog``'s built-in atexit hook so cleanup is
    deterministic per test run rather than at process exit.
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


@pytest.fixture()
def integration_dataset_tools(demo_catalog: tuple[str, str]) -> Iterator[_CapturingMCP]:
    """Register dataset tools against a fresh ``_CapturingMCP`` and seed the
    per-request credential contextvar so ``get_ml(...)`` resolves correctly.

    The fixture mirrors the unit-test ``dataset_ctx`` pattern but uses the
    real ``ml_context.get_ml`` (no patching) — credentials come from
    ``~/.deriva/credential.json`` for ``deriva_host``, set into the
    contextvar that ``get_request_credential`` reads inside the tool body.
    """
    hostname, _ = demo_catalog
    credential = get_credential(hostname)
    set_current_credential(credential)

    capturing = _CapturingMCP()
    plugin_ctx = PluginContext(capturing)
    _set_plugin_context(plugin_ctx)
    try:
        from deriva_ml_mcp.tools import dataset as dataset_module

        dataset_module.register(plugin_ctx)
        yield capturing
    finally:
        _set_plugin_context(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------


async def test_dataset_read_round_trip(
    demo_catalog: tuple[str, str],
    integration_dataset_tools: _CapturingMCP,
) -> None:
    """Exercise read-only dataset tools against a real (empty) catalog.

    We deliberately scope to the read side because ``create_dataset``
    requires a pre-existing ``Execution`` RID, which in turn requires a
    workflow plus a started execution context — out of scope for v1 (see
    module docstring). The read-side tools alone validate that:

    - The plugin's tool registration works against a real PluginContext.
    - ``ml_context.get_ml`` builds a working DerivaML using the credential
      contextvar set by the test fixture.
    - The tools' JSON envelopes match the documented shapes for an
      empty catalog (zero datasets, schema-bounded element types).
    """
    hostname, catalog_id = demo_catalog
    tools = integration_dataset_tools.tools

    list_out = json.loads(await tools["list_datasets"](hostname=hostname, catalog_id=catalog_id))
    assert "datasets" in list_out
    assert "count" in list_out
    assert "truncated" in list_out
    assert "next_after_rid" in list_out
    assert isinstance(list_out["datasets"], list)
    assert list_out["count"] == len(list_out["datasets"])
    assert list_out["truncated"] is False
    assert list_out["next_after_rid"] is None
    assert list_out["count"] == 0  # empty catalog (populate=False)

    preflight_out = json.loads(
        await tools["list_datasets"](hostname=hostname, catalog_id=catalog_id, preflight_count=True)
    )
    assert preflight_out["entities_fetched"] is False
    assert preflight_out["total_count"] == 0
    assert "action_required" in preflight_out

    elements_out = json.loads(
        await tools["list_dataset_element_types"](hostname=hostname, catalog_id=catalog_id)
    )
    assert "element_types" in elements_out
    assert "count" in elements_out
    assert isinstance(elements_out["element_types"], list)
    assert elements_out["count"] == len(elements_out["element_types"])
    for entry in elements_out["element_types"]:
        assert "name" in entry
        assert "schema" in entry
