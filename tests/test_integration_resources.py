"""Live-catalog integration tests for deriva-ml-mcp ML resources.

Same shape as ``tests/test_integration_workflow.py`` and
``tests/test_integration_feature.py`` -- gated by the ``integration``
pytest marker and a ``skipif`` that probes
``${DERIVA_HOST:-localhost}:443``. The shared ``_server_reachable``
helper and the ``deriva_host`` / ``demo_catalog`` session fixtures
live in ``tests/conftest.py``.

Scope (per Phase 6.5 plan): a read-only round-trip across the 9 MCP
resource URIs registered by ``src/deriva_ml_mcp/resources/ml.py``'s
``register(ctx)`` function. Resources are read-only by definition, so
this file uses the shared ``demo_catalog`` fixture (NOT the
``demo_mutation_catalog`` reserved for write-side integration tests).

The demo catalog is created with ``populate=False``,
``create_features=False``, ``create_datasets=False`` (see
``conftest.py:demo_catalog``), so list endpoints are expected to
return zero rows and detail endpoints called with a clearly-fake RID
are expected to return an ``error`` envelope. The ``registries``
snapshot still returns all four vocabulary keys; the ones the demo
schema seeds (e.g. ``Workflow_Type``, ``Execution_Status``) come back
populated, the ones it doesn't (``Dataset_Type``, ``Asset_Type``)
come back as ``[]`` per the resource's best-effort contract.
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
# contextvar that ml_context.get_ml() reads -- mirroring what stdio-mode
# server startup does in deriva_mcp_core/server.py. Same rationale as in
# test_integration_workflow.py / test_integration_feature.py; see those
# files' import comments for details.
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


# Resource URI templates -- copies of the keys that resources/ml.py registers.
# Kept here as constants so test functions read like the unit tests in
# tests/test_resources.py.
_DATASETS_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/datasets"
_DATASET_DETAIL_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}"
_DATASET_MEMBERS_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/members"
_WORKFLOWS_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/workflows"
_WORKFLOW_DETAIL_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/workflow/{workflow_rid}"
_EXECUTIONS_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/executions"
_EXECUTION_DETAIL_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/execution/{execution_rid}"
_FEATURES_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/features/{table_name}"
_REGISTRIES_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/registries"

# A clearly-fake RID for the detail-error tests. The catalog is empty so
# any RID would do, but one obviously not produced by Deriva makes the
# intent clear when reading the test.
_FAKE_RID = "99-FAKE"

# Demo schema target table for the features endpoint. The demo catalog's
# ``demo-schema`` (built by ``deriva_ml.demo_catalog.create_demo_catalog``
# at conftest.py:demo_catalog) seeds Subject, Image, Observation, Report,
# OCR_Report, ClinicalRecord, ClinicalRecord_Observation. We use
# ``Subject`` because it's a top-level entity that ``create_features=True``
# would attach a "Health" feature to -- so on this empty-features fixture
# the resource correctly returns ``features == []`` for a real table.
_DEMO_FEATURE_TABLE = "Subject"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def integration_resources(
    demo_catalog: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_CapturingMCP]:
    """Register ML resources against a fresh ``_CapturingMCP`` and seed the
    per-request credential contextvar so ``get_ml(...)`` resolves correctly.

    Mirrors ``integration_workflow_tools`` / ``integration_feature_tools``:
    same PluginContext / contextvar pattern, swapping in the
    ``resources.ml`` module. Sets ``DERIVA_ML_ALLOW_DIRTY=true`` for the
    same reason the workflow integration fixture does -- DerivaML's
    Workflow / Execution validators call git provenance helpers eagerly
    even on read paths, and the test runner is almost certainly executing
    from a dirty checkout. ``monkeypatch`` ensures the var is unset
    after the test.

    Note: deliberately registers ONLY ``resources.ml``, NOT
    ``resources.rag``. RAG indexing fires ``on_catalog_connect`` hooks
    that try to talk to the live catalog; Phase 6.5 is resources-only.
    """
    monkeypatch.setenv("DERIVA_ML_ALLOW_DIRTY", "true")

    hostname, _ = demo_catalog
    credential = get_credential(hostname)
    set_current_credential(credential)

    capturing = _CapturingMCP()
    plugin_ctx = PluginContext(capturing)
    _set_plugin_context(plugin_ctx)
    try:
        from deriva_ml_mcp.resources import ml as ml_module

        ml_module.register(plugin_ctx)
        yield capturing
    finally:
        _set_plugin_context(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# List endpoints -- empty-catalog invariants
# ---------------------------------------------------------------------------


async def test_resource_ml_datasets_empty_catalog(
    demo_catalog: tuple[str, str],
    integration_resources: _CapturingMCP,
) -> None:
    """``ml/datasets`` on an empty catalog returns an empty page envelope."""
    hostname, catalog_id = demo_catalog
    out = json.loads(
        await integration_resources.resources[_DATASETS_URI](
            hostname=hostname, catalog_id=catalog_id
        )
    )
    assert out.get("error") is None, f"resource returned error: {out}"
    assert out["datasets"] == []
    assert out["count"] == 0
    assert out["truncated"] is False
    assert out["next_after_rid"] is None


async def test_resource_ml_workflows_empty_catalog(
    demo_catalog: tuple[str, str],
    integration_resources: _CapturingMCP,
) -> None:
    """``ml/workflows`` on an empty catalog returns an empty page envelope."""
    hostname, catalog_id = demo_catalog
    out = json.loads(
        await integration_resources.resources[_WORKFLOWS_URI](
            hostname=hostname, catalog_id=catalog_id
        )
    )
    assert out.get("error") is None, f"resource returned error: {out}"
    assert out["workflows"] == []
    assert out["count"] == 0
    assert out["truncated"] is False
    assert out["next_after_rid"] is None


async def test_resource_ml_executions_empty_catalog(
    demo_catalog: tuple[str, str],
    integration_resources: _CapturingMCP,
) -> None:
    """``ml/executions`` on an empty catalog returns an empty page envelope."""
    hostname, catalog_id = demo_catalog
    out = json.loads(
        await integration_resources.resources[_EXECUTIONS_URI](
            hostname=hostname, catalog_id=catalog_id
        )
    )
    assert out.get("error") is None, f"resource returned error: {out}"
    assert out["executions"] == []
    assert out["count"] == 0
    assert out["truncated"] is False
    assert out["next_after_rid"] is None


# ---------------------------------------------------------------------------
# Detail endpoints -- unknown RID returns ``{"error": ...}``
# ---------------------------------------------------------------------------


async def test_resource_ml_dataset_detail_returns_error_for_unknown_rid(
    demo_catalog: tuple[str, str],
    integration_resources: _CapturingMCP,
) -> None:
    """``ml/dataset/{rid}`` with a fake RID returns an error envelope."""
    hostname, catalog_id = demo_catalog
    out = json.loads(
        await integration_resources.resources[_DATASET_DETAIL_URI](
            hostname=hostname, catalog_id=catalog_id, dataset_rid=_FAKE_RID
        )
    )
    assert "error" in out
    assert isinstance(out["error"], str) and out["error"]


async def test_resource_ml_dataset_members_returns_error_for_unknown_rid(
    demo_catalog: tuple[str, str],
    integration_resources: _CapturingMCP,
) -> None:
    """``ml/dataset/{rid}/members`` with a fake RID returns an error envelope."""
    hostname, catalog_id = demo_catalog
    out = json.loads(
        await integration_resources.resources[_DATASET_MEMBERS_URI](
            hostname=hostname, catalog_id=catalog_id, dataset_rid=_FAKE_RID
        )
    )
    assert "error" in out
    assert isinstance(out["error"], str) and out["error"]


async def test_resource_ml_workflow_detail_returns_error_for_unknown_rid(
    demo_catalog: tuple[str, str],
    integration_resources: _CapturingMCP,
) -> None:
    """``ml/workflow/{rid}`` with a fake RID returns an error envelope."""
    hostname, catalog_id = demo_catalog
    out = json.loads(
        await integration_resources.resources[_WORKFLOW_DETAIL_URI](
            hostname=hostname, catalog_id=catalog_id, workflow_rid=_FAKE_RID
        )
    )
    assert "error" in out
    assert isinstance(out["error"], str) and out["error"]


async def test_resource_ml_execution_detail_returns_error_for_unknown_rid(
    demo_catalog: tuple[str, str],
    integration_resources: _CapturingMCP,
) -> None:
    """``ml/execution/{rid}`` with a fake RID returns an error envelope."""
    hostname, catalog_id = demo_catalog
    out = json.loads(
        await integration_resources.resources[_EXECUTION_DETAIL_URI](
            hostname=hostname, catalog_id=catalog_id, execution_rid=_FAKE_RID
        )
    )
    assert "error" in out
    assert isinstance(out["error"], str) and out["error"]


# ---------------------------------------------------------------------------
# Features endpoint -- real demo-schema table, no features defined
# ---------------------------------------------------------------------------


async def test_resource_ml_features_empty_table(
    demo_catalog: tuple[str, str],
    integration_resources: _CapturingMCP,
) -> None:
    """``ml/features/{table_name}`` against a real demo-schema table.

    The demo catalog is built with ``create_features=False``, so even
    for ``Subject`` (which the demo's feature seeder would otherwise
    target with a ``Health`` feature), the resource returns an empty
    feature list.
    """
    hostname, catalog_id = demo_catalog
    out = json.loads(
        await integration_resources.resources[_FEATURES_URI](
            hostname=hostname, catalog_id=catalog_id, table_name=_DEMO_FEATURE_TABLE
        )
    )
    assert out.get("error") is None, f"resource returned error: {out}"
    assert out["features"] == []
    assert out["count"] == 0
    assert out["truncated"] is False
    assert out["next_after_rid"] is None


# ---------------------------------------------------------------------------
# Registries snapshot -- four vocab keys, some seeded by demo schema
# ---------------------------------------------------------------------------


async def test_resource_ml_registries_returns_all_four_vocabs(
    demo_catalog: tuple[str, str],
    integration_resources: _CapturingMCP,
) -> None:
    """``ml/registries`` returns all four vocab keys; missing vocabs are ``[]``.

    The demo catalog seeds the standard ML vocabularies (``Workflow_Type``
    and ``Execution_Status`` always have terms; ``Dataset_Type`` and
    ``Asset_Type`` may or may not). We assert structure (all four keys
    present, each a list) and that ``workflow_types`` and
    ``execution_statuses`` are non-empty -- the rest may be empty without
    failing the test, per the resource's best-effort contract.
    """
    hostname, catalog_id = demo_catalog
    out = json.loads(
        await integration_resources.resources[_REGISTRIES_URI](
            hostname=hostname, catalog_id=catalog_id
        )
    )
    assert out.get("error") is None, f"resource returned error: {out}"
    assert set(out.keys()) == {
        "dataset_types",
        "workflow_types",
        "asset_types",
        "execution_statuses",
    }
    for key, value in out.items():
        assert isinstance(value, list), f"{key!r} is not a list: {value!r}"
    # Demo schema seeds these unconditionally -- if either is empty, the
    # demo catalog vocab seeding has regressed.
    assert len(out["workflow_types"]) > 0, "demo catalog should seed Workflow_Type terms"
    assert len(out["execution_statuses"]) > 0, "demo catalog should seed Execution_Status terms"
    # Spot-check term shape on the first workflow type.
    sample = out["workflow_types"][0]
    assert set(sample.keys()) == {"name", "description", "synonyms", "rid"}
    assert isinstance(sample["name"], str) and sample["name"]
    assert isinstance(sample["synonyms"], list)
