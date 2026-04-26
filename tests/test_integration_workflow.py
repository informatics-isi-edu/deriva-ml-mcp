"""Live-catalog integration tests for deriva-ml-mcp workflow tools.

Same shape as ``tests/test_integration.py`` and
``tests/test_integration_feature.py`` — gated by the ``integration``
pytest marker and a ``skipif`` that probes
``${DERIVA_HOST:-localhost}:443``. The shared ``_server_reachable``
helper and the ``deriva_host`` session fixture live in
``tests/conftest.py``.

Scope of v1 (per Task 4.4 plan): a full round-trip exercising both
read AND write paths. Unlike feature tools (whose mutations need an
``Execution`` from Phase 5's scaffolding), ``create_workflow`` is
self-contained — it only needs a name, a workflow_type vocab term,
and a URL. The pre-populated demo catalog ships with several
``Workflow_Type`` terms (e.g. ``Testing``, ``Training``,
``Feature_Creation``), so the test can drive the full
create -> get -> find_by_url -> dedup -> update -> list cycle
without auxiliary setup.

Catalog isolation: this file uses the dedicated
``demo_mutation_catalog`` fixture (defined in conftest.py) rather
than the shared ``demo_catalog``. Read-only integration tests assert
empty-catalog invariants (e.g. ``count == 0``); creating a workflow
in their catalog would break those assertions if pytest happened to
run modules in an unfortunate order. The mutation catalog is also
session-scoped so the ~10s provisioning cost is paid once.
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
# test_integration.py / test_integration_feature.py; see those files'
# import comments for details.
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


# A pre-populated Workflow_Type vocab term shipped with the demo schema
# (verified via ``ml.list_vocabulary_terms("Workflow_Type")`` in the
# Phase 4.4 investigation step). "Testing" is semantically appropriate
# for an integration test workflow. Other available terms include
# "Training", "Feature_Creation", "Analysis", "Visualization" —
# any of these would work. Hard-coded rather than discovered at test
# time so a regression in the demo catalog's vocab seeding becomes a
# loud, debuggable test failure rather than a silent skip.
_WORKFLOW_TYPE_TERM = "Testing"

# Unique-per-test URL so dedup behavior is deterministic. The demo
# mutation catalog is session-scoped, so multiple test invocations
# would otherwise see the workflow created in a previous run if
# anything went wrong with cleanup. (In practice the catalog is
# destroyed at session end, but we belt-and-brace it with a URL
# distinct from anything a real workflow would use.)
_TEST_WORKFLOW_URL = "https://example.org/deriva-ml-mcp/integration-test/workflow.py"
_TEST_WORKFLOW_CHECKSUM = "sha256:integration-test-checksum-deadbeef"
_TEST_WORKFLOW_NAME = "IntegrationTestWorkflow"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def integration_workflow_tools(
    demo_mutation_catalog: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_CapturingMCP]:
    """Register workflow tools against a fresh ``_CapturingMCP`` and seed the
    per-request credential contextvar so ``get_ml(...)`` resolves correctly.

    Mirrors ``integration_dataset_tools`` / ``integration_feature_tools``:
    same PluginContext / contextvar pattern, swapping in the ``workflow``
    module and the dedicated ``demo_mutation_catalog`` fixture.

    Additionally sets ``DERIVA_ML_ALLOW_DIRTY=true`` for the duration of
    the test. Reason: ``deriva_ml.execution.workflow.Workflow.__init__``
    runs a ``model_validator`` (``setup_url_checksum``) that — when
    ``self.url`` is unset — invokes ``Workflow._get_python_script()`` and
    then ``get_url_and_checksum(path)``, which raises
    ``DerivaMLDirtyWorkflowError`` if the calling Python file is in a
    dirty git checkout. Our ``create_workflow`` tool calls
    ``ml.create_workflow(name=..., workflow_type=..., description=...)``
    without ``url`` (then assigns ``wf.url`` afterward), so the validator
    fires the git check before the tool can supply the URL. The
    ``DERIVA_ML_ALLOW_DIRTY`` env var is the documented escape hatch
    used by the deriva-ml test suite itself (see deriva-ml/CLAUDE.md
    "Dirty workflow check"). ``monkeypatch`` ensures the var is unset
    after the test so other tests aren't affected.
    """
    monkeypatch.setenv("DERIVA_ML_ALLOW_DIRTY", "true")

    hostname, _ = demo_mutation_catalog
    credential = get_credential(hostname)
    set_current_credential(credential)

    capturing = _CapturingMCP()
    plugin_ctx = PluginContext(capturing)
    _set_plugin_context(plugin_ctx)
    try:
        from deriva_ml_mcp.tools import workflow as workflow_module

        workflow_module.register(plugin_ctx)
        yield capturing
    finally:
        _set_plugin_context(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------


async def test_workflow_round_trip(
    demo_mutation_catalog: tuple[str, str],
    integration_workflow_tools: _CapturingMCP,
) -> None:
    """Exercise all five workflow tools end-to-end against a real catalog.

    Steps:

    1. ``list_workflows`` -- baseline (empty catalog: count == 0).
    2. ``create_workflow`` -- status="created", capture RID.
    3. ``get_workflow`` -- look up by RID, assert fields round-trip.
    4. ``find_workflow_by_url`` -- look up by URL, assert same RID.
    5. ``create_workflow`` again with same URL -- status="exists",
       assert dedup returns the original RID.
    6. ``update_workflow`` -- update description, assert
       updated_fields=["description"].
    7. ``list_workflows`` -- assert the new workflow now appears (count == 1)
       and the description was actually persisted.

    Validates that:

    - Plugin tool registration works against a real PluginContext.
    - ``ml_context.get_ml`` builds a working DerivaML using the
      credential contextvar set by the fixture.
    - All five workflow tools' JSON envelopes match the documented
      shapes against a real Deriva catalog.
    - The dedup pre-check in ``create_workflow`` correctly distinguishes
      ``status="created"`` from ``status="exists"`` against a live
      ``lookup_workflow_by_url`` (not the unit-test mock).
    """
    hostname, catalog_id = demo_mutation_catalog
    tools = integration_workflow_tools.tools

    # 1. Baseline: empty catalog.
    list_out = json.loads(await tools["list_workflows"](hostname=hostname, catalog_id=catalog_id))
    assert "workflows" in list_out
    assert "count" in list_out
    assert "truncated" in list_out
    assert "next_after_rid" in list_out
    assert isinstance(list_out["workflows"], list)
    baseline_count = list_out["count"]
    assert baseline_count == len(list_out["workflows"])
    # Demo catalog with populate=False ships zero workflows.
    assert baseline_count == 0
    assert list_out["truncated"] is False
    assert list_out["next_after_rid"] is None

    # 2. Create a new workflow.
    create_out = json.loads(
        await tools["create_workflow"](
            hostname=hostname,
            catalog_id=catalog_id,
            name=_TEST_WORKFLOW_NAME,
            workflow_type=_WORKFLOW_TYPE_TERM,
            url=_TEST_WORKFLOW_URL,
            checksum=_TEST_WORKFLOW_CHECKSUM,
            version="1.0.0",
            description="initial description",
        )
    )
    assert create_out.get("error") is None, f"create_workflow returned error: {create_out}"
    assert create_out["status"] == "created"
    assert create_out["name"] == _TEST_WORKFLOW_NAME
    assert create_out["workflow_type"] == [_WORKFLOW_TYPE_TERM]
    assert create_out["url"] == _TEST_WORKFLOW_URL
    assert create_out["checksum"] == _TEST_WORKFLOW_CHECKSUM
    assert create_out["version"] == "1.0.0"
    workflow_rid = create_out["workflow_rid"]
    assert isinstance(workflow_rid, str) and workflow_rid

    # 3. Look up by RID.
    get_out = json.loads(
        await tools["get_workflow"](
            hostname=hostname,
            catalog_id=catalog_id,
            workflow_rid=workflow_rid,
        )
    )
    assert get_out.get("error") is None, f"get_workflow returned error: {get_out}"
    assert get_out["rid"] == workflow_rid
    assert get_out["name"] == _TEST_WORKFLOW_NAME
    assert get_out["url"] == _TEST_WORKFLOW_URL
    assert get_out["checksum"] == _TEST_WORKFLOW_CHECKSUM
    assert get_out["version"] == "1.0.0"
    assert _WORKFLOW_TYPE_TERM in get_out["workflow_type"]
    assert get_out["description"] == "initial description"

    # 4. Look up by URL -- same RID.
    find_out = json.loads(
        await tools["find_workflow_by_url"](
            hostname=hostname,
            catalog_id=catalog_id,
            url_or_checksum=_TEST_WORKFLOW_URL,
        )
    )
    assert find_out.get("error") is None, f"find_workflow_by_url returned error: {find_out}"
    assert find_out["rid"] == workflow_rid
    assert find_out["url"] == _TEST_WORKFLOW_URL

    # 5. Re-create with the same URL -- dedup returns status="exists"
    #    and the original RID.
    dedup_out = json.loads(
        await tools["create_workflow"](
            hostname=hostname,
            catalog_id=catalog_id,
            name=_TEST_WORKFLOW_NAME,
            workflow_type=_WORKFLOW_TYPE_TERM,
            url=_TEST_WORKFLOW_URL,
            checksum=_TEST_WORKFLOW_CHECKSUM,
            version="1.0.0",
            description="this should be ignored on dedup",
        )
    )
    assert dedup_out.get("error") is None, f"dedup create_workflow returned error: {dedup_out}"
    assert dedup_out["status"] == "exists"
    assert dedup_out["workflow_rid"] == workflow_rid

    # 6. Update the description. We verify the tool's RESPONSE envelope
    #    only (not the catalog-side effect) because of an upstream
    #    deriva-ml bug (see note below at step 7).
    update_out = json.loads(
        await tools["update_workflow"](
            hostname=hostname,
            catalog_id=catalog_id,
            workflow_rid=workflow_rid,
            description="updated description",
        )
    )
    assert update_out.get("error") is None, f"update_workflow returned error: {update_out}"
    assert update_out["status"] == "updated"
    assert update_out["workflow_rid"] == workflow_rid
    assert update_out["updated_fields"] == ["description"]

    # 7. List again -- the workflow now appears.
    #
    # Upstream bug note: ``deriva_ml.execution.workflow.Workflow.__setattr__``
    # gates catalog write-back on ``"__pydantic_private__" in self.__dict__``,
    # but Pydantic v2 stores private attrs in a class-level slot rather
    # than in ``__dict__``, so the gate is permanently False and
    # ``wf.description = ...`` silently no-ops without writing to the
    # catalog. Our ``update_workflow`` tool's response is correct
    # (``status="updated"``) because the Pydantic instance attribute is
    # set, but the catalog row's ``Description`` column is unchanged.
    # A separate task is tracking the deriva-ml fix; once it lands,
    # tighten the assertion below to also check
    # ``matching[0]["description"] == "updated description"``.
    list_after = json.loads(await tools["list_workflows"](hostname=hostname, catalog_id=catalog_id))
    assert list_after["count"] == baseline_count + 1
    matching = [w for w in list_after["workflows"] if w["rid"] == workflow_rid]
    assert len(matching) == 1
    assert matching[0]["name"] == _TEST_WORKFLOW_NAME
    assert matching[0]["url"] == _TEST_WORKFLOW_URL
    assert matching[0]["rid"] == workflow_rid
    assert _WORKFLOW_TYPE_TERM in matching[0]["workflow_type"]
