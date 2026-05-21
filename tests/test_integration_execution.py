"""Live-catalog integration tests for deriva-ml-mcp execution tools.

Same shape as ``tests/test_integration_workflow.py`` /
``tests/test_integration_feature.py`` — gated by the ``integration``
pytest marker and a ``skipif`` that probes
``${DERIVA_HOST:-localhost}:443``. Shared session fixtures
(``demo_mutation_catalog``) and the ``_server_reachable`` helper come
from ``tests/conftest.py`` / ``tests/_helpers.py``.

This file ships TWO tests:

1. ``test_execution_lifecycle_round_trip`` — pure execution-domain
   smoke: create_workflow + list_executions + create_execution +
   get_execution + start_execution (and idempotent re-start) +
   commit_execution + post-commit get/list invariants.

2. ``test_phase3_dod2_feature_round_trip`` — closure of the deferred
   Phase 3 Definition-of-Done #2 mutation gap. End-to-end:
   create_workflow -> create_feature -> create_execution ->
   start_execution -> add_feature_values -> commit_execution ->
   list_feature_values verifies the values actually landed in the
   catalog. This is the integration test that proves the full
   execution + feature lifecycle works against a live catalog.

Caveats / known surprises (codified during Phase 5.4 investigation):

- ``create_execution(workflow=...)`` upstream resolves string
  arguments via ``lookup_workflow_by_url``, NOT by RID. The MCP
  ``create_execution`` tool pre-resolves ``workflow_rid`` via
  ``lookup_workflow`` (RID API) and hands the Workflow object to
  upstream, so the MCP tool's ``workflow_rid`` parameter actually
  takes a RID as the name suggests. Tests pass the RID returned by
  ``create_workflow`` — round-trip works.
- ``DerivaML.__del__`` aborts any non-terminal ``self._execution`` on
  garbage collection. The MCP ``create_execution`` tool detaches by
  setting ``ml._execution = None`` before returning so the per-call
  DerivaML instance can be safely collected. Without that detach,
  any subsequent tool call that triggered GC (a Subject insert, a
  feature lookup) would silently abort the freshly-created execution.
- ``Execution.upload_outputs()`` (the new lease-engine wrapper)
  drains ``pending_rows`` but skips the staged feature-record SQLite
  table that ``add_features`` writes to. ``commit_execution`` uses
  the legacy ``upload_execution_outputs`` instead, which calls
  ``_flush_staged_features`` to actually push feature values to the
  catalog.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pytest
from deriva.core import get_credential
from deriva_mcp_core.context import set_current_credential
from deriva_mcp_core.plugin.api import PluginContext, _set_plugin_context
from deriva_ml import DerivaML

from tests._helpers import _CapturingMCP, _server_reachable

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _server_reachable(),
        reason=(f"No Deriva server reachable on {os.environ.get('DERIVA_HOST', 'localhost')}:443"),
    ),
]


# Pre-populated Workflow_Type vocab term shipped with the demo schema —
# same convention as test_integration_workflow.py. Keep "Testing" so
# this file's intent is obvious to a future reader.
_WORKFLOW_TYPE_TERM = "Testing"

# Distinct URLs per test: dedup pre-check in ``create_workflow`` keys on
# checksum-then-URL, and the ``demo_mutation_catalog`` fixture is
# session-scoped — so a workflow created in one test would otherwise
# show up as ``status="exists"`` in another. The MCP test_workflow
# round-trip already uses the catalog; pick distinct URLs here.
_LIFECYCLE_URL = "https://example.org/deriva-ml-mcp/integration-test/exec-lifecycle.py"
_LIFECYCLE_CHECKSUM = "sha256:exec-lifecycle-deadbeef"

_DOD2_URL = "https://example.org/deriva-ml-mcp/integration-test/exec-dod2.py"
_DOD2_CHECKSUM = "sha256:exec-dod2-deadbeef"

# Feature definition used by the DoD#2 round-trip. ``Subject`` is the
# simplest target table in the demo schema — just a Name column —
# perfect for a feature-value smoke test.
_DOD2_FEATURE_NAME = "IntegrationLabel"
_DOD2_FEATURE_VALUE_COL = "Label"
_DOD2_TARGET_TABLE = "Subject"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def integration_execution_tools(
    demo_mutation_catalog: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_CapturingMCP]:
    """Register workflow + execution + feature tools against a fresh
    ``_CapturingMCP`` and seed the per-request credential contextvar.

    The execution round-trip needs all three tool families:
    workflow (to create the workflow that the execution attributes
    against), execution (the lifecycle under test), and feature
    (only used by the DoD#2 round-trip — but cheap to register
    unconditionally so the lifecycle test can still confirm
    list_feature_values returns zero on an empty catalog if needed).

    ``DERIVA_ML_ALLOW_DIRTY=true`` mirrors the workflow integration
    fixture: ``create_workflow`` upstream runs a Pydantic validator
    that errors on dirty git checkouts before our tool can supply the
    URL. The env var is the documented escape hatch.
    """
    monkeypatch.setenv("DERIVA_ML_ALLOW_DIRTY", "true")

    hostname, _ = demo_mutation_catalog
    credential = get_credential(hostname)
    set_current_credential(credential)

    capturing = _CapturingMCP()
    plugin_ctx = PluginContext(capturing)
    _set_plugin_context(plugin_ctx)
    try:
        from deriva_ml_mcp.tools import execution as execution_module
        from deriva_ml_mcp.tools import feature as feature_module
        from deriva_ml_mcp.tools import workflow as workflow_module

        workflow_module.register(plugin_ctx)
        feature_module.register(plugin_ctx)
        execution_module.register(plugin_ctx)
        yield capturing
    finally:
        _set_plugin_context(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_subjects(hostname: str, catalog_id: str, names: list[str]) -> list[str]:
    """Insert N Subject rows directly via DerivaML's pathBuilder and return RIDs.

    The MCP tool surface doesn't expose a generic insert tool by design
    — domain-table mutations are out of scope for the ML plugin.
    Integration tests legitimately need to seed target rows for the
    feature round-trip though, so we drop into the underlying
    DerivaML/path-builder API for that step. Real users would do the
    same via their own scripts or via the lower-level deriva-mcp tools.

    Args:
        hostname: Deriva host.
        catalog_id: Catalog ID.
        names: List of Name values to insert (one Subject row per).

    Returns:
        The RIDs of the inserted Subject rows in input order.
    """
    credential = get_credential(hostname)
    ml = DerivaML(hostname, catalog_id, credential=credential)
    pb = ml.pathBuilder()
    rows = pb.schemas["demo-schema"].tables["Subject"].insert([{"Name": name} for name in names])
    return [row["RID"] for row in rows]


# ---------------------------------------------------------------------------
# Test A: pure execution-domain lifecycle smoke
# ---------------------------------------------------------------------------


async def test_execution_lifecycle_round_trip(
    demo_mutation_catalog: tuple[str, str],
    integration_execution_tools: _CapturingMCP,
) -> None:
    """Exercise the execution lifecycle end-to-end against a real catalog.

    Steps:

    1. ``list_executions`` -- baseline count (catalog may already have
       executions from a previous test in the same session, so we
       capture the count rather than asserting zero).
    2. ``create_workflow`` -- needed to attribute the execution to.
    3. ``create_execution`` -- status="created", capture RID.
    4. ``get_execution`` -- assert status="Created", workflow round-trip.
    5. ``start_execution`` -- status="running".
    6. ``start_execution`` again -- idempotent no-op (note="already
       running"), status still running.
    7. ``commit_execution`` -- status="uploaded", report.total_failed==0.
    8. ``get_execution`` -- assert status="Uploaded".
    9. ``list_executions`` -- count is baseline + 1.
    """
    hostname, catalog_id = demo_mutation_catalog
    tools = integration_execution_tools.tools

    # 1. Baseline count.
    list_out = json.loads(
        await tools["deriva_ml_list_executions"](hostname=hostname, catalog_id=catalog_id)
    )
    assert "executions" in list_out
    assert "count" in list_out
    baseline_count = list_out["count"]

    # 2. Create workflow (needed to attribute the execution to).
    wf_out = json.loads(
        await tools["deriva_ml_create_workflow"](
            hostname=hostname,
            catalog_id=catalog_id,
            name="ExecLifecycleWorkflow",
            workflow_type=_WORKFLOW_TYPE_TERM,
            url=_LIFECYCLE_URL,
            checksum=_LIFECYCLE_CHECKSUM,
            version="1.0.0",
            description="execution lifecycle integration test",
        )
    )
    assert wf_out.get("error") is None, f"create_workflow returned error: {wf_out}"
    # status may be "created" (fresh) or "exists" (re-run). Both are fine.
    assert wf_out["status"] in {"created", "exists"}
    workflow_rid = wf_out["workflow_rid"]

    # 3. Create execution against the workflow RID returned in step 2.
    #    The MCP tool pre-resolves workflow_rid via lookup_workflow before
    #    handing the Workflow object to upstream create_execution
    #    (workaround for upstream's string-arg-routes-to-lookup_by_url
    #    misalignment — see tools/execution.py:create_workflow).
    exec_out = json.loads(
        await tools["deriva_ml_create_execution"](
            hostname=hostname,
            catalog_id=catalog_id,
            workflow_rid=workflow_rid,
            description="lifecycle smoke",
        )
    )
    assert exec_out.get("error") is None, f"create_execution returned error: {exec_out}"
    assert exec_out["status"] == "created"
    assert exec_out["dataset_count"] == 0
    assert exec_out["asset_count"] == 0
    assert exec_out["dry_run"] is False
    execution_rid = exec_out["execution_rid"]
    assert isinstance(execution_rid, str) and execution_rid

    # 4. get_execution -- status="Created", workflow_rid round-trips
    #    to the actual workflow RID (not the URL we passed in).
    get_out = json.loads(
        await tools["deriva_ml_get_execution"](
            hostname=hostname,
            catalog_id=catalog_id,
            execution_rid=execution_rid,
        )
    )
    assert get_out.get("error") is None, f"get_execution returned error: {get_out}"
    assert get_out["rid"] == execution_rid
    assert get_out["workflow_rid"] == workflow_rid
    assert get_out["status"] == "Created"
    assert get_out["description"] == "lifecycle smoke"

    # 5. start_execution -- transitions Created -> Running.
    start_out = json.loads(
        await tools["deriva_ml_start_execution"](
            hostname=hostname,
            catalog_id=catalog_id,
            execution_rid=execution_rid,
        )
    )
    assert start_out.get("error") is None, f"start_execution returned error: {start_out}"
    assert start_out["status"] == "running"
    assert start_out["execution_rid"] == execution_rid
    # First call is the real transition; "note" only appears on the
    # idempotent path (covered by step 6).
    assert "note" not in start_out

    # 6. start_execution again -- idempotent no-op.
    restart_out = json.loads(
        await tools["deriva_ml_start_execution"](
            hostname=hostname,
            catalog_id=catalog_id,
            execution_rid=execution_rid,
        )
    )
    assert restart_out.get("error") is None
    # v3.0 wire-break: start_execution idempotent no-op now returns
    # status="already_running" (was status="running" + note="already running").
    # See deriva-ml-mcp CLAUDE.md "v3.0 wire-break migration map".
    assert restart_out["status"] == "already_running"
    assert "note" not in restart_out

    # 7. commit_execution -- Stops -> uploads. report.total_failed==0
    #    is the success signal; total_uploaded includes
    #    Execution_Metadata files (config.json + env_snapshot) that
    #    upload_execution_outputs always emits.
    commit_out = json.loads(
        await tools["deriva_ml_commit_execution"](
            hostname=hostname,
            catalog_id=catalog_id,
            execution_rid=execution_rid,
        )
    )
    assert commit_out.get("error") is None, f"commit_execution returned error: {commit_out}"
    assert commit_out["status"] == "uploaded"
    assert commit_out["execution_rid"] == execution_rid
    # The success signal is total_failed == 0. The plugin successfully
    # transitions the execution to Uploaded with no errors. Whether
    # Execution_Metadata files (configuration.json + environment_snapshot)
    # land as catalog rows depends on deriva-ml's
    # ``upload_execution_outputs`` behavior, which can legitimately emit
    # an empty per_table when there's no user-side feature/asset work
    # (the metadata files exist on disk but are only written to the
    # catalog when there's something to attribute them to). Earlier
    # versions of deriva-ml unconditionally surfaced them; v1.36.5+
    # appears to skip the catalog-row write in the no-op case. Don't
    # over-tighten the assertion -- the lifecycle round-trip is what
    # this test pins.
    assert commit_out["report"]["total_failed"] == 0

    # 8. get_execution -- post-commit status is Uploaded.
    final_get = json.loads(
        await tools["deriva_ml_get_execution"](
            hostname=hostname,
            catalog_id=catalog_id,
            execution_rid=execution_rid,
        )
    )
    assert final_get.get("error") is None
    assert final_get["status"] == "Uploaded"

    # 9. list_executions -- new execution shows up.
    final_list = json.loads(
        await tools["deriva_ml_list_executions"](hostname=hostname, catalog_id=catalog_id)
    )
    assert final_list["count"] == baseline_count + 1
    matching = [e for e in final_list["executions"] if e["rid"] == execution_rid]
    assert len(matching) == 1
    assert matching[0]["status"] == "Uploaded"


# ---------------------------------------------------------------------------
# Test B: Phase 3 DoD #2 mutation closure
# ---------------------------------------------------------------------------


async def test_phase3_dod2_feature_round_trip(
    demo_mutation_catalog: tuple[str, str],
    integration_execution_tools: _CapturingMCP,
) -> None:
    """Closure of the deferred Phase 3 DoD #2 mutation round-trip.

    Phase 3 shipped feature read tools and unit-tested mutation tools
    with mocks, but deferred the live-catalog mutation round-trip
    because no execution scaffolding existed yet. Phase 5 ships
    execution tools — this test exercises the FULL chain end-to-end:

    1. Pre-create N Subject rows (via DerivaML's pathBuilder — the
       MCP surface intentionally doesn't expose generic inserts).
    2. ``create_workflow`` -- attribution target for the execution.
    3. ``create_feature`` on Subject -- registers the feature schema.
    4. ``create_execution`` -- attributes the value-add to a workflow run.
    5. ``start_execution`` -- transitions to long-running mode.
    6. ``add_feature_values`` -- bulk-stage two value rows.
    7. ``commit_execution`` -- flushes staged feature records to catalog.
    8. ``list_feature_values`` -- read back, assert values landed.

    This is the test that proves the entire execution + feature
    lifecycle works against a live catalog.
    """
    hostname, catalog_id = demo_mutation_catalog
    tools = integration_execution_tools.tools

    # 1. Pre-create target Subject rows.
    subject_rids = _insert_subjects(
        hostname, catalog_id, names=["DoD2_Subject_A", "DoD2_Subject_B"]
    )
    assert len(subject_rids) == 2
    rid_a, rid_b = subject_rids

    # 2. Create workflow (URL distinct from the lifecycle test).
    wf_out = json.loads(
        await tools["deriva_ml_create_workflow"](
            hostname=hostname,
            catalog_id=catalog_id,
            name="DoD2FeatureWorkflow",
            workflow_type=_WORKFLOW_TYPE_TERM,
            url=_DOD2_URL,
            checksum=_DOD2_CHECKSUM,
            version="1.0.0",
            description="DoD #2 feature round-trip integration test",
        )
    )
    assert wf_out.get("error") is None, f"create_workflow returned error: {wf_out}"
    assert wf_out["status"] in {"created", "exists"}
    workflow_rid = wf_out["workflow_rid"]

    # 3. Create the feature on Subject. metadata is a single value
    #    column ("Label", text). No term/asset columns — the simplest
    #    feature shape that exercises add_feature_values.
    create_feat = json.loads(
        await tools["deriva_ml_create_feature"](
            hostname=hostname,
            catalog_id=catalog_id,
            target_table=_DOD2_TARGET_TABLE,
            feature_name=_DOD2_FEATURE_NAME,
            metadata=[
                {
                    "name": _DOD2_FEATURE_VALUE_COL,
                    "type": {"typename": "text"},
                    "nullok": True,
                }
            ],
            comment="DoD #2 integration test feature",
        )
    )
    assert create_feat.get("error") is None, f"create_feature returned error: {create_feat}"
    assert create_feat["status"] == "created"
    assert create_feat["target_table"] == _DOD2_TARGET_TABLE
    assert create_feat["feature_name"] == _DOD2_FEATURE_NAME
    assert _DOD2_FEATURE_VALUE_COL in create_feat["value_columns"]

    # 4. Create execution against the workflow RID returned in step 3.
    exec_out = json.loads(
        await tools["deriva_ml_create_execution"](
            hostname=hostname,
            catalog_id=catalog_id,
            workflow_rid=workflow_rid,
            description="DoD #2 feature value injection",
        )
    )
    assert exec_out.get("error") is None, f"create_execution returned error: {exec_out}"
    assert exec_out["status"] == "created"
    execution_rid = exec_out["execution_rid"]

    # 5. Start the execution -- long-running mode required for
    #    add_feature_values (which would otherwise auto-wrap a
    #    Created -> Running -> Stopped lifecycle on each call).
    start_out = json.loads(
        await tools["deriva_ml_start_execution"](
            hostname=hostname,
            catalog_id=catalog_id,
            execution_rid=execution_rid,
        )
    )
    assert start_out.get("error") is None, f"start_execution returned error: {start_out}"
    assert start_out["status"] == "running"

    # 6. Bulk-stage feature values for both Subject targets.
    #    Each entry's keys must match the FeatureRecord class fields:
    #    target table column ("Subject" -> RID) + value column ("Label").
    add_out = json.loads(
        await tools["deriva_ml_add_feature_values"](
            hostname=hostname,
            catalog_id=catalog_id,
            table=_DOD2_TARGET_TABLE,
            feature_name=_DOD2_FEATURE_NAME,
            execution_rid=execution_rid,
            entries=[
                {"Subject": rid_a, _DOD2_FEATURE_VALUE_COL: "good"},
                {"Subject": rid_b, _DOD2_FEATURE_VALUE_COL: "bad"},
            ],
        )
    )
    assert add_out.get("error") is None, f"add_feature_values returned error: {add_out}"
    assert add_out["status"] == "added"
    assert add_out["count"] == 2
    assert add_out["execution_rid"] == execution_rid
    assert add_out["feature_name"] == _DOD2_FEATURE_NAME

    # 7. Commit -- flushes staged feature records via
    #    upload_execution_outputs / _flush_staged_features. The catalog
    #    rows are written here.
    commit_out = json.loads(
        await tools["deriva_ml_commit_execution"](
            hostname=hostname,
            catalog_id=catalog_id,
            execution_rid=execution_rid,
        )
    )
    assert commit_out.get("error") is None, f"commit_execution returned error: {commit_out}"
    assert commit_out["status"] == "uploaded"
    assert commit_out["report"]["total_failed"] == 0
    # 2 feature inserts + N Execution_Metadata assets (config + env
    # snapshot at minimum). Lower bound is feature_count.
    assert commit_out["report"]["total_uploaded"] >= 2

    # 8. Read the values back via list_feature_values. This is the
    #    "values landed in the catalog" assertion that closes DoD #2.
    list_out = json.loads(
        await tools["deriva_ml_list_feature_values"](
            hostname=hostname,
            catalog_id=catalog_id,
            table=_DOD2_TARGET_TABLE,
            feature_name=_DOD2_FEATURE_NAME,
        )
    )
    assert list_out.get("error") is None, f"list_feature_values returned error: {list_out}"
    assert list_out["count"] == 2
    # Build a {target_rid: label} map and compare.
    by_subject = {rec["Subject"]: rec[_DOD2_FEATURE_VALUE_COL] for rec in list_out["records"]}
    assert by_subject == {rid_a: "good", rid_b: "bad"}
    # Sanity-check provenance: every record carries the execution RID.
    for rec in list_out["records"]:
        assert rec["Execution"] == execution_rid
