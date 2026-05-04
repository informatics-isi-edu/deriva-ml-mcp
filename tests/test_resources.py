"""Unit tests for resources/ml.py (13 ML-domain MCP resources, v3.3).

Resources are read-only and emit no audit on success or failure. The
fixture wires ``mock_ml`` into the resources/ml.py registration call so
each test can stub the deriva-ml call surface independently.

Coverage target: at least 26 tests (2 per resource * 13 resources). List
resources also get a ``truncated`` test; detail resources get a 404 path.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def resource_ctx(ctx, mock_ml):
    """Register resources/ml.py with mock_ml as the DerivaML stand-in.

    Patches at the use-site (``deriva_ml_mcp.resources.ml.get_ml``) and
    imports the module *inside* the patch block so registration sees the
    mock.
    """
    with patch("deriva_ml_mcp.resources.ml.get_ml", return_value=mock_ml):
        from deriva_ml_mcp.resources import ml as ml_resources

        ml_resources.register(ctx)
        yield ctx


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


_DATASETS_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/datasets"
_DATASET_DETAIL_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}"
_DATASET_SPEC_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/spec"
_DATASET_MEMBERS_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/members"
_WORKFLOWS_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/workflows"
_WORKFLOW_DETAIL_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/workflow/{workflow_rid}"
_EXECUTIONS_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/executions"
_EXECUTION_DETAIL_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/execution/{execution_rid}"
_LINEAGE_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/lineage/{rid}"
_FEATURES_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/features/{table_name}"
_ASSET_TABLES_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/asset-tables"
_ASSET_DETAIL_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/asset/{asset_rid}"
_REGISTRIES_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/registries"


def _make_dataset_mock(
    rid: str,
    description: str = "",
    dataset_types: list[str] | None = None,
    current_version: str = "0.1.0",
    chaise_url: str = "https://example.org/chaise",
) -> MagicMock:
    ds = MagicMock()
    ds.dataset_rid = rid
    ds.description = description
    ds.dataset_types = dataset_types or []
    ds.current_version = current_version
    ds.get_chaise_url.return_value = chaise_url
    return ds


def _make_workflow_mock(
    rid: str = "1-WF",
    name: str = "MyPipeline",
    workflow_type: list[str] | None = None,
    url: str = "https://github.com/example/repo",
    checksum: str = "abc123",
    version: str = "1.0.0",
    description: str = "",
) -> MagicMock:
    wf = MagicMock()
    wf.rid = rid
    wf.name = name
    wf.workflow_type = workflow_type or ["Model_Training"]
    wf.url = url
    wf.checksum = checksum
    wf.version = version
    wf.description = description
    return wf


def _make_execution_record_mock(
    rid: str = "1-EXEC",
    workflow_rid: str = "1-WF",
    description: str = "",
) -> MagicMock:
    from deriva_ml.execution.execution import ExecutionStatus

    record = MagicMock()
    record.execution_rid = rid
    record.workflow_rid = workflow_rid
    workflow = MagicMock()
    workflow.rid = workflow_rid
    record.workflow = workflow
    record.status = ExecutionStatus.Stopped
    record.description = description
    record.start_time = datetime(2026, 1, 1, 12, 0, 0)
    record.stop_time = datetime(2026, 1, 1, 12, 30, 0)
    record.duration = "0:30:00"
    record.list_input_datasets.return_value = []
    record.list_assets.return_value = []
    return record


def _make_feature_mock(
    feature_name: str,
    target_table: str = "Image",
    feature_table: str = "Execution_Image_Quality",
) -> MagicMock:
    feature = MagicMock()
    feature.feature_name = feature_name
    feature.target_table = MagicMock()
    feature.target_table.name = target_table
    feature.feature_table = MagicMock()
    feature.feature_table.name = feature_table
    feature.term_columns = []
    feature.asset_columns = []
    feature.value_columns = []
    return feature


def _make_vocab_term_mock(name: str, description: str = "") -> MagicMock:
    term = MagicMock()
    term.name = name
    term.description = description
    term.synonyms = []
    term.rid = f"VRID-{name}"
    return term


# ---------------------------------------------------------------------------
# Resource registration smoke test
# ---------------------------------------------------------------------------


def test_register_adds_thirteen_resources(resource_ctx, capturing_mcp):
    """resources/ml.py.register() must register exactly 13 URIs (12 prior + dataset/{rid}/spec, v3.3)."""
    expected = {
        "deriva://catalog/{hostname}/{catalog_id}/ml/datasets",
        "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/spec",
        "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/members",
        "deriva://catalog/{hostname}/{catalog_id}/ml/workflows",
        "deriva://catalog/{hostname}/{catalog_id}/ml/workflow/{workflow_rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/executions",
        "deriva://catalog/{hostname}/{catalog_id}/ml/execution/{execution_rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/lineage/{rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/features/{table_name}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/asset-tables",
        "deriva://catalog/{hostname}/{catalog_id}/ml/asset/{asset_rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/registries",
    }
    actual = set(capturing_mcp.resources.keys())
    assert actual == expected, (
        f"missing: {sorted(expected - actual)}; extra: {sorted(actual - expected)}"
    )


# ---------------------------------------------------------------------------
# ml/datasets
# ---------------------------------------------------------------------------


async def test_ml_datasets_success(resource_ctx, capturing_mcp, mock_ml):
    mock_ml.find_datasets.return_value = [
        _make_dataset_mock("1-AAAA", "first", ["Training"], "1.0.0"),
        _make_dataset_mock("1-BBBB", "second", ["Testing"], "1.1.0"),
    ]
    out = json.loads(await capturing_mcp.resources[_DATASETS_URI](hostname="h", catalog_id="1"))
    assert out["count"] == 2
    assert out["truncated"] is False
    assert {d["rid"] for d in out["datasets"]} == {"1-AAAA", "1-BBBB"}


async def test_ml_datasets_truncated_at_max_limit(resource_ctx, capturing_mcp, mock_ml):
    """1001 datasets truncates to 1000 and sets ``truncated=True``."""
    mock_ml.find_datasets.return_value = [_make_dataset_mock(f"1-{i:04d}") for i in range(1001)]
    out = json.loads(await capturing_mcp.resources[_DATASETS_URI](hostname="h", catalog_id="1"))
    assert out["count"] == 1000
    assert out["truncated"] is True
    assert out["next_after_rid"] == "1-0999"


async def test_ml_datasets_error_path_is_silent(resource_ctx, capturing_mcp, mock_ml):
    """Errors return ``{"error": ...}`` and emit NO audit (resource is read-only)."""
    mock_ml.find_datasets.side_effect = RuntimeError("kaboom")
    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(await capturing_mcp.resources[_DATASETS_URI](hostname="h", catalog_id="1"))
    assert out == {"error": "kaboom"}
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# ml/dataset/{dataset_rid}
# ---------------------------------------------------------------------------


async def test_ml_dataset_detail_includes_version_history(resource_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", "desc", ["Training"], "1.0.0")
    history_entry = MagicMock()
    history_entry.dataset_version = "1.0.0"
    history_entry.snapshot = "snap1"
    history_entry.description = "v1"
    history_entry.execution_rid = "EXEC-1"
    ds.dataset_history.return_value = [history_entry]
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.resources[_DATASET_DETAIL_URI](
            hostname="h", catalog_id="1", dataset_rid="1-AAAA"
        )
    )
    assert out["rid"] == "1-AAAA"
    assert out["chaise_url"] == "https://example.org/chaise"
    assert out["version_history"] == [
        {
            "version": "1.0.0",
            "snapshot": "snap1",
            "description": "v1",
            "execution_rid": "EXEC-1",
        }
    ]


async def test_ml_dataset_detail_not_found(resource_ctx, capturing_mcp, mock_ml):
    """A missing RID returns ``{"error": ...}`` with no audit."""
    mock_ml.lookup_dataset.side_effect = RuntimeError("Dataset not found")
    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(
            await capturing_mcp.resources[_DATASET_DETAIL_URI](
                hostname="h", catalog_id="1", dataset_rid="missing"
            )
        )
    assert out == {"error": "Dataset not found"}
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# ml/dataset/{dataset_rid}/members
# ---------------------------------------------------------------------------


async def test_ml_dataset_members_success(resource_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.list_dataset_members.return_value = {
        "Image": [{"RID": "i1"}, {"RID": "i2"}, {"RID": "i3"}],
        "Subject": [{"RID": "s1"}],
    }
    mock_ml.lookup_dataset.return_value = ds
    out = json.loads(
        await capturing_mcp.resources[_DATASET_MEMBERS_URI](
            hostname="h", catalog_id="1", dataset_rid="1-AAAA"
        )
    )
    assert out["dataset_rid"] == "1-AAAA"
    assert out["summary"] == {"Image": 3, "Subject": 1}
    assert out["total_count"] == 4
    assert out["truncated"] is False
    assert sorted(out["tables"]) == ["Image", "Subject"]
    rids = {m["rid"] for m in out["members"]}
    assert rids == {"i1", "i2", "i3", "s1"}


async def test_ml_dataset_members_truncated(resource_ctx, capturing_mcp, mock_ml):
    """When >1000 members across tables, the flattened list is capped."""
    ds = _make_dataset_mock("1-AAAA")
    ds.list_dataset_members.return_value = {
        "Image": [{"RID": f"i{i}"} for i in range(1500)],
    }
    mock_ml.lookup_dataset.return_value = ds
    out = json.loads(
        await capturing_mcp.resources[_DATASET_MEMBERS_URI](
            hostname="h", catalog_id="1", dataset_rid="1-AAAA"
        )
    )
    assert out["total_count"] == 1500
    assert len(out["members"]) == 1000
    assert out["truncated"] is True


async def test_ml_dataset_members_error_path(resource_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_dataset.side_effect = RuntimeError("nope")
    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(
            await capturing_mcp.resources[_DATASET_MEMBERS_URI](
                hostname="h", catalog_id="1", dataset_rid="bad"
            )
        )
    assert out == {"error": "nope"}
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# ml/dataset/{rid}/spec
# ---------------------------------------------------------------------------


async def test_ml_dataset_spec_uses_current_version(resource_ctx, capturing_mcp, mock_ml):
    """Resource form omits the ``version`` parameter and always uses
    the dataset's current version (with a warning recommending an
    explicit pin). The shared helper is what the tool uses too, so a
    drift would break both."""
    ds = MagicMock()
    ds.current_version = "1.2.0"
    ds.description = "Training set"
    ds.dataset_types = ["Training"]
    mock_ml.lookup_dataset.return_value = ds

    out = json.loads(
        await capturing_mcp.resources[_DATASET_SPEC_URI](
            hostname="h", catalog_id="1", dataset_rid="1-DSAA"
        )
    )
    assert out["dataset_rid"] == "1-DSAA"
    assert out["version"] == "1.2.0"
    assert out["spec"] == 'DatasetSpecConfig(rid="1-DSAA", version="1.2.0")'
    assert out["dataset_types"] == ["Training"]
    # Resource always uses current version, so warning is always set.
    assert out["warning"] is not None
    assert "version not specified" in out["warning"]


async def test_ml_dataset_spec_error_path_is_silent(resource_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_dataset.side_effect = RuntimeError("Dataset not found")
    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(
            await capturing_mcp.resources[_DATASET_SPEC_URI](
                hostname="h", catalog_id="1", dataset_rid="missing"
            )
        )
    assert out == {"error": "Dataset not found"}
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# ml/workflows
# ---------------------------------------------------------------------------


async def test_ml_workflows_success(resource_ctx, capturing_mcp, mock_ml):
    mock_ml.find_workflows.return_value = [
        _make_workflow_mock(rid="1-AAA", name="A"),
        _make_workflow_mock(rid="1-BBB", name="B"),
    ]
    out = json.loads(await capturing_mcp.resources[_WORKFLOWS_URI](hostname="h", catalog_id="1"))
    assert out["count"] == 2
    assert {w["rid"] for w in out["workflows"]} == {"1-AAA", "1-BBB"}
    assert out["truncated"] is False


async def test_ml_workflows_error_path(resource_ctx, capturing_mcp, mock_ml):
    mock_ml.find_workflows.side_effect = RuntimeError("kaboom")
    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(
            await capturing_mcp.resources[_WORKFLOWS_URI](hostname="h", catalog_id="1")
        )
    assert out == {"error": "kaboom"}
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# ml/workflow/{workflow_rid}
# ---------------------------------------------------------------------------


async def test_ml_workflow_detail_success(resource_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_workflow.return_value = _make_workflow_mock(rid="1-WF", name="Pipe")
    out = json.loads(
        await capturing_mcp.resources[_WORKFLOW_DETAIL_URI](
            hostname="h", catalog_id="1", workflow_rid="1-WF"
        )
    )
    assert out["rid"] == "1-WF"
    assert out["name"] == "Pipe"
    assert out["url"] == "https://github.com/example/repo"
    mock_ml.lookup_workflow.assert_called_once_with("1-WF")


async def test_ml_workflow_detail_not_found(resource_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_workflow.side_effect = RuntimeError("Workflow not found")
    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(
            await capturing_mcp.resources[_WORKFLOW_DETAIL_URI](
                hostname="h", catalog_id="1", workflow_rid="missing"
            )
        )
    assert out == {"error": "Workflow not found"}
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# ml/executions
# ---------------------------------------------------------------------------


async def test_ml_executions_success(resource_ctx, capturing_mcp, mock_ml):
    mock_ml.find_executions.return_value = [
        _make_execution_record_mock(rid="1-AAA"),
        _make_execution_record_mock(rid="1-BBB"),
    ]
    out = json.loads(await capturing_mcp.resources[_EXECUTIONS_URI](hostname="h", catalog_id="1"))
    assert out["count"] == 2
    assert {e["rid"] for e in out["executions"]} == {"1-AAA", "1-BBB"}
    assert out["truncated"] is False


async def test_ml_executions_error_path(resource_ctx, capturing_mcp, mock_ml):
    mock_ml.find_executions.side_effect = RuntimeError("boom")
    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(
            await capturing_mcp.resources[_EXECUTIONS_URI](hostname="h", catalog_id="1")
        )
    assert out == {"error": "boom"}
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# ml/execution/{execution_rid}
# ---------------------------------------------------------------------------


async def test_ml_execution_detail_includes_inputs_outputs(resource_ctx, capturing_mcp, mock_ml):
    """Detail bundles inputs and outputs; experiment key absent when no experiment."""
    record = _make_execution_record_mock(rid="1-EXEC", workflow_rid="1-WF")

    # Stub one input dataset.
    input_ds = MagicMock()
    input_ds.dataset_rid = "1-DS"
    input_ds.current_version = "1.0.0"
    record.list_input_datasets.return_value = [input_ds]

    # Stub one input asset and one output asset.
    input_asset = MagicMock()
    input_asset.asset_rid = "1-IN"
    input_asset.filename = "in.txt"
    output_asset = MagicMock()
    output_asset.asset_rid = "1-OUT"
    output_asset.filename = "out.txt"

    def _list_assets(asset_role: str | None = None):
        if asset_role == "Input":
            return [input_asset]
        if asset_role == "Output":
            return [output_asset]
        return [input_asset, output_asset]

    record.list_assets.side_effect = _list_assets
    mock_ml.lookup_execution.return_value = record
    # Most executions are not experiments -- lookup_experiment raises.
    mock_ml.lookup_experiment.side_effect = RuntimeError("Execution has no Experiment")

    out = json.loads(
        await capturing_mcp.resources[_EXECUTION_DETAIL_URI](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    )
    assert out["rid"] == "1-EXEC"
    assert out["status"] == "Stopped"
    assert out["inputs"] == {
        "datasets": [{"rid": "1-DS", "version": "1.0.0"}],
        "assets": [{"rid": "1-IN", "filename": "in.txt"}],
    }
    assert out["outputs"] == {
        "assets": [{"rid": "1-OUT", "filename": "out.txt"}],
    }
    # No metadata bucket (omit-when-no-upstream-API; see TODO in
    # _get_execution_detail_impl).
    assert "metadata" not in out
    # v2.0 wire change: experiment is always present, set to None when
    # the execution is not a Hydra-driven Experiment. v1.x omitted the
    # key entirely; the typed contract from PR 2 (#6) now surfaces it
    # explicitly so consumers can introspect without a KeyError.
    assert out["experiment"] is None


async def test_ml_execution_detail_includes_experiment_when_present(
    resource_ctx, capturing_mcp, mock_ml
):
    """When the execution IS an Experiment, the key surfaces name + config."""
    record = _make_execution_record_mock(rid="1-EXP", workflow_rid="1-WF")
    record.list_input_datasets.return_value = []
    record.list_assets.return_value = []
    mock_ml.lookup_execution.return_value = record

    # Stub the Experiment shape: name + config_choices + model_config (cheap
    # accessors per Experiment.__init__; no hydra_config download).
    experiment = MagicMock()
    experiment.name = "lr-sweep-trial-3"
    experiment.config_choices = {"model": "resnet50", "optimizer": "adam"}
    experiment.model_config = {"lr": 0.001, "batch_size": 32}
    mock_ml.lookup_experiment.return_value = experiment

    out = json.loads(
        await capturing_mcp.resources[_EXECUTION_DETAIL_URI](
            hostname="h", catalog_id="1", execution_rid="1-EXP"
        )
    )
    assert "experiment" in out
    assert out["experiment"]["name"] == "lr-sweep-trial-3"
    assert out["experiment"]["config_choices"] == {"model": "resnet50", "optimizer": "adam"}
    assert out["experiment"]["model_config"] == {"lr": 0.001, "batch_size": 32}
    # We deliberately do NOT surface the full hydra_config dict (can be 10-100 KB);
    # the resource exposes the cheap accessors only.
    assert "hydra_config" not in out["experiment"]


async def test_ml_execution_detail_not_found(resource_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_execution.side_effect = RuntimeError("Execution not found")
    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(
            await capturing_mcp.resources[_EXECUTION_DETAIL_URI](
                hostname="h", catalog_id="1", execution_rid="missing"
            )
        )
    assert out == {"error": "Execution not found"}
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# ml/lineage/{rid}
# ---------------------------------------------------------------------------


async def test_ml_lineage_success(resource_ctx, capturing_mcp, mock_ml):
    """The resource returns the LineageResult Pydantic model serialized
    to JSON. Underlying call is ``ml.lookup_lineage(rid, depth=None,
    max_executions=500)``; resource form has no overrides for those
    parameters."""
    from deriva_ml.execution.lineage import (
        ExecutionSummary,
        LineageNode,
        LineageResult,
        RootDescriptor,
    )

    payload = LineageResult(
        root=RootDescriptor(
            rid="2-PRED",
            type="Asset",
            description="predictions.csv",
            producing_execution=ExecutionSummary(
                rid="1-EXEC",
                description="Train ResNet-50",
                workflow=None,
                status="Completed",
            ),
        ),
        lineage=LineageNode(
            execution=ExecutionSummary(
                rid="1-EXEC",
                description="Train ResNet-50",
                workflow=None,
                status="Completed",
            ),
            consumed_datasets=[],
            consumed_assets=[],
            parents=[],
        ),
        executions_visited=1,
        walked_complete=True,
        cycle_detected=False,
        depth_capped=False,
    )
    mock_ml.lookup_lineage.return_value = payload

    out = json.loads(
        await capturing_mcp.resources[_LINEAGE_URI](
            hostname="h", catalog_id="1", rid="2-PRED"
        )
    )
    assert out["root"]["rid"] == "2-PRED"
    assert out["root"]["type"] == "Asset"
    assert out["walked_complete"] is True
    assert out["cycle_detected"] is False
    # Resource always uses unbounded depth + default max_executions=500.
    mock_ml.lookup_lineage.assert_called_once_with(
        "2-PRED", depth=None, max_executions=500
    )


async def test_ml_lineage_error_path_is_silent(resource_ctx, capturing_mcp, mock_ml):
    """A workflow RID (or any unclassifiable) raises ``DerivaMLException``;
    the resource wraps it as ``{"error": ...}`` and emits no audit row
    (resources are read-only and silent on failure)."""
    mock_ml.lookup_lineage.side_effect = RuntimeError(
        "Workflow RIDs are not lineage-shaped"
    )
    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(
            await capturing_mcp.resources[_LINEAGE_URI](
                hostname="h", catalog_id="1", rid="1-WF"
            )
        )
    assert out == {"error": "Workflow RIDs are not lineage-shaped"}
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# ml/features/{table_name}
# ---------------------------------------------------------------------------


async def test_ml_features_for_table_success(resource_ctx, capturing_mcp, mock_ml):
    mock_ml.find_features.return_value = [
        _make_feature_mock(
            "Quality", target_table="Image", feature_table="Execution_Image_Quality"
        ),
        _make_feature_mock("Tag", target_table="Image", feature_table="Execution_Image_Tag"),
    ]
    out = json.loads(
        await capturing_mcp.resources[_FEATURES_URI](
            hostname="h", catalog_id="1", table_name="Image"
        )
    )
    assert out["count"] == 2
    assert {f["feature_name"] for f in out["features"]} == {"Quality", "Tag"}
    mock_ml.find_features.assert_called_once_with(table="Image")


async def test_ml_features_for_table_error_path(resource_ctx, capturing_mcp, mock_ml):
    mock_ml.find_features.side_effect = RuntimeError("nope")
    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(
            await capturing_mcp.resources[_FEATURES_URI](
                hostname="h", catalog_id="1", table_name="Image"
            )
        )
    assert out == {"error": "nope"}
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# ml/registries
# ---------------------------------------------------------------------------


async def test_ml_registries_bundles_four_vocabularies(resource_ctx, capturing_mcp, mock_ml):
    """Registries snapshot returns the four ML vocabularies as parallel lists."""

    def _terms_for(table_name):
        if table_name == "Dataset_Type":
            return [_make_vocab_term_mock("Training"), _make_vocab_term_mock("Testing")]
        if table_name == "Workflow_Type":
            return [_make_vocab_term_mock("Model_Training")]
        if table_name == "Asset_Type":
            return [_make_vocab_term_mock("Image")]
        if table_name == "Execution_Status":
            return [_make_vocab_term_mock("Created"), _make_vocab_term_mock("Running")]
        return []

    mock_ml.list_vocabulary_terms.side_effect = _terms_for

    out = json.loads(await capturing_mcp.resources[_REGISTRIES_URI](hostname="h", catalog_id="1"))
    assert {t["name"] for t in out["dataset_types"]} == {"Training", "Testing"}
    assert {t["name"] for t in out["workflow_types"]} == {"Model_Training"}
    assert {t["name"] for t in out["asset_types"]} == {"Image"}
    assert {t["name"] for t in out["execution_statuses"]} == {"Created", "Running"}


async def test_ml_registries_missing_vocab_yields_empty_list(resource_ctx, capturing_mcp, mock_ml):
    """A vocab table that raises (e.g. doesn't exist) maps to ``[]`` -- best-effort."""

    def _terms_for(table_name):
        if table_name == "Dataset_Type":
            return [_make_vocab_term_mock("Training")]
        raise RuntimeError(f"table {table_name} not found")

    mock_ml.list_vocabulary_terms.side_effect = _terms_for

    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(
            await capturing_mcp.resources[_REGISTRIES_URI](hostname="h", catalog_id="1")
        )
    # The successful vocab still lands; the failed ones are silently empty.
    # Compact shape: name + rid only (description/synonyms deliberately
    # omitted to keep the snapshot under ~1 KB; see _vocab_terms docstring).
    assert out["dataset_types"] == [
        {
            "name": "Training",
            "rid": "VRID-Training",
        }
    ]
    # Verify description and synonyms are NOT in the payload (token economy:
    # this resource is read on every "what types are available" question;
    # a 12 KB payload would dominate context for routine vocab checks).
    term = out["dataset_types"][0]
    assert "description" not in term
    assert "synonyms" not in term
    assert out["workflow_types"] == []
    assert out["asset_types"] == []
    assert out["execution_statuses"] == []
    # No audit on the per-vocab failures (resource is read-only).
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# ml/asset-tables (v1.2)
# ---------------------------------------------------------------------------


def _make_asset_table_mock(name: str, schema: str = "demo-schema") -> MagicMock:
    t = MagicMock()
    t.name = name
    t.schema.name = schema
    return t


async def test_ml_asset_tables_smoke(resource_ctx, capturing_mcp, mock_ml):
    """asset-tables resource returns the same shape as the tool."""
    mock_ml.list_asset_tables.return_value = [
        _make_asset_table_mock("Image", "demo-schema"),
        _make_asset_table_mock("Execution_Metadata", "deriva-ml"),
    ]
    out = json.loads(await capturing_mcp.resources[_ASSET_TABLES_URI](hostname="h", catalog_id="1"))
    assert out["count"] == 2
    assert {t["name"] for t in out["asset_tables"]} == {"Image", "Execution_Metadata"}


# ---------------------------------------------------------------------------
# ml/asset/{rid} (v1.2)
# ---------------------------------------------------------------------------


def _make_asset_detail_mock(
    asset_rid: str = "1-AAAA",
    description: str = "MRI scan",
    asset_types: list[str] | None = None,
    executions: list | None = None,
) -> MagicMock:
    a = MagicMock()
    a.asset_rid = asset_rid
    a.filename = "scan.png"
    a.length = 12345
    a.md5 = "abc"
    a.url = "/hatrac/scan.png"
    a.description = description
    a.asset_table = "Image"
    a.asset_types = list(asset_types) if asset_types else []
    a.list_executions.return_value = list(executions) if executions else []
    return a


async def test_ml_asset_detail_bundles_executions(resource_ctx, capturing_mcp, mock_ml):
    """asset/{rid} resource returns the bundled detail with executions."""
    exec_record = MagicMock()
    exec_record.execution_rid = "1-EXEC"
    exec_record.asset_role = "Output"
    asset = _make_asset_detail_mock(asset_types=["Training_Data"], executions=[exec_record])
    mock_ml.lookup_asset.return_value = asset

    out = json.loads(
        await capturing_mcp.resources[_ASSET_DETAIL_URI](
            hostname="h", catalog_id="1", asset_rid="1-AAAA"
        )
    )
    assert out["rid"] == "1-AAAA"
    assert out["asset_table"] == "Image"
    assert out["asset_types"] == ["Training_Data"]
    assert out["executions"] == [{"rid": "1-EXEC", "asset_role": "Output"}]


async def test_ml_asset_detail_omits_no_data(resource_ctx, capturing_mcp, mock_ml):
    """Asset with no executions and no description still serializes cleanly."""
    asset = _make_asset_detail_mock(description="", executions=[])
    mock_ml.lookup_asset.return_value = asset

    out = json.loads(
        await capturing_mcp.resources[_ASSET_DETAIL_URI](
            hostname="h", catalog_id="1", asset_rid="1-AAAA"
        )
    )
    # Empty containers come back as actual empty values, not missing keys --
    # the resource is part of the contract surface (LLM reads `executions`
    # unconditionally), so absent keys would force defensive get(...) calls.
    assert out["description"] == ""
    assert out["executions"] == []
    assert out["asset_types"] == []
