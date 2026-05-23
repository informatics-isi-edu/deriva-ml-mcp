"""Unit tests for resources/ml.py (14 ML-domain MCP resources, v3.4).

Resources are read-only and emit no audit on success or failure. The
fixture wires ``mock_ml`` into the resources/ml.py registration call so
each test can stub the deriva-ml call surface independently.

v3.4 retired ``ml/registries`` and ``ml/asset-tables`` and added four
schema-scoped resources covering vocabularies and asset tables:
``ml/vocabularies/{schema}``,
``ml/vocabularies/{schema}/{vocab_name}``, ``ml/assets/{schema}``, and
``ml/assets/{schema}/{asset_table}``.
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
_DATASET_BAG_PREVIEW_URI = (
    "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/bag-preview"
)
_DATASET_MEMBERS_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/members"
_WORKFLOWS_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/workflows"
_WORKFLOW_DETAIL_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/workflow/{workflow_rid}"
_EXECUTIONS_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/executions"
_EXECUTION_DETAIL_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/execution/{execution_rid}"
_LINEAGE_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/lineage/{rid}"
_FEATURES_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/features/{table_name}"
_ASSET_DETAIL_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/asset/{asset_rid}"
_ASSETS_IN_SCHEMA_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/assets/{schema}"
_ASSET_TABLE_CONTENTS_URI = (
    "deriva://catalog/{hostname}/{catalog_id}/ml/assets/{schema}/{asset_table}"
)
_VOCABS_IN_SCHEMA_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/vocabularies/{schema}"
_VOCAB_TERMS_URI = "deriva://catalog/{hostname}/{catalog_id}/ml/vocabularies/{schema}/{vocab_name}"


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


def test_register_adds_all_resources(resource_ctx, capturing_mcp):
    """resources/ml.py.register() must register exactly the expected URI set.

    Two static cold-start resources (``deriva://deriva-ml/concepts`` and
    ``deriva://deriva-ml/getting-started``) plus 15 catalog-scoped ML
    resource templates. The cold-start resources expose the same content
    as the matching MCP prompts -- a second discovery channel for
    resource-walking clients. See ``resources/ml.py`` register() preamble
    for the rationale (closes Section A / C1c of the 2026-05-16 findings).
    """
    expected = {
        # Cold-start orientation -- static URIs, no parameters.
        "deriva://deriva-ml/concepts",
        "deriva://deriva-ml/getting-started",
        # Catalog-scoped ML resource templates.
        "deriva://catalog/{hostname}/{catalog_id}/ml/datasets",
        "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/spec",
        "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/bag-preview",
        "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/members",
        "deriva://catalog/{hostname}/{catalog_id}/ml/workflows",
        "deriva://catalog/{hostname}/{catalog_id}/ml/workflow/{workflow_rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/executions",
        "deriva://catalog/{hostname}/{catalog_id}/ml/execution/{execution_rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/lineage/{rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/features/{table_name}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/asset/{asset_rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/assets/{schema}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/assets/{schema}/{asset_table}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/vocabularies/{schema}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/vocabularies/{schema}/{vocab_name}",
    }
    actual = set(capturing_mcp.resources.keys())
    assert actual == expected, (
        f"missing: {sorted(expected - actual)}; extra: {sorted(actual - expected)}"
    )


# ---------------------------------------------------------------------------
# Cold-start static resources (deriva://deriva-ml/{concepts,getting-started})
# ---------------------------------------------------------------------------


async def test_static_resources_return_prompt_constants(resource_ctx, capturing_mcp):
    """Static cold-start resources return the SAME content as their MCP prompts.

    Pins the "two channels, one source of truth" contract: both
    ``deriva_ml_concepts`` (prompt) and ``deriva://deriva-ml/concepts``
    (resource) MUST return ``_CONCEPTS_GUIDE``, and the same for
    ``getting-started``. If the resource handler ever wraps or trims the
    content, this test catches it.
    """
    from deriva_ml_mcp.prompts import _CONCEPTS_GUIDE, _GETTING_STARTED_GUIDE

    concepts_handler = capturing_mcp.resources["deriva://deriva-ml/concepts"]
    getting_started_handler = capturing_mcp.resources["deriva://deriva-ml/getting-started"]

    assert (await concepts_handler()) == _CONCEPTS_GUIDE
    assert (await getting_started_handler()) == _GETTING_STARTED_GUIDE


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
    # _cite_dataset_version_url interpolates ml.host_name and
    # ml.catalog_id into the snapshot URL; set them to real strings so
    # the f-string produces a real URL rather than embedding MagicMock
    # reprs.
    mock_ml.host_name = "data.example.org"
    mock_ml.catalog_id = "1"

    out = json.loads(
        await capturing_mcp.resources[_DATASET_DETAIL_URI](
            hostname="h", catalog_id="1", dataset_rid="1-AAAA"
        )
    )
    assert out["rid"] == "1-AAAA"
    # current_version "1.0.0" is a released label (not is_devrelease),
    # found in dataset_history with snapshot "snap1" -- the cite_url
    # routes to the snaptime-pinned form.
    assert out["cite_url"] == "https://data.example.org/id/1/1-AAAA@snap1"
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
# ml/dataset/{rid}/bag-preview
# ---------------------------------------------------------------------------


async def test_ml_dataset_bag_preview_uses_current_version(resource_ctx, capturing_mcp, mock_ml):
    """Resource form omits ``version`` -- always inspects current version
    with a warning recommending an explicit pin. The shared
    ``_bag_info_impl`` helper is what the tool uses too, so a drift in
    response shape (excluding the stateless-rule-stripped fields) would
    break both.

    v4.0.0: the resource strips ``cache_status`` / ``cache_path`` from
    the response because those describe the MCP server's local cache,
    not the caller's (stateless rule, docs/audit-2026-05-23.md). The
    tool path still surfaces them.
    """
    ds = MagicMock()
    ds.current_version = "1.2.0"
    mock_ml.lookup_dataset.return_value = ds
    mock_ml.bag_info.return_value = {
        "tables": {"Image": {"row_count": 100, "is_asset": True, "asset_bytes": 12345}},
        "total_rows": 100,
        "total_asset_bytes": 12345,
        "total_asset_size": "12 KB",
        "cache_status": "not_cached",
        "cache_path": None,
    }

    out = json.loads(
        await capturing_mcp.resources[_DATASET_BAG_PREVIEW_URI](
            hostname="h", catalog_id="1", dataset_rid="1-DSAA"
        )
    )
    assert out["dataset_rid"] == "1-DSAA"
    assert out["version"] == "1.2.0"
    assert out["total_rows"] == 100
    # Server-side cache state stripped per the stateless rule.
    assert "cache_status" not in out
    assert "cache_path" not in out
    # Resource always uses current version, so warning is always set.
    assert out["warning"] is not None
    assert "version not specified" in out["warning"]
    # bag_info was called with a DatasetSpec carrying the resolved
    # version and no exclude_tables (resource omits that parameter).
    spec_arg = mock_ml.bag_info.call_args.args[0]
    assert spec_arg.rid == "1-DSAA"
    assert str(spec_arg.version) == "1.2.0"
    assert spec_arg.exclude_tables is None


async def test_ml_dataset_bag_preview_error_path_is_silent(resource_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_dataset.side_effect = RuntimeError("Dataset not found")
    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(
            await capturing_mcp.resources[_DATASET_BAG_PREVIEW_URI](
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

    # Stub one input asset and one output asset. All Asset attributes the
    # impl reads (asset_rid, filename, description, asset_types,
    # asset_table) must be set explicitly -- MagicMock auto-creates
    # missing attributes as nested MagicMocks, which would silently fail
    # the iterable-coercion in the production code's `list(asset_types)`
    # and trigger the except-pass guard.
    input_asset = MagicMock()
    input_asset.asset_rid = "1-IN"
    input_asset.filename = "in.txt"
    input_asset.asset_table = "Execution_Asset"
    input_asset.asset_types = []
    input_asset.description = None
    output_asset = MagicMock()
    output_asset.asset_rid = "1-OUT"
    output_asset.filename = "out.txt"
    output_asset.asset_table = "Execution_Asset"
    output_asset.asset_types = []
    output_asset.description = None

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
    # Asset rows include description, asset_types, asset_table, cite_url.
    # The mocks here only set asset_table; description/asset_types/cite_url
    # come back as their defaults (None / [] / None — the cite_url helper's
    # ml.cite call returns a MagicMock under this fixture, which the
    # impl coerces to None rather than failing Pydantic validation).
    assert out["inputs"] == {
        "datasets": [
            {"rid": "1-DS", "version": "1.0.0", "cite_url": None},
        ],
        "assets": [
            {
                "rid": "1-IN",
                "filename": "in.txt",
                "description": None,
                "asset_types": [],
                "asset_table": "Execution_Asset",
                "cite_url": None,
            }
        ],
    }
    # Outputs split into two buckets: ``assets`` (real Execution_Asset
    # outputs -- model weights, predictions, etc.) and ``metadata``
    # (Execution_Metadata outputs -- Hydra configs, env snapshots,
    # uv.lock, etc.). Both buckets are always present even when empty.
    assert out["outputs"] == {
        "assets": [
            {
                "rid": "1-OUT",
                "filename": "out.txt",
                "description": None,
                "asset_types": [],
                "asset_table": "Execution_Asset",
                "cite_url": None,
            }
        ],
        "metadata": [],
    }
    # v2.0 wire change: experiment is always present, set to None when
    # the execution is not a Hydra-driven Experiment. v1.x omitted the
    # key entirely; the typed contract from PR 2 (#6) now surfaces it
    # explicitly so consumers can introspect without a KeyError.
    assert out["experiment"] is None


async def test_ml_execution_detail_categorizes_metadata_outputs(
    resource_ctx, capturing_mcp, mock_ml
):
    """Output assets split by ``asset_table`` into ``assets`` vs ``metadata``.

    ``record.list_assets(asset_role="Output")`` returns BOTH ``Execution_Asset``
    rows (model weights, prediction CSVs, training logs) and
    ``Execution_Metadata`` rows (configuration.json, hydra-*.yaml, uv.lock,
    environment snapshots) mixed together. The detail payload categorizes
    them by their ``asset_table`` attribute so a consumer can answer "what
    real outputs did this run produce?" without filtering metadata
    out themselves.
    """
    record = _make_execution_record_mock(rid="1-EXEC", workflow_rid="1-WF")
    record.list_input_datasets.return_value = []

    # Three real outputs (Execution_Asset) and three metadata files
    # (Execution_Metadata), shaped like a typical CIFAR-10 training run.
    # All Asset attributes the impl reads (asset_rid, filename,
    # description, asset_types, asset_table) must be set explicitly --
    # MagicMock auto-creates missing attributes as nested MagicMocks,
    # which would silently fail the iterable-coercion in the production
    # code's `list(asset_types)` and trigger the except-pass guard.
    def _mock_asset(rid, fn, table):
        m = MagicMock()
        m.asset_rid = rid
        m.filename = fn
        m.asset_table = table
        m.asset_types = []
        m.description = None
        return m

    weights = _mock_asset("1-WEIGHT", "cifar10_cnn_weights.pt", "Execution_Asset")
    log = _mock_asset("1-LOG", "training_log.txt", "Execution_Asset")
    preds = _mock_asset("1-CSV", "prediction_probabilities.csv", "Execution_Asset")
    config = _mock_asset("1-CFG", "configuration.json", "Execution_Metadata")
    hydra = _mock_asset("1-HYDRA", "hydra-1-config.yaml", "Execution_Metadata")
    lockfile = _mock_asset("1-LOCK", "uv.lock", "Execution_Metadata")

    def _list_assets(asset_role: str | None = None):
        if asset_role == "Output":
            return [weights, log, preds, config, hydra, lockfile]
        return []

    record.list_assets.side_effect = _list_assets
    mock_ml.lookup_execution.return_value = record
    mock_ml.lookup_experiment.side_effect = RuntimeError("Execution has no Experiment")

    out = json.loads(
        await capturing_mcp.resources[_EXECUTION_DETAIL_URI](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    )
    # Mocks here set asset_table only; description / asset_types / cite_url
    # come back as their defaults (None / [] / None).
    _bare = lambda rid, fn, table: {  # noqa: E731
        "rid": rid,
        "filename": fn,
        "description": None,
        "asset_types": [],
        "asset_table": table,
        "cite_url": None,
    }
    assert out["outputs"]["assets"] == [
        _bare("1-WEIGHT", "cifar10_cnn_weights.pt", "Execution_Asset"),
        _bare("1-LOG", "training_log.txt", "Execution_Asset"),
        _bare("1-CSV", "prediction_probabilities.csv", "Execution_Asset"),
    ]
    assert out["outputs"]["metadata"] == [
        _bare("1-CFG", "configuration.json", "Execution_Metadata"),
        _bare("1-HYDRA", "hydra-1-config.yaml", "Execution_Metadata"),
        _bare("1-LOCK", "uv.lock", "Execution_Metadata"),
    ]


async def test_ml_execution_detail_includes_asset_descriptions_and_types(
    resource_ctx, capturing_mcp, mock_ml
):
    """Asset rows on the bundle carry description, asset_types, asset_table.

    Without these fields, an agent asking "what assets did this execution
    produce" gets only RID + filename and has to fetch ``ml/asset/{rid}``
    per row to see what each asset is for. The bundle should be
    sufficient on its own for the typical "what outputs are these" reader
    use case; these three fields cover description (one-line "what is
    this?"), asset_types (controlled-vocab tags like Model_File /
    Output_File / Hydra_Config), and asset_table (the underlying table,
    which the categorization step also keys off but is useful for
    downstream code that wants to know "is this an Image vs an
    Execution_Asset?").
    """
    record = _make_execution_record_mock(rid="1-EXEC", workflow_rid="1-WF")
    record.list_input_datasets.return_value = []

    # One Execution_Asset output and one Execution_Metadata output, each
    # with full attribute coverage.
    weights = MagicMock()
    weights.asset_rid = "1-WEIGHT"
    weights.filename = "cifar10_cnn_weights.pt"
    weights.asset_table = "Execution_Asset"
    weights.asset_types = ["Model_File"]
    weights.description = "Trained CNN model weights"

    hydra = MagicMock()
    hydra.asset_rid = "1-HYDRA"
    hydra.filename = "hydra-1-config.yaml"
    hydra.asset_table = "Execution_Metadata"
    hydra.asset_types = ["Hydra_Config"]
    hydra.description = "Resolved Hydra configuration for this execution"

    def _list_assets(asset_role: str | None = None):
        if asset_role == "Output":
            return [weights, hydra]
        return []

    record.list_assets.side_effect = _list_assets
    mock_ml.lookup_execution.return_value = record
    mock_ml.lookup_experiment.side_effect = RuntimeError("Execution has no Experiment")

    out = json.loads(
        await capturing_mcp.resources[_EXECUTION_DETAIL_URI](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    )
    assert out["outputs"]["assets"] == [
        {
            "rid": "1-WEIGHT",
            "filename": "cifar10_cnn_weights.pt",
            "description": "Trained CNN model weights",
            "asset_types": ["Model_File"],
            "asset_table": "Execution_Asset",
            "cite_url": None,
        },
    ]
    assert out["outputs"]["metadata"] == [
        {
            "rid": "1-HYDRA",
            "filename": "hydra-1-config.yaml",
            "description": "Resolved Hydra configuration for this execution",
            "asset_types": ["Hydra_Config"],
            "asset_table": "Execution_Metadata",
            "cite_url": None,
        },
    ]


async def test_ml_execution_detail_handles_missing_asset_attributes(
    resource_ctx, capturing_mcp, mock_ml
):
    """Asset objects missing description / asset_types degrade to None / [].

    Older Asset implementations or unbound mocks may not expose every
    field. The detail payload should fall back gracefully rather than
    raising.
    """
    record = _make_execution_record_mock(rid="1-EXEC", workflow_rid="1-WF")
    record.list_input_datasets.return_value = []

    # Asset with only the bare-minimum attributes (rid, filename,
    # asset_table for categorization). description and asset_types are
    # absent -- spec=[] gives a strict mock with NO auto-attribute
    # behavior, so getattr() returns the supplied default.
    bare = MagicMock(spec=["asset_rid", "filename", "asset_table"])
    bare.asset_rid = "1-BARE"
    bare.filename = "bare.bin"
    bare.asset_table = "Execution_Asset"

    def _list_assets(asset_role: str | None = None):
        if asset_role == "Output":
            return [bare]
        return []

    record.list_assets.side_effect = _list_assets
    mock_ml.lookup_execution.return_value = record
    mock_ml.lookup_experiment.side_effect = RuntimeError("Execution has no Experiment")

    out = json.loads(
        await capturing_mcp.resources[_EXECUTION_DETAIL_URI](
            hostname="h", catalog_id="1", execution_rid="1-EXEC"
        )
    )
    assert out["outputs"]["assets"] == [
        {
            "rid": "1-BARE",
            "filename": "bare.bin",
            "description": None,
            "asset_types": [],
            "asset_table": "Execution_Asset",
            "cite_url": None,
        },
    ]


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


async def test_ml_execution_detail_experiment_tolerates_non_primitive_values(
    resource_ctx, capturing_mcp, mock_ml
):
    """Regression: ExecutionExperiment must accept list / dict values in
    model_config and config_choices.

    Surfaced 2026-05-19 in the e2e platform test session: a real Hydra-zen
    config produces a model_config dict that includes keys like
    ``_zen_exclude: list[str]`` (e.g. ``['description']``). The earlier
    typing of ``model_cfg: dict[str, str | int | float | bool | None]``
    rejected the list value at validation time, the bare ``except Exception``
    at read.py:337-338 silently turned the failure into ``experiment=None``,
    and the resource reported ``"experiment": null`` for every real
    Hydra-driven execution. The typing has been broadened to accept any
    JSON-serializable value (``Any``); this test pins that contract.
    """
    record = _make_execution_record_mock(rid="1-EXP", workflow_rid="1-WF")
    record.list_input_datasets.return_value = []
    record.list_assets.return_value = []
    mock_ml.lookup_execution.return_value = record

    experiment = MagicMock()
    experiment.name = "cifar10_quick"
    experiment.config_choices = {"model": "cnn_2layer", "optimizer": "adam"}
    # Real hydra-zen model_config carries lists (and could carry nested dicts);
    # the validator must tolerate all of it without dropping the field.
    experiment.model_config = {
        "lr": 0.001,
        "batch_size": 32,
        "_zen_exclude": ["description"],  # list of strings
        "_target_": "models.cifar10_cnn.cifar10_cnn",
        "nested": {"weight_decay": 0.0},  # nested dict
    }
    mock_ml.lookup_experiment.return_value = experiment

    out = json.loads(
        await capturing_mcp.resources[_EXECUTION_DETAIL_URI](
            hostname="h", catalog_id="1", execution_rid="1-EXP"
        )
    )

    # Pre-fix this silently came back as experiment=None.
    assert out["experiment"] is not None, (
        "experiment field was silently dropped (likely the ValidationError "
        "was swallowed by the bare `except Exception` at read.py:337-338)"
    )
    assert out["experiment"]["name"] == "cifar10_quick"
    assert out["experiment"]["model_config"]["_zen_exclude"] == ["description"]
    assert out["experiment"]["model_config"]["nested"] == {"weight_decay": 0.0}


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
        await capturing_mcp.resources[_LINEAGE_URI](hostname="h", catalog_id="1", rid="2-PRED")
    )
    assert out["root"]["rid"] == "2-PRED"
    assert out["root"]["type"] == "Asset"
    assert out["walked_complete"] is True
    assert out["cycle_detected"] is False
    # Resource always uses unbounded depth + default max_executions=500.
    mock_ml.lookup_lineage.assert_called_once_with("2-PRED", depth=None, max_executions=500)


async def test_ml_lineage_error_path_is_silent(resource_ctx, capturing_mcp, mock_ml):
    """A workflow RID (or any unclassifiable) raises ``DerivaMLException``;
    the resource wraps it as ``{"error": ...}`` and emits no audit row
    (resources are read-only and silent on failure)."""
    mock_ml.lookup_lineage.side_effect = RuntimeError("Workflow RIDs are not lineage-shaped")
    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(
            await capturing_mcp.resources[_LINEAGE_URI](hostname="h", catalog_id="1", rid="1-WF")
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
# ml/vocabularies/{schema} and ml/vocabularies/{schema}/{vocab_name} (v3.4)
# ---------------------------------------------------------------------------


def _make_table_mock(
    name: str,
    schema: str,
    rid: str = "",
    *,
    is_vocabulary: bool = False,
    is_asset: bool = False,
) -> MagicMock:
    """Build a deriva-py-like Table mock with vocab/asset disposition.

    The new resources walk ``ml.model.schemas[schema].tables.values()``
    and call ``table.is_vocabulary()`` / ``table.is_asset()`` to filter
    candidates. This factory keeps the shape stable across tests.
    """
    t = MagicMock()
    t.name = name
    t.schema = MagicMock()
    t.schema.name = schema
    t.rid = rid
    t.is_vocabulary.return_value = is_vocabulary
    t.is_asset.return_value = is_asset
    return t


def _make_full_vocab_term_mock(
    name: str,
    description: str = "",
    synonyms: list[str] | None = None,
    curie: str | None = None,
    uri: str | None = None,
) -> MagicMock:
    """Mock for the full ``VocabularyTerm`` shape (six fields)."""
    term = MagicMock()
    term.name = name
    term.description = description
    term.synonyms = synonyms or []
    term.rid = f"VRID-{name}"
    term.id = curie
    term.uri = uri
    return term


def _wire_schema(mock_ml: MagicMock, schema: str, tables: list[MagicMock]) -> None:
    """Wire ``mock_ml.model.schemas[schema].tables`` to the given tables."""
    schema_obj = MagicMock()
    schema_obj.tables = {t.name: t for t in tables}
    # ``model.schemas`` is read both via ``in`` and via ``.get(...)``;
    # using a real dict satisfies both call shapes without further setup.
    mock_ml.model.schemas = {schema: schema_obj}


async def test_ml_vocabularies_in_schema_lists_only_vocab_tables(
    resource_ctx, capturing_mcp, mock_ml
):
    """Walking a schema returns only its vocab tables, with term counts."""
    dataset_type = _make_table_mock("Dataset_Type", "deriva-ml", rid="T-1", is_vocabulary=True)
    workflow_type = _make_table_mock("Workflow_Type", "deriva-ml", rid="T-2", is_vocabulary=True)
    not_a_vocab = _make_table_mock("Workflow", "deriva-ml", rid="T-3")
    _wire_schema(mock_ml, "deriva-ml", [dataset_type, workflow_type, not_a_vocab])

    def _terms_for(table):
        if table is dataset_type:
            return [_make_full_vocab_term_mock("Training"), _make_full_vocab_term_mock("Testing")]
        if table is workflow_type:
            return [_make_full_vocab_term_mock("Model_Training")]
        return []

    mock_ml.list_vocabulary_terms.side_effect = _terms_for

    out = json.loads(
        await capturing_mcp.resources[_VOCABS_IN_SCHEMA_URI](
            hostname="h", catalog_id="1", schema="deriva-ml"
        )
    )
    assert out["schema"] == "deriva-ml"
    assert out["count"] == 2
    by_name = {v["name"]: v for v in out["vocabularies"]}
    assert by_name["Dataset_Type"]["term_count"] == 2
    assert by_name["Workflow_Type"]["term_count"] == 1
    assert "Workflow" not in by_name


async def test_ml_vocabularies_in_schema_empty_when_no_vocabs(resource_ctx, capturing_mcp, mock_ml):
    """A schema that exists but has no vocab tables returns an empty list."""
    _wire_schema(
        mock_ml,
        "domain",
        [_make_table_mock("Image", "domain", rid="T-IMG", is_asset=True)],
    )
    out = json.loads(
        await capturing_mcp.resources[_VOCABS_IN_SCHEMA_URI](
            hostname="h", catalog_id="1", schema="domain"
        )
    )
    assert out["count"] == 0
    assert out["vocabularies"] == []


async def test_ml_vocabularies_in_schema_unknown_schema_errors(
    resource_ctx, capturing_mcp, mock_ml
):
    """An unknown schema returns an error envelope."""
    _wire_schema(mock_ml, "deriva-ml", [])
    with patch("deriva_ml_mcp._helpers.audit_event") as mock_audit:
        out = json.loads(
            await capturing_mcp.resources[_VOCABS_IN_SCHEMA_URI](
                hostname="h", catalog_id="1", schema="nope"
            )
        )
    assert "error" in out
    assert "schema not found" in out["error"]
    assert mock_audit.call_count == 0


async def test_ml_vocabularies_in_schema_term_count_best_effort(
    resource_ctx, capturing_mcp, mock_ml
):
    """A failing per-vocab term fetch surfaces as ``term_count = None``."""
    good = _make_table_mock("Dataset_Type", "deriva-ml", rid="T-1", is_vocabulary=True)
    bad = _make_table_mock("Workflow_Type", "deriva-ml", rid="T-2", is_vocabulary=True)
    _wire_schema(mock_ml, "deriva-ml", [good, bad])

    def _terms_for(table):
        if table is good:
            return [_make_full_vocab_term_mock("Training")]
        raise RuntimeError("transient failure")

    mock_ml.list_vocabulary_terms.side_effect = _terms_for
    out = json.loads(
        await capturing_mcp.resources[_VOCABS_IN_SCHEMA_URI](
            hostname="h", catalog_id="1", schema="deriva-ml"
        )
    )
    by_name = {v["name"]: v for v in out["vocabularies"]}
    assert by_name["Dataset_Type"]["term_count"] == 1
    assert by_name["Workflow_Type"]["term_count"] is None


async def test_ml_vocabulary_terms_returns_full_six_field_shape(
    resource_ctx, capturing_mcp, mock_ml
):
    """A vocabulary read returns all six fields including CURIE and URI."""
    table = _make_table_mock("Tissue_Type", "domain", rid="T-T", is_vocabulary=True)
    _wire_schema(mock_ml, "domain", [table])
    mock_ml.list_vocabulary_terms.return_value = [
        _make_full_vocab_term_mock(
            "Epithelial",
            description="Epithelial tissue",
            synonyms=["Epithelium"],
            curie="tissue:0001",
            uri="http://example.org/tissue/0001",
        )
    ]
    out = json.loads(
        await capturing_mcp.resources[_VOCAB_TERMS_URI](
            hostname="h", catalog_id="1", schema="domain", vocab_name="Tissue_Type"
        )
    )
    assert out["schema"] == "domain"
    assert out["vocabulary"] == "Tissue_Type"
    assert out["count"] == 1
    term = out["terms"][0]
    assert term["name"] == "Epithelial"
    assert term["description"] == "Epithelial tissue"
    assert term["synonyms"] == ["Epithelium"]
    assert term["id"] == "tissue:0001"
    assert term["uri"] == "http://example.org/tissue/0001"
    assert term["rid"] == "VRID-Epithelial"


async def test_ml_vocabulary_terms_unknown_table_errors(resource_ctx, capturing_mcp, mock_ml):
    """An unknown table in a known schema returns an error envelope."""
    _wire_schema(mock_ml, "deriva-ml", [])
    out = json.loads(
        await capturing_mcp.resources[_VOCAB_TERMS_URI](
            hostname="h", catalog_id="1", schema="deriva-ml", vocab_name="Missing"
        )
    )
    assert "error" in out
    assert "vocabulary not found" in out["error"]


async def test_ml_vocabulary_terms_not_a_vocabulary_errors(resource_ctx, capturing_mcp, mock_ml):
    """A table that exists but isn't a vocabulary returns a specific error."""
    table = _make_table_mock("Workflow", "deriva-ml", rid="T-WF")  # is_vocabulary=False
    _wire_schema(mock_ml, "deriva-ml", [table])
    out = json.loads(
        await capturing_mcp.resources[_VOCAB_TERMS_URI](
            hostname="h", catalog_id="1", schema="deriva-ml", vocab_name="Workflow"
        )
    )
    assert "error" in out
    assert "is not a vocabulary" in out["error"]


# ---------------------------------------------------------------------------
# ml/assets/{schema} and ml/assets/{schema}/{asset_table} (v3.4)
# ---------------------------------------------------------------------------


def _make_asset_row_mock(rid: str = "1-AAAA") -> MagicMock:
    a = MagicMock()
    a.asset_rid = rid
    a.filename = f"file-{rid}.png"
    a.length = 100
    a.md5 = "abc"
    a.asset_table = "Image"
    a.asset_types = ["Training_Data"]
    return a


async def test_ml_assets_in_schema_lists_only_asset_tables(resource_ctx, capturing_mcp, mock_ml):
    """Walking a schema returns only its asset tables, name+rid only."""
    image = _make_table_mock("Image", "domain", rid="T-IMG", is_asset=True)
    model = _make_table_mock("Trained_Model", "domain", rid="T-M", is_asset=True)
    not_an_asset = _make_table_mock("Patient", "domain", rid="T-P")
    _wire_schema(mock_ml, "domain", [image, model, not_an_asset])

    out = json.loads(
        await capturing_mcp.resources[_ASSETS_IN_SCHEMA_URI](
            hostname="h", catalog_id="1", schema="domain"
        )
    )
    assert out["schema"] == "domain"
    assert out["count"] == 2
    names = {t["name"] for t in out["asset_tables"]}
    assert names == {"Image", "Trained_Model"}
    # No term_count / asset_count on this surface.
    for entry in out["asset_tables"]:
        assert set(entry.keys()) == {"name", "rid"}


async def test_ml_assets_in_schema_unknown_schema_errors(resource_ctx, capturing_mcp, mock_ml):
    """An unknown schema returns an error envelope."""
    _wire_schema(mock_ml, "domain", [])
    out = json.loads(
        await capturing_mcp.resources[_ASSETS_IN_SCHEMA_URI](
            hostname="h", catalog_id="1", schema="nope"
        )
    )
    assert "error" in out
    assert "schema not found" in out["error"]


async def test_ml_asset_table_contents_returns_summaries(resource_ctx, capturing_mcp, mock_ml):
    """Reading an asset table returns ``AssetSummary`` rows."""
    table = _make_table_mock("Image", "domain", rid="T-IMG", is_asset=True)
    _wire_schema(mock_ml, "domain", [table])
    mock_ml.list_assets.return_value = [
        _make_asset_row_mock("1-AAAA"),
        _make_asset_row_mock("1-BBBB"),
    ]
    out = json.loads(
        await capturing_mcp.resources[_ASSET_TABLE_CONTENTS_URI](
            hostname="h", catalog_id="1", schema="domain", asset_table="Image"
        )
    )
    assert out["schema"] == "domain"
    assert out["asset_table"] == "Image"
    assert out["count"] == 2
    assert out["truncated"] is False
    rids = {a["rid"] for a in out["assets"]}
    assert rids == {"1-AAAA", "1-BBBB"}


async def test_ml_asset_table_contents_unknown_table_errors(resource_ctx, capturing_mcp, mock_ml):
    """An unknown table in a known schema returns an error envelope."""
    _wire_schema(mock_ml, "domain", [])
    out = json.loads(
        await capturing_mcp.resources[_ASSET_TABLE_CONTENTS_URI](
            hostname="h", catalog_id="1", schema="domain", asset_table="Missing"
        )
    )
    assert "error" in out
    assert "asset table not found" in out["error"]


async def test_ml_asset_table_contents_not_an_asset_errors(resource_ctx, capturing_mcp, mock_ml):
    """A table that exists but isn't an asset returns a specific error."""
    table = _make_table_mock("Patient", "domain", rid="T-P")  # is_asset=False
    _wire_schema(mock_ml, "domain", [table])
    out = json.loads(
        await capturing_mcp.resources[_ASSET_TABLE_CONTENTS_URI](
            hostname="h", catalog_id="1", schema="domain", asset_table="Patient"
        )
    )
    assert "error" in out
    assert "is not an asset table" in out["error"]


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
