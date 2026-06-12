"""Unit tests for the deriva_ml_describe_rid tool (tools/resolve.py).

``deriva_ml_describe_rid`` is the "step 0" of the ordered query
strategy: given a bare RID of unknown type, resolve it to its
(schema, table), classify it into the DerivaML abstractions, and
suggest the right typed tool / resource to call next -- replacing the
try-get_dataset-then-get_execution guessing an LLM otherwise does.

All tests are fully mocked: ``get_ml`` is patched at the module's
import site and the fake DerivaML exposes ``resolve_rid`` plus the
``model.is_vocabulary`` / ``model.is_asset`` / ``model.is_association``
classifiers the tool consults.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def resolve_ctx(ctx):
    """Register the resolve tools on a fresh PluginContext."""
    from deriva_ml_mcp_plugin.tools import resolve as resolve_module

    resolve_module.register(ctx)
    return ctx


def _fake_ml(
    *,
    table_name: str,
    schema_name: str,
    ml_schema: str = "deriva-ml",
    is_vocabulary: bool = False,
    is_asset: bool = False,
    is_association: bool = False,
):
    """Build a fake DerivaML whose resolve_rid lands in (schema, table)."""
    ml = MagicMock()
    ml.ml_schema = ml_schema
    table = MagicMock()
    table.name = table_name
    table.schema.name = schema_name
    result = MagicMock()
    result.table = table
    ml.resolve_rid.return_value = result
    ml.model.is_vocabulary.return_value = is_vocabulary
    ml.model.is_asset.return_value = is_asset
    ml.model.is_association.return_value = is_association
    return ml


async def _describe(capturing_mcp, fake_ml, rid="1-ABCD"):
    from deriva_ml_mcp_plugin.tools import resolve as resolve_module

    with patch.object(resolve_module, "get_ml", return_value=fake_ml):
        raw = await capturing_mcp.tools["deriva_ml_describe_rid"](
            hostname="h.example", catalog_id="1", rid=rid
        )
    return json.loads(raw)


async def test_describe_rid_classifies_dataset(resolve_ctx, capturing_mcp):
    """A RID in the ML schema's Dataset table classifies as kind=dataset."""
    payload = await _describe(
        capturing_mcp, _fake_ml(table_name="Dataset", schema_name="deriva-ml")
    )
    assert payload["rid"] == "1-ABCD"
    assert payload["schema"] == "deriva-ml"
    assert payload["table"] == "Dataset"
    assert payload["kind"] == "dataset"
    assert "deriva_ml_get_dataset" in payload["suggestion"]
    assert payload["resource_uri"].endswith("/deriva-ml/dataset/1-ABCD")


async def test_describe_rid_classifies_execution_with_lineage_chain(resolve_ctx, capturing_mcp):
    """An Execution RID suggests get_execution AND the lineage chain."""
    payload = await _describe(
        capturing_mcp, _fake_ml(table_name="Execution", schema_name="deriva-ml")
    )
    assert payload["kind"] == "execution"
    assert "deriva_ml_get_execution" in payload["suggestion"]
    assert "deriva_ml_get_lineage" in payload["suggestion"]
    assert payload["resource_uri"].endswith("/deriva-ml/execution/1-ABCD")


async def test_describe_rid_classifies_workflow(resolve_ctx, capturing_mcp):
    """A Workflow RID suggests get_workflow."""
    payload = await _describe(
        capturing_mcp, _fake_ml(table_name="Workflow", schema_name="deriva-ml")
    )
    assert payload["kind"] == "workflow"
    assert "deriva_ml_get_workflow" in payload["suggestion"]
    assert payload["resource_uri"].endswith("/deriva-ml/workflow/1-ABCD")


async def test_describe_rid_classifies_vocabulary_term(resolve_ctx, capturing_mcp):
    """A RID in a vocabulary table classifies as a term, not a dataset."""
    payload = await _describe(
        capturing_mcp,
        _fake_ml(table_name="Diagnosis_Tag", schema_name="eye-ai", is_vocabulary=True),
    )
    assert payload["kind"] == "vocabulary_term"
    assert payload["table"] == "Diagnosis_Tag"
    assert payload["resource_uri"] is None


async def test_describe_rid_classifies_asset(resolve_ctx, capturing_mcp):
    """A RID in an asset table suggests lookup_asset + lineage."""
    payload = await _describe(
        capturing_mcp,
        _fake_ml(table_name="Model_Artifact", schema_name="eye-ai", is_asset=True),
    )
    assert payload["kind"] == "asset"
    assert "deriva_ml_lookup_asset" in payload["suggestion"]
    assert "deriva_ml_get_lineage" in payload["suggestion"]
    assert payload["resource_uri"].endswith("/deriva-ml/asset/1-ABCD")


async def test_describe_rid_classifies_association(resolve_ctx, capturing_mcp):
    """A RID in an association table is named as such (possible feature values)."""
    payload = await _describe(
        capturing_mcp,
        _fake_ml(table_name="Image_Diagnosis", schema_name="eye-ai", is_association=True),
    )
    assert payload["kind"] == "association"
    assert "deriva_ml_list_feature_values" in payload["suggestion"]
    assert payload["resource_uri"] is None


async def test_describe_rid_classifies_domain_entity(resolve_ctx, capturing_mcp):
    """A plain domain-table RID falls through to kind=entity with generic guidance."""
    payload = await _describe(capturing_mcp, _fake_ml(table_name="Image", schema_name="eye-ai"))
    assert payload["kind"] == "entity"
    assert "deriva_ml_get_lineage" in payload["suggestion"]
    assert payload["resource_uri"] is None


async def test_describe_rid_invalid_rid_returns_error_envelope(resolve_ctx, capturing_mcp):
    """An unresolvable RID returns {"error": ...} without raising."""
    from deriva_ml_mcp_plugin.tools import resolve as resolve_module

    ml = MagicMock()
    ml.resolve_rid.side_effect = RuntimeError("Invalid RID 9-NOPE")
    with patch.object(resolve_module, "get_ml", return_value=ml):
        raw = await capturing_mcp.tools["deriva_ml_describe_rid"](
            hostname="h.example", catalog_id="1", rid="9-NOPE"
        )
    payload = json.loads(raw)
    assert "error" in payload
    assert "9-NOPE" in payload["error"]


def test_describe_rid_registered_read_only(ctx):
    """describe_rid is mutates=False (pure read; no audit on success).

    ``PluginContext.tool`` strips ``mutates`` before forwarding to
    FastMCP, so we wrap ``ctx.tool`` to capture the declared value --
    same pattern as test_plugin.py's plugin-wide mutates check.
    """
    from deriva_ml_mcp_plugin.tools import resolve as resolve_module

    seen: dict[str, bool] = {}
    original_tool = ctx.tool

    def capturing_tool(*args, mutates, **kwargs):
        decorator = original_tool(*args, mutates=mutates, **kwargs)

        def wrapper(fn):
            seen[fn.__name__] = mutates
            return decorator(fn)

        return wrapper

    ctx.tool = capturing_tool
    try:
        resolve_module.register(ctx)
    finally:
        ctx.tool = original_tool

    assert seen == {"deriva_ml_describe_rid": False}
