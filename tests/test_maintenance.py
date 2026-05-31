"""Unit tests for the RAG cache management tools.

``tools/maintenance.py`` exposes two manual cache-refresh tools:

- ``deriva_ml_reindex_vocabularies`` (v1.1) -- thin wrapper around
  ``resources.rag._index_vocabularies`` (the same coroutine the
  on_catalog_connect hook fires); covered by the ``vocab_*`` tests
  in the first half of this file.
- ``deriva_ml_resync_indexes`` (v1.4) -- cross-user freshness bridge
  for the per-user-trio sources; thin wrapper around
  ``resources.rag._resync_user_sources``; covered by the ``resync_*``
  tests in the second half of this file.

Both tools are ``mutates=False`` (cache refresh, not catalog mutation).
The tests focus on the wrappers' contracts: success-path JSON shape,
argument forwarding, and the no-audit-on-failure guarantee.

Note on audit patching: unlike the other domain test files, this one
patches only ``deriva_ml_mcp_plugin._helpers.audit_event`` -- the failure-path
bind site. Neither tool module ever imports ``audit_event`` directly
(both are ``mutates=False``, so no emission anywhere). A single patch
is sufficient to assert "audit_event was not called".

History: this module was named ``test_vocabulary.py`` in v1.1 when it
held only the vocab tool; v1.4 added the resync tool, which isn't
vocabulary-specific, so the file was renamed alongside the
``tools/vocabulary.py -> tools/maintenance.py`` rename. The fixture
name ``vocab_ctx`` is preserved for symmetry with the underlying
domain (both tools live alongside the vocab indexer in spirit).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


def _patch_helpers_audit():
    """Context manager that patches the helpers-module audit_event bind site.

    Both tools in ``tools/maintenance.py`` are ``mutates=False`` on
    both success and failure paths, so neither bind site should ever
    fire. This single patch is sufficient to assert that contract
    (helpers.audit_event is the failure-path entry; neither tool
    module imports the success-path binding because neither emits
    one).
    """
    return patch("deriva_ml_mcp_plugin._helpers.audit_event")


@pytest.fixture()
def vocab_ctx(ctx):
    """Register the maintenance tools on a fresh PluginContext.

    The fixture name ``vocab_ctx`` is preserved from the v1.1 module
    layout for symmetry; the registered surface now includes both
    ``deriva_ml_reindex_vocabularies`` and ``deriva_ml_resync_indexes``.
    """
    from deriva_ml_mcp_plugin.tools import maintenance as maintenance_module

    maintenance_module.register(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


async def test_reindex_all_vocabs_returns_indexed_dict(vocab_ctx, capturing_mcp):
    """No ``vocab`` arg -> _index_vocabularies called with only_vocab=None."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    indexed = {
        "deriva-ml.Dataset_Type": 7,
        "deriva-ml.Workflow_Type": 4,
    }
    with (
        patch.object(rag_module, "_index_vocabularies", return_value=indexed) as fake_index,
        _patch_helpers_audit() as mock_audit,
    ):
        result = await capturing_mcp.tools["deriva_ml_reindex_vocabularies"](
            hostname="h.example", catalog_id="1"
        )

    fake_index.assert_awaited_once_with("h.example", "1", only_vocab=None)
    payload = json.loads(result)
    assert payload == {"reindexed": indexed}
    # mutates=False -> no audit emission on success.
    mock_audit.assert_not_called()


async def test_reindex_specific_vocab_forwards_filter(vocab_ctx, capturing_mcp):
    """``vocab="schema.X"`` is forwarded to _index_vocabularies as ``only_vocab``."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    indexed = {"deriva-ml.Workflow_Type": 4}
    with patch.object(rag_module, "_index_vocabularies", return_value=indexed) as fake_index:
        result = await capturing_mcp.tools["deriva_ml_reindex_vocabularies"](
            hostname="h.example", catalog_id="1", vocab="deriva-ml.Workflow_Type"
        )

    fake_index.assert_awaited_once_with("h.example", "1", only_vocab="deriva-ml.Workflow_Type")
    payload = json.loads(result)
    assert payload == {"reindexed": indexed}


async def test_reindex_empty_result_is_propagated(vocab_ctx, capturing_mcp):
    """If no vocab indexed (e.g. RAG disabled, _index returns {}) we still return success shape."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    with patch.object(rag_module, "_index_vocabularies", return_value={}):
        result = await capturing_mcp.tools["deriva_ml_reindex_vocabularies"](
            hostname="h.example", catalog_id="1"
        )

    payload = json.loads(result)
    assert payload == {"reindexed": {}}


# ---------------------------------------------------------------------------
# Error path -- mutates=False, so no audit emission even on failure
# ---------------------------------------------------------------------------


async def test_reindex_failure_returns_error_envelope_without_audit(vocab_ctx, capturing_mcp):
    """A raised exception lands as ``{"error": ...}`` and does NOT emit an audit event.

    ``mutates=False`` is the contract: the audit log is for catalog
    state changes; a cache refresh failure (vector-store outage,
    catalog auth glitch) doesn't change catalog state and so doesn't
    rate an audit row -- only an error log line.
    """
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    with (
        patch.object(rag_module, "_index_vocabularies", side_effect=RuntimeError("boom")),
        _patch_helpers_audit() as mock_audit,
    ):
        result = await capturing_mcp.tools["deriva_ml_reindex_vocabularies"](
            hostname="h.example", catalog_id="1"
        )

    payload = json.loads(result)
    assert "error" in payload
    assert "boom" in payload["error"]
    mock_audit.assert_not_called()


# ---------------------------------------------------------------------------
# deriva_ml_resync_indexes (v1.4) -- cross-user freshness manual refresh
# ---------------------------------------------------------------------------


async def test_resync_indexes_all_returns_counts_dict(vocab_ctx, capturing_mcp):
    """No ``target`` arg -> _resync_user_sources called with target=None; counts forwarded."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    counts = {"dataset": 5, "workflow": 2, "execution": 7}
    with (
        patch.object(rag_module, "_resync_user_sources", return_value=counts) as fake_resync,
        _patch_helpers_audit() as mock_audit,
    ):
        result = await capturing_mcp.tools["deriva_ml_resync_indexes"](
            hostname="h.example", catalog_id="1"
        )

    fake_resync.assert_awaited_once_with("h.example", "1", target=None)
    payload = json.loads(result)
    assert payload == {"resynced": counts}
    # mutates=False -> no audit emission on success.
    mock_audit.assert_not_called()


async def test_resync_indexes_targeted_forwards_target(vocab_ctx, capturing_mcp):
    """``target="dataset:1-AAAA"`` is forwarded to _resync_user_sources surgically."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    counts = {"dataset": 1, "workflow": 0, "execution": 0}
    with patch.object(rag_module, "_resync_user_sources", return_value=counts) as fake_resync:
        result = await capturing_mcp.tools["deriva_ml_resync_indexes"](
            hostname="h.example", catalog_id="1", target="dataset:1-AAAA"
        )

    fake_resync.assert_awaited_once_with("h.example", "1", target="dataset:1-AAAA")
    payload = json.loads(result)
    assert payload == {"resynced": counts}


async def test_resync_indexes_partial_progress_is_success(vocab_ctx, capturing_mcp):
    """Per-RID failures inside _resync_user_sources are swallowed; partial counts return as success."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    # Realistic scenario: dataset reindex partly failed (4/6 succeeded), workflow
    # totally failed (0/3), execution clean. _resync_user_sources's contract is
    # "return what landed"; the wrapping tool just forwards.
    counts = {"dataset": 4, "workflow": 0, "execution": 7}
    with patch.object(rag_module, "_resync_user_sources", return_value=counts):
        result = await capturing_mcp.tools["deriva_ml_resync_indexes"](
            hostname="h.example", catalog_id="1"
        )

    payload = json.loads(result)
    assert "error" not in payload
    assert payload == {"resynced": counts}


async def test_resync_indexes_malformed_target_returns_error(vocab_ctx, capturing_mcp):
    """A malformed ``target`` string surfaces as ``{"error": ...}`` (no audit).

    ``_resync_user_sources`` raises ValueError for malformed targets;
    the tool wraps it. The error message should mention the expected
    format so the LLM can recover.
    """
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    with (
        patch.object(
            rag_module,
            "_resync_user_sources",
            side_effect=ValueError("target must be in '<table>:<rid>' form; got 'badtarget'"),
        ),
        _patch_helpers_audit() as mock_audit,
    ):
        result = await capturing_mcp.tools["deriva_ml_resync_indexes"](
            hostname="h.example", catalog_id="1", target="badtarget"
        )

    payload = json.loads(result)
    assert "error" in payload
    assert "<table>:<rid>" in payload["error"]
    mock_audit.assert_not_called()


async def test_resync_indexes_failure_returns_error_envelope_without_audit(
    vocab_ctx, capturing_mcp
):
    """A raised exception lands as ``{"error": ...}`` and does NOT emit an audit event.

    Same ``mutates=False`` contract as deriva_ml_reindex_vocabularies:
    cache refresh failures don't change catalog state and so don't
    rate an audit row.
    """
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    with (
        patch.object(
            rag_module, "_resync_user_sources", side_effect=RuntimeError("vector store down")
        ),
        _patch_helpers_audit() as mock_audit,
    ):
        result = await capturing_mcp.tools["deriva_ml_resync_indexes"](
            hostname="h.example", catalog_id="1"
        )

    payload = json.loads(result)
    assert "error" in payload
    assert "vector store down" in payload["error"]
    mock_audit.assert_not_called()
