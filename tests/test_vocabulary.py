"""Unit tests for the vocabulary domain tool.

The single tool exposed by ``tools/vocabulary.py`` is the manual
``deriva_ml_reindex_vocabularies`` re-indexer. It's a thin wrapper
around ``resources.rag._index_vocabularies`` (the same coroutine the
on_catalog_connect hook fires) -- the tests focus on the wrapper's
contract: success-path JSON shape, the ``vocab=`` qname filter
forwarding, and the ``mutates=False`` no-audit-on-failure guarantee
(audit log is for catalog state changes; cache refreshes don't count).

Note on audit patching: unlike the other domain test files, this one
patches only ``deriva_ml_mcp._helpers.audit_event`` -- the failure-path
bind site. The tool module itself never imports ``audit_event``
(success path is mutates=False, so no emission anywhere). A single
patch is sufficient to assert "audit_event was not called".
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


def _patch_helpers_audit():
    """Context manager that patches the helpers-module audit_event bind site.

    The vocabulary tool is mutates=False on both success and failure
    paths, so neither bind site should ever fire. This single patch is
    sufficient to assert that contract (helpers.audit_event is the
    failure-path entry; the tool module never imports the success-path
    binding because it never emits one).
    """
    return patch("deriva_ml_mcp._helpers.audit_event")


@pytest.fixture()
def vocab_ctx(ctx):
    """Register vocabulary tools on a fresh PluginContext."""
    from deriva_ml_mcp.tools import vocabulary as vocabulary_module

    vocabulary_module.register(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


async def test_reindex_all_vocabs_returns_indexed_dict(vocab_ctx, capturing_mcp):
    """No ``vocab`` arg -> _index_vocabularies called with only_vocab=None."""
    from deriva_ml_mcp.resources import rag as rag_module

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
    from deriva_ml_mcp.resources import rag as rag_module

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
    from deriva_ml_mcp.resources import rag as rag_module

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
    from deriva_ml_mcp.resources import rag as rag_module

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
