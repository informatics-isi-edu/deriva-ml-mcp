"""Unit tests for resources/rag.py (per-user RAG indexing + docs source).

Two surfaces under test:

1. ``register_rag_sources(ctx)`` -- declarative wiring: one
   ``rag_github_source`` declaration plus three ``on_catalog_connect``
   callbacks.

2. The serializers and per-user hook bodies -- rich Markdown rendering
   for Dataset / Workflow / Execution rows, fall-through to the generic
   serializer for unknown tables, and best-effort exception handling
   inside hooks (a hook that raises must not propagate).

All tests are fully mocked. ``index_table_data`` is patched so we can
assert on ``user_id`` without touching a real vector store; the row
fetchers are patched to keep the hook tests independent of the underlying
``_list_*_impl`` shape.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


# Hook registration order in register_rag_sources(ctx):
#   0 -> Dataset, 1 -> Workflow, 2 -> Execution, 3 -> Vocabularies.
# Tests pull hooks via index rather than by name because the factory
# refactor produces anonymous closures (no module-level attribute to import).
_DATASET_HOOK_IDX = 0
_WORKFLOW_HOOK_IDX = 1
_EXECUTION_HOOK_IDX = 2
_VOCAB_HOOK_IDX = 3


def _hook_at(idx: int):
    """Build a fresh PluginContext, register rag sources, return the hook at idx."""
    from deriva_mcp_core.plugin.api import PluginContext

    from deriva_ml_mcp.resources import rag as rag_module
    from tests._helpers import _CapturingMCP

    plugin_ctx = PluginContext(_CapturingMCP())
    rag_module.register_rag_sources(plugin_ctx)
    return plugin_ctx._catalog_connect_hooks[idx]


@pytest.fixture()
def rag_ctx(ctx):
    """A fresh PluginContext with rag.register_rag_sources(ctx) applied."""
    from deriva_ml_mcp.resources import rag as rag_module

    rag_module.register_rag_sources(ctx)
    return ctx


# ---------------------------------------------------------------------------
# 1. Declarative wiring -- register_rag_sources(ctx)
# ---------------------------------------------------------------------------


def test_register_rag_sources_declares_one_github_source(rag_ctx) -> None:
    """register_rag_sources adds exactly one GitHub documentation source."""
    assert len(rag_ctx._rag_sources) == 1


def test_register_rag_sources_github_source_targets_deriva_ml_docs(rag_ctx) -> None:
    """The GitHub source points at informatics-isi-edu/deriva-ml docs/."""
    decl = rag_ctx._rag_sources[0]
    assert decl.name == "deriva-ml-docs"
    assert decl.repo_owner == "informatics-isi-edu"
    assert decl.repo_name == "deriva-ml"
    assert decl.branch == "main"
    assert decl.path_prefix == "docs/"
    assert decl.doc_type == "ml-docs"


def test_register_rag_sources_registers_four_catalog_hooks(rag_ctx) -> None:
    """Four on_catalog_connect callbacks are registered (Dataset, Workflow, Execution, Vocab)."""
    assert len(rag_ctx._catalog_connect_hooks) == 4


def test_register_rag_sources_does_not_use_rag_dataset_indexer(rag_ctx) -> None:
    """rag_dataset_indexer is the unsafe global path -- must not be touched."""
    assert rag_ctx._rag_dataset_indexers == []


# ---------------------------------------------------------------------------
# 2. _DatasetSerializer
# ---------------------------------------------------------------------------


def test_dataset_serializer_renders_full_row() -> None:
    """A populated Dataset row renders all sections with real values."""
    from deriva_ml_mcp.resources.rag import _DatasetSerializer

    row = {
        "rid": "1-DSAA",
        "name": "cifar-10-train",
        "description": "Training partition of CIFAR-10.",
        "dataset_types": ["image-classification", "supervised"],
        "current_version": "1.2.0",
        "member_count": 50000,
    }
    md = _DatasetSerializer().serialize("Dataset", row)

    assert md is not None
    assert md.startswith("## Dataset: 1-DSAA")
    assert "**Name:** cifar-10-train" in md
    assert "**Description:** Training partition of CIFAR-10." in md
    assert "**Types:** image-classification, supervised" in md
    assert "**Version:** 1.2.0" in md
    assert "**Members:** 50000" in md


def test_dataset_serializer_omits_empty_fields() -> None:
    """Fields with None / empty list / empty string render as no line at all."""
    from deriva_ml_mcp.resources.rag import _DatasetSerializer

    row = {
        "rid": "1-DSBB",
        "name": "minimal",
        "description": None,
        "dataset_types": [],
        "current_version": "",
        "member_count": None,
    }
    md = _DatasetSerializer().serialize("Dataset", row)

    assert md is not None
    assert "**Name:** minimal" in md
    assert "Description" not in md
    assert "Types" not in md
    assert "Version" not in md
    assert "Members" not in md


def test_dataset_serializer_returns_none_for_other_table() -> None:
    """Other tables fall through to the generic renderer (return None)."""
    from deriva_ml_mcp.resources.rag import _DatasetSerializer

    assert _DatasetSerializer().serialize("Workflow", {"rid": "x"}) is None
    assert _DatasetSerializer().serialize("Image", {"RID": "x"}) is None


# ---------------------------------------------------------------------------
# 3. _WorkflowSerializer
# ---------------------------------------------------------------------------


def test_workflow_serializer_renders_full_row() -> None:
    """A populated Workflow row renders all sections."""
    from deriva_ml_mcp.resources.rag import _WorkflowSerializer

    row = {
        "rid": "1-WFAA",
        "name": "TrainCifarCNN",
        "workflow_type": ["Model_Training"],
        "url": "https://github.com/example/repo",
        "checksum": "abc123",
        "version": "1.0.0",
        "description": "CIFAR-10 baseline trainer.",
    }
    md = _WorkflowSerializer().serialize("Workflow", row)

    assert md is not None
    assert md.startswith("## Workflow: 1-WFAA")
    assert "**Name:** TrainCifarCNN" in md
    assert "**Type:** Model_Training" in md
    assert "**URL:** https://github.com/example/repo" in md
    assert "**Checksum:** abc123" in md
    assert "**Version:** 1.0.0" in md
    assert "**Description:** CIFAR-10 baseline trainer." in md


def test_workflow_serializer_omits_empty_fields() -> None:
    """Empty workflow fields are dropped from the output."""
    from deriva_ml_mcp.resources.rag import _WorkflowSerializer

    row = {
        "rid": "1-WFBB",
        "name": "Bare",
        "workflow_type": None,
        "url": None,
        "checksum": "",
        "version": None,
        "description": "",
    }
    md = _WorkflowSerializer().serialize("Workflow", row)

    assert md is not None
    assert "**Name:** Bare" in md
    assert "Type" not in md
    assert "URL" not in md
    assert "Checksum" not in md
    assert "Version" not in md
    assert "Description" not in md


def test_workflow_serializer_returns_none_for_other_table() -> None:
    """Workflow serializer returns None for non-Workflow tables."""
    from deriva_ml_mcp.resources.rag import _WorkflowSerializer

    assert _WorkflowSerializer().serialize("Dataset", {"rid": "x"}) is None
    assert _WorkflowSerializer().serialize("Execution", {"rid": "x"}) is None


# ---------------------------------------------------------------------------
# 4. _ExecutionSerializer
# ---------------------------------------------------------------------------


def test_execution_serializer_renders_full_row() -> None:
    """A populated Execution row renders all sections (timestamps via str())."""
    from deriva_ml_mcp.resources.rag import _ExecutionSerializer

    row = {
        "rid": "1-EXAA",
        "workflow_rid": "1-WFAA",
        "status": "Stopped",
        "description": "Run #1.",
        "start_time": datetime(2026, 1, 1, 12, 0, 0),
        "stop_time": datetime(2026, 1, 1, 12, 30, 0),
        "duration": "0:30:00",
    }
    md = _ExecutionSerializer().serialize("Execution", row)

    assert md is not None
    assert md.startswith("## Execution: 1-EXAA")
    assert "**Status:** Stopped" in md
    assert "**Workflow:** 1-WFAA" in md
    assert "**Description:** Run #1." in md
    assert "**Start Time:** 2026-01-01 12:00:00" in md
    assert "**Stop Time:** 2026-01-01 12:30:00" in md
    assert "**Duration:** 0:30:00" in md


def test_execution_serializer_omits_unset_timestamps() -> None:
    """Missing start/stop timestamps don't render their lines."""
    from deriva_ml_mcp.resources.rag import _ExecutionSerializer

    row = {
        "rid": "1-EXBB",
        "workflow_rid": "1-WFAA",
        "status": "Created",
        "description": None,
        "start_time": None,
        "stop_time": None,
        "duration": None,
    }
    md = _ExecutionSerializer().serialize("Execution", row)

    assert md is not None
    assert "**Status:** Created" in md
    assert "**Workflow:** 1-WFAA" in md
    assert "Description" not in md
    assert "Start Time" not in md
    assert "Stop Time" not in md
    assert "Duration" not in md


def test_execution_serializer_returns_none_for_other_table() -> None:
    """Execution serializer returns None for non-Execution tables."""
    from deriva_ml_mcp.resources.rag import _ExecutionSerializer

    assert _ExecutionSerializer().serialize("Dataset", {"rid": "x"}) is None


# ---------------------------------------------------------------------------
# 5. Per-user hook flow -- index_table_data is called with resolved user_id
# ---------------------------------------------------------------------------


_USER_ID = "https://auth.example.org/sub/test-user-1"


def test_dataset_hook_calls_index_table_data_with_resolved_user_id() -> None:
    """The Dataset hook resolves the caller's user_id and forwards it."""
    from deriva_ml_mcp.resources import rag as rag_module

    captured: dict[str, Any] = {}
    rows = [{"rid": "1-DSAA", "name": "x"}]

    async def fake_index_table_data(
        store, hostname, catalog_id, table_name, the_rows, *, user_id, serializer, **_kw
    ):
        captured["user_id"] = user_id
        captured["table_name"] = table_name
        captured["rows"] = the_rows
        captured["hostname"] = hostname
        captured["catalog_id"] = catalog_id

    with (
        patch.object(rag_module, "_fetch_dataset_rows", return_value=rows),
        patch.object(rag_module, "resolve_user_identity", return_value=_USER_ID),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "index_table_data", side_effect=fake_index_table_data),
    ):
        _run(_hook_at(_DATASET_HOOK_IDX)("h.example", "1", "hash", {}))

    assert captured["user_id"] == _USER_ID
    assert captured["table_name"] == "Dataset"
    assert captured["hostname"] == "h.example"
    assert captured["catalog_id"] == "1"
    assert captured["rows"] == rows


def test_workflow_hook_calls_index_table_data_with_resolved_user_id() -> None:
    """The Workflow hook resolves the caller's user_id and forwards it."""
    from deriva_ml_mcp.resources import rag as rag_module

    captured: dict[str, Any] = {}
    rows = [{"rid": "1-WFAA", "name": "y"}]

    async def fake_index_table_data(
        store, hostname, catalog_id, table_name, the_rows, *, user_id, serializer, **_kw
    ):
        captured["user_id"] = user_id
        captured["table_name"] = table_name

    with (
        patch.object(rag_module, "_fetch_workflow_rows", return_value=rows),
        patch.object(rag_module, "resolve_user_identity", return_value=_USER_ID),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "index_table_data", side_effect=fake_index_table_data),
    ):
        _run(_hook_at(_WORKFLOW_HOOK_IDX)("h.example", "1", "hash", {}))

    assert captured["user_id"] == _USER_ID
    assert captured["table_name"] == "Workflow"


def test_execution_hook_calls_index_table_data_with_resolved_user_id() -> None:
    """The Execution hook resolves the caller's user_id and forwards it."""
    from deriva_ml_mcp.resources import rag as rag_module

    captured: dict[str, Any] = {}
    rows = [{"rid": "1-EXAA", "workflow_rid": "1-WFAA", "status": "Created"}]

    async def fake_index_table_data(
        store, hostname, catalog_id, table_name, the_rows, *, user_id, serializer, **_kw
    ):
        captured["user_id"] = user_id
        captured["table_name"] = table_name

    with (
        patch.object(rag_module, "_fetch_execution_rows", return_value=rows),
        patch.object(rag_module, "resolve_user_identity", return_value=_USER_ID),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "index_table_data", side_effect=fake_index_table_data),
    ):
        _run(_hook_at(_EXECUTION_HOOK_IDX)("h.example", "1", "hash", {}))

    assert captured["user_id"] == _USER_ID
    assert captured["table_name"] == "Execution"


# ---------------------------------------------------------------------------
# 6. Best-effort behavior -- exceptions never propagate from hooks
# ---------------------------------------------------------------------------


def test_dataset_hook_swallows_fetch_exception() -> None:
    """If the row fetch raises, the hook logs and returns without propagating.

    The fetch-side ``try/except`` is retained inside the hook because
    a deriva-ml read failure is domain-specific (table-name context in
    the log helps debugging). The framework's ``_safe_call`` would catch
    it too, but with a generic message.
    """
    from deriva_ml_mcp.resources import rag as rag_module

    fake_index = AsyncMock()
    with (
        patch.object(rag_module, "_fetch_dataset_rows", side_effect=RuntimeError("boom")),
        patch.object(rag_module, "resolve_user_identity", return_value=_USER_ID),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "index_table_data", new=fake_index),
    ):
        # Must not raise
        _run(_hook_at(_DATASET_HOOK_IDX)("h.example", "1", "hash", {}))

    fake_index.assert_not_called()


def test_workflow_hook_propagates_index_exception() -> None:
    """If index_table_data raises, the hook lets it propagate.

    Phase 6.3 follow-on: the redundant index-side ``try/except`` was
    dropped. The framework's ``_safe_call`` already wraps every dispatched
    hook, so a duplicated suppression here added nothing -- only stripped
    the traceback. The contract is now: fetch errors are swallowed with
    table-name context; index errors propagate to the framework.
    """
    from deriva_ml_mcp.resources import rag as rag_module

    rows = [{"rid": "1-WFAA"}]
    with (
        patch.object(rag_module, "_fetch_workflow_rows", return_value=rows),
        patch.object(rag_module, "resolve_user_identity", return_value=_USER_ID),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "index_table_data", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        _run(_hook_at(_WORKFLOW_HOOK_IDX)("h.example", "1", "hash", {}))


def test_hook_short_circuits_when_rag_store_is_none() -> None:
    """When RAG is disabled (get_rag_store -> None), the hook does nothing."""
    from deriva_ml_mcp.resources import rag as rag_module

    fake_index = AsyncMock()
    rows = [{"rid": "1-DSAA"}]
    with (
        patch.object(rag_module, "_fetch_dataset_rows", return_value=rows),
        patch.object(rag_module, "resolve_user_identity", return_value=_USER_ID),
        patch.object(rag_module, "get_rag_store", return_value=None),
        patch.object(rag_module, "index_table_data", new=fake_index),
    ):
        _run(_hook_at(_DATASET_HOOK_IDX)("h.example", "1", "hash", {}))

    fake_index.assert_not_called()


def test_dataset_hook_partitions_per_user() -> None:
    """Two consecutive calls under different identities partition by user_id.

    The contract under test: same hostname/catalog_id, two different
    callers, two distinct ``user_id`` values forwarded into
    ``index_table_data`` -- one per call. This pins the per-user
    partitioning that keeps user A's rows out of user B's index.
    """
    from deriva_ml_mcp.resources import rag as rag_module

    rows = [{"rid": "1-DSAA", "name": "shared"}]
    fake_index = AsyncMock()

    with (
        patch.object(rag_module, "_fetch_dataset_rows", return_value=rows),
        patch.object(rag_module, "resolve_user_identity", side_effect=["userA", "userB"]),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "index_table_data", new=fake_index),
    ):
        # Build the hook inside the patch context so the factory's closure
        # captures the patched fetch fn rather than the real one.
        hook = _hook_at(_DATASET_HOOK_IDX)
        _run(hook("h.example", "1", "hash", {}))
        _run(hook("h.example", "1", "hash", {}))

    assert fake_index.call_count == 2
    assert fake_index.call_args_list[0].kwargs["user_id"] == "userA"
    assert fake_index.call_args_list[1].kwargs["user_id"] == "userB"


# ---------------------------------------------------------------------------
# 7. _VocabSerializer
# ---------------------------------------------------------------------------


def test_vocab_serializer_renders_full_term() -> None:
    """A populated vocab term renders all sections + parent table in header."""
    from deriva_ml_mcp.resources.rag import _VocabSerializer

    row = {
        "name": "epithelial",
        "description": "Epithelial tissue type.",
        "synonyms": ["epithelium", "epi"],
        "rid": "1-VOAA",
    }
    md = _VocabSerializer().serialize("demo-schema.Tissue_Type", row)

    assert md is not None
    assert md.startswith("## Vocab Term: epithelial (demo-schema.Tissue_Type)")
    assert "**Description:** Epithelial tissue type." in md
    assert "**Synonyms:** epithelium, epi" in md
    assert "**RID:** 1-VOAA" in md


def test_vocab_serializer_omits_empty_description() -> None:
    """Term without description still renders header + synonyms + RID."""
    from deriva_ml_mcp.resources.rag import _VocabSerializer

    row = {
        "name": "stromal",
        "description": None,
        "synonyms": ["stroma"],
        "rid": "1-VOBB",
    }
    md = _VocabSerializer().serialize("demo-schema.Tissue_Type", row)

    assert md is not None
    assert md.startswith("## Vocab Term: stromal")
    assert "Description" not in md
    assert "**Synonyms:** stroma" in md
    assert "**RID:** 1-VOBB" in md


def test_vocab_serializer_omits_empty_synonyms() -> None:
    """Term with empty synonyms list drops the line."""
    from deriva_ml_mcp.resources.rag import _VocabSerializer

    row = {
        "name": "muscle",
        "description": "Muscle tissue.",
        "synonyms": [],
        "rid": "1-VOCC",
    }
    md = _VocabSerializer().serialize("demo-schema.Tissue_Type", row)

    assert md is not None
    assert "Synonyms" not in md
    assert "**Description:** Muscle tissue." in md


def test_vocab_serializer_omits_both_empty() -> None:
    """Term with neither description nor synonyms renders header + RID only."""
    from deriva_ml_mcp.resources.rag import _VocabSerializer

    row = {
        "name": "bare",
        "description": None,
        "synonyms": None,
        "rid": "1-VODD",
    }
    md = _VocabSerializer().serialize("demo-schema.Tissue_Type", row)

    assert md is not None
    assert md.startswith("## Vocab Term: bare")
    assert "Description" not in md
    assert "Synonyms" not in md
    assert "**RID:** 1-VODD" in md


def test_vocab_serializer_returns_none_when_name_missing() -> None:
    """A row without a name is skipped (returns None)."""
    from deriva_ml_mcp.resources.rag import _VocabSerializer

    assert _VocabSerializer().serialize("schema.Vocab", {"description": "x"}) is None
    assert _VocabSerializer().serialize("schema.Vocab", {"name": ""}) is None


def test_vocab_serializer_accepts_capitalized_keys() -> None:
    """ERMrest-style ``Name``/``Synonyms``/``Description``/``RID`` keys also work."""
    from deriva_ml_mcp.resources.rag import _VocabSerializer

    row = {
        "Name": "foo",
        "Description": "Foo desc",
        "Synonyms": ["f"],
        "RID": "1-VOEE",
    }
    md = _VocabSerializer().serialize("schema.Vocab", row)

    assert md is not None
    assert "## Vocab Term: foo (schema.Vocab)" in md
    assert "**Description:** Foo desc" in md
    assert "**Synonyms:** f" in md
    assert "**RID:** 1-VOEE" in md


# ---------------------------------------------------------------------------
# 8. _index_vocabularies discovery / filter / failure-isolation / shared source
# ---------------------------------------------------------------------------


def _vocab_table(schema_name: str, table_name: str) -> MagicMock:
    """Build a MagicMock standing in for a deriva-py Table with schema/name."""
    t = MagicMock()
    t.name = table_name
    t.schema.name = schema_name
    return t


def _vocab_term(name: str, description: str | None, synonyms: list[str], rid: str) -> MagicMock:
    """Build a MagicMock standing in for a VocabularyTerm."""
    term = MagicMock()
    term.name = name
    term.description = description
    term.synonyms = synonyms
    term.rid = rid
    return term


def test_index_vocabularies_writes_one_source_per_vocab() -> None:
    """All discovered vocabs land under their own ``vocab:`` source."""
    from deriva_ml_mcp.resources import rag as rag_module

    vocabs = [
        _vocab_table("deriva-ml", "Dataset_Type"),
        _vocab_table("deriva-ml", "Workflow_Type"),
        _vocab_table("demo-schema", "Tissue_Type"),
    ]
    terms_by_table = {
        "deriva-ml.Dataset_Type": [_vocab_term("Training", "train", [], "1-T1")],
        "deriva-ml.Workflow_Type": [
            _vocab_term("Model_Training", None, ["MT"], "1-T2"),
            _vocab_term("Eval", "eval", [], "1-T3"),
        ],
        "demo-schema.Tissue_Type": [_vocab_term("epi", None, [], "1-T4")],
    }

    fake_ml = MagicMock()
    fake_ml.find_vocabularies.return_value = vocabs
    fake_ml.list_vocabulary_terms.side_effect = lambda t: terms_by_table[
        f"{t.schema.name}.{t.name}"
    ]
    fake_store = MagicMock()
    fake_store.delete_source = AsyncMock()
    fake_store.add = AsyncMock()

    with (
        patch.object(rag_module, "get_rag_store", return_value=fake_store),
        patch.object(rag_module, "get_ml", return_value=fake_ml),
    ):
        result = _run(rag_module._index_vocabularies("h.example", "1"))

    assert result == {
        "deriva-ml.Dataset_Type": 1,
        "deriva-ml.Workflow_Type": 2,
        "demo-schema.Tissue_Type": 1,
    }
    deleted_sources = [c.args[0] for c in fake_store.delete_source.call_args_list]
    assert "vocab:h.example:1:deriva-ml.Dataset_Type" in deleted_sources
    assert "vocab:h.example:1:deriva-ml.Workflow_Type" in deleted_sources
    assert "vocab:h.example:1:demo-schema.Tissue_Type" in deleted_sources

    # Every chunk written carries the vocab: prefix (catalog-public carve-out).
    for call in fake_store.add.call_args_list:
        chunks = call.args[0]
        for chunk in chunks:
            assert chunk.source.startswith("vocab:h.example:1:")


def test_index_vocabularies_filter_only_writes_requested_vocab() -> None:
    """``only_vocab='schema.X'`` indexes only that vocab; others are skipped."""
    from deriva_ml_mcp.resources import rag as rag_module

    vocabs = [
        _vocab_table("deriva-ml", "Dataset_Type"),
        _vocab_table("deriva-ml", "Workflow_Type"),
    ]
    fake_ml = MagicMock()
    fake_ml.find_vocabularies.return_value = vocabs
    fake_ml.list_vocabulary_terms.return_value = [_vocab_term("X", None, [], "1-T")]

    fake_store = MagicMock()
    fake_store.delete_source = AsyncMock()
    fake_store.add = AsyncMock()

    with (
        patch.object(rag_module, "get_rag_store", return_value=fake_store),
        patch.object(rag_module, "get_ml", return_value=fake_ml),
    ):
        result = _run(
            rag_module._index_vocabularies("h.example", "1", only_vocab="deriva-ml.Workflow_Type")
        )

    assert result == {"deriva-ml.Workflow_Type": 1}
    deleted_sources = [c.args[0] for c in fake_store.delete_source.call_args_list]
    assert deleted_sources == ["vocab:h.example:1:deriva-ml.Workflow_Type"]


def test_index_vocabularies_per_vocab_failures_are_isolated(caplog) -> None:
    """A fetch error on one vocab does not abort the whole pass."""
    from deriva_ml_mcp.resources import rag as rag_module

    bad = _vocab_table("deriva-ml", "Broken_Vocab")
    good = _vocab_table("deriva-ml", "Working_Vocab")
    fake_ml = MagicMock()
    fake_ml.find_vocabularies.return_value = [bad, good]

    def fake_terms(t):
        if t is bad:
            raise RuntimeError("boom")
        return [_vocab_term("ok", None, [], "1-OK")]

    fake_ml.list_vocabulary_terms.side_effect = fake_terms

    fake_store = MagicMock()
    fake_store.delete_source = AsyncMock()
    fake_store.add = AsyncMock()

    with (
        patch.object(rag_module, "get_rag_store", return_value=fake_store),
        patch.object(rag_module, "get_ml", return_value=fake_ml),
        caplog.at_level("ERROR", logger="deriva_ml_mcp.resources.rag"),
    ):
        result = _run(rag_module._index_vocabularies("h.example", "1"))

    assert result == {"deriva-ml.Working_Vocab": 1}
    assert any("Broken_Vocab" in record.message for record in caplog.records)


def test_index_vocabularies_short_circuits_when_store_none() -> None:
    """``get_rag_store() is None`` -> empty result, no ml call."""
    from deriva_ml_mcp.resources import rag as rag_module

    fake_ml = MagicMock()  # should not even be touched
    with (
        patch.object(rag_module, "get_rag_store", return_value=None),
        patch.object(rag_module, "get_ml", return_value=fake_ml),
    ):
        result = _run(rag_module._index_vocabularies("h.example", "1"))

    assert result == {}
    fake_ml.find_vocabularies.assert_not_called()


def test_index_vocabularies_uses_shared_source_across_users() -> None:
    """Two consecutive runs (different identities) write under the SAME source name.

    Vocabularies have no per-user ACL -- the source name is keyed only
    by (hostname, catalog_id, vocab qname). Two runs back-to-back must
    target the same ``vocab:...`` source regardless of caller, so the
    second run replaces the first (drain-then-add) instead of forking
    a per-user partition.
    """
    from deriva_ml_mcp.resources import rag as rag_module

    vocabs = [_vocab_table("deriva-ml", "Dataset_Type")]
    fake_ml = MagicMock()
    fake_ml.find_vocabularies.return_value = vocabs
    fake_ml.list_vocabulary_terms.return_value = [_vocab_term("Training", None, [], "1-T")]

    fake_store = MagicMock()
    fake_store.delete_source = AsyncMock()
    fake_store.add = AsyncMock()

    with (
        patch.object(rag_module, "get_rag_store", return_value=fake_store),
        patch.object(rag_module, "get_ml", return_value=fake_ml),
    ):
        _run(rag_module._index_vocabularies("h.example", "1"))
        _run(rag_module._index_vocabularies("h.example", "1"))

    deleted_sources = [c.args[0] for c in fake_store.delete_source.call_args_list]
    expected = "vocab:h.example:1:deriva-ml.Dataset_Type"
    assert deleted_sources == [expected, expected]


# ---------------------------------------------------------------------------
# 9. Vocab on_catalog_connect hook
# ---------------------------------------------------------------------------


def test_vocab_hook_calls_index_vocabularies_with_args() -> None:
    """The vocab hook forwards (hostname, catalog_id) to ``_index_vocabularies``."""
    from deriva_ml_mcp.resources import rag as rag_module

    fake = AsyncMock(return_value={})
    with patch.object(rag_module, "_index_vocabularies", new=fake):
        _run(_hook_at(_VOCAB_HOOK_IDX)("h.example", "1", "hash", {}))

    fake.assert_awaited_once_with("h.example", "1")


def test_vocab_hook_swallows_index_exception() -> None:
    """If _index_vocabularies raises, the hook logs and does not propagate."""
    from deriva_ml_mcp.resources import rag as rag_module

    fake = AsyncMock(side_effect=RuntimeError("boom"))
    with patch.object(rag_module, "_index_vocabularies", new=fake):
        # Must not raise.
        _run(_hook_at(_VOCAB_HOOK_IDX)("h.example", "1", "hash", {}))


# ---------------------------------------------------------------------------
# 10. _write_vocab_chunks delete-then-add semantics
# ---------------------------------------------------------------------------


def test_write_vocab_chunks_deletes_then_adds() -> None:
    """``delete_source`` is called first, then chunks are added via ``store.add``."""
    from deriva_ml_mcp.resources.rag import _VocabSerializer, _write_vocab_chunks

    fake_store = MagicMock()
    fake_store.delete_source = AsyncMock()
    fake_store.add = AsyncMock()
    terms = [
        {"name": "a", "description": "alpha", "synonyms": [], "rid": "1-A"},
        {"name": "b", "description": "beta", "synonyms": ["B"], "rid": "1-B"},
    ]
    count = _run(
        _write_vocab_chunks(
            fake_store,
            "vocab:h:1:s.t",
            "s.t",
            terms,
            _VocabSerializer(),
        )
    )

    assert count == 2
    fake_store.delete_source.assert_awaited_once_with("vocab:h:1:s.t")
    assert fake_store.add.await_count == 1
    chunks = fake_store.add.await_args.args[0]
    assert all(c.source == "vocab:h:1:s.t" for c in chunks)
    assert all(c.doc_type == "catalog-data" for c in chunks)


def test_write_vocab_chunks_empty_terms_drains_only() -> None:
    """An empty term list still drains the prior source but does not call add."""
    from deriva_ml_mcp.resources.rag import _VocabSerializer, _write_vocab_chunks

    fake_store = MagicMock()
    fake_store.delete_source = AsyncMock()
    fake_store.add = AsyncMock()
    count = _run(_write_vocab_chunks(fake_store, "vocab:h:1:s.t", "s.t", [], _VocabSerializer()))

    assert count == 0
    fake_store.delete_source.assert_awaited_once_with("vocab:h:1:s.t")
    fake_store.add.assert_not_called()
