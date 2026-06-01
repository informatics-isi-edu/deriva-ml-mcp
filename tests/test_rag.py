"""Unit tests for resources/rag.py (per-user RAG indexing + docs source).

Two surfaces under test:

1. ``register_rag_sources(ctx)`` -- declarative wiring: one
   ``rag_github_source`` declaration plus four ``on_catalog_connect``
   callbacks.

2. The serializers, per-user hook bodies, and v1.3 surgical re-index
   helpers -- rich Markdown rendering for Dataset / Workflow / Execution
   rows, fall-through to the generic serializer for unknown tables,
   per-row write isolation inside the first-connect hooks (one bad row
   doesn't poison the rest), and best-effort failure handling inside
   ``_reindex_<entity>`` (the catalog mutation already succeeded; a
   re-index hiccup must not propagate).

All tests are fully mocked. ``_write_row_chunk`` is patched in the
hook-flow tests so we can assert on per-RID source naming without
touching a real vector store; the row fetchers are patched to keep
the hook tests independent of the underlying ``_list_*_impl`` shape.
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

    from deriva_ml_mcp_plugin.resources import rag as rag_module
    from tests._helpers import _CapturingMCP

    plugin_ctx = PluginContext(_CapturingMCP())
    rag_module.register_rag_sources(plugin_ctx)
    return plugin_ctx._catalog_connect_hooks[idx]


@pytest.fixture()
def rag_ctx(ctx):
    """A fresh PluginContext with rag.register_rag_sources(ctx) applied."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    rag_module.register_rag_sources(ctx)
    return ctx


# ---------------------------------------------------------------------------
# 1. Declarative wiring -- register_rag_sources(ctx)
# ---------------------------------------------------------------------------


def test_register_rag_sources_declares_two_github_sources(rag_ctx) -> None:
    """register_rag_sources adds two GitHub documentation sources.

    v3.x added a second source ('deriva-ml-mcp-plugin-docs') so the
    deriva-ml-mcp-plugin repo's own README is searchable via rag_search
    alongside the deriva-ml library docs.
    """
    assert len(rag_ctx._rag_sources) == 2


def test_register_rag_sources_github_source_targets_deriva_ml_docs(rag_ctx) -> None:
    """The first GitHub source points at informatics-isi-edu/deriva-ml repo root.

    v3.x widened path_prefix from "docs/" to "" to also cover the
    top-level README.md and CHANGELOG.md (the docs/ subdirectory's
    contents are still indexed; the GitHub crawler filters to .md).
    """
    decl = next(s for s in rag_ctx._rag_sources if s.name == "deriva-ml-docs")
    assert decl.repo_owner == "informatics-isi-edu"
    assert decl.repo_name == "deriva-ml"
    assert decl.branch == "main"
    assert decl.path_prefix == ""
    assert decl.doc_type == "ml-docs"


def test_register_rag_sources_mcp_source_targets_deriva_ml_mcp_readme(rag_ctx) -> None:
    """The second GitHub source points at informatics-isi-edu/deriva-ml-mcp-plugin repo root."""
    decl = next(s for s in rag_ctx._rag_sources if s.name == "deriva-ml-mcp-plugin-docs")
    assert decl.repo_owner == "informatics-isi-edu"
    assert decl.repo_name == "deriva-ml-mcp-plugin"
    assert decl.branch == "main"
    assert decl.path_prefix == ""
    assert decl.doc_type == "ml-mcp-docs"


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
    from deriva_ml_mcp_plugin.resources.rag import _DatasetSerializer

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
    from deriva_ml_mcp_plugin.resources.rag import _DatasetSerializer

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
    from deriva_ml_mcp_plugin.resources.rag import _DatasetSerializer

    assert _DatasetSerializer().serialize("Workflow", {"rid": "x"}) is None
    assert _DatasetSerializer().serialize("Image", {"RID": "x"}) is None


# ---------------------------------------------------------------------------
# 3. _WorkflowSerializer
# ---------------------------------------------------------------------------


def test_workflow_serializer_renders_full_row() -> None:
    """A populated Workflow row renders all sections."""
    from deriva_ml_mcp_plugin.resources.rag import _WorkflowSerializer

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
    from deriva_ml_mcp_plugin.resources.rag import _WorkflowSerializer

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
    from deriva_ml_mcp_plugin.resources.rag import _WorkflowSerializer

    assert _WorkflowSerializer().serialize("Dataset", {"rid": "x"}) is None
    assert _WorkflowSerializer().serialize("Execution", {"rid": "x"}) is None


# ---------------------------------------------------------------------------
# 4. _ExecutionSerializer
# ---------------------------------------------------------------------------


def test_execution_serializer_renders_full_row() -> None:
    """A populated Execution row renders all sections (timestamps via str())."""
    from deriva_ml_mcp_plugin.resources.rag import _ExecutionSerializer

    row = {
        "rid": "1-EXAA",
        "workflow_rid": "1-WFAA",
        "status": "Stopped",
        "description": "Run #1.",
        "start_time": datetime(2026, 1, 1, 12, 0, 0),
        "stop_time": datetime(2026, 1, 1, 12, 30, 0),
        "duration": "0:30:00",
        "download_duration": "0:00:05",
        "upload_duration": "0:00:12",
    }
    md = _ExecutionSerializer().serialize("Execution", row)

    assert md is not None
    assert md.startswith("## Execution: 1-EXAA")
    assert "**Status:** Stopped" in md
    assert "**Workflow:** 1-WFAA" in md
    assert "**Description:** Run #1." in md
    assert "**Start Time:** 2026-01-01 12:00:00" in md
    assert "**Stop Time:** 2026-01-01 12:30:00" in md
    # Three duration phases each get their own line; the legacy
    # "Duration" label was renamed to "Execution Duration" in PR 2
    # (2026-05-19) for consistency with Download/Upload Duration.
    assert "**Execution Duration:** 0:30:00" in md
    assert "**Download Duration:** 0:00:05" in md
    assert "**Upload Duration:** 0:00:12" in md


def test_execution_serializer_omits_unset_timestamps() -> None:
    """Missing start/stop timestamps don't render their lines."""
    from deriva_ml_mcp_plugin.resources.rag import _ExecutionSerializer

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
    from deriva_ml_mcp_plugin.resources.rag import _ExecutionSerializer

    assert _ExecutionSerializer().serialize("Dataset", {"rid": "x"}) is None


# ---------------------------------------------------------------------------
# 5. Per-user hook flow -- index_table_data is called with resolved user_id
# ---------------------------------------------------------------------------


_USER_ID = "https://auth.example.org/sub/test-user-1"


def test_dataset_hook_writes_one_per_rid_source_per_row() -> None:
    """First-connect Dataset hook writes one per-RID source per row for the caller.

    Each row's chunk lands under
    ``data:{host}:{cat}:{user_id}:dataset:{rid}`` (the v1.3 surgical
    naming) so a later mutating tool can refresh that one source via
    ``_reindex_dataset`` without touching anything else.
    """
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    captured_calls: list[dict[str, Any]] = []
    rows = [{"rid": "1-DSAA", "name": "x"}, {"rid": "1-DSBB", "name": "y"}]

    async def fake_write_row_chunk(store, source, table_name, row, serializer):
        captured_calls.append(
            {
                "source": source,
                "table_name": table_name,
                "row_rid": row["rid"],
            }
        )
        return 1

    with (
        patch.object(rag_module, "_fetch_dataset_rows", return_value=rows),
        patch.object(rag_module, "resolve_user_identity", return_value=_USER_ID),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "_write_row_chunk", side_effect=fake_write_row_chunk),
    ):
        _run(_hook_at(_DATASET_HOOK_IDX)("h.example", "1", "hash", {}))

    assert len(captured_calls) == 2
    expected_sources = {
        f"data:h.example:1:{_USER_ID}:dataset:1-DSAA",
        f"data:h.example:1:{_USER_ID}:dataset:1-DSBB",
    }
    assert {c["source"] for c in captured_calls} == expected_sources
    assert all(c["table_name"] == "Dataset" for c in captured_calls)


def test_workflow_hook_writes_one_per_rid_source_per_row() -> None:
    """First-connect Workflow hook writes one per-RID source per row for the caller."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    captured_calls: list[dict[str, Any]] = []
    rows = [{"rid": "1-WFAA", "name": "y"}]

    async def fake_write_row_chunk(store, source, table_name, row, serializer):
        captured_calls.append({"source": source, "table_name": table_name})
        return 1

    with (
        patch.object(rag_module, "_fetch_workflow_rows", return_value=rows),
        patch.object(rag_module, "resolve_user_identity", return_value=_USER_ID),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "_write_row_chunk", side_effect=fake_write_row_chunk),
    ):
        _run(_hook_at(_WORKFLOW_HOOK_IDX)("h.example", "1", "hash", {}))

    assert captured_calls == [
        {
            "source": f"data:h.example:1:{_USER_ID}:workflow:1-WFAA",
            "table_name": "Workflow",
        }
    ]


def test_execution_hook_writes_one_per_rid_source_per_row() -> None:
    """First-connect Execution hook writes one per-RID source per row for the caller."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    captured_calls: list[dict[str, Any]] = []
    rows = [{"rid": "1-EXAA", "workflow_rid": "1-WFAA", "status": "Created"}]

    async def fake_write_row_chunk(store, source, table_name, row, serializer):
        captured_calls.append({"source": source, "table_name": table_name})
        return 1

    with (
        patch.object(rag_module, "_fetch_execution_rows", return_value=rows),
        patch.object(rag_module, "resolve_user_identity", return_value=_USER_ID),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "_write_row_chunk", side_effect=fake_write_row_chunk),
    ):
        _run(_hook_at(_EXECUTION_HOOK_IDX)("h.example", "1", "hash", {}))

    assert captured_calls == [
        {
            "source": f"data:h.example:1:{_USER_ID}:execution:1-EXAA",
            "table_name": "Execution",
        }
    ]


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
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_write = AsyncMock()
    with (
        patch.object(rag_module, "_fetch_dataset_rows", side_effect=RuntimeError("boom")),
        patch.object(rag_module, "resolve_user_identity", return_value=_USER_ID),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "_write_row_chunk", new=fake_write),
    ):
        # Must not raise
        _run(_hook_at(_DATASET_HOOK_IDX)("h.example", "1", "hash", {}))

    fake_write.assert_not_called()


def test_workflow_hook_isolates_per_row_write_failures(caplog) -> None:
    """If one row's write raises, the hook logs and continues with the rest.

    v1.3 contract: per-row writes are independently wrapped so one
    degenerate row cannot poison the whole user's first-connect index.
    The previous v1.0 contract (let ``index_table_data`` raise out)
    no longer applies: the hook now drives N writes, and a single bad
    row is the most likely failure shape.
    """
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    rows = [{"rid": "1-WFAA"}, {"rid": "1-WFBB"}, {"rid": "1-WFCC"}]
    write_calls: list[str] = []

    async def fake_write(store, source, table_name, row, serializer):
        write_calls.append(source)
        if row["rid"] == "1-WFBB":
            raise RuntimeError("boom on middle row")
        return 1

    with (
        patch.object(rag_module, "_fetch_workflow_rows", return_value=rows),
        patch.object(rag_module, "resolve_user_identity", return_value=_USER_ID),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "_write_row_chunk", side_effect=fake_write),
        caplog.at_level("ERROR", logger="deriva_ml_mcp_plugin.resources.rag"),
    ):
        # Must not raise -- the hook isolates the bad row.
        _run(_hook_at(_WORKFLOW_HOOK_IDX)("h.example", "1", "hash", {}))

    # All three rows attempted (the bad one in the middle did not abort the loop).
    assert len(write_calls) == 3
    assert any("1-WFBB" in record.message for record in caplog.records)


def test_hook_short_circuits_when_rag_store_is_none() -> None:
    """When RAG is disabled (get_rag_store -> None), the hook does nothing."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_write = AsyncMock()
    rows = [{"rid": "1-DSAA"}]
    with (
        patch.object(rag_module, "_fetch_dataset_rows", return_value=rows),
        patch.object(rag_module, "resolve_user_identity", return_value=_USER_ID),
        patch.object(rag_module, "get_rag_store", return_value=None),
        patch.object(rag_module, "_write_row_chunk", new=fake_write),
    ):
        _run(_hook_at(_DATASET_HOOK_IDX)("h.example", "1", "hash", {}))

    fake_write.assert_not_called()


def test_dataset_hook_skips_rows_without_rid() -> None:
    """Rows with empty/missing rid are skipped (cannot form a stable per-RID source)."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    rows = [{"rid": "1-DSAA"}, {"rid": "", "name": "broken"}, {"name": "no-rid-key"}]
    captured_sources: list[str] = []

    async def fake_write(store, source, table_name, row, serializer):
        captured_sources.append(source)
        return 1

    with (
        patch.object(rag_module, "_fetch_dataset_rows", return_value=rows),
        patch.object(rag_module, "resolve_user_identity", return_value=_USER_ID),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "_write_row_chunk", side_effect=fake_write),
    ):
        _run(_hook_at(_DATASET_HOOK_IDX)("h.example", "1", "hash", {}))

    assert captured_sources == [f"data:h.example:1:{_USER_ID}:dataset:1-DSAA"]


def test_dataset_hook_partitions_per_user() -> None:
    """Two consecutive calls under different identities partition by user_id.

    v1.3 update: the partition is now visible in the per-RID source
    name shape (``data:h:c:userA:dataset:RID`` vs
    ``data:h:c:userB:dataset:RID``). Same contract as v1.0 (user A's
    rows must not bleed into user B's index); the assertion just
    targets the new naming.
    """
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    rows = [{"rid": "1-DSAA", "name": "shared"}]
    captured: list[str] = []

    async def fake_write(store, source, table_name, row, serializer):
        captured.append(source)
        return 1

    with (
        patch.object(rag_module, "_fetch_dataset_rows", return_value=rows),
        patch.object(rag_module, "resolve_user_identity", side_effect=["userA", "userB"]),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "_write_row_chunk", side_effect=fake_write),
    ):
        # Build the hook inside the patch context so the factory's closure
        # captures the patched fetch fn rather than the real one.
        hook = _hook_at(_DATASET_HOOK_IDX)
        _run(hook("h.example", "1", "hash", {}))
        _run(hook("h.example", "1", "hash", {}))

    assert captured == [
        "data:h.example:1:userA:dataset:1-DSAA",
        "data:h.example:1:userB:dataset:1-DSAA",
    ]


# ---------------------------------------------------------------------------
# 7. _VocabSerializer
# ---------------------------------------------------------------------------


def test_vocab_serializer_renders_full_term() -> None:
    """A populated vocab term renders all sections + parent table in header."""
    from deriva_ml_mcp_plugin.resources.rag import _VocabSerializer

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
    from deriva_ml_mcp_plugin.resources.rag import _VocabSerializer

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
    from deriva_ml_mcp_plugin.resources.rag import _VocabSerializer

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
    from deriva_ml_mcp_plugin.resources.rag import _VocabSerializer

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
    from deriva_ml_mcp_plugin.resources.rag import _VocabSerializer

    assert _VocabSerializer().serialize("schema.Vocab", {"description": "x"}) is None
    assert _VocabSerializer().serialize("schema.Vocab", {"name": ""}) is None


def test_vocab_serializer_accepts_capitalized_keys() -> None:
    """ERMrest-style ``Name``/``Synonyms``/``Description``/``RID`` keys also work."""
    from deriva_ml_mcp_plugin.resources.rag import _VocabSerializer

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
    from deriva_ml_mcp_plugin.resources import rag as rag_module

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
    from deriva_ml_mcp_plugin.resources import rag as rag_module

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
    from deriva_ml_mcp_plugin.resources import rag as rag_module

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
        caplog.at_level("ERROR", logger="deriva_ml_mcp_plugin.resources.rag"),
    ):
        result = _run(rag_module._index_vocabularies("h.example", "1"))

    assert result == {"deriva-ml.Working_Vocab": 1}
    assert any("Broken_Vocab" in record.message for record in caplog.records)


def test_index_vocabularies_short_circuits_when_store_none() -> None:
    """``get_rag_store() is None`` -> empty result, no ml call."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

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
    from deriva_ml_mcp_plugin.resources import rag as rag_module

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
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake = AsyncMock(return_value={})
    with patch.object(rag_module, "_index_vocabularies", new=fake):
        _run(_hook_at(_VOCAB_HOOK_IDX)("h.example", "1", "hash", {}))

    fake.assert_awaited_once_with("h.example", "1")


def test_vocab_hook_swallows_index_exception() -> None:
    """If _index_vocabularies raises, the hook logs and does not propagate."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake = AsyncMock(side_effect=RuntimeError("boom"))
    with patch.object(rag_module, "_index_vocabularies", new=fake):
        # Must not raise.
        _run(_hook_at(_VOCAB_HOOK_IDX)("h.example", "1", "hash", {}))


# ---------------------------------------------------------------------------
# 10. _write_vocab_chunks delete-then-add semantics
# ---------------------------------------------------------------------------


def test_write_vocab_chunks_deletes_then_adds() -> None:
    """``delete_source`` is called first, then chunks are added via ``store.add``."""
    from deriva_ml_mcp_plugin.resources.rag import _VocabSerializer, _write_vocab_chunks

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
    from deriva_ml_mcp_plugin.resources.rag import _VocabSerializer, _write_vocab_chunks

    fake_store = MagicMock()
    fake_store.delete_source = AsyncMock()
    fake_store.add = AsyncMock()
    count = _run(_write_vocab_chunks(fake_store, "vocab:h:1:s.t", "s.t", [], _VocabSerializer()))

    assert count == 0
    fake_store.delete_source.assert_awaited_once_with("vocab:h:1:s.t")
    fake_store.add.assert_not_called()


# ---------------------------------------------------------------------------
# 11. v1.3 _row_source_name + _write_row_chunk
# ---------------------------------------------------------------------------


def test_row_source_name_concatenates_segments() -> None:
    """``_row_source_name`` produces ``data:{host}:{cat}:{user_id}:{table}:{rid}``."""
    from deriva_ml_mcp_plugin.resources.rag import _row_source_name

    assert (
        _row_source_name("h.example", "1", "userA", "dataset", "1-AAAA")
        == "data:h.example:1:userA:dataset:1-AAAA"
    )
    # Source name is a strict superset of the user-id-only prefix so
    # upstream's filter (which prefix-matches on data:host:cat:user_id:)
    # still gates correctly.
    src = _row_source_name("h.example", "1", "userA", "execution", "1-EXAA")
    assert src.startswith("data:h.example:1:userA:")


def test_write_row_chunk_deletes_then_adds() -> None:
    """``_write_row_chunk`` drains the source and writes the rendered chunks."""
    from deriva_ml_mcp_plugin.resources.rag import _DatasetSerializer, _write_row_chunk

    fake_store = MagicMock()
    fake_store.delete_source = AsyncMock()
    fake_store.add = AsyncMock()
    row = {
        "rid": "1-DSAA",
        "name": "x",
        "description": "demo",
        "dataset_types": ["Training"],
        "current_version": "1.0.0",
        "member_count": 10,
    }

    count = _run(
        _write_row_chunk(
            fake_store,
            "data:h:1:userA:dataset:1-DSAA",
            "Dataset",
            row,
            _DatasetSerializer(),
        )
    )

    assert count >= 1
    fake_store.delete_source.assert_awaited_once_with("data:h:1:userA:dataset:1-DSAA")
    assert fake_store.add.await_count == 1
    chunks = fake_store.add.await_args.args[0]
    assert all(c.source == "data:h:1:userA:dataset:1-DSAA" for c in chunks)
    assert all(c.doc_type == "catalog-data" for c in chunks)


def test_write_row_chunk_serializer_returns_none_drains_only() -> None:
    """A serializer that returns None still drains the prior source but writes nothing."""
    from deriva_ml_mcp_plugin.resources.rag import _DatasetSerializer, _write_row_chunk

    fake_store = MagicMock()
    fake_store.delete_source = AsyncMock()
    fake_store.add = AsyncMock()

    # Dataset serializer returns None for non-Dataset tables.
    count = _run(
        _write_row_chunk(
            fake_store,
            "data:h:1:userA:workflow:1-WFAA",
            "Workflow",
            {"rid": "1-WFAA"},
            _DatasetSerializer(),
        )
    )

    assert count == 0
    fake_store.delete_source.assert_awaited_once_with("data:h:1:userA:workflow:1-WFAA")
    fake_store.add.assert_not_called()


# ---------------------------------------------------------------------------
# 12. v1.3 _reindex_<entity> surgical helpers
# ---------------------------------------------------------------------------


def test_reindex_dataset_writes_one_per_rid_source() -> None:
    """``_reindex_dataset`` looks up the row + writes one source for the caller."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_ml = MagicMock()
    fake_ds = MagicMock()
    fake_ds.dataset_rid = "1-DSAA"
    fake_ds.description = "x"
    fake_ds.dataset_types = ["Training"]
    fake_ds.current_version = "1.0.0"
    fake_ml.lookup_dataset.return_value = fake_ds

    captured: dict[str, Any] = {}

    async def fake_write(store, source, table_name, row, serializer):
        captured["source"] = source
        captured["table_name"] = table_name
        captured["row_rid"] = row.get("rid")
        return 1

    with (
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "get_ml", return_value=fake_ml),
        patch.object(rag_module, "resolve_user_identity", return_value="userA"),
        patch.object(rag_module, "_write_row_chunk", side_effect=fake_write),
    ):
        n = _run(rag_module._reindex_dataset("h.example", "1", "1-DSAA"))

    assert n == 1
    assert captured["source"] == "data:h.example:1:userA:dataset:1-DSAA"
    assert captured["table_name"] == "Dataset"
    assert captured["row_rid"] == "1-DSAA"
    fake_ml.lookup_dataset.assert_called_once_with("1-DSAA")


def test_reindex_workflow_writes_one_per_rid_source() -> None:
    """``_reindex_workflow`` looks up the row + writes one source for the caller."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_ml = MagicMock()
    fake_wf = MagicMock()
    # ``Workflow.rid`` -> ``workflow_rid`` (deriva-ml #226, 2026-05-25).
    fake_wf.workflow_rid = "1-WFAA"
    fake_wf.name = "MyPipe"
    fake_wf.url = "https://example/repo"
    fake_wf.checksum = "abc"
    fake_wf.version = "1.0.0"
    fake_wf.workflow_type = ["Model_Training"]
    fake_wf.description = "demo"
    del fake_wf.rid
    fake_ml.lookup_workflow.return_value = fake_wf

    captured: dict[str, Any] = {}

    async def fake_write(store, source, table_name, row, serializer):
        captured["source"] = source
        captured["table_name"] = table_name
        return 1

    with (
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "get_ml", return_value=fake_ml),
        patch.object(rag_module, "resolve_user_identity", return_value="userA"),
        patch.object(rag_module, "_write_row_chunk", side_effect=fake_write),
    ):
        n = _run(rag_module._reindex_workflow("h.example", "1", "1-WFAA"))

    assert n == 1
    assert captured["source"] == "data:h.example:1:userA:workflow:1-WFAA"
    assert captured["table_name"] == "Workflow"


def test_reindex_execution_writes_one_per_rid_source() -> None:
    """``_reindex_execution`` looks up the row + writes one source for the caller."""
    from deriva_ml.execution.execution import ExecutionStatus

    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_ml = MagicMock()
    fake_record = MagicMock()
    fake_record.execution_rid = "1-EXAA"
    fake_record.workflow_rid = "1-WFAA"
    fake_record.status = ExecutionStatus.Created
    fake_record.description = "demo"
    fake_record.start_time = None
    fake_record.stop_time = None
    fake_record.duration = None
    fake_ml.lookup_execution.return_value = fake_record

    captured: dict[str, Any] = {}

    async def fake_write(store, source, table_name, row, serializer):
        captured["source"] = source
        captured["table_name"] = table_name
        return 1

    with (
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "get_ml", return_value=fake_ml),
        patch.object(rag_module, "resolve_user_identity", return_value="userA"),
        patch.object(rag_module, "_write_row_chunk", side_effect=fake_write),
    ):
        n = _run(rag_module._reindex_execution("h.example", "1", "1-EXAA"))

    assert n == 1
    assert captured["source"] == "data:h.example:1:userA:execution:1-EXAA"
    assert captured["table_name"] == "Execution"


def test_reindex_dataset_short_circuits_when_store_none() -> None:
    """``_reindex_dataset`` returns 0 (no ml call) when RAG is disabled."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_ml = MagicMock()  # must not be touched
    with (
        patch.object(rag_module, "get_rag_store", return_value=None),
        patch.object(rag_module, "get_ml", return_value=fake_ml),
    ):
        n = _run(rag_module._reindex_dataset("h.example", "1", "1-DSAA"))

    assert n == 0
    fake_ml.lookup_dataset.assert_not_called()


def test_reindex_dataset_swallows_lookup_failure(caplog) -> None:
    """A best-effort failure (lookup raise) returns 0 + logs without propagating."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_ml = MagicMock()
    fake_ml.lookup_dataset.side_effect = RuntimeError("not found")
    with (
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "get_ml", return_value=fake_ml),
        patch.object(rag_module, "resolve_user_identity", return_value="userA"),
        caplog.at_level("ERROR", logger="deriva_ml_mcp_plugin.resources.rag"),
    ):
        n = _run(rag_module._reindex_dataset("h.example", "1", "1-DSAA"))

    assert n == 0
    assert any("1-DSAA" in record.message for record in caplog.records)


def test_delete_dataset_source_drops_per_rid_source() -> None:
    """``_delete_dataset_source`` calls ``store.delete_source`` for the row's source."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_store = MagicMock()
    fake_store.delete_source = AsyncMock()
    with (
        patch.object(rag_module, "get_rag_store", return_value=fake_store),
        patch.object(rag_module, "resolve_user_identity", return_value="userA"),
    ):
        ok = _run(rag_module._delete_dataset_source("h.example", "1", "1-DSAA"))

    assert ok is True
    fake_store.delete_source.assert_awaited_once_with("data:h.example:1:userA:dataset:1-DSAA")


def test_delete_dataset_source_short_circuits_when_store_none() -> None:
    """``_delete_dataset_source`` returns False when RAG is disabled."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    with patch.object(rag_module, "get_rag_store", return_value=None):
        assert _run(rag_module._delete_dataset_source("h.example", "1", "1-DSAA")) is False


def test_delete_dataset_source_swallows_failure(caplog) -> None:
    """A delete failure returns False + logs without propagating."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_store = MagicMock()
    fake_store.delete_source = AsyncMock(side_effect=RuntimeError("boom"))
    with (
        patch.object(rag_module, "get_rag_store", return_value=fake_store),
        patch.object(rag_module, "resolve_user_identity", return_value="userA"),
        caplog.at_level("ERROR", logger="deriva_ml_mcp_plugin.resources.rag"),
    ):
        ok = _run(rag_module._delete_dataset_source("h.example", "1", "1-DSAA"))

    assert ok is False
    assert any("1-DSAA" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# 13. v1.4 _resync_user_sources orchestrator (cross-user freshness bridge)
# ---------------------------------------------------------------------------


def test_resync_user_sources_all_iterates_all_three_tables() -> None:
    """``target=None`` iterates dataset/workflow/execution row fetchers + reindexes each RID."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_reindex_ds = AsyncMock(return_value=1)
    fake_reindex_wf = AsyncMock(return_value=1)
    fake_reindex_ex = AsyncMock(return_value=1)
    with (
        patch.object(
            rag_module,
            "_fetch_dataset_rows",
            return_value=[{"rid": "1-DSAA"}, {"rid": "1-DSBB"}],
        ),
        patch.object(rag_module, "_fetch_workflow_rows", return_value=[{"rid": "1-WFAA"}]),
        patch.object(
            rag_module,
            "_fetch_execution_rows",
            return_value=[{"rid": "1-EXAA"}, {"rid": "1-EXBB"}, {"rid": "1-EXCC"}],
        ),
        patch.object(rag_module, "_reindex_dataset", new=fake_reindex_ds),
        patch.object(rag_module, "_reindex_workflow", new=fake_reindex_wf),
        patch.object(rag_module, "_reindex_execution", new=fake_reindex_ex),
    ):
        counts = _run(rag_module._resync_user_sources("h.example", "1"))

    assert counts == {"dataset": 2, "workflow": 1, "execution": 3}
    assert fake_reindex_ds.await_count == 2
    assert fake_reindex_wf.await_count == 1
    assert fake_reindex_ex.await_count == 3


def test_resync_user_sources_targeted_dispatches_to_one_helper() -> None:
    """``target="dataset:1-AAAA"`` calls _reindex_dataset only; others not touched."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_reindex_ds = AsyncMock(return_value=1)
    fake_reindex_wf = AsyncMock(return_value=1)
    fake_reindex_ex = AsyncMock(return_value=1)
    with (
        patch.object(rag_module, "_reindex_dataset", new=fake_reindex_ds),
        patch.object(rag_module, "_reindex_workflow", new=fake_reindex_wf),
        patch.object(rag_module, "_reindex_execution", new=fake_reindex_ex),
    ):
        counts = _run(rag_module._resync_user_sources("h.example", "1", target="dataset:1-AAAA"))

    fake_reindex_ds.assert_awaited_once_with("h.example", "1", "1-AAAA")
    fake_reindex_wf.assert_not_awaited()
    fake_reindex_ex.assert_not_awaited()
    assert counts == {"dataset": 1, "workflow": 0, "execution": 0}


def test_resync_user_sources_targeted_workflow_dispatches_correctly() -> None:
    """``target="workflow:..."`` routes to _reindex_workflow."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_reindex_wf = AsyncMock(return_value=1)
    with patch.object(rag_module, "_reindex_workflow", new=fake_reindex_wf):
        counts = _run(rag_module._resync_user_sources("h.example", "1", target="workflow:1-WFAA"))

    fake_reindex_wf.assert_awaited_once_with("h.example", "1", "1-WFAA")
    assert counts["workflow"] == 1


def test_resync_user_sources_targeted_execution_dispatches_correctly() -> None:
    """``target="execution:..."`` routes to _reindex_execution."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_reindex_ex = AsyncMock(return_value=1)
    with patch.object(rag_module, "_reindex_execution", new=fake_reindex_ex):
        counts = _run(rag_module._resync_user_sources("h.example", "1", target="execution:1-EXAA"))

    fake_reindex_ex.assert_awaited_once_with("h.example", "1", "1-EXAA")
    assert counts["execution"] == 1


def test_resync_user_sources_malformed_target_raises_valueerror() -> None:
    """Malformed ``target`` (no colon, or unknown table) raises ValueError."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    with pytest.raises(ValueError, match="<table>:<rid>"):
        _run(rag_module._resync_user_sources("h.example", "1", target="badtarget"))

    with pytest.raises(ValueError, match="dataset/workflow/execution"):
        _run(rag_module._resync_user_sources("h.example", "1", target="schema:1-AAAA"))


def test_resync_user_sources_per_rid_failure_is_isolated(caplog) -> None:
    """One row's reindex failure logs + continues; other rows still refresh."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    # Dataset 1-DSAA succeeds, 1-DSBB raises; both should be attempted.
    async def _fake_reindex_ds(host, cat, rid):
        if rid == "1-DSBB":
            raise RuntimeError("dataset boom")
        return 1

    with (
        patch.object(
            rag_module,
            "_fetch_dataset_rows",
            return_value=[{"rid": "1-DSAA"}, {"rid": "1-DSBB"}],
        ),
        patch.object(rag_module, "_fetch_workflow_rows", return_value=[]),
        patch.object(rag_module, "_fetch_execution_rows", return_value=[]),
        patch.object(rag_module, "_reindex_dataset", side_effect=_fake_reindex_ds),
        caplog.at_level("ERROR", logger="deriva_ml_mcp_plugin.resources.rag"),
    ):
        counts = _run(rag_module._resync_user_sources("h.example", "1"))

    # The successful RID still landed; the failed one didn't increment the count.
    assert counts == {"dataset": 1, "workflow": 0, "execution": 0}
    # The failure was logged with the failing RID identified.
    assert any("1-DSBB" in record.message for record in caplog.records)


def test_resync_user_sources_fetcher_failure_skips_table_but_continues_others(caplog) -> None:
    """If a row fetcher raises, that table reports zero but other tables still resync."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_reindex_ds = AsyncMock(return_value=1)
    fake_reindex_wf = AsyncMock(return_value=1)
    fake_reindex_ex = AsyncMock(return_value=1)
    with (
        patch.object(
            rag_module, "_fetch_dataset_rows", side_effect=RuntimeError("dataset fetch boom")
        ),
        patch.object(rag_module, "_fetch_workflow_rows", return_value=[{"rid": "1-WFAA"}]),
        patch.object(rag_module, "_fetch_execution_rows", return_value=[{"rid": "1-EXAA"}]),
        patch.object(rag_module, "_reindex_dataset", new=fake_reindex_ds),
        patch.object(rag_module, "_reindex_workflow", new=fake_reindex_wf),
        patch.object(rag_module, "_reindex_execution", new=fake_reindex_ex),
        caplog.at_level("ERROR", logger="deriva_ml_mcp_plugin.resources.rag"),
    ):
        counts = _run(rag_module._resync_user_sources("h.example", "1"))

    # Dataset table reports zero (fetcher raised; no rows to reindex).
    assert counts == {"dataset": 0, "workflow": 1, "execution": 1}
    # Dataset reindex was never attempted.
    fake_reindex_ds.assert_not_awaited()
    # Workflow + execution still ran.
    fake_reindex_wf.assert_awaited_once_with("h.example", "1", "1-WFAA")
    fake_reindex_ex.assert_awaited_once_with("h.example", "1", "1-EXAA")
    # The fetcher failure was logged with the dataset-table context.
    # (The "dataset fetch boom" RuntimeError text lives in the traceback
    # via logger.exception, not in record.message itself.)
    assert any("failed to enumerate datasets" in record.message for record in caplog.records)


def test_resync_user_sources_skips_rows_with_empty_rid() -> None:
    """A row dict with empty/missing 'rid' is silently skipped (no None RID in source name)."""
    from deriva_ml_mcp_plugin.resources import rag as rag_module

    fake_reindex_ds = AsyncMock(return_value=1)
    with (
        patch.object(
            rag_module,
            "_fetch_dataset_rows",
            return_value=[{"rid": "1-DSAA"}, {"rid": ""}, {"name": "no rid here"}],
        ),
        patch.object(rag_module, "_fetch_workflow_rows", return_value=[]),
        patch.object(rag_module, "_fetch_execution_rows", return_value=[]),
        patch.object(rag_module, "_reindex_dataset", new=fake_reindex_ds),
    ):
        counts = _run(rag_module._resync_user_sources("h.example", "1"))

    assert counts["dataset"] == 1
    fake_reindex_ds.assert_awaited_once_with("h.example", "1", "1-DSAA")
