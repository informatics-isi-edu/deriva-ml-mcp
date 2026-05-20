"""Per-user RAG indexing + vocab indexing + DerivaML docs source for deriva-ml-mcp.

Phase 6.3 wired three things into the RAG subsystem; v1.1 added a
fourth (vocabularies); v1.3 (this file) reshapes the per-user trio into
per-user-per-RID surgical indexers:

1. **One GitHub doc source** -- ``informatics-isi-edu/deriva-ml`` ``docs/``
   (declared synchronously via ``ctx.rag_github_source``). The indexer
   crawls it once when RAG is enabled.

2. **Three per-user-per-RID ``on_catalog_connect`` hooks** -- one each
   for ``Dataset``, ``Workflow``, and ``Execution`` rows. Each hook
   resolves the calling user's identity, fetches rows under that user's
   credential, and writes one source per (user, table, RID) tuple to
   the vector store under
   ``data:{host}:{cat}:{user_id}:{table}:{rid}``. The fetch path goes
   through the plugin's existing ``_list_*_impl`` helpers, so ACL
   behavior matches the read tools. The per-RID source naming lets
   mutating tools surgically refresh just the affected source via
   ``_reindex_<entity>(...)`` immediately after the catalog mutation,
   so ``rag_search`` finds the change on the very next call.

3. **One catalog-public ``on_catalog_connect`` hook for vocabularies**
   (v1.1) -- discovers all vocabulary tables via
   ``ml.find_vocabularies()`` (built-in ML vocabs + any user-defined
   domain vocabs) and writes each vocab's terms to the vector store
   under a custom source prefix
   ``vocab:{hostname}:{catalog_id}:{schema}.{table}``. The ``vocab:``
   prefix bypasses upstream's ``data:`` user-id filter so chunks are
   served to all users in the catalog -- vocabularies have no per-user
   ACL. Each term renders to markdown with name + description +
   synonyms + RID; the parent vocab table appears inline in the H2
   header so a search hit carries the disambiguating context.

4. **Four ``RowSerializer`` implementations** -- one per per-user table
   (Dataset, Workflow, Execution) and one for vocabulary terms
   (``_VocabSerializer``), each producing rich Markdown sections for
   the LLM (header, fields, subsections; empties omitted FaceBase-style).

Why per-user hooks (not ``ctx.rag_dataset_indexer``)?
``rag_dataset_indexer`` produces a single global enriched source shared
across all users (the enricher fires under whichever credential happens
to connect first). For ML data with per-user ACLs that would leak rows
across users. The per-user pattern here partitions the source name by
user identity and fetches under the calling user's credential -- same
posture as every other tool in this plugin.

Why direct store writes for the per-user trio (not ``index_table_data``)?
``index_table_data``'s source naming is hardcoded to
``data:{host}:{cat}:{user_id}`` (one source per user, all rows lumped
together). v1.3's surgical re-index needs one source per (user, table,
RID) so a single RID's chunk can be replaced without touching the rest
of the user's index. Direct ``store.delete_source + store.add`` per
row gives us that. Source-name shape stays under the ``data:`` prefix
so upstream's ``rag_search`` user-id filter continues to gate access.

Why direct store writes (not ``index_table_data``) for vocabularies?
``index_table_data``'s source naming is hardcoded to
``data:{host}:{cat}:{user_id}``, which would (a) be incorrectly
per-user-partitioned for public vocab data, and (b) get filtered out
by upstream's ``data:`` user-id filter for users whose ``user_id`` is
not the indexer's. The vocab path writes to the store directly with
its own ``vocab:`` prefix; the upstream filter only applies to
``schema:``, ``data:``, and ``enriched:`` prefixes (verified in
``deriva-mcp-core/src/deriva_mcp_core/rag/tools.py`` around lines
425-440), so ``vocab:`` chunks are served to all users.

Vocab freshness (v1.1):
    First catalog connect per server lifetime indexes all vocabs in
    bulk via the new on_catalog_connect hook. The
    ``deriva_ml_reindex_vocabularies`` tool re-runs the same
    discovery + per-vocab write -- use it after adding terms via
    core's ``add_term``, which doesn't fire any framework lifecycle
    hook (tracked upstream as deriva-mcp-core#3). v1.1 deliberately
    does NOT add an ``on_schema_change`` hook -- ``add_term`` doesn't
    fire it, and the cost/benefit is wrong for v1.1; the manual tool
    is the bridge until upstream ships ``on_data_change``.

Wiring into ``plugin.py`` is in ``register_rag_sources(ctx)``. This
module is import-safe and idempotent on its own.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from deriva_mcp_core.context import resolve_user_identity
from deriva_mcp_core.rag import get_rag_store
from deriva_mcp_core.rag.chunker import chunk_markdown
from deriva_mcp_core.rag.data import RowSerializer
from deriva_mcp_core.rag.store import Chunk

from deriva_ml_mcp._helpers import _MAX_LIMIT, _table_qname
from deriva_ml_mcp.ml_context import get_ml
from deriva_ml_mcp.tools.dataset import _list_datasets_impl, _summarize_dataset
from deriva_ml_mcp.tools.execution import _list_executions_impl, _summarize_execution
from deriva_ml_mcp.tools.workflow import _list_workflows_impl, _summarize_workflow

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext
    from deriva_mcp_core.rag.store import VectorStore

logger = logging.getLogger(__name__)


_GITHUB_DOCS_NAME = "deriva-ml-docs"
_GITHUB_DOCS_OWNER = "informatics-isi-edu"
_GITHUB_DOCS_REPO = "deriva-ml"
_GITHUB_DOCS_BRANCH = "main"
# v3.x widening: was "docs/" (excluded the top-level README.md and
# CHANGELOG.md). Repo-root indexing covers the missing top-level
# files. The GitHub crawler already filters to .md only, so non-doc
# files are skipped. CLAUDE.md (maintainer-only Claude Code
# instructions) is also indexed as mild noise -- pending the
# upstream exclude_paths=[...] addition to deriva-mcp-core's
# GitHubCrawler that would let us drop it cleanly.
_GITHUB_DOCS_PATH_PREFIX = ""
_GITHUB_DOCS_DOC_TYPE = "ml-docs"


# v3.x: index the deriva-ml-mcp repo's own top-level README so MCP
# server documentation is searchable via rag_search alongside the
# deriva-ml library docs. Same noise caveat as the deriva-ml source
# above -- repo-root indexing also picks up CLAUDE.md and the
# docs/scratch/*.md planning notes; the upstream exclude_paths=[...]
# addition would let us drop these cleanly.
_GITHUB_MCP_DOCS_NAME = "deriva-ml-mcp-docs"
_GITHUB_MCP_DOCS_OWNER = "informatics-isi-edu"
_GITHUB_MCP_DOCS_REPO = "deriva-ml-mcp"
_GITHUB_MCP_DOCS_BRANCH = "main"
_GITHUB_MCP_DOCS_PATH_PREFIX = ""
_GITHUB_MCP_DOCS_DOC_TYPE = "ml-mcp-docs"


_DATASET_TABLE = "Dataset"
_WORKFLOW_TABLE = "Workflow"
_EXECUTION_TABLE = "Execution"

# URL-safe slug used as the {table} routing key inside per-RID source
# names ``data:{host}:{cat}:{user_id}:{table}:{rid}``. Distinct from the
# ERMrest-cased catalog table names above so the source name stays a
# stable identifier even if a future table rename shifts the catalog
# label.
_DATASET_TOKEN = "dataset"
_WORKFLOW_TOKEN = "workflow"
_EXECUTION_TOKEN = "execution"


# ---------------------------------------------------------------------------
# Row-rendering helpers
# ---------------------------------------------------------------------------


def _kv_line(label: str, value: Any) -> str | None:
    """Render one ``**Label:** value`` line, or None if value is empty.

    Empties (None, ``""``, ``[]``, ``{}``) are omitted so the surrounding
    Markdown stays clean rather than littered with ``**Foo:** None`` rows.

    Args:
        label: Bolded label (no trailing colon needed).
        value: Any JSON-friendly value. Lists are joined with ``, ``.

    Returns:
        Formatted line, or ``None`` to signal the caller should skip it.

    Example:
        >>> _kv_line("Name", "ds-1")
        '**Name:** ds-1'
        >>> _kv_line("Types", ["a", "b"])
        '**Types:** a, b'
        >>> _kv_line("Empty", None) is None
        True
        >>> _kv_line("Empty", []) is None
        True
    """
    if value is None or value == "" or value == [] or value == {}:
        return None
    if isinstance(value, list):
        text = ", ".join(str(v) for v in value)
    else:
        text = str(value)
    return f"**{label}:** {text}"


def _render_block(header: str, lines: list[str | None]) -> str:
    """Assemble a Markdown block: header + non-empty body lines.

    Args:
        header: The ``## ...`` header line for the block.
        lines: A list of lines (any ``None`` entries are dropped).

    Returns:
        ``"<header>\\n\\n<line1>\\n<line2>..."``. If every body line was
        ``None``, returns just the header (with a trailing blank line).
    """
    body = [ln for ln in lines if ln]
    if body:
        return header + "\n\n" + "\n".join(body)
    return header + "\n"


# ---------------------------------------------------------------------------
# Row serializers
# ---------------------------------------------------------------------------


class _DatasetSerializer(RowSerializer):
    """Render a ``Dataset`` summary dict as Markdown.

    Returns ``None`` for any other table so the framework's generic
    serializer (``_generic_row_markdown``) can take over -- per
    contract, our serializers are domain-specific and the generic
    fallback handles unknown tables.
    """

    def serialize(self, table_name: str, row: dict) -> str | None:  # noqa: D401
        """Serialize one Dataset row, or return None for unrelated tables."""
        if table_name != _DATASET_TABLE:
            return None
        rid = row.get("rid", "")
        lines: list[str | None] = [
            _kv_line("Name", row.get("name")),
            _kv_line("Description", row.get("description")),
            _kv_line("Types", row.get("dataset_types")),
            _kv_line("Version", row.get("current_version")),
            _kv_line("Members", row.get("member_count")),
        ]
        return _render_block(f"## Dataset: {rid}", lines)


class _WorkflowSerializer(RowSerializer):
    """Render a ``Workflow`` summary dict as Markdown."""

    def serialize(self, table_name: str, row: dict) -> str | None:  # noqa: D401
        """Serialize one Workflow row, or return None for unrelated tables."""
        if table_name != _WORKFLOW_TABLE:
            return None
        rid = row.get("rid", "")
        lines: list[str | None] = [
            _kv_line("Name", row.get("name")),
            _kv_line("Type", row.get("workflow_type")),
            _kv_line("URL", row.get("url")),
            _kv_line("Checksum", row.get("checksum")),
            _kv_line("Version", row.get("version")),
            _kv_line("Description", row.get("description")),
        ]
        return _render_block(f"## Workflow: {rid}", lines)


class _ExecutionSerializer(RowSerializer):
    """Render an ``Execution`` summary dict as Markdown."""

    def serialize(self, table_name: str, row: dict) -> str | None:  # noqa: D401
        """Serialize one Execution row, or return None for unrelated tables."""
        if table_name != _EXECUTION_TABLE:
            return None
        rid = row.get("rid", "")
        lines: list[str | None] = [
            _kv_line("Status", row.get("status")),
            _kv_line("Workflow", row.get("workflow_rid")),
            _kv_line("Description", row.get("description")),
            # Datetime fields are passed through str() because
            # _list_executions_impl returns datetime objects unmodified.
            _kv_line(
                "Start Time",
                str(row["start_time"]) if row.get("start_time") is not None else None,
            ),
            _kv_line(
                "Stop Time",
                str(row["stop_time"]) if row.get("stop_time") is not None else None,
            ),
            _kv_line("Execution Duration", row.get("duration")),
            _kv_line("Download Duration", row.get("download_duration")),
            _kv_line("Upload Duration", row.get("upload_duration")),
        ]
        return _render_block(f"## Execution: {rid}", lines)


class _VocabSerializer(RowSerializer):
    """Serialize a vocabulary term as Markdown for RAG indexing.

    Used for all discovered vocabulary tables (built-in ML vocabs and
    user-defined domain vocabs). The parent vocab table name appears
    inline in the H2 header so a search hit carries enough context for
    the LLM to know which vocabulary the term belongs to.

    Contract:
        Returns rendered markdown if the row carries a ``"name"`` (or
        ``"Name"``) value; ``None`` otherwise. The ``table_name``
        argument is treated as an opaque identifier for the inline
        header -- typically the qualified ``schema.table`` form -- and
        does not gate dispatch the way the per-user serializers do.
    """

    def serialize(self, table_name: str, row: dict) -> str | None:  # noqa: D401
        """Serialize one vocabulary term, or return None when ``name`` is absent."""
        # Contract divergence vs _DatasetSerializer/_WorkflowSerializer/
        # _ExecutionSerializer: those gate on table_name to dispatch among
        # multiple known table types. This serializer accepts ANY vocab
        # table; dispatch is handled by the caller's discovery loop in
        # _index_vocabularies (which iterates ml.find_vocabularies()), not
        # by the serializer itself. table_name is only used for the inline
        # header so a search hit carries its parent vocab inline.
        name = row.get("name") or row.get("Name")
        if not name:
            return None
        synonyms = row.get("synonyms") or row.get("Synonyms")
        synonyms_list = list(synonyms) if synonyms else None
        lines: list[str | None] = [
            _kv_line("Description", row.get("description") or row.get("Description")),
            _kv_line("Synonyms", synonyms_list),
            _kv_line("RID", row.get("rid") or row.get("RID")),
        ]
        return _render_block(f"## Vocab Term: {name} ({table_name})", lines)


# ---------------------------------------------------------------------------
# Row fetchers
# ---------------------------------------------------------------------------


def _fetch_dataset_rows(hostname: str, catalog_id: str) -> list[dict[str, Any]]:
    """Fetch up to ``_MAX_LIMIT`` Dataset summary rows under the caller's credential.

    Wraps ``_list_datasets_impl`` and pulls the ``"datasets"`` array.
    Since v2.2 the helper returns a ``DatasetListResponse`` Pydantic
    instance; ``.model_dump(mode="json")`` produces JSON-serializable
    dicts (datetimes coerced) for the chunk serializer.
    """
    ml = get_ml(hostname, catalog_id)
    payload = _list_datasets_impl(ml, after_rid=None, limit=_MAX_LIMIT)
    return [d.model_dump(mode="json") for d in payload.datasets]


def _fetch_workflow_rows(hostname: str, catalog_id: str) -> list[dict[str, Any]]:
    """Fetch up to ``_MAX_LIMIT`` Workflow summary rows under the caller's credential."""
    ml = get_ml(hostname, catalog_id)
    payload = _list_workflows_impl(ml, after_rid=None, limit=_MAX_LIMIT)
    return [w.model_dump(mode="json") for w in payload.workflows]


def _fetch_execution_rows(hostname: str, catalog_id: str) -> list[dict[str, Any]]:
    """Fetch up to ``_MAX_LIMIT`` Execution summary rows under the caller's credential."""
    ml = get_ml(hostname, catalog_id)
    payload = _list_executions_impl(
        ml,
        workflow_rid=None,
        workflow_type=None,
        status=None,
        after_rid=None,
        limit=_MAX_LIMIT,
    )
    return [e.model_dump(mode="json") for e in payload.executions]


def _coerce_for_index(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-trip rows through JSON to drop non-serializable values.

    The execution summary contains ``datetime`` objects; the serializer
    renders them directly via ``str()`` but the round-trip leaves
    everything as plain JSON-compatible scalars before chunking, which
    keeps the per-row write path simple. Datasets and workflows pass
    through unchanged.
    """
    return [json.loads(json.dumps(r, default=str)) for r in rows]


# ---------------------------------------------------------------------------
# Per-RID source naming + single-row write (v1.3 surgical re-index path)
# ---------------------------------------------------------------------------


def _row_source_name(
    hostname: str,
    catalog_id: str,
    user_id: str,
    table_token: str,
    rid: str,
) -> str:
    """Build the canonical per-RID ``data:`` source name for one user-table-RID tuple.

    The trailing ``{table}:{rid}`` segment is what makes the per-user
    trio's source naming surgical: each row lands in its own source so
    a mutating tool can replace just that one source via
    ``store.delete_source + store.add`` without touching the rest of
    the user's index.

    The ``data:{host}:{cat}:{user_id}`` prefix is preserved (verbatim)
    so upstream's ``rag_search`` user-id filter (which accepts both the
    bulk form ``data:{host}:{cat}:{user_id}`` and prefix-match on
    ``data:{host}:{cat}:{user_id}:`` for the per-RID form) continues
    to gate access correctly.

    Args:
        hostname: Deriva hostname.
        catalog_id: Catalog ID or alias.
        user_id: Subject identifier (``sub`` claim) for the calling user.
        table_token: One of ``_DATASET_TOKEN`` / ``_WORKFLOW_TOKEN`` /
            ``_EXECUTION_TOKEN``. URL-safe slug, distinct from the
            ERMrest-cased catalog table name.
        rid: RID of the row (e.g. ``"1-AAAA"``).

    Returns:
        ``f"data:{hostname}:{catalog_id}:{user_id}:{table_token}:{rid}"``.

    Example:
        >>> _row_source_name("h", "1", "u", "dataset", "1-AAAA")
        'data:h:1:u:dataset:1-AAAA'
    """
    return f"data:{hostname}:{catalog_id}:{user_id}:{table_token}:{rid}"


async def _write_row_chunk(
    store: VectorStore,
    source: str,
    table_name: str,
    row: dict[str, Any],
    serializer: RowSerializer,
) -> int:
    """Render one row and replace the existing source with its chunks.

    Pattern: ``delete_source`` (idempotent -- drains any prior version
    of this RID's chunks if present); render the row via the serializer;
    chunk via ``chunk_markdown``; ``store.add`` the resulting chunks.
    Mirrors ``_write_vocab_chunks`` from v1.1 but for a single row.

    Args:
        store: An active ``VectorStore`` (call site checks for ``None``).
        source: Canonical per-RID source name from ``_row_source_name``.
        table_name: ERMrest-cased catalog table name. Passed through to
            the serializer for dispatch.
        row: A single summary-shape row dict (the form
            ``_summarize_dataset`` / ``_summarize_workflow`` /
            ``_summarize_execution`` produce).
        serializer: The ``RowSerializer`` for the row's table.

    Returns:
        Count of chunks actually written (typically 1 -- one summary
        renders to one chunk under the 800-token chunk budget). 0 if
        the serializer returned None for the row.
    """
    await store.delete_source(source)
    rendered = serializer.serialize(table_name, row)
    if rendered is None:
        return 0
    chunks: list[Chunk] = []
    chunk_index = 0
    # TODO(upstream-rag-doctype): tracked as deriva-mcp-core#2.
    # Once chunks can carry a domain-specific doc_type, switch to
    # "ml-dataset" / "ml-workflow" / "ml-execution" so
    # rag_search(doc_type=...) can distinguish these from generic
    # catalog-data chunks.
    for c in chunk_markdown(rendered, source=source, doc_type="catalog-data"):
        chunks.append(
            Chunk(
                text=c.text,
                source=source,
                doc_type="catalog-data",
                section_heading=c.section_heading,
                heading_hierarchy=c.heading_hierarchy,
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1
    if chunks:
        await store.add(chunks)
    return len(chunks)


# ---------------------------------------------------------------------------
# Surgical per-RID re-index entry points called inline from mutating tools
# ---------------------------------------------------------------------------


async def _reindex_dataset(hostname: str, catalog_id: str, dataset_rid: str) -> int:
    """Refresh just one Dataset row's chunk for the calling user.

    Best-effort: any failure (fetch, render, store I/O) is logged and
    swallowed -- a re-index hiccup must not propagate to the calling
    tool's success path (the catalog mutation already succeeded). On
    failure returns 0; the next first-connect for this user picks the
    row up via the bulk path.

    Args:
        hostname: Deriva hostname.
        catalog_id: Catalog ID or alias.
        dataset_rid: RID of the dataset to refresh.

    Returns:
        Count of chunks written (typically 1; 0 on best-effort failure
        or when RAG is disabled).
    """
    store = get_rag_store()
    if store is None:
        return 0
    try:
        # get_ml + lookup_dataset are synchronous deriva-py HTTP I/O;
        # offload so the event loop is not blocked during the surgical
        # re-index that runs immediately after a mutating tool.
        ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
        ds = await asyncio.to_thread(ml.lookup_dataset, dataset_rid)
        # _summarize_dataset returns a Pydantic ``DatasetSummary`` (since v2.2);
        # ``model_dump(mode="json")`` produces a JSON-serializable dict
        # (datetimes etc. coerced) -- equivalent to the old
        # ``json.loads(json.dumps(row, default=str))`` round-trip.
        row = _summarize_dataset(ds).model_dump(mode="json")
        user_id = await asyncio.to_thread(resolve_user_identity, hostname)
        source = _row_source_name(hostname, catalog_id, user_id, _DATASET_TOKEN, dataset_rid)
        return await _write_row_chunk(store, source, _DATASET_TABLE, row, _DatasetSerializer())
    except Exception:  # noqa: BLE001 -- best-effort cache refresh
        logger.exception(
            "deriva-ml RAG: surgical re-index failed for dataset %s in %s/%s",
            dataset_rid,
            hostname,
            catalog_id,
        )
        return 0


async def _reindex_workflow(hostname: str, catalog_id: str, workflow_rid: str) -> int:
    """Refresh just one Workflow row's chunk for the calling user.

    Best-effort; see ``_reindex_dataset`` for the failure-handling contract.
    """
    store = get_rag_store()
    if store is None:
        return 0
    try:
        # See _reindex_dataset for the asyncio.to_thread rationale.
        ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
        wf = await asyncio.to_thread(ml.lookup_workflow, workflow_rid)
        # See _reindex_dataset for the v2.2 ``model_dump(mode="json")`` rationale.
        row = _summarize_workflow(wf).model_dump(mode="json")
        user_id = await asyncio.to_thread(resolve_user_identity, hostname)
        source = _row_source_name(hostname, catalog_id, user_id, _WORKFLOW_TOKEN, workflow_rid)
        return await _write_row_chunk(store, source, _WORKFLOW_TABLE, row, _WorkflowSerializer())
    except Exception:  # noqa: BLE001 -- best-effort cache refresh
        logger.exception(
            "deriva-ml RAG: surgical re-index failed for workflow %s in %s/%s",
            workflow_rid,
            hostname,
            catalog_id,
        )
        return 0


async def _reindex_execution(hostname: str, catalog_id: str, execution_rid: str) -> int:
    """Refresh just one Execution row's chunk for the calling user.

    Best-effort; see ``_reindex_dataset`` for the failure-handling contract.
    """
    store = get_rag_store()
    if store is None:
        return 0
    try:
        # See _reindex_dataset for the asyncio.to_thread rationale.
        ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
        record = await asyncio.to_thread(ml.lookup_execution, execution_rid)
        # See _reindex_dataset for the v2.2 ``model_dump(mode="json")`` rationale.
        row = _summarize_execution(record).model_dump(mode="json")
        user_id = await asyncio.to_thread(resolve_user_identity, hostname)
        source = _row_source_name(hostname, catalog_id, user_id, _EXECUTION_TOKEN, execution_rid)
        return await _write_row_chunk(store, source, _EXECUTION_TABLE, row, _ExecutionSerializer())
    except Exception:  # noqa: BLE001 -- best-effort cache refresh
        logger.exception(
            "deriva-ml RAG: surgical re-index failed for execution %s in %s/%s",
            execution_rid,
            hostname,
            catalog_id,
        )
        return 0


async def _delete_dataset_source(hostname: str, catalog_id: str, dataset_rid: str) -> bool:
    """Drop a single Dataset's per-RID source from the calling user's index.

    Used by ``deriva_ml_delete_dataset`` -- the row is gone from the
    catalog, so its chunk should not survive in RAG either. Best-effort:
    a delete failure is logged and swallowed.

    Args:
        hostname: Deriva hostname.
        catalog_id: Catalog ID or alias.
        dataset_rid: RID of the dataset whose source should be dropped.

    Returns:
        True on success, False on failure or when RAG is disabled.
    """
    store = get_rag_store()
    if store is None:
        return False
    try:
        # See _reindex_dataset for the asyncio.to_thread rationale.
        user_id = await asyncio.to_thread(resolve_user_identity, hostname)
        source = _row_source_name(hostname, catalog_id, user_id, _DATASET_TOKEN, dataset_rid)
        await store.delete_source(source)
        return True
    except Exception:  # noqa: BLE001 -- best-effort cache refresh
        logger.exception(
            "deriva-ml RAG: failed to drop source for deleted dataset %s in %s/%s",
            dataset_rid,
            hostname,
            catalog_id,
        )
        return False


async def _resync_user_sources(
    hostname: str,
    catalog_id: str,
    target: str | None = None,
) -> dict[str, int]:
    """Refresh the calling user's per-user-trio RAG sources.

    Wrapping orchestrator for the v1.4 manual cross-user-resync tool.
    The v1.3 surgical re-index covers freshness for the calling user's
    OWN mutations, but mutations by OTHER users (or by non-MCP clients
    like Chaise) leave the calling user's per-user sources stale until
    they reconnect. This helper is the bridge: when called, it re-runs
    the same per-RID re-index loop that ``_make_hook`` runs at first
    connect, but on demand instead of on connect.

    Two modes:

    - ``target=None`` (default): refresh ALL of the calling user's
      per-user sources for this catalog. Iterates dataset / workflow
      / execution row fetchers, calls ``_reindex_<entity>`` per RID.
      The bigger hammer; LLM-friendly default.

    - ``target="<table>:<rid>"``: refresh just one source. ``<table>``
      is one of ``dataset`` / ``workflow`` / ``execution``; ``<rid>``
      is the row RID. For when the LLM (or a skill) knows exactly
      which RID needs refreshing.

    Best-effort throughout: per-RID failures are logged and swallowed
    (same contract as ``_make_hook``). The returned dict reports the
    count of sources successfully refreshed per table -- a partial
    success is normal (e.g. one RID raised; the rest landed).

    Args:
        hostname: Deriva hostname.
        catalog_id: Catalog ID or alias.
        target: Optional ``"<table>:<rid>"`` selector. ``None`` means
            refresh all of the calling user's per-user sources.

    Returns:
        Dict with keys ``dataset``, ``workflow``, ``execution`` mapping
        to the count of sources successfully refreshed in each table.
        For ``target="<table>:<rid>"`` mode, only the targeted table's
        key has a non-zero value.

    Raises:
        ValueError: If ``target`` is malformed (not ``"<table>:<rid>"``
            shape or ``<table>`` not in the supported set). Caller
            should wrap and surface as an error envelope.
    """
    counts: dict[str, int] = {"dataset": 0, "workflow": 0, "execution": 0}

    if target is not None:
        # Surgical mode: parse "table:rid" and dispatch to one helper.
        if ":" not in target:
            raise ValueError(
                f"target must be in '<table>:<rid>' form (e.g. 'dataset:1-AAAA'); got {target!r}"
            )
        table_token, rid = target.split(":", 1)
        if table_token == _DATASET_TOKEN:
            chunks = await _reindex_dataset(hostname, catalog_id, rid)
            if chunks > 0:
                counts["dataset"] = 1
        elif table_token == _WORKFLOW_TOKEN:
            chunks = await _reindex_workflow(hostname, catalog_id, rid)
            if chunks > 0:
                counts["workflow"] = 1
        elif table_token == _EXECUTION_TOKEN:
            chunks = await _reindex_execution(hostname, catalog_id, rid)
            if chunks > 0:
                counts["execution"] = 1
        else:
            raise ValueError(
                f"target table must be one of dataset/workflow/execution; got {table_token!r}"
            )
        return counts

    # All-of-user-sources mode: iterate each table's row fetcher and
    # re-index every visible RID for the calling user. Each table
    # routes through ``_resync_one_table``, which encapsulates the
    # two-level try/except (per-table fetcher + per-RID reindex).
    counts["dataset"] = await _resync_one_table(
        hostname, catalog_id, "dataset", _fetch_dataset_rows, _reindex_dataset
    )
    counts["workflow"] = await _resync_one_table(
        hostname, catalog_id, "workflow", _fetch_workflow_rows, _reindex_workflow
    )
    counts["execution"] = await _resync_one_table(
        hostname, catalog_id, "execution", _fetch_execution_rows, _reindex_execution
    )

    return counts


async def _resync_one_table(
    hostname: str,
    catalog_id: str,
    table_label: str,
    fetch_fn: Callable[[str, str], list[dict[str, Any]]],
    reindex_fn: Callable[[str, str, str], Awaitable[int]],
) -> int:
    """Refresh every visible RID in one of the per-user-trio tables.

    Two-level try/except discipline: the outer ``try`` isolates a
    per-table fetcher failure (one bad fetcher doesn't cancel the
    other tables); the inner ``try`` isolates a per-RID reindex
    failure (one bad row doesn't cancel the rest of the loop). Both
    layers are needed because the underlying ``_reindex_<entity>``
    helpers swallow internally -- but defense in depth says the
    orchestrator should still isolate in case an unexpected raise
    escapes (lazy import failure, coroutine-creation issue, etc).

    Empty/missing RIDs are silently skipped so a malformed row dict
    never produces a None-bearing source name.

    Args:
        hostname: Deriva hostname.
        catalog_id: Catalog ID or alias.
        table_label: Lowercase singular ("dataset" / "workflow" /
            "execution"). Used in log messages and matches the key in
            ``_resync_user_sources``'s returned counts dict.
        fetch_fn: One of the row fetchers (``_fetch_<table>_rows``).
            Called synchronously inside the outer try.
        reindex_fn: The matching ``_reindex_<entity>`` coroutine.
            Awaited per RID inside the inner try.

    Returns:
        Count of RIDs whose reindex returned at least one chunk.
        Failures (fetcher or per-RID) don't increment the count but
        do emit a log line; the function never raises.
    """
    try:
        # fetch_fn wraps synchronous deriva-py HTTP I/O; offload to a
        # worker thread so the event loop is not blocked while the
        # full table is enumerated.
        rows = await asyncio.to_thread(fetch_fn, hostname, catalog_id)
    except Exception:  # noqa: BLE001 -- best-effort, log + continue
        logger.exception(
            "deriva-ml RAG: failed to enumerate %ss for resync in %s/%s",
            table_label,
            hostname,
            catalog_id,
        )
        return 0
    count = 0
    for row in rows:
        rid = row.get("rid")
        if not rid:
            continue
        try:
            chunks = await reindex_fn(hostname, catalog_id, rid)
            if chunks > 0:
                count += 1
        except Exception:  # noqa: BLE001 -- best-effort, per-RID
            logger.exception(
                "deriva-ml RAG: resync failed for %s %s in %s/%s",
                table_label,
                rid,
                hostname,
                catalog_id,
            )
    return count


# ---------------------------------------------------------------------------
# Hook factory (first-connect bulk index, per-RID source naming)
# ---------------------------------------------------------------------------


def _make_hook(
    fetch_fn: Callable[[str, str], list[dict[str, Any]]],
    table_name: str,
    table_token: str,
    serializer: RowSerializer,
) -> Callable[[str, str, str, dict], Awaitable[None]]:
    """Build an ``on_catalog_connect`` hook that indexes ``table_name`` per-user-per-RID.

    The returned coroutine fetches rows under the caller's credential,
    resolves the calling user's identity, and writes one source per
    (user, table_token, RID) tuple to the vector store. Same total
    chunk count as v1.0/v1.1's bulk pattern; the difference is that
    they land in N sources instead of 1, so individual rows can later
    be replaced surgically by ``_reindex_<entity>``.

    Behavior contract:

    - Wraps the fetch in ``try/except`` so a deriva-ml read failure is
      logged with table context but does not propagate.
    - Skips silently (debug log) when ``get_rag_store()`` returns
      ``None`` -- i.e. when RAG is disabled in this deployment.
    - Per-row write failures are isolated: a bad row is logged and the
      loop continues with the rest, so one degenerate row does not
      poison the whole user's first-connect index pass.
    - Logs a debug success line including row count + resolved user_id
      so an operator can investigate "why isn't user X's data showing up?"
      by flipping on debug logging without overwhelming production logs.

    Args:
        fetch_fn: Callable that takes ``(hostname, catalog_id)`` and
            returns summary-shape row dicts.
        table_name: Catalog table the rows belong to (ERMrest case).
            Used as the serializer dispatch key.
        table_token: URL-safe slug for the source-name routing segment
            (one of ``_DATASET_TOKEN`` / ``_WORKFLOW_TOKEN`` /
            ``_EXECUTION_TOKEN``).
        serializer: The ``RowSerializer`` to render each row.

    Returns:
        An async hook with the ``on_catalog_connect`` signature
        ``(hostname, catalog_id, schema_hash, schema_json) -> None``.
    """

    # schema_hash + schema_json are received because deriva-mcp-core's
    # _dispatch_catalog_connect signature requires them. We deliberately
    # ignore both: indexing depends only on (host, catalog_id, user_id,
    # rows), and re-indexing on schema changes is handled separately by
    # the framework's TTL gating, not by the schema_hash here.
    async def hook(
        hostname: str,
        catalog_id: str,
        schema_hash: str,  # noqa: ARG001 -- hook signature requires it
        schema_json: dict,  # noqa: ARG001 -- hook signature requires it
    ) -> None:
        try:
            # fetch_fn wraps synchronous deriva-py HTTP I/O; run it in a
            # worker thread so the event loop stays responsive while the
            # catalog is being indexed.
            raw_rows = await asyncio.to_thread(fetch_fn, hostname, catalog_id)
            rows = _coerce_for_index(raw_rows)
        except Exception:  # noqa: BLE001 -- fetch errors are domain-specific
            logger.exception(
                "deriva-ml RAG: failed to fetch %s rows for %s/%s",
                table_name,
                hostname,
                catalog_id,
            )
            return
        store = get_rag_store()
        if store is None:
            logger.debug("rag store unavailable, skipping %s first-connect index", table_name)
            return
        # resolve_user_identity may issue a sync GET /authn/session in
        # stdio mode; offload so the loop is not blocked on first call.
        user_id = await asyncio.to_thread(resolve_user_identity, hostname)
        for row in rows:
            rid = row.get("rid")
            if not rid:
                # Defensive: a row without a RID can't get a stable
                # per-RID source name; skip with a debug log rather
                # than silently dropping it under an empty-RID source.
                logger.debug(
                    "skipping %s first-connect row with empty rid: %r",
                    table_name,
                    row,
                )
                continue
            source = _row_source_name(hostname, catalog_id, user_id, table_token, rid)
            try:
                await _write_row_chunk(store, source, table_name, row, serializer)
            except Exception:  # noqa: BLE001 -- one bad row should not poison the pass
                logger.exception(
                    "deriva-ml RAG: failed to first-connect index %s row %s",
                    table_name,
                    rid,
                )
                continue
        logger.debug(
            "first-connect indexed %d %s rows for user=%s host=%s catalog=%s",
            len(rows),
            table_name,
            user_id,
            hostname,
            catalog_id,
        )

    return hook


# ---------------------------------------------------------------------------
# Vocabulary indexing (catalog-public, custom ``vocab:`` source prefix)
# ---------------------------------------------------------------------------


def _vocab_source_name(hostname: str, catalog_id: str, qname: str) -> str:
    """Build the canonical ``vocab:`` source name for one vocab table.

    The ``vocab:`` prefix is a deliberate carve-out from upstream's
    ``data:`` user-id filter (see module docstring) -- chunks under
    this prefix are returned to all users in the catalog, regardless
    of which user's connect triggered the index pass. This is correct
    for vocabularies, which carry no per-user ACL.
    """
    return f"vocab:{hostname}:{catalog_id}:{qname}"


async def _write_vocab_chunks(
    store: VectorStore,
    source: str,
    vocab_table: str,
    terms: list[dict[str, Any]],
    serializer: _VocabSerializer,
) -> int:
    """Render terms, chunk them, and replace the existing vocab source.

    The delete-then-add pattern mirrors what ``index_table_data`` does
    internally. Idempotent: a prior version of this source is drained
    before the new chunks land, so re-running on an unchanged vocab is
    a no-op from the caller's perspective (the chunks are equivalent).

    Args:
        store: An active ``VectorStore`` (call site checks for ``None``).
        source: Canonical source name from ``_vocab_source_name``.
        vocab_table: Qualified ``schema.table`` -- passed to the
            serializer as the inline header context.
        terms: List of term dicts (``name``, ``description``, ``synonyms``,
            ``rid``). Empty list short-circuits without touching the store.
        serializer: A ``_VocabSerializer`` instance.

    Returns:
        Count of indexed terms (the rendered, chunked count -- one term
        typically yields one chunk under our 800-token budget).
    """
    await store.delete_source(source)
    if not terms:
        return 0
    chunks: list[Chunk] = []
    chunk_index = 0
    for term in terms:
        rendered = serializer.serialize(vocab_table, term)
        if rendered is None:
            continue
        # TODO(upstream-rag-doctype): tracked as deriva-mcp-core#2
        # (https://github.com/informatics-isi-edu/deriva-mcp-core/issues/2).
        # Once chunks can carry a domain-specific doc_type, switch to
        # "ml-vocab" so rag_search(doc_type=...) can distinguish vocab
        # term chunks from generic catalog-data chunks.
        for c in chunk_markdown(rendered, source=source, doc_type="catalog-data"):
            chunks.append(
                Chunk(
                    text=c.text,
                    source=source,
                    doc_type="catalog-data",
                    section_heading=c.section_heading,
                    heading_hierarchy=c.heading_hierarchy,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
    if chunks:
        await store.add(chunks)
    return len(terms)


async def _index_vocabularies(
    hostname: str,
    catalog_id: str,
    only_vocab: str | None = None,
) -> dict[str, int]:
    """Discover and index vocabulary tables for the catalog.

    Iterates ``ml.find_vocabularies()`` (built-in ML vocabs +
    user-defined domain vocabs in any schema) and writes one shared
    source per vocab table at
    ``vocab:{hostname}:{catalog_id}:{schema}.{table}``. The custom
    ``vocab:`` prefix bypasses upstream's ``data:`` user-id filter so
    chunks are served to all users in the catalog (vocabularies carry
    no per-user ACL).

    Best-effort: per-vocab failures (missing vocab table, fetch error)
    are logged and skipped without aborting the whole index pass.

    Args:
        hostname: Deriva hostname.
        catalog_id: Catalog ID.
        only_vocab: If provided, only re-index this vocab (qualified
            ``"schema.table"`` form). If ``None``, re-index all
            discovered vocabs.

    Returns:
        Dict mapping vocab qname to indexed-term count for each vocab
        that was successfully indexed. Vocabs that errored or were
        filtered out by ``only_vocab`` do not appear.
    """
    store = get_rag_store()
    if store is None:
        logger.debug("rag store unavailable, skipping vocab index")
        return {}

    # get_ml() and find_vocabularies() both perform synchronous deriva-py
    # HTTP calls; offload so the event loop is not blocked while the
    # vocabulary catalog is enumerated.
    ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
    serializer = _VocabSerializer()
    indexed: dict[str, int] = {}

    vocab_tables = await asyncio.to_thread(ml.find_vocabularies)
    for vocab_table in vocab_tables:
        qname = _table_qname(vocab_table)
        if only_vocab is not None and qname != only_vocab:
            continue
        try:
            # Per-vocab term fetch is a sync HTTP round-trip in deriva-py.
            terms = await asyncio.to_thread(ml.list_vocabulary_terms, vocab_table)
            term_dicts = [
                {
                    "name": t.name,
                    "description": t.description,
                    "synonyms": list(t.synonyms or []),
                    "rid": t.rid,
                }
                for t in terms
            ]
            source = _vocab_source_name(hostname, catalog_id, qname)
            count = await _write_vocab_chunks(store, source, qname, term_dicts, serializer)
            indexed[qname] = count
            logger.debug("indexed %d terms for vocab %s", count, qname)
        except Exception:
            logger.exception("failed to index vocab %s", qname)
            continue

    return indexed


def _make_vocab_hook() -> Callable[[str, str, str, dict], Awaitable[None]]:
    """Build the on_catalog_connect hook that bulk-indexes vocabularies.

    Distinct from ``_make_hook`` because vocab indexing has a different
    shape: one hook fires N writes (one per discovered vocab) rather
    than one write. Forcing this into the per-user factory's
    ``(fetch_fn, table_name, serializer)`` signature would obscure the
    difference. Two clear factories beat one general one.

    Returns:
        An async hook with the ``on_catalog_connect`` signature
        ``(hostname, catalog_id, schema_hash, schema_json) -> None``.
    """

    async def hook(
        hostname: str,
        catalog_id: str,
        schema_hash: str,  # noqa: ARG001 -- hook signature requires it
        schema_json: dict,  # noqa: ARG001 -- hook signature requires it
    ) -> None:
        try:
            await _index_vocabularies(hostname, catalog_id)
        except Exception:  # noqa: BLE001 -- vocab discovery is best-effort
            logger.exception(
                "deriva-ml RAG: vocab index pass failed for %s/%s",
                hostname,
                catalog_id,
            )

    return hook


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def register_rag_sources(ctx: PluginContext) -> None:
    """Register the DerivaML docs source and four catalog-connect hooks.

    Hooks fire in registration order on every catalog connect:

    1. Dataset (per-user-per-RID, ``data:{host}:{cat}:{user_id}:dataset:{rid}``)
    2. Workflow (per-user-per-RID, ``data:{host}:{cat}:{user_id}:workflow:{rid}``)
    3. Execution (per-user-per-RID, ``data:{host}:{cat}:{user_id}:execution:{rid}``)
    4. Vocabularies (catalog-public, custom ``vocab:`` source prefix --
       v1.1; bypasses upstream's ``data:`` user-id filter, served to
       all users in the catalog)

    The first three are first-connect bulk indexers but write under
    per-RID source names so mutating tools can later replace just one
    affected source via the ``_reindex_<entity>`` helpers.

    Called from ``plugin.register(ctx)``. Safe to call without any RAG
    config -- ``ctx.rag_github_source`` is a no-op when RAG is
    disabled, and the hooks are best-effort: they swallow fetch
    exceptions and short-circuit when the vector store is unavailable.

    Args:
        ctx: PluginContext supplied by deriva-mcp-core at startup.

    Returns:
        None.

    Example:
        >>> from deriva_mcp_core.plugin.api import PluginContext
        >>> # ctx provided by the framework; not constructed by user code
        >>> register_rag_sources(ctx)  # doctest: +SKIP
    """
    ctx.rag_github_source(
        name=_GITHUB_DOCS_NAME,
        repo_owner=_GITHUB_DOCS_OWNER,
        repo_name=_GITHUB_DOCS_REPO,
        branch=_GITHUB_DOCS_BRANCH,
        path_prefix=_GITHUB_DOCS_PATH_PREFIX,
        doc_type=_GITHUB_DOCS_DOC_TYPE,
    )
    ctx.rag_github_source(
        name=_GITHUB_MCP_DOCS_NAME,
        repo_owner=_GITHUB_MCP_DOCS_OWNER,
        repo_name=_GITHUB_MCP_DOCS_REPO,
        branch=_GITHUB_MCP_DOCS_BRANCH,
        path_prefix=_GITHUB_MCP_DOCS_PATH_PREFIX,
        doc_type=_GITHUB_MCP_DOCS_DOC_TYPE,
    )
    ctx.on_catalog_connect(
        _make_hook(_fetch_dataset_rows, _DATASET_TABLE, _DATASET_TOKEN, _DatasetSerializer())
    )
    ctx.on_catalog_connect(
        _make_hook(_fetch_workflow_rows, _WORKFLOW_TABLE, _WORKFLOW_TOKEN, _WorkflowSerializer())
    )
    ctx.on_catalog_connect(
        _make_hook(
            _fetch_execution_rows, _EXECUTION_TABLE, _EXECUTION_TOKEN, _ExecutionSerializer()
        )
    )
    ctx.on_catalog_connect(_make_vocab_hook())
