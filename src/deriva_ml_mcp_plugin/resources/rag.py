"""Per-user RAG indexing + vocab indexing + DerivaML docs source for deriva-ml-mcp-plugin.

Phase 6.3 wired three things into the RAG subsystem; v1.1 added a
fourth (vocabularies); v1.3 reshaped the per-user trio into
per-user-per-RID surgical indexers; v1.5 (this file) drops the
on-connect bulk pass for those three tables in favor of read-through
(index-on-find) warming:

1. **One GitHub doc source** -- ``informatics-isi-edu/deriva-ml`` ``docs/``
   (declared synchronously via ``ctx.rag_github_source``). The indexer
   crawls it once when RAG is enabled.

2. **Read-through (index-on-find) per-user-per-RID writers** -- one
   logical path each for ``Dataset``, ``Workflow``, and ``Execution``
   rows. These are NO LONGER ``on_catalog_connect`` hooks (the three
   bulk first-connect passes were removed in v1.5 because they re-fetched
   and re-embedded every visible row on every catalog connect). Instead a
   row's chunk is warmed lazily: the list/get tools call
   ``_index_rows_on_find(...)`` after a read to schedule a best-effort
   background ``_reindex_<entity>(...)`` for each row just seen, and the
   mutating tools call ``_reindex_<entity>(...)`` surgically right after
   the catalog mutation. Both write one source per (user, table, RID)
   tuple to the vector store under
   ``data:{host}:{cat}:{user_id}:{table}:{rid}``. The fetch path goes
   through the plugin's existing ``_list_*_impl`` / ``lookup_*`` helpers,
   so ACL behavior matches the read tools, and the per-RID source naming
   lets a single row be refreshed without touching the rest of the
   user's index, so ``rag_search`` finds the change on the very next
   call. The ``deriva_ml_resync_indexes`` tool re-runs the same per-RID
   re-index loop over every visible row on demand (manual "warm
   everything" backfill via ``_resync_*`` + ``_fetch_*_rows``) -- the
   replacement for the removed first-connect bulk pass.

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

Why per-user writers (not ``ctx.rag_dataset_indexer``)?
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

First-connect lag (no startup prewarm):
    The vocab ``on_catalog_connect`` hook fires lazily on the first tool
    call that connects to a given catalog, so the first user to touch a
    catalog after a (re)start pays the vocab indexing cost inline. The
    per-user trio no longer indexes on connect at all (v1.5 removed the
    three bulk first-connect passes); their per-RID chunks are warmed
    read-through by the list/get tools via ``_index_rows_on_find`` and
    surgically on mutate via ``_reindex_<entity>``, so there is no
    per-user bulk cost on connect. The per-server GitHub doc sources ARE
    pre-warmed at startup (core's ``_startup_update``), but there is no
    startup prewarm for per-catalog indexing. The catalog-public parts
    (core's schema index + this plugin's vocab hook) COULD be primed at
    startup from an operator-configured catalog list; the per-user trio
    cannot (no calling-user credential exists at startup -- correct by
    design). Tracked upstream as deriva-mcp-core#10 (startup prewarm
    for catalog-public per-catalog indexing). Workaround until then:
    call ``deriva_ml_reindex_vocabularies(hostname, catalog_id)`` once
    per catalog after a restart to remove the vocab portion of the
    first-connect lag, and ``deriva_ml_resync_indexes(hostname,
    catalog_id)`` to warm the calling user's per-user trio in bulk.

Wiring into ``plugin.py`` is in ``register_rag_sources(ctx)``. This
module is import-safe and idempotent on its own.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from deriva_mcp_core.context import resolve_user_identity
from deriva_mcp_core.rag import get_rag_store
from deriva_mcp_core.rag.chunker import chunk_markdown
from deriva_mcp_core.rag.data import RowSerializer
from deriva_mcp_core.rag.store import Chunk

from deriva_ml_mcp_plugin._helpers import _MAX_LIMIT, _table_qname
from deriva_ml_mcp_plugin.ml_context import get_ml
from deriva_ml_mcp_plugin.tools.dataset import _list_datasets_impl, _summarize_dataset
from deriva_ml_mcp_plugin.tools.execution import _list_executions_impl, _summarize_execution
from deriva_ml_mcp_plugin.tools.workflow import _list_workflows_impl, _summarize_workflow

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext
    from deriva_mcp_core.rag.store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# First-connect guard (cold-start regression fix, 2026-06-04 bug report)
# ---------------------------------------------------------------------------
#
# The vocab ``on_catalog_connect`` hook fires as a background task
# (deriva-mcp-core's ``_dispatch_catalog_connect`` schedules one
# ``asyncio.create_task`` per hook). A single user's bootstrapping sequence
# fires several catalog-touching tool calls (``get_catalog_info``, then two
# ``get_schema`` calls), each independently triggering ``on_catalog_connect``.
# In ``stateless_http`` mode core's per-``(host, catalog, user)`` connect dedup
# does not reliably hold across them: it re-fires sequentially when a prior
# schema-fetch errored and discarded the dedup key, and it re-fires concurrently
# when calls interleave at ``get_catalog()`` before any adds the key. Without a
# guard the vocab pass ran on EVERY connect, re-fetching and re-embedding every
# vocab. The ICD10_Eye vocab (1209 terms) re-embedded each time; the passes
# fought the GIL with ``rag_search``'s own query embedding -- 60-261s first-turn
# latency, 97% CPU.
#
# The fix (per the bug report) is a TWO-LAYER guard, matching the documented
# "first catalog connect per server lifetime indexes all vocabs in bulk" intent
# that the code previously failed to enforce:
#
#   1. A per-server-lifetime "already indexed" SET. Once a work unit has been
#      indexed it is NOT re-indexed on any later connect until server restart.
#      Cleared only by an explicit re-index tool (see _clear_* below) or restart.
#   2. A per-work-unit ``asyncio.Lock`` with DOUBLE-CHECKED membership.
#      Concurrent firings serialize on the lock; the winner runs the pass and
#      records the unit in the set; waiters re-check the set after acquiring the
#      lock and skip. The check + ``async with`` is race-free under asyncio
#      cooperative scheduling -- no context switch between them.
#
# The guard lives on the vocab HOOK only, never on ``_index_vocabularies``
# directly: that backs the explicit ``deriva_ml_reindex_vocabularies`` tool,
# which is a deliberate user request that must always run. That tool instead
# CLEARS the relevant guard entry (``_clear_vocab_indexed``) so a subsequent
# connect re-indexes too.
#
# A FAILED pass is not recorded in the set, so a transient catalog error stays
# retryable on the next connect rather than stranding the catalog until restart.
#
# Work-unit key:
#   vocab pass -> ``(host, catalog)``
_indexed_vocab_catalogs: set[tuple[str, str]] = set()
_vocab_connect_locks: dict[tuple[str, str], asyncio.Lock] = {}


def _reset_index_state() -> None:
    """Drop all first-connect guard state (set + locks). Test-only isolation hook.

    The guard registries are module-global and per-server-lifetime in
    production; tests call this to start each scenario from a clean slate.
    """
    _indexed_vocab_catalogs.clear()
    _vocab_connect_locks.clear()


def _clear_vocab_indexed(hostname: str, catalog_id: str) -> None:
    """Forget that ``(hostname, catalog_id)`` vocabs were indexed this lifetime.

    Called by ``deriva_ml_reindex_vocabularies`` so an explicit re-index request
    is honored AND so the next ``on_catalog_connect`` re-runs the bulk pass
    (otherwise the persistent guard would suppress it).
    """
    _indexed_vocab_catalogs.discard((hostname, catalog_id))


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


# v3.x: index the deriva-ml-mcp-plugin repo's own top-level README so
# MCP server documentation is searchable via rag_search alongside the
# deriva-ml library docs. Same noise caveat as the deriva-ml source
# above -- repo-root indexing also picks up CLAUDE.md and the
# docs/scratch/*.md planning notes; the upstream exclude_paths=[...]
# addition would let us drop these cleanly.
_GITHUB_MCP_DOCS_NAME = "deriva-ml-mcp-plugin-docs"
_GITHUB_MCP_DOCS_OWNER = "informatics-isi-edu"
_GITHUB_MCP_DOCS_REPO = "deriva-ml-mcp-plugin"
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

# ML-kind doc_type per catalog table, carried on every chunk so
# ``rag_search(doc_type="ml-dataset" | "ml-workflow" | "ml-execution")``
# can filter to one kind (issue #7, resolved plugin-side). We write
# chunks directly to the store -- core's hardcoded-doc_type
# ``index_table_data`` path (deriva-mcp-core#2) is not on our write
# path, so the tag is ours to set. The per-user ``data:`` source-name
# filter in core's rag_search is independent of doc_type; ACL posture
# is unchanged. Unknown tables fall back to the generic "catalog-data".
_DOC_TYPE_BY_TABLE: dict[str, str] = {
    "Dataset": "ml-dataset",
    "Workflow": "ml-workflow",
    "Execution": "ml-execution",
}
_VOCAB_DOC_TYPE = "ml-vocab"


# Strong references to in-flight index-on-find background tasks. Without
# this set the event loop may garbage-collect a pending task before it
# runs (asyncio holds only a weak reference). Mirrors the
# ``_connect_tasks`` pattern in deriva-mcp-core/tools/catalog.py.
_on_find_tasks: set[asyncio.Task] = set()

# Maps a per-user-trio table token to the *name* of the surgical
# re-index coroutine that warms one row of that table. Storing the name
# (not the function object) lets ``_index_rows_on_find`` resolve the
# current module-level binding at call time, so ``patch.object`` on a
# ``_reindex_*`` coroutine is honored. Used by ``_index_rows_on_find``.
_REINDEX_BY_TOKEN: dict[str, str] = {}


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
    doc_type = _DOC_TYPE_BY_TABLE.get(table_name, "catalog-data")
    for c in chunk_markdown(rendered, source=source, doc_type=doc_type):
        chunks.append(
            Chunk(
                text=c.text,
                source=source,
                doc_type=doc_type,
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
    failure returns 0; the row is picked up on the next read-through
    warm (a later list/get of this row via ``_index_rows_on_find``) or
    via ``deriva_ml_resync_indexes``.

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
    except ValueError as exc:
        # A ValueError here is almost always a catalog row carrying a value
        # deriva-ml's enums can't parse -- most commonly a legacy
        # Execution.Status (e.g. "Completed") that predates the current
        # ExecutionStatus set. ``lookup_execution`` raises it while parsing the
        # row, so this single execution can't be indexed. This recurs on every
        # read of that row, so log a CONCISE, actionable single line (no
        # traceback) rather than spamming a stack dump. Fix is catalog-data: map
        # the legacy value to a current enum term. The row is skipped, not
        # crashed -- it warms once the data is corrected.
        logger.warning(
            "deriva-ml RAG: skipping execution %s in %s/%s -- the row holds a "
            "value deriva-ml cannot parse (likely a legacy Execution.Status / "
            "ExecutionStatus enum mismatch): %s. Fix the catalog row's value to "
            "a current enum term.",
            execution_rid,
            hostname,
            catalog_id,
            exc,
        )
        return 0
    except Exception:  # noqa: BLE001 -- best-effort cache refresh
        logger.exception(
            "deriva-ml RAG: surgical re-index failed for execution %s in %s/%s",
            execution_rid,
            hostname,
            catalog_id,
        )
        return 0


_REINDEX_BY_TOKEN[_DATASET_TOKEN] = _reindex_dataset.__name__
_REINDEX_BY_TOKEN[_WORKFLOW_TOKEN] = _reindex_workflow.__name__
_REINDEX_BY_TOKEN[_EXECUTION_TOKEN] = _reindex_execution.__name__


def _index_rows_on_find(
    hostname: str,
    catalog_id: str,
    table_token: str,
    rows: list[dict[str, Any]],
) -> None:
    """Warm the calling user's per-RID RAG sources for rows just read.

    Read-through (index-on-find) indexing: a list/get tool that just
    fetched ``rows`` calls this to schedule a best-effort background
    re-index of each row, so a later ``rag_search`` finds them. Rows are
    indexed in the order given (the order the read returned them) -- no
    re-sorting.

    Fire-and-forget: schedules one background task per row and returns
    immediately. Never blocks the caller's response and never raises --
    a missing event loop (import-time / sync test) is a silent no-op, and
    each per-row reindex swallows its own errors (see ``_reindex_*``).

    Args:
        hostname: Deriva hostname.
        catalog_id: Catalog ID or alias.
        table_token: One of ``_DATASET_TOKEN`` / ``_WORKFLOW_TOKEN`` /
            ``_EXECUTION_TOKEN``. Selects the per-row reindex coroutine.
        rows: Summary-shape row dicts (must carry ``"rid"``). Rows
            without a RID are skipped.

    Returns:
        None.

    Example:
        >>> _index_rows_on_find(  # doctest: +SKIP
        ...     "host", "1", _DATASET_TOKEN, [{"rid": "1-AAAA"}],
        ... )
    """
    reindex_name = _REINDEX_BY_TOKEN.get(table_token)
    if reindex_name is None:
        return
    # Resolve the current module-level binding at call time (not the
    # binding captured at import) so ``patch.object`` on a ``_reindex_*``
    # coroutine -- as the tests and the per-tool call sites rely on -- is
    # honored.
    reindex_fn = globals()[reindex_name]

    async def _do() -> None:
        for row in rows:
            rid = row.get("rid")
            if not rid:
                continue
            try:
                await reindex_fn(hostname, catalog_id, rid)
            except Exception:  # noqa: BLE001 -- best-effort warm
                logger.debug(
                    "index-on-find reindex failed for %s %s in %s/%s",
                    table_token,
                    rid,
                    hostname,
                    catalog_id,
                    exc_info=True,
                )

    try:
        task = asyncio.get_running_loop().create_task(_do())
        _on_find_tasks.add(task)
        task.add_done_callback(_on_find_tasks.discard)
    except RuntimeError:
        # No running event loop (import-time, sync test) -- silent no-op.
        pass


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
    OWN mutations, and read-through warming (``_index_rows_on_find``)
    covers rows the user reads, but mutations by OTHER users (or by
    non-MCP clients like Chaise) leave the calling user's per-user
    sources stale until they read or re-index them. This helper is the
    bridge -- and the replacement for the removed first-connect bulk
    pass: when called, it re-runs the per-RID re-index loop over every
    visible row on demand.

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
    (same contract as ``_reindex_<entity>``). The returned dict reports
    the count of sources successfully refreshed per table -- a partial
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
    except ValueError as exc:
        # A ValueError at the fetch boundary is almost always a catalog row
        # carrying a value deriva-ml's enums can't parse -- most commonly a
        # legacy Execution.Status (e.g. "Completed") that predates the current
        # ExecutionStatus set. ``find_executions()`` materializes lookup_execution
        # per row, so ONE such row aborts the whole table fetch. (This boundary
        # is generic across dataset/workflow/execution; only the execution
        # fetcher realistically raises it.) Log a CONCISE, actionable single line
        # (no traceback) instead of a stack dump -- the row recurs on every
        # resync. Fix is catalog-data: map the legacy value to a current enum
        # term. Skip this table's resync gracefully, like the fetcher-failure
        # branch below.
        logger.warning(
            "deriva-ml RAG: skipping %s resync in %s/%s -- a row holds a value "
            "deriva-ml cannot parse (likely a legacy Execution.Status / "
            "ExecutionStatus enum mismatch): %s. Fix the catalog row's value to "
            "a current enum term.",
            table_label,
            hostname,
            catalog_id,
            exc,
        )
        return 0
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
        for c in chunk_markdown(rendered, source=source, doc_type=_VOCAB_DOC_TYPE):
            chunks.append(
                Chunk(
                    text=c.text,
                    source=source,
                    doc_type=_VOCAB_DOC_TYPE,
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

    # find_vocabularies lives on the DerivaModel (ml.model), not on the
    # DerivaML facade -- deriva-ml >=1.41 exposes it there. (list_vocabulary_terms
    # below remains on ml directly.)
    vocab_tables = await asyncio.to_thread(ml.model.find_vocabularies)
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

    This is the only ``on_catalog_connect`` hook the plugin registers
    (v1.5 removed the three per-user bulk row hooks). Vocabularies are
    catalog-public (no per-user ACL) and small, so a single first-connect
    bulk pass over all discovered vocab tables -- writing one shared
    ``vocab:`` source per table -- is the right shape. The per-user row
    tables, by contrast, are warmed read-through and on mutate rather
    than on connect; see the module docstring.

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
        # Two-layer first-connect guard (see the _indexed_vocab_catalogs comment
        # block). Vocab content is catalog-public (no per-user partition), so the
        # work unit is just (host, catalog). This is the single most expensive
        # cold-start pass -- the ICD10_Eye vocab alone is 1209 embedded terms --
        # so suppressing redundant re-runs here removes most of the regression.
        key = (hostname, catalog_id)
        if key in _indexed_vocab_catalogs:
            return
        lock = _vocab_connect_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in _indexed_vocab_catalogs:  # re-check after acquiring the lock
                return
            try:
                await _index_vocabularies(hostname, catalog_id)
                # Record only after a successful pass so a transient failure
                # stays retryable on the next connect.
                _indexed_vocab_catalogs.add(key)
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
    """Register the GitHub doc sources and the vocabulary catalog-connect hook.

    Wires:

    - Two ``rag_github_source`` declarations (the deriva-ml library docs
      and this plugin's own README) -- crawled once when RAG is enabled.
    - One ``on_catalog_connect`` hook for vocabularies (catalog-public,
      custom ``vocab:`` source prefix -- v1.1; bypasses upstream's
      ``data:`` user-id filter, served to all users in the catalog).

    v1.5 removed the three per-user bulk row indexers (Dataset,
    Workflow, Execution) that used to register as ``on_catalog_connect``
    hooks here -- they re-fetched and re-embedded every visible row on
    every catalog connect. Those per-user ``data:`` per-RID sources
    (``data:{host}:{cat}:{user_id}:{table}:{rid}``) are now warmed
    read-through by the list/get tools (via ``_index_rows_on_find``) and
    surgically on mutate (via ``_reindex_<entity>``), with the
    ``deriva_ml_resync_indexes`` tool (``_resync_*`` + ``_fetch_*_rows``)
    available for manual bulk backfill. The per-RID ``data:`` source
    shape is unchanged.

    Called from ``plugin.register(ctx)``. Safe to call without any RAG
    config -- ``ctx.rag_github_source`` is a no-op when RAG is
    disabled, and the vocab hook is best-effort: it swallows fetch
    exceptions and short-circuits when the vector store is unavailable.

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
    ctx.on_catalog_connect(_make_vocab_hook())
