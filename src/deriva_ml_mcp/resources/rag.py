"""Per-user RAG indexing + DerivaML docs source for deriva-ml-mcp.

Phase 6.3 wires three things into the RAG subsystem:

1. **One GitHub doc source** -- ``informatics-isi-edu/deriva-ml`` ``docs/``
   (declared synchronously via ``ctx.rag_github_source``). The indexer
   crawls it once when RAG is enabled.

2. **Three ``on_catalog_connect`` hooks** -- one each for ``Dataset``,
   ``Workflow``, and ``Execution`` rows. Each hook resolves the calling
   user's identity, fetches rows under that user's credential, and
   upserts into the vector store with a user-scoped source name
   (``data:{host}:{cat}:{user_id}``). The fetch path goes through the
   plugin's existing ``_list_*_impl`` helpers, so ACL behavior matches
   the read tools.

3. **Three ``RowSerializer`` implementations** -- one per table, each
   producing rich Markdown sections for the LLM (header, fields,
   subsections; empties omitted FaceBase-style).

Why hooks + ``index_table_data`` (not ``ctx.rag_dataset_indexer``)?
``rag_dataset_indexer`` produces a single global enriched source shared
across all users (the enricher fires under whichever credential happens
to connect first). For ML data with per-user ACLs that would leak rows
across users. The per-user pattern here partitions the source name by
user identity and fetches under the calling user's credential -- same
posture as every other tool in this plugin.

Wiring into ``plugin.py`` is deferred to Phase 6.4. This module is
import-safe and idempotent on its own.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

# TODO(upstream-rag-userfilter): tracked as deriva-mcp-core#1
# (https://github.com/informatics-isi-edu/deriva-mcp-core/issues/1).
# rag_search must filter data: sources by user_id (symmetric to the
# existing schema-source filter). Until it lands, any user with read
# access to the vector store can match against another user's chunks by
# source name. See docs/coverage.md "Upstream gaps (Phase 6 RAG)".
from deriva_mcp_core.context import resolve_user_identity
from deriva_mcp_core.rag import get_rag_store
from deriva_mcp_core.rag.data import RowSerializer, index_table_data

from deriva_ml_mcp._helpers import _MAX_LIMIT
from deriva_ml_mcp.ml_context import get_ml
from deriva_ml_mcp.tools.dataset import _list_datasets_impl
from deriva_ml_mcp.tools.execution import _list_executions_impl
from deriva_ml_mcp.tools.workflow import _list_workflows_impl

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext

logger = logging.getLogger(__name__)


_GITHUB_DOCS_NAME = "deriva-ml-docs"
_GITHUB_DOCS_OWNER = "informatics-isi-edu"
_GITHUB_DOCS_REPO = "deriva-ml"
_GITHUB_DOCS_BRANCH = "main"
_GITHUB_DOCS_PATH_PREFIX = "docs/"
_GITHUB_DOCS_DOC_TYPE = "ml-docs"


_DATASET_TABLE = "Dataset"
_WORKFLOW_TABLE = "Workflow"
_EXECUTION_TABLE = "Execution"


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
            _kv_line("Duration", row.get("duration")),
        ]
        return _render_block(f"## Execution: {rid}", lines)


# ---------------------------------------------------------------------------
# Row fetchers
# ---------------------------------------------------------------------------


def _fetch_dataset_rows(hostname: str, catalog_id: str) -> list[dict[str, Any]]:
    """Fetch up to ``_MAX_LIMIT`` Dataset summary rows under the caller's credential.

    Wraps ``_list_datasets_impl`` and pulls the ``"datasets"`` array. The
    helper already returns JSON-friendly dicts so no shape conversion is
    needed before passing to the serializer.
    """
    ml = get_ml(hostname, catalog_id)
    payload = _list_datasets_impl(ml, after_rid=None, limit=_MAX_LIMIT)
    return list(payload.get("datasets", []))


def _fetch_workflow_rows(hostname: str, catalog_id: str) -> list[dict[str, Any]]:
    """Fetch up to ``_MAX_LIMIT`` Workflow summary rows under the caller's credential."""
    ml = get_ml(hostname, catalog_id)
    payload = _list_workflows_impl(ml, after_rid=None, limit=_MAX_LIMIT)
    return list(payload.get("workflows", []))


def _fetch_execution_rows(hostname: str, catalog_id: str) -> list[dict[str, Any]]:
    """Fetch up to ``_MAX_LIMIT`` Execution summary rows under the caller's credential."""
    ml = get_ml(hostname, catalog_id)
    payload = _list_executions_impl(
        ml,
        workflow_rid=None,
        status=None,
        after_rid=None,
        limit=_MAX_LIMIT,
    )
    return list(payload.get("executions", []))


def _coerce_for_index(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-trip rows through JSON to drop non-serializable values.

    The execution summary contains ``datetime`` objects that
    ``index_table_data`` does not need (the serializer renders them
    directly), but other downstream consumers may. The cheapest way to
    keep the rows JSON-friendly is to round-trip through ``json.dumps``
    + ``json.loads`` with ``default=str``. Datasets and workflows pass
    through unchanged.
    """
    return [json.loads(json.dumps(r, default=str)) for r in rows]


# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------


def _make_hook(
    fetch_fn: Callable[[str, str], list[dict[str, Any]]],
    table_name: str,
    serializer: RowSerializer,
) -> Callable[[str, str, str, dict], Awaitable[None]]:
    """Build an ``on_catalog_connect`` hook that indexes ``table_name`` per-user.

    The returned coroutine fetches rows under the caller's credential,
    resolves the calling user's identity, and upserts the rendered chunks
    into the vector store with a user-scoped source name. Adding a
    fourth indexed table is a one-liner at the call site.

    Behavior contract:

    - Wraps the fetch in ``try/except`` so a deriva-ml read failure is
      logged with table context but does not propagate. The framework's
      ``_safe_call`` wraps the index step on its own, so we don't double
      up there -- letting ``index_table_data`` raise gives the framework
      a clean traceback to log.
    - Skips silently (debug log) when ``get_rag_store()`` returns
      ``None`` -- i.e. when RAG is disabled in this deployment.
    - Logs a debug success line including row count + resolved user_id
      so an operator can investigate "why isn't user X's data showing up?"
      by flipping on debug logging without overwhelming production logs.

    Args:
        fetch_fn: Callable that takes ``(hostname, catalog_id)`` and
            returns summary-shape row dicts.
        table_name: Catalog table the rows belong to. Used as both the
            serializer dispatch key and the chunk header.
        serializer: The ``RowSerializer`` to render each row.

    Returns:
        An async hook with the ``on_catalog_connect`` signature
        ``(hostname, catalog_id, schema_hash, schema_json) -> None``.
    """

    # TODO(upstream-rag-doctype): tracked as deriva-mcp-core#2
    # (https://github.com/informatics-isi-edu/deriva-mcp-core/issues/2).
    # Once index_table_data accepts a doc_type param, pass
    # "ml-dataset" / "ml-workflow" / "ml-execution" so
    # rag_search(doc_type=...) can distinguish these from generic
    # catalog-data chunks. See docs/coverage.md "Upstream gaps".
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
            rows = _coerce_for_index(fetch_fn(hostname, catalog_id))
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
            logger.debug("rag store unavailable, skipping %s index", table_name)
            return
        user_id = resolve_user_identity(hostname)
        await index_table_data(
            store,
            hostname,
            catalog_id,
            table_name,
            rows,
            user_id=user_id,
            serializer=serializer,
        )
        logger.debug(
            "indexed %d %s rows for user=%s host=%s catalog=%s",
            len(rows),
            table_name,
            user_id,
            hostname,
            catalog_id,
        )

    return hook


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def register_rag_sources(ctx: PluginContext) -> None:
    """Register the DerivaML docs source and three per-user catalog hooks.

    Called from ``plugin.register(ctx)`` (wired in Phase 6.4). Safe to
    call without any RAG config -- ``ctx.rag_github_source`` is a no-op
    when RAG is disabled, and the hooks are best-effort: they swallow
    fetch exceptions and short-circuit when the vector store is
    unavailable.

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
    ctx.on_catalog_connect(_make_hook(_fetch_dataset_rows, _DATASET_TABLE, _DatasetSerializer()))
    ctx.on_catalog_connect(_make_hook(_fetch_workflow_rows, _WORKFLOW_TABLE, _WorkflowSerializer()))
    ctx.on_catalog_connect(
        _make_hook(_fetch_execution_rows, _EXECUTION_TABLE, _ExecutionSerializer())
    )
