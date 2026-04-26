"""Shared helpers for deriva-ml-mcp tool modules.

These were originally factored out during Phase 2 (dataset domain) and
are lifted here so Phase 3+ domain modules (feature, workflow,
execution) can reuse them without copy-paste. They have zero domain-
specific coupling -- every helper takes operation/key/etc. as
parameters.

Public API:
    _error_envelope: Standard tool error response (with optional audit
        emission and partial-state response fields).
    _paginate: Shape-agnostic cursor pagination via caller-supplied key.
    _read_rid: Key extractor for objects with a known RID attribute or
        dicts with a known key.
    _row_rid_for: Key extractor factory for denormalized-row dicts
        with Table.column-style keys.
    _MAX_LIMIT: Pagination cap (1000 rows per page).

The leading underscore on each helper is preserved -- they're internal
to deriva-ml-mcp, not part of the package's public API.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from deriva_mcp_core.telemetry import audit_event

_MAX_LIMIT = 1000

logger = logging.getLogger(__name__)


def _error_envelope(
    exc: Exception,
    *,
    operation: str,
    hostname: str,
    catalog_id: str,
    audit: bool = True,
    response_fields: dict[str, Any] | None = None,
    **audit_fields: Any,
) -> str:
    """Standard tool error response.

    Always logs the exception with traceback (operators get the stack
    trace without it leaking into the JSON return shape). For mutation
    tools, also emits a ``<operation>_failed`` audit event so failures
    show up in the audit log alongside successes.

    Args:
        exc: The exception that triggered the error path.
        operation: The audit-event base name (without the ``_failed``
            suffix or ``deriva_ml_`` prefix). e.g. ``"create_dataset"``
            -> audit event ``"deriva_ml_create_dataset_failed"``.
        hostname: The Deriva hostname (passed to audit_event).
        catalog_id: The catalog ID (passed to audit_event).
        audit: If True (default), emit the failure audit event. Set
            False for read-only tools -- failures are still logged with
            traceback, but no audit row is written. The convention is
            "audit logs are for state changes; reads neither succeed
            nor fail in the audit-log sense."
        response_fields: Optional extra keys to merge into the JSON
            error response (in addition to ``"error"``). Use for
            partial-state visibility -- e.g. for a tool that does N
            sub-operations in a loop, pass ``{"completed": [...]}`` so
            the LLM can reason about what was actually mutated before
            the failure.
        **audit_fields: Additional keyword fields included in the
            audit event payload (e.g. ``dataset_rid``). Ignored when
            ``audit=False``.

    Returns:
        JSON string. Default shape is ``{"error": str(exc)}``; with
        ``response_fields`` it becomes ``{"error": ..., **response_fields}``.

    Example (illustrative -- runs only inside an except block):
        >>> # return _error_envelope(exc, operation="create_dataset",
        >>> #                        hostname=h, catalog_id=c, execution_rid=e)
    """
    logger.error(
        "deriva_ml_%s failed: %s: %s",
        operation,
        type(exc).__name__,
        exc,
        exc_info=True,
    )
    if audit:
        audit_event(
            f"deriva_ml_{operation}_failed",
            hostname=hostname,
            catalog_id=catalog_id,
            error_type=type(exc).__name__,
            **audit_fields,
        )
    payload: dict[str, Any] = {"error": str(exc)}
    if response_fields:
        payload.update(response_fields)
    return json.dumps(payload)


def _paginate(
    items: list[Any],
    *,
    after_rid: str | None,
    limit: int,
    key: Callable[[Any], str],
) -> tuple[list[Any], bool, str | None]:
    """Apply cursor pagination to a sorted list using a caller-supplied key fn.

    Args:
        items: Source list, sorted by ``key`` in ascending order.
        after_rid: Skip items where ``key(item) <= after_rid``.
        limit: Page size (already capped by caller).
        key: Function that extracts the comparable RID from one item.
            Use ``functools.partial(_read_rid, rid_key="...")`` for
            object/dict items with a known RID attribute, or ``_row_rid``
            for denormalized rows whose RID column is auto-detected.

    Returns:
        Tuple of (page, truncated, next_after_rid).

    Note:
        ``truncated`` is True whenever the page returned exactly ``limit``
        rows, even if there happen to be no more rows. This may produce a
        false positive in the edge case where ``len(items) == limit``
        exactly, but the convention matches deriva-mcp-core's
        ``get_entities`` so callers can use one pagination idiom across
        both layers.

    Example:
        >>> from functools import partial
        >>> items = [{"RID": "1-AAAA"}, {"RID": "1-BBBB"}, {"RID": "1-CCCC"}]
        >>> page, truncated, nxt = _paginate(
        ...     items, after_rid=None, limit=2,
        ...     key=partial(_read_rid, rid_key="RID"),
        ... )
        >>> [p["RID"] for p in page]
        ['1-AAAA', '1-BBBB']
        >>> truncated
        True
        >>> nxt
        '1-BBBB'
    """
    if after_rid is not None:
        items = [it for it in items if key(it) > after_rid]
    page = items[:limit]
    truncated = len(page) == limit
    next_after_rid = key(page[-1]) if page and truncated else None
    return page, truncated, next_after_rid


def _read_rid(item: Any, rid_key: str) -> str:
    """Read a RID from either a dict or an object.

    Args:
        item: Either a dict (with ``"RID"`` or ``rid_key`` as a key) or
            an object exposing ``rid_key`` as an attribute.
        rid_key: The attribute or dict key to read. For dicts, ``"RID"``
            is also tried first to honour the catalog convention.

    Returns:
        The RID as a string.

    Raises:
        KeyError: If the item is a dict and neither ``"RID"`` nor
            ``rid_key`` is present.

    Example:
        >>> _read_rid({"RID": "1-AAAA"}, "dataset_rid")
        '1-AAAA'
        >>> class Obj:
        ...     dataset_rid = "1-BBBB"
        >>> _read_rid(Obj(), "dataset_rid")
        '1-BBBB'
    """
    if isinstance(item, dict):
        # Dict members may use "RID" (catalog convention) or the configured key.
        for candidate in ("RID", rid_key):
            if candidate in item:
                return str(item[candidate])
        raise KeyError(f"no RID-like key in member dict (looked for RID, {rid_key})")
    return str(getattr(item, rid_key))


def _row_rid_for(row_per: str | None) -> Callable[[dict[str, Any]], str]:
    """Build a callable that extracts the RID from a denormalized-row dict.

    Denormalizer rows use ``Table.column`` keys (e.g. ``Image.RID``).
    When ``row_per`` is known, prefer the unambiguous ``f"{row_per}.RID"``
    column. Otherwise fall back to the first key matching ``RID`` or
    ``*.RID`` in iteration order (which by denormalizer convention is
    the anchor table's RID, but not guaranteed for multi-table joins
    without a known anchor).

    Args:
        row_per: The row-per anchor table name, or ``None`` if unknown.

    Returns:
        A function ``row -> rid_str``. Returns empty string if no
        RID-like key is present (sorts first; pagination terminates).

    Example:
        >>> extract = _row_rid_for("Image")
        >>> extract({"Image.RID": "1-AAAA", "Image.name": "x.png"})
        '1-AAAA'
        >>> extract_anon = _row_rid_for(None)
        >>> extract_anon({"Subject.RID": "1-BBBB"})
        '1-BBBB'
    """
    preferred = f"{row_per}.RID" if row_per else None

    def extract(row: dict[str, Any]) -> str:
        if preferred and preferred in row:
            return str(row[preferred])
        if "RID" in row:
            return str(row["RID"])
        for key, value in row.items():
            if key == "RID" or key.endswith(".RID"):
                return str(value)
        return ""

    return extract
