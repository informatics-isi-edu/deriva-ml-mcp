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
    _table_qname: Render a deriva-py Table as ``"schema.name"``.
    _table_to_dict: Render a deriva-py Table as ``{"name", "schema"}``.
    _MAX_LIMIT: Pagination cap (1000 rows per page).

The leading underscore on each helper is preserved -- they're internal
to deriva-ml-mcp, not part of the package's public API.

Note on testing audit events: ``audit_event`` (imported below from
``deriva_mcp_core.telemetry``) is called from inside ``_error_envelope``
on the failure path. Tool modules also import ``audit_event`` directly
into their own namespace for success-path emission. Tests that need to
capture both paths must patch BOTH the tool-module name AND
``deriva_ml_mcp._helpers.audit_event`` to the same mock — see
``make_patch_audit(module_name)`` in ``tests/_helpers.py`` (the
canonical factory; per-domain test files bind it as
``_patch_audit = make_patch_audit("dataset")`` etc.). Python's
``from X import name`` binds ``name`` in the importing module's
namespace at import time, so a single-patch facade isn't possible
without changing every call site to use attribute lookup.
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
    # Structured-error uplift (audit T2.3): when the underlying
    # exception carries useful structured attributes, surface them on
    # the envelope so LLM callers can react without parsing the
    # message string. Currently handles:
    #
    #   - DerivaMLRidsNotFound.missing_rids -- per-RID failure list
    #     from deriva-ml's resolve_rids batch lookup. Lets a caller
    #     pinpoint which RIDs in a batch were unknown without
    #     splitting the request.
    #
    # Late import to avoid coupling _helpers.py to deriva-ml's import
    # tree at module load time. The except is defensive in case the
    # exception class moves between deriva-ml versions.
    try:
        from deriva_ml.core.exceptions import DerivaMLRidsNotFound

        if isinstance(exc, DerivaMLRidsNotFound):
            # Sort for deterministic wire shape regardless of set
            # iteration order.
            payload["missing_rids"] = sorted(exc.missing_rids)
    except ImportError:
        pass
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


def _table_qname(table: Any) -> str:
    """Return the qualified ``schema.name`` form of a deriva-py Table.

    Args:
        table: A ``deriva.core.ermrest_model.Table`` instance (the type
            returned by ``ml.find_vocabularies()`` and similar deriva-py
            model accessors).

    Returns:
        The string ``f"{table.schema.name}.{table.name}"``.

    Example:
        >>> from unittest.mock import MagicMock
        >>> t = MagicMock()
        >>> t.name = "Dataset_Type"
        >>> t.schema.name = "deriva-ml"
        >>> _table_qname(t)
        'deriva-ml.Dataset_Type'
    """
    return f"{table.schema.name}.{table.name}"


def _table_to_dict(table: Any) -> dict[str, str]:
    """Return a JSON-friendly ``{schema, name}`` dict for a deriva-py Table.

    The canonical shape used across the plugin's tool/resource responses
    when emitting table references. Use ``_table_qname`` when a string
    form (e.g., for log messages or markdown headers) is preferable.

    Args:
        table: A ``deriva.core.ermrest_model.Table`` instance.

    Returns:
        ``{"name": table.name, "schema": table.schema.name}``.

    Example:
        >>> from unittest.mock import MagicMock
        >>> t = MagicMock()
        >>> t.name = "Image"
        >>> t.schema.name = "demo-schema"
        >>> _table_to_dict(t)
        {'name': 'Image', 'schema': 'demo-schema'}
    """
    return {"name": table.name, "schema": table.schema.name}


# NOTE: _set_row_description was removed in v0.5.0. It was a
# pathBuilder workaround for Asset.description / Dataset.description
# lacking write-through setters in deriva-ml pre-1.38.0. v1.38.0
# introduced both setters, so callers now do ``asset.description = ...``
# / ``ds.description = ...`` directly. See git history for the prior
# implementation (and tests/test_helpers.py history for the unit
# tests that pinned the pathBuilder shape).


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


# ---------------------------------------------------------------------------
# Cite URLs (RFC-style /id/{cat}/{rid}[@snaptime] resolver links)
# ---------------------------------------------------------------------------


def _cite_url(ml: Any, rid: str) -> str | None:
    """Return the live (no-snaptime) cite URL for ``rid``.

    Used by per-row response models for non-dataset entities (assets,
    executions, workflows, etc.). Always returns the live form; clients
    that need a snaptime-pinned URL for a dataset version go through
    ``_cite_dataset_version_url`` instead.

    The URL form is ``https://{host}/id/{catalog}/{rid}``.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        rid: The entity RID to cite.

    Returns:
        The cite URL string, or ``None`` if construction fails (e.g.
        the rid does not resolve, or ``ml.cite`` raises). The
        ``cite_url`` field is best-effort from a presentation
        standpoint -- a missing URL is recoverable; a missing field
        would break the whole bundle.

    Note:
        Uses ``ml.cite(rid, current=True)`` under the hood. The
        ``current=True`` argument selects the no-snaptime form;
        the default ``current=False`` would append the *latest*
        catalog snaptime, which is the wrong form for a per-row
        navigation link in a bundle resource.
    """
    try:
        return ml.cite(rid, current=True)
    except Exception:  # noqa: BLE001 -- presentation field, never raise
        return None


def _cite_dataset_version_url(
    ml: Any, dataset_rid: str, version: str | None, *, ds: Any | None = None
) -> str | None:
    """Return the cite URL for a specific dataset version.

    Routing per ADR-0003 / 0004:

    - **Released version** (PEP 440 not-devrelease, e.g. ``"0.4.0"``)
      → snaptime-pinned URL ``.../id/{cat}/{rid}@{snaptime}``.
      The snaptime comes from the dataset's version-history row for
      that release.
    - **Current dev version** (PEP 440 devrelease matching the
      dataset's current dev row, e.g. ``"0.4.0.post1.dev3"``) →
      no-snaptime URL ``.../id/{cat}/{rid}``. Dev rows have no
      ``Snapshot``; per ADR-0003 they resolve to the live state.
    - **`None` / current_version** → resolves to whichever shape
      ``current_version`` takes (released or dev), routed as above.

    Args:
        ml: A connected ``deriva_ml.DerivaML`` instance.
        dataset_rid: The Dataset RID.
        version: A PEP 440 version string, or None to use the
            dataset's ``current_version``.
        ds: Optional pre-fetched ``Dataset`` object for ``dataset_rid``.
            Pass it when the caller has already looked up the dataset
            to avoid a redundant ``ml.lookup_dataset`` call.

    Returns:
        The cite URL string, or ``None`` if construction fails. A
        degraded-to-live URL is preferred over None when the live
        form is constructible; ``None`` is returned only when even
        ``ml.cite(rid, current=True)`` raises.

    Note:
        Defensive: if anything in the version-history walk raises,
        falls back to the live URL. The cite_url field is best-effort
        from a presentation standpoint -- a missing snaptime is
        recoverable; a missing field would break the whole bundle.
    """
    try:
        if ds is None:
            ds = ml.lookup_dataset(dataset_rid)
        if version is None:
            version = str(ds.current_version)

        # Dev versions have no snapshot. Detect via the PEP 440 typed
        # property. Construct a DatasetVersion to query is_devrelease.
        from deriva_ml.dataset.aux_classes import DatasetVersion

        try:
            parsed = DatasetVersion.parse(version)
        except Exception:  # noqa: BLE001 -- malformed version strings degrade
            parsed = None

        if parsed is not None and parsed.is_devrelease:
            return _cite_url(ml, dataset_rid)

        # Released version: look up snapshot via dataset_history().
        for entry in ds.dataset_history():
            if str(entry.dataset_version) == version:
                snapshot = getattr(entry, "snapshot", None)
                if snapshot:
                    return f"https://{ml.host_name}/id/{ml.catalog_id}/{dataset_rid}@{snapshot}"
                # Released version with no recorded snapshot is a catalog
                # inconsistency (released rows must carry one); fall
                # through to the live URL rather than synthesising one.
                break

        # Version not found in history (or had no snapshot): degrade.
        return _cite_url(ml, dataset_rid)
    except Exception:  # noqa: BLE001 -- presentation field, never raise
        return _cite_url(ml, dataset_rid)
