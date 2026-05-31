"""Unit tests for ``deriva_ml_mcp_plugin._helpers``.

The other helpers (``_error_envelope``, ``_paginate``, ``_read_rid``,
``_row_rid_for``) are exercised end-to-end by the per-domain test suites
and by the doctest examples in their own docstrings -- they have no
dedicated test file. The small Table-rendering helpers added in v1.1
(``_table_qname`` / ``_table_to_dict``) are pure value/IO transforms
with no domain coupling, so they get isolated tests here rather than
being rolled into the dataset / asset suites.

v0.5.0: the ``_set_row_description`` pathBuilder workaround was
removed when deriva-ml v1.38.0 introduced write-through
``Asset.description`` / ``Dataset.description`` setters. The two unit
tests that pinned the pathBuilder shape went with it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from deriva_ml_mcp_plugin._helpers import (
    _table_qname,
    _table_to_dict,
)


def test_table_qname_joins_schema_and_name() -> None:
    """``_table_qname`` produces the canonical ``schema.name`` string."""
    t = MagicMock()
    t.name = "Dataset_Type"
    t.schema.name = "deriva-ml"
    assert _table_qname(t) == "deriva-ml.Dataset_Type"


def test_table_qname_handles_domain_schema() -> None:
    """Works for domain-schema vocabularies, not just the ML schema."""
    t = MagicMock()
    t.name = "Tissue_Type"
    t.schema.name = "demo-schema"
    assert _table_qname(t) == "demo-schema.Tissue_Type"


def test_table_to_dict_returns_canonical_shape() -> None:
    """``_table_to_dict`` returns ``{"name", "schema"}`` -- the JSON shape used by tools."""
    t = MagicMock()
    t.name = "Image"
    t.schema.name = "demo-schema"
    assert _table_to_dict(t) == {"name": "Image", "schema": "demo-schema"}


def test_table_to_dict_keys_are_lowercase() -> None:
    """Pin the key names so any rename is caught here (callers depend on these keys)."""
    t = MagicMock()
    t.name = "X"
    t.schema.name = "y"
    out = _table_to_dict(t)
    assert set(out.keys()) == {"name", "schema"}


# _set_row_description tests removed in v0.5.0 along with the helper
# (replaced by deriva-ml v1.38.0 write-through description setters).


def test_error_envelope_surfaces_missing_rids_from_typed_exception() -> None:
    """``DerivaMLRidsNotFound`` -> ``missing_rids`` field on the envelope.

    Audit T2.3: structured-error uplift. The deriva-ml
    ``DerivaMLRidsNotFound`` exception carries a ``missing_rids: set[str]``
    attribute that lets a caller pinpoint per-RID failure in a batch
    request. ``_error_envelope`` auto-detects the type and injects the
    sorted list onto the wire response so LLM callers can react
    without parsing the error message string.
    """
    import json as _json

    from deriva_ml.core.exceptions import DerivaMLRidsNotFound

    from deriva_ml_mcp_plugin._helpers import _error_envelope

    # audit=False to avoid touching the audit-event side; the
    # structured-fields injection runs regardless of the audit flag.
    exc = DerivaMLRidsNotFound({"3JSE", "WX01", "1-AAAA"})
    out = _json.loads(
        _error_envelope(
            exc,
            operation="add_dataset_members",
            hostname="h",
            catalog_id="1",
            audit=False,
        )
    )
    assert "error" in out
    # Sorted for deterministic wire shape (set iteration order varies).
    assert out["missing_rids"] == ["1-AAAA", "3JSE", "WX01"]


def test_error_envelope_no_missing_rids_for_generic_exception() -> None:
    """Non-RidsNotFound exceptions get the plain ``{"error": ...}`` envelope."""
    import json as _json

    from deriva_ml_mcp_plugin._helpers import _error_envelope

    out = _json.loads(
        _error_envelope(
            RuntimeError("something else broke"),
            operation="add_dataset_members",
            hostname="h",
            catalog_id="1",
            audit=False,
        )
    )
    assert out == {"error": "something else broke"}
    assert "missing_rids" not in out
