"""Unit tests for ``deriva_ml_mcp._helpers``.

The other helpers (``_error_envelope``, ``_paginate``, ``_read_rid``,
``_row_rid_for``) are exercised end-to-end by the per-domain test suites
and by the doctest examples in their own docstrings -- they have no
dedicated test file. The small Table-rendering helpers added in v1.1
(``_table_qname`` / ``_table_to_dict``) are pure value transforms with
no domain coupling, so they get isolated tests here rather than being
rolled into the dataset suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from deriva_ml_mcp._helpers import _table_qname, _table_to_dict


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
