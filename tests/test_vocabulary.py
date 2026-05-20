"""Unit tests for vocabulary domain tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import _success_calls, make_patch_audit

# Dual-patch context manager for the vocabulary module. See
# ``make_patch_audit`` in ``tests/_helpers.py`` for the canonical
# explanation of why both bind sites are patched together.
_patch_vocabulary_audit = make_patch_audit("vocabulary")


def _make_vocab_table_mock(
    schema_name: str = "my_project",
    columns: list[str] | None = None,
) -> MagicMock:
    """Build a deriva-py Table-shaped MagicMock for ml.create_vocabulary's return.

    ``ml.create_vocabulary`` returns the freshly-created
    ``deriva.core.ermrest_model.Table``. The tool reads
    ``vocab_table.schema.name`` and ``[c.name for c in vocab_table.columns]``;
    everything else can stay defaulted.
    """
    table = MagicMock()
    table.schema = MagicMock()
    table.schema.name = schema_name
    column_names = columns or [
        "RID",
        "RCT",
        "RMT",
        "RCB",
        "RMB",
        "Name",
        "ID",
        "URI",
        "Description",
        "Synonyms",
    ]
    table.columns = [MagicMock(name=n) for n in column_names]
    # Match the .name attribute since MagicMock's `name` kwarg sets
    # the mock's own name (not the .name attribute we read).
    for mock_col, real_name in zip(table.columns, column_names):
        mock_col.name = real_name
    return table


@pytest.fixture()
def vocabulary_ctx(ctx, mock_ml):
    """Register vocabulary tools with mock_ml as the DerivaML stand-in.

    Patches at the use-site (``deriva_ml_mcp.tools.vocabulary.get_ml``)
    and imports the tool module *inside* the patch block so the
    register call resolves ``get_ml`` to the mock.
    """
    with patch("deriva_ml_mcp.tools.vocabulary.get_ml", return_value=mock_ml):
        from deriva_ml_mcp.tools import vocabulary as vocabulary_module

        vocabulary_module.register(ctx)
        yield ctx


# ---------------------------------------------------------------------------
# create_vocabulary
# ---------------------------------------------------------------------------


async def test_create_vocabulary_success(vocabulary_ctx, capturing_mcp, mock_ml):
    """Happy path: returns the new vocab schema and emits success audit.

    The tool must:
      * call ``ml.create_vocabulary`` with the supplied args
      * return ``status="created"`` with schema, vocab_name, columns
      * audit ``deriva_ml_create_vocabulary`` on success
    """
    new_vocab = _make_vocab_table_mock(
        schema_name="my_project",
        columns=["RID", "Name", "ID", "URI", "Description", "Synonyms"],
    )
    mock_ml.create_vocabulary.return_value = new_vocab

    # Patch fire_schema_change so the test doesn't touch the framework
    # plugin registry. _patch_vocabulary_audit handles audit_event.
    with (
        patch("deriva_ml_mcp.tools.vocabulary.fire_schema_change"),
        _patch_vocabulary_audit() as mock_audit,
    ):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_create_vocabulary"](
                hostname="h",
                catalog_id="1",
                vocab_name="Tissue_Type",
                comment="Standard tissue classifications",
            )
        )

    assert out["status"] == "created"
    assert out["schema"] == "my_project"
    assert out["vocab_name"] == "Tissue_Type"
    assert "Name" in out["columns"]
    assert "URI" in out["columns"]
    assert "ID" in out["columns"]

    mock_ml.create_vocabulary.assert_called_once_with(
        "Tissue_Type",
        comment="Standard tissue classifications",
        schema=None,
        update_navbar=True,
    )

    success = _success_calls(mock_audit, "deriva_ml_create_vocabulary")
    assert success, "expected deriva_ml_create_vocabulary audit event"
    assert success[0].kwargs["schema"] == "my_project"
    assert success[0].kwargs["vocab_name"] == "Tissue_Type"
    assert success[0].kwargs["update_navbar"] is True
    # No `comment` in audit fields (free text).
    assert "comment" not in success[0].kwargs


async def test_create_vocabulary_explicit_schema(vocabulary_ctx, capturing_mcp, mock_ml):
    """When schema= is supplied, it's passed through to ml.create_vocabulary."""
    new_vocab = _make_vocab_table_mock(schema_name="other_schema")
    mock_ml.create_vocabulary.return_value = new_vocab

    with (
        patch("deriva_ml_mcp.tools.vocabulary.fire_schema_change"),
        _patch_vocabulary_audit(),
    ):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_create_vocabulary"](
                hostname="h",
                catalog_id="1",
                vocab_name="Custom_Type",
                schema="other_schema",
                update_navbar=False,
            )
        )

    assert out["status"] == "created"
    assert out["schema"] == "other_schema"
    mock_ml.create_vocabulary.assert_called_once_with(
        "Custom_Type",
        comment="",
        schema="other_schema",
        update_navbar=False,
    )


async def test_create_vocabulary_failure_emits_failed_audit(
    vocabulary_ctx, capturing_mcp, mock_ml
):
    """Failure path: error envelope returned, failed-audit emitted, no success-audit."""
    mock_ml.create_vocabulary.side_effect = RuntimeError("Table Tissue_Type already exist")

    with (
        patch("deriva_ml_mcp.tools.vocabulary.fire_schema_change"),
        _patch_vocabulary_audit() as mock_audit,
    ):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_create_vocabulary"](
                hostname="h",
                catalog_id="1",
                vocab_name="Tissue_Type",
            )
        )

    assert out == {"error": "Table Tissue_Type already exist"}

    failed = _success_calls(mock_audit, "deriva_ml_create_vocabulary_failed")
    assert failed, "expected deriva_ml_create_vocabulary_failed audit event"
    assert failed[0].kwargs["error_type"] == "RuntimeError"
    assert failed[0].kwargs["vocab_name"] == "Tissue_Type"
    assert not _success_calls(mock_audit, "deriva_ml_create_vocabulary")
