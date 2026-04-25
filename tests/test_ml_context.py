"""Unit tests for ml_context.get_ml()."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_get_ml_uses_core_credential_and_constructs_derivaml():
    """get_ml(host, catalog_id) must call core's get_request_credential() and
    pass the result as DerivaML's ``credential=`` kwarg."""
    fake_credential = {"cookie": "test-cookie"}

    with (
        patch(
            "deriva_ml_mcp.ml_context.get_request_credential", return_value=fake_credential
        ) as mock_get_cred,
        patch("deriva_ml_mcp.ml_context.DerivaML") as mock_derivaml_cls,
    ):
        from deriva_ml_mcp.ml_context import get_ml

        ml = get_ml("host.example.org", "1")

    mock_get_cred.assert_called_once_with()
    mock_derivaml_cls.assert_called_once_with("host.example.org", "1", credential=fake_credential)
    assert ml is mock_derivaml_cls.return_value


def test_get_ml_propagates_credential_errors():
    """If core has no current credential, get_ml() must propagate the error
    without wrapping it — callers handle the message uniformly."""
    with patch(
        "deriva_ml_mcp.ml_context.get_request_credential",
        side_effect=RuntimeError("no current request"),
    ):
        from deriva_ml_mcp.ml_context import get_ml

        with pytest.raises(RuntimeError, match="no current request"):
            get_ml("host.example.org", "1")
