"""Unit tests for ml_context.get_ml()."""

from __future__ import annotations

from unittest.mock import patch


def test_get_ml_uses_contextvar_credential_in_http_mode():
    """When core's request contextvar is set (HTTP mode -- auth verifier
    populated it), ``get_ml`` uses that credential and does NOT touch the
    on-disk fallback."""
    fake_credential = {"bearer-token": "test-bearer"}

    with (
        patch(
            "deriva_ml_mcp.ml_context.get_request_credential", return_value=fake_credential
        ) as mock_get_cred,
        patch("deriva_ml_mcp.ml_context._disk_get_credential") as mock_disk,
        patch("deriva_ml_mcp.ml_context.DerivaML") as mock_derivaml_cls,
    ):
        from deriva_ml_mcp.ml_context import get_ml

        ml = get_ml("host.example.org", "1")

    mock_get_cred.assert_called_once_with()
    mock_disk.assert_not_called()  # disk fallback must not fire when contextvar is set
    mock_derivaml_cls.assert_called_once_with("host.example.org", "1", credential=fake_credential)
    assert ml is mock_derivaml_cls.return_value


def test_get_ml_falls_back_to_disk_credential_in_stdio_mode():
    """When core's request contextvar is unset (stdio mode -- the contextvar
    is never populated there), ``get_ml`` catches the RuntimeError raised by
    ``get_request_credential`` and falls back to the on-disk credential read
    via ``deriva.core.get_credential(hostname)``. Mirrors the same fallback
    that core's generic tools use via the swap-aware ``_get_credential_fn``.
    """
    fake_disk_credential = {"cookie": "webauthn=disk-cookie"}

    with (
        patch(
            "deriva_ml_mcp.ml_context.get_request_credential",
            side_effect=RuntimeError("No credential in current request context."),
        ) as mock_get_cred,
        patch(
            "deriva_ml_mcp.ml_context._disk_get_credential", return_value=fake_disk_credential
        ) as mock_disk,
        patch("deriva_ml_mcp.ml_context.DerivaML") as mock_derivaml_cls,
    ):
        from deriva_ml_mcp.ml_context import get_ml

        ml = get_ml("host.example.org", "1")

    mock_get_cred.assert_called_once_with()
    # Disk lookup is per-hostname so the right credential is selected when
    # ~/.deriva/credential.json contains entries for multiple hosts.
    mock_disk.assert_called_once_with("host.example.org")
    mock_derivaml_cls.assert_called_once_with(
        "host.example.org", "1", credential=fake_disk_credential
    )
    assert ml is mock_derivaml_cls.return_value
