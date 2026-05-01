"""Unit tests for ml_context.get_ml()."""

from __future__ import annotations

from unittest.mock import patch


def test_get_ml_resolves_credential_via_core_get_credential():
    """``get_ml`` resolves the credential through ``deriva_mcp_core.get_credential``
    -- the public swap-aware accessor that works in both HTTP and stdio mode --
    and passes the result to ``DerivaML(credential=...)``.

    The single call site replaces the v3.x stdio fallback (try
    ``get_request_credential`` then disk) with a single call to core's new
    transport-aware API. Both HTTP and stdio code paths are now exercised
    by core's own implementation, not duplicated in the plugin.
    """
    fake_credential = {"bearer-token": "from-core"}

    with (
        patch(
            "deriva_ml_mcp.ml_context.get_credential", return_value=fake_credential
        ) as mock_get_cred,
        patch("deriva_ml_mcp.ml_context.DerivaML") as mock_derivaml_cls,
    ):
        from deriva_ml_mcp.ml_context import get_ml

        ml = get_ml("host.example.org", "1")

    # The hostname is forwarded so stdio mode can route to the right
    # ~/.deriva/credential.json entry. HTTP mode ignores it (the contextvar
    # is server-scoped already), but the plugin can't tell the modes apart
    # and shouldn't have to -- that's exactly what core's get_credential
    # encapsulates.
    mock_get_cred.assert_called_once_with("host.example.org")
    mock_derivaml_cls.assert_called_once_with("host.example.org", "1", credential=fake_credential)
    assert ml is mock_derivaml_cls.return_value


def test_get_ml_propagates_credential_errors():
    """If core's ``get_credential`` raises (e.g. HTTP mode called outside a
    request context), the error propagates to the caller for the
    surrounding tool's ``_error_envelope`` to wrap. ``get_ml`` doesn't
    swallow or rewrite credential errors -- callers handle the message
    uniformly via the standard error path."""
    with patch(
        "deriva_ml_mcp.ml_context.get_credential",
        side_effect=RuntimeError("No credential in current request context."),
    ):
        from deriva_ml_mcp.ml_context import get_ml

        try:
            get_ml("host.example.org", "1")
        except RuntimeError as exc:
            assert "No credential" in str(exc)
        else:
            raise AssertionError("RuntimeError was not propagated")
