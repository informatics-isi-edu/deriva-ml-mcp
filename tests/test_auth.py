"""Unit tests for the deriva_ml_check_authentication tool (tools/auth.py).

``deriva_ml_check_authentication`` answers "can the MCP server reach and
authenticate to this catalog, and as whom?" -- a pre-flight a user runs
before catalog operations to confirm the server's credential is present,
unexpired, and accepted by the catalog's auth layer.

Fully mocked: ``get_ml`` is patched at the module's import site and the
fake DerivaML exposes ``whoami()`` (the identity dict or ``None``).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def auth_ctx(ctx):
    """Register the auth tool on a fresh PluginContext."""
    from deriva_ml_mcp_plugin.tools import auth as auth_module

    auth_module.register(ctx)
    return ctx


def _fake_ml(*, whoami_return=None, whoami_raises=None):
    ml = MagicMock()
    if whoami_raises is not None:
        ml.whoami.side_effect = whoami_raises
    else:
        ml.whoami.return_value = whoami_return
    return ml


async def test_check_authentication_authenticated(auth_ctx, capturing_mcp):
    """A live session -> authenticated True + the identity dict echoed back."""
    from deriva_ml_mcp_plugin.tools import auth as auth_module

    identity = {
        "id": "https://auth.globus.org/abc",
        "display_name": "Test User",
        "email": "user@example.org",
        "full_name": "Test User",
    }
    fake = _fake_ml(whoami_return=identity)
    with patch.object(auth_module, "get_ml", return_value=fake):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_check_authentication"](
                hostname="h", catalog_id="1"
            )
        )
    assert out["authenticated"] is True
    assert out["hostname"] == "h"
    assert out["catalog_id"] == "1"
    assert out["identity"]["display_name"] == "Test User"
    assert out["identity"]["email"] == "user@example.org"
    fake.whoami.assert_called_once_with()


async def test_check_authentication_not_authenticated(auth_ctx, capturing_mcp):
    """No session (whoami -> None) -> authenticated False, identity null."""
    from deriva_ml_mcp_plugin.tools import auth as auth_module

    fake = _fake_ml(whoami_return=None)
    with patch.object(auth_module, "get_ml", return_value=fake):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_check_authentication"](
                hostname="h", catalog_id="1"
            )
        )
    assert out == {
        "authenticated": False,
        "identity": None,
        "hostname": "h",
        "catalog_id": "1",
    }


async def test_check_authentication_error_envelope(auth_ctx, capturing_mcp):
    """A real error (non-auth HTTP / connect failure) -> error envelope, not a False."""
    from deriva_ml_mcp_plugin.tools import auth as auth_module

    fake = _fake_ml(whoami_raises=RuntimeError("could not connect to host h"))
    with patch.object(auth_module, "get_ml", return_value=fake):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_check_authentication"](
                hostname="h", catalog_id="1"
            )
        )
    assert out == {
        "error": "could not connect to host h",
        "error_type": "RuntimeError",
    }


def test_check_authentication_is_read_only(ctx):
    """The tool must be registered mutates=False (it makes no catalog change).

    ``PluginContext.tool`` strips ``mutates`` before forwarding, so we wrap
    ``ctx.tool`` to capture the declared value -- same idiom as
    ``test_resolve.py``'s read-only check.
    """
    from deriva_ml_mcp_plugin.tools import auth as auth_module

    seen: dict[str, bool] = {}
    original_tool = ctx.tool

    def capturing_tool(*args, mutates, **kwargs):
        decorator = original_tool(*args, mutates=mutates, **kwargs)

        def wrapper(fn):
            seen[fn.__name__] = mutates
            return decorator(fn)

        return wrapper

    ctx.tool = capturing_tool
    try:
        auth_module.register(ctx)
    finally:
        ctx.tool = original_tool

    assert seen == {"deriva_ml_check_authentication": False}
