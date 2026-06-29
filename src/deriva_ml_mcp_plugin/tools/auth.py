"""Authentication pre-flight tool.

``deriva_ml_check_authentication`` answers "can the MCP server reach and
authenticate to this catalog, and as whom?" -- the pre-flight a user runs
before catalog operations to confirm the server's credential is present,
unexpired, and accepted by the catalog's auth layer (the thing that
otherwise blanket-401s every operation).

Authentication is a GENERIC Deriva concern (it's an ERMrest session
check), but the wrapper lives here because the MCP server's credential
model -- one server-side credential per (host, calling user), built by
``ml_context.get_ml`` from core's request credential -- is what the
caller is actually asking about ("is *the server* authenticated for
*me* against this catalog?"). The tool wraps deriva-ml's
``DerivaML.whoami`` (added in deriva-ml #353): one network call to
``GET /authn/session``.

The check confirms *authentication*, not *authorization*: a True result
proves the server knows who the credential is, not that any specific
privileged operation will pass the catalog's ACLs.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from deriva_mcp_core import deriva_call

from deriva_ml_mcp_plugin._helpers import _error_envelope
from deriva_ml_mcp_plugin._response_models import CheckAuthenticationResponse
from deriva_ml_mcp_plugin.ml_context import get_ml

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def register(ctx: PluginContext) -> None:
    """Register the authentication pre-flight tool with the plugin context.

    Args:
        ctx: The plugin context supplied by ``deriva-mcp-core`` at
            server startup.

    Example:
        >>> # ctx provided by the framework; not constructed by user code
        >>> register(ctx)  # doctest: +SKIP
    """

    @ctx.tool(mutates=False)
    async def deriva_ml_check_authentication(
        hostname: str,
        catalog_id: str,
    ) -> str:
        """Check whether the MCP server is authenticated to a catalog, and as whom.

        A PRE-FLIGHT to run before catalog operations when you're not
        sure the server's credential is valid for the target catalog
        (e.g. a fresh session, a host you haven't touched, or after a
        run of 401-looking failures). One network call to the catalog's
        ``GET /authn/session`` via ``DerivaML.whoami``.

        Confirms *authentication* (the server knows who the credential
        is), NOT *authorization*: a True result is a safe bet that
        reads will work and proves the credential is present, unexpired,
        and accepted by the auth layer -- but a privileged write can
        still be refused by a table's ACLs even when authenticated.

        ``authenticated: false`` means specifically "reached the server,
        there is no valid session" (the endpoint returned 401/404). A
        connection failure or a non-auth HTTP error (5xx, DNS, TLS) is
        a different thing -- it comes back as ``{"error": ...}``, not as
        ``authenticated: false`` -- so the two failure modes are
        distinguishable.

        Args:
            hostname: Catalog host (e.g. ``"data.example.org"``).
            catalog_id: Catalog id (e.g. ``"1"``).

        Returns:
            JSON string ``{"authenticated": bool, "identity": {...} |
            null, "hostname", "catalog_id"}``. ``identity`` is the
            logged-in client dict (``id``, ``display_name``, ``email``,
            ``full_name``, ``identities``) when authenticated, else
            ``null``.

        Raises:
            RuntimeError / HTTPError: Wrapped as ``{"error": ...,
                "error_type": ...}`` for connection failures or non-auth
                HTTP errors -- distinct from the ``authenticated: false``
                "no session" answer.

        Example:
            ``{"authenticated": true, "identity": {"display_name":
            "Test User", "email": "user@example.org", ...},
            "hostname": "data.example.org", "catalog_id": "1"}``
        """
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. ``whoami`` makes a
                # blocking ``GET /authn/session`` network call. See
                # deriva-mcp-core plugin-authoring-guide.md §"Synchronous
                # work in threads".
                ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
                identity = await asyncio.to_thread(ml.whoami)
            return CheckAuthenticationResponse(
                authenticated=identity is not None,
                identity=identity,
                hostname=hostname,
                catalog_id=catalog_id,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            # Read-only tool: log + return without an audit row. A
            # connection / non-auth HTTP failure is a real error, NOT a
            # "not authenticated" answer (which whoami signals by
            # returning None, handled above).
            return _error_envelope(
                exc,
                operation="check_authentication",
                hostname=hostname,
                catalog_id=catalog_id,
                audit=False,
            )
