"""Helpers for obtaining a :class:`deriva_ml.DerivaML` instance.

This is the **only** module in deriva-ml-mcp that touches deriva-mcp-core's
connection plumbing. Every tool gets its DerivaML through :func:`get_ml`;
no tool calls into core's credential surface directly or re-instantiates
connections.
"""

from __future__ import annotations

from deriva_mcp_core import get_credential, remap_hostname
from deriva_ml import DerivaML


def get_ml(hostname: str, catalog_id: str) -> DerivaML:
    """Build a DerivaML instance for the current MCP request.

    Resolves the credential via :func:`deriva_mcp_core.get_credential`,
    which is transport-aware: in HTTP mode it returns the per-request
    derived token from the auth verifier's contextvar, and in stdio mode
    it returns the credential from ``~/.deriva/credential.json`` for the
    given hostname. The same call site works under both transports.

    Applies the deriva-mcp-core hostname remap so connections opened by
    deriva-ml (which constructs its own ``DerivaServer`` internally) route
    through the same internal alias as connections opened by core's
    ``get_catalog()``. Without this, an external-facing hostname like
    ``"localhost"`` would resolve to the MCP container's own loopback
    (where nothing listens on 443) instead of the configured internal
    Deriva host (e.g. ``"deriva"`` inside the deriva-docker stack).

    Args:
        hostname: The Deriva server hostname (e.g. ``"my.deriva.org"``).
            Used as the lookup key in stdio mode; the HTTP path ignores
            it (the derived token is already scoped to the right server).
        catalog_id: The catalog ID as a string (e.g. ``"1"``).

    Returns:
        A connected :class:`DerivaML` instance.

    Raises:
        RuntimeError: In HTTP mode, if called outside a tool or resource
            handler (no per-request credential available). In stdio mode,
            ``deriva.core.get_credential`` returns an empty dict for
            unknown hosts rather than raising, so the typical failure
            mode is a downstream auth error from DERIVA itself.

    Example:
        Inside a tool body::

            from deriva_ml_mcp.ml_context import get_ml

            ml = get_ml(hostname, catalog_id)
            datasets = ml.find_datasets()
    """
    internal = remap_hostname(hostname)
    return DerivaML(internal, catalog_id, credential=get_credential(internal))
