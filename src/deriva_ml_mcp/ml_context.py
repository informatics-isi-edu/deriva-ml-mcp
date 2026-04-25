"""Helpers for obtaining a :class:`deriva_ml.DerivaML` instance.

This is the **only** module in deriva-ml-mcp that touches deriva-mcp-core's
connection plumbing. Every tool gets its DerivaML through :func:`get_ml`;
no tool calls :func:`deriva_mcp_core.get_request_credential` directly or
re-instantiates connections.
"""

from __future__ import annotations

from deriva_mcp_core import get_request_credential
from deriva_ml import DerivaML


def get_ml(hostname: str, catalog_id: str) -> DerivaML:
    """Build a DerivaML instance for the current MCP request.

    Reads the per-request credential from deriva-mcp-core's contextvar and
    passes it to DerivaML. In stdio mode, core falls back to
    ``~/.deriva/credential.json``; the caller does not need to handle this
    distinction.

    Args:
        hostname: The Deriva server hostname (e.g. ``"my.deriva.org"``).
        catalog_id: The catalog ID as a string (e.g. ``"1"``).

    Returns:
        A connected :class:`DerivaML` instance.

    Raises:
        RuntimeError: If called outside an MCP request context, propagated
            from :func:`deriva_mcp_core.get_request_credential`.

    Example:
        Inside a tool body::

            from deriva_ml_mcp.ml_context import get_ml

            ml = get_ml(hostname, catalog_id)
            datasets = ml.find_datasets()
    """
    credential = get_request_credential()
    return DerivaML(hostname, catalog_id, credential=credential)
