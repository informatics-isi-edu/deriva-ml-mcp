"""Helpers for obtaining a :class:`deriva_ml.DerivaML` instance.

This is the **only** module in deriva-ml-mcp that touches deriva-mcp-core's
connection plumbing. Every tool gets its DerivaML through :func:`get_ml`;
no tool calls into core's credential surface directly or re-instantiates
connections.
"""

from __future__ import annotations

from deriva.core import get_credential as _disk_get_credential
from deriva_mcp_core import get_request_credential
from deriva_ml import DerivaML


def get_ml(hostname: str, catalog_id: str) -> DerivaML:
    """Build a DerivaML instance for the current MCP request.

    Resolves the credential in a transport-aware way so the same call site
    works under both deriva-mcp-core transports:

    - **HTTP transport:** the auth verifier sets a per-request credential
      contextvar before tool dispatch. :func:`get_request_credential` reads
      it. This is the common path in HTTP mode and never raises.
    - **stdio transport:** the contextvar is never populated -- stdio is
      single-user and core's design has the *generic* tools resolve
      credentials from ``~/.deriva/credential.json`` at call time via a
      hostname-keyed swap that core installs at startup. The swap is not
      exposed publicly, so we mirror it here: catch the ``RuntimeError``
      raised by :func:`get_request_credential` and fall back to
      :func:`deriva.core.get_credential`, which reads the same file core's
      stdio path does.

    Boundary note: this duplicates a small slice of core's stdio-fallback
    logic in the plugin. The cleaner long-term fix is for core to ship a
    public swap-aware accessor (e.g. ``get_credential_for_hostname``); when
    that lands, this function should switch to it and the disk import can
    be dropped.

    Args:
        hostname: The Deriva server hostname (e.g. ``"my.deriva.org"``).
            Passed to ``deriva.core.get_credential`` in the stdio fallback;
            ignored in HTTP mode.
        catalog_id: The catalog ID as a string (e.g. ``"1"``).

    Returns:
        A connected :class:`DerivaML` instance.

    Raises:
        RuntimeError: Only if BOTH the HTTP contextvar is unset AND
            ``deriva.core.get_credential`` returns no credential for
            ``hostname`` (e.g. user has not run
            ``deriva-globus-auth-utils login --host <hostname>``). The
            stdio fallback path returns an empty dict for unknown hosts
            rather than raising, so this case is rare.

    Example:
        Inside a tool body::

            from deriva_ml_mcp.ml_context import get_ml

            ml = get_ml(hostname, catalog_id)
            datasets = ml.find_datasets()
    """
    try:
        credential = get_request_credential()
    except RuntimeError:
        # stdio mode: the contextvar is never populated. Fall back to the
        # same on-disk credential file that core's stdio path reads.
        credential = _disk_get_credential(hostname)
    return DerivaML(hostname, catalog_id, credential=credential)
