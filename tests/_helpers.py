"""Module-level test utilities for deriva-ml-mcp-plugin.

Pure helpers (functions, classes, factories) that don't depend on
pytest's fixture machinery. Pytest-fixture definitions live in
``tests/conftest.py`` and consume these utilities where needed.

Public API:
    _success_calls: Filter ``audit_event`` mock call records by
        event name.
    make_patch_audit: Factory producing a dual-patch context manager
        for a domain module's ``audit_event`` (one mock captures
        success-path emission from the tool module + failure-path
        emission from ``_helpers._error_envelope``).
    _CapturingMCP: Minimal FastMCP stand-in that stores registered
        tools, resources, and prompts so unit tests can invoke them
        directly without spinning up a real MCP server.
    _server_reachable: Quick TCP probe to
        ``${DERIVA_HOST:-localhost}:443``; used by integration test
        modules in their ``pytestmark`` ``skipif`` gate.
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch


def _success_calls(mock_audit: Any, event_name: str) -> list:
    """Return audit_event call records whose first positional arg matches.

    Generic over any ``audit_event`` MagicMock — works for dataset,
    feature, workflow, execution, and any future domain that emits
    ``deriva_ml_<op>`` / ``deriva_ml_<op>_failed`` audit events.

    Args:
        mock_audit: A ``MagicMock`` standing in for
            ``deriva_mcp_core.telemetry.audit_event``.
        event_name: The exact event name (positional arg 0) to filter
            for, e.g. ``"deriva_ml_create_dataset"`` or
            ``"deriva_ml_create_dataset_failed"``.

    Returns:
        List of ``unittest.mock.call`` records where ``call.args[0] ==
        event_name``. Empty list if no matching calls were made.

    Example:
        >>> from unittest.mock import MagicMock
        >>> m = MagicMock()
        >>> m("deriva_ml_create_dataset", dataset_rid="1-AAAA")
        >>> calls = _success_calls(m, "deriva_ml_create_dataset")
        >>> len(calls)
        1
        >>> calls[0].kwargs["dataset_rid"]
        '1-AAAA'
    """
    return [c for c in mock_audit.call_args_list if c.args and c.args[0] == event_name]


def make_patch_audit(module_name: str):
    """Build a dual-patch context manager for a domain module's audit_event.

    Each domain tool module imports ``audit_event`` from
    ``deriva_mcp_core.telemetry`` directly (success-path emission). The
    failure path goes through ``_error_envelope`` in ``_helpers``, which
    has its own bound name. To capture both with one mock, we patch BOTH
    ``deriva_ml_mcp_plugin.tools.<module_name>.audit_event`` and
    ``deriva_ml_mcp_plugin._helpers.audit_event`` to the same MagicMock.

    Why two patches: Python's ``from X import name`` binds ``name`` in
    the importing module's namespace at import time, so patching only
    the source module wouldn't redirect calls in modules that already
    bound the name. See ``deriva_ml_mcp_plugin/_helpers.py`` docstring for the
    canonical explanation.

    Args:
        module_name: Bare domain module name (e.g. ``"dataset"``,
            ``"feature"``, ``"workflow"``, ``"execution"``). Used to
            construct the import path
            ``deriva_ml_mcp_plugin.tools.<module_name>.audit_event``.

    Returns:
        A no-arg context manager. Entering yields a single
        ``MagicMock`` that has been substituted into both bind sites.

    Example:
        >>> _patch_workflow_audit = make_patch_audit("workflow")
        >>> with _patch_workflow_audit() as mock_audit:  # doctest: +SKIP
        ...     # ... invoke a workflow tool ...
        ...     pass
        >>> # mock_audit captures both success and failure-path audits
    """
    target = f"deriva_ml_mcp_plugin.tools.{module_name}.audit_event"

    @contextmanager
    def _patch():
        with (
            patch(target) as mock_audit,
            patch("deriva_ml_mcp_plugin._helpers.audit_event", new=mock_audit),
        ):
            yield mock_audit

    return _patch


class _CapturingMCP:
    """Minimal FastMCP stand-in that stores registered tools and resources.

    Used by unit tests to invoke registered tools directly without spinning
    up a real MCP server. Mirrors the fixture in
    ``deriva-mcp-core/tests/test_tools.py``.
    """

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.tool_kwargs: dict[str, dict[str, Any]] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(self, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            self.tool_kwargs[fn.__name__] = kwargs
            return fn

        return decorator

    def resource(self, uri, *args, **kwargs):
        def decorator(fn):
            self.resources[uri] = fn
            return fn

        return decorator

    def prompt(self, name=None, *args, **kwargs):
        def decorator(fn):
            self.prompts[name or fn.__name__] = fn
            return fn

        return decorator


def _server_reachable() -> bool:
    """Quick TCP probe to ``${DERIVA_HOST:-localhost}:443``.

    Used by integration test modules in their ``pytestmark`` ``skipif``
    gate. Lives here so multiple integration test files can import the
    same helper rather than defining it independently.

    Returns:
        True if a TCP connect succeeded within 2 seconds, False otherwise.

    Example:
        >>> # In an integration test file:
        >>> from tests._helpers import _server_reachable
        >>> import pytest
        >>> pytestmark = [
        ...     pytest.mark.integration,
        ...     pytest.mark.skipif(not _server_reachable(), reason="..."),
        ... ]
    """
    host = os.environ.get("DERIVA_HOST", "localhost")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, 443))
        sock.close()
        return result == 0
    except Exception:
        return False
