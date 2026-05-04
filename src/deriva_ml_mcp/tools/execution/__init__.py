"""Execution domain tools for deriva-ml-mcp.

Why this is a package, not a single file: the prior single
``tools/execution.py`` had grown to 12 tools / 1363 lines, past the
inline-grep threshold the same engineer-persona review flagged for
``tools/dataset.py`` (issue #4). The split mirrors the read / mutate
grouping that was already visible in the source's declaration order:

- ``read.py``: 5 read-only tools (``deriva_ml_list_executions``,
  ``deriva_ml_get_execution``, ``deriva_ml_find_workflow_executions``,
  ``deriva_ml_list_execution_children``, ``deriva_ml_list_execution_parents``)
  plus the three shared execution helpers (``_summarize_execution``,
  ``_list_executions_impl``, ``_get_execution_detail_impl``) that this
  package re-exports for ``resources/ml.py`` and ``resources/rag.py``.
- ``mutate.py``: 7 mutating tools (``deriva_ml_create_execution``,
  ``deriva_ml_start_execution``, ``deriva_ml_commit_execution``,
  ``deriva_ml_update_execution``, ``deriva_ml_abort_execution``,
  ``deriva_ml_create_execution_dataset``, ``deriva_ml_add_nested_execution``)
  plus the state-machine constants and the ``_summarize_upload_dict``
  helper they consume.

Public surface (preserved from the pre-split single-file shape):

- ``register(ctx)`` -- aggregator; dispatches to the two submodules'
  own ``register`` functions. ``plugin.py`` calls this.
- ``audit_event`` -- re-exported from ``deriva_mcp_core.telemetry`` so
  ``patch("deriva_ml_mcp.tools.execution.audit_event")`` continues to
  redirect every success-path emission across read / mutate in one
  shot. The submodules access this via attribute lookup on the
  package (``_pkg.audit_event``) rather than ``from ... import`` so
  the package binding IS the canonical patch site.
- ``get_ml`` -- same patch-target rationale; the
  ``deriva_ml_mcp.tools.execution.get_ml`` patch in ``test_execution.py``
  must keep redirecting through this re-export.
- The three ``_*_impl`` helpers re-exported from ``read.py`` so
  ``from deriva_ml_mcp.tools.execution import _list_executions_impl, ...``
  (used by ``resources/ml.py`` and ``resources/rag.py``) keeps
  resolving identically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-exports below preserve the pre-split single-file patch surface.
# Test code patches ``deriva_ml_mcp.tools.execution.{audit_event, get_ml}``
# as a single canonical site, and the read/mutate submodules deliberately
# access those names via attribute lookup on this package
# (``_pkg.audit_event``, ``_pkg.get_ml``) so a single ``patch(...)``
# redirects every call across both submodules in one shot.
# ``from ... import`` in each submodule would create per-submodule
# bindings that the patch wouldn't reach.
from deriva_mcp_core.telemetry import audit_event

from deriva_ml_mcp.ml_context import get_ml

# Re-export of the three execution helpers consumed by ``resources/ml.py``
# and ``resources/rag.py``. They live in ``read.py`` (their natural
# domain home) but the import path
# ``from deriva_ml_mcp.tools.execution import _list_executions_impl, ...``
# is preserved here so the resource modules don't need to change.
from deriva_ml_mcp.tools.execution.read import (
    _get_execution_detail_impl,
    _get_lineage_impl,
    _list_executions_impl,
    _summarize_execution,
)

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


__all__ = [
    "audit_event",
    "get_ml",
    "register",
    "_get_execution_detail_impl",
    "_get_lineage_impl",
    "_list_executions_impl",
    "_summarize_execution",
]


def register(ctx: PluginContext) -> None:
    """Register all execution domain tools with the plugin context.

    Mirrors the ``tools/dataset/__init__.py`` aggregator pattern:
    submodule imports are deferred to inside ``register`` because
    ``mutate.py`` imports ``deriva_ml_mcp.tools.execution`` at module
    load time (to access ``_pkg.audit_event`` / ``_pkg.get_ml``);
    doing the submodule imports up here would create an import-time
    cycle. The deferred form lets ``__init__.py`` finish populating
    its module attributes (notably ``audit_event`` and ``get_ml``)
    before any submodule reaches back through the package object.

    Args:
        ctx: PluginContext supplied by deriva-mcp-core at startup.

    Returns:
        None.

    Example:
        >>> from deriva_mcp_core.plugin.api import PluginContext
        >>> # ctx provided by the framework
        >>> register(ctx)  # doctest: +SKIP
    """
    from deriva_ml_mcp.tools.execution import mutate as _mutate
    from deriva_ml_mcp.tools.execution import read as _read

    _read.register(ctx)
    _mutate.register(ctx)
