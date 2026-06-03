"""Execution domain tools for deriva-ml-mcp-plugin.

**Read-only domain.** As of v0.5.0, this package registers only
read-side tools. The execution lifecycle (``create``, ``start``,
``commit``, ``abort``, ``update``, ``add_feature_values``,
``create_execution_dataset``, ``add_nested_execution``) is owned by
the caller's local Python environment (the ``with
ml.create_execution(...) as exe:`` pattern). For the runnable how-to,
call ``rag_search(query="execution lifecycle", doc_type="ml-docs")``
(the ``user-guide/executions.md`` doc, retrievable over this MCP);
Claude Code clients also have the ``deriva-skills``
``execution-lifecycle`` skill.

Per the stateless / bounded-resource rule (CLAUDE.md), executions
originate in the user's environment because:

1. The workflow code lives in the user's git checkout; the MCP server
   has no access to it.
2. Feature staging writes to a per-process SQLite manifest store; a
   server-side stage couldn't pair with a user-side commit.
3. Asset uploads need local file bytes; bytes can't reasonably travel
   over the MCP wire.
4. The DerivaML intended pattern is ``with Execution(...) as exe:``;
   the context manager owns cleanup + abort-on-exception semantics,
   and MCP can't participate in that because the context lives in
   the caller's process.

The result is a clean architectural cut: **execution lifecycle =
local Python; MCP = catalog observation only.** This package therefore
exposes only read tools so an LLM can query, debug, compare, and
walk lineage across executions the user has produced.

Why this is still a package, not a single file: the package shape
predates the v0.5.0 contraction and the per-domain pattern is
preserved across other domains (``tools/dataset/`` is multi-file).
Keeping the package shape leaves room for future read-side tools to
land without forcing a re-split.

Public surface:

- ``register(ctx)`` -- aggregator; dispatches to ``read.register(ctx)``.
  ``plugin.py`` calls this.
- ``audit_event`` -- re-exported from ``deriva_mcp_core.telemetry`` so
  ``patch("deriva_ml_mcp_plugin.tools.execution.audit_event")`` continues to
  redirect emissions through the canonical patch site (even though no
  read tools emit audit events today, the re-export stays for tests
  and for any future read-tool side-effect that wants to audit).
- ``get_ml`` -- same patch-target rationale; the
  ``deriva_ml_mcp_plugin.tools.execution.get_ml`` patch in ``test_execution.py``
  must keep redirecting through this re-export.
- The three ``_*_impl`` helpers re-exported from ``read.py`` so
  ``from deriva_ml_mcp_plugin.tools.execution import _list_executions_impl, ...``
  (used by ``resources/ml.py`` and ``resources/rag.py``) keeps
  resolving identically.

History: ``mutate.py`` was deleted in the v0.5.0 cut. Recover from
git history if you need to see the prior shape; the audit doc at
``docs/audit-2026-05-23.md`` records the architectural rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-exports below preserve the pre-split single-file patch surface.
# Test code patches ``deriva_ml_mcp_plugin.tools.execution.{audit_event, get_ml}``
# as a single canonical site, and the read/mutate submodules deliberately
# access those names via attribute lookup on this package
# (``_pkg.audit_event``, ``_pkg.get_ml``) so a single ``patch(...)``
# redirects every call across both submodules in one shot.
# ``from ... import`` in each submodule would create per-submodule
# bindings that the patch wouldn't reach.
from deriva_mcp_core.telemetry import audit_event

from deriva_ml_mcp_plugin.ml_context import get_ml

# Re-export of the three execution helpers consumed by ``resources/ml.py``
# and ``resources/rag.py``. They live in ``read.py`` (their natural
# domain home) but the import path
# ``from deriva_ml_mcp_plugin.tools.execution import _list_executions_impl, ...``
# is preserved here so the resource modules don't need to change.
from deriva_ml_mcp_plugin.tools.execution.read import (
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

    v0.5.0+ registers only ``read.register(ctx)``. The mutate side was
    deleted because execution lifecycle belongs in the user's local
    environment per the stateless / bounded-resource rule (CLAUDE.md
    + docs/audit-2026-05-23.md).

    Submodule import is deferred to inside ``register`` because
    ``read.py`` accesses ``_pkg.audit_event`` / ``_pkg.get_ml`` via
    attribute lookup on this package; doing the submodule import up
    here would create an import-time cycle. The deferred form lets
    ``__init__.py`` finish populating its module attributes before
    ``read.py`` reaches back through the package object.

    Args:
        ctx: PluginContext supplied by deriva-mcp-core at startup.

    Returns:
        None.

    Example:
        >>> from deriva_mcp_core.plugin.api import PluginContext
        >>> # ctx provided by the framework
        >>> register(ctx)  # doctest: +SKIP
    """
    from deriva_ml_mcp_plugin.tools.execution import read as _read

    _read.register(ctx)
