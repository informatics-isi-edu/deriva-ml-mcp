"""Dataset domain tools for deriva-ml-mcp.

Why this is a package, not a single file: the prior single
``tools/dataset.py`` had grown to 17 tools / 1745 lines, well past the
inline-grep threshold the engineer-persona review flagged during v1.0
("when I'm asked to add ``download_dataset_bag``, I'm scrolling through
1700 lines of unrelated tools to find my anchor"). The split mirrors
the read / mutate / complex grouping that was already visible in the
source's declaration order:

- ``read.py``: 7 read-only tools (``deriva_ml_list_datasets``,
  ``deriva_ml_get_dataset``, ``deriva_ml_list_dataset_members``,
  ``deriva_ml_list_dataset_relations``, ``deriva_ml_list_dataset_element_types``,
  ``deriva_ml_bag_info``, ``deriva_ml_get_dataset_spec``) plus the
  four shared dataset helpers (``_summarize_dataset``,
  ``_list_datasets_impl``, ``_get_dataset_detail_impl``,
  ``_list_dataset_members_summary_impl``) that this package re-exports
  for ``resources/ml.py`` and ``resources/rag.py``.
- ``mutate.py``: 7 simple-mutation tools (each a thin wrapper around
  one DerivaML method, an audit emission, and a JSON envelope).
- ``complex.py``: ``deriva_ml_denormalize_dataset`` (the biggest
  tool in the dataset domain). v5.0.0 retired
  ``deriva_ml_cache_dataset`` per the stateless rule -- bag
  materialization is now a user-local Python operation. See
  ``docs/audit-2026-05-23.md``.

See deriva-ml-mcp issue #4 for the full backlog context.

Public surface (preserved from the pre-split single-file shape):

- ``register(ctx)`` -- aggregator; dispatches to the three submodules'
  own ``register`` functions. ``plugin.py`` calls this.
- ``audit_event`` -- re-exported from ``deriva_mcp_core.telemetry`` so
  ``patch("deriva_ml_mcp.tools.dataset.audit_event")`` continues to
  redirect every success-path emission across read / mutate / complex
  in one shot. The submodules access this via attribute lookup on the
  package (``_pkg.audit_event``) rather than ``from ... import`` so
  the package binding IS the canonical patch site.
- The four ``_*_impl`` helpers re-exported from ``read.py`` so
  ``from deriva_ml_mcp.tools.dataset import _list_datasets_impl, ...``
  (used by ``resources/ml.py`` and ``resources/rag.py``) keeps
  resolving identically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-exports below preserve the pre-split single-file patch surface.
# Test code patches ``deriva_ml_mcp.tools.dataset.{audit_event,
# get_ml, Dataset}`` as a single canonical site, and the
# read/mutate/complex submodules deliberately access those three
# names via attribute lookup on this package (``_pkg.audit_event``,
# ``_pkg.get_ml``, etc.) so a single ``patch(...)`` redirects every
# call across the submodules in one shot. ``from ... import`` in
# each submodule would create per-submodule bindings that the patch
# wouldn't reach.
from deriva_mcp_core.telemetry import audit_event
from deriva_ml.dataset.dataset import Dataset

from deriva_ml_mcp.ml_context import get_ml

# Re-export of the four dataset helpers consumed by ``resources/ml.py``
# and ``resources/rag.py``. They live in ``read.py`` (their natural
# domain home) but the import path
# ``from deriva_ml_mcp.tools.dataset import _list_datasets_impl, ...``
# is preserved here so the resource modules don't need to change.
from deriva_ml_mcp.tools.dataset.read import (
    _bag_info_impl,
    _get_dataset_detail_impl,
    _get_dataset_spec_impl,
    _list_dataset_members_summary_impl,
    _list_datasets_impl,
    _summarize_dataset,
)

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


__all__ = [
    "Dataset",
    "audit_event",
    "get_ml",
    "register",
    "_bag_info_impl",
    "_get_dataset_detail_impl",
    "_get_dataset_spec_impl",
    "_list_dataset_members_summary_impl",
    "_list_datasets_impl",
    "_summarize_dataset",
]


def register(ctx: PluginContext) -> None:
    """Register all dataset domain tools with the plugin context.

    Mirrors the top-level ``plugin.py`` register pattern: this aggregator
    dispatches to focused submodules (``read.py`` / ``mutate.py`` /
    ``complex.py``) so the 17 dataset tools split cleanly along
    read / mutate / complex lines without ballooning a single file past
    the comfortable inline-grep threshold (per issue #4 in the v1.x
    backlog).

    Submodule imports are deferred to inside ``register`` because
    ``mutate.py`` and ``complex.py`` import ``deriva_ml_mcp.tools.dataset``
    at module load time (to access ``_pkg.audit_event``); doing the
    submodule imports up here would create an import-time cycle. The
    deferred form lets ``__init__.py`` finish populating its module
    attributes (notably ``audit_event`` and the four helper re-exports)
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
    from deriva_ml_mcp.tools.dataset import complex as _complex
    from deriva_ml_mcp.tools.dataset import mutate as _mutate
    from deriva_ml_mcp.tools.dataset import read as _read

    _read.register(ctx)
    _mutate.register(ctx)
    _complex.register(ctx)
