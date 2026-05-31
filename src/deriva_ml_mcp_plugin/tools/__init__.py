"""Tool modules for deriva-ml-mcp-plugin.

Each module under this package owns a single ML domain area
(dataset, feature, workflow, execution) and exposes a
``register(ctx)`` function that ``deriva_ml_mcp_plugin.plugin.register``
will dispatch to once the corresponding phase lands.
"""

from __future__ import annotations
