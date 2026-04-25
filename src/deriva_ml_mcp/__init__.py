"""deriva-ml-mcp: DerivaML domain plugin for deriva-mcp-core.

This package registers MCP tools and resources that express ML-domain
workflows (datasets, features, workflows, executions). Generic Deriva
primitives are provided by deriva-mcp-core; this plugin layers ML
semantics on top.

Example:
    Operators load this plugin by installing the package alongside
    deriva-mcp-core::

        uv pip install deriva-mcp-core deriva-ml-mcp

    The plugin is then discovered automatically via the
    ``deriva_mcp.plugins`` entry-point group at server startup.
"""

from __future__ import annotations
