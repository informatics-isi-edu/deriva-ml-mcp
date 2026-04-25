# deriva-ml-mcp

DerivaML domain plugin for [deriva-mcp-core](../deriva-mcp-core).

Exposes ML-domain workflows (datasets, features, workflows, executions) as MCP tools and resources for LLM clients. Generic Deriva primitives (entity CRUD, schema, vocabulary, hatrac) come from `deriva-mcp-core`; this plugin adds only the ML-specific layer on top.

## Status

Pre-alpha — under active development.

## Install

Install alongside `deriva-mcp-core`:

```bash
uv pip install deriva-mcp-core deriva-ml-mcp
```

The plugin is discovered via the `deriva_mcp.plugins` entry-point group at server startup.

## Development

See workspace conventions in [`../CLAUDE.md`](../CLAUDE.md). Standard commands:

```bash
uv sync
uv run pytest
uv run ruff check src tests
uv run ruff format src tests
```

Live-catalog tests:

```bash
uv run pytest -m integration   # requires a Deriva server reachable on localhost
```
