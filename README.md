# deriva-ml-mcp

DerivaML domain plugin for [deriva-mcp-core](../deriva-mcp-core).

Exposes ML-domain workflows (datasets, features, workflows, executions) as MCP tools and resources for LLM clients. Generic Deriva primitives (entity CRUD, schema, vocabulary, hatrac) come from `deriva-mcp-core`; this plugin adds only the ML-specific layer on top.

## Status

Pre-alpha — under active development.

## Install

The plugin is discovered automatically by `deriva-mcp-core` via the
`deriva_mcp.plugins` entry-point group -- no registration step is required
beyond having both packages installed in the same Python environment.

### Production (uv tool)

```bash
uv tool install \
  --from git+https://github.com/informatics-isi-edu/deriva-mcp-core.git \
  --with deriva-ml-mcp \
  --with deriva-ml \
  deriva-mcp-core
```

`deriva-ml` is pulled in transitively, but listing it explicitly with
`--with` gives you a single-tool venv where all three packages can be
upgraded in lockstep with `uv tool upgrade deriva-mcp-core`.

### Development (editable checkout)

For working on the plugin source, clone the workspace and install
everything editable into one tool venv:

```bash
git clone https://github.com/informatics-isi-edu/deriva-mcp-core.git
git clone https://github.com/informatics-isi-edu/deriva-ml-mcp.git
git clone https://github.com/informatics-isi-edu/deriva-ml.git

uv tool install --reinstall \
  --with-editable ./deriva-ml-mcp \
  --with-editable ./deriva-ml \
  ./deriva-mcp-core
```

Source edits in any of the three checkouts take effect on the next MCP
server restart -- no reinstall needed.

### Docker (deriva-docker)

To run the plugin inside a [`deriva-docker`](https://github.com/informatics-isi-edu/deriva-docker)
deployment that uses `DERIVA_MCP_EXTRA_PACKAGES` to install plugins
into the MCP server's image, add this line to your deriva-docker env
file (typically `~/.deriva-docker/env/localhost.env`):

```bash
DERIVA_MCP_EXTRA_PACKAGES="deriva-ml-mcp@git+https://github.com/informatics-isi-edu/deriva-ml-mcp.git@main deriva-ml@git+https://github.com/informatics-isi-edu/deriva-ml.git@main deriva@git+https://github.com/informatics-isi-edu/deriva-py@deriva-ml"
```

The three packages pin against `main` (or a working branch for
`deriva-py`) so each rebuild picks up the latest commits. Pin to a
specific tag/commit for reproducible deployments.

The plugin's entry-point name (set in `pyproject.toml` under
`[project.entry-points."deriva_mcp.plugins"]`) is **`deriva-ml-mcp`** --
deliberately the same as the PyPI package name so the deriva-docker
default config (`mcp/config/deriva-mcp.env`,
`DERIVA_MCP_PLUGIN_ALLOWLIST=facebase,deriva-ml-mcp`) loads the
plugin out of the box, no override needed.

The same config file sets `DERIVA_MCP_DISABLE_MUTATING_TOOLS=false`,
so the server runs in mutable mode -- write tools (`create_dataset`,
`add_dataset_members`, `start_execution`, etc.) are exposed.

Confirm both on startup by grepping the container log for:

```
INFO ... Mutating tools are ENABLED (DERIVA_MCP_DISABLE_MUTATING_TOOLS=false).
INFO ... Loaded plugin: deriva-ml-mcp (deriva_ml_mcp.plugin:register)
```

To pick up new commits, rebuild and restart the MCP service:

```bash
docker-compose --env-file ~/.deriva-docker/env/localhost.env down deriva-mcp-test
docker-compose --env-file ~/.deriva-docker/env/localhost.env build deriva-mcp-test --no-cache
docker-compose --env-file ~/.deriva-docker/env/localhost.env up -d deriva-mcp-test
```

The `scripts/rebuild-deriva-docker-mcp.sh` helper in this repo wraps
those three commands. From a deriva-docker checkout (where
`docker-compose.yml` lives):

```bash
/path/to/deriva-ml-mcp/scripts/rebuild-deriva-docker-mcp.sh
# or pass a non-default env file:
/path/to/deriva-ml-mcp/scripts/rebuild-deriva-docker-mcp.sh /path/to/env
```

### Connecting Claude Code to the dockerized server

Once the container is up, point Claude Code at the HTTP MCP endpoint:

```bash
claude mcp add -t http dev-localhost https://localhost/mcp \
    --client-id deriva-mcp --callback-port 8080
```

Verify with `claude mcp list` -- the entry should show
`dev-localhost: https://localhost/mcp (HTTP) - ✓ Connected`. The OAuth
client-id (`deriva-mcp`) is the one pre-registered with the credenza
auth service in the deriva-docker deployment; `--callback-port 8080`
is where Claude listens for the auth callback. Remove stale stdio-mode
entries first (`claude mcp remove deriva -s <scope>`) so the tools
surface in the new HTTP server isn't shadowed.

> **This is the current canonical recipe for dev-localhost.** It will
> be superseded by `deriva-docker`'s built-in plugin-installation
> workflow once that ships (currently the plugin is bolted in via
> `DERIVA_MCP_EXTRA_PACKAGES` + `--no-cache` rebuild rather than as a
> first-class deriva-docker service). Until then, follow this section
> exactly -- it is not a placeholder.

## Configuration

The plugin reads no environment variables directly; its behavior is
controlled by `deriva-mcp-core`'s settings (see core's
[Configuration Reference](https://github.com/informatics-isi-edu/deriva-mcp-core#configuration-reference)
and [Deployment Guide](https://github.com/informatics-isi-edu/deriva-mcp-core/blob/main/docs/deployment-guide.md)
for the full list). The settings most relevant to this plugin:

| Variable                              | Default | What it does to the plugin                                                                                                                                                                                       |
|---------------------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DERIVA_MCP_RAG_ENABLED`              | `false` | Required to activate the plugin's RAG indexing. With it `false`, the plugin still loads but its GitHub-docs source registration and per-user catalog indexer hooks are no-ops, and `rag_search` is not exposed. |
| `DERIVA_MCP_RAG_AUTO_ENRICH`          | `false` | When `true`, the plugin's `on_catalog_connect` hooks index the calling user's visible Dataset / Workflow / Execution rows on first connect. Off by default for cost reasons; turn on for semantic-search use.    |
| `DERIVA_MCP_DISABLE_MUTATING_TOOLS`   | `true`  | When `true`, all 30+ `mutates=True` plugin tools (everything that creates/updates/deletes datasets, workflows, executions, features, assets) return an error envelope without executing. Set `false` for write access. |
| `DERIVA_MCP_MUTATION_REQUIRED_CLAIM`  | unset   | HTTP transport only: gates plugin mutating tools behind a token claim. Example: `{"groups": ["deriva-mcp-mutator"]}` requires the user's introspection payload to include that group.                            |
| `DERIVA_MCP_PLUGIN_ALLOWLIST`         | unset   | If set, only listed plugins load. Use `deriva-ml` to load only this plugin: `DERIVA_MCP_PLUGIN_ALLOWLIST=deriva-ml`.                                                                                              |
| `DERIVA_ML_ALLOW_DIRTY`               | `false` | Not a `DERIVA_MCP_*` variable -- read by `deriva-ml` itself. When `true`, `deriva_ml_create_workflow` bypasses the dirty-tree git check that records a provenance-poisoning warning. Use only in test runs.      |

Environment variables can be set in the shell, in the systemd unit, or in
a `deriva-mcp.env` file (search order: `/etc/deriva-mcp/deriva-mcp.env`,
`~/deriva-mcp.env`, `./deriva-mcp.env`).

### Quickstart recipes

**Read-only stdio (single-user dev, no auth setup needed).** Reads
credentials from `~/.deriva/credential.json` for hosts you've authenticated
to via `deriva-globus-auth-utils login --host <hostname>`. No env file
required; mutating tools are off by default.

```jsonc
// ~/.mcp.json
{
  "mcpServers": {
    "deriva": {
      "command": "deriva-mcp-core",
      "args": ["--transport", "stdio"]
    }
  }
}
```

**Full-feature stdio** -- mutation + RAG indexing of deriva-ml docs. Add a
`~/deriva-mcp.env`:

```ini
DERIVA_MCP_DISABLE_MUTATING_TOOLS=false
DERIVA_MCP_RAG_ENABLED=true
DERIVA_MCP_RAG_AUTO_ENRICH=true       # also index per-user catalog rows
```

First server start after enabling RAG downloads the embedding model (~79 MB)
and crawls the `informatics-isi-edu/deriva-ml` `docs/` directory; subsequent
starts skip sources crawled within the last 24 hours.

**HTTP transport (multi-user / production).** See core's
[Deployment Guide](https://github.com/informatics-isi-edu/deriva-mcp-core/blob/main/docs/deployment-guide.md)
for the full Credenza setup; the plugin-specific additions are the same
three RAG / mutation lines as the stdio recipe above.

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

## Design

- Design spec: [`docs/superpowers/specs/2026-04-24-deriva-ml-mcp-design.md`](docs/superpowers/specs/2026-04-24-deriva-ml-mcp-design.md)
- Implementation plan: [`docs/superpowers/plans/2026-04-24-deriva-ml-mcp.md`](docs/superpowers/plans/2026-04-24-deriva-ml-mcp.md)
- Migration coverage tracker: [`docs/coverage.md`](docs/coverage.md)
