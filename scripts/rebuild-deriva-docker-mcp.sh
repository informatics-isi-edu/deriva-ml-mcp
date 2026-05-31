#!/usr/bin/env bash
# Rebuild and restart the deriva-mcp-test container in a deriva-docker
# deployment, picking up new DERIVA_MCP_EXTRA_PACKAGES versions.
#
# Usage:
#   ./scripts/rebuild-deriva-docker-mcp.sh [env-file]
#
# Default env file: ~/.deriva-docker/env/localhost.env
# Default deriva-docker compose dir: $DERIVA_DOCKER_DIR, falling back to
#   $HOME/GitHub/deriva-docker/deriva (the conventional checkout location).
#
# The script `cd`s into that directory so it can be invoked from anywhere
# (e.g. `/path/to/deriva-ml-mcp/scripts/rebuild-deriva-docker-mcp.sh` from
# the deriva-ml-mcp repo). Override with `DERIVA_DOCKER_DIR=/path/to/deriva`
# if your checkout lives elsewhere.
#
# This is a development/testing helper while the deriva-docker support
# for installing the deriva-ml-mcp-plugin package via DERIVA_MCP_EXTRA_PACKAGES
# is still pre-release. When deriva-docker ships final support, this
# script will be replaced by the deriva-docker docs' canonical workflow.

set -euo pipefail

ENV_FILE="${1:-$HOME/.deriva-docker/env/localhost.env}"
SERVICE="${DERIVA_MCP_SERVICE:-deriva-mcp-test}"
DERIVA_DOCKER_DIR="${DERIVA_DOCKER_DIR:-$HOME/GitHub/deriva-docker/deriva}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: env file not found at $ENV_FILE" >&2
    echo "Pass a different path as the first argument, or set up the file." >&2
    exit 1
fi

if [[ ! -f "$DERIVA_DOCKER_DIR/docker-compose.yml" ]]; then
    echo "Error: docker-compose.yml not found at $DERIVA_DOCKER_DIR/docker-compose.yml" >&2
    echo "Set DERIVA_DOCKER_DIR to your deriva-docker checkout's compose directory." >&2
    exit 1
fi

cd "$DERIVA_DOCKER_DIR"
echo ">>> Working from: $DERIVA_DOCKER_DIR"

echo ">>> Stopping $SERVICE..."
docker-compose --env-file "$ENV_FILE" down "$SERVICE"

echo ">>> Rebuilding $SERVICE (--no-cache; picks up new DERIVA_MCP_EXTRA_PACKAGES)..."
docker-compose --env-file "$ENV_FILE" build "$SERVICE" --no-cache

echo ">>> Starting $SERVICE..."
docker-compose --env-file "$ENV_FILE" up -d "$SERVICE"

echo ">>> Done. Tail the logs with:"
echo "    docker-compose --env-file $ENV_FILE logs -f $SERVICE"
