#!/usr/bin/env bash
# Rebuild and restart the deriva-mcp-test container in a deriva-docker
# deployment, picking up new DERIVA_MCP_EXTRA_PACKAGES versions.
#
# Usage:
#   ./scripts/rebuild-deriva-docker-mcp.sh [env-file]
#
# Default env file: ~/.deriva-docker/env/localhost.env
#
# This is a development/testing helper while the deriva-docker support
# for installing the deriva-ml-mcp plugin via DERIVA_MCP_EXTRA_PACKAGES
# is still pre-release. When deriva-docker ships final support, this
# script will be replaced by the deriva-docker docs' canonical workflow.

set -euo pipefail

ENV_FILE="${1:-$HOME/.deriva-docker/env/localhost.env}"
SERVICE="${DERIVA_MCP_SERVICE:-deriva-mcp-test}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: env file not found at $ENV_FILE" >&2
    echo "Pass a different path as the first argument, or set up the file." >&2
    exit 1
fi

echo ">>> Stopping $SERVICE..."
docker-compose --env-file "$ENV_FILE" down "$SERVICE"

echo ">>> Rebuilding $SERVICE (--no-cache; picks up new DERIVA_MCP_EXTRA_PACKAGES)..."
docker-compose --env-file "$ENV_FILE" build "$SERVICE" --no-cache

echo ">>> Starting $SERVICE..."
docker-compose --env-file "$ENV_FILE" up -d "$SERVICE"

echo ">>> Done. Tail the logs with:"
echo "    docker-compose --env-file $ENV_FILE logs -f $SERVICE"
