#!/usr/bin/env bash
# Rebuild and restart the deriva-mcp-test container in a deriva-docker
# deployment, picking up new DERIVA_MCP_EXTRA_PACKAGES versions.
#
# Usage:
#   ./scripts/rebuild-deriva-docker-mcp.sh [target|env-file]
#
# The first argument selects which MCP-server target to (re)build:
#   localhost  ->  ~/.deriva-docker/env/localhost.env  (deriva-ml plugin
#                  only; auths against the in-container keycloak)
#   eye-ai     ->  ~/.deriva-docker/env/eye-ai.env     (eye-ai + deriva-ml
#                  plugins; auths against dev.eye-ai.org)
# Both targets rebuild the same shared deriva-mcp-test service -- they
# differ only in the MCP package set and the auth host. Default: localhost.
#
# A first argument that looks like a path (contains '/' or ends in '.env')
# is used verbatim as the env file instead, preserving direct-path usage.
#
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

ENV_DIR="$HOME/.deriva-docker/env"
VALID_TARGETS=(localhost eye-ai)

# Resolve the first argument into an env file. A value containing '/' or
# ending in '.env' is treated as a direct path; otherwise it must name one
# of VALID_TARGETS and resolves to "$ENV_DIR/<target>.env".
TARGET="${1:-localhost}"
if [[ "$TARGET" == */* || "$TARGET" == *.env ]]; then
    ENV_FILE="$TARGET"
else
    valid=0
    for t in "${VALID_TARGETS[@]}"; do
        [[ "$TARGET" == "$t" ]] && valid=1 && break
    done
    if [[ "$valid" -ne 1 ]]; then
        echo "Error: unknown target '$TARGET'." >&2
        echo "Valid targets: ${VALID_TARGETS[*]} (or pass a path to an env file)." >&2
        exit 1
    fi
    ENV_FILE="$ENV_DIR/$TARGET.env"
fi

SERVICE="${DERIVA_MCP_SERVICE:-deriva-mcp-test}"
DERIVA_DOCKER_DIR="${DERIVA_DOCKER_DIR:-$HOME/GitHub/deriva-docker/deriva}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: env file not found at $ENV_FILE" >&2
    echo "Pass a valid target (${VALID_TARGETS[*]}) or a path as the first argument." >&2
    exit 1
fi

echo ">>> Target env file: $ENV_FILE"

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
