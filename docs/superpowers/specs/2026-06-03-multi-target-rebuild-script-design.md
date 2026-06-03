# Multi-Target Rebuild Script — Design

**Date:** 2026-06-03
**Status:** Approved

## Goal

Let `scripts/rebuild-deriva-docker-mcp.sh` select between two named
MCP-server targets — **`localhost`** and **`eye-ai`** — via a first
positional argument, and set up the two corresponding env files in
`~/.deriva-docker/env/`.

The two targets differ along three axes:

| Axis | `localhost` | `eye-ai` |
|---|---|---|
| `AUTH_HOSTNAME` | `localhost` (in-container keycloak) | `dev.eye-ai.org` |
| `DERIVA_MCP_EXTRA_PACKAGES` | deriva-ml plugin only (+ deriva-ml + deriva-py) | eye-ai plugin **and** deriva-ml plugin (+ deps) |
| Target catalog host | served via local stack | the eye-ai-loaded catalog |

Everything else — the Docker stack identity, certs, secrets, ports,
`HOSTNAME_MAP` — is shared. Both targets rebuild the **same**
`deriva-mcp-test` service of one shared stack; switching targets just
rebuilds the MCP service with a different package set + auth host.

## Part 1 — Env files in `~/.deriva-docker/env/`

### Rename `localhost.env` → `eye-ai.env`

The current `localhost.env` *is already* the eye-ai config: it auths
against `dev.eye-ai.org` and loads both the eye-ai and deriva-ml
plugins. Renaming it to `eye-ai.env` makes the filename match its
actual role.

**Self-reference fix (only change inside the renamed file):**

- `STACK_ENV_FILE=…/localhost.env` → `STACK_ENV_FILE=…/eye-ai.env`

**Left unchanged** (these describe the shared Docker stack / cert /
secrets infrastructure, NOT which catalog the MCP serves):

- `COMPOSE_PROJECT_NAME=deriva-localhost`
- `CONTAINER_HOSTNAME=localhost`, `AUTHN_SESSION_HOST=localhost`
- `LETSENCRYPT_CERTDIR=…/certs/localhost/…`, `CERT_FILENAME`,
  `KEY_FILENAME`, `CERT_DIR`
- `SECRETS_DIR=…/secrets/localhost/test`
- `DERIVA_MCP_HOSTNAME_MAP={"localhost":"deriva"}`

### Create a new `localhost.env` (deriva-ml-only variant)

A copy of the pre-rename file with exactly three changes:

1. `DERIVA_MCP_EXTRA_PACKAGES`: drop the `eye-ai-deriva-mcp-plugin`
   entry; keep `deriva-ml-mcp-plugin` + `deriva-ml` + `deriva`
   (deriva-py).
2. `AUTH_HOSTNAME=dev.eye-ai.org` → `AUTH_HOSTNAME=localhost`.
3. `STACK_ENV_FILE` stays `…/localhost.env` (matches this file's
   own name).

All stack infra keys are identical to `eye-ai.env`, so both files
drive the same running `deriva-mcp-test` container — only the MCP
package set and auth host differ.

The live `DERIVA_CHATBOT_LLM_API_KEY` value carries over to both files
as-is (it is unrelated to the MCP target choice).

## Part 2 — `scripts/rebuild-deriva-docker-mcp.sh`

Change `$1` from a raw env-file path into a **target name**:

- `$1 ∈ {localhost, eye-ai}` (default `localhost`) resolves to
  `~/.deriva-docker/env/<target>.env`.
- A `$1` that looks like a path (contains `/` or ends in `.env`) is
  still honored as a direct override, preserving existing usage.
- Unknown target names error out with the list of valid targets.
- The usage comment block and the final "tail the logs" hint show the
  resolved env file.

The three `docker-compose` commands (`down` / `build --no-cache` /
`up -d`) are unchanged.

## Testing

- `bash -n` syntax check on the script.
- Resolution logic dry-run: `localhost`→`localhost.env`,
  `eye-ai`→`eye-ai.env`, a raw path passes through, an unknown name
  errors with the valid-targets list.
- Verify both env files exist and that their
  `STACK_ENV_FILE` / `AUTH_HOSTNAME` / `DERIVA_MCP_EXTRA_PACKAGES`
  lines are correct.
- No actual docker rebuild is run.
