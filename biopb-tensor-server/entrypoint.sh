#!/bin/bash
set -e

# Headless data-plane entrypoint. By default runs `biopb-tensor-server serve`
# (Arrow Flight only) in the foreground as PID 1: the container is a pure gRPC
# data-plane endpoint with no HTTP surface at all (biopb/biopb#604 case 2).
# Browsing that data happens downstream — a machine running the full biopb stack
# adds this container as a `grpc://`/`grpcs://` remote source, so its browser
# only ever talks to its own loopback origin.
#
# Set BIOPB_ENABLE_HTTP_SIDECAR=1 to get the old shape back (`launch`: Flight +
# the FastAPI HTTP sidecar), for a browser app served elsewhere that calls the
# data-plane HTTP API directly.
#
# Configuration via environment variables:
# CONFIG_FILE    - Path to a JSON config file (if set and exists,
#                  uses this file). Otherwise generates JSON config from env vars below
# DATA_DIR       - Directory to monitor (default: /data)
# MONITOR        - Enable live fs monitoring (default: true)
# BIOPB_BASE_PORT - Base port for all services (default: 8810)
#                  gRPC Flight=BASE+5; HTTP sidecar=BASE+4 (only with the sidecar opt-in)
# BIOPB_ENABLE_HTTP_SIDECAR - Truthy (1/true/yes/on) to also serve the FastAPI
#                  data-plane HTTP API on BASE+4 (runs `launch` instead of `serve`).
#                  Off by default: the container is Flight-only.
# BIOPB_TENSOR_TLS - Truthy to serve Flight over TLS with the self-signed cert in
#                  the state dir (auto-generated on first use); clients dial
#                  grpcs:// and pin it on first connect (TOFU). The cert lives in
#                  the container's state dir, so mount a volume at
#                  /root/.local/state (or set BIOPB_STATE_HOME) to keep the same
#                  cert across `docker rm` -- otherwise every recreate mints a new
#                  one and pinned clients refuse to reconnect.
# BIOPB_TLS_CERT / BIOPB_TLS_KEY - Paths (in-container) to a BYO PEM cert + key to
#                  serve instead of the self-signed one. Takes precedence over
#                  BIOPB_TENSOR_TLS; both must be set together. With the sidecar
#                  opted in, the cert must carry a loopback SAN (localhost /
#                  127.0.0.1) -- that is how the co-located sidecar reaches Flight.
# BIOPB_TENSOR_TOKEN - Access token for gRPC (and the sidecar, if enabled;
#                  auto-generated if not set)
# BIOPB_TENSOR_ALLOW_NO_TOKEN - Set truthy (1/true/yes/on, case-insensitive) to serve
#                  the data API WITHOUT a token even on the public 0.0.0.0 bind
#                  (insecure; trusted networks only). Ignored when BIOPB_TENSOR_TOKEN is set.
# BIOPB_BIND_LOCALHOST - Set to "true" to bind to localhost only (Singularity/HPC only)
# BIOPB_EXTERNAL_HOST - External hostname/IP shown in the printed endpoint URLs only;
#                  display-only, does not affect binding (auto-detected if not set)
# BIOPB_CORS_ORIGINS - Space-separated extra CORS origins (→ launch --cors,
#                  repeatable). Only meaningful with BIOPB_ENABLE_HTTP_SIDECAR;
#                  set this to allow a browser SPA served from a different
#                  origin (e.g. "http://localhost:5173 http://my.host:8813").
# BIOPB_TMP      - Base temp directory (default: /tmp/biopb-${USER:-$$})
# CACHE_MAX_SEGMENT_MB - Max segment size for file cache (unset: server default, ~64 MB)
# CACHE_MAX_TOTAL_GB   - Max total size for file cache (default: 16)

# Normalize a boolean env var exactly like the Python predicate
# _allow_no_token_from_env() (strip + lowercase, accept 1/true/yes/on), so the
# shell and Python readings of the same variable can never diverge.
#
# Lowercasing goes through `tr`, not `${v,,}`: that expansion needs bash 4, and
# macOS still ships bash 3.2 -- which parses the whole script before running it,
# so a single `${v,,}` makes the file unusable there, not just this branch.
_is_truthy() {
    local v="$1"
    v="${v#"${v%%[![:space:]]*}"}"
    v="${v%"${v##*[![:space:]]}"}"
    case "$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# Single base port env var - all ports derived from it
# Default 8810 → gRPC=8815 (HTTP sidecar=8814 when opted in)
BIOPB_BASE_PORT="${BIOPB_BASE_PORT:-8810}"

HTTP_PORT=$((BIOPB_BASE_PORT + 4))
GRPC_PORT=$((BIOPB_BASE_PORT + 5))

ENABLE_SIDECAR=false
if _is_truthy "$BIOPB_ENABLE_HTTP_SIDECAR"; then
    ENABLE_SIDECAR=true
fi

# GRPC_PORT (Flight) is the data-plane endpoint for SDK clients. HTTP_PORT is the
# FastAPI sidecar, served only when opted in. Both are token-authenticated.
if [ "$ENABLE_SIDECAR" = true ]; then
    echo "Ports: gRPC=$GRPC_PORT  HTTP(sidecar)=$HTTP_PORT"
else
    echo "Ports: gRPC=$GRPC_PORT  (Flight only; set BIOPB_ENABLE_HTTP_SIDECAR=1 for the HTTP API)"
fi

# Create unique temp directory prefix to avoid multi-user collisions on shared /tmp
# Use USER env var if available, else use PID as unique identifier
BIOPB_TMP="${BIOPB_TMP:-/tmp/biopb-${USER:-$$}}"
mkdir -p "$BIOPB_TMP"

# Bind address for the gRPC Flight server ([server] host, below) and, when
# enabled, the HTTP sidecar (--web-host). Default: all interfaces, so Docker's -p
# forwarding reaches the service (which is token-authenticated).
# BIOPB_BIND_LOCALHOST restricts to loopback for shared HPC nodes; ignored in
# Docker, where a 127.0.0.1 bind inside the container cannot be reached through
# -p forwarding.
BIND_ADDR="0.0.0.0"
if [ "${BIOPB_BIND_LOCALHOST}" = "true" ] || [ "${BIOPB_BIND_LOCALHOST}" = "1" ]; then
    if [ -f "/.dockerenv" ]; then
        echo "WARNING: BIOPB_BIND_LOCALHOST ignored in Docker (would break external access)"
        echo "         Use '-p 127.0.0.1:PORT:PORT' to restrict to localhost instead"
    else
        BIND_ADDR="127.0.0.1"
    fi
fi

# Use existing config file if provided, otherwise generate from env vars. The
# Flight bind is NOT in the config any more (biopb/biopb#604): --host/--port are
# passed below, so a supplied CONFIG_FILE cannot move the published gRPC port out
# from under the -p mapping. It still owns everything else (sources, cache, ...).
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    echo "Using config file: $CONFIG_FILE"
else
    echo "Generating runtime config from environment variables"
    DATA_DIR="${DATA_DIR:-/data}"
    MONITOR="${MONITOR:-true}"
    # Segment cap: omit the key unless explicitly set, so the server's built-in
    # default (64 MB) applies. Set CACHE_MAX_SEGMENT_MB only to override it.
    SEGMENT_CFG=""
    if [ -n "$CACHE_MAX_SEGMENT_MB" ]; then
        SEGMENT_CFG="    \"file_max_segment_mb\": ${CACHE_MAX_SEGMENT_MB},"
    fi
    cat > "$BIOPB_TMP/runtime-config.json" << EOF
{
  "server": {
    "aggressive_dir_pruning": true
  },
  "cache": {
    "backend": "file",
${SEGMENT_CFG}
    "file_max_total_gb": ${CACHE_MAX_TOTAL_GB:-16}
  },
  "sources": [
    {
      "url": "${DATA_DIR}",
      "monitor": $MONITOR
    }
  ]
}
EOF
    CONFIG_FILE="$BIOPB_TMP/runtime-config.json"
fi

# Construct best-effort external host for the printed endpoint URLs
# Priority: env var override > hostname > IP from default route > localhost
if [ -n "$BIOPB_EXTERNAL_HOST" ]; then
    WEB_HOST="$BIOPB_EXTERNAL_HOST"
elif hostname -f 2>/dev/null | grep -q '\.'; then
    # Has FQDN (e.g., server.example.com)
    WEB_HOST="$(hostname -f)"
elif hostname -I 2>/dev/null | grep -qE '^[0-9]+\.'; then
    # Pick first non-localhost IP from hostname -I output
    WEB_HOST="$(hostname -I | awk '{for(i=1;i<=NF;i++) if($i !~ /^127\./ && $i !~ /^169\.254\./) {print $i; exit}}')"
    # Fallback if nothing found
    [ -z "$WEB_HOST" ] && WEB_HOST="$(hostname -I | awk '{print $1}')"
else
    WEB_HOST="localhost"
fi

# Resolve TLS material for the Flight plane. A BYO cert wins over the self-signed
# one; both files must be given together (`serve`/`launch` re-validate and refuse
# one). Both commands take these flags: the sidecar reaches the Flight plane over
# loopback and trusts the same certificate directly, so TLS and the sidecar are
# no longer mutually exclusive. A BYO cert must carry a loopback SAN for that dial
# to pass gRPC's name check.
TLS_ARGS=()
if [ -n "$BIOPB_TLS_CERT" ] || [ -n "$BIOPB_TLS_KEY" ]; then
    if [ -z "$BIOPB_TLS_CERT" ] || [ -z "$BIOPB_TLS_KEY" ]; then
        echo "ERROR: BIOPB_TLS_CERT and BIOPB_TLS_KEY must be set together." >&2
        exit 2
    fi
    TLS_ARGS=(--tls-cert "$BIOPB_TLS_CERT" --tls-key "$BIOPB_TLS_KEY")
elif _is_truthy "$BIOPB_TENSOR_TLS"; then
    TLS_ARGS=(--tls)
fi

# Resolve the tensor-server access token. `serve`/`launch` re-validate it on the
# Flight gRPC port (and the sidecar, if enabled). A public bind with no token
# would be open, so generate one; a loopback-only bind (BIOPB_BIND_LOCALHOST)
# runs in local mode with no token. The CLI also enforces this fail-closed, but
# generating here keeps the token in the logs (and the env) deterministically.
TOKEN_ARGS=()
if [ -n "$BIOPB_TENSOR_TOKEN" ]; then
    TOKEN_ARGS=(--token "$BIOPB_TENSOR_TOKEN")
elif [ "$BIND_ADDR" = "127.0.0.1" ]; then
    # Loopback-only bind (Singularity BIOPB_BIND_LOCALHOST): local mode, no token.
    # Every listener is same-machine, so no token is enforced -- pass nothing.
    TOKEN_ARGS=()
elif _is_truthy "$BIOPB_TENSOR_ALLOW_NO_TOKEN"; then
    # Deliberate insecure opt-out: run tokenless even on the public 0.0.0.0 bind
    # (e.g. a host-loopback-published container on a trusted machine). Pass no
    # --token and skip generation; the CLI honors the same env var and serves the
    # data API OPEN. The var is already in the container env (-e), so it sees it.
    echo "WARNING: BIOPB_TENSOR_ALLOW_NO_TOKEN set -- serving the data API WITHOUT a token. Trusted networks only."
    TOKEN_ARGS=()
else
    GEN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    export BIOPB_TENSOR_TOKEN="$GEN_TOKEN"
    echo "Generated access token: $GEN_TOKEN"
    TOKEN_ARGS=(--token "$GEN_TOKEN")
fi

# Run the server in the foreground (container PID 1).
#
# PID 1 note: serve/launch install a SIGTERM handler so `docker stop` tears down
# gracefully (releasing the file-cache process lock). PID 1 does not reap
# *unrelated* orphaned grandchildren -- run the container with `docker run --init`
# (or a tini shim) if you want a reaping init as PID 1.
if [ "$ENABLE_SIDECAR" = true ]; then
    # Flight + the FastAPI HTTP sidecar in a single process. CORS is configurable
    # via BIOPB_CORS_ORIGINS (→ repeated --cors) for a browser SPA served from a
    # different origin; otherwise launch defaults to localhost variants.
    LAUNCH_ARGS=(
        --config "$CONFIG_FILE"
        --host "$BIND_ADDR"
        --port "$GRPC_PORT"
        --web-host "$BIND_ADDR"
        --web-port "$HTTP_PORT"
    )
    for origin in $BIOPB_CORS_ORIGINS; do
        LAUNCH_ARGS+=(--cors "$origin")
    done
    GRPC_SCHEME="grpc"
    [ ${#TLS_ARGS[@]} -gt 0 ] && GRPC_SCHEME="grpcs"
    echo "HTTP sidecar: http://${WEB_HOST}:${HTTP_PORT}   Flight: ${GRPC_SCHEME}://${WEB_HOST}:${GRPC_PORT}"
    exec biopb-tensor-server launch "${LAUNCH_ARGS[@]}" \
        "${TLS_ARGS[@]}" "${TOKEN_ARGS[@]}"
else
    # Flight only: a pure gRPC data-plane endpoint, no HTTP surface.
    GRPC_SCHEME="grpc"
    [ ${#TLS_ARGS[@]} -gt 0 ] && GRPC_SCHEME="grpcs"
    echo "Flight: ${GRPC_SCHEME}://${WEB_HOST}:${GRPC_PORT}"
    exec biopb-tensor-server serve --config "$CONFIG_FILE" \
        --host "$BIND_ADDR" --port "$GRPC_PORT" \
        "${TLS_ARGS[@]}" "${TOKEN_ARGS[@]}"
fi
