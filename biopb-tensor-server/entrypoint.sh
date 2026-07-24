#!/bin/bash
set -e

# Headless data-plane entrypoint. Runs `biopb-tensor-server launch` (Arrow Flight
# server + FastAPI HTTP sidecar) in the foreground as PID 1. There is no control
# plane and no bundled webapp in this image.
#
# Configuration via environment variables:
# CONFIG_FILE    - Path to a JSON config file (if set and exists,
#                  uses this file). Otherwise generates JSON config from env vars below
# DATA_DIR       - Directory to monitor (default: /data)
# MONITOR        - Enable live fs monitoring (default: true)
# BIOPB_BASE_PORT - Base port for all services (default: 8810)
#                  HTTP sidecar=BASE+4, gRPC Flight=BASE+5
# COMPUTE_BACKEND - auto/cpu/gpu
# BIOPB_TENSOR_TOKEN - Access token for the sidecar and gRPC (auto-generated if not set)
# BIOPB_BIND_LOCALHOST - Set to "true" to bind both HTTP and gRPC to localhost only (Singularity/HPC only)
# BIOPB_EXTERNAL_HOST - External hostname/IP for the printed access URL (auto-detected if not set)
# BIOPB_TMP      - Base temp directory (default: /tmp/biopb-${USER:-$$})
# CACHE_MAX_SEGMENT_MB - Max segment size for file cache (default: 256)
# CACHE_MAX_TOTAL_GB   - Max total size for file cache (default: 16)

# Single base port env var - all ports derived from it
# Default 8810 → HTTP=8814, gRPC=8815
BIOPB_BASE_PORT="${BIOPB_BASE_PORT:-8810}"

HTTP_PORT=$((BIOPB_BASE_PORT + 4))
GRPC_PORT=$((BIOPB_BASE_PORT + 5))

# HTTP_PORT is the public data-plane API (the FastAPI sidecar). GRPC_PORT (Flight)
# is the direct Arrow Flight endpoint for SDK clients. Both are token-authenticated.
echo "Ports: HTTP(sidecar)=$HTTP_PORT  gRPC=$GRPC_PORT"

# Create unique temp directory prefix to avoid multi-user collisions on shared /tmp
# Use USER env var if available, else use PID as unique identifier
BIOPB_TMP="${BIOPB_TMP:-/tmp/biopb-${USER:-$$}}"
mkdir -p "$BIOPB_TMP"

# Bind address shared by the gRPC Flight server ([server] host, below) and the
# HTTP sidecar (--web-host). Default: all interfaces, so Docker's -p forwarding
# reaches both services (each is token-authenticated). BIOPB_BIND_LOCALHOST
# restricts both to loopback for shared HPC nodes; ignored in Docker, where a
# 127.0.0.1 bind inside the container cannot be reached through -p forwarding.
BIND_ADDR="0.0.0.0"
if [ "${BIOPB_BIND_LOCALHOST}" = "true" ] || [ "${BIOPB_BIND_LOCALHOST}" = "1" ]; then
    if [ -f "/.dockerenv" ]; then
        echo "WARNING: BIOPB_BIND_LOCALHOST ignored in Docker (would break external access)"
        echo "         Use '-p 127.0.0.1:PORT:PORT' to restrict to localhost instead"
    else
        BIND_ADDR="127.0.0.1"
    fi
fi

# Use existing config file if provided, otherwise generate from env vars. A
# supplied CONFIG_FILE owns [server].host/port (the Flight bind); keep
# [server].port == BASE+5 so the published gRPC port matches. The sidecar's own
# HTTP port is set by --web-port below regardless of the config.
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    echo "Using config file: $CONFIG_FILE"
else
    echo "Generating runtime config from environment variables"
    DATA_DIR="${DATA_DIR:-/data}"
    MONITOR="${MONITOR:-true}"
    cat > "$BIOPB_TMP/runtime-config.json" << EOF
{
  "server": {
    "host": "$BIND_ADDR",
    "port": $GRPC_PORT,
    "aggressive_dir_pruning": true
  },
  "cache": {
    "backend": "file",
    "file_max_segment_mb": ${CACHE_MAX_SEGMENT_MB:-256},
    "file_max_total_gb": ${CACHE_MAX_TOTAL_GB:-16}
  },
  "compute": {
    "backend": "${COMPUTE_BACKEND:-auto}"
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

# Construct best-effort external URL for the printed access URL
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

# Resolve the tensor-server access token. `launch` re-validates the token on both
# the Flight gRPC port and the HTTP sidecar. A public bind with no token would be
# open, so generate one; a loopback-only bind (BIOPB_BIND_LOCALHOST) runs in local
# mode with no token. `launch` also enforces this fail-closed, but generating here
# keeps the token in the logs (and the env) deterministically.
TOKEN_ARGS=()
if [ -n "$BIOPB_TENSOR_TOKEN" ]; then
    TOKEN_ARGS=(--token "$BIOPB_TENSOR_TOKEN")
elif [ "$BIND_ADDR" = "127.0.0.1" ]; then
    # Loopback-only bind (Singularity BIOPB_BIND_LOCALHOST): local mode, no token.
    # Every listener is same-machine, so no token is enforced -- pass nothing.
    TOKEN_ARGS=()
else
    GEN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    export BIOPB_TENSOR_TOKEN="$GEN_TOKEN"
    echo "Generated access token: $GEN_TOKEN"
    TOKEN_ARGS=(--token "$GEN_TOKEN")
fi

# Run `biopb-tensor-server launch` in the foreground (container PID 1). It starts
# the Arrow Flight server (binds [server].host/port from the config) and the
# FastAPI HTTP sidecar (--web-host/--web-port) in a single process.
#   --web-host $BIND_ADDR : bind the sidecar publicly (0.0.0.0 -> reachable via -p)
#   --web-port $HTTP_PORT : the published data-plane API port
#   --web-url             : external origin used only for the printed access URL
#
# PID 1 note: launch installs a SIGTERM handler so `docker stop` tears down
# gracefully (releasing the file-cache process lock). PID 1 does not reap
# *unrelated* orphaned grandchildren -- run the container with `docker run --init`
# (or a tini shim) if you want a reaping init as PID 1.
LAUNCH_ARGS=(
    --config "$CONFIG_FILE"
    --web-host "$BIND_ADDR"
    --web-port "$HTTP_PORT"
    --web-url "http://${WEB_HOST}:${HTTP_PORT}"
)

echo "HTTP sidecar: http://${WEB_HOST}:${HTTP_PORT}   Flight gRPC: ${WEB_HOST}:${GRPC_PORT}"
exec biopb-tensor-server launch "${LAUNCH_ARGS[@]}" "${TOKEN_ARGS[@]}"
