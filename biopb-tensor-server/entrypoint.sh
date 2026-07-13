#!/bin/bash
set -e

# Configuration via environment variables:
# CONFIG_FILE    - Path to JSON (or legacy TOML) config file (if set and exists,
#                  uses this file). Otherwise generates JSON config from env vars below
# DATA_DIR       - Directory to monitor (default: /data)
# MONITOR        - Enable live fs monitoring (default: true)
# BIOPB_BASE_PORT - Base port for all services (default: 8810)
#                  HTTP=BASE+4, gRPC=BASE+5
# COMPUTE_BACKEND - auto/cpu/gpu
# BIOPB_TENSOR_TOKEN - Access token for webapp and gRPC (auto-generated if not set)
# BIOPB_BIND_LOCALHOST - Set to "true" to bind both HTTP and gRPC to localhost only (Singularity/HPC only)
# BIOPB_EXTERNAL_HOST - External hostname/IP for webapp URL (auto-detected if not set)
# BIOPB_TMP      - Base temp directory (default: /tmp/biopb-${USER:-$$})
# CACHE_MAX_SEGMENT_MB - Max segment size for file cache (default: 256)
# CACHE_MAX_TOTAL_GB   - Max total size for file cache (default: 16)

# Single base port env var - all ports derived from it
# Default 8810 → HTTP=8814, gRPC=8815
BIOPB_BASE_PORT="${BIOPB_BASE_PORT:-8810}"

CONTROL_PORT=$((BIOPB_BASE_PORT + 3))
HTTP_PORT=$((BIOPB_BASE_PORT + 4))
GRPC_PORT=$((BIOPB_BASE_PORT + 5))

# CONTROL_PORT is the single public web origin (the control plane). HTTP_PORT is
# the tensor sidecar, now PRIVATE (loopback) behind the control. GRPC_PORT (Flight)
# stays public for SDK clients. See mcp-dedaemonization-migration.md §6.1.
echo "Ports: CONTROL(web)=$CONTROL_PORT  HTTP(sidecar, private)=$HTTP_PORT  gRPC=$GRPC_PORT"

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

# Use existing config file if provided, otherwise generate from env vars.
# NOTE: a supplied CONFIG_FILE owns [server].host/port, and the control probes the
# data plane's liveness at 127.0.0.1:$GRPC_PORT (BIOPB_BASE_PORT+5). Keep
# [server].port == BASE+5 and [server].host reachable over loopback (0.0.0.0 or
# 127.0.0.1); a config that binds Flight elsewhere leaves the control unable to
# see the plane as "serving" (the sidecar HTTP port is overridden to $HTTP_PORT
# regardless). The generated config below already satisfies this.
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

# Enable debug logging if dev bypass mode is active
if [ "${BIOPB_WEB_DEV_BYPASS}" = "true" ] || [ "${BIOPB_WEB_DEV_BYPASS}" = "1" ]; then
    export BIOPB_LOG_LEVEL="${BIOPB_LOG_LEVEL:-DEBUG}"
fi

# Construct best-effort external URL for webapp access
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

# Resolve the tensor-server access token. The control forwards the browser's
# Bearer token to the (private) sidecar, which re-validates; the Flight gRPC port
# re-validates it too. A public bind with no token would be open, so generate one
# (mirroring `biopb server start`); a loopback-only bind may skip enforcement.
TOKEN_ARGS=()
if [ -n "$BIOPB_TENSOR_TOKEN" ]; then
    TOKEN_ARGS=(--token "$BIOPB_TENSOR_TOKEN")
elif [ "$BIND_ADDR" = "127.0.0.1" ]; then
    TOKEN_ARGS=(--local-bypass)
else
    GEN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    export BIOPB_TENSOR_TOKEN="$GEN_TOKEN"
    echo "Generated access token: $GEN_TOKEN"
    TOKEN_ARGS=(--token "$GEN_TOKEN")
fi

# Run the control plane in the foreground (container PID 1). It is the single
# public web origin on $CONTROL_PORT: it serves the web/ SPA bundle (/app/webapp,
# passed via --static-dir above) at its root (dashboard /, dataviewer /viewer,
# per-session observe), supervises the tensor server as a PRIVATE loopback
# subprocess (sidecar on 127.0.0.1:$HTTP_PORT, never exposed), and reverse-proxies
# its data API under /data_plane. The Flight gRPC port stays directly exposed for
# SDK clients.
#
# PID 1 note: run_control installs SIGTERM/SIGINT handlers so `docker stop` tears
# down gracefully, and the supervisor reaps its own tensor-server child. But PID 1
# does not reap *unrelated* orphaned grandchildren -- run the container with
# `docker run --init` (or a tini shim) if you want a reaping init as PID 1.
#   --control-host $BIND_ADDR : bind the public origin (0.0.0.0 -> reachable via -p)
#   --grpc-host 127.0.0.1     : the liveness-probe CONNECT address; the server BINDs
#                               server.host from the config (0.0.0.0 above), so
#                               Flight itself stays public
#   --web-host 127.0.0.1      : bind the sidecar privately (only the control reaches it)
CONTROL_ARGS=(
    --config "$CONFIG_FILE"
    --control-host "$BIND_ADDR"
    --control-port "$CONTROL_PORT"
    --grpc-host 127.0.0.1
    --grpc-port "$GRPC_PORT"
    --web-host 127.0.0.1
    --web-port "$HTTP_PORT"
)
if [ -d "/app/webapp" ]; then
    CONTROL_ARGS+=(--static-dir /app/webapp)
fi

echo "Web origin (control): http://${WEB_HOST}:${CONTROL_PORT}   Flight gRPC: ${WEB_HOST}:${GRPC_PORT}"
exec python -m biopb_control run "${CONTROL_ARGS[@]}" "${TOKEN_ARGS[@]}"
