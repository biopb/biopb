#!/bin/bash
set -e

# Configuration via environment variables:
# CONFIG_FILE    - Path to TOML config file (if set and exists, uses this file)
#                  Otherwise generates config from env vars below
# DATA_DIR       - Directory to monitor (default: /data)
# MONITOR        - Enable live fs monitoring (default: true)
# BIOPB_BASE_PORT - Base port for all services (default: 8810)
#                  HTTP=BASE+4, gRPC=BASE+5
# COMPUTE_BACKEND - auto/cpu/gpu
# BIOPB_TENSOR_TOKEN - Access token for webapp and gRPC (auto-generated if not set)
# BIOPB_BIND_LOCALHOST - Set to "true" to bind HTTP to localhost only (Singularity/HPC only)
# BIOPB_EXTERNAL_HOST - External hostname/IP for webapp URL (auto-detected if not set)
# BIOPB_TMP      - Base temp directory (default: /tmp/biopb-${USER:-$$})
# CACHE_MAX_SEGMENT_MB - Max segment size for file cache (default: 256)
# CACHE_MAX_TOTAL_GB   - Max total size for file cache (default: 128)

# Single base port env var - all ports derived from it
# Default 8810 → HTTP=8814, gRPC=8815
BIOPB_BASE_PORT="${BIOPB_BASE_PORT:-8810}"

HTTP_PORT=$((BIOPB_BASE_PORT + 4))
GRPC_PORT=$((BIOPB_BASE_PORT + 5))

echo "Ports: HTTP=$HTTP_PORT gRPC=$GRPC_PORT"

# Create unique temp directory prefix to avoid multi-user collisions on shared /tmp
# Use USER env var if available, else use PID as unique identifier
BIOPB_TMP="${BIOPB_TMP:-/tmp/biopb-${USER:-$$}}"
mkdir -p "$BIOPB_TMP"

# Use existing config file if provided, otherwise generate from env vars
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    echo "Using config file: $CONFIG_FILE"
else
    echo "Generating runtime config from environment variables"
    DATA_DIR="${DATA_DIR:-/data}"
    MONITOR="${MONITOR:-true}"
    cat > "$BIOPB_TMP/runtime-config.toml" << EOF
[server]
host = "127.0.0.1"
port = $GRPC_PORT
aggressive_dir_pruning = true

[cache]
backend = "file"
file_max_segment_mb = ${CACHE_MAX_SEGMENT_MB:-256}
file_max_total_gb = ${CACHE_MAX_TOTAL_GB:-128}

[metadata_db]
enabled = true

[compute]
backend = "${COMPUTE_BACKEND:-auto}"

[[sources]]
url = "${DATA_DIR}"
monitor = $MONITOR
EOF
    CONFIG_FILE="$BIOPB_TMP/runtime-config.toml"
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

# Build command args
COMMAND="${1:-launch}"
shift 2>/dev/null || true

# HTTP bind address: localhost only or all interfaces
HTTP_BIND="0.0.0.0"
if [ "${BIOPB_BIND_LOCALHOST}" = "true" ] || [ "${BIOPB_BIND_LOCALHOST}" = "1" ]; then
    if [ -f "/.dockerenv" ]; then
        echo "WARNING: BIOPB_BIND_LOCALHOST ignored in Docker (would break external access)"
        echo "         Use '-p 127.0.0.1:PORT:PORT' to restrict to localhost instead"
    else
        HTTP_BIND="127.0.0.1"
    fi
fi

ARGS=(
    --config "$CONFIG_FILE"
    --web-host "$HTTP_BIND"
    --web-port "$HTTP_PORT"
    --web-url "http://${WEB_HOST}:${HTTP_PORT}"
    --cors "*"
)

# Add static-dir only if webapp directory exists
if [ -d "/app/webapp" ]; then
    ARGS+=(--static-dir /app/webapp)
fi

exec biopb-tensor-server "$COMMAND" "${ARGS[@]}" "$@"