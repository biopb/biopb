#!/bin/bash
set -e

# Configuration via environment variables:
# CONFIG_FILE    - Path to TOML config file (if set and exists, uses this file)
#                  Otherwise generates config from env vars below
# DATA_DIR       - Directory to monitor (default: /data)
# MONITOR        - Enable live fs monitoring (default: true)
# BIOPB_BASE_PORT - Base port for all services (default: 8810)
#                  HTTP=BASE+4, gRPC=BASE+5, Sidecar=BASE+6, Flight=BASE+7
# COMPUTE_BACKEND - auto/cpu/gpu
# BIOPB_TENSOR_TOKEN - Access token for webapp and gRPC (auto-generated if not set)
# BIOPB_WEB_DEV_BYPASS - Set to "true" for dev mode
# BIOPB_EXTERNAL_HOST - External hostname/IP for webapp URL (auto-detected if not set)
# BIOPB_TMP      - Base temp directory (default: /tmp/biopb-${USER:-$$})

# Single base port env var - all ports derived from it
# Default 8810 → HTTP=8814, gRPC=8815, Sidecar=8816, Flight=8817
BIOPB_BASE_PORT="${BIOPB_BASE_PORT:-8810}"

# Find available port starting from base, scanning upward
find_available_port() {
    local base=$1
    local max_attempts=100
    for port in $(seq $base $((base + max_attempts))); do
        if ! ss -tuln 2>/dev/null | grep -q ":$port " && \
           ! netstat -tuln 2>/dev/null | grep -q ":$port "; then
            echo $port
            return 0
        fi
    done
    echo $base  # fallback
}

# Derive all ports sequentially, using previous port as starting point
# This ensures no overlap between discovered ports
NGINX_HTTP_PORT=$(find_available_port $((BIOPB_BASE_PORT + 4)))
NGINX_GRPC_PORT=$(find_available_port $((NGINX_HTTP_PORT + 1)))
WEB_PORT=$(find_available_port $((NGINX_GRPC_PORT + 1)))
PORT=$(find_available_port $((WEB_PORT + 1)))

echo "Ports: HTTP=$NGINX_HTTP_PORT gRPC=$NGINX_GRPC_PORT Sidecar=$WEB_PORT Flight=$PORT"

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
port = $PORT

[compute]
backend = "${COMPUTE_BACKEND:-auto}"

[[sources]]
url = "${DATA_DIR}"
monitor = $MONITOR
EOF
    CONFIG_FILE="$BIOPB_TMP/runtime-config.toml"
fi

# Copy nginx.conf to temp location and update paths
cp /etc/nginx/nginx.conf "$BIOPB_TMP/nginx.conf"

# Update all /tmp paths in nginx.conf to use our unique prefix
sed -i "s|/tmp/biopb|${BIOPB_TMP}|g" "$BIOPB_TMP/nginx.conf"

# Update nginx listen ports (bind to localhost only in dev bypass mode for security)
if [ "${BIOPB_WEB_DEV_BYPASS}" = "true" ] || [ "${BIOPB_WEB_DEV_BYPASS}" = "1" ]; then
    sed -i "s/listen 8814;/listen 127.0.0.1:${NGINX_HTTP_PORT};/" "$BIOPB_TMP/nginx.conf"
    sed -i "s/listen 8815;/listen 127.0.0.1:${NGINX_GRPC_PORT};/" "$BIOPB_TMP/nginx.conf"
else
    sed -i "s/listen 8814;/listen ${NGINX_HTTP_PORT};/" "$BIOPB_TMP/nginx.conf"
    sed -i "s/listen 8815;/listen ${NGINX_GRPC_PORT};/" "$BIOPB_TMP/nginx.conf"
fi

# Update nginx proxy targets to use discovered internal ports
sed -i "s/:8817/:${PORT}/g" "$BIOPB_TMP/nginx.conf"
sed -i "s/:8816/:${WEB_PORT}/g" "$BIOPB_TMP/nginx.conf"

# Create nginx temp directories
mkdir -p "$BIOPB_TMP/nginx_client_body" "$BIOPB_TMP/nginx_proxy" "$BIOPB_TMP/nginx_fastcgi" "$BIOPB_TMP/nginx_uwsgi" "$BIOPB_TMP/nginx_scgi"

# Start nginx using temp config
nginx -c "$BIOPB_TMP/nginx.conf"

# Start tensor server (foreground process)
# First argument is the subcommand (launch/serve), rest are passed through
COMMAND="${1:-launch}"
shift 2>/dev/null || true

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

exec biopb-tensor-server "$COMMAND" \
    --config "$CONFIG_FILE" \
    --web-host 127.0.0.1 \
    --web-port ${WEB_PORT} \
    --web-url "http://${WEB_HOST}:${NGINX_HTTP_PORT}" \
    --cors "*" \
    "$@"