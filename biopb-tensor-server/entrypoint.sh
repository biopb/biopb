#!/bin/bash
set -e

# Configuration via environment variables:
# CONFIG_FILE    - Path to TOML config file (if set and exists, uses this file)
#                  Otherwise generates config from env vars below
# DATA_DIR       - Directory to monitor (default: /data)
# MONITOR        - Enable live fs monitoring (default: false, recommended for NFS/Lustre)
# HOST           - gRPC server host (default: 127.0.0.1 for nginx proxy)
# PORT           - gRPC server port (default: 8817 for nginx proxy)
# WEB_HOST       - HTTP sidecar host (default: 127.0.0.1 for nginx proxy)
# WEB_PORT       - HTTP sidecar port (default: 8816)
# NGINX_HTTP_PORT - nginx/webapp HTTP port (default: 8814)
# NGINX_GRPC_PORT - nginx gRPC port (default: 8815)
# COMPUTE_BACKEND - auto/cpu/gpu
# BIOPB_TENSOR_TOKEN - Pre-set access token (skips prompt)
# BIOPB_WEB_DEV_BYPASS - Set to "true" for dev mode
# BIOPB_TMP      - Base temp directory (default: /tmp/biopb-${USER:-$$})
#                  Uses USER env var or PID to avoid multi-user collisions

NGINX_HTTP_PORT="${NGINX_HTTP_PORT:-8814}"
NGINX_GRPC_PORT="${NGINX_GRPC_PORT:-8815}"

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
host = "${HOST:-127.0.0.1}"
port = ${PORT:-8817}

[compute]
backend = "${COMPUTE_BACKEND:-auto}"

[[sources]]
url = "${DATA_DIR}"
monitor = ${MONITOR}
EOF
    CONFIG_FILE="$BIOPB_TMP/runtime-config.toml"
fi

# Copy nginx.conf to temp location and update paths
cp /etc/nginx/nginx.conf "$BIOPB_TMP/nginx.conf"

# Update all /tmp paths in nginx.conf to use our unique prefix
sed -i "s|/tmp/biopb|${BIOPB_TMP}|g" "$BIOPB_TMP/nginx.conf"

# Update nginx ports if not default
if [ "$NGINX_HTTP_PORT" != "8814" ]; then
    sed -i "s/listen 8814;/listen ${NGINX_HTTP_PORT};/" "$BIOPB_TMP/nginx.conf"
fi
if [ "$NGINX_GRPC_PORT" != "8815" ]; then
    sed -i "s/listen 8815 http2;/listen ${NGINX_GRPC_PORT} http2;/" "$BIOPB_TMP/nginx.conf"
fi

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

exec biopb-tensor "$COMMAND" \
    --config "$CONFIG_FILE" \
    --web-host 127.0.0.1 \
    --web-port ${WEB_PORT:-8816} \
    --web-url http://localhost:${NGINX_HTTP_PORT} \
    "$@"