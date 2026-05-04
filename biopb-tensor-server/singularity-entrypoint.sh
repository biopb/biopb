#!/bin/bash
set -e

# Configuration via environment variables:
# CONFIG_FILE   - Path to TOML config file (default: /app/config/default-config.toml)
# NGINX_PORT    - nginx/webapp port (default: 80, use higher port on HPC)
# WEB_HOST      - HTTP sidecar host (default: 127.0.0.1 for nginx proxy)
# WEB_PORT      - HTTP sidecar port (default: 8816)
# BIOPB_TENSOR_TOKEN - Pre-set access token (skips prompt)
# BIOPB_WEB_DEV_BYPASS - Set to "true" for dev mode
# BIOPB_TMP     - Base temp directory (default: /tmp/biopb-${USER:-$$})

CONFIG_FILE="${CONFIG_FILE:-/app/config/default-config.toml}"
NGINX_PORT="${NGINX_PORT:-80}"

# Create unique temp directory prefix to avoid multi-user collisions on shared /tmp
BIOPB_TMP="${BIOPB_TMP:-/tmp/biopb-${USER:-$$}}"
mkdir -p "$BIOPB_TMP"

# Copy nginx.conf to temp location and update paths
cp /etc/nginx/nginx.conf "$BIOPB_TMP/nginx.conf"

# Update all /tmp paths in nginx.conf to use our unique prefix
sed -i "s|/tmp/biopb|${BIOPB_TMP}|g" "$BIOPB_TMP/nginx.conf"

# Update nginx port if not default
if [ "$NGINX_PORT" != "80" ]; then
    sed -i "s/listen 80;/listen ${NGINX_PORT};/" "$BIOPB_TMP/nginx.conf"
fi

# Create nginx temp directories
mkdir -p "$BIOPB_TMP/nginx_client_body" "$BIOPB_TMP/nginx_proxy" \
         "$BIOPB_TMP/nginx_fastcgi" "$BIOPB_TMP/nginx_uwsgi" "$BIOPB_TMP/nginx_scgi"

# Start nginx using temp config
nginx -c "$BIOPB_TMP/nginx.conf"

# Start tensor server (foreground process)
COMMAND="${1:-launch}"
shift 2>/dev/null || true

exec biopb-tensor "$COMMAND" \
    --config "$CONFIG_FILE" \
    --web-host 127.0.0.1 \
    --web-port ${WEB_PORT:-8816} \
    --web-url http://localhost \
    "$@"