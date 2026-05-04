#!/bin/bash
set -e

# Configuration via environment variables:
# DATA_DIR      - Directory to monitor (default: /data)
# MONITOR       - Enable live fs monitoring (default: false, recommended for NFS/Lustre)
# HOST          - gRPC server host (default: 0.0.0.0)
# PORT          - gRPC server port (default: 8815)
# WEB_HOST      - HTTP sidecar host (default: 127.0.0.1 for nginx proxy)
# WEB_PORT      - HTTP sidecar port (default: 8816)
# NGINX_PORT    - nginx/webapp port (default: 80, use higher port on HPC)
# COMPUTE_BACKEND - auto/cpu/gpu
# BIOPB_TENSOR_TOKEN - Pre-set access token (skips prompt)
# BIOPB_WEB_DEV_BYPASS - Set to "true" for dev mode
# BIOPB_TMP     - Base temp directory (default: /tmp/biopb-${USER:-$$})
#                 Uses USER env var or PID to avoid multi-user collisions

DATA_DIR="${DATA_DIR:-/data}"
MONITOR="${MONITOR:-false}"
NGINX_PORT="${NGINX_PORT:-80}"

# Create unique temp directory prefix to avoid multi-user collisions on shared /tmp
# Use USER env var if available, else use PID as unique identifier
BIOPB_TMP="${BIOPB_TMP:-/tmp/biopb-${USER:-$$}}"
mkdir -p "$BIOPB_TMP"

# Generate runtime config
cat > "$BIOPB_TMP/runtime-config.toml" << EOF
[server]
host = "${HOST:-0.0.0.0}"
port = ${PORT:-8815}

[compute]
backend = "${COMPUTE_BACKEND:-auto}"

[[sources]]
url = "${DATA_DIR}"
monitor = ${MONITOR}
EOF

# Copy nginx.conf to temp location and update paths
cp /etc/nginx/nginx.conf "$BIOPB_TMP/nginx.conf"

# Update all /tmp paths in nginx.conf to use our unique prefix
sed -i "s|/tmp/biopb|${BIOPB_TMP}|g" "$BIOPB_TMP/nginx.conf"

# Update nginx port if not default
if [ "$NGINX_PORT" != "80" ]; then
    sed -i "s/listen 80;/listen ${NGINX_PORT};/" "$BIOPB_TMP/nginx.conf"
fi

# Create nginx temp directories
mkdir -p "$BIOPB_TMP/nginx_client_body" "$BIOPB_TMP/nginx_proxy" "$BIOPB_TMP/nginx_fastcgi" "$BIOPB_TMP/nginx_uwsgi" "$BIOPB_TMP/nginx_scgi"

# Start nginx using temp config
nginx -c "$BIOPB_TMP/nginx.conf"

# Start tensor server (foreground process)
# First argument is the subcommand (launch/serve), rest are passed through
COMMAND="${1:-launch}"
shift 2>/dev/null || true

exec biopb-tensor "$COMMAND" \
    --config "$BIOPB_TMP/runtime-config.toml" \
    --web-host 127.0.0.1 \
    --web-port ${WEB_PORT:-8816} \
    --web-url http://localhost \
    "$@"