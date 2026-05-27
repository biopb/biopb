#!/bin/bash
# Build and launch a test environment for install.sh.
# Usage: ./run.sh [scenario]
#
# Scenarios:
#   clean            Fresh Ubuntu, no uv, no Python extras  (default)
#   uv-preinstalled  uv already on PATH before installer runs
#   old-python       System Python 3.7 present (too old, should fall back)
#   rerun            Pre-staged env simulating a prior install (idempotency)

set -euo pipefail

SCENARIO="${1:-clean}"
DOCKERFILE="Dockerfile.$SCENARIO"
IMAGE="biopb-install-test:$SCENARIO"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$SCRIPT_DIR/$DOCKERFILE" ]; then
    echo "ERROR: Unknown scenario '$SCENARIO'"
    echo "Available: clean  uv-preinstalled  old-python  rerun"
    exit 1
fi

echo "Building $IMAGE from $DOCKERFILE..."
docker build \
    --file "$SCRIPT_DIR/$DOCKERFILE" \
    --tag "$IMAGE" \
    "$SCRIPT_DIR/.."

echo ""
echo "Launching — run the installer with:"
echo "  bash /install.sh"
echo ""
docker run --rm -it "$IMAGE"
