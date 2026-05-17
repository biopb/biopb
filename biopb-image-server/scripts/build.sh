#!/bin/bash
# Build script for biopb-image-server Docker images
#
# Versioning: SDK version from biopb wheel (setuptools_scm),
# image revision (-N) from git SHA
#
# Usage:
#   ./build.sh [--no-cache]
#   ./build.sh --no-cache

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
SERVICE_DIR="$SCRIPT_DIR/.."

# Registry settings (consistent with biopb-server)
REGISTRY="${REGISTRY:-docker.io}"
IMAGE_PREFIX="${IMAGE_PREFIX:-jiyuuchc}"
IMAGE_NAME="biopb-image-base"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

get_sha() {
    git rev-parse --short HEAD 2>/dev/null || echo "local"
}

# Parse arguments
NO_CACHE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-cache]"
            exit 1
            ;;
    esac
done

sha=$(get_sha)
image_base="$REGISTRY/$IMAGE_PREFIX/$IMAGE_NAME"

log_info "Building biopb-image-base"
echo "  SHA: $sha"
echo "  Image: $image_base"
if [ -n "$NO_CACHE" ]; then
    echo "  Cache: disabled"
fi

# Build biopb wheel first (SDK version from setuptools_scm)
log_info "Building biopb wheel..."
cd "$ROOT_DIR"
rm -f wheels/biopb-*.whl
pip wheel . --no-deps -w wheels/ --quiet

# Build with tags: sha, latest (no VERSION file needed)
tags="--tag $image_base:$sha --tag $image_base:latest"

docker buildx build \
    --platform linux/amd64 \
    $tags \
    $NO_CACHE \
    --load \
    -f "$SERVICE_DIR/Dockerfile" \
    "$ROOT_DIR"

log_info "Built $image_base:$sha"
echo ""
echo "Images:"
docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"