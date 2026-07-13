#!/bin/bash
# Build script for biopb-image-runtime Docker images
#
# By default builds wheels from the latest released git tags:
#   biopb         → latest v* tag (SDK line)
#   biopb-tensor-server → latest release-v* tag (product line)
#
# Usage:
#   ./build.sh [options]
#
# Options:
#   --no-cache   Disable Docker layer cache
#   --dev        Build wheels from current working tree instead of latest tags

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
SERVICE_DIR="$SCRIPT_DIR/.."

# Registry settings
REGISTRY="${REGISTRY:-docker.io}"
IMAGE_PREFIX="${IMAGE_PREFIX:-jiyuuchc}"
IMAGE_NAME="biopb-image-base"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

get_sha() {
    git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo "local"
}

# Parse arguments
NO_CACHE=""
DEV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --dev)
            DEV=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-cache] [--dev]"
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

cd "$ROOT_DIR"
rm -f wheels/biopb-*.whl wheels/biopb_tensor_server-*.whl

if [ "$DEV" = true ]; then
    log_info "Building wheels from current working tree..."
    pip wheel . --no-deps -w wheels/ --quiet
    pip wheel biopb-tensor-server/ --no-deps -w wheels/ --quiet
else
    BIOPB_TAG=$(git tag --sort=-version:refname | grep -E '^v[0-9]' | head -1)
    TENSOR_TAG=$(git tag --sort=-version:refname | grep -E '^release-v[0-9]' | head -1)

    if [ -z "$BIOPB_TAG" ]; then
        echo "Error: no biopb SDK release tag found (expected v*)"
        exit 1
    fi
    if [ -z "$TENSOR_TAG" ]; then
        echo "Error: no product release tag found (expected release-v*)"
        exit 1
    fi

    log_info "Building wheels from latest release tags..."
    echo "  biopb: $BIOPB_TAG"
    echo "  biopb-tensor-server: $TENSOR_TAG"

    pip wheel "git+file://${ROOT_DIR}@${BIOPB_TAG}" \
        --no-deps -w wheels/ --quiet
    pip wheel "git+file://${ROOT_DIR}@${TENSOR_TAG}#subdirectory=biopb-tensor-server" \
        --no-deps -w wheels/ --quiet
fi

# Build Docker image with tags: sha, latest
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
