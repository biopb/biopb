#!/bin/bash
# Build and launch a test environment for install.sh.
# Usage: ./run.sh [scenario]
#
# Scenarios:
#   clean            Fresh Ubuntu, no uv, no Python extras  (default)
#   uv-preinstalled  uv already on PATH before installer runs
#   old-python       System Python 3.7 present (too old, should fall back)
#   rerun            Pre-staged env simulating a prior install (idempotency)
#   bioformats       Bio-Formats/ZVI end-to-end (install with
#                    BIOPB_INSTALL_BIOFORMATS=1, then run /verify_bioformats.sh;
#                    no system Java present)
#
# Mount a ZVI sample for the bioformats scenario:
#   BIOPB_TEST_DATA=/dir/with/zvi ./run.sh bioformats

set -euo pipefail

SCENARIO="${1:-clean}"
DOCKERFILE="Dockerfile.$SCENARIO"
IMAGE="biopb-install-test:$SCENARIO"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$SCRIPT_DIR/$DOCKERFILE" ]; then
    echo "ERROR: Unknown scenario '$SCENARIO'"
    echo "Available: clean  uv-preinstalled  old-python  rerun  bioformats"
    exit 1
fi

echo "Building $IMAGE from $DOCKERFILE..."
docker build \
    --file "$SCRIPT_DIR/$DOCKERFILE" \
    --tag "$IMAGE" \
    "$SCRIPT_DIR/.."

echo ""
echo "Launching — run the installer with:"
if [ "$SCENARIO" = "bioformats" ]; then
    # The image bakes a config pointing at /data, so install.sh keeps it and
    # BIOPB_DATA_DIR would be ignored -- don't set it here.
    echo "  BIOPB_INSTALL_BIOFORMATS=1 bash /install.sh"
    echo "then verify Bio-Formats/ZVI support with:"
    echo "  /verify_bioformats.sh"
else
    # A bare fresh install now seeds the sample bundle from the latest release and
    # points the config there (no data-dir prompt). NOTE: seeding pulls
    # biopb-samples.tar.gz from the latest *release* (not this branch build), so
    # the seed path only actually populates once a release shipping that asset
    # exists -- until then a bare run fails soft to an empty folder (that is not a
    # seeding failure). To exercise discovery deterministically, point the install
    # at the TIFFs seeded at /root instead:
    echo "  BIOPB_DATA_DIR=/root bash /install.sh   # discover the seeded /root TIFFs"
    echo "  bash /install.sh                        # or: seed + serve the sample bundle (needs a release with the asset)"
fi
echo ""

DOCKER_RUN_ARGS=(--rm -it)
if [ -n "${BIOPB_TEST_DATA:-}" ]; then
    echo "Mounting $BIOPB_TEST_DATA -> /data (read-only)"
    DOCKER_RUN_ARGS+=(-v "$BIOPB_TEST_DATA:/data:ro")
fi
docker run "${DOCKER_RUN_ARGS[@]}" "$IMAGE"
