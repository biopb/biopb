#!/bin/bash
# Build and launch a test environment for install.sh.
# Usage: ./run.sh [--assert] [scenario]
#
# Scenarios:
#   clean            Fresh Ubuntu, no uv, no Python extras  (default)
#   uv-preinstalled  uv already on PATH before installer runs
#   old-python       System Python 3.7 present (too old, should fall back)
#   rerun            Pre-staged env simulating a prior install (idempotency)
#   bioformats       Bio-Formats/ZVI end-to-end (install with
#                    BIOPB_INSTALL_BIOFORMATS=1, then run /verify_bioformats.sh;
#                    no system Java present)
#   all              (--assert only) every scenario in turn
#
# Without --assert this is a workbench: it builds the image and drops you at a
# shell with instructions, asserting nothing. That is still the mode to use when
# poking at a failure by hand.
#
# With --assert it runs install/test/assert.sh in the container instead --
# non-interactive, exit 0 only if every post-condition holds. This is the only
# layer that executes the installer end to end (see assert.sh for the checks and
# the reasoning); it takes minutes and needs network, so it is nightly /
# pre-release, not per-PR. See biopb/biopb#653.
#
# Mount a ZVI sample for the bioformats scenario:
#   BIOPB_TEST_DATA=/dir/with/zvi ./run.sh bioformats

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ALL_SCENARIOS=(clean uv-preinstalled old-python rerun bioformats)

ASSERT=0
SCENARIO=""
for arg in "$@"; do
    case "$arg" in
        --assert)  ASSERT=1 ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        -*)        echo "ERROR: Unknown option '$arg'" >&2; exit 2 ;;
        *)         SCENARIO="$arg" ;;
    esac
done
SCENARIO="${SCENARIO:-clean}"

if [ "$SCENARIO" = "all" ] && [ "$ASSERT" != "1" ]; then
    echo "ERROR: 'all' only makes sense with --assert -- there is one shell to drop into" >&2
    exit 2
fi

# Build one scenario image. Split out from running it so `all` can build and run
# each in turn without duplicating either half. Build chatter goes to stderr so
# it never lands in the middle of the assertion output.
build() {
    local scenario="$1" dockerfile="$SCRIPT_DIR/Dockerfile.$1"
    if [ ! -f "$dockerfile" ]; then
        echo "ERROR: Unknown scenario '$scenario'" >&2
        echo "Available: ${ALL_SCENARIOS[*]}  all" >&2
        return 1
    fi
    echo "Building biopb-install-test:$scenario from Dockerfile.$scenario..." >&2
    docker build --file "$dockerfile" --tag "biopb-install-test:$scenario" "$SCRIPT_DIR/.." >&2
}

# Scenario-independent: bioformats wants a ZVI sample, and any scenario can be
# handed a folder to index.
data_mount=()
if [ -n "${BIOPB_TEST_DATA:-}" ]; then
    echo "Mounting $BIOPB_TEST_DATA -> /data (read-only)"
    data_mount=(-v "$BIOPB_TEST_DATA:/data:ro")
fi

if [ "$ASSERT" = "1" ]; then
    scenarios=("$SCENARIO")
    [ "$SCENARIO" = "all" ] && scenarios=("${ALL_SCENARIOS[@]}")

    failed=()
    for s in "${scenarios[@]}"; do
        printf '\n========== %s ==========\n' "$s"
        build "$s"
        # assert.sh is MOUNTED rather than baked into the images: editing a check
        # would otherwise mean rebuilding five Dockerfiles to try it. `|| true`
        # via the if/else -- one failing scenario must not stop the rest, since
        # the point of `all` is to learn everything wrong in a single run.
        if docker run --rm "${data_mount[@]}" \
                -v "$SCRIPT_DIR/assert.sh:/assert.sh:ro" \
                "biopb-install-test:$s" bash /assert.sh "$s"; then
            :
        else
            failed+=("$s")
        fi
    done

    printf '\n===============================\n'
    if [ "${#failed[@]}" -eq 0 ]; then
        printf 'PASS  %s scenario(s): %s\n' "${#scenarios[@]}" "${scenarios[*]}"
        exit 0
    fi
    printf 'FAIL  %s\n' "${failed[*]}"
    exit 1
fi

build "$SCENARIO"

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
echo "Or run the same checks unattended, asserting instead of showing:"
echo "  ./run.sh --assert $SCENARIO"
echo ""

docker run --rm -it "${data_mount[@]}" "biopb-install-test:$SCENARIO"
