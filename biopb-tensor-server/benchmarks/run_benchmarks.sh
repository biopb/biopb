#!/bin/bash
#SBATCH --job-name=biopb-bench
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks-%j.out
#SBATCH --error=benchmarks-%j.err

# BioPB Tensor Server Benchmark Suite
# SLURM batch script for HPC deployment
#
# Prerequisites:
#   1. docker build -t jiyuuchc/biopb-tensor-server ../
#   2. singularity build biopb-bench.sif benchmarks/biopb-bench.def
#
# Usage:
#   sbatch run_benchmarks.sh              # Ephemeral test servers
#   ./run_benchmarks.sh --bench-only      # Existing production server
#
# Environment variables (forwarded to container):
#   TEST_DATA_DIR=/path/to/data           # For placeholder.json sources
#   BIOPB_CACHE_BACKEND=file|memory       # Cache backend mode
#   BIOPB_SYNTHETIC_DATA_DIR=/scratch/bench-data  # Synthetic data location
#
# pytest -k keyword filters (matches test names):
#   Client type:
#     -k "flight"          # Flight server tests only
#     -k "baseline"        # Baseline (direct library) tests only
#     -k "not baseline"    # Skip baseline tests
#
#   Source type:
#     -k "synthetic"       # Only synthetic sources (no network)
#     -k "allen"           # Allen Institute S3 sources
#     -k "zarr"            # All zarr sources
#     -k "tiff"            # All TIFF sources
#     -k "hdf5"            # HDF5 sources
#
#   Test class:
#     -k "TestReadLatency"            # First/warm read latency
#     -k "TestScaledColdRead"         # Cold cache scaled reads
#     -k "TestScaledWarmRead"         # Warm cache scaled reads
#     -k "TestScaledSequentialScan"   # Sequential scan tests
#     -k "TestUnscaledRead"           # Unscaled reads (file backend)
#
#   Combinations:
#     -k "flight and synthetic"       # No network required
#     -k "baseline and allen"         # Baseline on S3 sources
#     -k "TestScaledWarmRead and zarr" # Warm cache on zarr
#
# pytest -m markers (matches @pytest.mark.xxx):
#     -m "s3"              # Only tests marked with @pytest.mark.s3
#     -m "nfs"             # Only tests marked with @pytest.mark.nfs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="biopb-bench.sif"
RESULTS_DIR="/tmp/${USER}/biopb-results"

# Build quoted PYTEST_ARGS preserving argument grouping
PYTEST_ARGS=""
for arg in "$@"; do
    if [[ "$arg" == --bench-only ]]; then
        BENCH_ONLY=1
    else
        # Quote args that contain spaces using single quotes for singularity
        if [[ "$arg" =~ [[:space:]] ]]; then
            PYTEST_ARGS="$PYTEST_ARGS '$arg'"
        else
            PYTEST_ARGS="$PYTEST_ARGS $arg"
        fi
    fi
done

mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "BioPB Tensor Server Benchmark Suite"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Results: ${RESULTS_DIR}"
echo "Args: ${PYTEST_ARGS:-<default>}"
echo "=========================================="

cd "${SCRIPT_DIR}"

if [[ ! -f "${IMAGE_NAME}" ]]; then
    echo "Building SIF from Docker image..."
    singularity build "${IMAGE_NAME}" biopb-bench.def
fi

singularity run \
    --bind "${RESULTS_DIR}:/scratch/results" \
    --env TEST_DATA_DIR="${TEST_DATA_DIR:-}" \
    --env BIOPB_CACHE_BACKEND="${BIOPB_CACHE_BACKEND:-file}" \
    --env BIOPB_SYNTHETIC_DATA_DIR="${BIOPB_SYNTHETIC_DATA_DIR:-}" \
    --env PYTEST_ARGS="${PYTEST_ARGS}" \
    --env BENCH_ONLY="${BENCH_ONLY:-0}" \
    "${IMAGE_NAME}"

# Print latest results (pytest-benchmark stores in nested subdir)
LATEST_JSON=$(ls -t "${RESULTS_DIR}"/*.json 2>/dev/null | head -1)
if [[ -z "${LATEST_JSON}" ]]; then
    # pytest-benchmark may store in nested platform subdir
    LATEST_JSON=$(find "${RESULTS_DIR}" -name "*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
fi
if [[ -n "${LATEST_JSON}" ]]; then
    echo ""
    echo "=== Latest Results ==="
    python -c "
import json
with open('${LATEST_JSON}') as fp:
    data = json.load(fp)
for bench in data.get('benchmarks', []):
    name = bench['name']
    mean = bench['stats']['mean'] * 1000
    stddev = bench['stats']['stddev'] * 1000
    print(f'{name}: {mean:.2f}ms (+/- {stddev:.2f}ms)')
"
fi

echo "=========================================="
echo "Done"
