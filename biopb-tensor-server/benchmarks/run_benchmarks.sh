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
# This script runs the benchmark suite on HPC infrastructure
# using Singularity containers with NFS data mounts.
#
# Prerequisites:
#   1. Build Singularity image: singularity build biopb-bench.sif benchmarks/biopb-bench.def
#   2. Set NFS_TEST_DATA_DIR to your microscopy data path
#   3. Ensure scratch filesystem is available for cache/results
#
# Output:
#   /scratch/$USER/results/*.json     - Raw benchmark data
#   /scratch/$USER/results/*.svg      - Histogram plots

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="biopb-bench.sif"
RESULTS_DIR="/scratch/${USER}/biopb-results"

# Test data paths (customize for your environment)
NFS_DATA_DIR="${NFS_TEST_DATA_DIR:-/data/microscopy}"
S3_DATA_URL="${S3_TEST_DATA_URL:-s3://idr-public/ngff/6001240.zarr}"

# Create results directory
mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "BioPB Tensor Server Benchmark Suite"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Results dir: ${RESULTS_DIR}"
echo "=========================================="

# Check if Singularity image exists
if [[ ! -f "${IMAGE_NAME}" ]]; then
    echo "Building Singularity image..."
    cd "${SCRIPT_DIR}"
    singularity build "${IMAGE_NAME}" biopb-bench.def
fi

# Run benchmarks with NFS mount
echo "Running benchmarks..."

singularity exec \
    --bind "${NFS_DATA_DIR}:/data:ro" \
    --bind "${RESULTS_DIR}:/scratch/results" \
    --env S3_TEST_DATA_URL="${S3_DATA_URL}" \
    --env NFS_TEST_DATA_DIR="/data" \
    "${IMAGE_NAME}" \
    python -m pytest benchmarks/ \
        --benchmark-only \
        --benchmark-autosave \
        --benchmark-save-data \
        --benchmark-histogram \
        --benchmark-storage="/scratch/results" \
        -v \
        --tb=short

# Copy results to permanent location
echo "Copying results to ${RESULTS_DIR}..."

# Summarize results
echo "=========================================="
echo "Benchmark Summary"
echo "=========================================="

if [[ -f "${RESULTS_DIR}/.benchmarks" ]]; then
    python -c "
import json
import glob

results_files = glob.glob('${RESULTS_DIR}/*.json')
for f in results_files:
    with open(f) as fp:
        data = json.load(fp)
    print(f'File: {f}')
    for bench in data.get('benchmarks', []):
        name = bench['name']
        mean = bench['stats']['mean'] * 1000
        stddev = bench['stats']['stddev'] * 1000
        print(f'  {name}: {mean:.2f}ms (+/- {stddev:.2f}ms)')
"
fi

echo "=========================================="
echo "Results saved to: ${RESULTS_DIR}"
echo "=========================================="

# Clean up old results (keep last 10 runs)
cd "${RESULTS_DIR}"
ls -t *.json | tail -n +11 | xargs -r rm
ls -t *.svg | tail -n +11 | xargs -r rm

echo "Done!"