"""Benchmark suite for biopb-tensor-server.

Measures performance under realistic HPC/Singularity deployment conditions
with NFS storage. Validates caching algorithms and throughput for
production microscopy workflows.

Usage:
    pytest benchmarks/ --benchmark-only

For HPC deployment:
    sbatch benchmarks/run_benchmarks.sh
"""

__all__ = []