"""Benchmark: how long a user waits for one scaled X/Y plane (#640).

The metric is **first-read latency** -- request a full X/Y plane at some scale
and wait until every pixel of it has arrived. That is the blank-screen event:
the tensor browser opens a scene at a coarse level (#818), and this is what it
costs. Every other number in the scaled-read work (per-chunk time, kernel
throughput, peak RSS) is a component of it.

It is deliberately narrow:

- **One plane** is what is *requested*: full X and Y, index 0 on every other
  axis. It is not necessarily what is delivered -- a read plan snaps outward to
  the scaled virtual grid, which grows along whatever axis is free until it fills
  its byte budget. Asking for one Z plane of a 12-plane TIFF at scale 32 returns
  a chunk covering ten of them; asking for one channel of a 3-channel ND2 returns
  all three. That inflation IS part of what a user waits for, so it stays in the
  timing -- but the ``covers`` column reports it, because a latency that silently
  includes 10x the requested data is not interpretable.
- **Serial resolve**, in plan order. A real client may fetch endpoints
  concurrently, so this is the pessimistic bound; the chunk count is reported so
  a parallel estimate can be derived.
- **A fresh file cache per cell**, because a first read has nothing warmed. The
  cache write is part of what the user waits for, so it stays in the timing.

Cold means the source is evicted from page cache with ``posix_fadvise`` after
the adapter -- and therefore its mmap -- has been dropped and collected. Both
halves are required and neither is obvious:

- ``utils.clear_os_page_cache`` needs root and *silently does nothing* without
  it;
- ``posix_fadvise`` cannot reclaim pages that a live mapping still maps, and
  every reader here holds one, so evicting without dropping the adapter first
  leaves the file entirely resident.

Each failure reports a warm read as cold, and it does not look wrong: it just
prints a fast number. So a cold cell asserts afterwards that the timed read did
real block I/O, and prints ``NOT-COLD`` instead of a latency if it did not.

Run:
    python benchmarks/bench_plane_latency.py FILE [--scale 32,8,4]
                                                  [--method nearest,area]
                                                  [--state cold,warm] [--rounds 3]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.adapter_base import unpack_chunk_array
from biopb_tensor_server.core.config import CacheConfig


def _open_adapter(path: Path):
    """A tensor adapter for ``path``, scene/tensor 0.

    ``source_url`` is passed explicitly and is not optional: without it the ND2
    direct read path cannot reopen the file and every read falls back silently
    to a dask compute -- correct pixels, ~15x the time, no warning.
    """
    suffix = path.suffix.lower()
    if suffix == ".nd2":
        from bioio import BioImage
        from biopb_tensor_server.adapters.bioio import NikonAdapter

        source = NikonAdapter(BioImage(str(path)), None, "bench", source_url=str(path))
    elif suffix in {".tif", ".tiff"}:
        # OME-TIFF carries its own OME-XML and takes a different adapter from a
        # plain TIFF, so dispatch on the compound suffix, not just the last one.
        if path.name.lower().endswith((".ome.tif", ".ome.tiff")):
            from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

            source = OmeTiffAdapter(str(path), "bench")
        else:
            from biopb_tensor_server.adapters.tifffile_adapter import TiffAdapter

            source = TiffAdapter(str(path), "bench")
    elif path.name.lower().endswith(".zarr") or suffix == ".zarr":
        from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
        from biopb_tensor_server.core.config import SourceConfig

        source = OmeZarrAdapter.create_from_config(
            SourceConfig(url=str(path), source_id="bench")
        )
    else:
        sys.exit(f"no adapter wired for {suffix} in this benchmark")

    descriptors = source.list_tensor_descriptors()
    descriptor = descriptors[0]
    adapter = source.get_tensor_adapter(descriptor.array_id)
    return adapter, descriptor


def _plane_request(descriptor, scale: int, method: str) -> TensorDescriptor:
    """Full X/Y at ``scale``, index 0 on every other axis."""
    labels = [str(label).upper() for label in descriptor.dim_labels]
    shape = [int(dim) for dim in descriptor.shape]
    xy = {labels.index("Y"), labels.index("X")}

    request = TensorDescriptor(array_id=descriptor.array_id)
    request.slice_hint.start[:] = [0] * len(shape)
    request.slice_hint.stop[:] = [
        size if axis in xy else 1 for axis, size in enumerate(shape)
    ]
    request.scale_hint[:] = [scale if axis in xy else 1 for axis in range(len(shape))]
    request.reduction_method = method
    return request


def _read_bytes() -> int:
    with open("/proc/self/io") as handle:
        for line in handle:
            if line.startswith("read_bytes"):
                return int(line.split()[1])
    return 0


def _evict(path: Path) -> None:
    """Drop ``path`` from page cache. Caller must already have dropped the reader."""
    import gc

    gc.collect()
    for member in _source_files(path):
        fd = os.open(str(member), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _source_files(path: Path) -> list[Path]:
    """Every file backing this source.

    A source is not always one file: a zarr store is a directory of chunk files,
    and warming or evicting only the top-level path there is a no-op that leaves
    the whole store resident.
    """
    if path.is_dir():
        return sorted(f for f in path.rglob("*") if f.is_file())
    return [path]


def _warm(path: Path) -> None:
    """Pull the whole source into page cache.

    Over the files rather than through the adapter: warming only the chunks the
    plan happens to touch leaves the rest cold and reports a half-cold read as
    warm.
    """
    for member in _source_files(path):
        with open(member, "rb") as handle:
            while handle.read(16 << 20):
                pass


def _one_cell(path, open_adapter, descriptor, scale, method, cold, rounds):
    latencies, disk = [], []
    for _ in range(rounds):
        # Rebuilt per round: a cold read needs the previous round's mmap gone
        # before the eviction, and the reader is what holds it.
        adapter, descriptor = open_adapter()
        request = _plane_request(descriptor, scale, method)
        endpoints = adapter.get_read_plan(request).chunk_endpoints
        cache_dir = tempfile.mkdtemp(prefix="planebench-")
        cache = CacheManager(
            CacheConfig(
                backend="file",
                file_cache_dir=Path(cache_dir),
                file_max_total_bytes=16 * 1024**3,
            )
        )
        try:
            if cold:
                del adapter
                _evict(path)
                adapter, _ = open_adapter()
            else:
                _warm(path)

            before = _read_bytes()
            start = time.perf_counter()
            pixels = 0
            for endpoint in endpoints:
                batch = adapter.resolve_chunk_data(endpoint.chunk_id, cache)
                pixels += unpack_chunk_array(batch).nbytes
            latencies.append((time.perf_counter() - start) * 1e3)
            # What the plan actually covered, against what was asked for.
            covered = [0] * len(descriptor.shape)
            for endpoint in endpoints:
                for axis, stop in enumerate(endpoint.bounds.stop):
                    covered[axis] = max(covered[axis], int(stop))
            covers = covered
            disk.append(_read_bytes() - before)
        finally:
            cache.close()
            shutil.rmtree(cache_dir, ignore_errors=True)

    latency = sorted(latencies)[len(latencies) // 2]
    read = sorted(disk)[len(disk) // 2]
    # A cold read of a source larger than its output must have touched the disk.
    # Zero means the eviction did not take, and the latency below is a warm one.
    verified = (not cold) or read > 0
    return {
        "verified": verified,
        "covers": covers,
        "latency_ms": latency,
        "chunks": len(endpoints),
        "out_mib": pixels / 2**20,
        "disk_mib": read / 2**20,
        "rate": (read / 2**20) / (latency / 1e3) if latency else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--scale", default="32,8,4")
    parser.add_argument("--method", default="nearest,area")
    parser.add_argument("--state", default="cold,warm")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    def open_adapter():
        return _open_adapter(args.file)

    adapter, descriptor = open_adapter()
    shape = [int(dim) for dim in descriptor.shape]
    print(
        f"{args.file.name}  {shape} {descriptor.dtype} "
        f"labels={list(descriptor.dim_labels)}  "
        f"grid={list(adapter.get_transfer_chunk_size())}"
    )
    print(
        f"{'state':6} {'method':8} {'scale':>5} {'latency':>10} {'chunks':>7} "
        f"{'plane out':>10} {'disk read':>10} {'MiB/s':>8} {'covers':>12}",
        flush=True,
    )

    for state in args.state.split(","):
        for method in args.method.split(","):
            for scale in (int(s) for s in args.scale.split(",")):
                try:
                    row = _one_cell(
                        args.file,
                        open_adapter,
                        descriptor,
                        scale,
                        method,
                        state == "cold",
                        args.rounds,
                    )
                except Exception as exc:
                    # One unsupported cell must not take the sweep with it:
                    # `precompute` legitimately has no level at some scales, and
                    # that is a result, not a crash.
                    print(
                        f"{state:6} {method:8} {scale:>5} "
                        f"{type(exc).__name__}: {str(exc)[:60]}",
                        flush=True,
                    )
                    continue
                latency = (
                    f"{row['latency_ms']:>8.0f}ms"
                    if row["verified"]
                    else "NOT-COLD".rjust(10)
                )
                print(
                    f"{state:6} {method:8} {scale:>5} {latency} "
                    f"{row['chunks']:>7} {row['out_mib']:>8.1f}MiB "
                    f"{row['disk_mib']:>8.0f}MiB {row['rate']:>8.0f}"
                    f" {'x'.join(str(c) for c in row['covers']):>12}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
