"""Benchmark NikonAdapter's direct ND2 frame path against BioIO/Dask (#640).

Run from the repository root:

    uv run --no-sync python \
      biopb-tensor-server/benchmarks/bench_nd2_direct_read.py FILE [ROUNDS]

The timed region reads one planner-sized channel tile. ``direct`` includes
opening and closing the ND2 file, matching the adapter's normal handle policy.
BioIO construction/metadata access is reported separately and remains shared by
both paths.
"""

import gc
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from bioio import BioImage
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters.bioio import NikonAdapter


def _time_ms(fn):
    gc.collect()
    started = time.perf_counter()
    result = fn()
    return (time.perf_counter() - started) * 1e3, result


def _summary(samples):
    return (
        f"{statistics.median(samples):.2f} ms [{min(samples):.2f}-{max(samples):.2f}]"
    )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: bench_nd2_direct_read.py FILE [ROUNDS]")
    path = Path(sys.argv[1])
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    metadata_ms, image = _time_ms(lambda: BioImage(path))
    metadata_access_ms, metadata = _time_ms(
        lambda: (image.scenes, image.dims.order, image.shape, image.dtype)
    )
    image.set_scene(0)
    adapter = NikonAdapter(
        image,
        scene_index=0,
        source_id="benchmark",
        source_url=str(path),
    )
    desc = adapter.get_tensor_descriptor()
    transfer = adapter.get_transfer_chunk_size()
    starts = [0] * len(desc.shape)
    stops = [
        min(int(size), int(chunk))
        for size, chunk in zip(desc.shape, transfer, strict=True)
    ]
    bounds = ChunkBounds(start=starts, stop=stops)

    timings = {"bioio/dask": [], "direct": []}
    expected = None
    for _ in range(rounds):
        for name, read in (
            ("bioio/dask", lambda: adapter._get_data_via_bioio(bounds)),
            ("direct", lambda: adapter.get_data(bounds)),
        ):
            elapsed, data = _time_ms(read)
            timings[name].append(elapsed)
            if expected is None:
                expected = data
            elif not np.array_equal(data, expected):
                raise RuntimeError(f"{name} returned different pixels")

    print(f"file: {path}")
    print(f"shape: {tuple(desc.shape)}, native: {tuple(desc.chunk_shape)}")
    print(
        f"transfer/read bounds: {tuple(stops)} ({np.prod(stops) * image.dtype.itemsize / 2**20:.2f} MiB)"
    )
    print(f"BioImage construction: {metadata_ms:.2f} ms")
    print(f"metadata access: {metadata_access_ms:.2f} ms ({metadata[1]})")
    for name, samples in timings.items():
        print(f"{name:>12}: {_summary(samples)}")
    print(
        f"speedup: {statistics.median(timings['bioio/dask']) / statistics.median(timings['direct']):.2f}x"
    )


if __name__ == "__main__":
    main()
