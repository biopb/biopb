"""What the cyclic garbage of a dropped OME-TIFF adapter is, and what frees it (#1284).

Run from the repository root:

    uv run --no-sync python \
      biopb-tensor-server/benchmarks/bench_tifffile_cycles.py FILE

Counts the objects only the cyclic GC can reclaim (``gc.DEBUG_SAVEALL``) after:
a plain ``TiffFile`` that has evaluated ``series``; an adapter parsed and read
once; and a ``TiffFile`` whose page references were cleared on close. The cycle
is tifffile's own -- ``TiffFile`` -> ``TiffPages`` -> ``TiffFrame`` -> ``parent``
-- and the series keeps a second list of the same frames, so both must go.
Those are private attributes of tifffile, which the repo already pins.
"""

import gc
import sys
import time

import numpy as np
import tifffile
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters import OmeTiffAdapter
from biopb_tensor_server.core.config import SourceConfig


def unreachable(fn):
    """Objects only the cycle collector frees after *fn* runs."""
    gc.collect()
    gc.disable()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        fn()
        n = gc.collect()
    finally:
        gc.garbage.clear()
        gc.set_debug(0)
        gc.enable()
    return n


def read_once(adapter):
    scene = adapter.get_tensor_adapter(adapter.list_tensor_descriptors()[0].array_id)
    shape = list(scene.get_tensor_descriptor().shape)
    stop = [min(s, 64) for s in shape]
    np.asarray(scene.get_data(ChunkBounds(start=[0] * len(shape), stop=stop)))


def open_and_close(path, break_with=None):
    tf = tifffile.TiffFile(path)
    series = tf.series[0]
    tf.close()
    if break_with is not None:
        break_with(tf, series)


def clear_pages(tf, series):
    tf.pages._pages.clear()


def clear_series(tf, series):
    tf._series = None
    series._pages = None


def clear_both(tf, series):
    clear_pages(tf, series)
    clear_series(tf, series)


def main(path):
    config = SourceConfig(url=path, source_id="s0")
    with tifffile.TiffFile(path) as tf:
        pages = len(tf.pages)
    print(f"{pages} pages, tifffile {tifffile.__version__}")

    print(
        f"{'adapter parsed, dropped':44s} {unreachable(lambda: OmeTiffAdapter.create_from_config(config))}"
    )

    def parsed_and_read():
        adapter = OmeTiffAdapter.create_from_config(config)
        read_once(adapter)

    print(
        f"{'adapter parsed and read once, dropped':44s} {unreachable(parsed_and_read)}"
    )
    print(
        f"{'TiffFile + series, closed':44s} {unreachable(lambda: open_and_close(path))}"
    )
    for label, fn in (
        ("  ... pages cleared", clear_pages),
        ("  ... series cleared", clear_series),
        ("  ... pages and series cleared", clear_both),
    ):
        count = unreachable(lambda fn=fn: open_and_close(path, fn))
        print(f"{label:44s} {count}")

    open_and_close(path)
    t = time.perf_counter()
    gc.collect()
    print(
        f"deferred gc.collect() of one dropped file: {(time.perf_counter() - t) * 1e3:.0f} ms"
    )


if __name__ == "__main__":
    main(sys.argv[1])
