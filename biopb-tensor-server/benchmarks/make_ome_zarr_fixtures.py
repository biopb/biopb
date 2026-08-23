"""Build OME-Zarr fixtures for bench_plane_latency (#640).

Four stores crossing the two axes ND2 and TIFF cannot test:

- **native pyramid** -- stored levels, so a scaled read becomes a small direct
  read of a level (the ``precompute`` path) instead of reducing level 0. Nothing
  in the local catalog has one.
- **compression** -- ND2 and TIFF here are both uncompressed, so "cold latency is
  I/O-bound" may be a property of that rather than a general truth. A compressed
  store reads fewer bytes and pays decode, which can move the bound to CPU.

**The pixel content is chosen, not incidental**, because the compression axis is
meaningless otherwise. Two obvious choices are both wrong:

- ``fixtures.py``'s OME-Zarr builder writes a constant per level
  (``arr[:] = level``), which blosc squeezes to nothing -- a 48 MiB level lands as
  ~0.2 MiB. Benchmarking that measures a store that is not there.
- Uniform ``rng.integers`` over the full dtype range is *incompressible*: measured
  at 1.00x, so the compressed and raw stores come out byte-identical in size and
  the axis collapses the other way.

Real 16-bit microscopy is neither: it usually occupies ~12 bits and is spatially
correlated, which is what makes it compress at all. So the field here is a coarse
random field upsampled to full resolution with fine noise added on top -- giving a
ratio in the range real data shows. The achieved ratio is printed, because it is
the number that decides whether the compressed arm is measuring anything.

Usage:
    python benchmarks/make_ome_zarr_fixtures.py OUTDIR [--size 8192] [--levels 6]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

# Structure scale: how many full-resolution pixels share one coarse value before
# noise. Larger means smoother and more compressible.
_FEATURE_PX = 8
# Dynamic range of the signal. Real 16-bit cameras are typically 12-bit.
_SIGNAL_MAX = 4096
# Fine detail on top of the structure, as a fraction of _SIGNAL_MAX.
_NOISE_FRACTION = 0.05


def _plausible_field(rng, rows: int, cols: int, dtype: str) -> np.ndarray:
    """A block that compresses like microscopy rather than like noise or a constant."""
    coarse = rng.integers(
        0,
        _SIGNAL_MAX,
        size=(max(1, rows // _FEATURE_PX), max(1, cols // _FEATURE_PX)),
        dtype=np.int32,
    )
    field = np.repeat(np.repeat(coarse, _FEATURE_PX, axis=0), _FEATURE_PX, axis=1)
    field = field[:rows, :cols]
    if field.shape != (rows, cols):  # sizes not divisible by _FEATURE_PX
        field = np.pad(
            field,
            ((0, rows - field.shape[0]), (0, cols - field.shape[1])),
            mode="edge",
        )
    noise = rng.integers(
        0, max(1, int(_SIGNAL_MAX * _NOISE_FRACTION)), size=(rows, cols), dtype=np.int32
    )
    return np.clip(field + noise, 0, np.iinfo(dtype).max).astype(dtype)


def _write_store(
    path: Path,
    base_size: int,
    n_levels: int,
    chunk: int,
    compressed: bool,
    dtype: str = "uint16",
) -> Path:
    import numcodecs
    import zarr

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    root = zarr.open_group(str(path), mode="w")
    compressor = numcodecs.Blosc(cname="zstd", clevel=3) if compressed else None

    rng = np.random.default_rng(0)
    datasets = []
    for level in range(n_levels):
        factor = 2**level
        size = base_size // factor
        if size < 1:
            break
        # Chunks never exceed the level: a coarse level is smaller than the base
        # chunk, and stopping the ladder there would leave the deepest scales --
        # exactly the ones a browser opens at -- with no stored level to serve.
        level_chunk = min(chunk, size)
        arr = root.create_dataset(
            str(level),
            shape=(size, size),
            chunks=(level_chunk, level_chunk),
            dtype=dtype,
            compressor=compressor,
        )
        # Row-block at a time: an 8192^2 uint16 level is 128 MiB, and holding the
        # whole thing plus its compressed buffers is avoidable.
        for start in range(0, size, level_chunk):
            stop = min(start + level_chunk, size)
            arr[start:stop, :] = _plausible_field(rng, stop - start, size, dtype)
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [factor, factor]}
                ],
            }
        )

    (path / ".zattrs").write_text(
        json.dumps(
            {
                "multiscales": [
                    {
                        "version": "0.4",
                        "name": path.name,
                        "axes": [
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ],
                        "datasets": datasets,
                    }
                ]
            },
            indent=2,
        )
    )
    return path


def _du(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw = args.size * args.size * 2 / 2**20
    print(f"level 0 = {args.size}^2 uint16 = {raw:.0f} MiB raw, chunk {args.chunk}")
    for pyramid in (True, False):
        for compressed in (True, False):
            name = (
                f"{'pyramid' if pyramid else 'flat'}-"
                f"{'blosc' if compressed else 'raw'}.ome.zarr"
            )
            store = _write_store(
                args.outdir / name,
                args.size,
                args.levels if pyramid else 1,
                args.chunk,
                compressed,
            )
            on_disk = _du(store) / 2**20
            stored_raw = raw * (4 / 3 if pyramid else 1)  # pyramid adds ~1/3
            print(
                f"  {name:28} {on_disk:8.1f} MiB on disk  "
                f"({stored_raw / on_disk:.2f}x compression)"
            )


if __name__ == "__main__":
    main()
