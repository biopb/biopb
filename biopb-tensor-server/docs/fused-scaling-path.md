# The scaled read path

How the tensor server serves a *scaled* chunk (a pyramid level computed on
demand) without holding its full-resolution extent. Code: `core/adapter_base.py`
(`get_scaled_data`, `get_decimated_data`, `read_block_shape`),
`core/stream_reduce.py`, `core/cache_source.py`, `core/downsample.py`, and the
adapter overrides named below.

## 1. The seam

Every scaled read reaches the pixels through `TensorAdapter.resolve_chunk_data`,
which decodes the scale and the reduction method from the chunk_id and calls

```python
get_scaled_data(bounds, scale_hint, reduction_method, cache_manager)
```

An unscaled chunk calls `get_data(bounds)` instead. `NormalizingAdapter` needs
nothing: it forwards the chunk_id verbatim, so the wrapped adapter reduces in
native axis order and the wrapper transposes the small result.
`RemoteTensorAdapter` forwards the chunk_id upstream and never reduces locally.

### The contract

An override of `get_scaled_data` must return exactly what the default returns:

1. **Shape**: `ceil((stop - start) / scale)` per axis, edge padding included.
2. **Dtype**: the input's own (`get_output_dtype`).
3. **Values**: bit-identical to
   `downsample_block(self.get_data(bounds), scale_hint, method)`. A backend
   whose reduction differs from a method must not be reached for that method.
4. **Ownership**: an owned array. No view onto a reader-owned mapping or a
   cache entry may leave the call.
5. **Fallback**: anything the override cannot express bit-identically calls
   `super().get_scaled_data(...)`, forwarding `cache_manager`.

`get_data` carries the same ownership rule: the returned memory must not be
tied to a handle the reaper can close.

### The extent

`scaled_virtual_chunk_size` sizes a scaled chunk's source extent as
`transfer * scale`, clamped to the tensor. The delivered chunk is therefore one
transfer chunk, chunks tile without splitting a reduction block, and a ragged
block can only occur where the extent ends at the tensor's end. Nothing else
shapes the extent. There is no memory ceiling, because the extent is streamed,
never held.

## 2. The default `get_scaled_data`

In order:

1. **`nearest` asks for a decimated read** (§4). If the adapter returns one, it
   is the output and nothing is streamed.
2. **Size the streaming unit**: `streaming_unit` takes the transfer grid,
   floors it at `read_block_shape` (§3), and rounds it up to whole reduction
   blocks.
3. **Offer the extent to the cache** (§6). `cache_sourced_units` either supplies
   a `fetch` that assembles units from cached full-resolution chunks, or
   declines and leaves `fetch` reading the source.
4. **One unit covers the extent**: read it and call `downsample_block` once.
   A unit borrowed from the cache is materialised before it is returned.
5. **Otherwise `stream_reduce`**: walk the units in row-major order, reduce each
   with `downsample_block`, and write it into a preallocated output.

Units land on block boundaries, so every output element depends on one unit
only. That makes the streamed result identical to reducing the whole extent,
for every method and dtype, including float `area` and non-dyadic scales.
Peak residency is one unit, not the extent.

## 3. Read granularity: `read_block_shape`

The transfer grid is derived from the backend's native granularity (the
`native=` seed): the sizer either coalesces it up to the 8 MiB target or
divides it down. Dividing puts a streaming unit inside a block the backend can
only read whole, so every unit overlapping that block pays for all of it. The
default path therefore floors the unit at `read_block_shape`, which is the
identity wherever the grid coalesced.

`read_block_shape` answers "what can this reader not read below?", which is not
always the seed:

| declares | adapters | block |
| --- | --- | --- |
| a block | `ZarrAdapter` and its subclasses, `_QptiffLevelAdapter` | store chunk |
| | `Hdf5Adapter` | dataset chunk; `None` when contiguous |
| | `OmeTiffAdapter`, `TiffAdapter`, `LsmAdapter` | one page (`aszarr(chunkmode="page")` decodes it whole) |
| | `TiffSequenceAdapter`, `MicroManagerLegacyAdapter` | one strile |
| | `DicomAdapter`, `DicomSeriesAdapter` | one frame / slice file |
| | `NdTiffAdapter`, `EmdAdapter`, `_BioioAdapterBase` family | the dask block |
| | `LifAdapter` | one plane (readlif has no ROI read) |
| `None` | `MrcAdapter`, `DeltaVisionAdapter` | memmap; a crop costs its own pages |
| | `NiftiAdapter` | dataobj slicing |
| | `NikonAdapter`, `Nd2Adapter` | `read_frame` returns an mmap view; the seed (a whole C/Y/X frame) would floor every scaled read at the frame |
| | `CziAdapter` | a libCZI ROI read composes only the subblocks it touches |
| | `RemoteTensorAdapter`, `CachedSourceAdapter` | bounds forwarded upstream / served by chunk_id |

A wrong declaration is silent (values stay identical, the read costs more), so
`tests/adapter_read_block_test.py` requires every `TensorAdapter` subclass to be
listed as quantized or unquantized, with its reason.

**Why an OME-TIFF block is the whole page.** Under `chunkmode="page"`,
tifffile's zarr store decodes the entire page and slices the window out, and
nothing caches the decoded page, so N units cost N page decodes. The per-strile
mode reads proportionally but loses strip coalescing and threaded tile decode
(`maxworkers` applies only to page mode): on a 4096² page it is 9x slower for
one-row strips and 1.6-2.4x slower for compressed 512 tiles. Page mode is right
for reading a page, so the page is the block.

Measured effect of the floor, first-read latency for one plane, warm:

| source | unfloored | floored |
| --- | ---: | ---: |
| tiled OME-TIFF, 8192² page | 1719-1800 ms | 157-206 ms |
| OME-Zarr chunked at 4096², blosc | 369-416 ms | 94-131 ms |

## 4. `nearest` as a decimated read: `get_decimated_data`

`downsample_block(data, scale, "nearest")` is `data[::scale]`, so a backend that
can stride its own read serves `nearest` without touching the skipped bytes.
`get_decimated_data(bounds, step)` returns that strided read, or `None` (the
default) to leave the caller on the streamed path. The result must match
`get_data(bounds)[::step]` in shape and values, and be owned.

| adapter | how it strides |
| --- | --- |
| `MrcAdapter`, `DeltaVisionAdapter` | memmap indexing with a step |
| `NiftiAdapter` | `dataobj[slices]`; nibabel plans the read from the slice |
| `NikonAdapter`, `Nd2Adapter` | the step selects frames on the sequence axes and strides the mmap view inside a frame |
| `LifAdapter` | the step skips whole planes on T/C/Z/M; Y/X crop a decoded plane |

The capability correlates with `read_block_shape is None`: a quantized backend
decodes a whole block to return any of it, so a stride inside the block saves
only a memcpy. `LifAdapter` is the exception that fits the same logic: its
block is one plane, and the step skips whole planes. `tests/decimated_read_test.py`
classifies every adapter as decimating or not, checks values against
`get_data(...)[::step]` on real files including ragged off-origin extents, and
asserts on ND2 that only the frames on the step are decoded.

Measured on ND2, `nearest`, against the streamed default:

| file | scale | warm | cold |
| --- | ---: | --- | --- |
| 3789², 4C, 110 MiB extent | 4 | 46 -> 22 ms | 123 -> 97 ms |
| | 8 | 40 -> 13 ms | 115 -> **152 ms** |
| | 32 | 38 -> 8 ms | 113 -> 47 ms |
| 14234², 4C, 1.5 GiB extent | 8 | 470 -> 83 ms | 1854 -> 1058 ms |
| | 32 | 424 -> 17 ms | 1629 -> 270 ms |

Warm always wins: the full-resolution memcpy no longer happens. Cold depends on
readahead. A strided cold read runs at about half the sequential rate, and it
skips I/O only when the gap between picked rows, `(step - 1) x row_bytes`,
exceeds the kernel readahead window (128 KiB by default). Below that the kernel
faults the skipped rows anyway, which is the 1.3x cold loss at scale 8 on the
3789² file (53 KiB gap). It is shipped unconditionally: the loss is cold-only
and bounded, and gating it would mean predicting a kernel tunable from inside
an adapter.

## 5. CZI: `read(zoom=)`

`CziAdapter` overrides `get_scaled_data` to serve `nearest` from libCZI's own
`zoom=`, which produces the reduced pixels without decoding full resolution.
It is an override, not a capability, because no other backend scales during
the read.

It applies only when all of these hold, and otherwise falls through to the
default:

1. The method is `nearest`. `zoom=` differs from `area` in 100% of pixels.
2. Y and X reduce by the same factor, at least 2: one read takes one zoom.
3. Every other axis has scale 1. A scaled plane axis is a reduction across
   reads.
4. The factor divides both the Y and X extents. `zoom=` returns
   `floor(extent / f)` where the contract requires `ceil`, so the two agree
   exactly when `f` divides the extent. Divisibility is the predicate, not
   powers of two: 3x on a 510-wide ROI is exact.

A runtime check compares the returned shape with the contract's; a mismatch
logs and falls back rather than failing the read.

Measured on a synthetic 8192² x 3C uint16 CZI with no stored pyramid, identical
checksums:

| scale | streamed default | `zoom=` |
| ---: | ---: | ---: |
| 4 | 292.8 ms | 91.5 ms |
| 8 | 260.6 ms | 51.7 ms |
| 32 | 531.0 ms | 38.9 ms |

`tests/czi_adapter_test.py::TestZoomServesNearest` pins bit-identity across
factors 2-8 including 3 and 5, every declining case, that the zoom actually
reaches libCZI, and the fallback on a rounding mismatch.

## 6. Scaled reads from the cache

A plan-minted scaled chunk's extent tiles exactly into the absolute transfer
grid, so its full-resolution chunks are ordinary cache entries. Where every one
of them is cached, `cache_sourced_units` assembles units from the entries
instead of decoding the source again. It declines, with one index lookup on a
cold extent, when there is no cache, when the extent or unit is off the chunk
grid, or when any chunk is missing. An entry evicted between the probe and the
read falls back to a source read of that unit.

The unit follows the method:

- **`nearest`** sizes its unit at one chunk and *borrows* the entry's mapping,
  picking straight out of it; the returned array is materialised.
- **`area`** reads every byte anyway, so it keeps the source-path unit and
  copies chunks into one reused buffer (a fresh buffer per unit faults its
  pages during the copy: 31.7 ms against 8.3 ms for 32 MiB).

On an 8192² uint16 zarr chunked at 4096, one scale-16 chunk, warm: `nearest`
99 -> 9 ms, `area` 141 -> 56 ms, with compression making little difference.
Adapters whose `get_data` is already an mmap crop have nothing to gain, and they
serve `nearest` through §4 before reaching this.

On by default (`cache.source_scaled_reads`). Beyond the time saved, reading the
coarse level from the full-resolution entries leaves their pages resident for
the full-resolution read that usually follows.

## 7. The `area` kernel

Integer `area` with power-of-two scales sums each block into an accumulator
sized from `block_size * dtype_max` (`_plan_integer_area`), then divides,
rounds and clips on the reduced array. Other inputs (float, bool, non-dyadic
scales) take the float-accumulator path, which is unchanged.

Two kernels compute the same exact integer sum; the block size picks between
them:

| block | kernel | why |
| --- | --- | --- |
| <= 256 (`_STRIDED_ADD_MAX_BLOCK`) | strided adds, one pass per block offset | 1.4-7.6x faster at every input size measured |
| > 256 | per-axis reshape-sum, last axis first | reshape-sum gets cheaper as the block grows; strided loses ~2x at block 1024 |

The bound matters because the coarsest level is the first read a client makes:
the tensor browser opens a 14234-wide scene at scale 32, block 1024. The
accumulator must be the sized one, never a fixed width: a narrow accumulator
wraps silently. `tests/downsample_test.py` checks both kernels against an
in-test legacy oracle across the crossover in 2D and 3D, with saturated
`iinfo.max` / `iinfo.min` inputs, and checks the gate routes by block size.

## 8. Tests

| file | pins |
| --- | --- |
| `tests/scaled_data_seam_test.py` | scaled chunk_ids route through `get_scaled_data`, unscaled through `get_data`; the default matches the old inline path; cache-sourced reads and their ownership |
| `tests/stream_reduce_test.py` | streamed = whole-extent reduction for every method, float `area` and non-dyadic scales included |
| `tests/adapter_read_block_test.py` | every adapter classified quantized or not |
| `tests/decimated_read_test.py` | every adapter classified decimating or not; values, ragged extents, ND2 frame skipping |
| `tests/czi_adapter_test.py` | the `zoom=` gate (§5) |
| `tests/downsample_test.py` | the kernels, the gate, overflow |

Benchmarks: `benchmarks/bench_plane_latency.py` (first-read latency per plane,
cold or warm), `benchmarks/bench_nd2_direct_read.py`.
