# The fused scaling path (#640)

Plan for serving a *scaled* chunk without ever materialising its full-resolution
extent. Scope: `core/downsample.py`, `TensorAdapter.resolve_chunk_data`, and the
adapters that can read at a finer granularity than the extent.

## 1. Re-baseline: the issue's numbers no longer describe the tree

#640 was written before #797 (ND2 dask bypass), #643 / #686 (integer and float
accumulators in `downsample_block`), #809-#812 (adapter-owned transfer grid) and
#818 (demand tier). Its headline — 420.7 ms -> 95.3 ms, 4.4x — measured the dask
read path that no longer exists on ND2. Everything below is measured on today's
`dev`.

Setup: `ND040.nd2`, scene 0, `[1, 3, 1, 14234, 14234] uint16`, uncompressed,
interleaved `(Y, X, C)` on disk. **Real** chunk_ids taken from `get_read_plan`,
`pack_chunk_batch` included. Ryzen 5 5600X, warm page cache, median of 3.

**On the memory instrument.** These numbers are peak RSS (`VmHWM`, reset per arm
via `/proc/self/clear_refs`), one arm per process, split into anonymous and
file-backed by sampling `RssAnon`/`RssFile` at 2 ms. An earlier revision of this
document used `tracemalloc`, which is wrong here twice over: it cannot see
allocations made inside a C++ reader (libCZI hands back a buffer it owns), it
does not count mmap'd pages at all (the ND2 path reads through a mapping), and
its instrumentation inflates the timings of allocation-heavy arms — the coarse
`nearest` arm measured 495 ms under `tracemalloc` and 225 ms without it. Every
figure below is from the RSS runs.

The distinction between the two RSS columns is the point, not bookkeeping:
**anonymous pages are what a server OOMs on**, file-backed pages are page cache
the kernel reclaims under pressure.

`area`, extent `[1, 3, 1, 4728, 4728]` (128 MiB) at scale 4 and
`[1, 1, 1, 14208, 14208]` (385 MiB) at scale 32:

| scale | path | time | anon | file |
| ---: | --- | ---: | ---: | ---: |
| 4 | today | 330.7 ms | +191.9 MiB | +385.1 MiB |
| 4 | fused (band + fold) | **111.6 ms** | **+40.0 MiB** | +385.1 MiB |
| 32 | today | 340.3 ms | +406.1 MiB | +1152.8 MiB |
| 32 | fused (band + fold) | 466.8 ms | **+0.0 MiB** | +1157.3 MiB |

`np.array_equal`, max diff 0 on every row. Two corrections to what an earlier
draft claimed:

- **Fusing `area` does not reduce the bytes read, only the heap.** The same
  source rows must be touched either way, so file-backed residency is unchanged
  (+385 MiB at scale 4 — note that is *three times* the 128 MiB extent, because
  a sub-tile of an interleaved frame faults whole 85404-byte rows). What fusion
  removes is the anonymous half: 192 -> 40 MiB, and at scale 32 essentially all
  of it. That is the half that matters for #641's OOM path, but "18x less
  memory" was `tracemalloc` measuring only what it could see.
- **The scale-32 row is 1.4x slower, and it is the kernel gate, not the
  banding.** That prototype uses strided adds at block 1024, where §3's sweep
  says reshape-sum wins. With the gate it tracks today's time at zero anonymous
  cost.

`nearest` is the same seam and a much larger win, because a strided pick never
has to touch the bytes it discards — the one path where fusion cuts the *reads*
and not just the heap:

| scale | path | time | anon | file |
| ---: | --- | ---: | ---: | ---: |
| 4 | today | 84.4 ms | +135.8 MiB | +385.1 MiB |
| 4 | fused (strided pick) | **19.9 ms** | **+8.8 MiB** | **+105.7 MiB** |
| 32 | today | 225.4 ms | +385.1 MiB | +1157.3 MiB |
| 32 | fused (strided pick) | **11.2 ms** | **+1.2 MiB** | **+55.5 MiB** |

Pixels identical. 4.2x and **20x** on time; file-backed residency falls with the
scale (1/4 and 1/32 of the rows faulted) where `area`'s cannot. Two more things
fall out of it:

- **The deeper the level, the better `nearest` fuses**, because the extent grows
  with the scale while the output does not: at scale 32 the fused read copies
  1/1024 of the bytes and the unfused one copies all of them.
- **Today's `nearest` spends most of its cost making the view contiguous.**
  `downsample_block` returns a strided view and `pack_chunk_batch` then gathers
  a 0.38 MiB result out of a 385 MiB base through a stride of 32 rows x 6 bytes.
  The "free" view is free only until something has to materialise it, and it
  pins the full-resolution base until that happens. `TestZeroCopyContract`
  documents the ownership half; the cost half is not documented anywhere.

The split of today's 341.5 ms (`area`) is the part that redirects the rest of
the plan:

| stage | time |
| --- | ---: |
| `read_frame` mmap + strided copy of the tile (`get_data`) | ~40-90 ms |
| `downsample_block` on the resulting contiguous array | **~250 ms** |

**The read is no longer the cost; the reduction is** — about three quarters of a
warm scaled chunk. And the reduction is nowhere near its floor: the same sums
computed with strided adds instead of `reshape().sum()` take 65-71 ms.

That splits #640 into three wins that are worth different amounts and carry very
different risk:

- **W1 — the kernel.** `_area_reduce_integer`'s per-axis reshape-sum is 4x off a
  strided-add kernel at the scales clients actually open at. No seam change, no
  adapter change, bit-identical, helps every format *and* #816.
- **W2 — a bounded working set.** Reduce band by band so the extent is never
  resident. This is what the demand tier (#818) needs: it walks whole sources at
  the observed level, so peak resident per in-flight read is the OOM lever
  (#641's 5.4 GB field observation), not latency.
- **W3 — no full-resolution intermediate at all.** Fold the adapter's own native
  read units (an ND2 frame, a TIFF page, a CZI plane) straight into the output.
  Subsumes W2 for the adapters that can do it, and is the only one that needs
  the new seam. For `nearest` it is also the only one that removes the *bytes*:
  W1 and W2 cannot help a method that already does no arithmetic.

## 2. Where the code is today

```python
# core/adapter_base.py:855 -- resolve_chunk_data.compute_fn
result_arr = self.get_data(bounds)
if is_scaled_chunk_flag:
    scale_hint = decode_scale_info(chunk_id)
    reduction_method = decode_reduction_method(chunk_id)
    result_arr = downsample_block(result_arr, scale_hint, reduction_method)
result = pack_chunk_batch(result_arr)
```

This is the only seam that matters: `do_get`, the localhost locate handoff, the
HTTP tile route and precache all reach a scaled chunk through
`resolve_chunk_data`.

Three facts about the surroundings, all verified in the tree, that make the
change smaller than #640 assumed:

- **`NormalizingAdapter` needs no change.** `_permute_plan` carries `chunk_id`
  verbatim (`core/normalize.py:468`), so the wrapped adapter decodes *native*
  bounds and a *native* scale_hint, reduces in native order, and the wrapper
  transposes the reduced (kilobyte) result. Fusion propagates through
  normalisation for free; no scale hint has to be permuted by hand.
- **`RemoteTensorAdapter` and `CachedSourceAdapter` never reach it.** The first
  forwards the chunk_id upstream; the second has no backend (`get_data` raises).
- **The extent is a multiple of the scale except at the tensor edge.**
  `scaled_virtual_chunk_size` grows it in whole `lcm(transfer, scale)` units, so
  ragged blocks only occur where the extent ends on the tensor's own end.

## 3. Phase 0 — the block-sum kernel (no seam change)

`_area_reduce_integer` reduces one axis at a time with
`reshape(...).sum(axis, dtype=acc)`. Replace it, for small blocks, with a sum of
strided slices:

```python
def _area_reduce_strided(padded, scale_hint, accumulator):
    """Block-sum `padded` by strided adds. Same contract as _area_reduce_integer.

    `accumulator` MUST be the one _plan_integer_area sized, never a hardcoded
    uint32: it is chosen from block_size * dtype_max, so a uint16 input reduces
    into uint32 but a uint32 input needs uint64. Hardcoding the width is the
    overflow #639's scope note warned about, and it wraps silently -- the sum
    of block_size elements is exact in the sized accumulator and garbage in a
    narrow one, with no error either way.

    `padded` is already a multiple of the scale on every axis (_pad_array_edge
    ran first), which is what makes the strided slices tile it exactly. Do not
    substitute a trim: the pad is edge-replicated and divided by the FULL block
    size, so trimming changes the values at a tensor boundary.
    """
    acc = None
    for offsets in product(*(range(s) for s in scale_hint)):
        piece = padded[tuple(slice(o, None, s) for o, s in zip(offsets, scale_hint))]
        if acc is None:
            acc = piece.astype(accumulator)     # widens once, on the first term
        else:
            np.add(acc, piece, out=acc)         # every later term adds in place
    return acc
```

The sum is exact and order-independent for integers, so this is the identical
sum `reshape(...).sum()` computes — in a form numpy vectorises. Checked at the
extremes rather than assumed: every integer dtype with all elements at
`iinfo.max` and at `iinfo.min`, blocks 16/64/256, bit-identical to
`downsample_block` in all cases, with the worst-case block sum verified against
the chosen accumulator's own maximum.

The crossover is real and it is not noise: the two kernels scale in opposite
directions. Reshape-sum gets *cheaper* as the block grows (longer contiguous
inner sums, smaller output), while the strided kernel costs one pass over the
source per offset regardless. Swept over both axes that could move it — block
size and input size — on contiguous `uint16` square planes, `area`. The cell is
`downsample_block / strided-adds`; above 1.0 the strided kernel wins:

| scale (block) | 0.24 MiB | 0.95 MiB | 3.96 MiB | 15.8 MiB | 64 MiB | 256 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 (4) | 4.8x | 6.5x | 7.6x | 6.9x | 4.8x | 4.7x |
| 4 (16) | 3.3x | 4.6x | 5.5x | 7.1x | 5.8x | 3.9x |
| 8 (64) | 1.7x | 2.6x | 2.8x | 4.0x | 4.4x | 2.1x |
| 16 (256) | **0.53x** | 1.00x | 1.4x | 1.6x | 2.0x | 1.9x |
| 32 (1024) | 0.14x | 0.26x | 0.49x | 0.52x | 0.53x | 0.75x |

**The choice is a function of the block, not of the input size** — every row is
flat in size except block 256, which is the one that flips (2x loss at a quarter
MiB, 1.4-2.0x win from 4 MiB up). A sub-MiB piece costs a fraction of a
millisecond either way, so it does not earn a second term in the gate:

```python
_STRIDED_ADD_MAX_BLOCK = 256   # blocks <= 256 win at every realistic input size
```

Why bound it at all, when blocks <= 256 covers the levels a client browses
through: **the coarse end is not rare, it is the first thing requested.** The
tensor browser opens a 14234-wide scene at scale 32 (#818's `_tile_levels`
comparison), and that open is both the blank-screen event and the level the
demand tier then warms. Block 1024 loses 2x at every size, so an ungated kernel
would regress precisely the read a user waits on. The absolute numbers there are
small (76 vs 102 ms on 256 MiB) — which is the argument for keeping the bound
cheap rather than for removing it. Block 512 is untested; 256 stays on the
measured side of it.

**Evaluate the gate on the array actually being reduced**, not once per request.
Under banding (§5) or #816's compose the kernel is handed a band or a single
transfer chunk, and the 0.24 MiB column is where the answer changes.

`block_size` is already computed by `_plan_integer_area`, so the gate stays one
comparison. Which side of it matters more depends on the client: the tensor
browser *opens* at the coarse end (scale 32 on a 14234-wide scene, #818), where
the current kernel is already good, but every level it zooms into after that
lands in 2-8, where the current kernel is worst — and those are also the levels
whose extents are largest.

Notes for the implementer:

- **Bit-identity is the gate, and it is already written.**
  `tests/downsample_test.py::test_area_is_bit_identical_to_legacy` compares
  against `legacy_downsample_block`, an in-test oracle. Extend `_SCALES` with a
  block-size sweep that straddles 64 in both 2D and 3D (e.g. `(4,4)`, `(8,8)`,
  `(4,4,4)`, `(2,8,8)`) rather than adding a second oracle.
- **`nearest` and the float path are untouched.** `nearest` is a strided view
  and has no arithmetic to speed up (its cost is elsewhere — see §6.1); float
  `area` is not reassociable, so `_area_reduce` and `_float_accumulator` stay as
  they are. `_plan_integer_area` returning `None`
  (float, bool, 64-bit input, non-dyadic scale) keeps today's path exactly.
- **The trailing-pad copy is a separate cost, deliberately left alone.**
  `_pad_array_edge` copies the *whole* padded array whenever the extent is not a
  multiple of the scale. Trimming to the aligned interior and handling the
  margin strip separately is a real win at tensor edges, but it changes which
  elements are summed together and so needs its own bit-identity argument. File
  it; do not fold it in here.
- This lands independently of everything below and improves #816's streamed
  fold, the HTTP tile route and every non-fusing adapter at once.

## 4. Phase 1 — the seam

Add one optional method to `TensorAdapter` (`core/adapter_base.py`), with a
default that is exactly today's behaviour:

```python
def get_scaled_data(
    self,
    bounds: ChunkBounds,
    scale_hint: Tuple[int, ...],
    reduction_method: str,
) -> np.ndarray:
    """Read ``bounds`` and reduce it by ``scale_hint`` in one step.

    The default reads the whole extent and reduces it. An adapter whose reader
    can deliver the extent in pieces overrides this and folds each piece as it
    arrives, so the full-resolution extent is never resident -- and so no view
    onto a reader-owned mapping ever leaves the adapter's lock.
    """
    return downsample_block(self.get_data(bounds), scale_hint, reduction_method)
```

and call it from `resolve_chunk_data`:

```python
def compute_fn():
    if is_scaled_chunk_flag:
        result_arr = self.get_scaled_data(
            bounds, decode_scale_info(chunk_id), decode_reduction_method(chunk_id)
        )
    else:
        result_arr = self.get_data(bounds)
    return pack_chunk_batch(result_arr), result_arr.nbytes
```

Mechanical detail that will bite immediately: `core/adapter_base.py` asserts at
**import time** that `TensorAdapter`'s public API equals `_TENSOR_SCOPED_API`.
The new name has to be added to that frozenset in the same commit.

### Why a method and not a `scale_hint=` kwarg on `get_data`

As argued in #640's comment, and it holds up against the tree: 16 `get_data`
overrides each calling `super().get_data(bounds)` for validation; core cannot
tell whether an adapter honoured a kwarg (an adapter that ignores it returns
full-resolution pixels that get served as reduced — wrong pixels, no error); and
a base implementation that is always correct beats a capability flag that has to
stay in sync with a kwarg.

### The contract to document on the ABC

`get_data`'s docstring currently promises only "Numpy array with data within the
requested bounds". State the real invariant, which `core/normalize.py:419`
already relies on: **the returned memory's lifetime must not be tied to a
closable handle.** A transpose view over an owned array is fine; a view over an
mmap the `_handle_reaper` can close is not. Then state, on `get_scaled_data`:

1. **Shape** = `ceil((stop - start) / scale)` per axis — identical to
   `downsample_block`'s output, padding included.
2. **Dtype** = `get_output_dtype(base_dtype, method)`, i.e. the input dtype.
3. **Values** = bit-identical to
   `downsample_block(self.get_data(bounds), scale_hint, method)`. An override
   that cannot be bit-identical must not be reached by default (see CZI `zoom=`
   below).
4. **Ownership** = an owned array. In particular a fused `nearest` must
   materialise; only the default may return the strided view that
   `TestZeroCopyContract` pins today, because only there is the base array owned
   and already off the reader.
5. **Fallback** = anything the fused path cannot express bit-identically calls
   `super().get_scaled_data(...)`.

### The shared streaming reducer

Every fused caller needs the same primitive: fold a piece of the extent, given
its local offset, into an accumulator indexed by output position. #816 already
wrote it (`core/compose.py`: `_block_breaks`, `_accumulate_area`,
`_assign_nearest`, `_edge_padded`, and `streaming_area_plan` exported from
`downsample.py` for exactly this reason). Promote it rather than writing a
second one:

```python
# core/scaled_reduce.py
class ScaledReducer:
    def __init__(self, extent_shape, scale_hint, reduction_method, dtype): ...
    @property
    def can_fuse(self) -> bool: ...        # False -> caller must read whole + downsample_block
    def add(self, piece, local_start): ...  # fold one piece, then drop it
    def result(self) -> np.ndarray: ...     # divide / round / clip / cast, owned
```

- `can_fuse` is `can_compose`'s body: `nearest` always; `area` only when
  `streaming_area_plan` yields an accumulator (integer input, dyadic scales);
  `precompute` never.
- `add` must reduce **the last axis first**. This is not cosmetic: on the ND2
  frame view, folding Y before X cost 394 ms against 208 ms for X before Y on
  the same data. `downsample_block` already iterates `reversed(range(ndim))`;
  `compose.py::_accumulate_area` iterates forward and should be flipped —
  a free win for #816 as well.
- `add` uses the Phase 0 kernel for pieces that start on a block boundary (the
  common case) and `np.add.reduceat` with `_block_breaks` for a piece that
  straddles one, so a straddling block receives from both sides.
- `_edge_padded` carries `downsample_block`'s edge-replicate pad on whichever
  piece ends the extent. This is load-bearing for bit-identity: the pad
  replicates the extent's own last element and the divide is by the **full**
  block size, so a partial edge block averages that element repeated.

`compose.py` then imports from here, and the two paths cannot drift on which
inputs qualify.

## 5. Phase 1.5 — a banded default (optional, per adapter)

W2 without any per-format code: the base `get_scaled_data` splits `bounds` into
row bands, calls its own `get_data` per band, and folds each.

Measured on the ND2 chunk above (band = 256 source rows, ~7 MiB):

| path | scale 4 | scale 32 (384 MiB extent) |
| --- | ---: | ---: |
| today (whole extent + `downsample_block`) | 287.5 ms / 128 MiB | 219.6 ms / 384 MiB |
| banded + reshape-sum | 263.5 ms / 7 MiB | **184.4 ms** / 12 MiB |
| banded + strided adds | **64.8 ms** / 7 MiB | 322.3 ms / 12 MiB |

Banding never lost, and with the Phase 0 kernel selection it wins at both ends
while cutting resident memory by 18-32x.

### Bounded memory is not the main reason

An earlier draft presented banding as a memory measure that happens not to cost
time. That undersells it: **banding is a throughput win in its own right, and it
has nothing to do with the OS.** On an array already in RAM — no file, no mmap,
no page cache — with byte-identical arithmetic:

| scale | band | whole | banded | |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 4 MiB | 61.5 ms | 30.1 ms | **2.05x** |
| 4 | 16 MiB | 61.5 ms | 30.0 ms | **2.05x** |
| 4 | 64 MiB | 61.5 ms | 51.4 ms | 1.20x |
| 32 | 4 MiB | 37.5 ms | 37.3 ms | 1.01x |

This is loop tiling against L3, not paging. Reducing the whole extent makes two
passes over it — `get_data` materialises it, then the reduction re-reads it from
DRAM. Banding fuses them: each band is still in L3 when the reduction touches it.
The 64 MiB row is the control — one band over this box's 32 MiB L3 collapses the
win to 1.20x. It would hold identically on tmpfs or a RAM-backed array.

At scale 32 it is neutral (1.01-1.04x): reshape-sum already streams, so there is
little locality left to recover. The win lives in the 2-8 band, which is where
the extents are largest and where clients zoom.

### The band budget is a cache size, not a knob

Swept on contiguous `uint16`, ratio against the unbanded reduction:

| band | 0.5 | 1 | 2 | 4 | **8** | 16 | 32 | 64 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scale 4 | 1.45x | 1.68x | 1.90x | 2.03x | **2.06x** | 2.05x | 1.71x | 1.20x |
| scale 8 | 0.85x | 1.17x | 1.42x | 1.69x | **1.79x** | 1.88x | 1.78x | 1.55x |

Both ends are real: under ~2 MiB per-band overhead starts to bite (0.85x — an
actual loss), over ~16 MiB the band stops fitting in cache. **8 MiB**, which is
also the ~7 MiB band the table above was measured with. An earlier draft proposed
~32 MiB; that is exactly this box's L3 and already past the knee.

So it is a constant, not a setting. The right value is a property of the cache
hierarchy, and an operator tuning it would be guessing at L3. If it ever has to
move it should be *derived* from cache size, not configured.

### It does not become a second `max_read_block_mb`

`PyramidConfig.max_read_block_mb` (default 512) already exists for this concern —
"Ceiling, in MiB, on the source pixels one computed-scale chunk may materialize.
A resident-memory bound, not a throughput knob". It reaches that bound by a
different route: it clamps the virtual chunk size at plan time
(`scaled_virtual_chunk_size`), so a memory-constrained server gets **smaller
chunks and more round trips** — and this read path is latency-bound, capped
around 5.4k Flight dispatches/s.

Banding bounds the same residency by streaming instead, at no round-trip cost.
The two must not both become knobs for one concern. Once an adapter bands, the
extent is never resident, so `max_read_block_mb`'s justification no longer
applies to it and the clamp is only costing round trips: a banded adapter should
be allowed to **relax** that ceiling. That is a throughput win this plan can
claim on top of the memory one, and it is the reason the band budget stays
internal — the config surface for "how much may one read materialize" is already
taken, and adding a second one would leave an operator with two dials for one
outcome and no way to tell them apart.

Any config here is better spent on an escape hatch (force-disable banding while
debugging) than on a tuning dial.

### Constraints on making it the default

- **Band along rows, never into tiles.** #816 measured square retiling on this
  exact file at +1.7 s: a 1182-wide tile out of a 14234-wide frame is 1182
  memcpys of ~7 kB. Full-width bands keep the rows long.
- **Bands must land on block boundaries.** A band that splits a reduction block
  would fold the two halves independently and produce a different pixel, so band
  height is a whole multiple of the row scale. Only the last band may be ragged,
  and that is the one place `downsample_block`'s edge-replicate pad applies —
  which is exactly where it applies for the unbanded read, since the extent ends
  where the tensor does. That is what keeps the banded path bit-identical rather
  than merely close.
- **`nearest` bands too, and it is a memory trade there, not a speed win.**
  Measured through `pack_chunk_batch` on a 128 MiB extent: `area` 62.0 -> 29.7 ms
  (2.09x), `nearest` 1.8 -> 2.0 ms (0.90x). There is no arithmetic pass for the
  read to fuse with, so the locality win does not exist. Take it anyway: unbanded
  `nearest` returns a strided view that pins the full-resolution base until
  `pack_chunk_batch` materialises it, so its peak IS the extent — and `nearest`
  is now the default method. What banding cannot give `nearest` is fewer bytes
  read; that needs the reader to skip rows (§6), since doing it generically would
  mean one call per picked row.
- **Only where a band read is proportionally cheaper than the extent read.**
  True for mmap/uncompressed sources (ND2, MRC, plain TIFF via the `aszarr` page
  store, zarr, HDF5). False where the reader must decode a whole plane per call
  — banding there multiplies decode work by the band count. So this is opt-in: a
  class flag (`BANDED_SCALED_READ`), not a behaviour change for every adapter at
  once. An adapter that already overrides `get_scaled_data` (Phase 2) ignores it
  by construction, since it never calls `super()`.

  The flag stays a class attribute rather than config on purpose: it asserts a
  property of the *reader* ("a band read costs proportionally less than the
  extent read"), which an operator has no way to know — nobody outside the code
  knows whether libCZI decodes a whole plane per call.

- **Two gaps, stated rather than assumed.** Every number here is warm page cache,
  median of 3; the cold path is unmeasured (banding should still win there, since
  compute on band N overlaps readahead of band N+1, but it is not measured).
  And the flag is keyed on the adapter class while the cost model is really about
  *storage*: an ND2 on NFS — which is what biopb.org runs — is still mmap-able,
  but a band read there multiplies round trips and works against server-side
  readahead. `core/fs_detect.py` already classifies exactly this
  (`network_filesystem_type`), and the band decision should consult it before
  this is turned on for a network-backed source.

## 6. Phase 2 — per-adapter fusion

| adapter | native unit | technique | gate | expected |
| --- | --- | --- | --- | --- |
| `NikonAdapter` (ND2) | sequence frame, mmap, zero-copy | fold row bands of `read_frame(seq)[slices]` inside `with self._io_lock, nd2.ND2File(...)`; strided pick for `nearest` | direct path available (`_nd2_frame_indices` not None, labels in TCZYXS) | **`area` 331 -> 112 ms, anon +192 -> +40 MiB; `nearest` 84 -> 20 ms at scale 4 and 225 -> 11 ms at scale 32, anon +385 -> +1.2 MiB. Measured, pixels identical** (§1) |
| `CziAdapter` | one `reader.read(plane=, roi=)` per plane, owned buffer | band-wise ROI reads folded as they arrive; `read(zoom=1/f)` for `nearest` | `zoom=` is bit-identical to **`nearest` only** (100% of pixels differ from `area`); needs a power-of-two shape guard (3x -> 1365 vs 1366) | **`area` 387 -> 150 ms, +274 -> +104 MiB at scale 4; `nearest` via `zoom=` 131 -> 26 ms at scale 4 and 384 -> 37 ms, +641 -> +0.3 MiB at scale 32. Measured, identical** (§6.0) |
| `OmeTiffAdapter` / `TiffAdapter` / `LsmAdapter` | one TIFF page per plane (`aszarr`, `chunkmode="page"`) | fold per page inside `_read_region` | none beyond `can_fuse` | kernel + bounded memory; unmeasured |
| `ZarrAdapter` / `OmeZarrAdapter` | store chunk | prefer the native pyramid (`precompute`); fold per store chunk otherwise | `can_fuse` | overlaps #816; low priority |
| `NdTiffAdapter`, `DicomAdapter`, `Hdf5Adapter`, `MrcAdapter`, `NiftiAdapter`, `EmdAdapter` | — | default path; Phase 0 kernel only | — | 2.5-4.7x on the reduce at scale <= 8 |

ND2 first: it is measured, it is the format this catalog is dominated by (151
sources, 1.62 GB per frame on the plates), and the fused fold sits inside a lock
scope that already exists.

### 6.0 CZI, measured

The CZI datapoint posted in #640's comments **reproduces on today's tree** — the
plane-level table (4096x4096 uint16, no stored pyramid) is within a few percent
on every cell, and `read(zoom=1/f)` is still bit-identical to
`downsample_block(..., "nearest")` at 2x, 4x, 8x and 16x:

| factor | read+`nearest` posted -> now | read+`area` posted -> now | `zoom=` posted -> now |
| ---: | --- | --- | --- |
| 2x | 28.2 -> 26.5 ms | 154.2 -> 149.0 ms | 13.5 -> 14.3 ms |
| 4x | 28.1 -> 25.7 ms | 88.2 -> 84.7 ms | 5.5 -> 5.8 ms |
| 8x | 28.0 -> 25.6 ms | 60.4 -> 58.5 ms | 3.8 -> 3.6 ms |
| 16x | 28.1 -> 26.3 ms | — | 3.4 -> 3.0 ms |

At adapter scale, through `CziAdapter` with real chunk_ids from `get_read_plan`
(synthetic `[1, 3, 1, 8192, 8192] uint16`, `pack_chunk_batch` included, every row
bit-identical to today's output). Peak RSS delta over the arm's own baseline,
one arm per process — libCZI reads with `pread` rather than mmap, so unlike the
ND2 tables this is all anonymous:

| scale | extent | `area` today | `area` fused (band ROI + fold) | `nearest` today | `nearest` fused (`zoom=`) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 128 MiB | 387.4 ms / +274.3 MiB | **150.3 ms / +103.5 MiB** | 131.3 ms / +264.4 MiB | **25.9 ms / +16.0 MiB** |
| 32 | 384 MiB | 507.2 ms / +649.7 MiB | 618.8 ms / **+44.2 MiB** | 384.1 ms / +640.7 MiB | **37.3 ms / +0.3 MiB** |

The `zoom=` row is the one that answers "does libCZI decode full resolution
internally and throw it away?" — **no**: at scale 32 the whole arm adds 0.3 MiB
over baseline for a 0.38 MiB chunk out of a 384 MiB extent. Nothing
full-resolution is ever allocated, in C++ or anywhere else. (That claim needs
RSS to make; `tracemalloc` cannot see a buffer the binding owns, and reported
the same arm at 1.3 MiB by accident of measuring only the numpy wrapper.)

Three more things to take from it.

**One posted conclusion needs qualifying.** The comment says of the `area` cost:
"there is no downsampler optimization that captures this; the only way past it
is to not materialize full resolution." Its evidence was that `downsample_block`
is at par with a hand-rolled reshape-sum (126 vs 107 ms) — but those are two
spellings of the *same* kernel. The Phase 0 strided-add kernel is a different
one, and on the same plane it takes read+`area` from 149.0 to **49.0 ms** at 2x
and 84.7 to 34.2 ms at 4x, bit-identical. So a downsampler optimisation does
capture most of it — in the 2x-8x band. By 16x the read (26 ms of a 43 ms total)
dominates and the claim holds again: there, only `zoom=` or a native level helps.

**The `area` fused row at scale 32 is a regression, and it is the expected one.**
That prototype applies strided adds unconditionally; at scale 32 the block is
1024 and §3's crossover says reshape-sum wins. With the Phase 0 gate in place the
banded path tracks today's time at 30x less memory instead of costing 1.2x. It is
worth stating plainly because it is the same trap in both formats: banding is
always a memory win, but the kernel inside the band has to be chosen by block
size or it turns into a time loss at the coarse end.

**`zoom=` is the largest single win anywhere in this plan** — 6.5x at scale 4 and
11.2x at scale 32, with peak allocation falling from 512 MiB to 0.5 MiB, because
libCZI never decodes the full-resolution pixels at all. It stays gated to
`nearest` (100% of pixels differ from `area`) and needs the power-of-two shape
guard the comment names (3x -> 1365 vs 1366). Since `nearest` is now the default,
that gate is on the common path rather than an opt-in one.

Unchanged from the posted comment: these fixtures have **no stored pyramid**, so
route 2 (native levels via `get_native_pyramid_levels()` + `precompute`) is still
unmeasured, and on a real whole-slide CZI it should beat `zoom=` outright.

**Reachability: settled, ahead of the implementation.** When this was written no
client requested `nearest` — biopb-mcp asks for the advertised level's method or
omits the field, and the HTTP tile route's `red` parameter defaults to unset —
so a fused `nearest` decode would have been live code on a dead path. The
**default reduction method is now `nearest`** (committed separately, `#640`), so
every method-free read reaches it: the mcp fallback ladder, the tile route, and
anything else that omits the field. `area` remains a first-class request for a
caller that wants averaging. The aliasing trade that decision carries is
described where the default is defined (`core/downsample.py`).

### 6.1 Do `nearest` first within each adapter

Within an adapter, the `nearest` override is both the cheaper half to write and
the larger win, so it should not wait behind `area`:

- **No accumulator, no bit-identity argument.** A strided pick is exact by
  construction for every dtype and every scale — integer, float, non-dyadic
  alike. None of `_plan_integer_area`'s preconditions apply, so `can_fuse` is
  unconditionally true and there is no fallback branch to test.
- **It is the only path where fusion removes bytes rather than passes.** `area`
  must read every source element whatever it does with them, so fusing it leaves
  file-backed residency untouched; `nearest` faults 1/scale of the rows. That is
  why its ratio *improves* with depth (4.2x at scale 4, 20x at scale 32) where
  `area`'s decays.
- **It is now the default path.** With the default flipped to `nearest`, this is
  what an unqualified scaled read runs, so it is where fusion is worth the most
  and where a regression would be felt first.

Consequences elsewhere in this plan:

- The fused `nearest` **must materialise** (contract rule 4, §4). It is picking
  straight off a reader-owned mapping inside the lock, so returning the view
  would hand the caller a dangling mapping — and it would keep the whole extent
  alive as its base, which is the memory this phase exists to not spend. Copy
  into a preallocated output as the pick happens; the output is the chunk, so
  the copy is the delivery.
- A sequence axis that is *scaled* should be **skipped, not read**: at
  `scale_hint` 2 on T, only every second frame contributes, so the frame loop
  must step by the scale rather than read-and-discard. (T = Z = 1 on the
  measured file, so this is untested — it is the one place a fused `nearest`
  can silently read 2-32x more than it needs.)
- CZI's `read(zoom=)` route (§6 table) is `nearest`-only, which looked like a
  narrow gate when `area` was the default worth optimising. Given the numbers
  above it is the *interesting* gate: it stacks a decode-side 2-8x on top of the
  transfer-side win, for the method that fuses best anyway.

CZI's native-pyramid route (`get_native_pyramid_levels()` +
`reduction_method="precompute"`, already modelled for OME-Zarr) is strictly
better than `zoom=` where the file has stored subblocks — it sidesteps the
reduction rather than swapping which one runs, and carries no fidelity conflict.
It is blocked on a genuinely pyramidal CZI fixture (pylibCZIrw's writer emits
none; same gap as #799). Do not let the `zoom=` route (opt-in, `nearest`-only)
become the reason the pyramid route never lands.

## 7. Tests

- **Kernel (Phase 0).** Extend the existing `legacy_downsample_block` sweep in
  `tests/downsample_test.py` across the block-size crossover, 2D and 3D, every
  integer dtype. No new oracle.
- **Kernel overflow.** A saturated case per dtype — every element at `iinfo.max`,
  and at `iinfo.min` for the signed ones — at the largest block the gate admits.
  `TestNoAccumulatorOverflow` already exists for the reshape-sum path; the
  strided kernel needs the same cases pointed at it, because its failure mode is
  a silent wrap rather than an exception. This is the test that fails if someone
  later replaces the sized accumulator with a literal `np.uint32`.
- **Seam (Phase 1).** A spy asserting `resolve_chunk_data` routes a scaled
  chunk_id through `get_scaled_data` and an unscaled one through `get_data`; and
  that the base implementation is byte-identical to the old inline path.
- **Differential suite (Phase 2), new `tests/scaled_read_test.py`.** For every
  adapter that overrides the method: geometries x dtypes x methods against
  `downsample_block(adapter.get_data(bounds), ...)`. Reuse `compose_test.py`'s
  36-case shape — its geometry list (ragged edges, corner pads, offset extents,
  anisotropic scales) is the one that catches the straddling-block and edge-pad
  mistakes.
- **Ownership.** The returned array must survive the reader: read, then
  `adapter.close()` (or trip `_handle_reaper`), then assert the values still
  read back. Plus `not isinstance(arr.base, np.memmap)` and, for fused
  `nearest`, that it does not share memory with anything the adapter holds.
  `nearest` is the case worth asserting hardest: it is the one where returning
  the view is both the obvious implementation and wrong twice over (dangling
  mapping, plus an 8 MiB chunk pinning a 385 MiB base). Assert
  `arr.nbytes == arr.base.nbytes` if a base exists at all, so a view that merely
  looks small cannot pass.
- **Scaled sequence axes.** A `nearest` read with `scale_hint > 1` on T or Z must
  read only the contributing frames — assert on the reader's call count, not on
  pixels, since reading and discarding gives the same answer.
- **Fallback.** Float `area`, scale 3, and `precompute` must reach
  `super().get_scaled_data` — assert by spying on `downsample_block`, not by
  comparing pixels (which would pass either way).
- **Memory.** A precache/demand-tier test that peak allocation during a warm
  pass stays bounded by the band budget rather than tracking
  `max_read_block_mb`.

## 8. Benchmarks

`benchmarks/bench_scaled_read.py`, following `bench_nd2_direct_read.py`: for a
given source, take real chunk_ids from `get_read_plan` at scales 2/4/8/32 **for
both `area` and `nearest`** and report time and peak allocation for today's path
against the fused one, with `np.array_equal` asserted in the harness (the ND2 bench already fails loudly on
a pixel mismatch; keep that). The per-scale crossover in §3 is the thing most
likely to differ on other hardware, so the bench should print it rather than the
plan asserting it forever.

## 9. Relationship to the other open work

### #816 (compose a scaled chunk from cached full-resolution chunks)

They nest rather than compete: compose sits *above* this seam (it decides not to
read the source at all), and where it declines, `get_scaled_data` runs. They
must share one streaming reducer (§4), and #816's fold should adopt the
last-axis-first order and the Phase 0 kernel.

The open question is #816's unresolved regression — composing fetches on the
narrow transfer grid, which cost it +1.72 s of `get_data` on ND040. Measured on
that same file, extent 4728x4728, grid 1182x1182, warm, all three producing
identical tiles:

| read pattern | time | vs wide |
| --- | ---: | ---: |
| 1 wide read 4728x4728 — what a scaled read does today | 79.2 ms | 1.00x |
| 16 narrow grid reads of 1182x1182 — what composing does | 203.7 ms | **2.57x** |
| 4 band reads of 1182x4728, each sliced into 4 grid tiles | 115.0 ms | **1.45x** |

**Mechanically this plan is neutral on the amplification.** It changes what a
scaled read does internally; the grid, the plan and the per-chunk promise
protocol are untouched, and when compose is on `get_scaled_data` is never
reached at all.

**Competitively it makes #816's position worse, in two ways.**

- *Phase 0 removes the arithmetic that was masking the read penalty.* Applying a
  conservative 3x to both arms of #816's own table: composed 5.82 + 2.41 + 0.14
  -> ~6.76 s, plain 4.10 + 2.59 -> ~4.96 s. The gap stays ~1.8 s in absolute
  terms and *grows* as a ratio, 1.28x -> 1.36x. A faster reducer helps the arm
  it is competing against exactly as much.
- *Fusion cuts the cost of the alternative.* On ND2 the direct scaled read goes
  331 -> 112 ms for a 128 MiB extent (§1). Composing has to beat that, not the
  old number, on every format where fusion applies.

**In shared infrastructure it makes #816's way out cheaper to build.** The PR
names it — "read wide, cache narrow: one `get_data` per row band of chunks,
split into the grid-aligned entries" — and calls it not small because it needs a
bulk path around the per-chunk promise protocol. The table above says that shape
recovers 2.57x -> 1.45x, and Phase 1.5/2 builds band iteration over an extent
anyway. Factor it as a reusable helper at the seam rather than inlining it in
each adapter, and #816's remaining work is exposing it, not writing it.

**Compose keeps the cases fusion cannot serve**, and they are the ones it was
filed for: a *warm* cache, where a 1/2 read costs 0.2 MB of disk and no decode
can compete; `CachedSourceAdapter`, whose `get_data` raises so composing is the
only way to serve a scaled read of an upload; and any adapter with no finer
native granularity to fuse into.

One synthesis worth naming and not adopting yet: a fused `area` read already
walks the full-resolution bytes band by band, so it could split each band into
grid-aligned tiles and cache them on the way past — #808's payoff with **zero**
read amplification, since the bytes are read either way. The cost is cache
*space*, not time: 128 MiB of full-resolution entries per scaled chunk, which
would have the demand tier hydrate whole sources at full resolution. It would
need its own budget before it could be turned on.
### #818 (demand tier)

This is the workload: an observed scaled read warms the whole source at that
level (20.1 GB decoded for a 200-frame timelapse). Fusion does not shorten that
walk, it bounds what it holds while walking — anonymous residency per in-flight
read goes from the extent (up to `max_read_block_mb`) to one band, which is the
half of peak RSS a server OOMs on (§1).

### #639 / #643 / #686

Closed. The float64 promotion, unit-axis walks and full-resolution rounding are
genuinely gone; do not re-derive them. What §3 fixes is the kernel *inside* the
integer path those PRs introduced.

## 10. Sequencing

| PR | content | risk | independent? |
| --- | --- | --- | --- |
| 1 | Phase 0 kernel + gate + sweep | low (existing bit-identity oracle) | yes |
| 2 | `get_scaled_data` seam, ABC contract, `ScaledReducer` extracted from `compose.py` | low (default preserves behaviour) | needs 1 for the kernel, not for correctness |
| 3a | ND2 fused `nearest` (strided pick) | low-medium — exact by construction, no accumulator | needs 2 |
| 3b | ND2 fused `area` (banded fold) | medium (lock scope, frame indexing, edge pads) | needs 2, 3a |
| 4 | banded default + opt-in flag | medium (per-adapter read economics) | needs 2 |
| 5 | OME-TIFF / TIFF per-page fold | medium | needs 2 |
| 6a | CZI fused `nearest` via `read(zoom=)` behind the method gate | low-medium; measured 6.5-11.2x | needs 2 |
| 6b | CZI banded ROI fold for `area` | medium | needs 1 (for the kernel gate) and 2 |
| 6c | CZI native pyramid route (`precompute`) | medium; blocked on a pyramidal fixture | needs 2 |

By payoff per unit of risk the order is **1, 2, 3a, 6a, 3b/6b, 4, 5, 6c** — PR 1
because it is the broadest win and gated by an oracle that already exists, then
the seam, then the two `nearest` fusions (largest ratios, exact by construction).
6a no longer carries a client-opt-in question: the default flip landed first, so
both `nearest` fusions sit on the path an unqualified read already takes.

## 11. Non-goals and open questions

- **`max_read_block_mb` should probably grow once memory is bounded** — its
  whole job is capping resident bytes, and a fused read no longer holds them.
  Bigger extents mean fewer chunk_ids and fewer reads. Deliberately out of scope
  here: it changes minted chunk_ids and therefore cache keys.
- **`_pad_array_edge`'s whole-array copy** at tensor edges (§3).
- **float32 `area` fusion.** Staged means do not reassociate; there is no
  bit-identical streamed form. Would need an explicit accuracy trade, as #686
  took for the accumulator width.
- **`nearest`'s byte-skipping is partly an OS question.** The fused pick touches
  1/scale of the rows, but page-cache readahead may fault a good deal of what it
  skips. The measured wins are warm-cache, so they are memcpy and cache-line
  wins that are certain; the cold-read saving is plausible and unmeasured. Worth
  a cold-cache arm in the bench before quoting an I/O number.
- **Measured on two formats, warm cache.** ND2 (real file) and CZI (synthetic,
  no stored pyramid) are measured; the TIFF/zarr rows in §6 are still inference
  from their read shapes, and #640's own closing caveat ("worth measuring per adapter before generalizing") still
  applies. `docs/dask-bypass-benchmarks.md` is the precedent for doing that
  measurement before the code.
