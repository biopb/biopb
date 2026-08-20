# Is bypassing dask worth it beyond ND2?

Follow-up to #640, which measured ND2 only and closed with "worth measuring per
adapter before generalizing". This is that measurement, across every format that
actually reaches `adapters/bioio.py`.

**Answer: yes, and the reason is not the one #640 gives.** The dominant cost is
not per-byte and not a fixed per-read overhead. It scales with the number of
blocks in the *whole* dask array, so it is worst on long time series — and
`_FRAME_CHUNK_DIMS` makes it worse, because forcing one block per frame
maximizes the block count.

## Scope

Only these reach the bioio adapter. OME-TIFF (`OmeTiffAdapter`), TIFF sequences
and MicroManager (`tiff.py`) are pure tifffile on a persistent `aszarr` store and
never touch dask, so they are **out of scope** — an earlier revision of this
document benchmarked them by mistake.

| adapter | extension | bioio plugin | native reader |
| --- | --- | --- | --- |
| AicsImageIoAdapter | `.tif`/`.tiff` (plain) | bioio-tifffile | tifffile |
| ZeissAdapter | `.lsm` | bioio-tifffile | tifffile |
| ZeissAdapter | `.czi` | bioio-czi | pylibCZIrw |
| LeicaAdapter | `.lif` | bioio-lif | readlif |
| NikonAdapter | `.nd2` | bioio-nd2 | nd2 (already bypassed) |
| DvAdapter | `.dv` | bioio-dv | mrc |

The rest of the claim surface is Java-only (`.oif .oib .zvi .lei .vsi` and most of
`MICROSCOPY_EXTENSIONS`) or has no installed plugin at all (`.klb .fit .fts .mhd
.mha .jpeg .mpeg .mpg`), which claim-then-error. `.cif` is an extension-table
artifact: `bioio-imageio`'s own `ReaderMetadata` does not list it.

## Method

`get_data` does `self._dask_data[slices].compute()`. Each row compares that
against the identical region read straight from the native library, verified
with `np.array_equal` (every row below is `match=True`; an earlier revision
failed this and was reading the wrong planes).

- `BioImage` is built **outside** the timed region, as the adapter holds it, and
  with the `chunk_dims` biopb passes (`("Y","X","S")` for `.tif`/`.lsm`/`.lif`).
- Best of 3, warm page cache — this isolates CPU/graph cost, which is what is
  under test. A cold-cache run would narrow every ratio.
- The native path always **copies**. A memmap view costs nothing and reads
  nothing; the first DV number measured here was inflated 3x by exactly that,
  which is the trap #640 flags about views escaping the lock.
- Ryzen 5 5600X, dask 2026.6.0, bioio 3.4.0.

`.tif` and `.nd2` are real files. `.lsm`, `.lif`, `.czi` and `.dv` are
synthesized — see "Fixtures" below.

## One frame, in biopb's configuration

Averaged over a walk of **distinct** frames on both sides. Re-reading a single
frame keeps its page offsets warm and flatters the native side -- on the
2800-page TIFF that alone was the difference between 0.05 and 0.086 ms.

| format | adapter | blocks | dask | native | |
| --- | --- | --- | --- | --- | --- |
| `.tif` | aics | 2800 | 235.2 ms | 0.086 ms | **2730x** |
| `.lsm` | zeiss | 30 | 11.3 ms | 0.009 ms | 1224x |
| `.lif` | leica | 20 | 2.5 ms | 0.044 ms | 57x |
| `.dv` | dv | 80 | 1.4 ms | 0.060 ms | 23x |
| `.czi` | zeiss | 40 | 3.6 ms | 2.019 ms | 2x |
| `.nd2` | nikon | 1 | 141.1 ms | 70.2 ms | 2x |

The ratio tracks block count, not format. `.nd2` is bottom of the table only
because that file is a single frame (T=1, Z=1) and therefore one block -- the
same file type with 40 000 frames sits where `.tif` is. `.czi` is last on merit:
pylibCZIrw genuinely spends ~2 ms decoding a 1024x1024 plane, so there is little
overhead to remove.

Native reads run at 1.5-1.9 GB/s against a 13-15 GB/s memcpy floor -- the gap is
tifffile's per-page Python work, not I/O. They are real reads: `pages.cache` is
False and `asarray()` returns a fresh array each call.

## Cost scales with blocks in the array

Same 8 KB frame every time; only the file's frame count changes.

| frames | `.tif` | `.lsm` | `.lif` | `.czi` | `.dv` |
| --- | --- | --- | --- | --- | --- |
| 100 | 10.1 ms | 16.5 ms | 7.0 ms | 6.5 ms | 1.7 ms |
| 500 | 30.4 ms | 44.7 ms | 27.5 ms | 23.4 ms | 1.7 ms |
| 2 000 | 126.4 ms | 171.9 ms | 107.1 ms | 111.6 ms | 1.5 ms |
| 8 000 | 713.6 ms | | | | |
| 20 000 | 2199.7 ms | | | | |

Native reads stay flat (0.03–0.18 ms) throughout.

**`.dv` is the exception, and it explains the mechanism.** `bioio-dv` builds its
array with `map_blocks`; every other plugin here uses the aicsimageio pattern,
`da.block([...from_delayed...])` — one task per chunk. `_optimized_dsk` is a
`cached_property`, but `arr[slices]` constructs a *fresh* expression on each
read, so the whole graph is re-optimized every time. Profiling a 40 000-frame
source: of 9.7 s, **5.5 s is `__dask_graph__` → `optimize`/`order`/`fuse`/`cull`**
and only 1.4 s is execution.

This is why `_FRAME_CHUNK_DIMS` (commit `0e5c7a50`) cuts both ways. It removes
read amplification — without it a one-plane read materialized a whole Z-stack —
but it maximizes block count. On `organoids.tif` the same single frame costs
48.3 ms under bioio's default chunking and **221.5 ms** under frame chunking.

## Frame chunking (#685) is shape-dependent

`_FRAME_CHUNK_DIMS` removes read amplification but multiplies block count, and
which effect wins depends entirely on the source's shape. Twelve single-plane
reads, fresh process per row:

| source | chunking | blocks | block size | time | peak RSS |
| --- | --- | --- | --- | --- | --- |
| deepstack.tif (1 T, 320 Z) | default | 1 | 153.6 MB | 1935.7 ms | +313 MB |
| deepstack.tif (1 T, 320 Z) | **frame** | 320 | 0.48 MB | **425.9 ms** | **+20 MB** |
| organoids.tif (20 T, 140 Z) | default | 20 | 4.62 MB | **858.4 ms** | **+28 MB** |
| organoids.tif (20 T, 140 Z) | **frame** | 2800 | 0.03 MB | 2778.5 ms | +33 MB |

#685 measured the first shape, where frame chunking is 4.5x faster and 15x
leaner. On the second it is 3.2x *slower* with no memory win, because the default
block was already small and T multiplies the block count. Neither setting is
right for every source, which is the argument for getting off the dask array
rather than tuning its chunking.

## No generic fix captures it

Measured on in-scope arrays at 2 000 blocks:

| approach | `.tif` | `.lsm` | `.czi` |
| --- | --- | --- | --- |
| `compute()` (today) | 105.1 ms | 176.6 ms | 104.4 ms |
| `compute(optimize_graph=False)` | 133.4 ms | 200.5 ms | 89.9 ms |
| `arr.blocks[idx]` | 106.8 ms | 176.5 ms | 103.6 ms |
| `arr.to_delayed()` (once) | 598.8 ms | 610.1 ms | 604.1 ms |

`to_delayed` also detaches DV's `resource_backed_dask_array`, so reads then fail
with "Cannot read from closed file".

Calling bioio's own per-chunk function directly — bypassing the outer graph but
keeping plugin code — separates the two layers:

| | dask | bioio chunk fn | native |
| --- | --- | --- | --- |
| `.tif`, 2 000 blocks | 105.1 ms | 6.6 ms | 0.06 ms |
| `.lsm`, 2 000 blocks | 176.6 ms | 76.3 ms | 0.05 ms |

Both layers are O(blocks): the outer dask graph, and bioio's chunk function
itself, which reopens the file and re-enumerates its pages on every call. Only a
native reader against a held handle is O(1).

## Recommendation

0. **Remote sources stay on bioio.** `OmeTiffAdapter` and `TiffSequenceAdapter`
   both decline remote URLs today, so a remote `.tif` already falls through to
   the bioio `aics` adapter and its fsspec path -- `ome_tiff.py`'s persistent
   `aszarr` store is local-only by choice ("persistent local handle N/A"). A
   dedicated `.tif`/`.lsm` adapter can follow the same shape: claim local,
   decline remote. That contains the gap rather than closing it -- remote reads
   keep the O(blocks) cost measured above.
1. **Hold the file handle open across reads.** This is where O(1) comes from and
   it is worth more than the dask bypass on many-block sources. It interacts with
   `_handle_reaper`, and must respect #640's constraint that no view escapes
   `_io_lock`.
2. **`.tif` and `.lsm`** — worst measured, and both are tifffile-backed, so they
   can reuse the existing `_read_aszarr_plane` / persistent-store machinery in
   `ome_tiff.py` and `tiff.py` rather than needing a new bypass.
3. **`.lif`** (readlif) and **`.czi`** (pylibCZIrw) — real O(blocks) cost, each
   needs its own small direct path.
4. **`.dv`** — flat in block count; lowest priority.

## Fixtures

`.lsm` and `.lif` have no local data and no writer, so both are synthesized:

- **LSM** — a TIFF carrying a valid `CZ_LSMINFO` tag (34412). tifffile's
  `_series_lsm` requires full-resolution and reduced pages to *alternate*
  (it takes `pages[0::2]` and `pages[1::2]`), channels carried as
  samples-per-pixel, and `metadata=None` on write so tifffile's own "shaped"
  description tag does not win the series dispatch.
- **LIF** — the container is written by hand from what readlif's parser reads:
  magic `70 00 00 00`, a UTF-16 XML header, then memory blocks. `bioio-lif`
  additionally requires `LUTName` on each `ChannelDescription`. Plane order
  inside a block is `n = t*(C*Z) + z*C + c`, matching `get_frame()`.
- **CZI** via `pylibCZIrw.create_czi`, **DV** via `mrc.save`. `mrc.save` records
  no channel count, so a multi-channel DV fails bioio's coordinate validation —
  keep synthetic DV single-channel.

Both generators verify their output through the native reader *and* through
`BioImage` before use.
