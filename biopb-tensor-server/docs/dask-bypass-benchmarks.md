# Is bypassing dask worth it beyond ND2?

Follow-up to #640, which measured ND2 only and closed with "worth measuring per
adapter before generalizing". This is that pre-Phase-1 measurement, across every
format that then reached `adapters/bioio.py`. Phase 1 now serves local plain
`.tif`/`.tiff` and `.lsm` through native persistent tifffile adapters; their rows below are the historical
baseline.

**Answer: yes, and the reason is not the one #640 gives.** The dominant cost is
not per-byte and not a fixed per-read overhead. It scales with the number of
blocks in the *whole* dask array, so it is worst on long time series. A second,
independent cost is over-read: a request smaller than a block pays for the whole
block, and on `dev` a `.tif`/`.lsm`/`.lif` block is an entire Z stack.

## Scope

The measurements below predate Phase 1. Local plain TIFF and LSM now use
`TiffAdapter` and `LsmAdapter`, respectively; remote TIFF and all other formats
listed here still use BioIO where their dedicated native phase is not complete.
OME-TIFF (`OmeTiffAdapter`), TIFF sequences and MicroManager (`tiff.py`) are
pure tifffile on a persistent `aszarr` store and never touch dask.

| adapter | extension | bioio plugin | native reader |
| --- | --- | --- | --- |
| TiffAdapter | `.tif`/`.tiff` (plain, local) | — | tifffile |
| LsmAdapter | `.lsm` (local) | — | tifffile |
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

- `BioImage` is built **outside** the timed region, as the adapter holds it.
  A `_FRAME_CHUNK_DIMS` override was proposed in #685 / #798 and closed; every
  headline number here is `dev`'s behaviour (bioio's own `chunk_dims`), and the
  frame-chunked configuration is measured separately below.
- Best of 3, warm page cache — this isolates CPU/graph cost, which is what is
  under test. A cold-cache run would narrow every ratio.
- The native path always **copies**. A memmap view costs nothing and reads
  nothing; the first DV number measured here was inflated 3x by exactly that,
  which is the trap #640 flags about views escaping the lock.
- Ryzen 5 5600X, dask 2026.6.0, bioio 3.4.0.

`.tif` and `.nd2` are real files. `.lsm`, `.lif`, `.czi` and `.dv` are
synthesized — see "Fixtures" below.

## One plane, as `dev` reads it today

Cost to serve a single (T, C, Z) plane, averaged over distinct planes on both
paths. Comparable regardless of chunking: asking dask for one plane out of a
Z-stack block still materializes the block, and that over-read is the thing
being measured.

| format | adapter | blocks | block size | dask | native | |
| --- | --- | --- | --- | --- | --- | --- |
| `.lsm` | zeiss | 6 | 0.04 MB | 32.60 ms | 0.022 ms | **1473x** |
| `.tif` | aics | 20 | 4.62 MB | 57.01 ms | 0.093 ms | **610x** |
| `.lif` | leica | 4 | 0.04 MB | 1.87 ms | 0.062 ms | 30x |
| `.dv` | dv | 80 | 0.52 MB | 1.44 ms | 0.074 ms | 20x |
| `.czi` | zeiss | 40 | 2.10 MB | 3.87 ms | 1.999 ms | 2x |
| `.nd2` | nikon | 1 | 202.57 MB | 124.12 ms | 70.08 ms | 2x |

`.nd2` is low only because that file is a single frame (T=1, Z=1) and therefore
one block; the same format with 40 000 frames sits at the top. `.czi` is low
because every request here is a whole plane, the one shape with no over-read to
remove -- see the next section.

Native reads run at 1.5-1.9 GB/s against a 13-15 GB/s memcpy floor; the gap is
tifffile's per-page Python work, not I/O. They are real reads: `pages.cache` is
False and `asarray()` returns a fresh array each call. Re-reading one hot plane
instead of distinct planes would flatter the native side by roughly 2x on a
many-page file.

## Sub-plane requests over-read the whole block

Every measurement above asks for a whole plane. A viewer asking for a tile does
not. For `.czi` the block is a whole plane (`bioio_czi` sets
`chunk_shape = shape[-2:]`, always per-plane -- never a Z stack), so a tile
request materializes the plane and throws most of it away. For
`.tif`/`.lsm`/`.lif` on `dev` the block is a whole *Z stack*, so the same
request is worse still.

CZI, a 256x256 tile out of a 4096x4096 plane (33.6 MB block):

| requested | dask | native `roi=` | |
| --- | --- | --- | --- |
| 256x256 (0.13 MB) | 31.4 ms | 1.09 ms | **28.8x** |
| 512x512 (0.52 MB) | 32.1 ms | 1.22 ms | 26.3x |
| 1024x1024 (2.10 MB) | 31.8 ms | 1.62 ms | 19.6x |

TIFF, a crop out of a 341 MB single-plane `E14.tif`. The file is **striped, not
tiled**, and still wins: tifffile's `aszarr` store exposes strip-level chunks
(11 rows x full width), so the crop reads a few strips instead of the plane.

| requested | dask | `aszarr` | |
| --- | --- | --- | --- |
| 256x256 | 339.4 ms | 7.9 ms | **42.9x** |
| 1024x1024 | 335.8 ms | 40.8 ms | 8.2x |

pylibCZIrw also reads downsampled natively: `zoom=0.25` returns the 1024x1024
reduction of that plane in **5.74 ms** against 36.4 ms for the full-plane dask
read. That fuses read and reduction the way #640 asks for on the precache path,
without materializing the full-resolution intermediate.

Not every format can do this. readlif has no ROI API -- `_get_item` reads the
whole plane's bytes into a PIL image -- so `.lif` gains nothing here. `.dv`
needs nothing: it is a memmap, so a crop only touches the pages it covers.

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

This is also why chunking cannot be tuned out of the problem: shrinking blocks
to remove the Z-stack over-read raises the block count, and the graph cost rises
with it. See the next section.

## Frame chunking (#685 / #798) was evaluated and closed

Forcing `chunk_dims = ("Y","X","S")` removes the Z-stack over-read but
multiplies block count, and which effect wins depends entirely on the source's
shape. Twelve single-plane reads, fresh process per row:

| source | chunking | blocks | time | peak RSS |
| --- | --- | --- | --- | --- |
| deepstack.tif (1 T, 320 Z) | default | 1 | 1935.7 ms | +313 MB |
| deepstack.tif (1 T, 320 Z) | **frame** | 320 | **425.9 ms** | **+20 MB** |
| organoids.tif (20 T, 140 Z) | default | 20 | **858.4 ms** | **+28 MB** |
| organoids.tif (20 T, 140 Z) | **frame** | 2800 | 2778.5 ms | +33 MB |

#685 measured the first shape, where it is 4.5x faster and 15x leaner. On the
second it is 3.2x *slower* with no memory win, because the default block was
already small and T multiplies the block count. Per plane across the corpus:

| format | dev default | frame chunked |
| --- | --- | --- |
| `.tif` | 57.01 ms | 216.23 ms |
| `.lif` | 1.87 ms | 2.67 ms |
| `.lsm` | 32.60 ms | 32.53 ms |

Neither setting is right for every source, which is why #798 was closed in
favour of getting off the dask array entirely (#799) rather than tuning its
chunking.

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

0. **Local TIFF/LSM (phase 1) and CZI (phase 2) are complete; remote sources
   stay on BioIO.**
   `OmeTiffAdapter` and `TiffSequenceAdapter`
   both decline remote URLs today, so a remote `.tif` already falls through to
   the bioio `aics` adapter and its fsspec path -- `ome_tiff.py`'s persistent
   `aszarr` store is local-only by choice ("persistent local handle N/A"). The
   dedicated adapters claim local and decline remote. That contains the gap
   rather than closing it -- remote reads keep the O(blocks) cost measured above.
1. **Hold the file handle open across reads.** This is where O(1) comes from and
   it is worth more than the dask bypass on many-block sources. It interacts with
   `_handle_reaper`, and must respect #640's constraint that no view escapes
   `_io_lock`.
2. **`.lsm` and `.tif`** — worst measured (1473x and 610x per plane), and both
   are tifffile-backed, so one path serves both and can reuse the existing
   `_read_aszarr_plane` / persistent-store machinery in `ome_tiff.py` and
   `tiff.py` rather than needing new read code. `.lsm` carries one decision the
   others do not: tifffile builds a second, reduced series from the interleaved
   thumbnail pages and bioio exposes it as a scene. It is **not** a pyramid
   level — `CZ_LSMINFO` records `ThumbnailX`/`ThumbnailY` as absolute pixel
   counts, so its scale varies per file. `LsmAdapter` deliberately exposes only the
   full-resolution series and drops the thumbnail; computed pyramid levels remain the
   server's responsibility.
3. **`.czi`** — only 2x on whole planes, but 20-29x on tiles plus a native
   `zoom=` that subsumes the fused read+reduce #640 asks for. The case rests on
   sub-plane and pyramid access, not on plane reads. **Delivered** by
   `CziAdapter` (`adapters/czi.py`), measured end to end through `get_data`
   against the same file read through BioIO:

   | blocks | plane read | 256x256 tile |
   | --- | --- | --- |
   | 40 (2C x 20Z, 512²) | 3.30 -> 0.161 ms (20.5x) | 3.25 -> 0.154 ms (21.1x) |
   | 1 000 (2C x 500Z, 512²) | 78.77 -> 0.172 ms (459x) | 79.32 -> 0.142 ms (560x) |
   | 4 (1C x 4Z, 2048²) | 3.65 -> 2.114 ms (1.7x) | 3.75 -> 0.887 ms (4.2x) |

   The block-count row is the O(blocks) graph cost; the 4-block row isolates the
   over-read, where only the tile column moves. `zoom=` is **not** wired up: the
   server downsamples a scaled chunk with `downsample_block`, whose reduction
   methods (`area`/`max`/...) libCZI's own scaling accessor does not reproduce,
   so routing scaled reads through it would silently change pixel values. That
   is a read-planner change, not an adapter one.
4. **`.lif`** — O(blocks) overhead only; readlif offers no ROI, so there is no
   over-read win on top.
5. **`.dv`** — flat in block count and a memmap underneath; lowest priority.

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
- **CZI** via `pylibCZIrw.create_czi` (`create_zeiss_czi`, plus
  `create_zeiss_czi_scenes` for a document whose scenes sit at different plane
  offsets — the layout that separates absolute from scene-relative ROI
  coordinates), **DV** via `mrc.save`. `mrc.save` records
  no channel count, so a multi-channel DV fails bioio's coordinate validation —
  keep synthetic DV single-channel.

Both generators verify their output through the native reader *and* through
`BioImage` before use.
