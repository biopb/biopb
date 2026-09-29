# Scaled read path: remaining work

What is left of the #640 plan. The shipped path is described in
`fused-scaling-path.md`. Every remaining item is a speed win for `area`: memory
is already bounded for every adapter by the streamed default.

## Open items

| item | what | status |
| --- | --- | --- |
| Row bands for contiguous readers | For adapters with `read_block_shape is None`, stream full-width row bands instead of transfer-grid tiles. A tile out of a wide frame reads strided (an 1182-wide tile of a 14234-wide ND2 frame is 1182 short memcpys). Band height is a whole multiple of the row scale, about 8 MiB (the L3 knee on this box: 2-8 MiB wins, past ~16 MiB the band stops fitting). | Measured 1852 -> 592 ms warm on ND2 at area/8. Not started. |
| ND2 fused `area` | Fold row bands inside the frame lock rather than copying tiles out. | Measured 331 -> 112 ms on a 128 MiB extent at scale 4, before streaming; re-measure against the streamed default first. Mostly subsumed by row bands. |
| CZI banded ROI `area` | Band-wise libCZI ROI reads folded as they arrive. | Measured 387 -> 150 ms at scale 4 before streaming; re-measure first. |
| CZI native pyramid | Advertise stored subblock levels via `get_native_pyramid_levels()` + `precompute`, as OME-Zarr and QPTIFF do. Strictly better than `zoom=` where the file has them. | Blocked on a pyramidal CZI fixture (#799). |
| Edge-pad copy | `_pad_array_edge` copies the whole padded array whenever the extent is not a multiple of the scale. Handling the margin separately needs its own bit-identity argument. | Not started. |
| Cold strided `nearest` | A decimated read loses ~1.3x cold where the row gap is inside the readahead window. `madvise(MADV_RANDOM)` on the strided mapping would make the skip real. | Only if the loss shows up in practice. |

## Constraints for the row-band work

- Bands along rows, never square retiles, and bands end on block boundaries so
  the result stays bit-identical. Only the last band may be ragged.
- Only where a band read costs in proportion to its size: mmap sources (ND2,
  MRC, NIfTI, DeltaVision). `read_block_shape is None` is the selector; do not
  add a per-class flag, since a flag that defaults on is a silent opt-in.
- On a network filesystem each extra band adds up to one RTT when readahead
  misses a band boundary. Estimated break-even around 2 ms RTT at scale 4; at
  coarse scales banding saves no time, only memory, which streaming already
  bounds. Consult `biopb._fs_detect.network_filesystem_type` before enabling it
  for a network-backed source.

## Still unmeasured

- Warm-precache peak memory as a test: should be bounded by one streaming unit.
- `benchmarks/bench_scaled_read.py`: real chunk_ids at scales 2/4/8/32 for both
  methods, time and peak RSS (not `tracemalloc`, which misses C++ and mmap
  pages), with `np.array_equal` asserted, printing the kernel crossover per
  machine.
- DICOM, NDTiff and the BioIO formats: their `read_block_shape` declarations
  are reasoned from their read paths, not benchmarked. MRC, NIfTI and
  DeltaVision decimation has no real file in the test catalog.
