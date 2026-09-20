# Java tensor Flight v2 migration

This is the implementation checklist for bringing the Java tensor SDK to the
same Flight v2 contract as Python. Remote cache-file transfer and persistent
disk caching are deliberately out of scope until the core protocol is stable.

- [x] Central typed Flight-error decoding and public Java exceptions.
- [x] Route core Java Flight calls and action iterators through the decoder;
  migrate remaining worker-side paths with their serialization redesign.
- [x] Introduce `FlightSession` for allocator, authentication, client ownership
  and common error-mapped calls; add TLS trust in its dedicated slice.
- [x] Replace descriptor caching and legacy read planning with an `array_id`
  first, FieldMask-driven v2 tensor planner.
- [x] Rebuild the lazy Imglib2 read adapter from immutable `FlightInfo` plans.
- [x] Make `SerializedTensor` the only cross-process tensor-handle format.
- [x] Implement catalog/metadata/resolve/warm parity.
- [x] Implement source lifecycle, upload and ROI parity.
- [x] Remove the legacy `(sourceId, tensorId)` entry points and the
  `DataSourceDescriptor` row decoders they fed.
- [x] Own Flight connections after the pool's removal: `FlightSessions` shares
  one per `(location, token)`, as `biopb.tensor._pool` does.
- [ ] Port the connect-time protocol check (`health`'s `protocol`, Python's
  `_check_protocol_version` / `_check_wire_protocol`). Without it a v1 server
  is reported as "returned no terminal result" rather than as too old.
- [ ] Consider remote cache-file/mmap and persistent caching separately, after
  protocol parity is covered by shared integration fixtures.

## What "parity" covers

| Python | Java |
| --- | --- |
| `add_source` / `remove_source` | `addSource` / `removeSource` |
| `label_sets` / `delete_labels` | `labelSets` / `deleteLabels` |
| `list_rois` / `put_rois` / `delete_rois` / `prune_rois` | `listRois` / `putRois` / `deleteRois` / `pruneRois` |
| `create_tensor` / `upload_array` / `upload_chunk` / `finish_upload` | `createTensor` / `uploadArray` / `uploadChunk` / `finishUpload` |

Cancellation transfers, by a different route. Python breaks out of the
`do_action` generator and pyarrow closes the stream; Arrow Java hands back a
bare `Iterator<Result>` with no close on it, but the call is still a gRPC call
-- created inside a `Context.CancellableContext`, it is cancelled with the
scope, and the server sees it. `streamAction` does that on every exit path.
Verified against a live server: a cancelled `addSource` over a directory of
eight images stops it at four, where abandoning the iterator alone let it
register all eight.

Two things do not transfer, and are not gaps:

- **No dask.** Python's `upload_array` hands the whole upload to `da.store`, so
  blocks ship from whichever worker computed them. `TensorUploads.uploadArray`
  walks the descriptor's chunk grid on the calling thread. The grid, the bounds
  and the empty-chunk skip for a label set are the same.
- **Connections are shared, not pooled per thread.** Python keys its pool by
  `(location, token)` *and* by thread, because a `FlightClient` is not
  pickle-safe and each dask worker dials its own. Java has no such constraint:
  one `FlightSession` per `(location, token)` for the whole process, held until
  exit. `TensorFlightClient` is unaffected -- it owns its session and closes it;
  the cache is for `SerializableTensorImg`, which hands its session to an
  imglib2 cell cache that outlives every call it can see.
- **No `RoiAnnotation` wire codec to share.** `biopb.image._roi_rows` is one
  module serving both the Python SDK and the server; Java has only a client, so
  `RoiRowCodec` is a second implementation of that row schema and has to track
  it column for column. Its Arrow schema compares equal to pyarrow's, which is
  what keeps the two honest.
