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
  one per `(location, token)`. The token belongs in the key as it does in both
  of `biopb.tensor._pool`'s pools; the third component Python's *connection*
  key has, `TlsTrust.key_id`, has to arrive with TLS (biopb/biopb#1072).
- [x] Port the two compatibility gates: the `health` action's `protocol`
  (checked once per connection) and the read plan's `chunk_wire_protocol`
  (checked where a plan becomes an image).
- [ ] Consider remote cache-file/mmap and persistent caching separately, after
  protocol parity is covered by shared integration fixtures.

## What "parity" covers

| Python | Java |
| --- | --- |
| `add_source` / `remove_source` | `addSource` / `removeSource` |
| `label_sets` | `labelSets` |
| `list_rois` / `put_rois` / `delete_rois` / `prune_rois` | `listRois` / `putRois` / `deleteRois` / `pruneRois` |
| `add_tensor` / `upload_array` / `upload_chunk` / `set_upload_status` | `addTensor` / `uploadArray` / `uploadChunk` / `setUploadStatus` |

### The two version gates

A server advertises exactly two version signals, and both are contracts:

| signal | where | meaning |
| --- | --- | --- |
| `protocol` | `health` action | the protocol *shape* -- which descriptors, tickets and put commands the server understands. v1 routed by a sentinel `source_id`; v2 is the oneofs this client sends; v3 replaced `finish` / `delete_labels` with one `set_upload_status`, and made an upload add a tensor to a source that already exists (`add_tensor`, `PutCommand.chunk_ticket`) rather than create one -- `register_source` went with the change, since a server serves its own `scratch` source to add to. |
| `chunk_wire_protocol` | schema metadata on every read plan | the chunk *encoding*. v1 was `data: list<T>`; v2 is one binary blob plus a numpy dtype string (biopb/biopb#293), which is what `ChunkDecoder` reads. |

Known gaps, each with an issue rather than a note here: uploads are sequential
where Python's are concurrent (biopb/biopb#1073), and a `grpc+tls://` location
gets the JDK default trust store with no way to configure an anchor, a
fingerprint or a hostname override (biopb/biopb#1072 -- which must also extend
the connection-cache key, see `FlightSessions`).

Known gap: the chunk value path decodes every dtype to `double` and scatters it
with `setReal`, so `i8`/`u8` tensors lose ids above 2^53 even though
`createType` gives them the right imglib2 type -- biopb/biopb#1071. The upload
side is already exact.

Those two are the whole set. A read plan's schema carries no other version key,
and `chunk_locate` advertises no segment format -- the server verifies a byte
range before handing it out instead (biopb/biopb#1070).

`FlightSession` probes the first on first use, so building a client stays free
of I/O and a v1 server is named rather than sent a request it will parse as
something else. `Imglib2TensorFactory` checks the second where a plan becomes
an image -- the same place Python checks it, and the one point both a
locally-planned read and a `SerializedTensor` from another process pass
through. An absent stamp means v1 in both cases: the key postdates that
version.

Both gates fail closed. A server that does not state a protocol is refused
however it declines to -- no reply, an empty body, a body that is not JSON, or
JSON without the key are one fact, and the gate is worth nothing if the
quietest server walks through it. The single exemption is an
`UNAUTHENTICATED`/`UNAUTHORIZED` `health` (which a biopb server no longer
answers, since `health` is ungated -- but a client must not depend on that): a
capability token cannot
reach the catalog tier `health` sits on, and the call it is about to make
authorizes itself.

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
