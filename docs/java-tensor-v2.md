# Java tensor Flight v2 migration

This is the implementation checklist for bringing the Java tensor SDK to the
same Flight v2 contract as Python. Remote cache-file transfer and persistent
disk caching are deliberately out of scope until the core protocol is stable.

- [x] Central typed Flight-error decoding and public Java exceptions.
- [x] Route core Java Flight calls and action iterators through the decoder;
  migrate remaining worker-side paths with their serialization redesign.
- [x] Introduce `FlightSession` for allocator, authentication, client ownership
  and common error-mapped calls; add TLS trust in its dedicated slice.
- [ ] Replace descriptor caching and legacy read planning with an `array_id`
  first, FieldMask-driven v2 tensor planner.
- [ ] Rebuild the lazy Imglib2 read adapter from immutable `FlightInfo` plans.
- [ ] Make `SerializedTensor` the only cross-process tensor-handle format.
- [ ] Implement catalog/metadata/resolve/warm parity.
- [ ] Implement source lifecycle, upload and ROI parity.
- [ ] Deprecate legacy `(sourceId, tensorId)` entry points and update examples.
- [ ] Consider remote cache-file/mmap and persistent caching separately, after
  protocol parity is covered by shared integration fixtures.
