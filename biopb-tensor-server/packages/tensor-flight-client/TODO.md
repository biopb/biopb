# tensor-flight-client TODO

## Move hysteresis out of the library into web/

`computeScaleHint` currently accepts `prevFactors` and applies a 20% hysteresis band
to avoid scale-level flickering near power-of-two boundaries. This is a render-loop
stability concern, not a tensor/data concern:

- Non-web consumers (e.g. server-side thumbnail generators) have no need for it and
  must pass `undefined` to opt out.
- The library cannot know when a zoom gesture has ended — the ideal time to commit to
  a new scale level — so the fixed 20% band is a blunt proxy for gesture state that
  only the web layer can reason about properly.

Proposed split:
- Remove `prevFactors` parameter and hysteresis from `computeScaleHint`. The function
  returns the optimal power-of-two factor for the given viewport/data size.
- `web/` adds a `useScaleHint()` hook (or a small stateful wrapper) that retains
  `prevFactors` across frames and applies hysteresis before calling `compute()`.
- `ScaleVector.snapped` can be removed from the public API along with the `HYSTERESIS`
  constant once they are no longer needed by library callers.

Low priority — current placement is functional and tested. Revisit if the library gains
non-web consumers.

---

## Client-side tile cache

### Motivation

Currently every `TensorArray.compute()` call is a fresh HTTP POST to `/api/slice`.
Although the server has a chunk cache that avoids re-reading from storage, each call
still pays:
- ~2–8 ms fixed FastAPI/ASGI dispatch cost per request (even on localhost loopback)
- A fresh `ArrayBuffer` allocation + copy for every response regardless of hit/miss

For a tiled microscopy viewer that re-requests the same tiles across pan/zoom gestures,
this means repeated network round-trips for data the JS process already holds in memory.

### Design

A `TileCache` class in a new `src/tile-cache.ts` file, used inside `TensorArray.compute()`:

**Key**: stable serialisation of `SliceRequest` (JSON.stringify is deterministic for plain
objects with fixed key order; or use a small hash over the binary fields).

**Eviction policy**: byte-budget LRU
- Track total held bytes: `buffer.byteLength` per entry.
- On every write, if `totalBytes > budgetBytes`, evict the LRU entry (Map insertion order
  is the cheapest LRU approximation in JS).
- Eviction triggered on every write, not lazily — a single large slice can blow the budget
  immediately.
- Suggested default budget: `256 * 1024 * 1024` (256 MB), configurable via
  `TensorFlightClient` constructor option.

**Do NOT use `WeakRef` for the primary cache** — `ArrayBuffer` backing a typed array used
in a Pixi.js texture will be kept alive by the renderer, so weak references would not help
reclaim memory. Use explicit eviction instead.

**Integration point**: `TensorArray.compute()` checks the cache before issuing the HTTP
request. On a miss, the response is stored before returning. Cache instance should be
shared across all `TensorArray` instances for the same `TensorFlightClient` (pass it from
the parent client).

### Concurrency

Add in-flight deduplication: if two `compute()` calls for the same key are in-flight
simultaneously (same key, not yet resolved), the second should await the same promise
rather than issuing a second HTTP request. Use a `Map<string, Promise<TypedNdArray>>` of
pending fetches alongside the result cache.

### Interface sketch

```typescript
interface TileCacheOptions {
  /** Maximum total bytes held in cache. Default: 256 MB. */
  budgetBytes?: number;
}

class TileCache {
  constructor(options?: TileCacheOptions);
  get(key: string): TypedNdArray | undefined;
  set(key: string, value: TypedNdArray): void;
  /** Number of bytes currently held. */
  get bytesUsed(): number;
  clear(): void;
}
```

### Notes

- `SliceRequest` key should normalise `undefined` optional fields to avoid duplicates
  (e.g. `scale_hint: undefined` vs omitted key).
- Consider exposing cache stats (hit rate, bytes used) via a `cacheStats()` method on
  `TensorFlightClient` to aid debugging — mirrors what the server exposes in `/api/diagnostics`.
- The server-side cache and client-side cache are complementary: server avoids re-reading
  storage; client avoids re-sending bytes over the network and re-allocating ArrayBuffers.
