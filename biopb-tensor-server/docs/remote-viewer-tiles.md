# Remote data viewer — client-side rendering over a tile API

**Status:** partially implemented. The tile API (`GET /api/tile_info`, `GET /api/tile`),
server-side cancellation, client `AbortSignal` plumbing and the Viv `PixelSource`
adapter have landed. **Viv is now the committed framework** (0.22.1, deck.gl/luma.gl
`~9.3.3`). The viewer component itself has not been built.
**Component:** `biopb-tensor-server` (HTTP sidecar, tile route); `web/` (viewer SPA);
`biopb-control` (proxy hop — see `biopb/biopb#762`).
**Related:** `http-server.md`, `remote-tensor-cache.md`, `../../docs/url-prefix.md`.

## Why

The viewer SPA renders server-side. `ws/render` takes a region plus a `scale_hint`,
composites it (percentile contrast, colormap) with PIL/VTK, and pushes a PNG/JPEG
over the socket; the client pans and zooms locally with a CSS transform against the
returned `loaded_region` rect until `shouldReload()` decides the viewport drifted,
then debounces a new render.

That is the right shape for a **local** server and the wrong shape for a remote one:

- **One monolithic image per reload.** A pan moving 10% of the viewport refetches
  100% of the pixels. Nothing is reused across reloads.
- **Nothing is cacheable.** Opaque blobs over a WebSocket are invisible to the
  browser cache, a Service Worker, and any CDN. The transport structurally cannot
  cache.
- **Appearance is baked server-side.** Changing `percentile_lo/hi` re-renders and
  re-transfers the whole region, so a contrast slider costs a round trip per frame.
- **Cancellation is client-side only.** `renderGenerationRef` discards stale blobs
  (`web/packages/app/src/hooks/useRenderWebSocket.ts`), but the server has already
  done the work and the bytes have already crossed the wire.
- **Reload is a cliff.** Between `shouldReload` firing and the response landing
  there are no pixels for the new region.

## Deployment topology — decided

Three topologies were considered:

1. Local server always present; browser↔server is loopback.
2. **Bare remote server, no local proxy.** ← target
3. Remote server plus a local caching proxy near the browser (today's
   `remote-tensor-cache.md` design, deployed differently).

**The SPA must work in topology 2.** It cannot assume an installed local server —
that requirement is exactly what a browser SPA trades on avoiding. A tile-over-HTTP
interface is agnostic to where the server sits, so building for (2) makes (1) and
(3) fall out as the fast path.

The consequence that drives the rest of this document: **the browser cannot mmap.**
The Python client takes whole chunks zero-copy through the `chunk_locate` fast path;
the browser always copies over a socket. These are two client classes with genuinely
different optimal request sizes, served by one arbitrary-rect interface.

## Architecture — client-side rendering from raw tiles

Move the render boundary: the server ships raw pixel data, the browser uploads it to
WebGL textures and applies contrast, colormap, and multi-channel blending in a
fragment shader. Contrast and channel toggles become zero-network; one fetch serves
every rendering setting for the rest of the session.

The load-bearing property is that **repaint must never require a round trip.** Any
design where a camera change implies network traffic is latency-bound at RTT and
cannot be fixed by compression. Today's debounce-plus-CSS-transform is a partial
version of this; tiling completes it.

### Bandwidth budget

Traffic is bounded by **screen size, not image size** — only a viewport's worth of
pixels at the current pyramid level is ever fetched. This is why the approach works
against an arbitrarily large dataset.

| | raw `uint16` |
|---|---|
| 1920×1080 viewport | 4.1 MB / channel |
| + prefetch ring (~1.5×) | ~6 MB / channel |
| 4 channels, fresh viewport | ~25 MB |

At 50 Mbps (6.25 MB/s): ~1 s for one channel, ~4 s for four. Single- and dual-channel
browsing is comfortable; **four-plus channels at full resolution is where raw
transport gets uncomfortable.**

The same viewport as server-rendered JPEG is ~300–500 KB — 10–40× less. That gap is
real and is widest in topology 2, which is why the tile route keeps a rendered mode
(below) rather than committing to raw only.

## Rendering framework — Viv (committed)

[Viv](https://github.com/hms-dbmi/viv) (`@hms-dbmi/viv`), deck.gl layers for
bioimaging. The decisive argument is that its data seam already matches ours. A
multiscale image is a `PixelSource[]`, one per pyramid level:

```ts
interface PixelSource {
  shape: number[]; labels: string[]; dtype: string; tileSize: number;
  getTile({ x, y, selection, signal }): Promise<{ data: TypedArray, width, height }>
  getRaster({ selection, signal }): Promise<{ data: TypedArray, width, height }>
}
```

`/api/slice` already returns `TypedNdArray` — `{ buffer, shape, dtype, dimLabels }`,
raw C-contiguous numpy bytes — which is the same thing modulo a reshape, and
`tensor-array.ts` already resolves t/z/c/y/x via `buildAxisMap`. **This is an adapter,
not a data layer:** no zarr, no OME-NGFF, no wire-protocol rewrite.

What it provides that is otherwise expensive to build:

- **Native `uint16`/`float32` rendering.** Viv requires WebGL2 specifically for
  integer textures (`R16UI`); 16-bit data reaches the GPU unconverted and contrast
  limits are applied in the shader.
- **N-channel additive blending** with per-channel LUT and contrast limits, all
  shader-side. The current single-`color`/`channel_name` round trip becomes a uniform
  update.
- **Tile scheduling and progressive refinement** via deck.gl `TileLayer`
  (`refinementStrategy: 'best-available'` draws cached parent/child tiles scaled
  while the correct level loads).
- **`AbortSignal` in the interface** — cancellation is designed in.
- **A 3D path** (ray-cast volume rendering) that matches our 3D spec exactly (below).

### Costs, honestly

- **Bundle size.** deck.gl + luma.gl will be the largest dependency by a wide margin
  in an app that is currently React + zustand + a router. Lazy-load the viewer route.
- **Version pinning.** Resolved: **Viv 0.22.1** declares `@deck.gl/*` and `@luma.gl/*`
  peers at **`~9.3.3`** — a tilde, so it will not follow a deck.gl minor. Pin all three
  together and re-check the peer range on any Viv bump.
- **Maintenance cadence.** Academic-lab project (HMS DBMI). Widely deployed and it
  works, but expect to vendor a patch occasionally rather than wait on a release.
- **Channel cap.** `MAX_CHANNELS = 10` in `@vivjs/constants` — a hard wall for highly
  multiplexed data, comfortable for ordinary microscopy.
- **`@vivjs/loaders` is not used.** It pulls geotiff, zarrita and zod to read formats
  we do not serve. The adapter depends on `@vivjs/types` only, which is types plus
  math.gl.

### Alternatives

- **deck.gl directly** (`TileLayer` + a custom layer). Not a competing choice — the
  eject hatch. Same ecosystem, so Viv-specific shader limits can be escaped without
  touching the app shell.
- **regl / hand-rolled WebGL2.** ~600–1000 LOC for tile cache, quad renderer, shader.
  Viable if bundle size dominates and 3D is dropped; reimplements the fiddly 16-bit
  texture handling.
- **Rejected:** OpenLayers / OpenSeadragon / Leaflet (8-bit RGB tools — correct for
  server-rendered tiles, wrong here); three.js (2D overkill); vtk.js (heavy, own
  idioms).

## Tile endpoint

A **GET** route, everything in the URL, so the browser cache works:

```
GET /data_plane/api/tile/{source}/{tensor}/{level}/{col}_{row}?fmt=raw&t=0&z=42&c=1
GET /data_plane/api/tile/{source}/{tensor}/{level}/{col}_{row}?fmt=jpeg&t=0&z=42&c=1&lo=1&hi=99&color=green
```

`POST /api/slice` cannot be cached by any browser under any header. It stays for the
3D slab and large-body requests; the tile read path needs its own GET.

**Headers.** `Cache-Control: private, max-age=3600` plus `Vary: Authorization` and an
`ETag`. Staleness is handled by versioning the `array_id` namespace, **not** by putting
a version in the cache key (same conclusion as the compact-grid work: `chunk_id` stays
an opaque server-side token) — so `immutable` and a long `max-age` wait on that
versioning, since re-indexing currently reuses the id.

**Auth stays out of the URL.** `ws/render` passes `?token=`; do not carry that over.
A token in a tile URL means one cache entry per token and tokens in access logs. Use
the `Authorization` header — the client is on `fetch` anyway, and the browser caches
by URL, so tokens can rotate without invalidating anything.

**Which is exactly why the cache must be `private`.** These two decisions interact,
and the first draft of this document got it wrong by recommending `public`. RFC 9111
§3.5 permits a **shared** cache to reuse a response to a request bearing
`Authorization` for a *different* request when the response carries `public`,
`s-maxage`, or `must-revalidate`. Since auth is a header, the cache key — the URL —
contains no token at all, so that "different request" can be an unauthenticated
stranger's. An nginx `proxy_cache`, CDN, or corporate proxy in front of a `--remote`
deployment would then serve tiles having checked the token exactly once, for someone
else. Local mode is unaffected (nothing shared is in the loopback path), but the
directive is an instruction to intermediaries we do not control, so it must not be
sent.

`private` keeps everything the tiled design actually needs: a per-user browser cache
across pan and zoom, and cheap `ETag` revalidation on reload. It gives up only CDN
caching, which bearer auth cannot make safe in any case — that would need the grant in
the cache key, i.e. signed URLs, not a header change.

**`fmt` is the escape hatch, and it must exist from day one.** A `PixelSource`
returning server-rendered 8-bit RGB tiles is still a valid `PixelSource` — Viv renders
interleaved RGB as well as raw integer channels. So server-side rendering becomes a
*transport mode* of this architecture rather than a competing design, selectable at
runtime: raw when channel count is low or contrast is being adjusted, rendered when
the link is slow or eight channels are loaded. Ship `fmt=raw` first; without the
parameter designed in, this ends up as two viewers.

**The server advertises the tile size** (below), rather than the client hardcoding it.

**Compression: measure before building.** `Content-Encoding` at the nginx edge is
free but weak on 16-bit image data (~1.2–1.5×). Byte-shuffle + zstd is the real win
on scientific integer data (often 2–4×) at the cost of a ~100 KB wasm decoder — do
not add it speculatively.

## Cancellation

A client-side `AbortController` stops the browser from *looking at* a tile it no
longer wants; it does not stop the server from producing it. The read handlers
therefore poll `request.is_disconnected()` before starting a read and answer 499
("client closed request", nginx's code) when the caller has already gone, counted
as `cancelled_reads` on `/api/diagnostics`.

**This only reclaims queued work, never in-flight work.** Neither the Flight read
nor the dask graph is interruptible, so a client that leaves *during* a compute is
not noticed. That is the right target anyway: the tile burst is exactly the case
where 20–60 requests queue and the viewport moves on before most of them run.

### The coupling that makes it work at all

Measured, not assumed: with the read running **on the event loop**, the check never
fires. A burst of 60 abandoned tiles produced **0** cancellations. The blocking
`.compute()` starves the loop of the turn it needs to process the socket closes, so
every queued handler checks `is_disconnected()` just before its own disconnect
callback is delivered, and sees `False` — systematically, not occasionally.

Moving the read to `run_in_threadpool` fixed it: the same burst now skips **14 of
60** (the rest were already running — with a 40-thread pool, most start
immediately). So the disconnect check and getting the read off the loop are one
change, not two; shipping the check alone would have been dead code that looked
correct in unit tests.

`is_disconnected()` itself is not the problem — a single poll detects a departed
client reliably once the loop is free to see the close.

`/api/slice` and `/api/render` carry the same pre-read check but still compute on
the loop, so their cancellation only fires when something else has yielded. They
benefit indirectly (tile reads now free the loop) but should move to a threadpool
before their check can be relied on.

## Chunk size vs tile size

One parameter is currently doing two jobs:

- **cache chunk** — sized for remote fetch amortization and mmap locality. Wants to
  be large.
- **transport tile** — sized for time-to-first-pixel, cancellation granularity, and
  incremental paint. Wants to be small.

These conflict only if they are the same number, and they need not be: `/api/slice`
takes an arbitrary rect plus `scale_hint`, so the transport unit is already the
*request*, not the chunk.

**The segment cache is mmap-backed and uncompressed** (segment writers/mmaps, the
localhost `chunk_locate` fast path). Carving a 256×256 tile out of a large cached
chunk is therefore a strided copy off a mapped page — not a decompress-the-whole-chunk
operation. That is the cost that normally kills this design in zarr/blosc-style
stores, and we do not pay it.

**Decision: keep one cache at large chunks and slice out of it.** A second
browser-shaped cache would double storage and introduce coherence problems to save a
memcpy we can already afford.

### The nesting rule

A tile must never **straddle** a chunk boundary. Equality is one degenerate case;
nesting is the general one:

```
tile_xy = chunk_xy / 2^k,  chosen so the tile lands near 256–512 px
```

One chunk then backs 16–64 tiles: the first tile touching it pays the fetch, the rest
are hits. Derive this server-side from `chunk_shape` (already on the descriptor,
`web/packages/tensor-flight-client/src/types.ts`) so the client need not know the rule.

Locally, straddling costs only extra page touches. The real cost is **remote fetch
amplification** — a straddling tile forces two cold chunk pulls from an upstream
instead of one — which is a latency problem only when the server is itself proxying
a remote source.

## Cache hierarchy

Four layers; Viv contributes none of them directly.

| | Cache | Owner | Unit | Survives |
|---|---|---|---|---|
| L1 | `Tileset2D` LRU | deck.gl `TileLayer` | decoded TypedArray + GPU texture | pan/zoom only |
| L2 | HTTP cache | browser | response bytes | reload, selection change |
| L3 | Service Worker / Cache API | us (deferred) | response bytes | reload, offline |
| L4 | segment cache | tensor server | chunks | everything, server-side |

**L1 must be bounded by bytes, not tile count.** The count-based default was designed
for 8-bit RGB map tiles; a 512×512 `uint16` tile is 512 KB per channel and the cache
holds both the array and a live GPU texture. Set `maxCacheByteSize` from a measured
VRAM budget.

**L1 is volatile across selection changes.** Because the tile fetch closes over the
channel/Z/T `selections`, changing them rebuilds the tileset rather than adding to it
— so scrubbing Z re-fetches everything. *Verify this against the pinned deck.gl
version before relying on the conclusion*; if it holds, it means L2 carries far more
weight than it appears to, and is the reason the GET/immutable shape is a
prerequisite rather than an optimization.

**L4 requires the nesting rule** to be hit at all.

## 3D volume path

**Spec (relaxed, agreed):** no pyramid in 3D. The volume is requested at a scale that
fits entirely in memory, uploaded as one 3D texture, and ray-cast.

This is precisely Viv's 3D model — it loads one resolution level into a single WebGL2
3D texture and does not stream out-of-core. (Multi-resolution volume streaming is
Neuroglancer's territory and a substantially harder system.) Our spec and the
framework's capability coincide; that is the main reason 3D does not force a second
architecture.

**Fetch the volume in one request.** Viv's generic loaders were built against
zarr/OME-TIFF, which cannot express a slab, so the default 3D path fetches per-Z-plane
and stacks — 200 round trips for a 200-plane volume. `/api/slice` already returns an
arbitrary N-d slab as C-contiguous bytes, which is exactly the layout `texImage3D`
wants. Assemble the volume in the adapter and hand Viv a finished buffer; do not
adopt the per-plane loop.

**Two ceilings, and the tighter one is not the obvious one.**

*GPU:* WebGL2 `MAX_3D_TEXTURE_SIZE` is commonly 2048, sometimes 1024 on integrated
hardware — a single dimension above it fails outright, not gracefully. Per channel:
256³ `uint16` = 32 MB, 512³ = 256 MB, 1024³ = 2 GB (not viable).

*Network:* 512³ `uint16` is 256 MB on the wire — ~40 s per channel at 50 Mbps. **The
remote-viable budget is ~256³ per channel**, well under what the GPU would tolerate.

So the server picks the level, not the client's GPU. Extend the existing
`pixel_budget`/`scale_hint` mechanism as a `voxel_budget` on the slice path: the
client asks for an ROI under a voxel count, the server resolves the pyramid level and
returns the slab. Same concept, same code shape.

**Consequences of no pyramid:**

- No progressive refinement — the user waits, then the whole volume appears. Fix falls
  out of the budget mechanism: request a coarse volume first (128³ ≈ 2 MB, sub-second),
  display it, then fetch the target level and swap the texture.
- No LOD during interaction — rotating a large volume is GPU-bound and will crawl on
  integrated graphics. Standard mitigation is to cut ray-sampling steps while the
  camera moves and restore on idle.
- No L1/L2 caching. Hold the volume explicitly, keyed on
  `(tensor, level, t, channel)`, bounded to 2–3 entries — these are large.

## Measured: the control proxy hop, and the tile-size floor

In topology 2 every tile crosses **browser → nginx → control (starlette + httpx) →
tensor server**. The control is a same-origin reverse proxy for `/data_plane/*`
(`biopb-control/src/biopb_control/_control.py`), which is good for CORS and is now on
a hot path it was not designed for: ~20–60 small responses per interaction instead of
one large one.

Measured on the dev box (12 cores, control :8813, tensor sidecar :8814, no auth, no
TLS, loopback). Same endpoint hit both ways, so the delta is the proxy alone:

```
direct  -> :8814    8711 req/s    tensor  = 100% of 1 core     (~115 µs CPU/req)
proxied -> :8813     596 req/s    control = 100% of 1 core   (~1677 µs CPU/req)
```

```
### livez (tiny GET)          ### slice 512KB tile (512x512 uint16, POST)
 conc            req/s          conc            req/s      MB/s
    1  direct   3843.5             1  direct    292.1     153.1
    1 proxied   1013.1             1 proxied    211.9     111.1
    8  direct   7934.5             8  direct    353.4     185.3
    8 proxied    798.1             8 proxied    346.1     181.5
   32  direct   8531.3            32  direct    359.3     188.4
   32 proxied    718.4            32 proxied    331.4     173.8
   64  direct   8567.5
   64 proxied    656.5     (p95 249 ms, p99 342 ms)
```

The control burns **~1.68 ms of CPU per proxied request — ~14× what the tensor server
spends serving it**, on a single-core-bound event loop. Proxied small-request
throughput *falls* as concurrency rises (1013 → 798 → 718 → 656): a saturated loop
with a growing queue. For reference, nginx proxies at ~10–20 µs/req.

Fitting both cases: **CPU_ms ≈ 1.7 + 0.0026 × KB.**

### The consequence for tile size

Cost is **per-request, not per-byte**, so small tiles are punished. Sustained
aggregate capacity through the control:

| tile payload | req/s | MB/s | concurrent users @ 50 Mbps |
|---|---|---|---|
| 64 KB | 535 | 34 | ~5 |
| 256 KB | 422 | 108 | ~17 |
| 512 KB | 330 | 169 | ~27 |
| 1 MB | 229 | 235 | ~38 |

This cuts against the instinct to shrink tiles for latency: **below ~256 KB of payload,
deployment capacity is spent on proxy overhead rather than pixels.** 512×512 `uint16`
(512 KB) is well placed; 256×256 `uint16` (128 KB) is marginal; smaller is wasteful.

### For a single user this does not matter

A user on 50 Mbps pulls ~6 MB/s ≈ 12 tiles/s at 512 KB. A 60-tile burst costs ~100 ms
of serialized proxy CPU against ~5 s of WAN transfer — **the WAN dominates by ~50×.**
The proxy is a **capacity** ceiling, not a latency problem, binding at roughly **20–30
concurrent active viewers**. The tensor server itself saturates near 30 users
(188 MB/s), so the two are comparable and neither blocks shipping.

Before multi-user remote deployment, move tile reads off the control — nginx proxies
`/data_plane` straight to the sidecar. That is a config change, but its real cost is
**auth**: the control's proxy is also the auth boundary (every `/api/` route is
gated), so the tensor server must then enforce its own token on the tile route. That
decision should be made deliberately, not under load. Proxy cost itself is tracked
separately in `biopb/biopb#762`.

Reproduce with `proxybench.py` / `cpuprobe2.py` (measurement fixture:
`ome-tiff_dace9c2e5006/Image:0`, `<u2`, `[1,3,16,512,512]`, chunk `[1,1,1,512,512]` —
a 512×512 slice is exactly 524288 B).

## Deferred

- **Server-side hydrate-ahead.** Cold-chunk latency amplification is about the
  server→upstream hop. In the production topology the sidecar reaches data over NFS on
  the LAN, so that hop is fast. Build this only when deploying a server that proxies a
  remote upstream (see `remote-tensor-cache.md`, and the Model A background
  hydrate-ahead in the cloud-resolve work).
- **Service Worker (L3).** Needs the immutable-GET shape first; add once hit rates are
  measurable.
- **zstd + byte-shuffle.** Measure `Content-Encoding` first.

## Sequencing

1. ~~**Benchmark the control proxy hop**~~ — done. Not a blocker for single users;
   sets a ~256 KB tile-payload floor and a ~20–30 concurrent-viewer ceiling.
2. ~~**GET tile endpoint**~~ — done: `/api/tile_info` + `/api/tile`, ETag +
   `max-age`, `fmt` parameter, tile size derived from `chunk_shape`.
3. ~~**Server-side cancellation**~~ — done, with the threadpool change it depends on.
4. ~~**Caller-supplied `AbortSignal`**~~ — done: every read method on
   `TensorHttpClient` takes `{ signal }`, composed with its own timeout, plus
   `tileInfo()` / `tile()` / `tileImage()` so the tile API is reachable. A caller
   abort raises `TensorAbortError` (`name: "AbortError"`) rather than the 408 the
   helpers used to synthesise, so a tile the viewport moved past is not reported
   as a server failure.
5. ~~**Viv `PixelSource` adapter**~~ — done:
   `web/packages/tensor-flight-client/src/viv-source.ts`. Types come from
   `@vivjs/types` rather than being restated, so the build fails if Viv changes the
   contract. Verified against a live server: the level-0 grid of a 1411×1411 RGB
   source tiles the plane **exactly** (1990921 px covered, no gaps or overlap,
   including the four ragged edge shapes).
6. ~~**Viewer component**~~ — done: `web/packages/app/src/components/TileViewer.tsx`,
   chosen per tensor by `ViewerPane.tsx`. Viv 0.22.1 with deck.gl/luma.gl `~9.3.3`,
   lazy-loaded so the 1.06 MB chunk (299 kB gzip) stays off every other route — the
   eager entry chunk is unchanged at 248 kB. Verified in headless chromium against a
   live server (below).
7. **Measure against a real link**, then decide on zstd and on the `fmt=jpeg` policy.

### What the adapter had to reconcile

- **`onTileError` is required** by `PixelSource` and is where a cancelled tile must
  be swallowed — an aborted tile is a normal outcome, not a load failure.
- **`labels` must end `y`, `x`** (`"_c"` after them for interleaved RGB(A)). The
  canonical `[..., Z, Y, X, S]` order satisfies this after lowercasing; a tensor that
  does not is rejected rather than rendered transposed.
- **`data` is a `TypedArray`,** not an `ArrayBuffer`, and `dtype` is Viv's spelling
  (`Uint16`), not numpy's. Both numpy spellings the server emits (`<u2` from
  `tile_info`, `uint16` from the tile header) map to it.
- **`getRaster` reads a whole level**, which is right for the overview and would be
  the entire image on level 0. It is served by one `/api/slice` at that level's
  `scale_hint` and refuses beyond `maxRasterPixels` (default 4096²).
- **Only `t`/`z`/`c` are selectable** through the tile API. Any other slider axis is
  served at index 0 — correct as a default, wrong for anything else — so a non-zero
  request on one is refused instead of quietly returning the wrong plane. Extending
  the endpoint is the fix if a dataset needs it.
- **Out-of-grid tiles are answered locally** with zeros. deck.gl can ask past the edge
  while a viewport settles, and a round trip for a routine 404 is waste.

### What the viewer component had to reconcile

Three things that the design above got wrong or left implicit, each found by
measurement rather than reading.

**`maxCacheByteSize` is unusable with Viv, and setting it is worse than not.**
The cache-hierarchy note above says to bound L1 by bytes. deck.gl reads
`tile.content.byteLength`, and Viv's `getTileData` returns a plain
`{ data, width, height }` object which has no such property: it logs
`byteLength not defined in tile data` **once per tile** and counts zero — measured,
6 errors in one probe run. Worse, passing `maxCacheByteSize` at all switches
`maxCacheSize` to `Infinity`, so the bound that *was* working is removed and the
cache grows without limit.

`maxCacheSize` (a count) is the control that works, and it loses nothing here:
tile edge, dtype and sample count are all known from `tile_info`, so a count **is**
a byte budget for this data. `tileCacheSize()` derives it from 128 MiB — ~256 tiles
of 512×512 `uint16`, which budgets ~256 MiB across RAM and VRAM once the GPU texture
is counted. Re-check on a Viv bump: if Viv starts reporting `byteLength`,
`maxCacheByteSize` becomes the better prop.

**Viv's `ImageLayer` diffs `selections` by reference, and zustand defeats that.**
`updateState` refetches when `props.selections !== oldProps.selections`, so any new
array identity re-reads the whole background raster. zustand hands out a fresh
`slice` object on *every* `setSlice`, so a selection derived straight from it changed
identity when only the contrast slider moved — and the "contrast costs no round trip"
claim quietly failed: **2 requests per contrast change**, measured. Deriving the
selection from its own content (and memoising the `selections` array separately)
takes it to **0**. This is the kind of thing the four-layer cache table cannot warn
about, because the refetch happens above all four.

**The coarsest level is kept resident for free.** `MultiscaleImageLayer` already
renders a background `ImageLayer` from `loader[loader.length - 1]` via `getRaster`,
so no extra work was needed. It also means a **single-level** tensor never uses the
tile grid at all — Viv picks `ImageLayer` when `loader.length === 1`. That is safe
only because `_tile_levels` stops halving once the plane fits one tile, so one level
implies an image no larger than `tile_size`; a single-level pyramid over a large
image would make Viv read the whole thing in one request.

**`@vivjs/loaders` cannot be avoided by import choice.** The framework note above
says it is not used. In the app it is: `@vivjs/views` imports `getImageSize` from it
and `@vivjs/layers` imports `isInterleaved`/`SIGNAL_ABORTED`, and the package ships
without `sideEffects: false`, so Rollup keeps geotiff's core in the graph. Importing
the subpackages directly instead of the `@hms-dbmi/viv` umbrella makes **no**
difference — measured byte-identical at 1,057.63 kB. Aliasing `@vivjs/loaders` to a
three-function stub removes 108 kB (38 kB gzip) from the lazy chunk plus ~120 kB of
decoder chunks that are never fetched. Not taken: it means vendoring three semantic
definitions that can drift silently, for 38 kB off a chunk that is already lazy.

**A t/c/z change has to be marked invalid; only zoom/pan may refine locally.**
Both Viv layers keep their previous raster until a new read resolves, so a plane
change left the old plane painted for exactly as long as the read took — frames
measured **byte-identical** across a 6 s held read, with nothing on screen saying
so. A stale plane is worse than a blank one: it is indistinguishable from the
right answer, and a plane that never changes reads as a hung viewer rather than a
slow one. deck.gl's `TileLayer` supplies the completion signal, `onViewportLoad`,
and Viv forwards unknown props down to it while pinning the background
`ImageLayer`'s own callback to `null` — so it fires once per completed viewport,
not twice. Keyed against the current selection it gives an explicit valid/invalid
state: the stale window drops from the read duration to **under ~270 ms** (a
commit and a paint) and stops scaling with the backend. It re-fires for a fully
cached selection, so stepping back to a visited plane clears the cover instead of
sticking on it. Zoom and pan never invalidate — they change which tiles are
wanted, not which plane — which is what keeps `refinementStrategy:
'best-available'` doing its job.

Two things the signal does not give for free. deck.gl counts a **failed** tile as
loaded — `_isLoaded = true` with `content = null`, so `onError` fires and the
tileset still reports the viewport complete — which would clear the cover over a
canvas that never received the plane. Requiring every selected tile to carry
content keeps it up. (An *aborted* tile is unaffected: `_isCancelled && !tileData`
leaves it unloaded, so it never reaches the callback.) And once a plane is known
to have landed, a featureless one still renders black, which is why the coarsest
level's contrast samples double as an emptiness check — `sorted[0] === sorted.at(-1)`
labels the plane rather than leaving black to mean three different things. That
label is keyed to the selection: the samples are deliberately kept across a plane
change so contrast does not flash, so an unkeyed label would describe the plane
before last.

And the signal arrives in **two shapes**. A pyramid reports the array of deck.gl
tiles; an image that needs only one level is rendered by Viv's plain `ImageLayer`,
which reports the single raster it read. Accepting only the array left every
single-level image permanently covered — the exact case the tile grid is skipped
for.

**Nothing cancels a superseded `getRaster`, so the viewer has to.**
`ImageLayer.updateState` makes a new `AbortController` per selection and drops the
old one un-aborted; only `finalizeState` aborts, and only the current one. Eight
z-steps 250 ms apart against a 6 s backend left **9 reads in flight**, monotonic,
three of them alive until `chunkTimeoutMs` killed them at 8 s — and past the
browser's six-connections-per-origin cap the newest read queues behind stale ones
nobody wants. The same trigger doubled the count: the background `ImageLayer` and
the contrast sampler both read the coarsest level for every selection, neither
aware of the other (**16** `/api/slice` for 8 steps). `RasterRequests` in
`viv-source.ts` keeps one in-flight read per (level, selection) — a second caller
joins it, a new selection aborts every older one, each caller's own signal stays
independent, and the shared read dies with its last waiter. Measured: **8**
requests for 8 steps, **1** in flight throughout.

It also removes a false clear. A superseded raster that resolves late calls
`onViewportLoad`, which reads the *current* selection — so the cover lifted at
+4229 ms over a canvas that then changed by **45%** when the real plane landed.
With superseded reads aborted the cover lifts at +6037 ms and the canvas changes
by **0%**: one backend read after the last step, which is the floor.

Aborts must reject with Viv's `SIGNAL_ABORTED` (`"__vivSignalAborted"`), not an
`AbortError`. `ImageLayer` ends its chain with
`catch(e => { if (e !== SIGNAL_ABORTED) throw e })`, so anything else is rethrown
inside a `.catch` nobody follows.

**A level that fits one tile is one tile, so the raster never needs
`/api/slice`.** `_tile_levels` stops halving as soon as the plane fits the edge,
so the coarsest level always has `cols === rows === 1` — and that is the only
level Viv reads a raster from, since both the background `ImageLayer` and the
contrast sampler take `loader[loader.length - 1]`. Verified against the live
server: `GET /api/tile?level=<coarsest>&col=0&row=0` and the `POST /api/slice`
the viewer used to send return the same `X-Shape`, the same dtype, and
**byte-for-byte identical** bodies — 425,042 B on a 922² `uint16` source, 373,827 B
on a 1411² interleaved RGB one. The tile route carries an ETag and
`max-age=3600`; `/api/slice` is a POST and cacheable by nothing.

Routing the raster there is four lines in `viv-source.ts` and no server change.
Measured per selection change:

| | before | after |
|---|---|---|
| 922², 2 levels, cold | 4 tiles + 1 slice (425 KB, 14–21 ms) | 5 tiles, 0 slices |
| 922², 2 levels, warm | 4 cache hits **+ 1 backend read** | 5 cache hits, **0 bytes** |
| 512², 1 level, cold | 0 tiles + 1 slice (525 KB) | 1 tile |
| 512², 1 level, warm | 1 slice, 525 KB, ~16 ms — *every time* | cached, 0 bytes, ~2 ms |

The single-level row is the larger win: with one level Viv uses plain
`ImageLayer`, so the raster is not a backdrop, it **is** the image — the whole
render was uncacheable and is now entirely cache-served on a revisit.

One honest caveat about the sharing: `RasterRequests` merges callers that
overlap *in time*. A cold read is slow enough that the background layer and the
contrast sampler always overlap and share one request; a warm read can complete
in ~2 ms, between the two, so a revisit sometimes issues two requests. Both are
cache hits costing no bytes, so this is a note rather than a defect.

### Verified in a browser, not just in node

`viv_browser_probe.mjs` drives the real `ViewerPane` in headless chromium over CDP
against a live sidecar, and reads the composited frame back (CDP screenshots, so it
does not need `preserveDrawingBuffer`). Against three real OME-TIFFs — a
single-level `uint16` stack, a 3-level interleaved RGB, and a 2-level 4-channel
`uint16`:

| claim | result |
|---|---|
| tiles reach the GPU | 86–100% of the canvas lit on all three |
| contrast is a shader uniform | repaints 14.2% of pixels, **0 requests** |
| a channel change is real data | repaints 12.7%, **3 requests** |
| the tile grid is used when it exists | 4 tile requests for the RGB pyramid |
| single-level takes the `ImageLayer` path | 0 tile requests, still renders |
| cache bound is live | no `byteLength not defined` in the console |

Pixel checks, not smoke tests: "no round trip" is asserted from
`performance.getEntriesByType("resource")` and "repaints" from a per-pixel luminance
diff of two screenshots.

### Threadpool size bounds what cancellation can reclaim

`run_in_threadpool` uses anyio's default 40-thread limiter, so a burst of ≤40 tiles
starts immediately and **none of it queues** — a 40-tile abort through the adapter
skipped 0 server-side reads, where a 60-tile burst skipped 34. That is the documented
queued-vs-in-flight boundary, not a defect, but it means the pool size sets the
reclaim ceiling. 40 concurrent Flight reads on a 12-core box is also oversubscribed;
worth tuning alongside biopb/biopb#762.

End-to-end, one `AbortController` per viewport abandoned mid-burst: 60 tiles → 11
completed, 49 aborted client-side, **34 reads skipped server-side**. The gap
between 49 and 34 is requests already running when the abort landed — the
queued/in-flight boundary, visible in practice.

Steps 2–4 are plumbing that is useful regardless and do not depend on the framework
choice — worth landing before committing to Viv, since they are also where the
surprises live (see the cancellation coupling above, which only showed up against a
running server).
