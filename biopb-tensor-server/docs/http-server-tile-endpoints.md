# Remote data viewer: client-side rendering over a tile API

Scope: `biopb-tensor-server` (HTTP sidecar, tile route), `web/` (viewer SPA),
`biopb-control` (same-origin proxy for `/data_plane/*`).

## Architecture

The server ships raw pixel data; the browser uploads it to WebGL textures and
applies contrast, colormap, and multi-channel blending in a fragment shader.
Traffic is bounded by **screen size, not image size** -- only a viewport's
worth of pixels at the current pyramid level is ever fetched.

## Rendering framework: Viv

[Viv](https://github.com/hms-dbmi/viv) (`@hms-dbmi/viv`, 0.22.1), deck.gl
layers for bioimaging, pinned together with `@deck.gl/*` and `@luma.gl/*` at
`~9.3.3` (Viv's peer range is a tilde -- re-check it on every Viv bump).

A multiscale image is a `PixelSource[]`, one per pyramid level:

```ts
interface PixelSource {
  shape: number[]; labels: string[]; dtype: string; tileSize: number;
  getTile({ x, y, selection, signal }): Promise<{ data: TypedArray, width, height }>
  getRaster({ selection, signal }): Promise<{ data: TypedArray, width, height }>
}
```

`web/packages/tensor-flight-client/src/viv-source.ts` adapts the tile API to
this interface.

**Known limitations:**
- WebGL2 is required for native textures (`R16UI`), so a browser without it gets a stated refusal.
- `MAX_CHANNELS = 10` (`@vivjs/constants`) is a hard wall for highly multiplexed data.
- Viv covers u1/i1/u2/i2/u4/i4/f4/f8 only -- `int64`, `float16`, `bool`, and complex have no GPU equivalent and are refused

## Tile endpoint

Two **GET** routes, everything in the URL, so the browser cache works:

```
GET /data_plane/api/tile_info/{array_id}
GET /data_plane/api/tile/{array_id}?level=&col=&row=&fmt=raw&t=&z=&c=&sel=
```

### `tile_info` response

`GET /api/tile_info/{array_id}` reports everything needed to address the
tensor as a tile grid -- shaped to drop into a Viv `PixelSource[]`:

```json
{
  "array_id": "src/Image:0", "dim_labels": ["T","C","Z","Y","X"],
  "shape": [1,3,16,512,512], "chunk_shape": [1,1,1,512,512], "dtype": "<u2",
  "tile_size": 512,
  "plane": {"y": 3, "x": 4, "s": null},
  "selectable": {"t": 0, "c": 1, "z": 2},
  "sel_axes": [],
  "levels": [{"level":0,"scale":1,"height":512,"width":512,"cols":1,"rows":1}],
  "pyramid": [{"scale_hint":[1,1,1,2,2],"shape":[1,3,16,256,256],
               "reduction_method":"precompute","native":true}],
  "volume": {"available": true, "reason": null,
             "axes": {"z":2,"y":3,"x":4}, "scale_hint": [1,1,1,1,1],
             "depth": 16, "height": 512, "width": 512, "bytes": 8388608,
             "spacing": null, "unit": null}
}
```

`levels` is the ladder a client addresses -- always powers of two, required by
Viv's `PixelSource[]` convention. **`level` 0 is full resolution**, each
level halving from there.

`pyramid` is the ladder the **server** advertises, and what each rung is actually *read from*.
Advisory: published for diagnosis, and as the only place a client can see *why* one source's tiles
are cheap and another's are not. A tile always decimates: `reduction_method` is not accepted on `/api/tile`
(below).

`volume` is **not** a rung of `levels` and not part of the tile grid, since a 3-D renderer takes one whole volume rather than tiles.
Instead, it describes what a `scale_policy: "volume"` read on `/api/slice` will return, or
`available: false` with a reason to show (no z axis, a z extent of 1, an
interleaved samples axis). `spacing` is that volume's physical voxel extent
(source size × `scale_hint`, reduced to one unit), `null` when unavailable.
`scale_hint` is normally `pyramid`'s coarsest entry but is **bounded**: a
native ladder downsampling only Y/X would otherwise leave a full-depth
volume with no 3-D voxel budget applied, so over budget the plan falls back
to a computed scale and `reduction_method` comes back `null` -- clients
should read the extents rather than assume the plan matches a `pyramid`
entry.

`selectable` gives the wire index of each **named** slider axis, or `null`.

`sel_axes` is the converse: non-plane axes with extent > 1 that `t`/`z`/`c`
cannot *name*. Empty for an ordinary TCZYX tensor.

`tile_size` is derived from `chunk_shape` -- the transfer grid, taken from a
`GetFlightInfo` describe of this tensor.

#### Content-versioned array_ids

`/api/tile_info` publishes `source_id "@" token [ "/" field ]` when the
tensor carries a `content_version` -- e.g. `zarr_a3f2@9f1c4e2b/Image:0`. The
token is the first 8 hex of `sha256(content_version, CHUNK_SEMANTICS_EPOCH)`.
`content_version` is a **serving field** on `TensorDescriptor` (like
`chunk_shape` and `pyramid`), filled by `GetFlightInfo` from the bound
adapter.

An **unversioned** source has no `content_version` and gets no token, so it
never gets the year-long `immutable` caching a versioned source does.

| request | result |
|---|---|
| versioned, current token | 200, `immutable` |
| versioned, superseded token | **404**, listing the array_ids that exist |
| unversioned | 200, `max-age=3600` |
| source publishes no version | 200, `max-age=3600` |

#### Serving a tile from the warm ladder

The tile ladder's coarsest rung (one tile) is far coarser than the warm
target precache keeps resident -- e.g. scale 32 vs. scale 8 on a 14234^2
scene (see [precache-policy.md](precache-policy.md)). Rather than warm both
ladders or leave one cold, `_tile_read` (`serving/http_server.py`) serves
every rung from the coarsest **advertised** level that still divides it, and
reduces the remainder in-process:

- A rung at or coarser than an advertised level's scale is read from that
  level's warm chunks and decimated the rest of the way -- no second cache
  entry, no separate scaled read from the data plane.
- A rung finer than every advertised level reads full resolution and
  decimates from there, for the same reason: reusing level-0's own chunks
  costs nothing extra, where asking the data plane for an unwarmed scale
  would mint one.
- A native on-disk level is addressed by an exact `(scale_hint,
  "precompute")` match instead, since a stored level is the writer's own
  downsampling, not a decimation this route can reproduce elsewhere. A
  level whose stored extent disagrees with the tile grid's `ceil(extent /
  scale)` arithmetic is skipped, so a rung never promises pixels the store
  doesn't have.

### `GET /api/tile` parameters and validation

Takes `level`, `col`, `row`, the selection `t` / `z` / `c` (default 0) and
`sel` (below), and `fmt` (`raw` | `png` | `jpeg`, plus `lo` / `hi` / `color` /
`use_min_max` for the rendered formats). Extents are the full-resolution ones.
Validation runs *before* the ETag check, so a nonexistent
tile cannot be turned into a cheap 304 by a stale or forged `If-None-Match`.

**`sel=<axis>:<index>`**, repeatable, selects an axis by its **wire index** --
the only handle an axis in `sel_axes` has. `GET /api/tile/seq?sel=0:154`
serves frame 154 of a 155-file TIFF sequence. It composes with the named parameters
(`?sel=0:4&c=2`) and is refused, **422**, when malformed, when it names the same
axis twice, an axis the tensor does not have, a plane axis, or an axis
`t`/`z`/`c` already name -- even when the two agree. Unlike `t`/`z`/`c`,
`sel` has no index-0 exemption. The ETag is computed over the
**resolved** selection rather than the raw parameters.

`(level, col, row)` is validated against exactly the grid `/api/tile_info`
publishes -- **404** otherwise. A selection index outside its axis is
**422**.

`t`/`z`/`c` are checked against the axis they name, not merely `ge=0`.

**`fmt` accepts only `raw`** -- **410** otherwise (the server-rendered
png/jpeg forms are gone).

`reduction_method` answers **410** unless it names the decimation tiles already do
(`nearest`, or an alias such as `decimate`, accepted and ignored)

> **`level` is not a harmless over-zoom.** `scale_hint` is honoured down into
> `downsample_block`, which edge-pads its input up to a multiple of the scale
> factor: level 17 on a 512px plane would ask the *data plane* to allocate and
> write a 65536x65536 array, in the Flight process shared by every other
> caller. Hence the level gate, which rejects the request before the read
> starts.

## Cancellation

A client `AbortController` stops the browser from waiting on a tile it no
longer wants. The shared mechanism (`is_disconnected()`, 499,
`run_in_threadpool` on both `/api/tile` and `/api/slice`) is in
[http-server.md](http-server.md). Tile-specific: the threadpool's size
(anyio's default 40) is the reclaim ceiling -- a burst within the limit
starts immediately and none of it is cancellable, so only requests still
queued when the abort lands get skipped.

## Cache hierarchy

Three layers; Viv contributes none of them directly.

| | Cache | Owner | Unit | Survives |
|---|---|---|---|---|
| L1 | `Tileset2D` LRU | deck.gl `TileLayer` | decoded TypedArray + GPU texture | pan/zoom only |
| L2 | HTTP cache | browser | response bytes | reload, selection change |
| L3 | segment cache | tensor server | chunks | everything, server-side |

## 3D volume path

No pyramid in 3D: the volume is requested at a scale that fits in memory as
one WebGL2 3D texture and ray-cast, matching Viv's own 3D model (it loads
one resolution level and does not stream out-of-core).

The server picks the scale, not the client's GPU: `scale_policy: "volume"`
on `/api/slice` resolves to the level the precache worker keeps warm.
The volume is fetched in one `/api/slice` request as a C-contiguous slab and
assembled into a single buffer -- not Viv's default per-Z-plane loop, which
would cost one round trip per plane. `XR3DLayer` is driven directly rather than through Viv's packaged
`VolumeViewer`, which would instead lock browser 3-D to the tile ladder.

## Known limitations

- **zstd + byte-shuffle compression** is not built; `Content-Encoding` at
  the nginx edge is free but weak on 16-bit data (~1.2-1.5x), byte-shuffle +
  zstd would do better (2-4x) at the cost of a wasm decoder -- not worth
  adding without a measured need.
- **Server-side hydrate-ahead** for cold-chunk latency only matters when the
  server itself proxies a remote upstream over a slow link; the production
  topology reaches data over local NFS, so this is not built.
- **The control proxy hop** A nginx proxy in front of control costs
  materially more CPU per request than the tensor server spends serving it, so
  proxied small-request throughput falls as concurrency rises. It
  caps remote deployments at roughly 20-30 concurrent viewers.
- **3D has no progressive refinement**, no LOD during interaction
  (mitigated by cutting ray-sampling steps while the camera moves), and no
  L1/L2 caching (the volume is held explicitly, keyed on `(tensor, level, t,
  channel)`, bounded to 2-3 entries).
