# The tensor-server HTTP sidecar (FastAPI)

Endpoint/contract reference for the **API-only** HTTP sidecar. For how it fits the
data plane, see `../ARCHITECTURE.md` (§ FastAPI HTTP Server): it wraps the Python
`TensorFlightClient` and re-exposes it as HTTP/JSON (+ binary slices) so browsers
reach the data plane without a gRPC-Web proxy. It serves **no** static assets —
the control plane owns the browser UI and reverse-proxies this sidecar under
`/data_plane/*`.

**Module:** `biopb_tensor_server.serving.http_server` ·
**Factory:** `create_app(flight_location, token, cache_bytes, cors_origins, config_path, supervised) → FastAPI` ·
**Port:** `8814` under the control plane (which passes `--web-port 8814`); a bare
`biopb-tensor-server launch` defaults to `8816`.

## Lifecycle

The app holds two pieces of shared mutable state created at factory time:

- **`_client_holder`** — lazily initialised `TensorFlightClient`; the first
  authenticated request that reaches any protected endpoint triggers the gRPC
  connection to `flight_location`.
- **`_DiagnosticsState`** — thread-safe container for latency samples, error
  events, cache counters, and per-session rate-limit state.

## Authentication

Two equivalent header schemes are accepted on every protected endpoint:

```
Authorization: Bearer <token>
X-Biopb-Token: <token>
```

`secrets.compare_digest` is used for timing-safe comparison. The auth check
compares against `expected = self.token`: a `None` token means no enforcement, a
token present means it is enforced on every protected endpoint. There is no
separate dev flag.

**Enforcement is independent of the network mode.** The `--host` flag decides
the *bind* (loopback = local, public = remote; `biopb control start --remote`
selects the public one), not whether a token exists. Remote mode **requires** one (auto-generated if not supplied — a public
listener is never left open). Local mode is tokenless *by default*, but a token
passed via `--token` / `BIOPB_TENSOR_TOKEN` is honored and enforced on the
loopback listeners too (`_resolve_launch_token` takes a supplied token before it
ever looks at the bind) — defense-in-depth on a shared machine. So "no token"
is the local **default**, not a property of local mode.

> **Caveat (biopb/biopb#470):** `/api/admin/status` reports `local` as
> "the token is `None`", which a token-gated *loopback* deployment fails — it
> reports `local: false`, and `/api/admin/browse` then 404s. Fails closed; the
> data endpoints are unaffected.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/livez` | ✗ | Liveness probe — `{"status":"ok","timestamp":"…"}`. Never contacts the backend |
| `GET` | `/readyz` | ✗ | Readiness — **200 when Flight reports `SERVING`, 503 otherwise**. Adds `ready`, `backend_health`, `backend_error`, `source_count`, `dev_mode`, `service`, `version` |
| `GET` | `/healthz` | ✗ | Alias for `/readyz` |

`/readyz` opens the Flight connection if none exists yet, so it answers from the
backend rather than from whatever traffic happened to arrive first, and it is
safe for a supervisor to gate on. `backend_health` is `null` exactly when the
backend was not reached, and `backend_error` then says why (`connect failed: …`
vs `health check failed: …`) — the two used to be indistinguishable, and both
looked the same as "nobody has asked yet" (biopb/biopb#755).
| `GET` | `/api/diagnostics` | ✓ | Diagnostics snapshot; rate-limited 1 req/s per session |
| `GET` | `/api/sources` | ✓ | JSON array of `DataSourceDescriptor` objects |
| `GET` | `/api/sources/{id}` | ✓ | Single descriptor |
| `GET` | `/api/sources/{id}/metadata` | ✓ | Parsed `metadata_json` field |
| `POST` | `/api/sources/query` | ✓ | Server-side DuckDB SQL over the catalog |
| `GET` | `/api/sources/{id}/ticket/{ticket_hex}` | ✓ | Resolve a Flight ticket to bytes |
| `GET` | `/api/tile_info/{array_id}` | ✓ | Tile grid, pyramid levels, selectable axes and the 3-D volume plan |
| `GET` | `/api/tile/{array_id}` | ✓ | One tile, cacheable (raw bytes) |
| `POST` | `/api/slice` | ✓ | Binary tensor sub-region; `scale_policy` delegates the scale |
| `GET` | `/api/rois/{array_id}` | ✓ | A tensor's whole ROI annotation set (`?set=` for one layer) |
| `POST` | `/api/rois/{array_id}` | ✓ | Create/update annotations (same-origin guarded) |
| `DELETE` | `/api/rois/{array_id}` | ✓ | Delete by `?ids=a,b`, else the whole set / one `?set=` (same-origin guarded) |
| `GET` | `/api/config` | ✓ | Current config (secrets redacted) |
| `PUT` | `/api/config` | ✓ | Update config (same-origin guarded) |
| `GET` | `/api/admin/status` | ✓ | Server/catalog status for the admin page |
| `GET` | `/api/admin/browse` | ✓ | Filesystem browse for the data-folder picker (local only — see the auth caveat) |

> **Route ordering:** `/api/sources/{id}/metadata` and `/ticket/{ticket_hex}` are
> registered *before* the greedy `{source_id:path}` catch-all to avoid Starlette
> first-match shadowing.

> **ROI annotations** (`/api/rois/*`) carry canonical proto3 JSON of
> `biopb.image.RoiAnnotation` in both directions. The version token is stripped
> from `array_id` before the store sees it — annotations anchor on the
> unversioned id so they outlive an in-place edit — and spliced back onto the
> response. `501` means the server does not offer annotations (disabled, or no
> metadata DB); `422` means the request was rejected (geometry the store does not
> accept, mismatched `array_id`, per-tensor cap). Design: `roi-annotations.md`.

> **Source listings are structural.** Each `tensors[]` entry on `/api/sources`
> carries `array_id` / `dim_labels` / `shape` / `dtype`; `chunk_shape` is `[]`
> there and is **not** a usable grid. The transfer grid belongs to the tensor the
> server binds to serve a read, so ask `/api/tile_info/{array_id}` for it
> (biopb/biopb#812).

## Tile endpoints

Design rationale in `remote-viewer-tiles.md`; this is the contract.

**Addressed by `array_id` alone**, per the identity policy at the top of
`proto/biopb/tensor/descriptor.proto`. array_id is globally unique and
authoritative; `source_id` is only the slash-free routing prefix. The path is `{array_id:path}`, so a field
containing `/` (HCS `plate/A01/0`) is captured whole, percent-encoded or not.

Resolution is **one targeted `GetFlightInfo`** — the catalog is never listed to
look up an id the request already carries (biopb/biopb#834).

**array_id policy is the Flight server's.** A bare source_id resolves to whatever
that server binds for it — its default tensor — and the sidecar does not
second-guess it with a tensor count of its own. The answer carries the array_id
it resolved to, and that is the id the sidecar reports and reads from, so the
geometry and the read cannot come from two derivations. That agreement is what
biopb/biopb#75 was about; a status code was only ever a proxy for it.

So `/api/tile_info/{source_id}` on a multi-tensor source is a **200** publishing
the qualified array_id the server chose, not a 404. First contact is where the
ambiguity ends — the viewer threads that id back through every tile after.

### Content-versioned array_ids

`/api/tile_info` publishes `source_id "@" token [ "/" field ]` when the tensor
carries a `content_version` — e.g. `zarr_a3f2@9f1c4e2b/Image:0`. The token is the
first 8 hex of `sha256(content_version)`; the raw value is a stat signature whose
mtime has no business in every tile URL and access log.

`content_version` is a **serving field** on `TensorDescriptor`, like `chunk_shape`
and `pyramid`: filled by `GetFlightInfo` from the bound adapter, empty on the
structural `DataSourceDescriptor.tensors[]` entries. It is a source-level property
repeated on the tensor deliberately — the check has to read the freshest thing in
the request.

This exists **only above the Flight wire**. The sidecar strips it before every
Flight call; no adapter, chunk_id, catalog row or descriptor carries it. A `@` in
a *field* name is untouched — only the half before the first `/` is parsed, and a
`source_id` is `<type>_<hex>` and can never contain one.

Nothing about the client changes. The viewer already fetches `tile_info` once per
tensor and threads the `array_id` it answers with through every subsequent tile
URL, so publishing the versioned form is the whole delivery mechanism.

| request | result |
|---|---|
| versioned, current token | 200, `immutable` |
| versioned, superseded token | **404**, listing the array_ids that exist |
| unversioned | 200, `max-age=3600` |
| source publishes no version | 200, `max-age=3600` |

A superseded token is a plain 404, not a distinct status: a stale bookmark and a
typo both want "ask again", and the 404 already lists what does exist.

The ETag folds in the source's **current** version, not the requested one. The
versioned URL changes on its own when content changes; this is what stops the
*unversioned* URL — stable across a re-index — from revalidating to a 304 for
bytes that changed.

> The token is read off the **descriptor**, never the source listing —
> `GetFlightInfo` is fetch-per-call by contract, while the listing was the
> expensive part and the obvious thing to cache. That independence is what let
> biopb/biopb#834 drop the listing from the resolution path. A test pins it: a
> listing frozen at a superseded version still yields a 404.

`GET /api/tile_info/{array_id}` reports everything needed to address the tensor as
a tile grid — shaped to drop into a Viv `PixelSource[]`:

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

`levels` is the ladder a client addresses — always powers of two, because Viv's
`PixelSource[]` convention requires it. `pyramid` is the ladder the **server**
advertises, and is what each rung is actually *read from*: a tile is served by
the coarsest entry whose `scale_hint` divides the rung's scale, with the
remainder decimated in-process. Level 0 is omitted (it names full resolution,
which a caller gets without asking) and the list is coarsest first.

Two kinds ride the same field. A **native** entry (`native: true`,
`reduction_method: "precompute"`) is a real on-disk level — OME-Zarr
multiscales, a pyramidal QPTIFF — and reading it reads that level's own store.
A **computed** entry is what the precache worker warms. Either way the rungs are
addressed identically, so `pyramid` is advisory: it is published for diagnosis,
and because it is the only place a client can see *why* one source's tiles are
cheap and another's are not.

A tile always decimates. `reduction_method` used to be accepted here and is
**withdrawn** (below): a tile is the display path, so what it selected in
practice was a *store* — opt out of the pyramid, decimate full resolution —
rather than a kernel, and a stored level carries the writer's downsampling
whatever a caller names. `POST /api/slice` is where a kernel is a real choice.

`volume` is the odd one out: it is not a rung of `levels` and does not belong to
the tile grid at all. A 3-D renderer takes one whole volume rather than tiles, so
it leaves the ladder entirely — but this is the one call a viewer already makes
before it can address the tensor, so the plan rides along rather than costing a
second round trip. It describes what a `scale_policy: "volume"` read (below)
will return, or says `available: false` with a reason a viewer can show: no z
axis, a z extent of 1, an interleaved samples axis. `spacing` is the physical
extent of one voxel *of that volume* — the source's physical size already
multiplied by `scale_hint`, and with the three axes reduced to one unit
(`physical_unit` is per-axis and adapters do not all normalise: NIfTI reports
`mm`, the EM readers `nm`). `null` when the source declares no size, or when
the axes carry units that differ and cannot all be placed on a common scale —
a plausible wrong ratio is worse than rendering isotropic.

`volume.scale_hint` is normally the coarsest entry of `pyramid`, but it is
**bounded** rather than taken on trust. A native ladder is advertised instead of
the server's computed plan, so one that downsamples only Y/X leaves a full-depth
volume that no 3-D voxel budget has been applied to (biopb/biopb#891) — at
gigabyte scale on the wire and in VRAM. Over the budget, the plan falls back to
the computed scale and `reduction_method` comes back `null`, saying the read is
no longer addressed to a stored level. Clients should read the extents rather
than assume the plan matches a `pyramid` entry.

`selectable` gives the wire index of each **named** slider axis, or `null`.
`sel_axes` is the converse and is the one worth reading: non-plane axes with
extent > 1 that `t`/`z`/`c` cannot *name* — an unlabelled axis
(`[{"axis":0,"label":"POS","extent":5}]`), a TIFF sequence's opaque file axis
(`"i"`), or the second of two axes sharing a label, since only the first
occurrence resolves. Empty for an ordinary TCZYX tensor.

Naming is not addressing: `sel_axes` entries **are** selectable, through the
`sel` parameter below. What the list says is that a client must reach them
positionally, and that the server has no semantic title to offer for the
slider — so show the source's own label (`i`), not an invented `Z`. Deriving
one positionally is the guess `core/axes.py` declines to make, and making it
client-side only moves it somewhere less visible.

**`level` 0 is full resolution** (Viv's `PixelSource[]` index convention, not the
map-tile one where z grows with detail); each level halves. `tile_size` is derived
from `chunk_shape` — the transfer grid, taken from a `GetFlightInfo` describe of
this tensor, not from the source listing — so a tile *nests* inside a delivered
chunk rather than straddling one; clients must not assume a constant.

`GET /api/tile/{array_id}` takes `level`, `col`, `row`, the selection
`t` / `z` / `c` (default 0) and `sel` (below), and `fmt` (`raw` | `png` |
`jpeg`, plus `lo` / `hi` / `color` / `use_min_max` for the rendered formats).

**`sel=<axis>:<index>`**, repeatable, selects an axis by its **wire index** —
the only handle an axis in `sel_axes` has. `GET /api/tile/seq?sel=0:154` serves
frame 154 of a 155-file TIFF sequence; before it, that tensor was a one-frame
image to every tiled client. It composes with the named parameters
(`?sel=0:4&c=2`) and is refused, 422, when it is malformed, names the same axis
twice, names an axis the tensor does not have, names a plane axis, or names an
axis `t`/`z`/`c` already name — that last one even when the two agree, because
one axis with two spellings in one URL is two cache entries for one tile.
Unlike `t`/`z`/`c`, `sel` has no index-0 exemption: it is never a default, so
`sel=9:0` on a 3-D tensor is a client addressing an axis it believes exists.

The ETag is computed over the **resolved** selection rather than the raw
parameters, so the two spellings of one plane share a cache entry and a
parameter the resolution ignored cannot vary the key.

| | |
|---|---|
| body | `application/octet-stream`, the tensor's own dtype, C-contiguous. `X-Shape` / `X-Dtype` / `X-Dim-Labels` as on `/api/slice` |
| always | `ETag`, `Cache-Control: private, max-age=3600`, `Vary: Authorization`, `X-Tile-Size` / `-Level` / `-Col` / `-Row` |

`If-None-Match` revalidates to **304 without reading tile data**. Revalidation
still consults the catalog to resolve the tensor descriptor and compute the
ETag, but it does not call `get_tensor()` or run a data read. Appearance is a
client-side shader concern and no appearance parameter is declared, so adjusting
contrast cannot fragment the tile cache.

`(level, col, row)` is validated against exactly the grid `/api/tile_info`
publishes — a level the tensor does not have, or a tile outside that level's
`cols`×`rows`, is **404**. A selection index outside its axis is **422**.

`fmt` accepts only `raw`. The server-composited `png` / `jpeg` forms were removed
with the server-rendered viewer and now answer **410** — a deliberate refusal
rather than silently handing raw bytes to a caller that asked for an image.

`reduction_method` answers **410** for the same reason, unless it names the
decimation tiles already do (`nearest`, or an alias of it such as `decimate`,
which is accepted and ignored). Serving pixels reduced by a different kernel
than the caller asked for is exactly the silent substitution the 410 exists to
prevent. It is no longer part of the ETag: nothing about a tile varies with it.

`t`/`z`/`c` are checked against the axis they name, not merely against `ge=0`: an
index past the axis extent is **422**, and so is a *non-zero* index on an axis the
tensor does not have. Index 0 there stays valid — it is the default every client
sends. Extents are the full-resolution ones, correct at every level because
`scale_hint` is 1 on non-plane axes.

Validation runs *before* the ETag check, so a nonexistent tile cannot be turned
into a cheap 304 by a stale or forged `If-None-Match`.

> **`level` is not a harmless over-zoom.** `scale_hint` is honoured down into
> `downsample_block`, which edge-pads its input up to a multiple of the scale
> factor: level 17 on a 512px plane would ask the *data plane* to allocate and
> write a 65536×65536 array, in the Flight process shared by every other caller.
> numpy refuses the absurd sizes (surfacing as 502), but the band that merely
> exhausts memory succeeds. Hence the level gate, and hence it rejects before
> `get_tensor()` rather than letting the read path discover it.

> **Two cache policies, chosen by the URL.** A tile whose `array_id` carries a
> content version (below) gets `private, max-age=31536000, immutable` — the URL
> names its own content, so a re-index mints a different id and the old URL 404s
> rather than answering stale pixels. A source that publishes no version keeps
> the old `max-age=3600` hedge: nothing distinguishes its content across a
> re-index, so an hour is still the honest ceiling.

> **`private`, never `public`.** The URL carries no token — auth is a header, so
> rotation does not bust the cache — and RFC 9111 §3.5 lets a *shared* cache reuse
> a response to an authenticated request for a *different* request when the
> response says `public` (or `s-maxage`, or `must-revalidate`). With no token in
> the cache key that other request can be an unauthenticated one, so an nginx
> `proxy_cache`, CDN, or corporate proxy in front of a `--remote` deployment would
> serve tiles with the token checked exactly once, for someone else. `private`
> keeps the per-user browser cache that the tiled design actually needs and
> withholds the shared-cache reuse that bearer auth cannot make safe. CDN caching
> would need a different grant in the cache key — signed URLs — not a header
> change.

## Cancellation

The read routes (`/api/tile`, `/api/slice`) check
`request.is_disconnected()` before starting a read and return **499** when the
caller has already hung up, counted as `cancelled_reads` in `/api/diagnostics`.
This reclaims *queued* work only — neither the Flight read nor the dask graph is
interruptible, so a client that leaves mid-compute is not noticed.

Both read via `run_in_threadpool`. That is load-bearing for cancellation, not
just for latency: a read blocking the event loop denies it the turn it needs to
observe other callers' disconnects, which silently defeats the check for every
request queued behind it (`remote-viewer-tiles.md` has the measurements).
`/api/slice` computed on the loop until volumes made that reachable in normal
use — a `scale_policy` read is a whole 3-D block, so the queue behind it can be
seconds long.

## Slice endpoint

**Request body** (`SliceRequest` Pydantic model):

```json
{
  "array_id":         "my-zarr",
  "slice_start":      [0, 0, 0],
  "slice_stop":       [1, 512, 512],
  "scale_hint":       [1, 2, 2],
  "scale_policy":     null,
  "reduction_method": "area",
  "pixel_budget":     1000000
}
```

`array_id` is required and is the whole address — `my-zarr` for a single-tensor
source, `my-zarr/well_A1` for a multi-tensor one. It resolves through the same
lookup the tile routes use, so one id cannot name two tensors depending on which
route asked, and the read is issued for the array_id that lookup came back with
(biopb/biopb#75).

A bare within-source field (`well_A1`) is **not** accepted; send the qualified
form. There is **no `"0"` sentinel** either: a single-tensor source's `array_id`
*is* its `source_id`, so `"0"` addresses a field literally named `0`.

> The route took a `(source_id, tensor_id)` pair until biopb/biopb#766. The split
> was rejoined before every read, and two derivations of one identity could
> disagree. A body carrying the old pair and no `array_id` is a **422**.

**Response:**
- `Content-Type: application/octet-stream` — C-contiguous `numpy.tobytes()`
- `X-Shape: 1,512,512`
- `X-Dtype: uint16`
- `X-Dim-Labels: z,y,x`
- `X-Scale-Hint: 1,2,2` — the per-axis scale actually read at

`scale_hint` and `reduction_method` are forwarded verbatim to
`TensorFlightClient.get_tensor(...)`, which resolves the appropriate
precomputed pyramid level (if available) or applies runtime downsampling.

### `scale_policy` — letting the server choose the scale

`scale_policy: "volume"` reads at the one scale the server keeps a whole 3-D
volume warm at: the Flight ladder's coarsest level, which is what the precache
worker warms and what napari's 3-D mode uploads as a single texture
(`precache-policy.md` §5). `/api/tile_info`'s `volume` block says what it will
resolve to for a given tensor, and a tensor with no volume is a **422** carrying
that block's own `reason`.

It exists because a client cannot compute that level without reimplementing the
server's pyramid planner, and a guess one rung away misses every warmed chunk —
`chunk_id` is `array_id + bounds + scale_hint + reduction_method`, so a
neighbouring scale shares no cache entry and pays a cold read of the source.

`scale_policy` and `scale_hint` are **mutually exclusive** (422): one read has one
scale, and letting the two disagree would make which one applies a silent policy.
`X-Scale-Hint` is echoed either way, but it is load-bearing here — the caller did
not choose, so the header is the only statement of what it got.

## Diagnostics

`_DiagnosticsState` tracks:

| Field | Implementation |
|-------|---------------|
| `latency_p50_ms` / `latency_p95_ms` | `_LatencyTracker` — rolling deque of 200 samples, thread-safe interpolated percentile |
| `last_error_code` / `last_error_message` | Ring buffer of 20 `_ErrorEvent` objects |
| `cache_hit_rate` | Pulled from `TensorFlightClient.cache_info()` on each diagnostics request |
| `connection_state` | `"disconnected"` → `"connected"` or `"error"` |
| Rate limiting | Per-session 1 req/s window, keyed by raw token header value |

All error messages are passed through `_redact()` before storage:
- Filesystem paths matching `/...` or `C:\...` → `[REDACTED]`
- Strings of ≥ 16 URL-safe characters (potential tokens) → `[REDACTED]`

## CORS

Both `create_app(cors_origins=None)` and the CLI launcher default to the loopback
variants of the sidecar's own bind: `http://localhost:8814`,
`http://127.0.0.1:8814`, `http://[::1]:8814` (substituting the actual
`--web-host:--web-port`). That covers the control front reaching the data API
over loopback. No web app is bundled with this package, so there is no frontend
origin in the default set.

A browser app served on another origin must be allowed explicitly: the
`cors_origins` argument to `create_app`, or `--cors` (repeatable) on the CLI
launcher.
