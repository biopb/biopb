# 3-D volume rendering — coarse-level, single-read

**Status:** proposed. Nothing implemented.
**Component:** `web/` (viewer SPA); `biopb-tensor-server` (HTTP sidecar `/api/slice`,
`/api/tile_info`; `core/chunk.py` pyramid).
**Related:** `remote-viewer-tiles.md` (the 2-D tiled viewer this sits beside),
`http-server.md`.

## Why

The viewer browses a z-stack one plane at a time. That answers "what is in this
plane" and nothing else: whether a structure is continuous across z, how two
objects sit relative to each other, how thick something is — each of those needs
the stack seen as a volume. It is also the one view a napari user gives up when
they move to the browser.

**Scope of this document is stage 1: coarse-level only.** One volume, sized to fit
a texture budget, fetched once, then the camera is free. No level-of-detail, no
streaming, no refetch on zoom. That is a complete and useful feature on its own,
and it is close to what the data plane already serves.

## What the data plane already provides

Three facts, all already true, which is why stage 1 needs almost no backend work.

**The pyramid is 3-D aware.** `compute_pyramid_scale_hints` (`core/chunk.py:573`)
downsamples Z alongside X and Y against a *cubic* budget
(`Lx*Ly*Lz <= PRECACHE_PIXEL_BUDGET_CUBIC_ROOT**3`), not a plane budget — the loop
was written for volumes. It is already parameterised on
`threshold` / `downscale_factor` / `pixel_budget_cubic_root`; the `PRECACHE_*`
constants are a default *policy*, not the algorithm.

**Reduction works on any axis.** `downsample_block` (`core/downsample.py:228`)
area-averages over an arbitrary per-axis `scale_hint`, with an exact-integer
accumulator when every factor is a power of two and a float64 path otherwise.
Z-averaging is not a special case, and it is not striding.

**Scaled reads are cached by identity.** `resolve_chunk_data`
(`core/adapter_base.py:740`): scaled chunks are *always* cached when a CacheManager
is present, keyed by `chunk_id` — which carries `array_id + bounds + scale_hint` —
with the reduction method advisory so entries are shared. The expensive half (read
full resolution, reduce) is paid once and persists on disk.

Together: *"the whole stack, quarter-scale in every axis, area-averaged"* is a
first-class, already-cached read.

## What does not reach the browser

**1. The tile ladder is a different ladder, and has to be.** `_tile_levels`
(`http_server.py:684`) synthesises a factor-2, YX-only ladder, and `_tile_slices`
pins `scale_hint[z] = 1` on every tile read (`:802`). This is not drift from the
server's factor-4 z-aware pyramid — it is forced by the consumer. Viv hard-codes
`resolution = Math.round(-z)` into a dense `loader[]` array with
`minZoom = -(loader.length - 1)`: the array index *is* the exponent. The `2 **`
arithmetic appears in the tile bounding box (`@vivjs/layers:849`), the background
layer's model matrix (`:1012`), the overview view scale (`@vivjs/views:174`), and
the 3-D path twice (`:2080`, `:2100`). deck.gl's own quadtree is swappable —
`TileLayer.defaultProps.TilesetClass = Tileset2D`, instantiated at
`tile-layer.js:68` — but Viv's is not, and it sits outside the tileset abstraction
entirely. The two ladders cannot be merged.

**2. The descriptor the sidecar reads carries neither pyramid nor physical scale.**
`_tensor_desc_by_array_id` (`:606`) goes through `client.list_sources()`, i.e.
`ListFlights`, which leaves `pyramid`, `physical_scale` and `metadata_json` empty by
construction — they are `GetFlightInfo` fields. `_tensor_desc_to_dict` (`:2231`)
then publishes only `array_id`/`dim_labels`/`shape`/`chunk_shape`/`dtype`.

**Physical scale is the one gap that blocks 3-D outright.** Without z-vs-xy spacing
there is no aspect ratio, and a typical confocal stack is 3–10x anisotropic — it
renders visibly squashed. Nothing in `web/` mentions physical size today.

**3. Native levels need an exact scale match.** `_plan_precomputed_read`
(`core/adapter_base.py:849`) raises `"No precomputed level matching scale_hint"`
unless the request equals a level's factors exactly, and only when the caller passes
`reduction_method="precompute"` — which the viewer never does. So on a pyramidal
OME-Zarr or QPTIFF, any client-chosen scale reads level 0 and reduces on the fly.
Not stage-1 blocking; it is the difference between "works" and "fast on exactly the
datasets that should be fastest".

## Decisions

### 1. Do not use Viv's `VolumeLayer`

`getVolume` (`@vivjs/layers:1937`) builds the volume by calling
`source.getRaster({selection: {...z}})` once per plane and stitching. Two problems:

- `downsampleDepth: 2 ** resolution` (`:2080`) ties the z stride to the XY level.
  Viv assumes an isotropic dyadic volume pyramid; ours is factor-4 and z-aware, and
  no ladder we can construct will satisfy both it and `loader[]`'s dense indexing.
- It *strides* z rather than pooling it. For a coarse overview that discards
  precisely the signal an overview exists to show.

`VolumeView` is not a way around this — `getLayers` constructs a `VolumeLayer`
(`@vivjs/views:471`). All it otherwise wraps is a deck.gl `OrbitView` (`:449`), so
we take that directly.

### 2. Do use Viv's `XR3DLayer`

`XR3DLayer` is the renderer, `VolumeLayer` is the loader policy. Only the latter is
unusable. `XR3DLayer` takes the volume as a prop and asks nothing about where it
came from:

```
channelData: { data, width, height, depth }   // one entry per channel
contrastLimits, colors, channelsVisible
dtype
xSlice / ySlice / zSlice, clippingPlanes      // free clipping
resolutionMatrix                              // our scale_hint goes here
```

It carries the same `DECKGL_PROCESS_INTENSITY` hook and the same
`_isHookDefinedByExtensions` all-or-nothing rule as `XRLayer` (`:1352`, `:1622`,
`:1642`), through its own `channelIntensity3D` module. So the existing
`GammaExtension` approach carries over — but note `XR3DLayer`'s default
`extensions` is `[ColorPalette3DExtensions.AdditiveBlendExtension()]`, and naming
`extensions` *replaces* the default rather than adding to it. Whether
`VivLayerExtension` is the right base for a 3-D extension needs checking against
`ColorPalette3DExtensions` before building on it.

### 3. Transport is `POST /api/slice` — not a new cacheable GET route

The natural instinct is to mirror `/api/tile`'s ETag discipline with a
`GET /api/volume`. Rejected, for three reasons in descending order of weight:

- **The access pattern does not repeat.** Tiles cache well because pan and zoom
  re-request the same bytes constantly. A volume is fetched once per `(t, c)`,
  uploaded to a texture, and then rotation costs zero requests. The only possible
  hits are reload, toggling `t`/`c` back and forth, and revisiting a source.
- **The server already caches the expensive half** (`resolve_chunk_data`, above).
  HTTP caching would save only the wire copy, which against a loopback sidecar is
  the cheap part.
- **Size.** Tens of MB in a single browser cache entry is where disk caches start
  refusing outright, and a stored one evicts the tile working set for every other
  open source. Active harm, not merely no benefit.

`/api/slice` in exchange gives what the volume actually needs: a **per-axis**
`scale_hint`, so one round trip returns an area-averaged block in the tensor's own
dtype with `X-Shape` / `X-Dtype` / `X-Dim-Labels` — a 3-D texture upload with no
reassembly.

### 4. The client picks the scale, against a browser budget

Do **not** take the server's coarsest level. Its budget is 512³ voxels and each
axis stops shrinking at 512, which is a sensible cache policy and a bad texture:

| tensor (uint16) | full | server coarsest | at a 64 MB client budget |
|---|---|---|---|
| 100 × 2048 × 2048 | 838 MB | `[4,4,4]` → 25×512×512, 13 MB | same |
| 400 × 2048 × 2048 | 3.35 GB | `[1,4,4]` → 400×512×512, **210 MB** | `[2,8,8]` → 200×256×256, 26 MB |
| 60 × 1024 × 1024 | 126 MB | `[1,4,4]` → 60×256×256, 7.9 MB | same |

The z axis is the one that gets away: `lz` below the 512 floor is never reduced, so
a deep stack keeps every plane. The client derives its own per-axis scale from
`tile_info`'s `shape` and `dtype` under a **byte** budget (not a voxel count — dtype
width varies by 4x), divided by the number of visible channels, since `XR3DLayer`
holds one volume per channel. Each axis additionally clamps to the context's
`MAX_3D_TEXTURE_SIZE` (commonly 2048), read from GL rather than assumed.

Prefer powers of two: it is what `downsample_block`'s exact-integer path requires,
and it keeps the door open to landing on an advertised level later.

### 5. Contrast and gamma stay client-side

Unchanged from the 2-D viewer. The existing sampler reads the coarsest level once
per selection and re-derives limits locally; that same sample set serves the volume.
Nothing about 3-D moves the render boundary back to the server.

## Stage 1 — scope

1. `SliceControls` gains a 2-D / 3-D toggle, shown only when the tensor has a z axis
   with extent > 1.
2. In 3-D, the z slider is replaced by the volume; `t`/`c` still select.
3. One `/api/slice` read per visible channel on entering 3-D and on any `t`/`c`
   change, aborted and reissued on change.
4. deck.gl `OrbitView` + `XR3DLayer`, `resolutionMatrix` from the applied
   `scale_hint`, physical scaling matrix from `physical_scale`.
5. Contrast, colour, gamma reuse the existing store and controls.
6. `xSlice`/`ySlice`/`zSlice` clipping if it is cheap; otherwise defer.

Explicitly **not** in stage 1: LOD on camera distance, progressive fill, maximum-
intensity projection as a separate mode, annotation overlays in 3-D.

## Backend changes required

Small, and two of the three are worth doing regardless of 3-D.

1. **Publish `physical_scale` / `physical_unit` on `/api/tile_info`.** *Blocking.*
   Cheap: `client.get_physical_scale(array_id)` (`client.py:260`) reads a cached
   descriptor when one exists, and the server fills these on *every* `GetFlightInfo`
   (issue #31) — no `with_metadata`, no opt-in `with_pyramid` cost. Do not dig them
   out of `metadata_json`; that forces the whole OME tree.
2. **`/api/slice`: move the compute off the event loop and cap the response.**
   `arr_lazy.compute()` runs inline in the async handler (`:1508`), unlike
   `/api/tile` which uses `run_in_threadpool` — and `:1409` already documents why
   that matters: a blocking compute starves the loop of the turn it needs to notice
   that *other* queued callers have hung up, defeating `_abort_if_client_gone` for
   the whole burst behind it. Separately, `pixel_budget` is currently
   "informational, stored in diagnostics" (`:310`) and nothing bounds the response.
   A volume read makes both of these reachable in normal use.
3. **Divisor-based native-level routing** (`_plan_precomputed_read`). Find the
   largest native level whose factors *divide* the request, plan against that
   level's store, apply the residual scale through `downsample_block`. Scale 8
   against native levels 1/2/4 opens level 4 and reduces by 2. This makes ladder
   alignment a performance optimisation rather than a correctness requirement, and
   it is what lets a factor-2 client ladder ride a factor-4 native pyramid. Not
   stage-1 blocking, but it is the largest available win and it benefits the 2-D
   tiled viewer equally.

## On "one source of truth for pyramid shape"

The original intent was a single server-owned pyramid. That is not achievable as
stated, and gap 1 above is why: the consumer's ladder shape is dictated by deck.gl
and Viv (factor 2, dense, indexed from zero), the server's by cache efficiency and
by having to be decidable with **no client present** (precache). Those are different
questions and cannot produce the same list.

What should be single-sourced is the *facts* — which levels are native, which are
warm, what the physical scale is — with reconciliation in the **read path**
(change 3) rather than in the metadata. Then the client picks whatever ladder its
renderer demands and the server always serves it from the best thing it has.

Corollary: do **not** parameterise `_tile_levels` on a client budget.
`/api/tile?level=N` indexes into the published list and `_resolve_tile_level` gates
on exactly that list, deliberately, "so the two cannot disagree about what exists".
Make the ladder depend on a caller-supplied budget and `N` stops being interpretable
without it, and `_tile_etag` has to carry it. Today the ladder is a pure function of
`(shape, chunk_shape)`, which is what makes level indices and ETags stable across
clients and sessions. The volume path needs no ladder at all — it sends an explicit
per-axis `scale_hint`.

## Deferred, with the trigger that would revive each

- **`GET /api/volume`** — if volumes turn out to be re-requested often enough to
  measure. Revisit only with a hit-rate number.
- **Slab reads (a `zdepth` parameter on `/api/tile`)** — the right answer if we want
  *progressive* fill or a WAN deployment where one 30 MB body is too slow. A few MB
  per slab caches at a size browsers accept, revalidates granularly, and fills the
  texture incrementally, all under the ETag machinery that already exists. This, not
  a monolithic volume route, is the cacheable design.
- **LOD on camera distance** — needs a second volume in flight and a swap policy.
  Only worth it if stage 1's fixed coarse level proves too coarse in practice.
- **Maximum-intensity projection** — cheap once the volume is on the GPU (a shader
  variant), and often what people actually want. Deliberately held back so stage 1
  ships one thing.

## Open questions

- Does `VivLayerExtension` compose with `XR3DLayer`, or does the 3-D path need a
  `ColorPalette3DExtensions`-derived base? Decides whether gamma is free.
- What byte budget? 64 MB is a guess; it should be measured against upload time on
  the slowest GPU we care about, not chosen from texture limits.
- Anisotropic tensors with no `physical_scale` at all (many TIFFs): render isotropic
  with a visible caveat, or refuse 3-D? Leaning toward rendering with a caveat.
- Does `t`/`c` switching in 3-D want the previous volume held while the next loads
  (memory) or a blank (latency)?
