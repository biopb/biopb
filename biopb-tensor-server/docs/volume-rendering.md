# 3-D volume rendering — coarse-level, single-read

**Status:** stage 1 implemented. Decision 4 was reversed on the way in — the
server picks the scale, not the client; see that section.
**Component:** `web/` (viewer SPA); `biopb-tensor-server` (HTTP sidecar `/api/slice`,
`/api/tile_info`; `core/chunk.py` pyramid).
**Related:** `remote-viewer-tiles.md` (the 2-D tiled viewer this sits beside),
`precache-policy.md` (which scale is warm, and why that is the one to read),
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
(`core/adapter_base.py`): scaled chunks are *always* cached when a CacheManager is
present, keyed by `chunk_id` — which carries `array_id + bounds + scale_hint +
reduction_method`. The expensive half (read full resolution, reduce) is paid once
and persists on disk.

That the *method* is part of the identity too (#578, no longer advisory) is what
makes decision 4 below more than a preference: a read has to match the warm entry
on both the scale and the method to hit it at all.

Together: *"the whole stack, quarter-scale in every axis"* is a first-class,
already-cached read.

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

**2. ~~The descriptor the sidecar reads carries neither pyramid nor physical
scale.~~ No longer true, and that is what unblocked this.**
`_tensor_desc_by_array_id` now resolves through `client.get_descriptor()`, i.e.
`GetFlightInfo`, so `physical_scale` and `physical_unit` are on the descriptor
every sidecar route already holds. (`pyramid` still is not — it is behind
`with_pyramid`, which this route does not ask for. The volume path needs no
advertised pyramid: it computes the plan from the same planner the precache
worker uses.)

**Physical scale was the one gap that blocked 3-D outright.** Without z-vs-xy
spacing there is no aspect ratio, and a typical confocal stack is 3–10x
anisotropic — it renders visibly squashed. `/api/tile_info`'s `volume.spacing`
carries it now.

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

### 4. ~~The client picks the scale~~ — the server does

**Reversed.** The original argument was that the server's coarsest level was a
cache policy and a bad texture: a 512³ voxel budget with a per-axis floor of 512,
so a deep stack kept every plane and a `[400, 2048, 2048]` came back at 210 MB.

Two things changed under it. The budget is now 448³, measured against a real GPU
rather than assumed (`precache-policy.md` §9.1), and the ladder's coarsest level
is now the level the precache worker **warms whole** — that is what the 3-D
target in §5 of that document is for, and N1 (napari reads the coarsest level
whole) is why it is sized as a texture in the first place.

That makes the level the client would have derived for itself the wrong one to
ask for. A client-chosen scale one rung away from the warm one shares no
`chunk_id` with it — the id carries `scale_hint` — so it misses every warmed
chunk and pays a cold read of the source to produce a blurrier picture. The
server's level is not merely acceptable; it is the only one that is cheap.

So `POST /api/slice` gained `scale_policy: "volume"`: the caller sends the region
and the server resolves the scale, echoing what it used in `X-Scale-Hint`. There
is no way to compute that client-side short of reimplementing the pyramid
planner, and a reimplementation is exactly the drift this avoids —
`biopb-mcp`'s copy of the ladder had already gone stale once.

`/api/tile_info` carries a `volume` block saying what the policy will resolve to:
the three axis indices, the scale, the extents, the wire size, and `spacing` (the
source's physical size already multiplied by the scale). Or `available: false`
with a reason, which is the useful answer for most of the catalog.

**The browser budget survives, as a refusal rather than as a scale.** The server
bounds voxels, not bytes, and Viv uploads every volume as Float32 regardless of
dtype — so VRAM follows the voxel budget, but the wire and the transient
`Float32Array` follow the dtype. A float64 tensor at 448³ is ~720 MB on the wire
and 1.4 GB of live heap before upload. `VOLUME_MAX_BYTES` (512 MiB) refuses that
in the viewer, where the browser-specific limit belongs. Nothing the catalog
actually holds comes near it: 448³ is 180 MB at uint16.

### 5. Contrast and gamma stay client-side

Unchanged from the 2-D viewer. The existing sampler reads the coarsest level once
per selection and re-derives limits locally; that same sample set serves the volume.
Nothing about 3-D moves the render boundary back to the server.

## Stage 1 — as shipped

1. `SliceControls` gains a 2-D / 3-D toggle — **unconditional**, not gated on the
   tensor having a z axis. Whether it has one is `tile_info.volume`'s answer, and
   the 3-D pane is what fetches that; gating the button would mean a second
   `tile_info` call on every source, made only to grey something out. A refusal
   instead shows the server's own reason and a "Show 2D" button.
2. In 3-D the z slider is dropped — the volume *is* that axis, read whole — and
   `t`/`c` still select.
3. One `/api/slice` read on entering 3-D and on any `t`/`c` change, aborted and
   reissued. Keyed on the resolved request rather than on the store slice, so a
   contrast drag or a colour change cannot re-issue hundreds of MB.
4. deck.gl `OrbitView` + `XR3DLayer`. `physicalSizeScalingMatrix` carries the
   anisotropy; `resolutionMatrix` is **identity**, not the applied `scale_hint` —
   the world here is this volume's own voxels, where Viv's `VolumeLayer` needs
   `scale(2 ** resolution)` because its world is level-0 pixels.
5. Contrast and colour reuse the existing store and controls. **Gamma does not:**
   it is a `VivLayerExtension` on the 2-D colour path, and the 3-D path is
   `ColorPalette3DExtensions`. Answering the open question below would make it
   free; until then the 3-D pane renders at gamma 1.
6. `xSlice`/`ySlice`/`zSlice` clipping — deferred. `XR3DLayer` takes them as
   props, so it stays cheap whenever it is wanted.

Explicitly **not** in stage 1: LOD on camera distance, progressive fill, maximum-
intensity projection as a separate mode, annotation overlays in 3-D.

## Backend changes required

Small, and two of the three are worth doing regardless of 3-D.

1. ~~**Publish `physical_scale` / `physical_unit` on `/api/tile_info`.**~~ **Done**,
   as `volume.spacing`: the source's physical size *times the plan's own scale*,
   in z/y/x order. The product rather than the two factors, because a renderer
   needs the anisotropy ratio and the two factors are in different orders —
   multiplying the wrong pair stretches the volume with nothing to show for it.
2. ~~**`/api/slice`: move the compute off the event loop**~~ **Done.** It now
   reads through `run_in_threadpool` and re-checks `_abort_if_client_gone` after
   resolving the geometry, exactly as `/api/tile` does. The response is still
   uncapped server-side; the bound is the pyramid's voxel budget on one side and
   the viewer's `VOLUME_MAX_BYTES` refusal on the other (decision 4).
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
  `ColorPalette3DExtensions`-derived base? Still open, and it is what decides
  whether gamma reaches 3-D.
- ~~What byte budget?~~ Moot as posed: the scale is no longer chosen against a
  budget. `VOLUME_MAX_BYTES` is a refusal threshold, and at 512 MiB it only ever
  fires on float64 — it is not a tuning knob.
- ~~Render isotropic with a caveat, or refuse?~~ **Render, with the caveat
  visible.** `volume.spacing` is null when the source declares no physical size,
  the ratio falls back to `[1, 1, 1]`, and the pane's badge says
  "isotropic (no physical scale)". Refusing would take 3-D away from most TIFFs
  over a fact the source failed to record.
- Does `t`/`c` switching in 3-D want the previous volume held while the next loads
  (memory) or a blank (latency)? Currently a blank, matching the 2-D pane's
  "Reading plane…" cover — a stale volume under a new `t` is indistinguishable
  from the right one.
