# Pre-cache policy

**Status:** design. Current behaviour is `serving/precache.py`; this document is
what replaces it. Supersedes the earlier admission-control proposal, whose
capacity apparatus (retention classes, per-source and class budgets, tri-state
`enabled`) is retired below rather than implemented.

## 1. Why the warm misses

Precache warms a level no browser client ever requests. Three independent
breaks, any one of which is sufficient:

1. **Wrong ladder.** The server carries *two* pyramid ladders.
   `build_pyramid_plan` (`core/chunk.py`) is advertised on
   `TensorDescriptor.pyramid` and read by napari; `_tile_levels`
   (`serving/http_server.py:688`) is published on `/api/tile_info` and read by
   Viv. Precache warms the first; the SPA reads the second.
2. **Wrong level.** On a 14234² ND2 scene the Flight ladder's coarsest is
   scale 4 (3559²); the tile ladder's is scale 32 (445²).
3. **Wrong reduction.** `_DEFAULT_REDUCTION_METHOD = "nearest"`
   (`core/downsample.py:29`), the tile route passes
   `reduction_method: Optional[str] = Query(None)` (`http_server.py:1391`), and
   `viv-source.ts` never sets it — so every tile asks for `nearest`. Precache
   warms `PyramidConfig.reduction_method`, default `"area"`
   (`core/config.py:549`). Since #578 the method is a mandatory chunk_id byte,
   so this alone is a total miss.

napari is unaffected: it honours the advertised levels
(`_tensor_utils._advertised_pyramid_levels`) and opens at the coarsest
(`scalar_field.py:279`), which is what precache warms. **The browser client is
the entire defect.**

## 2. What already works

The chunk grid is absolute at a given scale: `virtual_chunk_size =
transfer_chunk_size * scale_hint` clamped to the tensor (`core/chunk.py:834`),
and `_get_read_plan` snaps `realized_start` to that grid from origin 0
(`core/adapter_base.py:1571`). Neither depends on `slice_hint`.

So a tile read at scale *s* mints chunk_ids that are a strict subset of a
whole-tensor plan at scale *s*. **The cache is already shared; only the level
and the method disagree.** No change to the codec, the grid, or the cache is
needed.

## 3. Constraints

### 3.1 Viv / deck.gl — binding on `_tile_levels`

Verified against `@vivjs/layers@0.22.1` and `@deck.gl/geo-layers@9.3.10`.

- **V1. Level *i* is scale 2^*i*, contiguous, level 0 full resolution.**
  `resolution = Math.round(-z)` indexes `loader[]` (`index.mjs:951`);
  `renderSubLayers` derives world bounds from `scale = 2 ** Math.round(-z)`
  (`:846`); the background layer places the coarsest with
  `modelMatrix.scale(2 ** (loader.length - 1))` (`:1012`).
- **V2. X and Y share one scale.** `renderSubLayers` multiplies both
  `data.width` and `data.height` by one scalar; deck.gl's `getIndexingCoords`
  maps both axes through one `tileSize` (`tileset-2d/utils.js:151`), so a tile
  is square in world units. **Viv cannot express a Z-scaled level at all.**
- **V3. One `tileSize` for the whole ladder** — `loader[0]`'s (`:946`).
- **V4. Y and X are the last two axes** of each level's `shape` (last two before
  the samples axis when interleaved): `getImageSize` slices positionally and
  ignores `labels`. Holds only because the served order is canonical (#596).

Not constraints, though they look like them:

- **Ladder depth is free.** `minZoom = -(loader.length - 1)` and deck.gl clamps
  (`z = minZoom`; the `return []` branch needs a missing `extent`, and Viv always
  passes one). Truncating renders the coarsest upscaled; extending is free.
- **Tiles need not be full `tileSize²`** — bounds come from the returned
  `data.width`/`data.height`.
- **"Coarsest is one tile" is ours.** `viv-source.ts:470` routes `getRaster` to
  the cacheable tile route only when `cols === 1 && rows === 1`, else to
  uncacheable `POST /api/slice`. Viv only requires `getRaster` to return the
  whole plane. Kept deliberately (§4.2), because synthesis makes it free.

Viv's 3-D has **two** paths, and only the packaged one is tied to this ladder:

- **`VolumeViewer` / `VolumeView`** hardcodes `new VolumeLayer(...)`
  (`@vivjs/views` `:471`), and `VolumeLayer` reads `loader[resolution]` and calls
  `getVolume` (`:1937`) for `Z / 2**resolution` rasters, decimating Z
  client-side. Same chunk_ids as XY level *r*, so the 2-D warm target covers it.
- **`XR3DLayer`** is the actual renderer and has **no `loader` prop**. It takes
  `channelData: {data, width, height, depth}` plus `dtype`,
  `resolutionMatrix` and `physicalSizeScalingMatrix` as plain props, and is
  exported from `@hms-dbmi/viv`. Driven directly inside an `OrbitView` it
  renders one volume fetched any way we like — no ladder, no dyadic or isotropic
  constraint (V1/V2 bind `_tile_levels`, not this), one request instead of
  `Z / 2**r`.

**Decision: `XR3DLayer` direct, not `VolumeViewer`.** The SPA wires neither
today, so nothing is being migrated. Driven directly the browser consumes the
*same* coarsest level napari 3-D reads, which makes §5's 3-D target serve both
clients instead of napari alone; via `VolumeViewer` browser 3-D would instead be
locked to the tile ladder, with client-side Z decimation and `Z / 2**r` separate
raster requests for a volume the server can return in one.

The cost is owning the view wiring — an `OrbitView` plus the two matrices
`VolumeLayer` computes (`resolutionMatrix`, `physicalSizeScalingMatrix`) and at
least one extension, since `getRenderingFromExtensions` throws unless one defines
`rendering._RENDER` (`VolumeLayer` relies on the default `AdditiveBlendExtension`).

**Volumes are uploaded as Float32 regardless of source dtype**: `dataToTexture`
(`:1913`) builds a `dimension: "3d"` texture with `data: attrs.cast(data)`, where
`getRenderingAttrs()` is `getDtypeValues("Float32")` with
`cast: d => new Float32Array(d)`. So 4 B/voxel of VRAM per channel, plus a
transient `Float32Array` of the same size during upload.

### 3.2 napari — binding on `build_pyramid_plan`

- **N1. 3-D always reads the coarsest level, whole.**
  `_slice.py:249` (`level = len(self.data) - 1` when `ndisplay == 3`), confirmed
  again at `base.py:2163`. Not adjustable. **This is what
  `pixel_budget_cubic_root` exists for** (#29).
- **N2. 2-D opens at the coarsest** (`scalar_field.py:279`), then steps by canvas
  scale.
- **N3. Irregular ladders are accepted.** `MultiScaleData.__init__`
  (`layers/_multiscale_data.py:34`) validates only non-empty +
  `LayerDataProtocol`. `downsample_factors = np.divide(level_shapes[0],
  level_shapes)` (`scalar_field.py:408`) is per-axis and float;
  `compute_multiscale_level` is elementwise (`layer_utils.py:598`); 2-D
  selection consults `downsample_factors[:, displayed_axes]` only
  (`base.py:2133`), so Z anisotropy is invisible to it.
  The one requirement is **per-axis non-decreasing factors**
  (`layer_utils.py:590`), which `locations[-1]` depends on.
  *Code-read, not yet tested against a live napari — see §9.*

## 4. Two ladders, deliberately

V2 and N1 are irreconcilable: Viv cannot index a Z-scaled level, and napari's
3-D read must be bounded in Z. They are not two implementations of one policy.
Keep both, each with a named consumer.

| | consumer | rungs | stops at |
|---|---|---|---|
| **Flight** `build_pyramid_plan` | napari 2-D + 3-D, **and browser 3-D** (§3.1) | XY-only, then one final XYZ rung | voxel budget |
| **tile** `_tile_levels` | Viv 2-D | XY-only, dyadic | plane ≤ tile edge (rungs below the warm target synthesized, §4.2) |

Browser 3-D leaves the tile ladder entirely: `XR3DLayer` takes a volume, not a
`PixelSource[]`, so V1/V2 do not reach it. That is what lets one 3-D target
serve both clients.

### 4.1 Flight ladder shape

Today the planner downsamples X, Y and Z together toward a shared floor from the
first rung. That scales Z on rungs that only ever display one slice — pure loss.

New rule: **XY-only rungs until the plane fits the 2-D cap, then one final rung
that scales X, Y and Z toward the voxel budget** (the existing loop), subject to
per-axis monotonicity against the last 2-D rung (N3).

On `[1000, 6000, 6000]`:

| | L1 | coarsest |
|---|---|---|
| today `[1,1,1] [4,4,4] [4,16,16]` | `[250,1500,1500]` — Z already ÷4 | `[250,375,375]` |
| new `[1,1,1] [1,4,4] [4,16,16]` | `[1000,1500,1500]` — Z intact | `[250,375,375]` |

The final rung must stay near-isotropic rather than merely adding Z to the 2-D
rung: on a `[1024,1024,1024]` cube, "add Z" gives `[128,1024,1024]` — same voxel
count, 8:1 sampling anisotropy, texture spent on XY detail 3-D cannot show.

### 4.2 Tile ladder: synthesize the rungs below the warm target

The tile ladder keeps its "stop when the plane fits one tile" rule, because that
is what keeps Viv's unconditional background `getRaster` on the cacheable tile
route (§3.1). But that coarsest rung is far coarser than the 2-D warm target —
scale 32 vs scale 8 on the ND2 — so warming one of the two leaves the other cold
or uncacheable.

Resolution: warm only the 2-D target, and have the tile route **derive the rungs
below it in-process** by reducing the warm level, rather than asking the data
plane for a separate scaled read.

- The coarsest rung stays 1×1 tile, so the background raster stays cacheable.
- The fit-to-view tiles are the warm level itself.
- One warm target serves both open-time reads.
- Only the rungs between the warm target and the coarsest are synthesized — two
  on both the ND2 and the WSI.

**This is exact, given §6.** `nearest` is a strided pick, so
`data[::32] == data[::8][::4]` identically, ragged extents included — verified
across shapes with and without remainders. Under `area` the same composition
differs on 6–10% of pixels (edge padding, then re-averaging), which would have
made this an approximation the rest of the codebase refuses. The two decisions
were made independently and happen to compose.

The synthesized rungs mint no chunk-cache entry; each request re-reduces a plane
of a few Mpx from warm chunks, under the tile route's existing ETag. Reuse
`core/downsample.downsample_block` rather than writing a second reducer.

## 5. Warm targets

Two, gated **independently**. A tensor may qualify for either, both or neither.

| target | level | extent warmed | why that extent |
|---|---|---|---|
| **2-D** | rung whose plane first fits the cap | **one plane** | a 2-D renderer shows one plane |
| **3-D** | Flight ladder's coarsest | **whole volume** | N1: napari reads the level whole; `XR3DLayer` uploads it as one texture |

### 5.1 Gates

```
2-D:  plane_pixels = shape[y] * shape[x]  >  pyramid.plane_max_pixels    (4 Mpx)
3-D:  shape[z] > 1  and  shape[z]*shape[y]*shape[x] > pyramid budget     (448³, §9.1)
```

Neither implies the other:

| tensor | plane | 2-D | 2-D warm | volume | 3-D | 3-D warm |
|---|---:|:--:|---:|---:|:--:|---:|
| cube `[1024,1024,1024]` | 1.05 Mpx | skip | — | 1074 Mvox | **warm** | 32.0 MiB |
| ND2 `[Z1,14234²]`×C4 | 202.6 Mpx | **warm** | 6.0 MiB | 202.6 Mvox | skip | — |
| WSI `[Z1,100000²]` u1 | 10000 Mpx | **warm** | 2.3 MiB | — | skip | — |
| confocal `[Z200,2048²]`×C2 | 4.19 Mpx | **warm** | 2.0 MiB | 839 Mvox | **warm** | 100.0 MiB |
| lightsheet `[Z1200,2048²]`×C2 | 4.19 Mpx | **warm** | 2.0 MiB | 5033 Mvox | **warm** | 150.0 MiB |
| big3D `[1000,6000,6000]` | 36.0 Mpx | **warm** | 4.3 MiB | 36000 Mvox | **warm** | 67.1 MiB |
| thin stack `[Z40,1024²]`×C3 | 1.05 Mpx | skip | — | 41.9 Mvox | skip | — |
| timelapse `[Z1,1024²]`×T500C2 | 1.05 Mpx | skip | — | — | skip | — |

The cube is the case that forces the split: its 2-D slices are already cheap, but
3-D cannot render 2 GiB without a downscale, and computing that downscale *is*
the expensive read. Warming it is 32 MiB against a 2 GiB source.

### 5.2 The cap is set by the viewport, not by disk

4 Mpx = 2048² is what makes the 2-D warm target coincide with the level deck.gl
requests at fit-to-view (`z = Math.ceil(viewport.zoom + zoomOffset)`) for a
~1500–2000 px window. On the ND2 and the WSI the two agree exactly.

Do **not** lower it to save space. The cap is also the admission gate — the skip
fires when the coarsest rung is full resolution — so lowering it admits *more*
tensors. At 1 Mpx the `[T500,C2,1024²]` timelapse goes from skipped (0 MiB) to
warmed (500 MiB), while the ND2 saves 18 MiB.

### 5.3 Selection axes

A rung is XY-only, so **every non-XY axis is a selection axis** — T, C and, on the
2-D target, Z. Warm one combination, not the cross-product. Without this, "warm
the 2-D rung" on `[1000,6000,6000]` means 1000 planes at 1500² = 4.3 GiB
instead of 4.3 MiB, and the `[T500,C2,2048²]` timelapse costs 2000 MiB instead
of 2.0 MiB.

The 3-D target is exempt on Z by N1 — the renderer consumes the volume whole —
but still warms one T/C combination.

## 6. Reduction method: `nearest`

Set `PyramidConfig.reduction_method = "nearest"`.

- It is what the wire already defaults to (`downsample.py:29`), and what every
  tile request asks for. `/api/tile_info` carries no reduction field, so the
  browser client cannot honour an advertised method even in principle.
- It is correct for label images. Averaging segmentation labels produces values
  that name no object; `nearest` returns only real labels.
- On `MrcAdapter`, `NiftiAdapter` and `NikonAdapter` it also costs what it
  returns rather than what it spans (`get_decimated_data`,
  `tests/decimated_read_test.py:25`). Every other adapter decodes whole blocks
  and saves nothing.

Change it on `PyramidConfig`, not inside precache: that field drives both the
advertised `PyramidLevel.reduction_method` and the warm, and napari honours the
advertised value. Overriding it in precache alone would fix the browser by
breaking napari.

**Accepted tradeoff:** `nearest` aliases. Thin structures at scale 8–32 flicker
where `area` antialiases them. Reduction is really a per-tensor property (label
vs intensity) and the server has no signal to tell them apart, so the global
default picks which failure it prefers. Inventing label values is the worse one.

**Cleanup while there:** `CHUNK_ID_IMPLICIT_REDUCTION_METHOD = "area"` sits 13
lines from `_DEFAULT_REDUCTION_METHOD = "nearest"` in the same module. #578 made
the byte mandatory so the legacy decode is dead, but two constants named "the
default" with opposite values will be miswired eventually.

## 7. What this retires

The previous proposal's capacity apparatus assumed a warm set that does not fit
the cache. With §5.3 in place every row above is single- to low-triple-digit MiB,
so:

- **P3 retention class / `cache.derived_reserve`** — dropped. Nothing to protect.
- **P4 tri-state `enabled` / `min_cache_bytes` / class budgets** — dropped.
- **P1 plane gate** — kept, as `pyramid.plane_max_pixels`, but it is now the ladder's own
  stopping rule rather than a separate knob.
- **P2 selection restriction** — kept and widened to Z on the 2-D target (§5.3).
  It is the load-bearing one.
- **P5 `allow_deferred=False`** — kept, unchanged and still needed.
- **`compose=True` in precache** — moot. It existed to make a coarse warm also
  populate full resolution; the 2-D target is now the level clients read.

`enabled` stays a plain bool. It should default **true** again once §5 lands —
it was flipped to false (#826) because the warm set was unbounded, which is the
thing §5.3 fixes.

## 8. Config

```jsonc
"pyramid": {
  "reduction_method": "nearest", // §6
  "downscale_factor": 2,         // dyadic rungs; matches the tile ladder
  "pixel_budget_cubic_root": 448,// 3-D gate + target (§9.1)
  "plane_max_pixels": 4000000    // 2-D rungs: cap, target and gate (§5.2)
},
"precache": {
  "enabled": true,
  "warm_selection": "first",     // §5.3: "first" | "all" over T/C (and Z in 2-D)
  "warm_3d": true,               // §5: independent 3-D target
  "defer_writes": false          // P5
}
```

`pyramid.pixel_budget_cubic_root` keeps its meaning and becomes load-bearing: it
is the 3-D gate *and* the 3-D target, because N1 makes the coarsest level the
thing both renderers upload as a texture. **Set it to 448** (from 512) — see
§9.1.

## 9. To measure before committing

- ~~The 512³ budget against a real GPU.~~ **Measured — see §9.1.** Result:
  set `pixel_budget_cubic_root = 448`.
- **N3 empirically.** Build an anisotropic ladder (`[1,1,1] [1,4,4] [4,16,16]`)
  and open it in a live napari in both 2-D and 3-D.
- **`nearest` warm cost on ND2.** `get_decimated_data` should make it much
  cheaper than `area`, but a Y-stride skips rows while an X-stride inside a row
  faults the same pages — likely closer to the Y factor than the product.
- **Whether the tile ladder should stay at factor 2.** `locations[-1]` in
  napari and `Math.ceil(zoom)` in deck.gl both select the coarsest level still
  above threshold, so a ×4 gap can load up to 16× the canvas pixels.

### 9.1 Measured: the 3-D budget is 448, not 512

Quadro P2000 (Pascal, 5 GiB, ~4.1 GiB free), vispy `VolumeVisual` `method="mip"`
with `texture_format="auto"` — the path napari uses
(`_vispy/layers/image.py:130`) — offscreen EGL, 1200×900, single channel,
random-noise volumes.

| volume | Mvox | uint16 fps | float32 fps | f32 VRAM |
|---|---:|---:|---:|---:|
| 384³ | 56.6 | 100 | **89** | 216 MiB |
| 448³ | 89.9 | 83 | **50** | 343 MiB |
| 512³ *(old budget)* | 134.2 | 63 | **17** | 516 MiB |
| 640³ | 262.1 | 32 | **2.5** | allocation spilled to host |

Three findings:

1. **Render speed is not the constraint; dtype is.** Frame time scales with the
   longest dimension (ray steps), not voxel count — 64× the voxels from 128³ to
   512³ costs only 3× the frame time, and confocal `[200,512,512]` (52 Mvox,
   6.5 ms) beats 384³ (57 Mvox, 10.0 ms) because a slab has a shorter ray path
   than a cube. What breaks is **float32**, which falls off a cliff between
   343 MiB and 516 MiB.
2. **The two renderers have different headroom.** napari keeps native dtype, so
   a u16 volume is 2 B/voxel; Viv casts to Float32 unconditionally (§3.1) at
   4 B/voxel. The same level costs the browser 2× the VRAM and hits the wall
   ~2.3× sooner. **The budget must be set by the browser**, which also runs on
   weaker hardware than the machine holding the data.
3. **Upload is not a constraint**: 0.29 s at 512³. (An earlier run showing
   1.4–2.0 s was measuring vispy's legacy uint8-normalize path, not transfer.)

`448` puts the ceiling at 90 Mvox / 343 MiB float32 — 50 fps on this card, ~2×
headroom on napari's u16 path. It changes **no current shape**: confocal,
lightsheet, big3D and the cube already land at 17–79 Mvox. It is purely a
tripwire so a future dataset cannot land at 134 Mvox and render at 17 fps.
`400` was also tried and over-corrects, clipping lightsheet 4× for no measured
gain (79 Mvox already renders at 52 fps).

**Caveats.** Noise volumes defeat early ray termination, so these are
conservative against real sparse data — deliberate, since a P2000 is roughly the
floor for a discrete workstation card. Single channel: `XR3DLayer` uploads one
texture per channel, so a 2-channel confocal is 400 MiB, not 200. And this
measures only the *ceiling* — the floor (is the specimen still legible at that
downscale?) needs eyes on real data and is not yet done.

## 10. Order of work

0. ~~GPU measurement~~ — **done, §9.1**: set
   `pyramid.pixel_budget_cubic_root = 448`. One-line change, no current shape
   moves.
1. **§6 `nearest`** — one constant, independent of everything else, fixes a
   total miss on its own.
2. **§5.3 selection restriction** — bounds the warm set; unblocks re-enabling
   `enabled` by default.
3. **§4.1 Flight ladder shape** + **§5 two targets with independent gates**.
4. **§4.2 tile-ladder alignment** — the 2-D target must be a level
   `_tile_levels` publishes, and the rungs below it become synthesized. Depends
   on §6 for exactness, so it lands after it.

## 11. Validation

- The §5.1 table as a gate/footprint parametrisation: assert the verdict and the
  bytes warmed per row, both gates independently.
- A `[1024,1024,1024]` cube asserts 2-D skip + 3-D warm; an ND2 scene asserts
  2-D warm + 3-D skip.
- Warm a tensor, then issue the browser client's exact tile request, and assert a
  cache **hit** — the end-to-end property all of §1 is about. Pin the
  reduction byte explicitly so a default flip cannot silently un-fix it.
- §4.2 synthesis is bit-identical to a direct read at the same scale, on extents
  with and without remainders. Assert it *fails* under `area`, so the test
  records the coupling to §6 rather than silently passing if the method flips.
- P5: a precache warm leaves `deferred_write_bytes` at zero while a concurrent
  live read still defers.
