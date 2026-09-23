# Pre-cache policy

Scope: `biopb-tensor-server`'s background warming worker (`serving/precache.py`)
and the two pyramid ladders it and the HTTP sidecar reconcile.

## Two ladders, one cache

The server carries two pyramid ladders, because their consumers have
irreconcilable constraints:

| | consumer | rungs | stops at |
|---|---|---|---|
| **Flight** `build_pyramid_plan` (`core/chunk.py`) | napari 2-D + 3-D, and browser 3-D | XY-only, then one final XYZ rung | voxel budget |
| **tile** `_tile_levels` (`serving/http_server.py`) | Viv 2-D (the SPA) | XY-only, dyadic | plane fits one tile |

Viv/deck.gl cannot express a Z-scaled level (`renderSubLayers` scales X and Y
by one shared factor), while napari's 3-D read is always the coarsest level,
whole (`ndisplay == 3` reads `len(self.data) - 1`). Neither ladder can serve
the other's consumer, so both are kept, each with a named client.

**The load-bearing insight: the chunk grid is absolute at a given scale.**
`virtual_chunk_size = transfer_chunk_size * scale_hint`, clamped to the
tensor, and the read plan snaps every window to that grid from origin 0 --
independent of `slice_hint`. So a tile read at scale *s* mints chunk_ids that
are a strict subset of a whole-tensor plan at scale *s*: the cache is already
shared between the two ladders, only the level and the reduction method can
disagree. Precache warms the Flight ladder's targets; the tile route reads
the same warm chunks by construction (below), so nothing about the two
ladders requires warming twice.

## The worker

`PrecacheWorker` (`serving/precache.py`) warms the file cache so the first
view a client opens is already resident. It serves two tiers, in strict
priority order:

- **Live (primary).** Sources added to the catalog after startup, fed by
  `SourceManager`'s commit hook. Always drained first, and always warmed.
- **Backlog (secondary).** Local sources already present at startup, seeded
  once at boot and ordered newest-mtime-first. Drained only when the live
  queue is empty.

Before each chunk the worker waits for `idle_debounce_seconds` of Flight
server quiet (no in-flight `do_get`), re-checking between chunks so a live
read preempts it at chunk granularity. Reads also serialize behind live
reads through each source's `_io_lock`, so precache never races a
non-thread-safe adapter.

**Backlog is bounded so it never evicts live data.** The file cache evicts
globally on every write, so the backlog tier stops warming once the cache
passes `backlog_high_water` (0.8) of its budget, and yields the instant a
live source is enqueued. A backlog pass preempted mid-tensor is requeued
newest-first; re-warming is cheap since the interrupted chunks are already
hits.

A tensor whose coarsest advertised level is already full resolution is
skipped outright -- warming it would cache the source 1:1 and save an open
nothing. A tensor with a native on-disk pyramid is also skipped: it already
serves overviews from its own store cheaper than a computed level, and
warming it would evict a computed chunk (expensive to reproduce) to hold a
native one (cheap to re-read). A remote-proxy source is skipped too; the
real read path caches its chunks on demand and the upstream caches
independently.

## Warm targets

Two targets, gated independently. A tensor may qualify for either, both, or
neither:

| target | level | extent warmed | why |
|---|---|---|---|
| **2-D** | coarsest rung whose plane fits `pyramid.plane_max_pixels` (4 Mpx) | one plane | a 2-D renderer shows one plane |
| **3-D** | Flight ladder's coarsest rung | whole volume | napari's 3-D read and `XR3DLayer` both consume the coarsest level whole (bounded by `pixel_budget_cubic_root`, below) |

The Flight planner scales X and Y toward the 2-D cap first, then emits one
final rung that scales X, Y and Z together toward the voxel budget
`pyramid.pixel_budget_cubic_root` -- so a tensor with no Z axis
worth downsampling never pays anisotropic loss on Z-only rungs it will never
display in 2-D. The plan is sparse: full resolution plus at most these two
targets. Intermediate halving steps are walked but never emitted or warmed,
because an unwarmed intermediate level costs a client the same page-in as
level 0 and buys it nothing sharper than what interpolating the two emitted
levels would.

`4 Mpx` (2048^2) is set by the viewport, not by disk: it is the level
deck.gl's fit-to-view request lands on for a ~1500-2000 px window, so the
2-D warm target coincides with what the browser actually asks for at open.
It is also the admission gate for the 2-D target -- lowering it would admit
more tensors, not save space, since tensors already below the cap have
nothing to skip.

### Selection axes: a per-level byte budget

A warm level is a plane (or, for the 3-D target, a volume) repeated across
every other axis -- T, C, an unnamed plate/sequence axis. Warming that whole
cross-product is what would make the warm set unbounded (a `[1000,6000,6000]`
volume's 2-D target is 4.3 GiB at one plane per Z). Instead each level is
capped at `precache.warm_budget_bytes` (256 MiB): while a level is over
budget, `compute_warm_selection` narrows selection axes one at a time, in
order:

1. **unnamed axes** (a plate's `POS`, a sequence's `i`) -- independent
   acquisitions; a viewer opens on one of them.
2. **T** -- frames are independent and a user views very few at once.
3. **Z** -- scrubbed, but locally, so a contiguous slab keeps its value.
4. **C** -- last; typically 2-4 channels, cheap to keep whole, and toggling
   channels is the first thing a user does after opening.

The plane itself is never narrowed. Each narrowed axis is cut to the largest
extent that still fits the budget, floored at 1, and the window always
starts at index 0 -- not because that's an arbitrary convention, but because
it's where every client actually opens: napari's raw default centres the
dims slider, but the tensor browser suppresses that second move
(`_tensor_utils._origin_initial_view`) so the decoded slice and the
displayed slice are the same one, and the SPA's `store.ts` re-zeros on every
source change. All three agree on 0 because none of them can ask the server
what is warm.

The budget counts the window's *logical* extent; the cache stores whole
chunks, so the footprint rounds up by up to one chunk per narrowed axis. It
bounds one tensor's share of the cache, not the cache's total size --
`backlog_high_water` is what stops a filling cache. Above the first plane it
buys scrub headroom: 256 MiB keeps 128 of a 200-plane confocal's Z slices
where 64 MiB would keep only 32, so paging through a stack stays warm past
the opening slice.

**The 3-D target is exempt on Z**: the renderer consumes the volume whole, so
a Z-narrowed warm would miss chunks of it. It still narrows T and C.
`pixel_budget_cubic_root` (below) is what bounds a volume's own size; the
selection budget only bounds the cross-product around it.

## Reduction method: `nearest`

`PyramidConfig.reduction_method` defaults to `"nearest"`
(`core/config.py`), matching `_DEFAULT_REDUCTION_METHOD` in
`core/downsample.py`. This is both the advertised method (what napari reads
off the pyramid) and what precache warms, and it is also what every tile
request implicitly asks for -- `/api/tile_info` carries no reduction field,
so a browser client has no way to request anything else. Keeping the two in
sync here, rather than overriding the method inside precache alone, is what
keeps napari and the tile route reading the same chunk_id.

`nearest` is correct for label images (averaging segmentation labels invents
values that name no object) and, on adapters whose decimated read costs what
it returns rather than what it spans, cheaper than `area`. The accepted
tradeoff is aliasing: thin structures flicker at coarse scales where `area`
would antialias them. Reduction is really a per-tensor property (label vs.
intensity) the server has no signal to infer, so the global default picks
the failure mode that never invents label values.

## 3-D voxel budget

`pyramid.pixel_budget_cubic_root` (default `448`) is both the 3-D gate and
the 3-D target's voxel budget, because the coarsest level is the one thing
both renderers upload as a single texture. It's set from a measured GPU
ceiling, not a guess: on a Quadro P2000, Viv's unconditional Float32 upload
(2x the VRAM of napari's native-dtype path) falls off a frame-rate cliff
between 343 MiB (448^3, 50 fps) and 516 MiB (512^3, 17 fps). `448` is a
tripwire against a future dataset landing in that regime, not a bound any
shipped adapter currently approaches.

A client reaches this warm target through `scale_policy: "volume"` on
`/api/slice` ([http-server.md](http-server.md)) rather than guessing a scale itself; how the
tile route serves rungs out of the warm ladder is [http-server-tile-endpoints.md](http-server-tile-endpoints.md).

## Config

```jsonc
"pyramid": {
  "reduction_method": "nearest",
  "downscale_factor": 2,          // dyadic rungs; matches the tile ladder
  "pixel_budget_cubic_root": 448, // 3-D gate + target
  "plane_max_pixels": 4000000     // 2-D rungs: cap, target and gate
},
"precache": {
  "enabled": true,
  "warm_budget_bytes": 268435456, // per level, caps the T/Z/C cross-product
  "backlog_high_water": 0.8,
  "idle_debounce_seconds": 2.0
}
```
