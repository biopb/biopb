# Proposal: precache policy — admission control for a level-0-hydrating warm

**Status:** proposal, not implemented. Current behaviour is `serving/precache.py`.

## 1. Where this sits

- **#808** — a scaled chunk is computed from the source file even when the
  covering full-resolution chunks are cached. The *read* side already works
  (a 1/2 read after full-res was cached costs 0.2 MB of disk); the populate side
  is what is missing.
- **#814** (`proto/compose-scaled-reads`, prototype) — supplies the populate
  side: `resolve_chunk_data(..., compose=True)` builds a scaled chunk by
  streaming the full-resolution transfer chunks under it through
  `resolve_chunk_data` itself, which leaves them cached under the chunk_ids a
  full-resolution plan asks for. Also, independently, deferred cache writes.
- **#812** — the composed chunk_ids must be exactly the ones a full-resolution
  `get_read_plan` mints; a second opinion about the grid makes the fill dead
  weight.

#814 wires `compose` at the two `server.py` do_get sites and **deliberately not
at precache** (`adapter_base.py` docstring, and the call site itself):

> No compose= here, deliberately. This warms the *coarsest* level of every tensor
> in the catalog; composing would make each of those chunks materialize and cache
> its full-resolution source, turning a cheap overview warmer into a
> whole-catalog hydrator against a cache the live path is already competing for.
> `_has_headroom()` gates volume, not this change in kind.

That judgement is correct **for precache as it is today**. This document is the
policy that would make `compose=True` defensible there: it is entirely about
admission control and retention, not about warming anything new.

## 2. What precache does today

For every non-remote, non-native-pyramid source, warm the coarsest level of the
advertised pyramid, over the whole tensor:

- `precache.py:367` — `build_pyramid_plan(...)[-1]` picks the level.
- `precache.py:392` — `get_read_plan(...)` with **no `slice_hint`**: every T, C, Z.
- `precache.py:417` — `resolve_chunk_data(chunk_id, cache_manager)`, one entry per
  endpoint, keyed by `array_id + exact bounds + scale_hint` (`core/chunk.py:330`).

`area` reduction reads every full-resolution pixel under the output, so this is
already a full-resolution pass over the entire dataset. It reads level 0 and
throws it away, keeping the `1/prod(scale)` reduction. `compose=True` is exactly
the switch that stops throwing it away.

## 3. What composing in precache costs, measured

From #814 on ND040 (14234², 3ch u2) and ND030:

| | overview warm | later full-res read | bytes written / scene |
|---|---:|---:|---:|
| `compose=False` (today) | 5.88 s | 3.40 s | 76 MB |
| `compose=True` | 8.35 s | **0.02 s** | 1293 MB |

Two things follow, and they point in opposite directions:

- **The wall-clock tax is the one cost precache can absorb.** +2.5 s (~42 %) per
  scene, inherent to reusing the transfer grid (169 reads of 1182² instead of 16
  of 4728²), paid by a background thread behind a 2 s idle gate. Precache is the
  *only* caller for which that tax lands off the critical path.
- **The write volume is the cost that decides everything else.** 17× per scene.
  An 18-scene ND2 plate goes from ~1.4 GB of overview entries to ~23 GB of
  full-resolution chunks, against a 32 GiB cache.

So `compose=True` in precache is not a free side effect of #814; it is a capacity
decision, which is what `resolve_chunk_data`'s docstring says. P1–P4 below are
that capacity decision.

Note also that composing **refuses rather than approximates** — float `area`,
non-dyadic scales, `precompute`, misaligned extents, non-`ArrowFileBackend`,
depth > 0. A precache compose path must treat non-composition as normal, not as
an error, and must not assume the full-resolution chunks landed.

### Keep warming the coarsest level regardless

The two are not substitutes. The coarse entry removes disk **and** reduction CPU
(~18 ms open); level 0 removes only disk, leaving the ~2.3 s full-plane
reduction that #814 instrumented. Composing *produces* the coarse entry on the
way past, so keeping it costs nothing extra. The change is additive.

## 4. Cost model across shapes

Full-resolution bytes (what `compose=True` retains), the coarsest advertised
level, and what that level costs:

| source | full res = level-0 residency | full-res XY plane | coarsest scale | coarse entry |
|---|---|---|---|---|
| WSI 2D 100k² u8 | 9.31 GiB | 9536 MiB | 64×64 | **2.3 MiB** |
| ND2 position `[T1,C4,Z1,14234²]` u16 | 1.51 GiB | 386 MiB | 4×4 | 96.6 MiB |
| confocal `[C2,Z200,2048²]` u16 | 3.12 GiB | 8 MiB | 4×4 | 200 MiB |
| timelapse `[T500,C2,Z1,1024²]` u16 | 1.95 GiB | 2 MiB | **1×1** | **2000 MiB** |
| small 2D 1024² u16 | 2 MiB | 2 MiB | 1×1 | 2 MiB |

Against the deployed cache (**32 GiB** here, `file_max_total_gb: 32`; shipped
default 4 GiB, `core/config.py:511`):

- The timelapse row already *is* a level-0 warm today, because
  `pixel_budget_cubic_root` only counts X/Y/Z: 500 frames never trigger a level,
  so precache stores the whole 1.95 GiB at full resolution for one source.
- The confocal row spends 200 MiB warming all 200 z × 2 c, when the first view is
  one z-slice that reads 32 MiB and returns in ~0.2 s.
- The last two rows are pure cost: nothing about their first open was slow.

## 5. P1 — skip sources whose first read is already fast

Gate on **full-resolution X/Y plane bytes**, since `area` makes the coarse
overview read the full-res plane underneath it:

```
plane_bytes = shape[y] * shape[x] * itemsize      # full resolution, per plane
if plane_bytes < precache.min_plane_bytes:  skip the tensor
```

Default `min_plane_bytes = 64 MiB`. On the table above it skips the timelapse,
confocal and small rows and keeps the WSI and ND2 rows — the intended verdict on
all five, including the confocal row where the plane is small but the dataset is
not, because the *first view* there is a single 32 MiB z-slice.

The constant varies ~10× with codec and storage (uncompressed NVMe vs JPEG tiles
on NFS). Where that matters, confirm by measurement: warm the first endpoint,
time it, extrapolate over the endpoint count, and abandon the tensor if the
projection is under `probe_min_seconds` (default 1.0; 0 disables). Cost of being
wrong is one chunk. The static gate stays in front so the cheap majority never
pays even the probe.

With composing on, this gate does double duty: it admits the pass *and* the 17×
residency the pass leaves behind. Native-pyramid sources stay skipped
(`precache.py:349`).

## 6. P2 — warm the first view, not the T×C cross-product

Give the warm a `slice_hint` instead of planning over the whole tensor. T and C
are *selection* axes — the viewer shows one combination — so warming all of them
multiplies the pass by T·C, and with composing multiplies the level-0 residency
by T·C as well. Z stays whole (napari's 3-D mode reads it; the pyramid budget
already bounds it).

The timelapse row drops from 1.95 GiB and 500 full-resolution passes to ~4 MiB
and one, independent of P1.

## 7. P3 — retention class and overflow bypass

The file cache evicts globally by recency, and composed full-resolution entries
are 16–4093× the coarse entry they support. Unbounded, one large source's chunks
evict *every other source's overview* — catalog-wide overview coverage, the thing
precache exists to provide, becomes a rolling window over one dataset.

1. **Class split.** Reserve a fraction for derived/coarse entries
   (`cache.derived_reserve`, default 0.25), or mark speculative full-resolution
   entries first-to-evict. A coarse entry must not be evictable by a
   full-resolution entry that exists only to accelerate a hypothetical zoom-in.
2. **Overflow bypass.** Compose only when
   `source_bytes <= precache.level0_max_bytes` (default 25 % of the cache
   budget); above it, warm with `compose=False` — which is free, since that is
   the existing call.

A per-source cap is necessary but **not sufficient**: each ND2 scene is 1.62 GiB
and clears an 8 GiB cap comfortably, yet the 18-scene plate sums to 29 GiB and
still evicts everything. Hence the class budget in P4.

## 8. P4 — default off, or backlog-only, depending on cache size

Precache is speculative work whose payoff is *residency*. On a cache small
relative to the data it fronts, nothing survives and the worker is churn plus CPU
burn — which is why it runs with `[precache] enabled=false` on this box and why
it "confounds profiling" (`biopb-mcp/docs/viewer-prefetch.md`). Make that
judgement the default rather than a thing each operator rediscovers.

Turn `enabled` into a tri-state — `"auto"` (default), `true`, `false` — resolved
at startup against the cache budget:

| condition | policy |
|---|---|
| memory backend | off (already true, `_file_backend_active`) |
| `file_max_total_bytes < min_cache_bytes` (default 8 GiB) | **off entirely**, logged once with the reason |
| otherwise | backlog tier: `compose=False`, coarse only; live tier: `compose=True` within the P3/P4 budgets |

The tier split is the substantive part, and it is the direct answer to #814's
objection. The tiers differ in how speculative they are: a source *just
registered* is probably the one about to be opened, so hydrating it is a good
bet; the backlog is the whole catalog on the theory that someone might open any
of it, and its full-resolution residency will never fit at any cache size. So
composing is a **live-tier feature**, and the backlog keeps doing the cheap thing
it does well. "Whole-catalog hydrator" stops being the failure mode because the
catalog is never composed.

Budgets for the live tier, both through the existing headroom gate:

- `level0_max_bytes` — per source, default 25 % of the cache budget (P3).
- `level0_total_fraction` — the whole composed class, default 0.5. This is what
  catches the 18-scene plate.

`_has_headroom` currently gates on *total* cache fill; with the class split it
should gate the backlog on the derived reserve's fill and live composing on the
composed class's fill, not on the global number.

Worked through: **4 GiB (shipped default)** → off, which also removes the ~30 min
startup core burn as an out-of-the-box surprise. **32 GiB (this box)** → on;
backlog warms overviews catalog-wide (~25 MiB per admitted source after P1+P2, so
an 8 GiB reserve covers a few hundred), a newly registered ND2 scene also
composes its 1.62 GiB level 0, and the class budget stops the plate at ~16 GiB
instead of 23–29 GiB.

## 9. P5 — precache must not consume the deferred-write budget

#814's second half commits an entry from memory and lets one writer thread
persist it. Overflow behaviour is deliberate and load-bearing: *"reaching it
neither blocks nor queues: the caller writes its own entry inline."*

`complete_entry(..., allow_deferred=True)` is the default and `get_or_acquire`
does not override it, so **precache's writes would be deferred like anyone
else's** — and a composed warm writes 1293 MB per scene. Filling the queue
silently converts concurrent *live* cold reads back to synchronous writes: the
exact 3.31 s → 1.92 s win the flag buys, given away to a background thread that
gains nothing from it. Precache is never on a critical path.

The 2 s idle gate does not protect against this: the queue drains
asynchronously, so the bytes outlive the yield to live traffic.

So plumb `allow_deferred` from `resolve_chunk_data` down to `complete_entry` —
today only `CacheManager.put` (uploads) can set it — and have precache pass
`False`. It joins the same checklist #814 already maintains for callers that want
bytes on disk rather than data.

## 10. Precache is the experiment #814's open question needs

#814 ends unresolved: composing and deferring come out a wash back to back
(6.00 s vs 6.12 s), and *"composing's real case is the full-resolution read
arriving an hour later, cold — which this harness cannot see."*

That case is precisely precache's: the warm happens at registration or backlog
drain, the open happens minutes to hours later with the page cache gone. So the
discriminating measurement is a precache-shaped one — warm with `compose=True`,
drop the page cache, then read at full resolution — and it is worth running
before either flag stops being a prototype, because it decides whether composing
belongs anywhere *except* precache.

## 11. Config

```jsonc
"precache": {
  "enabled": "auto",                  // P4: "auto" | true | false
  "min_cache_bytes": 8589934592,      // P4: below this cache budget, "auto" means off
  "min_plane_bytes": 67108864,        // P1: skip below this full-res X/Y plane
  "probe_min_seconds": 1.0,           // P1: measured confirmation; 0 disables
  "warm_selection": "first",          // P2: "first" | "all" over T/C
  "compose": "live",                  // P3/P4: "live" | "off" | "all"
  "level0_max_bytes": null,           // P3: per source; null = 25% of the cache budget
  "level0_total_fraction": 0.5,       // P4: whole composed class
  "defer_writes": false               // P5: precache never takes the deferral budget
},
"cache": {
  "derived_reserve": 0.25             // P3: cache fraction reserved for scaled entries
}
```

`enabled` is a plain bool today; `"auto"` widens it. Accept `true`/`false`
unchanged so existing configs (including the `enabled=false` here) keep their
exact meaning. All of these are operational knobs on `PrecacheConfig`
(`core/config.py:572`) plus one on `CacheConfig`; none touch `PyramidConfig`,
which stays the single definition of the advertised levels.

## 12. Order of work

1. **P1, P2, P4** — independent of #808/#814, shippable now. P4 alone stops a
   small-cache box burning CPU for residency it cannot keep; P1+P2 make the warm
   set defensible at any cache size.
2. **P5** — small, and needed before #814's deferral flag defaults on.
3. **P3 + `compose="live"`** — last, once #814 lands, since the budgets and the
   class split are what make `compose=True` in precache anything other than the
   whole-catalog hydrator its docstring warns about.

The ordering matters: turning composing on in precache before P1/P3/P4 exist
would reproduce exactly the failure #814 declined to ship.

## 13. Validation

- Extend `tests/precache_test.py` with the §4 table as a gate/footprint
  parametrisation, asserting the verdict and the bytes retained per row.
- P4 resolution table: `"auto"` yields off at 4 GiB and backlog-coarse +
  live-compose at 32 GiB; explicit `true`/`false` still wins.
- P5: a precache warm with the deferral flag on must leave `deferred_write_bytes`
  at zero, and a concurrent live read must still defer.
- Catalog-scale check for P3: drain a backlog containing one >cache-sized source
  and assert the other sources' overview entries survive.
- The §10 spaced-phase measurement, as the gate on `compose="live"`.
- Re-measure the startup backlog wall-clock on the local `dnd://` catalog: P1+P2
  should move it by roughly the fraction of the catalog that is small-plane.
