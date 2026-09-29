# Image viewer — refactor proposal

Status: proposal (2026-09-28); steps 0 and 1 done. Current structure: [viewer-architecture.md](viewer-architecture.md).

## Why

#1156 fixed one reader of the "which tensor is on screen" question and missed
others. #1163 tried to fix the rest by snapping the selection when `tile_info`
lands. It went through five review rounds, and each round found another place
holding its own copy of the id: the remount key, the pixel-fetch guard, the
server's version token, the URL writer. That is a structural problem, not a
run of bad luck.

### Root causes

- **R1: identity is resolved inside the renderer.** The store only learns
  which tensor is on screen after `TileViewer` mounts, fetches `tile_info`,
  and publishes it back. Until then every tensor-scoped read works from a
  guess, and when the answer arrives the id changes spelling mid-view.
- **R2: scope lives in companion fields, not in the data's shape.** Ten
  `…For` fields are each stamped with whatever id the writer had on hand,
  then compared against one of two identity functions. Nothing ties a
  writer's id to its selector's id, so the `SELECTOR_ONLY_FIELDS` lint rule
  enforces the read side only.
- **R3: viewers produce shared state.** Tile grid, contrast track, applied
  window, plane limits, readiness and camera are all published from effects.
  That gives a one-commit lag, content-compare setters to stop loops, and
  derived values stored as state that then need scoping of their own.
- **R4: six scoping mechanisms** (reset list, a second reset list, read
  guard, id content, remount key, plane key). Each new field has to pick
  one, and picking the wrong one works until the next identity edge case.
- **R5: god objects.** One 1870-line store spans six lifetimes. One
  1200-line component does roughly eight jobs.

### Defects found by the audit (fixed by step 0)

All three come from `tile_info` returning `src@tok/field` (#780) for a
request of `src/field`. That happens on an **ordinary click of a specific
tensor**, not only for a bare source id:

1. **Contrast state is always hidden.** `observedLimitsFor` and
   `contrastTrackFor` are stamped `src/field` (the viewer prop), and
   `currentArrayId` is `src@tok/field`, so `selectObservedLimits` and
   `selectContrastTrack` both return null. Integer dtypes don't show it
   because the dtype fallback is the same range. For **float tensors** the
   panel draws the fixed-window bar on `[0, 1]` while the shader clamps into
   the plane's own limits, which is #955 back again. The track also never
   widens across planes.
2. **An unpinned link's `rs=` sets vanish** once `tile_info` lands
   (`visibleSetsFor = "src/field"` ≠ `"src@tok/field"`). Links the app writes
   are pinned and survive; hand-written or older links don't.
3. **The ROI listing is fetched twice per open.** `loadRois("src/field")` at
   mount, then again as `src@tok/field`. The first response is discarded.

Smaller issues found on the way:
- `SliceControls` shares one `debounceRef` across the axis, percentile, fixed
  and gamma sliders. Moving a second slider within 150 ms drops the first
  one's commit.
- `channelColors` and `channelNames` are keyed by `source_id`, so every
  tensor of a multi-tensor source shares one colour per channel index. The
  names are wrong too: `extractChannelNames(metadata, sceneId?)` can read a
  given scene, but `loadChannelNames` never passes one, so every scene of a
  multi-scene file is labelled with scene 0's channels.
- `applyViewerState` does not reset `playAxis`, `planeReady`,
  `appliedLimits` or `planeLimits`, which `selectSource` does. This is latent
  only because hydration runs once.

## Target design

### A. Resolve the view target in the store, before mount

```ts
type TensorKey = string;              // resolved stable address: "src/field", never bare, never pinned

interface ViewTarget {
  epoch: number;                      // bumps on every open(); guards every async landing
  requested: string | null;           // what the click/link named (bare, pinned, whatever); its pin stays in here
  linked: boolean;                    // opened from a link, so a catalog poll does not evict it
  status: "idle" | "resolving" | "ready" | "failed";
  key: TensorKey | null;              // set when ready
  info: TileInfo | null;              // the one tile_info, shared by 2-D and 3-D
  retrying: boolean;                  // the transport retry is in flight
  error: { reason: string; kind: ViewerErrorKind } | null;
  seedSets: string[] | null;          // a link's rs=, written under `key` on landing
}

openTensor(address: string, init?: Partial<ViewInit>): void   // the only entry path
```

Clicking the tree, opening a recent, landing a link and a catalog-poll
eviction (as `openTensor(null)`) all go through `openTensor`. It bumps `epoch`, applies the single reset
list, fetches `tile_info` (keeping today's transport retry), and on landing
sets `key = splitArrayVersion(info.array_id).arrayId`, dropping the landing if
`epoch` moved on. `ViewerPane` renders nothing until `status === "ready"`,
keys the viewer on `key` (which never changes spelling), and passes `info` in.
`TileViewer` builds its sources with the existing `pixelSourcesFromInfo`.
`VolumeViewer` gets the same `info` instead of fetching its own, which also
saves a round trip on every 2-D/3-D flip.

This removes: `tileInfoFor`, `requestedTensorId`, `currentArrayId`,
`selectTileInfo`'s guard, `setTileInfo` as a publish-back, and all of #1163's
`advanceViewerKey` / `sameStableAddress` / snap logic. Resolution happened
before the first render, so there is nothing left to snap.

Latency stays the same. The viewer already waits on this same serial
`tile_info` before it can draw anything.

### B. Make tensor scope structural: one record per tensor

```ts
interface TensorView {                // everything that "belongs to the tensor in view"
  rois: RoiAnnotation[]; roiSets: RoiSetInfo[];
  roiScopes: Record<string, RoiScopeState>; roisPending: string[]; roisError: string | null;
  visibleSets: string[] | null; broadcastAxes: number[] | null;
  draft: { shape: RoiDraft; sliceKey: string } | null;
  observedLimits: Record<number, [number, number]>;
}
views: Record<TensorKey, TensorView>  // small LRU (e.g. 8)

const useView = <T>(pick: (v: TensorView) => T) =>
  useAppStore((s) => pick(s.views[s.target.key!] ?? EMPTY_VIEW));
```

An async writer captures `key` when it starts and writes into `views[key]`.
A landing for a tensor the user has left writes harmlessly into that
tensor's own record. So the ten `…For` fields, the `stillPending` identity
checks, and most of `SELECTOR_ONLY_FIELDS` all go away. The lint rule shrinks
to "don't read `s.views` directly." Link-seeded state (`rs=`) goes into
`init` and is written under the resolved key, so it can't be keyed wrong.
The ROI cache persists across navigation: coming back to a tensor finds its
rows warm, still subject to the count-mismatch refetch. Today it is a single
slot tagged with `currentArrayId`'s spelling, token included, so A → B → A
refetches A and a re-index looks like a new tensor. Keyed by `TensorKey`
(token-free) it survives both, bounded by the LRU.

### C. Derive, don't publish

A viewer publishes only facts that only it can observe, each tagged with the
`epoch` it was observed under:

```ts
runtime: { epoch; shownSelection: string | null; planeReady: boolean; planeLimits: [number, number] | null }
```

The contrast track and the applied window become **pure selectors** over
`target.info.dtype`, `views[key].observedLimits`, `runtime.planeLimits` and
the display settings. The viewer's shader and the panel's bar call the same
selector, which is #955's guarantee without storing the result. This removes
`contrastTrack(+For)`, `appliedLimits`, and three content-compare setters, and
`useContrastWindow` shrinks to "sample, then note".

### D. Split `SliceState` by lifetime

`position: { t, z, c, axes }` is per tensor and reset by `openTensor`.
`display: { contrastMode, percentileScale, fixedLimits, gamma }` is a
preference, except that `fixedLimits` resets. The Viv selection, the volume
request and the draft's slice key then depend on `position` alone, so a
contrast drag no longer produces a new object that has to be JSON-keyed back
to a stable one.

### E. Store slices by lifetime

Split `store.ts` into zustand slice files that are composed into one store:
`connection+catalog`, `recents`, `jobs`, `preferences` (persisted ones in
one place), `target+position+camera`, `views` (ROI actions live here),
`runtime`. The lifetime table in the architecture doc turns into the file
layout, and each new field has to choose a home.

### F. Decompose TileViewer

Keep the render code and move each concern into a hook with a narrow
contract, sharing what the two viewers duplicate today:

| Hook | Owns | Replaces |
|------|------|----------|
| `usePixelSources(info)` | `pixelSourcesFromInfo`, tile errors | fetch effect + retry (retry moves to `openTensor`) |
| `usePlaneGate(position)` | `selectionKey`, `loadedKey`, label gate → `runtime.planeReady`, `shownSelection` | ~120 lines of key/ref plumbing |
| `useContrastSamples(sources, selection)` | overview raster → `runtime.planeLimits`, observed levels | samples effect + half of `useContrastWindow` |
| `useRoiOverlay(shownPlane)` / `useRoiAuthoring(...)` | layers, hit-test, draft, keys, finish | ~250 lines |
| `useLabelOverlayLayers(info, gate)` | label sources + keyed layers | ~110 lines |
| `useHoverReadout()` | ref-fed badge | as is, extracted |
| `useCameraMirror(kind)`, `useElementSize` | shared by both viewers | two copies each today |

### G. Move play out of the panel

`usePlayback()` is mounted once by `HomePage`, or implemented as a store
action with its own timer. It paces on `runtime.planeReady` for the current
`epoch`. `SliceControls` just toggles `playAxis`. Play no longer depends on a
panel staying mounted, and the "sets `planeReady=false` itself" handshake
becomes an epoch/selection comparison.

### H. Channel colour and names stay per source (known deficiency)

`channelColors` and `channelNames` stay keyed by `source_id`, and names stay
derived client-side from the source's single metadata dict. Colour is really
a per-tensor concern, but keying it by tensor needs to know which scene of
the source metadata a tensor is, and nothing in the client or on the wire
says so today.

The cost of leaving it: every tensor of a multi-tensor source shares one
colour per channel index, and every scene of a multi-scene file is labelled
with scene 0's channel names (`loadChannelNames` passes no `sceneId`). Revisit
when the server publishes channel info per tensor (as it does
`physical_size` / `spacing`); then colours can move to `TensorKey` with no
client-side scene mapping.

## Migration

Each step ships on its own and passes the existing test suite (adapted).
Steps 1–2 are the keystone. 3–5 are independent afterwards.

| Step | Scope | Fixes / deletes |
|------|-------|-----------------|
| 0 (**done**) | `viewKey` (token-free `currentArrayId`) stamps and scopes every `…For` field; contrast setters take the key from the store; `adoptResolution` moves bare-keyed state to the resolved field. No remount changes. | Defects 1–3 (3 only for the token; a bare source still fetches its listing twice) |
| 1 (**done**) | `ViewTarget` + `openTensor` + epoch; viewers take `info`; single entry path | the bare-source double fetch; `currentArrayId`, `viewKey`, `adoptResolution`, `tileInfoFor`, #1163's machinery; the second reset list |
| 2 | `views: Record<TensorKey, TensorView>` + `useView` | 10 `…For` fields, `stillPending` checks, most of the lint list |
| 3 | Derived contrast selectors + `runtime` slice; split `SliceState` | `contrastTrack`, `appliedLimits`, content-compare setters, JSON selection keys |
| 4 | TileViewer hooks, shared hooks with VolumeViewer, `usePlayback`; one debounce per slider | TileViewer to ~400 lines; the debounce bug |
| 5 | Store slice files | `store.ts` to about 6 × 300 lines |

**On #1163:** step 0 covers its store-level fix, and step 1 supersedes its
viewer-key machinery (`advanceViewerKey`/`sameStableAddress`), which would
otherwise be deleted again right away. Its tests describe the right behaviour
(race guard, pin preservation, no remount on resolution), so keep them as the
acceptance tests for step 1.

## Decisions

1. *(resolved: colour and names stay per source for now; see H.)*
2. *(resolved: the ROI cache persists, keyed by `TensorKey` in the LRU of `TensorView`s; see B.)*
3. *(resolved: `ViewerPane` reads resolution failures from `target.error`; the viewer's `onUnsupported` stays only for render-time refusals: dtype, WebGL, `volumeRefusal`.)*
