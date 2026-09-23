# ROI annotations — frontend design

Draw, edit and persist 2-D ROIs on a tensor in the viewer SPA. Backend
contract: [roi-annotations.md](../../biopb-tensor-server/docs/roi-annotations.md).

## Wire types

The ROI routes are the SPA's only proto3-canonical-JSON surface (everything
else in `@biopb/tensor-flight-client` is hand-written snake_case mirroring
FastAPI). Proto3 JSON's quirks:

| Proto | JSON on the wire |
|-------|------------------|
| `int64 rev`, `map<string, int64> plane` | strings, not numbers |
| `optional bytes drawn_against_version` | base64 string |
| `oneof shape` | a bare key: `{"polygon": {...}}` |

`roi-types.ts` (app-facing types) and `roi-json.ts` (encode/decode) handle
this by hand rather than through generated protobuf-es types, which would
still need the same int64/base64/oneof conversion at the SDK seam and would
reintroduce a Node-toolchain dependency.

```ts
export interface RoiAnnotation {
  roiId: string;
  arrayId: string;
  setName: string;
  label: string;
  geometry: RoiGeometry;              // discriminated union on `kind`
  plane: Record<number, number>;      // wire axis index -> index; absent = all
  props: Record<string, unknown>;     // parsed props_json
  rev: number;
  createdAtMs: number;
  updatedAtMs: number;
}
```

## Fetching

`GET /api/rois/{array_id}?set=<name>` returns a tensor's client-owned sets, all at once if `set` is ignored.
A server-owned (`@`) set only fetch with a specified set name, and only while
it's on screen. Every response also carries `sets` — every set on the tensor with its stored count, server-owned included
— which is how the panel knows a reserved set exists before it holds any of
its rows.

The store tracks what has landed per *scope* (`""` for the client-owned
listing, a set name for a named fetch), keyed under the tensor it belongs to.
A scope re-fetches whenever a later response's `sets` count disagrees with what's resident.

**Nothing is fetched in 3-D.** `ViewerPane` keys the viewer on the 2-D/3-D
mode, so flipping to the volume viewer unmounts the whole 2-D subtree
(`loadRois`'s only two callers, the overlay and the panel, are both unmounted
with it). `loadRois` being idempotent on already-landed scopes is load-bearing
here, not an optimization: without it, a 2-D → 3-D → 2-D round trip would
refetch a set that can reach megabytes on every mode flip.

A content-version on array_id is legal but ignored by the sidecar.

## Rendering

| Shape | Layer |
|-------|-------|
| polygon, rectangle, ellipse | `PolygonLayer` (ellipse tessellated client-side) |
| polyline | two `PathLayer`s — the band (width from geometry, not styling) and its centreline |
| point | `ScatterplotLayer` |

Every kind renders as a translucent area under an opaque, screen-unit-width
outline, from shared alpha constants. **An axis absent from `plane` means the ROI applies at every index of that
axis**.

**The overlay is drawn for the plane on screen, not the plane requested.**
They differ only while a read is outstanding; during play the "Reading
plane…" cover is deliberately dropped so the stale image stays visible while
the slice index has already moved on, so the overlay derives its pin from
`loadedKey` (the selection that actually landed), while the panel counts
against the requested plane. Getting this backwards would put plane N+1's
annotations over plane N's pixels for the whole of playback.

**Every overlay layer is `pickable: false`** — `TileViewer`'s hover badge
reads the pixel under the pointer via deck.gl picking, and a pickable overlay
on top would intercept that. Selection therefore hit-tests in JS
(`roiHitTest.ts`) against the already-resident set: a filled shape hits inside
it or within a few screen pixels of its outline, a point hits by proximity,
and a polyline hits within its own stroke plus that same tolerance (scaled by
world-units-per-pixel at click time) — the same half-width the store pads the
bbox by, so what's selectable and what SQL reports as covered agree.

## Authoring

**Creation is click-to-place**, not drag-based (see the `dragPan` constraint
above): a point is one click, a rectangle is two (opposite corners), a
polygon/polyline runs until `Enter`, the Finish button, or a click on the
first vertex; `Escape` abandons the draft and `Backspace` takes back the last
vertex. These keys bind on `window` (the canvas isn't focusable) but are
suppressed during IME composition and inside any text field, since
`Backspace` there must edit text.

The draft is its own layer set, memoised apart from the overlay, so the
pointer-following segment can re-render at pointer rate without
re-tessellating every other annotation. **A draft does not survive a plane,
render-mode, or tensor change** — it's hidden (not destroyed, so scrubbing
back restores it) rather than finished onto the wrong image, keyed off the
same `loadedKey`/`selectDraft` selector the overlay uses.

**Shapes authored in v1:** point, rectangle, polygon, polyline. Ellipse is
render-only (the store accepts one and the overlay draws its `rotation`) —
drawing one wants a rotate handle, which is the drag-editing machinery not yet
built.

**Writes are per completed shape, with `check_rev` on** — not a global Save
button, so a conflict is actionable per row and there's no dirty buffer to
lose on navigation. A create is optimistic: a provisional row carries a
`pending,`-prefixed id (the wire can't produce one, since delete splits ids on
`,`) and is swapped for the stored row when the write lands or removed if it
fails; it isn't selectable until the server assigns its real id. Delete is
optimistic the same way. Clearing a whole set is the exception and stays
blocking, since it's a deliberate confirmed action.

### Axis pinning

**Default: pin every pinnable axis except channel.** The viewer shows one
channel at a time and an annotation names an object, not a channel, so a
channel-pinned ROI would vanish on channel switch — marking a
channel-specific artifact is the exception, so it's available but not
default. Everything else pins by default because an over-specific pin
announces itself (scrub past it, the shape disappears) while an
over-broadcast pin is silent (it follows you through every plane and looks
correct until something downstream consumes it wrong).

The control is one row per pinnable axis (`Z: 12` / `Z: all`), on the
pre-draw default and on a selected ROI. Per-set pin defaults are remembered
client-side per set name for the session (there's no server-side field for
it). The selection readout shows its pin at minimum, since a broadcast and a
pinned ROI are pixel-identical on the plane where they coincide.

## Known gaps and guardrails

- **A lint rule (`no-restricted-syntax` in `eslint.config.mjs`) blocks reading
  `rois`/`draft`/`selectedRoiId`/etc. directly off `useAppStore`**, forcing
  every read through the tensor-scoped selectors (`selectRois`, `selectDraft`,
  ...). Three prior bugs (a stale hidden-sets list, a truncation warning for
  the wrong tensor, an enabled Finish for a dropped draft) came from a second
  unguarded reader; the selector fixes the instance, the lint rule stops the
  next one. `getState()`/`setState()` stay open for tests.
- **The tool strip takes the draft as a prop, not a store read** — a second
  unguarded read there could show a status line and an enabled Finish for a
  draft the viewer already considers gone.
- **A selection is only actionable while its plane is on screen.** It
  survives a plane change (scrubbing back brings it into reach) but the panel
  withholds it, and `Delete` is bound on the same condition, meanwhile.
- **deck.gl's click recognizer is reconfigured**
  (`eventRecognizerOptions: {click: {interval: 0}}`) because the default
  `requireFailure(['dblclick'])` deferred every click's emit by 300 ms, so
  vertices placed faster than that arrived as one click at the last position.
  `dblclick` keeps its own recognizer; `isRealClick` drops the duplicate
  `onClick` deck also fires for it.
- **Double-click zoom is off while a click would place something** — the two
  taps of a double click are two vertices, and zooming out from under them
  mid-gesture would move everything already placed. A `DetailView` subclass
  (`BiopbDetailView`) passes `{doubleClickZoom}` through per tool, reusing the
  same id/height/width so switching tools doesn't reset the camera.
- **Selection renders as its own layer, over one annotation**, rather than as
  an accessor on the annotation's own layer — folding it in would make
  deck.gl regenerate every layer's data (earcut included) on every click;
  measured flat at ~0.002 ms this way against 0.06-1.9 ms of rebuild the
  other way.
- **A polyline's width is set with the tool, before the first click** — it's
  geometry (the band of pixels the stroke covers), not styling, so choosing it
  after tracing would mean tracing blind. Default is 4 image pixels (0 renders
  at the layer's hairline floor at every zoom).
- **Delete is a key, not a button** — the panel's one-line selection readout
  leaves no room for one, and `Backspace` is reserved for "take back the last
  vertex" during drafting.

## State

New store slice, cleared on tensor change:

```
rois: RoiAnnotation[]         // every row held, across the landed scopes
roiSets: RoiSetInfo[]         // every set on the tensor, with its stored count
roisFor: string | null        // which array_id they belong to
roiScopes: Record<string, {truncated, skipped}>   // "" = client-owned listing; else a set name
roisPending / roisError
visibleSets: string[] | null  // the sets on screen; null = this tensor's default
tool: "none" | "point" | "rect" | "polygon" | "polyline"
draft: DraftShape | null
selectedRoiId: string | null
```

**`visibleSets` is a positive list, or `null` for the tensor's default**
(client-owned on, server-owned off) — a hidden-list shape couldn't express
"show `@ome`" from a URL before the listing lands. It's materialised into an
explicit list on the first toggle, and rides the URL as repeated `rs=` params
(absent = default; present, even `rs=` bare, means exactly these) alongside
the axis indices.

**The overlay toggle turns the whole annotation surface off**, not just the
rendered shapes — with it off, no tool is offered, a click selects and places
nothing, an in-progress draft is hidden, and the authoring panel (including
Delete) is gone. Anything narrower would let a draft stay finishable while
invisible.

`tool` and the overlay toggle are viewer preferences and outlive a tensor
change. Everything else is scoped by a selector rather than reset by a
writer, because `applyViewerState` can change the active tensor straight from
a URL without going through the write path a reset would otherwise hook.

## Not yet done

- **Vertex-drag editing** of an existing shape (move, reshape) — needs a
  `DetailView` subclass toggling `dragPan` for the duration of a drag; the
  current subclass only handles `doubleClickZoom`. There is also no store
  action to update an existing ROI's geometry or plane pin, only create and
  delete.
- **`RoiPutResult.conflicts` is decoded but never surfaced in the UI** —
  there's no edit path yet for a conflict to occur on, so this is dormant
  rather than broken.
- **Content staleness.** `drawn_against_version` is filled by Python/Flight
  clients but left empty by the SPA today; `tile_info` already publishes the
  current version token, so a content-pinned link still shows annotations
  correctly anchored, just unlabelled as possibly-stale. Filling it means the
  sidecar stamping the bytes on a save batch, at the cost of one
  `get_descriptor` per batch.
