# ROI annotations — frontend design

Draw, edit and persist 2-D ROIs on a tensor in the viewer SPA, against the store
and routes built in `biopb-tensor-server/docs/roi-annotations.md`. That doc is the
backend contract; this one covers the SPA: how an overlay composes with Viv, how
the wire JSON becomes an app type, what the authoring model is, and what is
deliberately deferred.

## Non-goals

Same scope boundary as the backend: **instance segmentation is a separate tensor**,
not an annotation set. Everything here assumes human-scale counts (the server caps
at 5000 per tensor), which is what allows a whole-set fetch and in-memory
hit-testing.

Also out: undo/redo, copy between tensors, import/export, and annotation on the
3-D volume viewer (`VolumeViewer`). The 2-D tiled viewer is the only surface.

## What the viewer already gives us

Three facts settled by reading Viv 0.22 and deck.gl 9.3 rather than assuming. Each
one removes or creates work, so they are recorded here.

**Coordinates already match — there is no transform.** `DetailView` renders in
level-0 image pixels; that is what `TileViewer.tsx` reads out of `info.coordinate`
for the hover badge. `annotation.proto` mandates geometry in "LEVEL-0 pixel
coordinates, in the tensor's own Y/X axes". A drawn vertex is a stored vertex.

**Overlay layers compose, but the layer id is load-bearing.** `VivViewer.render()`
appends `deckProps.layers` after its own. It also hardwires

```js
layerFilter: ({ layer, viewport }) => layer.id.includes(getVivId(viewport.id))
```

and `getVivId` (`` `-#${id}#` ``) is not exported from the `@hms-dbmi/viv` barrel.
A layer whose id lacks the literal `-#detail#` is never drawn and never picked,
with no error. So every overlay layer id ends in that suffix, from one constant,
covered by a test — this is the kind of thing that is invisible until something
silently does not render.

**A drag on a layer does not stop the camera.** A layer's `onDragStart` returning
`true` suppresses only the *root* handler (`@deck.gl/core` `deck.js`,
`_dispatchPickingEvent`); the `OrthographicView` controller subscribes to the event
manager independently, so the canvas pans out from under a dragged vertex. The fix
is `controller: {dragPan: false}` while dragging, and `DetailView` inherits
`getDeckGlView()` from `VivView` with a bare `controller: true` — so it needs a
subclass.

That last one is why **creation is click-based, not drag-based** (below): it is the
only hard integration problem here, and click-to-place does not need it.

One cosmetic consequence: Viv sets `getCursor` *after* spreading `deckProps`, so a
per-tool cursor cannot go through deck. `TileViewer` owns the host element, so it
is a CSS rule on the canvas.

## Wire types

The ROI routes are the SPA's only proto3-canonical-JSON surface; everything else in
`@biopb/tensor-flight-client` is hand-written snake_case interfaces mirroring
FastAPI. Proto3 JSON differs in ways that bite silently:

| Proto | JSON on the wire |
|-------|------------------|
| `int64 rev`, `map<string, int64> plane` | **strings**, not numbers |
| `optional bytes drawn_against_version` | base64 string |
| `oneof shape` | a bare key: `{"polygon": {...}}` |

**Decision: hand-write the DTO and codec in the SDK; do not wire `protoc-gen-es`.**

The alternative was generating protobuf-es types. Three things argue against it:

- The generated types are *not* the wire shape either — `rev: bigint`,
  `plane: {[k: string]: bigint}`, `drawnAgainstVersion?: Uint8Array`, oneofs as
  `{case, value}`. The app wants numbers and a discriminated union, so a
  conversion at the SDK seam is written either way. protobuf-es would supply a
  validated codec *inside* a conversion we are writing regardless.
- It was tried and removed. Commit `5c598b74` deleted a `local: protoc-gen-es`
  entry that forced `build_proto.py` (a setuptools `build_py` hook, so it ran on
  `pip install`) to probe `node_modules/.bin`, mutate `PATH`, and fall back to a
  second template when the plugin was absent. `remote: buf.build/bufbuild/es`
  would avoid all of that — verified, it generates `annotation_pb.ts` cleanly with
  no node involved — but it still points `buf generate` at `web/`, and
  `web/AGENTS.md` keeps the two toolchains apart on purpose.
- `web/packages/tensor-flight-client/src/gen/` is dead: nothing outside it imports
  it, `index.ts` does not export it, and no config produces it. It predates
  `Polyline` and has no `annotation_pb.ts`. **Delete it** as part of this work —
  keeping stale generated code that nothing generates is worse than having none.

So: `roi-types.ts` (app-facing types) + `roi-json.ts` (encode/decode), roughly 60
lines, with the int64-as-string and base64 cases pinned by tests. The app never
sees proto shapes.

App-facing type, for reference:

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

`GET /api/rois/{array_id}` returns the tensor's whole set. One fetch per tensor,
held in the store, filtered in memory.

**Why not a viewport filter.** The ROI being edited would vanish on a pan.

**Why not a per-plane filter**, which is the more tempting version and needs its
own answer — the viewport argument does *not* carry over, since a per-plane fetch
would return exactly the ROIs the user can interact with:

- **The play driver steps a slice axis every 100 ms** (`PLAY_FPS = 10`), and a
  slider drag does it at pointer rate. Per-plane means a round trip per frame,
  with the overlay trailing the image by at least one. This is the one that
  decides it.
- **The pin is sparse, so the filter does not push down cleanly.** A dimension
  absent from `plane` applies at every index, so the predicate is
  `(plane['z'] IS NULL OR plane['z'] = ?)` per pinnable dim, and every unpinned
  ROI matches every plane. On a 2-D tensor, or one whose axes cannot be named,
  *nothing* is pinned and a per-plane fetch returns the whole set anyway.
- **It does not exist server-side.** `RoiListRequest` carries only `array_id` and
  `set_name`. Adding a plane filter means a proto field, the Flight action, a
  per-tensor predicate over a `MAP(VARCHAR, BIGINT)` column, and the route — to
  optimise something the 5000-per-tensor cap already bounds.

It would also foreclose showing which planes carry annotations as tick marks on
the z/t slider, which is a one-line derivation with the set resident.

**The cost of whole-set, honestly.** There is no compression anywhere in the
stack (the sidecar mounts only `CORSMiddleware`; control's proxy adds none), so
at the cap a polygon set measures 2.6 MB (8 vertices each) to 9.7 MB (50)
uncompressed. Hand-drawn sets are tens of KB, so this bites only at the cap — and
the fix there is `GZipMiddleware` (the same payload gzips 3.7x), not a plane
filter.

### Nothing is fetched in 3-D

`ViewerPane` keys the viewer on `` `${tensorId}#${render3d ? "3d" : "2d"}#...` ``,
so flipping to the volume viewer unmounts the whole 2-D subtree. The overlay and
the panel are the only two callers of `loadRois`, and the panel is not rendered
in volume mode — so 3-D issues no request for a set it cannot draw.

That makes `loadRois` idempotent on `roisFor` load-bearing rather than an
optimisation: the mode flip is a full remount, so without it a 2-D → 3-D → 2-D
round trip would refetch a set that can reach megabytes.

Rendering ROIs *in* the volume is a separate feature, not a gap: `Point` has an
optional z, but a plane-pinned polygon under an orbit camera needs a
billboard-or-extrude decision the 2-D geometry does not answer.

The array_id passed is whatever the viewer is already using —
`requestedArrayId ?? activeTensorId`, possibly content-versioned. The sidecar
strips the token on the way in and splices it back on the way out, so the SPA needs
no version handling of its own. Writes go to the same address.

`DELETE` and `POST` both proxy correctly through control (`_control.py` routes the
`/data_plane` mount for every method) and pass `_require_same_origin`, which keys
on `Sec-Fetch-Site`.

## Rendering

One deck.gl layer per geometry family, all from packages already in
`packages/app/package.json`:

| Shape | Layer |
|-------|-------|
| polygon, rectangle, ellipse | `PolygonLayer` (ellipse tessellated client-side) |
| polyline | `PathLayer`, width from `Polyline.width` (geometry, not styling) |
| point | `ScatterplotLayer` |

Plane filtering follows the proto's rule exactly: **an axis absent from `plane`
means the ROI applies at every index of that axis.** That is what lets one ROI
follow a z-stack, and getting it backwards would hide most annotations.

The pin is keyed by wire axis index, so the only translation left is from
`SliceState`'s `SliderAxis.key` (`t`, `z`, `a3`) to that index — `sliderAxes`
gives both, and `planeFromSelection` does it in one place.

**The overlay is drawn for the plane on screen, not the plane requested.** They
differ only while a read is outstanding — and during play the "Reading plane…"
cover is deliberately dropped, so the stale image stays visible while the slice
index has already moved on. Driving the overlay from the requested slice there
puts plane N+1's annotations over plane N's pixels for the whole of playback: a
systematic off-by-one, not a flicker. So the overlay derives its pin from
`loadedKey` (the selection that actually landed) via `planeFromSelection`, while
the panel counts against the requested plane.

Hiding the overlay on `!dataValid` instead would strobe — the play driver paces
on exactly that flag, so it toggles ~10 times a second while playing. Outside
play it would also be redundant: the cover is opaque and full-bleed over the
deck canvas, so it already hides the overlay along with the stale image.

Selection state (which ROI is active) is SPA-local and never written.

**Every overlay layer is `pickable: false`.** `TileViewer`'s hover badge reads
`info.sourceLayer` and `info.tile` to report the pixel under the pointer; a
pickable overlay sits on top and would answer that hover itself, blanking the
readout wherever an annotation lies.

Selection therefore hit-tests in JS (`roiHitTest.ts`) rather than using deck.gl
picking. That costs nothing extra — the whole annotation set is already resident,
which is one of the reasons the read path fetches it whole — and it is what the
whole-set fetch was justified by in the first place. A filled shape hits anywhere
inside it *or* within a few screen pixels of its outline, so a thin sliver stays
selectable; an open path and a point hit by proximity. The tolerance is scaled by
world-units-per-pixel, read off `info.viewport.zoom` at click time rather than
from the store's mirrored camera, which trails a gesture by `CAMERA_MIRROR_MS`.

## Authoring

**Creation is click-to-place.** A point is one click and a rectangle is two
(opposite corners, normalised on store), because their vertex count is fixed and
a separate "finish" would be ceremony. A polygon or polyline runs until the user
ends it: `Enter`, the Finish button, or a click on the first vertex, which grows
into a handle once the shape has enough vertices to close. `Escape` abandons the
draft and `Backspace` takes back the last vertex; those keys are bound only while
a draft is open, so the viewer never swallows a key it has no use for.

This avoids the controller conflict entirely, and it is the better interaction for
tracing anyway — a drag-traced polygon at zoom is worse than placed vertices.

The click arrives through `deckProps.onClick`. Viv overrides `layerFilter`,
`layers`, `onViewStateChange`, `views`, `viewState`, `useDevicePixels` and
`getCursor` after spreading `deckProps`, but not the pointer callbacks — and
because the overlay is unpickable the click arrives with no picked layer and
`coordinate` set, which is exactly what placing a vertex needs.

**The draft is its own layer set**, memoised apart from the overlay. The segment
trailing the pointer has to follow it to be worth anything, so that one path
re-renders at pointer rate — but only while a draft is open, and rebuilding the
whole overlay there would re-tessellate every annotation on every mouse move
(the same cost the plane-switch rebuild pays).

**Shapes authored in v1:** point, rectangle, polygon, polyline. Ellipse is
**render-only** — the store accepts one and another client may write one, but the
proto's `Ellipse{center, radius}` is axis-aligned with no rotation, so authoring
it earns little.

**Writes are per completed shape, with `check_rev` on.** Not a global Save button:
per-shape writes make a conflict actionable (`RoiPutResult.conflicts` names the
`roi_id` and the stored `rev`), and they leave no dirty buffer to lose on
navigation. A failed write leaves the shape on screen marked unsaved, retryable.

### Which axes a new shape pins

Broadcast is not something the user sets — it is an axis the pin *omits* — so the
authoring question is which axes a new shape leaves out.

**Default: pin every pinnable axis except channel.** Channel is the clear
broadcast case: the viewer shows one channel at a time, an annotation names an
object rather than a channel, and a c-pinned ROI would vanish the moment the user
switches channel. Marking a channel-specific artifact is the exception, so it
must be available, not the default.

Everything else pins, and the asymmetry is what decides it. An over-specific pin
announces itself — draw on z=12, scrub to z=13, the shape is gone, and the user
reaches for the control. An over-broadcast pin is silent: the shape follows you
through all 40 planes and looks correct, and the mistake surfaces later in
whatever consumes the data. Given a visible mistake and an invisible one, pin.

**The control is one row per pinnable axis** (`Z: 12` / `Z: all`), appearing both
as the pre-draw default and on a selected ROI, where changing it is an ordinary
rev-bumping write. Per-set defaults — nuclei per-slice, a wound boundary global —
are natural, but stay **client-side**, remembered per set name for the session:
the proto has no set-level metadata, and inventing server state for a UI default
would be a dialect the Python SDK does not share.

**Broadcast has to be visible.** On the plane where they coincide, a broadcast ROI
and a pinned one are pixel-identical, so without a readout the only way to tell
them apart is to scrub. The selection shows its pin at minimum; a dashed outline
for anything with a broadcast axis is worth trying after that.

Every axis the sliders offer can be pinned, including unlabelled ones and two
sharing a label — that is what the positional pin bought, and it is why this
section can promise a control for every axis rather than a silent exception.

**Vertex editing** needs the `DetailView` subclass described above to toggle
`dragPan` for the duration of a drag.

**The draft is cleared on a render-mode flip as well as on a tensor change.** The
tool is a preference and can survive a trip through 3-D; a half-placed polygon
reappearing afterwards is only confusing.

## State

New store slice, cleared on tensor change alongside the existing per-tensor state:

```
rois: RoiAnnotation[]         // the fetched set
roisFor: string | null        // which array_id they belong to (as tileInfoFor does)
roisLoading / roisError
activeSet: string             // "" = every set
visibleSets: Set<string>
tool: "none" | "point" | "rect" | "polygon" | "polyline"
draft: DraftShape | null      // vertices placed so far
selectedRoiId: string | null
```

`tool` and the overlay toggle are viewer preferences and outlive a tensor change.
Overlay visibility belongs in `useViewerUrlSync` so a shared link carries it.

**Everything else is scoped by a selector, not reset by a writer.** Each piece
carries the tensor it belongs to (`roisFor`, `hiddenSetsFor`, `roisErrorFor`) and
is read through `selectRois` / `selectHiddenSets` / `selectRoisTruncated` and
friends, which hide it when it belongs to another tensor.

That is not tidiness. `selectSource` is *not* the only way the tensor in view
changes: `applyViewerState` writes `activeTensorId` straight from a URL without
going through it, so a reset written in `selectSource` is missed by every link
the app opens. The warnings are the ones that matter — a carried-over "this
tensor holds more annotations than are shown" reads as a fact about the image on
screen, and unlike a stale count nothing on screen contradicts it.

It is also the treatment `tileInfo` already gets, for the same reason, which
`applyViewerState`'s own comment spells out.

## Content staleness — deferred, and why

`RoiAnnotation.drawn_against_version` is meant to answer "did the image change since
this was drawn?". The SPA cannot fill it today, but the gap is narrower than it
looks and does not block anything.

`tile_info` **already publishes the current version token** — `_tensor_desc_by_array_id`
returns the token off the descriptor it bound, and `tile_info` splices it into the
returned `array_id` (`TestTileInfoPublishesTheVersion`). What the SPA lacks is only
the raw `content_version` bytes, and for an equality check it does not need them:
the server already stakes tile-cache correctness on 32 bits of the same hash.

So the only open question is what gets *stored*, and it is a backend one:

- **Defer** (this work). The field stays empty for SPA-drawn ROIs; Python/Flight
  clients still fill it. A staleness warning is a label, not a blocker.
- **Sidecar stamps the bytes on write and publishes the derived token on read** —
  the translation role it already plays for `array_id`. Costs one `get_descriptor`
  per save batch, on a write path that currently binds no tensor. This is the
  intended end state.
- **Retype the field to `string` and store the token.** Cheapest and single-dialect,
  but it freezes `_version_token`'s hash into persisted records; with the bytes
  stored, the hash stays an implementation detail.

Until one lands, a content-pinned link shows annotations anchored to the bare
array_id — correct behaviour (annotations must outlive an edit), just unlabelled.

## Implementation order

1. **SDK.** `listRois` / `putRois` / `deleteRois` on `TensorHttpClient` beside
   `tileInfo`; `roi-types.ts` + `roi-json.ts`; the `axis -> index` plane
   mapping from `TileInfo`. Delete the dead `src/gen/` tree. Vitest against a
   mocked `fetch`, as the rest of the client is tested.
2. **Read-only overlay.** Store slice, fetch on tensor change, plane filter, the
   three layers, a visibility toggle and set filter in the control column. Useful
   alone: anything written by the Python SDK becomes visible.
3. **Authoring.** Tool strip over the canvas, click-to-place, label and set entry,
   delete, per-shape writes with conflict reporting.
4. **Editing and polish.** Vertex drag (`DetailView` subclass, `dragPan`), move,
   truncation notice at the cap, URL sync, keyboard shortcuts.

## Testing

Component tests follow `ChatPane.test.tsx` (vitest + jsdom). deck.gl is not
rendered under test, so the coverage that matters is pure: the JSON codec, the
plane-visibility predicate, the geometry builders, the layer-id suffix, and the
draft state machine (place / close / cancel). The overlay's actual pixels are
checked by hand.
