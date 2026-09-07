/**
 * ROI annotations as deck.gl layers, composed into the Viv viewer.
 *
 * Two things about this integration are not obvious and are load-bearing:
 *
 * **The layer id must contain Viv's view id.** `VivViewer` appends
 * `deckProps.layers` to its own, then filters every layer through
 * `layer.id.includes(getVivId(viewport.id))` -- so a layer whose id lacks
 * `-#detail#` is never drawn and never picked, with no error anywhere. Viv does
 * not export `getVivId`, so {@link roiLayerId} rebuilds it from the exported
 * `DETAIL_VIEW_ID` rather than hardcoding the string.
 *
 * **Geometry needs no transform.** `DetailView` renders in level-0 image
 * pixels, which is exactly the space `annotation.proto` stores ROI coordinates
 * in.
 *
 * See docs/roi-annotations-ui.md.
 */

import { PathLayer, PolygonLayer, ScatterplotLayer } from "@deck.gl/layers";
import { DETAIL_VIEW_ID } from "@hms-dbmi/viv";
import { pinnableAxes, planePinFor, roiVisibleOnPlane, sliderAxes } from "@biopb/tensor-flight-client";
import type { RoiAnnotation, RoiGeometry, TileInfo } from "@biopb/tensor-flight-client";
import { vivSelection, type SliceIndices } from "./vivUtils";
import { CLOSE_HANDLE_PX, type RoiDraft } from "./roiDraft";

/** An `[x, y]` in level-0 image pixels, the space deck.gl draws these in. */
export type XY = [number, number];

/**
 * A layer id Viv's `layerFilter` will accept.
 *
 * Built from `DETAIL_VIEW_ID` so it cannot drift from the view it has to match.
 */
export function roiLayerId(name: string): string {
  return `roi-${name}-#${DETAIL_VIEW_ID}#`;
}

/**
 * `axis -> index` for a Viv selection (which is keyed by `SliderAxis.key`).
 *
 * The two spellings of a plane meet here: Viv addresses axes by `t`/`z`/`c`/`a3`
 * and the proto pins annotations by wire axis index. Both come from
 * `sliderAxes`, so they cannot disagree about which axis a key names.
 */
export function planeFromSelection(
  info: TileInfo | null,
  selection: Record<string, number>,
): Record<number, number> {
  if (!info) return {};
  const indexByAxis: Record<number, number> = {};
  for (const axis of sliderAxes(info.dim_labels, info.shape)) {
    indexByAxis[axis.axis] = selection[axis.key] ?? 0;
  }
  return planePinFor(pinnableAxes(info), indexByAxis);
}

/**
 * The plane the viewer has been *asked* for.
 *
 * What the panel counts against. The overlay instead draws the plane that is
 * actually on screen -- see `planeFromSelection` at the `loadedKey` call site in
 * TileViewer -- because during play those differ.
 *
 * One implementation shared by both: two derivations of "which plane is this"
 * could drift, and the symptom would be a panel count that does not match what
 * is drawn.
 */
export function currentPlaneFor(
  info: TileInfo | null,
  slice: SliceIndices,
): Record<number, number> {
  if (!info) return {};
  // Through vivSelection, so the indices are clamped to the tensor's extents
  // exactly as the read path clamps them.
  return planeFromSelection(info, vivSelection(info, slice));
}

/** Segments used to tessellate an ellipse into a polygon ring. */
export const ELLIPSE_SEGMENTS = 64;

/**
 * Per-set colours. Assigned by name rather than by position so a set keeps its
 * colour as others are added, hidden, or emptied.
 */
const PALETTE: Array<[number, number, number]> = [
  [56, 189, 248],
  [251, 191, 36],
  [74, 222, 128],
  [244, 114, 182],
  [167, 139, 250],
  [248, 113, 113],
  [45, 212, 191],
  [251, 146, 60],
];

/** Selection emphasis: one colour, so a selected shape reads the same anywhere. */
const SELECTED_COLOR: [number, number, number, number] = [255, 255, 255, 255];

/**
 * Every annotation is a translucent area under an opaque outline, and these are
 * what make that true of all of them rather than of each of them separately.
 *
 * The area is what the annotation claims and the pixels under it have to stay
 * readable through it; the outline is the shape itself, so it is opaque, drawn
 * in screen units, and is the only part selection touches. A polyline's band is
 * an area by this reckoning even though it is drawn by a stroke -- which is
 * what it was getting wrong: one opaque band, hiding the pixels it pointed at
 * and turning white as a whole when selected, where every other kind keeps its
 * fill and brightens an edge.
 */
const AREA_ALPHA = 40;
/** A dot is small enough that the area alpha would leave nothing to see. */
const POINT_AREA_ALPHA = 90;
const OUTLINE_ALPHA = 230;
const OUTLINE_PX = 1.5;
const SELECTED_OUTLINE_PX = 3;

export function setColor(setName: string): [number, number, number] {
  let hash = 0;
  for (let i = 0; i < setName.length; i++) {
    hash = (hash * 31 + setName.charCodeAt(i)) | 0;
  }
  return PALETTE[Math.abs(hash) % PALETTE.length] as [number, number, number];
}

/**
 * The closed ring for the arms drawn as polygons, or null for the others.
 *
 * A rectangle becomes its four corners and an ellipse is tessellated here
 * rather than by a dedicated layer, so all three share one `PolygonLayer` and
 * one styling path.
 */
export function roiRing(geometry: RoiGeometry, segments = ELLIPSE_SEGMENTS): XY[] | null {
  switch (geometry.kind) {
    case "polygon":
      return geometry.points.map((p) => [p.x, p.y] as XY);
    case "rectangle": {
      const { topLeft: a, bottomRight: b } = geometry;
      return [
        [a.x, a.y],
        [b.x, a.y],
        [b.x, b.y],
        [a.x, b.y],
      ];
    }
    case "ellipse": {
      const { center, radius, rotation } = geometry;
      const cos = Math.cos(rotation);
      const sin = Math.sin(rotation);
      const ring: XY[] = [];
      for (let i = 0; i < segments; i++) {
        const theta = (2 * Math.PI * i) / segments;
        const x = radius.x * Math.cos(theta);
        const y = radius.y * Math.sin(theta);
        // Rotated about the centre. Y grows downward in image coordinates, so a
        // positive rotation reads clockwise on screen -- the same convention
        // the geometry is stored in, not a separate screen-space one.
        ring.push([center.x + x * cos - y * sin, center.y + x * sin + y * cos]);
      }
      return ring;
    }
    default:
      return null;
  }
}

/** The open path of a polyline, or null for the other arms. */
export function roiPath(geometry: RoiGeometry): XY[] | null {
  return geometry.kind === "polyline" ? geometry.points.map((p) => [p.x, p.y] as XY) : null;
}

/**
 * The annotations to draw: on this plane, and in a set that is not switched off.
 *
 * Filtering in memory rather than fetching per plane is deliberate -- the play
 * driver steps an axis every 100 ms, and a fetch per frame would leave the
 * overlay trailing the image. See docs/roi-annotations-ui.md.
 */
export function visibleRois(
  rois: RoiAnnotation[],
  currentPlane: Record<number, number>,
  hiddenSets: string[],
): RoiAnnotation[] {
  const hidden = new Set(hiddenSets);
  return rois.filter(
    (roi) => !hidden.has(roi.setName) && roiVisibleOnPlane(roi.plane, currentPlane),
  );
}

/** Sets present in a fetched collection, with their counts, in first-seen order. */
export function roiSetCounts(rois: RoiAnnotation[]): Array<{ setName: string; count: number }> {
  const counts = new Map<string, number>();
  for (const roi of rois) {
    counts.set(roi.setName, (counts.get(roi.setName) ?? 0) + 1);
  }
  return [...counts].map(([setName, count]) => ({ setName, count }));
}

/**
 * Axes a new annotation should broadcast across unless the user says otherwise.
 *
 * Channel, and only channel. The viewer shows one channel at a time and an
 * annotation names an object rather than a channel, so a c-pinned ROI would
 * vanish the moment the user switches channel.
 *
 * Everything else pins, because the two mistakes are not symmetric: an
 * over-specific pin announces itself (draw on z=12, scrub to z=13, the shape is
 * gone) while an over-broadcast one is silent -- the shape follows you through
 * every plane and looks right. Prefer the mistake the user can see.
 */
export function defaultBroadcastAxes(info: TileInfo | null): number[] {
  if (!info) return [];
  return pinnableAxes(info)
    .filter((axis) => axis.named === "c")
    .map((axis) => axis.axis);
}

/** The pin a new annotation gets: every pinnable axis except the broadcast ones. */
export function pinForNewRoi(
  info: TileInfo | null,
  currentPlane: Record<number, number>,
  broadcastAxes: number[],
): Record<number, number> {
  if (!info) return {};
  const broadcast = new Set(broadcastAxes);
  const pin: Record<number, number> = {};
  for (const [axis, index] of Object.entries(currentPlane)) {
    if (!broadcast.has(Number(axis))) pin[Number(axis)] = index;
  }
  return pin;
}

export interface RoiLayerOptions {
  rois: RoiAnnotation[];
  /** `axis -> index` for the plane on screen; see `planePinFor`. */
  currentPlane: Record<number, number>;
  hiddenSets: string[];
  /** The overlay toggle. False yields no layers at all rather than hidden ones. */
  visible: boolean;
}

/**
 * The overlay's layers, ready for `VivViewer`'s `deckProps.layers`.
 *
 * Selection is deliberately not an input. Every call rebuilds each layer's
 * `data`, which deck.gl reads as a changed identity and answers by regenerating
 * every attribute -- earcut included, which is the dominant term. Folding
 * selection in here made a click pay that for the whole set; it is drawn by
 * {@link buildSelectionLayers} instead, over one annotation, so the layers
 * below keep their identity and deck.gl does nothing at all.
 *
 * `pickable: false` throughout, and not only because nothing is interactive
 * yet: the hover readout in `TileViewer` reads `info.sourceLayer` and
 * `info.tile` to report the pixel under the pointer, so a pickable overlay on
 * top would answer that hover itself and blank the readout wherever an
 * annotation lies. Phase 3 has to route around that rather than just flip this.
 */
export function buildRoiLayers(options: RoiLayerOptions): unknown[] {
  const { rois, currentPlane, hiddenSets, visible } = options;
  if (!visible) return [];
  const shown = visibleRois(rois, currentPlane, hiddenSets);
  if (shown.length === 0) return [];

  const rings = shown
    .map((roi) => ({ roi, ring: roiRing(roi.geometry) }))
    .filter((entry): entry is { roi: RoiAnnotation; ring: XY[] } => entry.ring !== null);
  const paths = shown
    .map((roi) => ({ roi, path: roiPath(roi.geometry) }))
    .filter((entry): entry is { roi: RoiAnnotation; path: XY[] } => entry.path !== null);
  const points = shown.filter((roi) => roi.geometry.kind === "point");

  const layers: unknown[] = [];

  if (rings.length > 0) {
    layers.push(
      new PolygonLayer({
        id: roiLayerId("shapes"),
        data: rings,
        pickable: false,
        filled: true,
        stroked: true,
        getPolygon: (d: { ring: XY[] }) => d.ring,
        getFillColor: (d: { roi: RoiAnnotation }) => [...setColor(d.roi.setName), AREA_ALPHA],
        getLineColor: (d: { roi: RoiAnnotation }) => [...setColor(d.roi.setName), OUTLINE_ALPHA],
        // Pixels, not world units: an outline is a way of seeing the shape, so
        // it should not thin out as the user zooms out of a large field.
        lineWidthUnits: "pixels",
        getLineWidth: OUTLINE_PX,
      }),
    );
  }

  if (paths.length > 0) {
    layers.push(
      // The band the stroke claims: this shape's area, so it takes the area's
      // alpha and world units -- the width is geometry, the pixels it covers,
      // so it scales with the image. No pixel floor, unlike the centreline
      // below: a stored width of 0 has no extent to draw, and drawing it a
      // floor's worth would invent one.
      new PathLayer({
        id: roiLayerId("path-bands"),
        data: paths,
        pickable: false,
        widthUnits: "common",
        widthMinPixels: 0,
        capRounded: true,
        jointRounded: true,
        getPath: (d: { path: XY[] }) => d.path,
        getColor: (d: { roi: RoiAnnotation }) => [...setColor(d.roi.setName), AREA_ALPHA],
        getWidth: (d: { roi: RoiAnnotation }) =>
          d.roi.geometry.kind === "polyline" ? Math.abs(d.roi.geometry.width) : 0,
      }),
      // The centreline stands in for the outline the other kinds have: a band
      // cannot be stroked along its edges here, and the centreline is the trace
      // the user actually made. Screen units and the outline alpha, so it reads
      // the same as a polygon's edge, and it is the only part selection touches.
      new PathLayer({
        id: roiLayerId("paths"),
        data: paths,
        pickable: false,
        widthUnits: "pixels",
        capRounded: true,
        jointRounded: true,
        getPath: (d: { path: XY[] }) => d.path,
        getColor: (d: { roi: RoiAnnotation }) => [...setColor(d.roi.setName), OUTLINE_ALPHA],
        getWidth: OUTLINE_PX,
      }),
    );
  }

  if (points.length > 0) {
    layers.push(
      new ScatterplotLayer({
        id: roiLayerId("points"),
        data: points,
        pickable: false,
        // A point marks a location, not an extent, so it keeps its screen size.
        radiusUnits: "pixels",
        getRadius: 4,
        stroked: true,
        lineWidthUnits: "pixels",
        getLineWidth: OUTLINE_PX,
        getPosition: (d: RoiAnnotation) =>
          d.geometry.kind === "point" ? [d.geometry.at.x, d.geometry.at.y] : [0, 0],
        getFillColor: (d: RoiAnnotation) => [...setColor(d.setName), POINT_AREA_ALPHA],
        getLineColor: (d: RoiAnnotation) => [...setColor(d.setName), OUTLINE_ALPHA],
      }),
    );
  }

  return layers;
}

/**
 * The emphasis on the selected annotation, drawn over the overlay.
 *
 * Its own layers, over one datum, so selecting costs the same whether the
 * tensor holds ten annotations or the server's five thousand -- see
 * {@link buildRoiLayers} for what it used to cost.
 *
 * Outline only. The area underneath is still drawn by the layer below, which is
 * both why this composes (a wider opaque outline covers the one it replaces)
 * and what keeps selection an emphasis rather than a different shape.
 */
export function buildSelectionLayers(roi: RoiAnnotation | null): unknown[] {
  if (!roi) return [];

  if (roi.geometry.kind === "point") {
    return [
      new ScatterplotLayer({
        id: roiLayerId("selection-point"),
        data: [roi],
        pickable: false,
        radiusUnits: "pixels",
        getRadius: 4,
        stroked: true,
        filled: false,
        lineWidthUnits: "pixels",
        getLineWidth: OUTLINE_PX,
        getPosition: (d: RoiAnnotation) =>
          d.geometry.kind === "point" ? [d.geometry.at.x, d.geometry.at.y] : [0, 0],
        getLineColor: SELECTED_COLOR,
      }),
    ];
  }

  const path = roiPath(roi.geometry);
  if (path) {
    // The centreline, as in the overlay: the band stays the set's colour,
    // because the area is what the annotation claims and selecting it does not
    // change that.
    return [
      new PathLayer({
        id: roiLayerId("selection-path"),
        data: [{ path }],
        pickable: false,
        widthUnits: "pixels",
        capRounded: true,
        jointRounded: true,
        getPath: (d: { path: XY[] }) => d.path,
        getColor: SELECTED_COLOR,
        getWidth: SELECTED_OUTLINE_PX,
      }),
    ];
  }

  const ring = roiRing(roi.geometry);
  if (!ring) return [];
  return [
    new PolygonLayer({
      id: roiLayerId("selection-shape"),
      data: [{ ring }],
      pickable: false,
      filled: false,
      stroked: true,
      getPolygon: (d: { ring: XY[] }) => d.ring,
      lineWidthUnits: "pixels",
      getLineWidth: SELECTED_OUTLINE_PX,
      getLineColor: SELECTED_COLOR,
    }),
  ];
}

// ---------------------------------------------------------------------------
// The shape being drawn
// ---------------------------------------------------------------------------

/** Colour of an in-progress draft: distinct from every set colour. */
const DRAFT_COLOR: [number, number, number, number] = [250, 204, 21, 240];
/** The same colour as the band a finished polyline becomes. */
const DRAFT_BAND_COLOR: [number, number, number, number] = [250, 204, 21, AREA_ALPHA];

export interface DraftLayerOptions {
  draft: RoiDraft | null;
  /** Where the pointer is, for the segment that trails it. Null when off-image. */
  cursor: XY | null;
  /** The draft has enough vertices to close, so its first vertex is a handle. */
  closeable: boolean;
  /** Width a polyline draft will be stored at, in image pixels. */
  polylineWidth: number;
}

/**
 * The in-progress shape: placed vertices, the edges between them, and the
 * segment trailing the pointer.
 *
 * Its own builder, memoised apart from {@link buildRoiLayers}, because the
 * trailing segment follows the pointer: rebuilding the whole overlay at pointer
 * rate would re-tessellate every annotation for every mouse move.
 */
export function buildDraftLayers(options: DraftLayerOptions): unknown[] {
  const { draft, cursor, closeable, polylineWidth } = options;
  if (!draft || draft.points.length === 0) return [];

  // A polyline previews as the band it will become -- same world units, same
  // area alpha -- because the width is the pixels it claims, and tracing at a
  // hairline only to find out afterwards is tracing blind. The construction
  // line stays on top of it, as the centreline does on a finished one, so what
  // is on screen while drawing is what commit leaves behind.
  const asBand = draft.tool === "polyline";

  const placed = draft.points;
  // A rectangle previews as the box it will become, not as the two clicks that
  // define it -- otherwise the second click's effect is invisible until it lands.
  const preview: XY[] =
    draft.tool === "rectangle" && cursor && placed.length === 1
      ? rectanglePreview(placed[0] as XY, cursor)
      : cursor
        ? [...placed, cursor]
        : placed;

  const layers: unknown[] = [
    ...(asBand
      ? [
          new PathLayer({
            id: roiLayerId("draft-band"),
            data: [{ path: preview }],
            pickable: false,
            widthUnits: "common",
            widthMinPixels: 0,
            getWidth: polylineWidth,
            capRounded: true,
            jointRounded: true,
            getPath: (d: { path: XY[] }) => d.path,
            getColor: DRAFT_BAND_COLOR,
            updateTriggers: { getPath: preview, getWidth: polylineWidth },
          }),
        ]
      : []),
    new PathLayer({
      id: roiLayerId("draft-path"),
      data: [{ path: preview }],
      pickable: false,
      widthUnits: "pixels",
      getWidth: 2,
      capRounded: true,
      jointRounded: true,
      getPath: (d: { path: XY[] }) => d.path,
      getColor: DRAFT_COLOR,
      updateTriggers: { getPath: preview },
    }),
    new ScatterplotLayer({
      id: roiLayerId("draft-vertices"),
      data: placed.map((at, i) => ({ at, first: i === 0 })),
      pickable: false,
      radiusUnits: "pixels",
      // The first vertex grows into a handle once clicking it would close the
      // shape, which is the only affordance saying that is possible.
      getRadius: (d: { first: boolean }) => (d.first && closeable ? CLOSE_HANDLE_PX / 2 : 3),
      stroked: true,
      lineWidthUnits: "pixels",
      getLineWidth: 1.5,
      getPosition: (d: { at: XY }) => d.at,
      getFillColor: [250, 204, 21, 140],
      getLineColor: DRAFT_COLOR,
      updateTriggers: { getRadius: closeable },
    }),
  ];
  return layers;
}

function rectanglePreview(a: XY, b: XY): XY[] {
  return [a, [b[0], a[1]], b, [a[0], b[1]], a];
}
