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
      const { center, radius } = geometry;
      const ring: XY[] = [];
      for (let i = 0; i < segments; i++) {
        const theta = (2 * Math.PI * i) / segments;
        // Axis-aligned: biopb.image.Ellipse carries no rotation (biopb#935).
        // When it gains one, it rotates this pair about the centre and nothing
        // else here changes.
        ring.push([center.x + radius.x * Math.cos(theta), center.y + radius.y * Math.sin(theta)]);
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
  /** Drawn brighter and thicker, so the panel and the canvas agree. */
  selectedRoiId?: string | null;
}

/**
 * The overlay's layers, ready for `VivViewer`'s `deckProps.layers`.
 *
 * `pickable: false` throughout, and not only because nothing is interactive
 * yet: the hover readout in `TileViewer` reads `info.sourceLayer` and
 * `info.tile` to report the pixel under the pointer, so a pickable overlay on
 * top would answer that hover itself and blank the readout wherever an
 * annotation lies. Phase 3 has to route around that rather than just flip this.
 */
export function buildRoiLayers(options: RoiLayerOptions): unknown[] {
  const { rois, currentPlane, hiddenSets, visible, selectedRoiId = null } = options;
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
        getFillColor: (d: { roi: RoiAnnotation }) => [...setColor(d.roi.setName), 40],
        getLineColor: (d: { roi: RoiAnnotation }) =>
          d.roi.roiId === selectedRoiId ? SELECTED_COLOR : [...setColor(d.roi.setName), 230],
        // Pixels, not world units: an outline is a way of seeing the shape, so
        // it should not thin out as the user zooms out of a large field.
        lineWidthUnits: "pixels",
        getLineWidth: (d: { roi: RoiAnnotation }) => (d.roi.roiId === selectedRoiId ? 3 : 1.5),
        updateTriggers: { getLineColor: selectedRoiId, getLineWidth: selectedRoiId },
      }),
    );
  }

  if (paths.length > 0) {
    layers.push(
      new PathLayer({
        id: roiLayerId("paths"),
        data: paths,
        pickable: false,
        // World units here, unlike the outlines above: a polyline's width is
        // geometry -- the band of pixels the stroke covers -- so it has to scale
        // with the image. The pixel floor only keeps a hairline visible.
        widthUnits: "common",
        widthMinPixels: 1.5,
        capRounded: true,
        jointRounded: true,
        getPath: (d: { path: XY[] }) => d.path,
        getColor: (d: { roi: RoiAnnotation }) =>
          d.roi.roiId === selectedRoiId ? SELECTED_COLOR : [...setColor(d.roi.setName), 230],
        getWidth: (d: { roi: RoiAnnotation }) =>
          d.roi.geometry.kind === "polyline" ? Math.abs(d.roi.geometry.width) : 0,
        updateTriggers: { getColor: selectedRoiId },
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
        getLineWidth: 1.5,
        getPosition: (d: RoiAnnotation) =>
          d.geometry.kind === "point" ? [d.geometry.at.x, d.geometry.at.y] : [0, 0],
        getFillColor: (d: RoiAnnotation) => [...setColor(d.setName), 90],
        getLineColor: (d: RoiAnnotation) =>
          d.roiId === selectedRoiId ? SELECTED_COLOR : [...setColor(d.setName), 230],
        updateTriggers: { getLineColor: selectedRoiId },
      }),
    );
  }

  return layers;
}

// ---------------------------------------------------------------------------
// The shape being drawn
// ---------------------------------------------------------------------------

/** Colour of an in-progress draft: distinct from every set colour. */
const DRAFT_COLOR: [number, number, number, number] = [250, 204, 21, 240];

export interface DraftLayerOptions {
  draft: RoiDraft | null;
  /** Where the pointer is, for the segment that trails it. Null when off-image. */
  cursor: XY | null;
  /** The draft has enough vertices to close, so its first vertex is a handle. */
  closeable: boolean;
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
  const { draft, cursor, closeable } = options;
  if (!draft || draft.points.length === 0) return [];

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
