/**
 * The shape being drawn: a small state machine over placed vertices.
 *
 * Pure, so the interaction is testable without a canvas — the DOM wiring in
 * TileViewer only turns pointer events into these calls.
 *
 * **Creation is click-to-place, not drag-to-trace.** A layer's drag does not
 * stop the OrthographicView controller (deck.gl dispatches picking events and
 * the controller independently), so a drag-traced shape would pan the canvas out
 * from under itself. Clicking needs none of that, and for tracing an outline at
 * zoom it is the better interaction anyway. See docs/roi-annotations-ui.md.
 */

import type { RoiGeometry } from "@biopb/tensor-flight-client";
import type { XY } from "./roiLayers";

/** What the pointer places. `select` draws nothing and picks instead. */
export type RoiTool = "select" | "point" | "rectangle" | "polygon" | "polyline";

/** Tools that build a shape from a run of clicks rather than a fixed count. */
export const OPEN_ENDED_TOOLS: ReadonlySet<RoiTool> = new Set(["polygon", "polyline"]);

export interface RoiDraft {
  tool: Exclude<RoiTool, "select">;
  /** Vertices placed so far, in level-0 pixel coordinates. */
  points: XY[];
}

/**
 * How close to the first vertex a click must land to close the shape, in
 * SCREEN pixels -- a world-space radius would grow unusable at low zoom and
 * unhittable at high zoom.
 */
export const CLOSE_HANDLE_PX = 10;

/** Fewest vertices each tool needs before it can be completed. */
export function minimumPoints(tool: Exclude<RoiTool, "select">): number {
  switch (tool) {
    case "point":
      return 1;
    case "rectangle":
      return 2;
    case "polyline":
      return 2;
    case "polygon":
      return 3;
  }
}

const TEXT_ENTRY_TAGS = new Set(["INPUT", "TEXTAREA", "SELECT"]);

/**
 * Whether a keystroke belongs to something being typed into rather than to the
 * drawing tools.
 *
 * The draft's keys are bound on `window`, because the canvas is not focusable
 * and a shape has to be finishable wherever the pointer is. That reach is the
 * problem: the viewer route keeps the source search, the chat composer, the
 * annotation's own label and set fields, and the slice inputs all within it, and
 * Backspace would take back a vertex instead of a character in any of them.
 */
export function isTextEntryTarget(target: EventTarget | null): boolean {
  const el = target as (HTMLElement & { tagName?: unknown }) | null;
  if (!el || typeof el.tagName !== "string") return false;
  if (el.isContentEditable) return true;
  return TEXT_ENTRY_TAGS.has(el.tagName.toUpperCase());
}

/** A draft that has enough vertices to become a shape. */
export function isCompletable(draft: RoiDraft | null): boolean {
  return draft !== null && draft.points.length >= minimumPoints(draft.tool);
}

export interface PlaceResult {
  /** The draft after the click, or null once it turned into a geometry. */
  draft: RoiDraft | null;
  /** Set when the click completed a shape. */
  completed?: RoiGeometry;
}

/**
 * Place one vertex.
 *
 * A point and a rectangle complete on their own -- one and two clicks -- because
 * their vertex count is fixed and asking for a separate "finish" would be
 * ceremony. A polygon or polyline runs until {@link closeDraft}, since only the
 * user knows where the outline ends.
 */
export function placePoint(draft: RoiDraft | null, tool: RoiTool, at: XY): PlaceResult {
  if (tool === "select") return { draft };
  const points = draft && draft.tool === tool ? [...draft.points, at] : [at];

  if (tool === "point") {
    return { draft: null, completed: { kind: "point", at: { x: at[0], y: at[1] } } };
  }
  if (tool === "rectangle" && points.length === 2) {
    const [a, b] = points as [XY, XY];
    // Normalised, so the stored corners are top-left and bottom-right whichever
    // way the user dragged their eye across the image.
    return {
      draft: null,
      completed: {
        kind: "rectangle",
        topLeft: { x: Math.min(a[0], b[0]), y: Math.min(a[1], b[1]) },
        bottomRight: { x: Math.max(a[0], b[0]), y: Math.max(a[1], b[1]) },
      },
    };
  }
  return { draft: { tool, points } };
}

/**
 * Width a new polyline gets, in image pixels.
 *
 * Not zero, which is what this used to be: a stroke of no extent renders at the
 * layer's hairline floor whatever the zoom, so every polyline came out looking
 * like a construction line rather than something traced over structure. Width
 * is geometry here -- the band of pixels the stroke covers -- so the default has
 * to be a width somebody would have chosen, not the additive identity.
 */
export const DEFAULT_POLYLINE_WIDTH = 4;

/**
 * What the width control offers, in image pixels.
 *
 * The floor is 1 rather than 0: zero is a legitimate stored value, but it is
 * not one to hand somebody a slider for -- it draws as a hairline and nothing
 * about the control would say why.
 */
export const MIN_POLYLINE_WIDTH = 1;
export const MAX_POLYLINE_WIDTH = 64;

/** Hold a width inside the range the control offers. */
export function clampPolylineWidth(width: number): number {
  if (!Number.isFinite(width)) return DEFAULT_POLYLINE_WIDTH;
  return Math.min(MAX_POLYLINE_WIDTH, Math.max(MIN_POLYLINE_WIDTH, width));
}

/**
 * Finish an open-ended draft, or return null when it has too few vertices.
 *
 * Null rather than a partial shape: a two-vertex "polygon" is not a lesser
 * polygon, it is a line the store would reject anyway.
 */
export function closeDraft(
  draft: RoiDraft | null,
  polylineWidth: number = DEFAULT_POLYLINE_WIDTH,
): RoiGeometry | null {
  if (!isCompletable(draft) || !draft) return null;
  const points = draft.points.map(([x, y]) => ({ x, y }));
  if (draft.tool === "polygon") return { kind: "polygon", points };
  if (draft.tool === "polyline") {
    return { kind: "polyline", points, width: clampPolylineWidth(polylineWidth) };
  }
  return null;
}

/** Undo the last placed vertex; an empty draft becomes null. */
export function undoPoint(draft: RoiDraft | null): RoiDraft | null {
  if (!draft) return null;
  const points = draft.points.slice(0, -1);
  return points.length === 0 ? null : { ...draft, points };
}

/**
 * Whether a click at `at` should close the draft by landing on its first vertex.
 *
 * `scale` is world units per screen pixel, so the handle keeps a constant size
 * on screen at any zoom.
 */
export function closesOnFirstVertex(draft: RoiDraft | null, at: XY, scale: number): boolean {
  if (!draft || !OPEN_ENDED_TOOLS.has(draft.tool) || !isCompletable(draft)) return false;
  const first = draft.points[0];
  if (!first) return false;
  const radius = CLOSE_HANDLE_PX * Math.max(scale, Number.EPSILON);
  return Math.hypot(at[0] - first[0], at[1] - first[1]) <= radius;
}
