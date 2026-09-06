/**
 * Which annotation is under the pointer.
 *
 * Done in JS, not by deck.gl picking, and that is deliberate. The overlay's
 * layers stay `pickable: false` because TileViewer's hover badge reads the pixel
 * under the pointer from `info.sourceLayer` / `info.tile`: a pickable overlay
 * would answer that hover itself and blank the readout wherever an annotation
 * lies. Hit-testing here costs nothing extra, because the whole annotation set
 * is already resident -- which is one of the reasons the read path fetches it
 * whole (docs/roi-annotations-ui.md).
 */

import type { RoiAnnotation } from "@biopb/tensor-flight-client";
import type { XY } from "./roiLayers";
import { roiPath, roiRing } from "./roiLayers";

/** How far from a stroke or a point still counts as a hit, in SCREEN pixels. */
export const HIT_SLOP_PX = 6;

/** Even-odd point-in-polygon. */
export function pointInRing(at: XY, ring: XY[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const a = ring[i] as XY;
    const b = ring[j] as XY;
    const straddles = a[1] > at[1] !== b[1] > at[1];
    if (straddles && at[0] < ((b[0] - a[0]) * (at[1] - a[1])) / (b[1] - a[1]) + a[0]) {
      inside = !inside;
    }
  }
  return inside;
}

/** Distance from a point to a segment, in the same units as the inputs. */
export function distanceToSegment(at: XY, a: XY, b: XY): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq === 0) return Math.hypot(at[0] - a[0], at[1] - a[1]);
  // Clamped projection, so the nearest point stays on the segment rather than
  // running off its infinite line.
  let t = ((at[0] - a[0]) * dx + (at[1] - a[1]) * dy) / lengthSq;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(at[0] - (a[0] + t * dx), at[1] - (a[1] + t * dy));
}

function distanceToPath(at: XY, path: XY[]): number {
  let best = Infinity;
  for (let i = 1; i < path.length; i++) {
    best = Math.min(best, distanceToSegment(at, path[i - 1] as XY, path[i] as XY));
  }
  return best;
}

/**
 * Whether `at` hits this annotation, with `slop` in world units.
 *
 * A filled shape hits anywhere inside it *or* within slop of its outline, so a
 * thin sliver is still selectable; an open path and a point hit by proximity.
 */
export function hitsRoi(roi: RoiAnnotation, at: XY, slop: number): boolean {
  if (roi.geometry.kind === "point") {
    const { x, y } = roi.geometry.at;
    return Math.hypot(at[0] - x, at[1] - y) <= slop;
  }
  const path = roiPath(roi.geometry);
  if (path) return path.length > 1 && distanceToPath(at, path) <= slop;
  const ring = roiRing(roi.geometry);
  if (!ring || ring.length === 0) return false;
  if (pointInRing(at, ring)) return true;
  return distanceToPath(at, [...ring, ring[0] as XY]) <= slop;
}

/**
 * The annotation under `at`, or null.
 *
 * `scale` is world units per screen pixel, so the tolerance is constant on
 * screen at any zoom. Searched from the end: later annotations are drawn over
 * earlier ones, so the last hit is the one the user sees on top.
 */
export function roiAt(rois: RoiAnnotation[], at: XY, scale: number): RoiAnnotation | null {
  const slop = HIT_SLOP_PX * Math.max(scale, Number.EPSILON);
  for (let i = rois.length - 1; i >= 0; i--) {
    const roi = rois[i] as RoiAnnotation;
    if (hitsRoi(roi, at, slop)) return roi;
  }
  return null;
}
