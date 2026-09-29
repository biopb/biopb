/**
 * Which axes an annotation can be pinned to, and whether one shows on the plane
 * currently in view.
 *
 * Pins are keyed by **wire axis index**, matching `RoiAnnotation.plane`. A label
 * cannot address every axis -- a TIFF sequence's opaque file axis has none, and
 * two axes of one tensor may share one -- and an axis a pin cannot name is an
 * axis every annotation silently broadcasts across.
 *
 * Pure and free of app state: the SPA's slice selection is keyed by
 * `SliderAxis.key`, so the caller translates its own selection into
 * `indexByAxis` and this owns the rest.
 */

import { sliderAxes, type SliderAxis } from "./tensor-array.js";
import type { TileInfo } from "./types.js";

/**
 * The axes a new annotation can be pinned to.
 *
 * `sliderAxes` already drops the display plane's own axes (`y`, `x`, and `s` for
 * interleaved RGB), which geometry addresses rather than the pin. This drops one
 * more: an axis of extent 1, where a pin says nothing because every annotation
 * is on index 0.
 *
 * Deliberately derived from `sliderAxes` rather than from `TileInfo.plane`: the
 * pin has to name axes the user can actually navigate, and those are the ones
 * the sliders offer. Two derivations of "is this a plane axis?" could disagree,
 * and the pin would then name an axis with no control behind it.
 */
export function pinnableAxes(tileInfo: TileInfo): SliderAxis[] {
  return sliderAxes(tileInfo.dim_labels, tileInfo.shape).filter((axis) => axis.extent > 1);
}

/** The `axis -> index` pin for a view sitting at `indexByAxis`. */
export function planePinFor(
  axes: SliderAxis[],
  indexByAxis: Record<number, number>,
): Record<number, number> {
  const pin: Record<number, number> = {};
  for (const { axis } of axes) {
    const index = indexByAxis[axis];
    if (typeof index === "number" && Number.isFinite(index)) {
      pin[axis] = Math.trunc(index);
    }
  }
  return pin;
}

/**
 * Whether an annotation shows on the plane described by `current`.
 *
 * A dimension **absent** from the annotation's pin applies at every index of
 * that dimension -- that is what lets one ROI follow a z-stack or a time course
 * without being duplicated per plane, and reading it the other way round would
 * hide nearly everything.
 *
 * A pin naming an axis `current` says nothing about is treated as visible: the
 * pin cannot be evaluated, and showing an extra ROI is recoverable where
 * silently hiding one is not.
 */
export function roiVisibleOnPlane(
  roiPlane: Record<number, number>,
  current: Record<number, number>,
): boolean {
  for (const [key, index] of Object.entries(roiPlane)) {
    const here = current[Number(key)];
    if (here !== undefined && here !== index) return false;
  }
  return true;
}
