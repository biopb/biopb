/**
 * Which axes an annotation can be pinned to, and whether one shows on the plane
 * currently in view.
 *
 * Pure and free of any app state on purpose: the SPA's slice selection is keyed
 * by `SliderAxis.key` (`a0`, `a3`), while the proto keys `plane` by *dim label*.
 * The caller translates its own selection into `indexByAxis` (wire index ->
 * index) and this owns the label half.
 */

import type { TileInfo } from "./types.js";

/** A non-plane axis that can carry a plane pin, with the label it is keyed by. */
export interface PinnableAxis {
  /** Wire index, i.e. position in `dim_labels`/`shape`. */
  axis: number;
  label: string;
}

/**
 * The axes a new annotation can be pinned to.
 *
 * Excluded, in each case because there is no key to hold the pin under:
 *
 * - The display plane's own axes (`y`, `x`, and `s` for interleaved RGB) --
 *   geometry addresses those, not the pin.
 * - Axes of extent 1: pinning one says nothing, since every ROI is on index 0.
 * - **Unlabelled or duplicate-labelled axes.** A TIFF sequence's `i` or the
 *   second of two axes sharing a label cannot be named, and inventing a
 *   synthetic key (`"3"`, say) would be a client-side dialect the Python SDK
 *   does not share. An ROI on such a tensor is simply not pinned on that axis,
 *   so it shows at every index of it.
 */
export function pinnableAxes(tileInfo: TileInfo): PinnableAxis[] {
  const planeAxes = new Set<number>([tileInfo.plane.y, tileInfo.plane.x]);
  if (tileInfo.plane.s !== null) planeAxes.add(tileInfo.plane.s);

  const seen = new Map<string, number>();
  for (const label of tileInfo.dim_labels) {
    seen.set(label, (seen.get(label) ?? 0) + 1);
  }

  const out: PinnableAxis[] = [];
  tileInfo.dim_labels.forEach((label, axis) => {
    if (planeAxes.has(axis)) return;
    if ((tileInfo.shape[axis] ?? 1) <= 1) return;
    if (!label || (seen.get(label) ?? 0) > 1) return;
    out.push({ axis, label });
  });
  return out;
}

/** The `dim_label -> index` pin for a view sitting at `indexByAxis`. */
export function planePinFor(
  axes: PinnableAxis[],
  indexByAxis: Record<number, number>,
): Record<string, number> {
  const pin: Record<string, number> = {};
  for (const { axis, label } of axes) {
    const index = indexByAxis[axis];
    if (typeof index === "number" && Number.isFinite(index)) {
      pin[label] = Math.trunc(index);
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
 * A pin naming a dimension `current` says nothing about is treated as visible:
 * the pin cannot be evaluated, and showing an extra ROI is recoverable where
 * silently hiding one is not.
 */
export function roiVisibleOnPlane(
  roiPlane: Record<string, number>,
  current: Record<string, number>,
): boolean {
  for (const [label, index] of Object.entries(roiPlane)) {
    const here = current[label];
    if (here !== undefined && here !== index) return false;
  }
  return true;
}
