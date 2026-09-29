import type { AppState } from "./types";

/**
 * Where in the tensor's grid the view is. Per tensor: an index means "index of
 * the tensor in view", so `openTensor` resets it.
 *
 * Separate from {@link DisplayState} because the two change for different
 * reasons and cost differently: a new position asks the server for other
 * pixels, a new display setting re-shades the same ones. Keeping the objects
 * apart is what lets a contrast drag leave `position` -- and everything keyed
 * on its identity: the Viv selection, the volume request, a draft's plane --
 * untouched.
 */
export interface PositionState {
  t: number;
  z: number;
  c: number;
  /**
   * Index chosen on each axis `t`/`z`/`c` cannot name, keyed by `SliderAxis.key`
   * (`a0`, `a3`, ...).
   *
   * A TIFF sequence's `i`, a plate's `POS`, the second of two axes sharing a
   * label: navigable, but with no semantic name to hold them under. Reset with
   * t/z/c on a source change, and for the same reason -- a key means "axis 0 of
   * the tensor in view", so it does not survive one.
   */
  axes: Record<string, number>;
}

/**
 * How the image is shaded. A preference that outlives a tensor, except
 * `fixedLimits`, which is a value of one tensor's dtype.
 */
export interface DisplayState {
  /**
   * How the contrast window is chosen: from the plane's own histogram, or from
   * two grey levels the user fixed.
   *
   * Fixed is what makes two planes comparable -- an automatic window rescales
   * per plane, so a channel that dims over a timelapse looks constant under it.
   */
  contrastMode: "auto" | "fixed";
  /** Width of the automatic percentile window: 0 = min-max, 1 = 1-99, 2 = 2-98. */
  percentileScale: number;
  /**
   * The window in `fixed` mode, in raw grey levels, or null for "not chosen".
   *
   * Null seeds from whatever is on screen when the mode is turned on, so
   * switching to fixed does not change the image. Dropped on a source change:
   * a level is a value of *this* tensor's dtype, and a uint16 window carried
   * onto a uint8 image is a white frame.
   */
  fixedLimits: [number, number] | null;
  // Display-only exponent applied to the normalized intensity, after the
  // contrast window and before the channel color. 1 leaves the ramp linear;
  // below 1 lifts the dim end, above 1 pushes it down.
  gamma: number;
}

/**
 * Position and display as one record. Only the URL codec uses it: a link names
 * both in one flat set of parameters.
 */
export type SliceState = PositionState & DisplayState;

export const pickPosition = ({ t, z, c, axes }: SliceState): PositionState => ({ t, z, c, axes });
export const pickDisplay = ({ contrastMode, percentileScale, fixedLimits, gamma }: SliceState): DisplayState => ({
  contrastMode,
  percentileScale,
  fixedLimits,
  gamma,
});

/**
 * The patch for moving to `next`: nothing when it is where the view already is,
 * else the new position with the plane marked not ready -- a new position always
 * means the canvas is no longer showing the plane asked for, and saying so in the
 * same write is what keeps a play timer from racing the viewer's own effect.
 */
export function moved(s: AppState, next: PositionState): Partial<AppState> {
  if (samePosition(s.position, next)) return {};
  return { position: next, runtime: { ...s.runtime, planeReady: false } };
}

/** `next`, unless it holds the same indices as `current`, in which case `current`. */
export function samePositionOr(current: PositionState, next: PositionState): PositionState {
  return samePosition(current, next) ? current : next;
}

/** Same indices, so a write that changes nothing can keep the object. */
function samePosition(a: PositionState, b: PositionState): boolean {
  if (a.t !== b.t || a.z !== b.z || a.c !== b.c) return false;
  const keys = Object.keys(a.axes);
  return keys.length === Object.keys(b.axes).length && keys.every((k) => a.axes[k] === b.axes[k]);
}

/**
 * A stable key for the slice position, for spotting that it has moved.
 *
 * Only the indices: contrast, gamma and the percentile window ride `SliceState`
 * too, and none of them invalidate a shape being drawn.
 */
export function sliceKey(slice: PositionState): string {
  return `${slice.t}|${slice.z}|${slice.c}|${JSON.stringify(slice.axes)}`;
}
