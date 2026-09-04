/**
 * How the slice controls present and animate the navigable axes.
 *
 * Kept out of the component for the reason `vivUtils` is: the ordering, the
 * thumb geometry and the playback pacing are all arithmetic, and none of it
 * needs a DOM to be checked.
 */

import type { SliderAxis } from "@biopb/tensor-flight-client";

/**
 * Named axes, outermost-looking first: Z, then C, then T.
 *
 * Display order only — the wire order stays whatever `sliderAxes` reports, and
 * nothing here touches how an axis is addressed. Z leads because it is the one
 * a user scrubs while looking at a single field; T is the slowest-moving of the
 * three and sits at the bottom next to nothing.
 */
export const AXIS_DISPLAY_ORDER = ["z", "c", "t"] as const;

/**
 * Reorder for display: named axes as {@link AXIS_DISPLAY_ORDER}, then the rest.
 *
 * Unnamed axes keep their wire order and follow the named ones. There is no
 * meaning to sort them by — an `i` or a `POS` is whatever the source called it
 * — so the only stable answer is the order the tensor lists them in.
 */
export function orderSliderAxes(axes: SliderAxis[]): SliderAxis[] {
  const rank = (axis: SliderAxis) => {
    const i = axis.named ? AXIS_DISPLAY_ORDER.indexOf(axis.named) : -1;
    return i < 0 ? AXIS_DISPLAY_ORDER.length : i;
  };
  // Index tiebreak rather than a bare `rank` diff: `sort` is stable in every
  // engine this runs on, but saying so costs one comparison and removes the
  // question.
  return axes
    .map((axis, i) => ({ axis, i }))
    .sort((a, b) => rank(a.axis) - rank(b.axis) || a.i - b.i)
    .map((e) => e.axis);
}

// ---------------------------------------------------------------------------
// Thumb geometry
// ---------------------------------------------------------------------------

/**
 * Narrowest thumb: below this an axis is no longer draggable by pointer.
 *
 * The floor bites only past a few dozen positions, where the true share is a
 * sliver anyway and the size has stopped carrying information.
 */
export const THUMB_MIN_PX = 12;

/**
 * How wide to draw the grab handle for an axis of `positions` steps on a track
 * of `trackPx`.
 *
 * A scrollbar's thumb is its share of the track, so a 2-channel axis gets half
 * the bar. That needs the measured track width -- CSS knows it and the
 * component has to look it up (see `useTrackWidth`), because a thumb cannot be
 * sized as a percentage of a range input.
 */
export function sliderThumbPx(positions: number, trackPx: number): number {
  // Before the first measurement there is no share to take; the floor keeps the
  // slider usable for the frame it takes to land.
  if (!Number.isFinite(trackPx) || trackPx <= 0) return THUMB_MIN_PX;
  if (!Number.isFinite(positions) || positions <= 1) return Math.round(trackPx);
  const share = Math.round(trackPx / positions);
  return Math.max(THUMB_MIN_PX, Math.min(Math.round(trackPx), share));
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

/** Frames per second play asks for, when the data plane can keep up. */
export const PLAY_FPS = 10;
/** The configured frame delay: what {@link PLAY_FPS} means in milliseconds. */
export const PLAY_FRAME_MS = Math.round(1000 / PLAY_FPS);
/** How often to re-check whether the frame in flight has landed. */
export const PLAY_READY_POLL_MS = 25;
/**
 * How long a frame may fail to arrive before play steps past it anyway.
 *
 * Play paces itself to the data plane, which means a plane that never loads
 * would otherwise stop the animation with no way to tell it from a very slow
 * one. Stepping on keeps a broken plane from freezing the whole sequence.
 */
export const PLAY_STALL_MS = 5_000;

/** The next index in a looping scrub. */
export function nextPlayIndex(current: number, extent: number): number {
  if (extent <= 1) return 0;
  // Guards a stored index that is out of range for this tensor: the modulo
  // alone would map a negative one to a negative one.
  const at = Math.max(0, Math.min(extent - 1, Math.trunc(current)));
  return (at + 1) % extent;
}
