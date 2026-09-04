// How wide the viewer's two side panes may be.
//
// Split out of HomePage for the reason `chatPaneWidth` is split out of
// ObservePage: a remembered width is the one part of a layout with a rule
// rather than a value. A width chosen on a wide monitor must not swallow the
// canvas on the next one, and with two panes the space each may take depends on
// what the other is already using.

/** Narrower than this and a source path is unreadable. */
export const MIN_SIDEBAR_WIDTH = 180;

/** Narrower than this and the slider rows start wrapping. */
export const MIN_CONTROL_WIDTH = 240;

/** Whatever is left has to be worth calling a viewer. */
export const MIN_CANVAS_WIDTH = 320;

/** The widths with nothing dragged; the CSS fallbacks say the same. */
export const DEFAULT_SIDEBAR_WIDTH = 320;
export const DEFAULT_CONTROL_WIDTH = 320;

/**
 * A width that leaves the other pane and the canvas usable, or 0 when `width`
 * is not a number.
 *
 * The upper bound follows the viewport and the *other* pane rather than being a
 * constant, so widening one pane on a narrow window stops where the canvas
 * would start disappearing instead of taking it silently.
 *
 * 0 for a non-number so a corrupt stored value falls back to the default rather
 * than pinning the pane at its minimum -- the two look alike, and one of them
 * reads as a broken layout.
 */
export function clampPaneWidth(
  width: number,
  min: number,
  otherPane: number,
  viewport: number,
): number {
  if (!Number.isFinite(width)) return 0;
  const max = Math.max(min, viewport - otherPane - MIN_CANVAS_WIDTH);
  return Math.round(Math.min(Math.max(width, min), max));
}
