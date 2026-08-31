// How wide the chat pane is, and what keeps that width sane.
//
// Split out of ObservePage because it is the one part of the layout with a rule
// rather than a value: a width is remembered per browser, and a width that made
// sense on the monitor it was chosen on can leave no room for the job list on
// the next one.

/** Narrower than this and the composer stops being usable. */
export const MIN_CHAT_WIDTH = 300;

/** Whatever is left must still show a job row's code preview. */
export const MIN_WORK_WIDTH = 360;

/** The width with nothing dragged. Mirrors the CSS fallback
 * `clamp(340px, 34%, 520px)`, which is what actually sizes an untouched pane —
 * this exists for the keyboard path, which needs a number to add to.
 *
 * A proportion rather than a fixed number because that is what reads well on
 * both a laptop and a wide monitor. */
export function defaultChatWidth(viewport: number): number {
  return Math.round(Math.min(Math.max(viewport * 0.34, 340), 520));
}

/**
 * A width that keeps both columns usable, or 0 when *width* is not a number.
 *
 * The upper bound follows the viewport rather than being a constant: a width
 * stored from a wide monitor would otherwise swallow the whole column on a
 * laptop, and the person would find the job list gone with no way to see that
 * a remembered preference was the cause.
 *
 * 0 for a non-number so a corrupt stored value falls back to the default rather
 * than pinning the pane at the minimum — the two are easy to confuse, and one
 * of them looks like the feature is broken.
 */
export function clampChatWidth(width: number, viewport: number): number {
  if (!Number.isFinite(width)) return 0;
  const max = Math.max(MIN_CHAT_WIDTH, viewport - MIN_WORK_WIDTH);
  return Math.round(Math.min(Math.max(width, MIN_CHAT_WIDTH), max));
}
