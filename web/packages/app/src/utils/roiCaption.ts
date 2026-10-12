/**
 * How big a caption is on screen.
 *
 * A caption is drawn in screen pixels (it keeps its size as the user zooms, like
 * the point marker), so its extent in the image changes with the zoom and the
 * store keeps only its anchor. Hit-testing needs that extent, and it has to be
 * answerable without a canvas, so it is an estimate from the text rather than a
 * measurement: a few pixels either way is inside the hit slop.
 */

/** Font size of a caption, in screen pixels. */
export const CAPTION_SIZE_PX = 14;
/** Distance between baselines, as a multiple of the font size. */
export const CAPTION_LINE_HEIGHT = 1.2;

/**
 * Advance of a Latin glyph, as a multiple of the font size. A little over the
 * average (about 0.55), so a run of wide capitals still falls inside the box.
 */
const NARROW_EM = 0.65;
/** CJK, Hangul, full-width forms and emoji advance a full em. */
const WIDE = /[\u{1100}-\u{115f}\u{2e80}-\u{a4cf}\u{ac00}-\u{d7a3}\u{f900}-\u{faff}\u{fe30}-\u{fe6f}\u{ff00}-\u{ff60}\u{ffe0}-\u{ffe6}\u{1f300}-\u{1faff}\u{20000}-\u{3fffd}]/u;

export interface CaptionSize {
  width: number;
  height: number;
}

/** The text block's estimated size in screen pixels. */
export function captionSizePx(text: string, sizePx: number = CAPTION_SIZE_PX): CaptionSize {
  const lines = text.split("\n");
  let widest = 0;
  for (const line of lines) {
    let em = 0;
    for (const ch of line) em += WIDE.test(ch) ? 1 : NARROW_EM;
    widest = Math.max(widest, em);
  }
  return { width: widest * sizePx, height: lines.length * sizePx * CAPTION_LINE_HEIGHT };
}
