/**
 * Composite rendering: every channel of a small multichannel image, each in its
 * own colour, blended into one picture.
 *
 * Everything here is arithmetic on the grid and the store's values, kept out of
 * the viewer so it can be tested without a WebGL context -- the same split as
 * `vivUtils`.
 */

import { sliderAxes, type TileInfo } from "@biopb/tensor-flight-client";
import { guessDefaultColor, type ColorValue } from "./colorUtils";
import {
  contrastLimitsFrom,
  percentileBounds,
  vivSelection,
  type SliceIndices,
} from "./vivUtils";

/**
 * The most channels one composite draws. Viv's shader is compiled per channel
 * count and the tile traffic grows with it; four covers the usual
 * DAPI / green / red / far-red panel, and past it the picture stops being
 * legible as colours anyway.
 */
export const COMPOSITE_MAX_CHANNELS = 4;

/** The tensor's named channel axis, if it has one. */
function channelAxis(info: TileInfo) {
  return sliderAxes(info.dim_labels, info.shape).find((a) => a.named === "c");
}

/**
 * The channel indices a composite would draw for this tensor, or null when it
 * does not apply: not asked for, interleaved RGB (already colour), no channel
 * axis, a single channel, or more than {@link COMPOSITE_MAX_CHANNELS}.
 *
 * "Asked for" is a preference that outlives a tensor, so this is what decides
 * whether it takes effect on the one in view.
 */
export function compositeChannels(info: TileInfo, composite: boolean): number[] | null {
  if (!composite || info.plane.s !== null) return null;
  const c = channelAxis(info);
  if (!c || c.extent < 2 || c.extent > COMPOSITE_MAX_CHANNELS) return null;
  return Array.from({ length: c.extent }, (_, i) => i);
}

/** One Viv selection per channel, at the plane `position` names on every other axis. */
export function compositeSelections(
  info: TileInfo,
  position: SliceIndices,
  channels: readonly number[],
): Record<string, number>[] {
  return channels.map((c) => vivSelection(info, { ...position, c }));
}

/**
 * `selection` moved to channel `c`.
 *
 * A composite is drawn from selections that all differ in `c`, but an
 * annotation belongs to one plane, and the plane the C slider names is the one
 * a user means. This is that plane, on whatever t/z/axes `selection` is at.
 */
export function withChannel(
  info: TileInfo,
  selection: Record<string, number>,
  c: number,
): Record<string, number> {
  const axis = channelAxis(info);
  if (!axis) return selection;
  return { ...selection, [axis.key]: Math.min(Math.max(0, c), Math.max(0, axis.extent - 1)) };
}

/**
 * Colours tried, in order, for a channel nothing else names.
 *
 * Keyed by channel count so the common cases get the conventional pairing:
 * green over magenta is the colour-blind-safe two-channel composite, and
 * red/green/blue is what three channels are read as. A fourth falls to grey,
 * which adds to the others without competing with them.
 */
const FALLBACK_BY_COUNT: Record<number, readonly string[]> = {
  2: ["green", "magenta"],
  3: ["red", "green", "blue"],
  4: ["red", "green", "blue", "gray"],
};

/** What is left once the count's own list is spent. */
const FALLBACK_REST = ["green", "magenta", "cyan", "yellow", "red", "blue"];

/**
 * A colour per channel for a composite.
 *
 * A colour the user picked stands. Next a channel's name is allowed to imply one
 * (DAPI is blue, GFP is green), first come first served: names collide -- Cy5
 * and Cy7 both read as far-red, which is magenta -- and two channels in one
 * colour cannot be told apart once blended. The loser, and every channel with
 * no recognisable name (which would be grey in a single-channel view), takes the
 * first colour of the count's fallback list that nothing else is wearing. Grey
 * is never *guessed* for a composite, since it is the one colour that says
 * nothing about the channel.
 */
export function compositeColors(
  stored: readonly ColorValue[],
  names: readonly (string | undefined)[],
): ColorValue[] {
  const out: (ColorValue | null)[] = stored.map((color) => (color === "auto" ? null : color));
  const used = new Set(out.filter((c): c is ColorValue => c !== null));
  out.forEach((color, i) => {
    if (color !== null) return;
    const name = names[i];
    const guess = name ? guessDefaultColor(name) : "gray";
    if (guess !== "gray" && !used.has(guess)) {
      out[i] = guess;
      used.add(guess);
    }
  });
  const pool = [...(FALLBACK_BY_COUNT[stored.length] ?? []), ...FALLBACK_REST];
  return out.map((color) => {
    if (color !== null) return color;
    const next = pool.find((c) => !used.has(c)) ?? "gray";
    used.add(next);
    return next;
  });
}

/**
 * The contrast window of each channel: the percentile window of that channel's
 * own samples. Null where a channel has not been sampled yet, which the viewer
 * leaves hidden rather than drawing it through a window that is a guess.
 */
export function compositeWindows(
  channels: readonly number[],
  samples: Readonly<Record<number, { values: Float64Array }>>,
  percentileScale: number,
): ([number, number] | null)[] {
  const [lo, hi] = percentileBounds(percentileScale);
  return channels.map((c) => {
    const s = samples[c];
    return s ? contrastLimitsFrom(s.values, lo, hi) : null;
  });
}
