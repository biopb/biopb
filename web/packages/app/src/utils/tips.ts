/**
 * The things about this viewer nobody can find by looking.
 *
 * Each tip earns its place by answering a question a first-time user would
 * otherwise have to ask — most often "why does it look broken?" (a black plane,
 * a plane that is not the one selected, a source list that is too short) — or by
 * naming a gesture with no visible affordance.
 *
 * Tips carry a `when` so the bar shows what is true right now: a tip about the
 * metadata panel while no source is selected is noise, and noise is what makes
 * people stop reading the strip that will later tell them something useful.
 */

export interface TipContext {
  /** Sources currently loaded in the tree. */
  sourceCount: number;
  /** A source *and* a tensor are selected, i.e. the viewer and panels exist. */
  hasSelection: boolean;
  /** The catalog scan is still running, so the list is not final. */
  scanning: boolean;
}

export interface Tip {
  /** Stable across edits to the text: it is the rotation cursor. */
  id: string;
  text: string;
  when?: (ctx: TipContext) => boolean;
}

const selected = (ctx: TipContext) => ctx.hasSelection;

/**
 * The threshold in SourceTree at which the search box stops filtering the
 * loaded list and starts asking the server.
 */
const SERVER_QUERY_THRESHOLD = 1000;

export const TIPS: Tip[] = [
  {
    id: "slice-scroll",
    text: "Hold T, Z or C and scroll over the image to step through that axis.",
    when: selected,
  },
  {
    id: "meta-expand",
    text: "Metadata rows expand and collapse; long lists stop at 10 and show “… N more” to reveal the rest.",
    when: selected,
  },
  {
    id: "server-filter",
    text: "With a large catalog the search box filters on the server, across the whole catalog rather than the loaded list.",
    when: (ctx) => ctx.sourceCount > SERVER_QUERY_THRESHOLD,
  },
  {
    id: "channel-colour",
    text: "Channel colour is remembered per channel, per source. “Auto” reads the channel name — DAPI renders blue, GFP green, Cy5 magenta.",
    when: selected,
  },
  {
    id: "min-max",
    text: "Min/Max is not the percentile slider at zero: it latches the contrast to the full range and stays there until the slider moves.",
    when: selected,
  },
  {
    id: "gamma",
    text: "Gamma reshapes the ramp between the contrast limits — lower lifts dim detail. “Linear” puts it back to 1.",
    when: selected,
  },
  {
    id: "empty-plane",
    text: "A badge reading “empty plane (all zeros)” means the data is black, not the viewer.",
    when: selected,
  },
  {
    id: "pinned-axis",
    text: "“Showing axis … — not selectable through the tile API” means the plane on screen is not the one you asked for.",
    when: selected,
  },
  {
    id: "transport-retry",
    text: "“Server did not answer in time” is not a verdict on the image — the Try again button re-runs it on the tiled viewer.",
  },
  {
    id: "indexing",
    text: "Sources appear as they are indexed, so a short list at startup is not the final one.",
    when: (ctx) => ctx.scanning,
  },
  {
    id: "hover-details",
    text: "Hover a source for its full path, or a tensor for its array id, shape and dtype.",
  },
  {
    id: "contrast-estimate",
    text: "Contrast limits come from the coarsest pyramid level, subsampled — an estimate of the plane's histogram, not the whole of it.",
    when: selected,
  },
];

/** The tips that apply right now, in declaration order. */
export function eligibleTips(ctx: TipContext, tips: Tip[] = TIPS): Tip[] {
  return tips.filter((tip) => !tip.when || tip.when(ctx));
}

/**
 * The tip after `currentId` in `tips`, wrapping at the end.
 *
 * Takes an id rather than an index because the eligible list changes underfoot —
 * selecting a source adds seven tips — and an index into the old list points at
 * an unrelated tip in the new one. An id that has dropped out restarts at the
 * front, which is the honest answer: its context is gone.
 */
export function nextTip(currentId: string | null, tips: Tip[]): Tip | null {
  if (tips.length === 0) return null;
  const at = tips.findIndex((tip) => tip.id === currentId);
  return tips[(at + 1) % tips.length] ?? null;
}
