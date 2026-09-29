import type { StateCreator } from "zustand";
import { vivDtype } from "@biopb/tensor-flight-client";
import {
  clampContrastLimits,
  contrastLimitsFrom,
  contrastTrack,
  percentileBounds,
} from "../utils/vivUtils";
import { selectView, withView } from "./tensorView";
import type { AppState } from "./types";

/** The sampled grey levels of one plane. */
export interface PlaneSamples {
  /** The plane they were read for, by identity: the Viv selection, or the volume. */
  plane: object;
  /** Sorted ascending, as `contrastSamples` returns them. */
  values: Float64Array;
}

/** Facts only a mounted viewer can observe. See {@link AppState.runtime}. */
export interface Runtime {
  /**
   * Whether what is on the canvas is the plane that was last asked for. Play
   * reads it to pace itself to the data plane rather than to a timer, and the
   * tiled viewer reads its own copy of the same fact to cover a stale plane.
   */
  planeReady: boolean;
  /**
   * The last plane sampled, deliberately kept across a plane change so the
   * contrast does not flash while the next read is in flight. Null before the
   * first.
   */
  samples: PlaneSamples | null;
}

/** What a viewer that has not reported yet has observed. */
export const INITIAL_RUNTIME: Runtime = { planeReady: false, samples: null };

export interface RuntimeSlice {
  /**
   * What only the mounted viewer can observe, tagged with the `target.epoch` it
   * was observed under. A write carrying another epoch is dropped, so a viewer
   * that outlives its tensor for a commit cannot leak into the next one.
   *
   * Everything else the contrast controls need is derived from this and from
   * `target.info`, `views[key].observedLimits` and `display`, by
   * {@link selectContrastTrack} and {@link selectContrastWindow}: the shader and
   * the panel call the same selectors (biopb/biopb#955), so they cannot disagree
   * and nothing derived is stored.
   */
  runtime: Runtime;
  /** Widen the tensor in view's observed levels on `channel`. */
  noteObservedLimits: (value: [number, number], channel: number) => void;
  /**
   * A viewer sampled a plane: keep its values for the contrast window, and note
   * their extremes as levels the tensor has shown, on `channel`.
   */
  notePlaneSamples: (samples: PlaneSamples, channel: number, epoch: number) => void;
  setPlaneReady: (value: boolean, epoch: number) => void;
}

export const createRuntimeSlice: StateCreator<AppState, [], [], RuntimeSlice> = (set, get) => ({
  runtime: INITIAL_RUNTIME,

  noteObservedLimits(value, channel) {
    const key = get().target.key;
    set((s) => widenObserved(s, key, value, channel));
  },

  notePlaneSamples(samples, channel, epoch) {
    const key = get().target.key;
    set((s) => {
      if (s.target.epoch !== epoch) return s;
      const { values } = samples;
      return {
        runtime: { ...s.runtime, samples },
        ...(values.length > 0
          ? widenObserved(s, key, [values[0] as number, values[values.length - 1] as number], channel)
          : {}),
      };
    });
  },

  setPlaneReady(value, epoch) {
    set((s) =>
      s.target.epoch !== epoch || s.runtime.planeReady === value
        ? s
        : { runtime: { ...s.runtime, planeReady: value } },
    );
  },
});

/** A patch widening `key`'s observed union on `channel` to cover `value`, or none. */
function widenObserved(
  s: AppState,
  key: string | null,
  value: [number, number],
  channel: number,
): Partial<AppState> {
  const prev = key ? s.views[key]?.observedLimits[channel] : undefined;
  if (prev && prev[0] <= value[0] && prev[1] >= value[1]) return {};
  const next: [number, number] = prev
    ? [Math.min(prev[0], value[0]), Math.max(prev[1], value[1])]
    : [value[0], value[1]];
  return withView(s, key, (cur) => ({
    observedLimits: { ...cur.observedLimits, [channel]: next },
  }));
}

/** The levels this tensor's current channel has shown, or null if none yet. */
export function selectObservedLimits(s: AppState): [number, number] | null {
  return selectView(s).observedLimits[s.position.c] ?? null;
}

/**
 * The sampled min and max grey level of the plane last sampled, or null before
 * one has been. What the automatic window would be with neither tail trimmed;
 * in fixed mode the window no longer says anything about the data, so this is
 * what a fixed window is reset onto.
 *
 * Returns a fresh array: read it with `useShallow`.
 */
export function selectPlaneLimits(s: AppState): [number, number] | null {
  const samples = s.runtime.samples;
  return samples ? contrastLimitsFrom(samples.values, 0, 100) : null;
}

/**
 * The track a contrast window is chosen on: the dtype's own range, or on a
 * float tensor the levels the data has shown. Null until the grid is there --
 * the panel then falls back to what the catalog says of the dtype.
 *
 * Derived, not published (biopb/biopb#955): the viewer's shader and the panel's
 * bar both call this, so a plane change cannot leave the bar on a track the
 * shader is not clamping into. Returns a fresh array: read it with `useShallow`.
 */
export function selectContrastTrack(s: AppState): [number, number] | null {
  const info = s.target.info;
  if (!info) return null;
  return contrastTrack(
    vivDtype(info.dtype),
    selectObservedLimits(s),
    selectPlaneLimits(s),
    s.display.fixedLimits,
  );
}

/**
 * The contrast window the shader uses, and the panel seeds a fixed window from:
 * the user's fixed levels brought inside the track, or the percentile window of
 * the plane last sampled -- or the whole track while none has been. Null until
 * the grid is there. Returns a fresh array: read it with `useShallow`.
 */
export function selectContrastWindow(s: AppState): [number, number] | null {
  const info = s.target.info;
  const track = selectContrastTrack(s);
  if (!info || !track) return null;
  const { contrastMode, fixedLimits, percentileScale } = s.display;
  // A fixed window is the user's, not the plane's: it is not re-derived per
  // plane, only brought inside the track it is being applied to.
  if (contrastMode === "fixed") {
    return fixedLimits ? clampContrastLimits(fixedLimits, track, vivDtype(info.dtype)) : track;
  }
  const samples = s.runtime.samples;
  if (!samples) return track;
  const [lo, hi] = percentileBounds(percentileScale);
  return contrastLimitsFrom(samples.values, lo, hi);
}
