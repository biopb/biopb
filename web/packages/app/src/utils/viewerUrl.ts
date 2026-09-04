/**
 * The viewer's shareable state, as query parameters.
 *
 * A link has to survive being pasted into an issue and hand-edited a week
 * later, so the parameters are literal (`?id=…&t=5&z=12`) rather than one
 * packed token. That also makes them additive: a link written before `g`
 * existed simply falls back to the store default, where a positional encoding
 * would need a version and a migration for every field.
 *
 * Camera (pan/zoom/orbit) is deliberately absent. Neither viewer reports its
 * view state outward -- TileViewer hands Viv a `viewStates` computed once, and
 * VolumeViewer uses deck.gl's uncontrolled `initialViewState` -- so putting the
 * camera here means making both controlled first. Separate job.
 *
 * Names are chosen so a `.N` pane suffix stays available for the eventual
 * multi-image grid: pane 0 would keep writing the unsuffixed names this module
 * already emits, which is what lets today's links keep working then.
 */

import type { SliceState } from "../store";
import { clampGamma } from "./vivUtils";
import { DEFAULT_VOLUME_RENDER_MODE, VOLUME_RENDER_MODES, type VolumeRenderMode } from "./volumeUtils";

/** The tensor's whole address; `source_id`, or `source_id/field`. */
export const PARAM_ID = "id";

/** Percentile-window width, matching the slider's own 0-4 range. */
const PERCENTILE_MIN = 0;
const PERCENTILE_MAX = 4;

/** Every parameter this module owns, so unrelated ones survive a write. */
const OWNED = new Set([PARAM_ID, "t", "z", "c", "p", "mm", "g", "v", "vm"]);

/** An axis index parameter: `a0`, `a3`, ... keyed exactly as `SliderAxis.key`. */
const AXIS_PARAM = /^a\d+$/;

export interface ViewerUrlState {
  arrayId: string;
  slice: SliceState;
  render3d: boolean;
  volumeRenderMode: VolumeRenderMode;
}

function isVolumeRenderMode(v: string): v is VolumeRenderMode {
  return VOLUME_RENDER_MODES.some((m) => m.key === v);
}

/** A finite number from `raw`, or null -- `Number("")` is 0, which is a value. */
function num(raw: string | null): number | null {
  if (raw === null || raw.trim() === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Read a viewing state out of `params`.
 *
 * Unbounded on purpose. A link may carry a *pinned* address, whose descriptor
 * does not exist until `tile_info` answers, so there is nothing to clamp
 * against here. `clampSliceTo` does it when the grid lands, which also covers a
 * case this could never see: a tensor that changed shape under a selection
 * already made. Values are still *validated* -- a non-number is ignored rather
 * than passed on as NaN.
 *
 * Axis keys are taken as given for the same reason: which ones exist is a
 * property of the grid, so the clamp is what drops the ones naming nothing.
 */
export function decodeViewerState(
  params: URLSearchParams,
  defaults: ViewerUrlState,
): ViewerUrlState {
  const named: Pick<SliceState, "t" | "z" | "c"> = {
    t: defaults.slice.t,
    z: defaults.slice.z,
    c: defaults.slice.c,
  };
  for (const key of ["t", "z", "c"] as const) {
    const v = num(params.get(key));
    if (v !== null) named[key] = Math.max(0, Math.round(v));
  }

  const unnamed: Record<string, number> = {};
  for (const [key, raw] of params.entries()) {
    if (!AXIS_PARAM.test(key)) continue;
    const v = num(raw);
    if (v !== null) unnamed[key] = Math.max(0, Math.round(v));
  }

  const p = num(params.get("p"));
  const g = num(params.get("g"));
  const vm = params.get("vm");

  return {
    arrayId: params.get(PARAM_ID) ?? defaults.arrayId,
    slice: {
      ...named,
      axes: unnamed,
      percentileScale:
        p === null ? defaults.slice.percentileScale : clamp(p, PERCENTILE_MIN, PERCENTILE_MAX),
      // An explicit percentile is a contradiction with min/max, and the control
      // writes them together; the flag wins only when it is the one that is set.
      useMinMax: params.get("mm") === "1" ? true : p !== null ? false : defaults.slice.useMinMax,
      gamma: g === null ? defaults.slice.gamma : clampGamma(g),
    },
    render3d: params.get("v") === "1" ? true : params.get("v") === "0" ? false : defaults.render3d,
    volumeRenderMode: vm !== null && isVolumeRenderMode(vm) ? vm : defaults.volumeRenderMode,
  };
}

/**
 * Write `state` into a copy of `params`, dropping anything left at its default.
 *
 * Omitting defaults keeps a link to what was deliberately set, so a shared URL
 * reads as an intent rather than a dump of the whole store. Parameters this
 * module does not own (`token`, `next`) are carried through untouched.
 */
export function encodeViewerState(params: URLSearchParams, state: ViewerUrlState, defaults: ViewerUrlState): URLSearchParams {
  const out = new URLSearchParams();
  for (const [key, value] of params.entries()) {
    if (!OWNED.has(key) && !AXIS_PARAM.test(key)) out.append(key, value);
  }

  out.set(PARAM_ID, state.arrayId);
  for (const key of ["t", "z", "c"] as const) {
    if (state.slice[key] !== defaults.slice[key]) out.set(key, String(state.slice[key]));
  }
  for (const [key, value] of Object.entries(state.slice.axes)) {
    if (value !== 0) out.set(key, String(value));
  }
  if (state.slice.useMinMax) {
    out.set("mm", "1");
  } else if (state.slice.percentileScale !== defaults.slice.percentileScale) {
    out.set("p", String(round(state.slice.percentileScale, 1)));
  }
  if (state.slice.gamma !== defaults.slice.gamma) out.set("g", String(round(state.slice.gamma, 3)));
  if (state.render3d !== defaults.render3d) out.set("v", state.render3d ? "1" : "0");
  if (state.volumeRenderMode !== defaults.volumeRenderMode) out.set("vm", state.volumeRenderMode);
  return out;
}

/** Trim a slider's float so a scrub does not write 17 digits into the bar. */
function round(value: number, decimals: number): number {
  const f = 10 ** decimals;
  return Math.round(value * f) / f;
}

export const DEFAULT_VIEWER_URL_STATE: Omit<ViewerUrlState, "arrayId"> = {
  slice: { t: 0, z: 0, c: 0, axes: {}, percentileScale: 1, useMinMax: false, gamma: 1 },
  render3d: false,
  volumeRenderMode: DEFAULT_VOLUME_RENDER_MODE,
};
