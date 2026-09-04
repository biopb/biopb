/**
 * The viewer's shareable state, as query parameters.
 *
 * A link has to survive being pasted into an issue and hand-edited a week
 * later, so the parameters are literal (`?id=…&t=5&z=12`) rather than one
 * packed token. That also makes them additive: a link written before `g`
 * existed simply falls back to the store default, where a positional encoding
 * would need a version and a migration for every field.
 *
 * Both cameras are carried, under one pair of names. Each viewer reports its
 * own view state through an `onViewStateChange` that returns nothing, so the
 * library keeps driving the camera and the store only mirrors it; a link then
 * carries whichever viewer was mounted.
 *
 * Names are chosen so a `.N` pane suffix stays available for the eventual
 * multi-image grid: pane 0 would keep writing the unsuffixed names this module
 * already emits, which is what lets today's links keep working then.
 */

import type { Camera2DState, Camera3DState, SliceState } from "../store";
import { clampGamma } from "./vivUtils";
import { DEFAULT_VOLUME_RENDER_MODE, VOLUME_RENDER_MODES, type VolumeRenderMode } from "./volumeUtils";

/** The tensor's whole address; `source_id`, or `source_id/field`. */
export const PARAM_ID = "id";

/** Percentile-window width, matching the slider's own 0-4 range. */
const PERCENTILE_MIN = 0;
const PERCENTILE_MAX = 4;

/** Every parameter this module owns, so unrelated ones survive a write. */
const OWNED = new Set([PARAM_ID, "t", "z", "c", "p", "mm", "g", "v", "vm", "tg", "zm", "rx", "ro"]);

/** `OrbitController` clamps pitch to this; a link may not ask for more. */
const ROTATION_X_LIMIT = 90;

/**
 * Zoom is log2(pixels per world unit), so this is already far past any real
 * volume -- it exists to keep a mistyped exponent from reaching the projection
 * matrix, not to express a policy about how far one may zoom.
 */
const ZOOM_LIMIT = 50;

/** An axis index parameter: `a0`, `a3`, ... keyed exactly as `SliderAxis.key`. */
const AXIS_PARAM = /^a\d+$/;

export interface ViewerUrlState {
  arrayId: string;
  slice: SliceState;
  render3d: boolean;
  volumeRenderMode: VolumeRenderMode;
  /** Null when the view was never orbited, i.e. "open at the fitted camera". */
  camera3d: Camera3DState | null;
  /** Null when the view was never panned, i.e. "open at the fitted camera". */
  camera2d: Camera2DState | null;
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
 * Wrap an orbit angle into [-180, 180), the range deck.gl reports it in.
 *
 * Wrapped rather than clamped: the axis is periodic, so 370 degrees is a real
 * bearing (10) where clamping would silently mean 180 -- a different view.
 */
function wrapDegrees(value: number): number {
  return ((((value + 180) % 360) + 360) % 360) - 180;
}

/** The comma target a link carries, as finite numbers, or null. */
function decodeTarget(params: URLSearchParams): number[] | null {
  const raw = params.get("tg");
  if (raw === null) return null;
  const parts = raw.split(",").map(Number);
  return parts.every(Number.isFinite) ? parts : null;
}

/**
 * The 2-D camera, or null when the link does not carry one.
 *
 * Both cameras share `tg`/`zm`, told apart by how many components the target
 * has: two is a plane, three is a volume. That is a property of the data rather
 * than a convention, so neither decoder needs to know which viewer is mounted,
 * and a link naming one camera can never be read as the other.
 */
function decodeCamera2d(params: URLSearchParams): Camera2DState | null {
  const target = decodeTarget(params);
  const zoom = num(params.get("zm"));
  if (target === null || target.length !== 2 || zoom === null) return null;
  const [x, y] = target as [number, number];
  return { target: [x, y], zoom: clamp(zoom, -ZOOM_LIMIT, ZOOM_LIMIT) };
}

/**
 * The 3-D camera, or null when the link does not carry one.
 *
 * All-or-nothing on `tg`+`zm`: a target without a zoom is not a camera, and
 * guessing the missing half would frame the volume somewhere nobody chose.
 * The rotations do default, since 0 is the fitted view's own orientation.
 *
 * Unlike the indices above this *is* bounded here: its limits come from
 * `OrbitController`, which is a property of the camera rather than of the
 * tensor, so there is nothing to wait for a grid to learn.
 */
function decodeCamera3d(params: URLSearchParams): Camera3DState | null {
  const target = decodeTarget(params);
  const zoom = num(params.get("zm"));
  if (target === null || target.length !== 3 || zoom === null) return null;
  const [x, y, z] = target as [number, number, number];
  return {
    target: [x, y, z],
    zoom: clamp(zoom, -ZOOM_LIMIT, ZOOM_LIMIT),
    rotationX: clamp(num(params.get("rx")) ?? 0, -ROTATION_X_LIMIT, ROTATION_X_LIMIT),
    rotationOrbit: wrapDegrees(num(params.get("ro")) ?? 0),
  };
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
    camera3d: decodeCamera3d(params) ?? defaults.camera3d,
    camera2d: decodeCamera2d(params) ?? defaults.camera2d,
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
  // The camera of the viewer that is actually mounted. Writing both would put a
  // camera in the link for a viewer the recipient will not open, and the two
  // share `tg`/`zm`.
  const camera = state.render3d ? state.camera3d : state.camera2d;
  if (camera) {
    // One decimal on the target: it is in scaled voxels (3-D) or image pixels
    // (2-D), where a tenth is far below what the screen can resolve.
    out.set("tg", camera.target.map((v) => round(v, 1)).join(","));
    out.set("zm", String(round(camera.zoom, 3)));
    if ("rotationX" in camera) {
      // Unlike the target these have a meaningful zero -- the fitted
      // orientation -- so they follow the same omit-the-default rule as
      // everything above.
      if (camera.rotationX !== 0) out.set("rx", String(round(camera.rotationX, 1)));
      if (camera.rotationOrbit !== 0) out.set("ro", String(round(camera.rotationOrbit, 1)));
    }
  }
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
  camera3d: null,
  camera2d: null,
};
