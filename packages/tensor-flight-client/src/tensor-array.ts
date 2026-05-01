/**
 * Lazy tensor accessor returned by getTensor().
 *
 * Wraps a TensorDescriptor and accumulates slice/scale parameters, then
 * issues a single POST /api/slice request on .compute().
 *
 * Axis semantics are inferred from dim_labels (t/time, z/depth/plane,
 * c/channel, y/height/row, x/width/col) with a positional heuristic fallback
 * for unknown labels.
 */

import type { TensorDescriptor, TypedNdArray, SliceRequest } from "./types.js";
import type { TensorHttpClient } from "./client.js";

// ---------------------------------------------------------------------------
// Axis helpers
// ---------------------------------------------------------------------------

const SPATIAL_Y = new Set(["y", "height", "row", "rows"]);
const SPATIAL_X = new Set(["x", "width", "col", "cols", "column", "columns"]);
const SPATIAL_Z = new Set(["z", "depth", "plane", "planes", "slice"]);
const TEMPORAL = new Set(["t", "time", "frame", "frames"]);
const CHANNEL = new Set(["c", "channel", "channels", "band", "bands"]);

export interface AxisMap {
  t: number | null;
  z: number | null;
  c: number | null;
  y: number | null;
  x: number | null;
}

/** Derive axis→dimension-index mapping from dim_labels. */
export function buildAxisMap(dimLabels: string[]): AxisMap {
  const map: AxisMap = { t: null, z: null, c: null, y: null, x: null };
  const labels = dimLabels.map((l) => l.toLowerCase().trim());

  for (let i = 0; i < labels.length; i++) {
    const l = labels[i] as string;
    if (TEMPORAL.has(l) && map.t === null) map.t = i;
    else if (SPATIAL_Z.has(l) && map.z === null) map.z = i;
    else if (CHANNEL.has(l) && map.c === null) map.c = i;
    else if (SPATIAL_Y.has(l) && map.y === null) map.y = i;
    else if (SPATIAL_X.has(l) && map.x === null) map.x = i;
  }

  // Positional heuristic fallback for unlabelled / unknown axes:
  // Last two unassigned dimensions → Y, X; one before that → Z
  const ndim = labels.length;
  if (ndim >= 1 && map.x === null) map.x = ndim - 1;
  if (ndim >= 2 && map.y === null) map.y = ndim - 2;
  if (ndim >= 3 && map.z === null) map.z = ndim - 3;
  if (ndim >= 4 && map.c === null) map.c = ndim - 4;
  if (ndim >= 5 && map.t === null) map.t = ndim - 5;

  return map;
}

/** Return true when any axis was inferred by heuristic (not explicit label). */
export function isAxisMapAmbiguous(dimLabels: string[]): boolean {
  const labels = dimLabels.map((l) => l.toLowerCase().trim());
  const allKnown = [...TEMPORAL, ...SPATIAL_Z, ...CHANNEL, ...SPATIAL_Y, ...SPATIAL_X];
  return labels.some((l) => !allKnown.includes(l));
}

// ---------------------------------------------------------------------------
// Scale selector
// ---------------------------------------------------------------------------

/** Per-dimension integer downsampling factors aligned to powers of two. */
export interface ScaleVector {
  factors: number[];
  /** True if any factor was snapped to a different power-of-two. */
  snapped: boolean;
}

const HYSTERESIS = 0.2; // 20% band before switching scale level

/**
 * Compute power-of-two scale hint for a 2D viewport.
 *
 * @param tensorShape   Full shape of the tensor.
 * @param axisMap       Axis index mapping from buildAxisMap().
 * @param viewportW     Viewport width in physical pixels (already DPR-scaled).
 * @param viewportH     Viewport height in physical pixels (already DPR-scaled).
 * @param pixelBudget   Maximum output megapixels (default 1.0).
 * @param prevFactors   Previously used scale factors (for hysteresis).
 */
export function computeScaleHint(
  tensorShape: number[],
  axisMap: AxisMap,
  viewportW: number,
  viewportH: number,
  pixelBudget = 1_000_000,
  prevFactors?: number[],
): ScaleVector {
  const ndim = tensorShape.length;
  const factors = new Array<number>(ndim).fill(1);

  const yIdx = axisMap.y;
  const xIdx = axisMap.x;

  if (yIdx === null || xIdx === null) {
    return { factors, snapped: false };
  }

  const dataH = tensorShape[yIdx] as number;
  const dataW = tensorShape[xIdx] as number;

  if (dataH <= 0 || dataW <= 0) {
    return { factors, snapped: false };
  }

  // How many data pixels fit into the viewport?
  const rawScaleY = dataH / viewportH;
  const rawScaleX = dataW / viewportW;
  // Use the larger factor (more conservative / lower resolution)
  const rawScale = Math.max(rawScaleY, rawScaleX);

  // Budget limit: don't request more pixels than the budget
  const budgetPixels = pixelBudget;
  const budgetFactor = Math.sqrt((dataH * dataW) / budgetPixels);
  const targetScale = Math.max(rawScale, budgetFactor, 1);

  // Snap to nearest power of two
  const log2 = Math.log2(targetScale);
  const snappedLog2 = Math.round(log2);
  let snappedFactor = Math.max(1, Math.pow(2, snappedLog2));

  // Hysteresis: don't switch scale unless deviation exceeds 20%
  if (prevFactors && prevFactors[yIdx] !== undefined) {
    const prev = prevFactors[yIdx];
    const ratio = snappedFactor / prev;
    if (ratio > 1 - HYSTERESIS && ratio < 1 + HYSTERESIS) {
      snappedFactor = prev; // stay on current level
    }
  }

  const snapped = Math.round(snappedFactor) !== Math.round(targetScale);
  factors[yIdx] = snappedFactor;
  factors[xIdx] = snappedFactor;
  // All other axes (T, Z, C) stay at 1

  return { factors: factors.map(Math.round), snapped };
}

// ---------------------------------------------------------------------------
// TensorArray
// ---------------------------------------------------------------------------

export interface SliceOptions {
  /** Fixed index (or start of range) per axis label: t, z, c, y, x. */
  t?: number | [number, number];
  z?: number | [number, number];
  c?: number | [number, number];
  y?: number | [number, number];
  x?: number | [number, number];
  scaleHint?: number[];
  reductionMethod?: string;
  pixelBudget?: number;
}

/** Expand a scalar or [start, stop] into a [start, stop] pair. */
function toRange(
  val: number | [number, number] | undefined,
  fullSize: number,
): [number, number] {
  if (val === undefined) return [0, fullSize];
  if (typeof val === "number") return [val, val + 1];
  return val;
}

/**
 * Lazy accessor for a single tensor within a data source.
 *
 * Call .compute(options) to fetch data from the server.
 */
export class TensorArray {
  readonly descriptor: TensorDescriptor;
  readonly sourceId: string;
  readonly axisMap: AxisMap;
  readonly axisMapAmbiguous: boolean;

  private readonly _client: TensorHttpClient;

  constructor(
    client: TensorHttpClient,
    sourceId: string,
    descriptor: TensorDescriptor,
  ) {
    this._client = client;
    this.sourceId = sourceId;
    this.descriptor = descriptor;
    this.axisMap = buildAxisMap(descriptor.dim_labels);
    this.axisMapAmbiguous = isAxisMapAmbiguous(descriptor.dim_labels);
  }

  get ndim(): number {
    return this.descriptor.shape.length;
  }

  get shape(): number[] {
    return this.descriptor.shape;
  }

  get dtype(): string {
    return this.descriptor.dtype;
  }

  /**
   * Fetch a sub-region of the tensor.
   *
   * @param options  Per-axis slice ranges + scale/reduction settings.
   * @returns        TypedNdArray with raw bytes, shape, dtype, and dim labels.
   */
  async compute(options: SliceOptions = {}): Promise<TypedNdArray> {
    const ndim = this.ndim;
    const shape = this.descriptor.shape;

    const sliceStart: number[] = new Array(ndim).fill(0);
    const sliceStop: number[] = [...shape];

    const setAxis = (idx: number | null, val: number | [number, number] | undefined) => {
      if (idx === null || val === undefined) return;
      const fullSize = shape[idx] as number;
      const [s, e] = toRange(val, fullSize);
      sliceStart[idx] = Math.max(0, s);
      sliceStop[idx] = Math.min(fullSize, e);
    };

    setAxis(this.axisMap.t, options.t);
    setAxis(this.axisMap.z, options.z);
    setAxis(this.axisMap.c, options.c);
    setAxis(this.axisMap.y, options.y);
    setAxis(this.axisMap.x, options.x);

    const req: SliceRequest = {
      source_id: this.sourceId,
      tensor_id: this.descriptor.array_id,
      slice_start: sliceStart,
      slice_stop: sliceStop,
      scale_hint: options.scaleHint,
      reduction_method: options.reductionMethod,
      pixel_budget: options.pixelBudget,
    };

    return this._client.slice(req);
  }
}
