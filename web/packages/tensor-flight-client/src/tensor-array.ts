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
const SAMPLES = new Set(["s", "samples"]);
const ALL_KNOWN_LABELS = new Set<string>([
  ...TEMPORAL, ...SPATIAL_Z, ...CHANNEL, ...SPATIAL_Y, ...SPATIAL_X, ...SAMPLES,
]);

export interface AxisMap {
  t: number | null;
  z: number | null;
  c: number | null;
  y: number | null;
  x: number | null;
  /** Interleaved RGB(A) samples axis, composited server-side. Always trailing. */
  s: number | null;
}

/**
 * Derive axis→dimension-index mapping from a descriptor's dim_labels.
 *
 * The data plane guarantees the axis order it serves (biopb/biopb#596): Z, Y, X
 * and S appear last, in that relative order, with T, C and any unrecognized
 * label keeping their relative order ahead of them. So the display plane is a
 * *position* — X last, Y before it, behind an interleaved samples axis when one
 * is there — and must be read the same way the server's own `plane_axes` reads
 * it, since this side picks the crop and the scale_hint while that side renders
 * the block those describe.
 *
 * T, C and Z stay label lookups. The canonical order does not position them (T
 * and C ride in the leading group), and the slider UI needs to tell them apart.
 */
export function buildAxisMap(dimLabels: string[]): AxisMap {
  const map: AxisMap = { t: null, z: null, c: null, y: null, x: null, s: null };
  const labels = dimLabels.map((l) => l.toLowerCase().trim());
  const ndim = labels.length;
  if (ndim === 0) return map;

  // Samples is honored only where the canonical order puts it — last, and 3 or
  // 4 wide is the server's gate, but shapes are not known here, so the label
  // alone decides. An S anywhere else is not an order the server serves.
  const hasTrailingSamples = ndim >= 3 && SAMPLES.has(labels[ndim - 1] as string);
  if (hasTrailingSamples) map.s = ndim - 1;

  const xIdx = hasTrailingSamples ? ndim - 2 : ndim - 1;
  const yIdx = xIdx - 1;
  if (xIdx >= 0) map.x = xIdx;
  if (yIdx >= 0) map.y = yIdx;

  // T/C/Z by label, and only among the axes ahead of the plane — a label
  // sitting on Y, X or S is describing the plane, not a slider axis.
  for (let i = 0; i < yIdx; i++) {
    const l = labels[i] as string;
    if (TEMPORAL.has(l) && map.t === null) map.t = i;
    else if (SPATIAL_Z.has(l) && map.z === null) map.z = i;
    else if (CHANNEL.has(l) && map.c === null) map.c = i;
  }

  // Positional fallback for leading axes no label claimed, nearest the plane
  // first (z, then c, then t). An unlabelled store still gets usable sliders;
  // it is only a name for something already navigable, never a claim about the
  // plane — which is why this is confined to the leading group and can no
  // longer swallow the samples axis the way the old whole-array fallback did.
  const unclaimed: number[] = [];
  for (let i = yIdx - 1; i >= 0; i--) {
    if (i !== map.t && i !== map.z && i !== map.c) unclaimed.push(i);
  }
  if (map.z === null) map.z = unclaimed.shift() ?? null;
  if (map.c === null) map.c = unclaimed.shift() ?? null;
  if (map.t === null) map.t = unclaimed.shift() ?? null;

  return map;
}

/**
 * True when the labels do not name every axis, so the plane came from position.
 *
 * Position is the contract, not a guess, so this no longer means "the plane may
 * be wrong" — it means the source did not say what its leading axes are, and
 * the slider UI has nothing to title them with.
 */
export function isAxisMapAmbiguous(dimLabels: string[]): boolean {
  return dimLabels.some((l) => !ALL_KNOWN_LABELS.has(l.toLowerCase().trim()));
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

/**
 * Compute power-of-two scale hint for a 2D viewport.
 *
 * Pure function: computes optimal scale factors without hysteresis.
 * Hysteresis should be implemented at the caller level.
 *
 * @param tensorShape   Full shape of the tensor.
 * @param axisMap       Axis index mapping from buildAxisMap().
 * @param viewportW     Viewport width in physical pixels (already DPR-scaled).
 * @param viewportH     Viewport height in physical pixels (already DPR-scaled).
 * @param pixelBudget   Maximum output megapixels (default 1.0).
 * @param viewportZoom  Current viewport zoom level (1.0 = fit-to-window).
 */
export function computeScaleHint(
  tensorShape: number[],
  axisMap: AxisMap,
  viewportW: number,
  viewportH: number,
  pixelBudget = 1_000_000,
  viewportZoom = 1,
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

  if (viewportW <= 0 || viewportH <= 0) {
    return { factors, snapped: false };
  }

  const maxScale = Math.max(1, Math.sqrt((dataH * dataW) / pixelBudget));

  // effectiveTargetScale: zoomed in (viewportZoom>1) means smaller scale (more detail)
  const effectiveTargetScale = Math.max(1 / viewportZoom, 1);

  const clampedTargetScale = Math.min(maxScale, effectiveTargetScale);

  // Snap to nearest power of two (pure, no hysteresis)
  const log2 = Math.log2(clampedTargetScale);
  const snappedLog2 = Math.round(log2);
  const snappedFactor = Math.max(1, Math.pow(2, snappedLog2));

  const snapped = snappedFactor !== effectiveTargetScale;
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
  protected _descriptor: TensorDescriptor;
  readonly sourceId: string;
  protected _axisMap: AxisMap;
  protected _axisMapAmbiguous: boolean;
  protected readonly _client: TensorHttpClient;

  get descriptor(): TensorDescriptor { return this._descriptor; }
  get axisMap(): AxisMap { return this._axisMap; }
  get axisMapAmbiguous(): boolean { return this._axisMapAmbiguous; }

  constructor(
    client: TensorHttpClient,
    sourceId: string,
    descriptor: TensorDescriptor,
  ) {
    this._client = client;
    this.sourceId = sourceId;
    this._descriptor = descriptor;
    this._axisMap = buildAxisMap(descriptor.dim_labels);
    this._axisMapAmbiguous = isAxisMapAmbiguous(descriptor.dim_labels);
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
