/**
 * Store state -> Viv props.
 *
 * The arithmetic the tiled viewer needs, kept out of the component so it can be
 * tested without a WebGL context: contrast limits from a sampled histogram, the
 * tile-cache bound, and the axis/colour translations.
 */

import { sliderAxes, type TileInfo } from "@biopb/tensor-flight-client";
import { getColorMultipliers, type ColorValue } from "./colorUtils";

// ---------------------------------------------------------------------------
// Contrast limits
// ---------------------------------------------------------------------------

/**
 * How many values the percentile estimate is allowed to sort.
 *
 * The coarsest pyramid level is at most one tile (the server stops halving once
 * the plane fits), so this only bites on a 512-edge tile — 262144 values down to
 * 65536. Percentiles of a smooth intensity histogram are insensitive to that,
 * and it keeps the sort off the critical path of a channel switch.
 */
export const CONTRAST_SAMPLE_LIMIT = 65_536;

/**
 * A sorted subsample of the plane, ready for repeated percentile queries.
 *
 * Sorting once is what makes the contrast slider free: dragging it re-reads this
 * array instead of asking the server to re-render, which is the whole point of
 * moving contrast into the shader.
 */
export function contrastSamples(
  data: ArrayLike<number>,
  limit: number = CONTRAST_SAMPLE_LIMIT,
): Float64Array {
  const stride = Math.max(1, Math.ceil(data.length / limit));
  const count = Math.ceil(data.length / stride);
  const out = new Float64Array(count);
  for (let i = 0, j = 0; j < count; i += stride, j++) out[j] = data[i] ?? 0;
  out.sort(); // TypedArray sorts numerically, unlike Array
  return out;
}

/** The [lo, hi] percentile pair the intensity control is asking for. */
export function percentileBounds(
  useMinMax: boolean,
  percentileScale: number,
): [number, number] {
  if (useMinMax) return [0, 100];
  const lo = Math.min(Math.max(percentileScale, 0), 50);
  return [lo, 100 - lo];
}

/**
 * Contrast limits from a {@link contrastSamples} array.
 *
 * The `hi <= lo` guard is not defensive noise: a flat region (a blank Z-plane, a
 * dark channel) makes both percentiles the same value, and the shader divides by
 * `hi - lo`. Nudging hi keeps that plane black instead of NaN.
 */
export function contrastLimitsFrom(
  sorted: Float64Array,
  loPct: number,
  hiPct: number,
): [number, number] {
  if (sorted.length === 0) return [0, 1];
  const at = (pct: number) => {
    const i = Math.round((pct / 100) * (sorted.length - 1));
    return sorted[Math.min(sorted.length - 1, Math.max(0, i))] ?? 0;
  };
  const lo = at(loPct);
  const hi = at(hiPct);
  return [lo, hi > lo ? hi : lo + 1];
}

// ---------------------------------------------------------------------------
// Gamma
// ---------------------------------------------------------------------------

/**
 * Half-width of the gamma slider, in octaves.
 *
 * The control is positioned in log2(gamma) rather than in gamma itself, because
 * gamma is a ratio: 0.5 and 2 are equal and opposite corrections, and only a log
 * scale puts them the same distance from neutral. On a linear 0.25-4 track,
 * everything that brightens would be squeezed into the first fifth of the
 * travel.
 */
export const GAMMA_OCTAVES = 2;

/** The exponents those ends of the track correspond to. */
export const GAMMA_MIN = 2 ** -GAMMA_OCTAVES;
export const GAMMA_MAX = 2 ** GAMMA_OCTAVES;

/** Slider position (octaves from neutral) -> gamma exponent. */
export function gammaFromOctaves(octaves: number): number {
  const clamped = Math.min(Math.max(octaves, -GAMMA_OCTAVES), GAMMA_OCTAVES);
  return 2 ** clamped;
}

/** Gamma exponent -> slider position. Inverse of {@link gammaFromOctaves}. */
export function octavesFromGamma(gamma: number): number {
  if (!Number.isFinite(gamma)) return 0;
  // Not log2 of a non-positive number: 0 is the direction of "brighter", so the
  // handle belongs at that end of the track rather than back at neutral.
  if (gamma <= 0) return -GAMMA_OCTAVES;
  return Math.min(Math.max(Math.log2(gamma), -GAMMA_OCTAVES), GAMMA_OCTAVES);
}

/**
 * A gamma safe to hand a shader or `np.power`.
 *
 * The server clamps to the same range (`renderer.clamp_gamma`), so the two
 * viewers cannot disagree about what a stored value means. Zero and negatives
 * are not dim -- as an exponent they are a uniform white plane -- and a value
 * that is not a number at all did not come from this control, so it reads as
 * neutral rather than as one end of the track.
 */
export function clampGamma(gamma: number): number {
  if (!Number.isFinite(gamma)) return 1;
  return Math.min(Math.max(gamma, GAMMA_MIN), GAMMA_MAX);
}

const RANGE_BY_DTYPE: Record<string, [number, number]> = {
  Uint8: [0, 255],
  Int8: [-128, 127],
  Uint16: [0, 65535],
  Int16: [-32768, 32767],
  Uint32: [0, 4294967295],
  Int32: [-2147483648, 2147483647],
  Float32: [0, 1],
  Float64: [0, 1],
};

/** Full-range limits, used until the first sampled plane comes back. */
export function dtypeContrastLimits(vivDtype: string): [number, number] {
  return RANGE_BY_DTYPE[vivDtype] ?? [0, 1];
}

// ---------------------------------------------------------------------------
// Tile cache
// ---------------------------------------------------------------------------

const BYTES_BY_DTYPE: Record<string, number> = {
  Uint8: 1, Int8: 1,
  Uint16: 2, Int16: 2,
  Uint32: 4, Int32: 4,
  Float32: 4, Float64: 8,
};

export function bytesPerElement(vivDtype: string): number {
  return BYTES_BY_DTYPE[vivDtype] ?? 1;
}

/**
 * Bytes the decoded-tile cache may hold.
 *
 * Each cached tile is counted twice in practice — the TypedArray plus the GPU
 * texture deck.gl uploaded from it — so this is roughly half the real footprint.
 * 128 MiB therefore budgets ~256 MiB across RAM and VRAM, which leaves room on
 * the integrated graphics that a browser-based viewer has to assume.
 */
export const TILE_CACHE_BUDGET_BYTES = 128 * 1024 * 1024;

/** Never evict below this, so the current viewport always fits. */
const MIN_CACHED_TILES = 16;

/**
 * How many tiles deck.gl may cache, from a byte budget.
 *
 * deck.gl offers `maxCacheByteSize`, which would express this directly, but Viv
 * never sets `byteLength` on the object its `getTileData` returns: deck.gl reads
 * `tile.content.byteLength`, gets `undefined`, logs an error per tile and counts
 * zero — and passing `maxCacheByteSize` *also* switches `maxCacheSize` to
 * Infinity, so the cache would grow without bound. The count is the working
 * control here, and it loses nothing: tile edge, dtype and sample count are all
 * known, so a count IS a byte budget for this data.
 *
 * (Re-check on a Viv or deck.gl bump — if Viv starts reporting `byteLength`,
 * `maxCacheByteSize` becomes the better prop.)
 */
export function tileCacheSize(
  tileSize: number,
  vivDtype: string,
  samplesPerPixel: number,
  channels: number,
  budgetBytes: number = TILE_CACHE_BUDGET_BYTES,
): number {
  const perTile =
    tileSize *
    tileSize *
    bytesPerElement(vivDtype) *
    Math.max(1, samplesPerPixel) *
    Math.max(1, channels);
  if (!(perTile > 0)) return MIN_CACHED_TILES;
  return Math.max(MIN_CACHED_TILES, Math.floor(budgetBytes / perTile));
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

export interface SliceIndices {
  t: number;
  z: number;
  c: number;
  /** Index per axis with no semantic name, keyed by `SliderAxis.key`. */
  axes: Record<string, number>;
}

/**
 * The store's slice position -> Viv's label-keyed selection.
 *
 * Every non-plane axis appears, under the key `sliderAxes` gave it: `t`/`z`/`c`
 * where the labels name the axis, `a<index>` where they do not. The second kind
 * used to be left out entirely — there was no way to ask for it — which is what
 * made a 155-file TIFF sequence a one-frame image in this viewer.
 *
 * Indices are clamped: the store carries a slice position across a source
 * change, so a tensor with fewer Z planes than the last one would otherwise ask
 * for a plane past the end and get a 422 on every tile.
 */
export function vivSelection(
  info: TileInfo,
  slice: SliceIndices,
): Record<string, number> {
  const selection: Record<string, number> = {};
  for (const axis of sliderAxes(info.dim_labels, info.shape)) {
    const want = axis.named ? slice[axis.named] : slice.axes[axis.key] ?? 0;
    selection[axis.key] = Math.min(Math.max(0, want), Math.max(0, axis.extent - 1));
  }
  return selection;
}

/** Values per pixel: >1 only for an interleaved RGB(A) samples axis. */
export function samplesPerPixel(info: TileInfo): number {
  return info.plane.s === null ? 1 : Math.max(1, info.shape[info.plane.s] ?? 1);
}

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

/** Store colour -> Viv's 0-255 RGB triple. */
export function vivColor(
  color: ColorValue,
  channelName?: string,
): [number, number, number] {
  const [r, g, b] = getColorMultipliers(color, channelName);
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}
