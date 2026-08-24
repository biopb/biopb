import { describe, expect, it } from "vitest";
import type { TileInfo } from "@biopb/tensor-flight-client";
import {
  CONTRAST_SAMPLE_LIMIT,
  GAMMA_OCTAVES,
  TILE_CACHE_BUDGET_BYTES,
  clampGamma,
  contrastLimitsFrom,
  contrastSamples,
  dtypeContrastLimits,
  gammaFromOctaves,
  octavesFromGamma,
  percentileBounds,
  samplesPerPixel,
  tileCacheSize,
  vivColor,
  vivSelection,
} from "./vivUtils";

const TCZYX: TileInfo = {
  array_id: "src/Image:0",
  dim_labels: ["T", "C", "Z", "Y", "X"],
  shape: [4, 3, 16, 1024, 1024],
  chunk_shape: [1, 1, 1, 512, 512],
  dtype: "<u2",
  tile_size: 512,
  plane: { y: 3, x: 4, s: null },
  selectable: { t: 0, c: 1, z: 2 },
  sel_axes: [],
  levels: [
    { level: 0, scale: 1, height: 1024, width: 1024, cols: 2, rows: 2 },
    { level: 1, scale: 2, height: 512, width: 512, cols: 1, rows: 1 },
  ],
};

/** 155 single-page TIFFs on an opaque file axis: the case `sel` exists for. */
const SEQUENCE: TileInfo = {
  ...TCZYX,
  array_id: "tiff-sequence_6bc95fdaaeb2",
  dim_labels: ["i", "y", "x"],
  shape: [155, 1024, 1344],
  plane: { y: 1, x: 2, s: null },
  selectable: { t: null, z: null, c: null },
  sel_axes: [{ axis: 0, label: "i", extent: 155 }],
};

const RGB: TileInfo = {
  ...TCZYX,
  dim_labels: ["T", "C", "Z", "Y", "X", "S"],
  shape: [1, 1, 1, 1024, 1024, 3],
  dtype: "|u1",
  plane: { y: 3, x: 4, s: 5 },
};

const PLAIN_2D: TileInfo = {
  ...TCZYX,
  dim_labels: ["Y", "X"],
  shape: [1024, 1024],
  plane: { y: 0, x: 1, s: null },
  selectable: { t: null, c: null, z: null },
};

describe("contrastSamples", () => {
  it("caps the sort at the limit and returns it sorted", () => {
    const data = new Uint16Array(4000);
    for (let i = 0; i < data.length; i++) data[i] = (data.length - i) * 3;
    const sorted = contrastSamples(data, 100);

    expect(sorted.length).toBeLessThanOrEqual(100);
    expect(sorted.length).toBeGreaterThan(50);
    for (let i = 1; i < sorted.length; i++) {
      expect(sorted[i]!).toBeGreaterThanOrEqual(sorted[i - 1]!);
    }
  });

  it("keeps every value when the data already fits", () => {
    expect(Array.from(contrastSamples(new Uint8Array([9, 1, 5]), 100))).toEqual([1, 5, 9]);
  });

  it("has a limit that keeps a full 512-edge tile's worth of sorting bounded", () => {
    expect(contrastSamples(new Uint16Array(512 * 512)).length).toBeLessThanOrEqual(
      CONTRAST_SAMPLE_LIMIT,
    );
  });
});

describe("contrastLimitsFrom", () => {
  const ramp = contrastSamples(
    Float64Array.from({ length: 100 }, (_, i) => i),
    1000,
  );

  it("spans the data at 0-100", () => {
    expect(contrastLimitsFrom(ramp, 0, 100)).toEqual([0, 99]);
  });

  it("trims the tails at a percentile", () => {
    expect(contrastLimitsFrom(ramp, 1, 99)).toEqual([1, 98]);
  });

  it("never returns a zero-width window on a flat plane", () => {
    // Both percentiles land on the same value; the shader divides by hi - lo.
    const flat = contrastSamples(new Uint16Array(64).fill(7), 1000);
    const [lo, hi] = contrastLimitsFrom(flat, 1, 99);
    expect(lo).toBe(7);
    expect(hi).toBeGreaterThan(lo);
  });

  it("survives an empty sample set", () => {
    const [lo, hi] = contrastLimitsFrom(new Float64Array(0), 1, 99);
    expect(hi).toBeGreaterThan(lo);
  });
});

describe("percentileBounds", () => {
  it("is the full range in min/max mode whatever the slider says", () => {
    expect(percentileBounds(true, 2.5)).toEqual([0, 100]);
  });

  it("is symmetric around the slider value", () => {
    expect(percentileBounds(false, 1)).toEqual([1, 99]);
  });

  it("cannot invert the window", () => {
    const [lo, hi] = percentileBounds(false, 90);
    expect(lo).toBeLessThanOrEqual(hi);
  });
});

describe("gamma", () => {
  it("is neutral at the centre of the slider", () => {
    expect(gammaFromOctaves(0)).toBe(1);
    expect(octavesFromGamma(1)).toBe(0);
  });

  it("puts halving and doubling the same distance from neutral", () => {
    // The reason the control is in octaves at all: on a linear track these two
    // equal and opposite corrections would be 0.5 and 1 apart.
    expect(octavesFromGamma(0.5)).toBe(-1);
    expect(octavesFromGamma(2)).toBe(1);
  });

  it("round-trips a slider position", () => {
    for (const octaves of [-2, -1.35, 0, 0.4, 2]) {
      expect(octavesFromGamma(gammaFromOctaves(octaves))).toBeCloseTo(octaves, 10);
    }
  });

  it("stops at the ends of the track", () => {
    expect(gammaFromOctaves(-99)).toBe(2 ** -GAMMA_OCTAVES);
    expect(gammaFromOctaves(99)).toBe(2 ** GAMMA_OCTAVES);
  });

  it("pulls a stored value that is not an exponent back to the track", () => {
    // 0 and below are not "dim": pow() would render a uniform white plane.
    expect(clampGamma(0)).toBe(2 ** -GAMMA_OCTAVES);
    expect(clampGamma(-3)).toBe(2 ** -GAMMA_OCTAVES);
    expect(clampGamma(1000)).toBe(2 ** GAMMA_OCTAVES);
  });

  it("treats a non-number as neutral rather than as an end of the track", () => {
    // NaN and Infinity are not "very dark" or "very bright" -- they are a value
    // that never came from this control, so neutral is the honest answer.
    expect(clampGamma(Number.NaN)).toBe(1);
    expect(clampGamma(Number.POSITIVE_INFINITY)).toBe(1);
  });

  it("leaves a value already on the track alone", () => {
    expect(clampGamma(0.7)).toBeCloseTo(0.7, 10);
  });
});

describe("dtypeContrastLimits", () => {
  it("covers the integer range so the first frame is not blank", () => {
    expect(dtypeContrastLimits("Uint16")).toEqual([0, 65535]);
  });

  it("falls back rather than throwing on an unknown dtype", () => {
    expect(dtypeContrastLimits("Float16")).toEqual([0, 1]);
  });
});

describe("tileCacheSize", () => {
  it("holds the budget in whole 512x512 uint16 tiles", () => {
    // 512 KiB per tile against 128 MiB.
    expect(tileCacheSize(512, "Uint16", 1, 1)).toBe(TILE_CACHE_BUDGET_BYTES / (512 * 512 * 2));
  });

  it("charges an interleaved RGB tile for all three samples", () => {
    expect(tileCacheSize(512, "Uint8", 3, 1)).toBe(
      Math.floor(tileCacheSize(512, "Uint8", 1, 1) / 3),
    );
  });

  it("shrinks as channels are added, since each is fetched separately", () => {
    expect(tileCacheSize(512, "Uint16", 1, 3)).toBeLessThan(tileCacheSize(512, "Uint16", 1, 1));
  });

  it("keeps a floor a viewport can fit in", () => {
    expect(tileCacheSize(8192, "Float64", 4, 10)).toBeGreaterThanOrEqual(16);
  });
});

describe("vivSelection", () => {
  const at = (t: number, z: number, c: number, axes: Record<string, number> = {}) => ({
    t,
    z,
    c,
    axes,
  });

  it("names only the axes the tensor has", () => {
    expect(vivSelection(TCZYX, at(1, 2, 0))).toEqual({ t: 1, z: 2, c: 0 });
    expect(vivSelection(PLAIN_2D, at(0, 0, 0))).toEqual({});
  });

  it("clamps a slice position carried over from a larger tensor", () => {
    // The store keeps t/z/c across a source change; unclamped this is a 422 per tile.
    expect(vivSelection(TCZYX, at(99, 99, 99))).toEqual({ t: 3, z: 15, c: 2 });
  });

  it("clamps a negative index", () => {
    expect(vivSelection(TCZYX, at(-1, 0, 0)).t).toBe(0);
  });

  it("selects an axis nothing names, under its wire-index key", () => {
    // Left out entirely before `sel` existed, which is what made a 155-file
    // TIFF sequence a one-frame image here.
    expect(vivSelection(SEQUENCE, at(0, 0, 0, { a0: 154 }))).toEqual({ a0: 154 });
  });

  it("defaults an unvisited unnamed axis to 0", () => {
    expect(vivSelection(SEQUENCE, at(0, 0, 0))).toEqual({ a0: 0 });
  });

  it("clamps an unnamed axis too", () => {
    expect(vivSelection(SEQUENCE, at(0, 0, 0, { a0: 999 }))).toEqual({ a0: 154 });
  });

  it("does not let t/z/c stand in for an unnamed axis", () => {
    // buildAxisMap would call this axis `z`. Scrubbing Z must not move it.
    expect(vivSelection(SEQUENCE, at(0, 7, 0))).toEqual({ a0: 0 });
  });
});

describe("samplesPerPixel", () => {
  it("is 1 without a samples axis and the axis extent with one", () => {
    expect(samplesPerPixel(TCZYX)).toBe(1);
    expect(samplesPerPixel(RGB)).toBe(3);
  });
});

describe("vivColor", () => {
  it("scales the store's multipliers to Viv's 0-255 channel colour", () => {
    expect(vivColor("green")).toEqual([0, 255, 0]);
    expect(vivColor("#804000")).toEqual([128, 64, 0]);
  });

  it("resolves auto from the channel name", () => {
    expect(vivColor("auto", "DAPI")).toEqual([0, 0, 255]);
  });
});
