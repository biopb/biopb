import { describe, it, expect } from "vitest";
import { pinnableAxes, planePinFor, roiVisibleOnPlane } from "./roi-plane.js";
import type { TileInfo } from "./types.js";

function tileInfo(over: Partial<TileInfo>): TileInfo {
  return {
    array_id: "src/Image:0",
    dim_labels: ["t", "c", "z", "y", "x"],
    shape: [10, 3, 20, 512, 512],
    chunk_shape: [1, 1, 1, 256, 256],
    dtype: "uint16",
    tile_size: 256,
    plane: { y: 3, x: 4, s: null },
    selectable: { t: 0, z: 2, c: 1 },
    sel_axes: [],
    levels: [],
    ...over,
  };
}

describe("pinnableAxes", () => {
  it("names every non-plane axis with extent", () => {
    expect(pinnableAxes(tileInfo({}))).toEqual([
      { axis: 0, label: "t" },
      { axis: 1, label: "c" },
      { axis: 2, label: "z" },
    ]);
  });

  it("leaves out the display plane's own axes", () => {
    const axes = pinnableAxes(tileInfo({})).map((a) => a.label);
    expect(axes).not.toContain("y");
    expect(axes).not.toContain("x");
  });

  it("leaves out the samples axis of an interleaved RGB tensor", () => {
    const info = tileInfo({
      dim_labels: ["z", "y", "x", "s"],
      shape: [8, 512, 512, 3],
      plane: { y: 1, x: 2, s: 3 },
    });
    expect(pinnableAxes(info)).toEqual([{ axis: 0, label: "z" }]);
  });

  it("leaves out a singleton axis, where a pin would say nothing", () => {
    const info = tileInfo({ shape: [1, 3, 20, 512, 512] });
    expect(pinnableAxes(info).map((a) => a.label)).toEqual(["c", "z"]);
  });

  it("leaves out an unlabelled axis rather than inventing a key", () => {
    // A TIFF sequence's opaque file axis: nothing to key a pin under, and a
    // synthetic key would be a dialect the Python SDK does not share.
    const info = tileInfo({
      dim_labels: ["", "z", "y", "x"],
      shape: [4, 8, 512, 512],
      plane: { y: 2, x: 3, s: null },
    });
    expect(pinnableAxes(info)).toEqual([{ axis: 1, label: "z" }]);
  });

  it("leaves out both axes when two share a label", () => {
    const info = tileInfo({
      dim_labels: ["z", "z", "y", "x"],
      shape: [4, 8, 512, 512],
      plane: { y: 2, x: 3, s: null },
    });
    expect(pinnableAxes(info)).toEqual([]);
  });
});

describe("planePinFor", () => {
  it("keys the current indices by label", () => {
    const axes = pinnableAxes(tileInfo({}));
    expect(planePinFor(axes, { 0: 4, 1: 1, 2: 12 })).toEqual({ t: 4, c: 1, z: 12 });
  });

  it("skips an axis the caller has no index for", () => {
    const axes = pinnableAxes(tileInfo({}));
    expect(planePinFor(axes, { 2: 12 })).toEqual({ z: 12 });
  });
});

describe("roiVisibleOnPlane", () => {
  it("shows an unpinned annotation everywhere", () => {
    expect(roiVisibleOnPlane({}, { z: 4, t: 2 })).toBe(true);
  });

  it("hides one pinned to another index", () => {
    expect(roiVisibleOnPlane({ z: 12 }, { z: 4 })).toBe(false);
  });

  it("shows one pinned to this index", () => {
    expect(roiVisibleOnPlane({ z: 4 }, { z: 4, t: 9 })).toBe(true);
  });

  it("follows a z-stack when only t is pinned", () => {
    // The absent dimension is the whole point: one ROI, every z.
    const pin = { t: 2 };
    expect(roiVisibleOnPlane(pin, { t: 2, z: 0 })).toBe(true);
    expect(roiVisibleOnPlane(pin, { t: 2, z: 19 })).toBe(true);
    expect(roiVisibleOnPlane(pin, { t: 3, z: 19 })).toBe(false);
  });

  it("needs every pinned dimension to agree", () => {
    expect(roiVisibleOnPlane({ t: 2, z: 4 }, { t: 2, z: 5 })).toBe(false);
    expect(roiVisibleOnPlane({ t: 2, z: 4 }, { t: 2, z: 4 })).toBe(true);
  });

  it("shows rather than hides when a pin cannot be evaluated", () => {
    // An extra ROI on screen is recoverable; a silently hidden one is not.
    expect(roiVisibleOnPlane({ q: 3 }, { z: 4 })).toBe(true);
  });
});
