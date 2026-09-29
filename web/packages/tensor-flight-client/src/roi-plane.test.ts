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
  it("offers every non-plane axis with extent, by position", () => {
    expect(pinnableAxes(tileInfo({})).map((a) => a.axis)).toEqual([0, 1, 2]);
  });

  it("leaves out the display plane's own axes", () => {
    // Geometry addresses y/x; a pin would be meaningless there.
    expect(pinnableAxes(tileInfo({})).map((a) => a.axis)).not.toContain(3);
    expect(pinnableAxes(tileInfo({})).map((a) => a.axis)).not.toContain(4);
  });

  it("leaves out the samples axis of an interleaved RGB tensor", () => {
    const info = tileInfo({
      dim_labels: ["z", "y", "x", "s"],
      shape: [8, 512, 512, 3],
      plane: { y: 1, x: 2, s: 3 },
    });
    expect(pinnableAxes(info).map((a) => a.axis)).toEqual([0]);
  });

  it("leaves out a singleton axis, where a pin would say nothing", () => {
    const info = tileInfo({ shape: [1, 3, 20, 512, 512] });
    expect(pinnableAxes(info).map((a) => a.axis)).toEqual([1, 2]);
  });

  it("OFFERS an unlabelled axis, which a label-keyed pin could not address", () => {
    // A TIFF sequence's opaque file axis. Under the old label-keyed pin this
    // had no key at all, so every annotation on it broadcast across every file.
    const info = tileInfo({
      dim_labels: ["", "z", "y", "x"],
      shape: [4, 8, 512, 512],
      plane: { y: 2, x: 3, s: null },
    });
    expect(pinnableAxes(info).map((a) => a.axis)).toEqual([0, 1]);
  });

  it("OFFERS both axes when two share a label", () => {
    const info = tileInfo({
      dim_labels: ["z", "z", "y", "x"],
      shape: [4, 8, 512, 512],
      plane: { y: 2, x: 3, s: null },
    });
    expect(pinnableAxes(info).map((a) => a.axis)).toEqual([0, 1]);
  });

  it("still carries a title, so the UI has something to call each axis", () => {
    const info = tileInfo({
      dim_labels: ["POS", "", "y", "x"],
      shape: [4, 8, 512, 512],
      plane: { y: 2, x: 3, s: null },
    });
    expect(pinnableAxes(info).map((a) => a.title)).toEqual(["POS", "axis 1"]);
  });
});

describe("planePinFor", () => {
  it("keys the current indices by axis position", () => {
    const axes = pinnableAxes(tileInfo({}));
    expect(planePinFor(axes, { 0: 4, 1: 1, 2: 12 })).toEqual({ 0: 4, 1: 1, 2: 12 });
  });

  it("skips an axis the caller has no index for", () => {
    const axes = pinnableAxes(tileInfo({}));
    expect(planePinFor(axes, { 2: 12 })).toEqual({ 2: 12 });
  });

  it("never pins a plane axis, even when handed an index for one", () => {
    const axes = pinnableAxes(tileInfo({}));
    expect(planePinFor(axes, { 2: 12, 3: 100, 4: 200 })).toEqual({ 2: 12 });
  });
});

describe("roiVisibleOnPlane", () => {
  it("shows an unpinned annotation everywhere", () => {
    expect(roiVisibleOnPlane({}, { 2: 4, 0: 2 })).toBe(true);
  });

  it("hides one pinned to another index", () => {
    expect(roiVisibleOnPlane({ 2: 12 }, { 2: 4 })).toBe(false);
  });

  it("shows one pinned to this index", () => {
    expect(roiVisibleOnPlane({ 2: 4 }, { 2: 4, 0: 9 })).toBe(true);
  });

  it("follows a z-stack when only t is pinned", () => {
    // The absent axis is the whole point: one ROI, every z.
    const pin = { 0: 2 };
    expect(roiVisibleOnPlane(pin, { 0: 2, 2: 0 })).toBe(true);
    expect(roiVisibleOnPlane(pin, { 0: 2, 2: 19 })).toBe(true);
    expect(roiVisibleOnPlane(pin, { 0: 3, 2: 19 })).toBe(false);
  });

  it("needs every pinned axis to agree", () => {
    expect(roiVisibleOnPlane({ 0: 2, 2: 4 }, { 0: 2, 2: 5 })).toBe(false);
    expect(roiVisibleOnPlane({ 0: 2, 2: 4 }, { 0: 2, 2: 4 })).toBe(true);
  });

  it("distinguishes two axes that a label-keyed pin would have collided", () => {
    // Both labelled "z": under the old scheme the second had no key of its own.
    expect(roiVisibleOnPlane({ 0: 1, 1: 7 }, { 0: 1, 1: 7 })).toBe(true);
    expect(roiVisibleOnPlane({ 0: 1, 1: 7 }, { 0: 1, 1: 8 })).toBe(false);
  });

  it("shows rather than hides when a pin cannot be evaluated", () => {
    // An extra ROI on screen is recoverable; a silently hidden one is not.
    expect(roiVisibleOnPlane({ 9: 3 }, { 2: 4 })).toBe(true);
  });
});
