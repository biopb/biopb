import { describe, expect, it } from "vitest";
import { sliderAxes } from "@biopb/tensor-flight-client";
import {
  PLAY_FRAME_MS,
  THUMB_MIN_PX,
  nextPlayIndex,
  orderSliderAxes,
  sliderThumbPx,
} from "./sliceUi";

describe("orderSliderAxes", () => {
  it("presents the named axes as Z, C, T whatever the wire order", () => {
    const axes = sliderAxes(["t", "c", "z", "y", "x"], [4, 3, 16, 512, 512]);
    expect(orderSliderAxes(axes).map((a) => a.title)).toEqual(["Z", "C", "T"]);
  });

  it("puts unnamed axes after the named ones, in wire order", () => {
    const axes = sliderAxes(["POS", "t", "i", "y", "x"], [5, 4, 3, 512, 512]);
    expect(orderSliderAxes(axes).map((a) => a.title)).toEqual(["T", "POS", "i"]);
  });

  it("leaves the wire order alone when nothing is named", () => {
    const axes = sliderAxes(["dim0", "dim1", "y", "x"], [4, 5, 512, 512]);
    expect(orderSliderAxes(axes).map((a) => a.axis)).toEqual([0, 1]);
  });

  it("does not drop or duplicate axes", () => {
    const axes = sliderAxes(["c", "c", "z", "y", "x"], [2, 3, 4, 512, 512]);
    const ordered = orderSliderAxes(axes);
    expect(ordered).toHaveLength(axes.length);
    expect(new Set(ordered.map((a) => a.key)).size).toBe(axes.length);
  });
});

describe("sliderThumbPx", () => {
  it("is the axis's share of the track: half the bar for two channels", () => {
    expect(sliderThumbPx(2, 200)).toBe(100);
    expect(sliderThumbPx(4, 200)).toBe(50);
  });

  it("floors at a grabbable width once the share is a sliver", () => {
    expect(sliderThumbPx(4000, 200)).toBe(THUMB_MIN_PX);
  });

  it("never exceeds the track, and survives an unmeasured one", () => {
    expect(sliderThumbPx(1, 200)).toBe(200);
    expect(sliderThumbPx(2, 8)).toBe(THUMB_MIN_PX);
    expect(sliderThumbPx(3, 0)).toBe(THUMB_MIN_PX);
    expect(sliderThumbPx(Number.NaN, 200)).toBe(200);
  });

  it("never grows as positions are added", () => {
    let prev = Infinity;
    for (let n = 1; n < 500; n++) {
      const w = sliderThumbPx(n, 200);
      expect(w).toBeLessThanOrEqual(prev);
      prev = w;
    }
  });
});

describe("nextPlayIndex", () => {
  it("wraps at the end so play loops", () => {
    expect(nextPlayIndex(0, 3)).toBe(1);
    expect(nextPlayIndex(2, 3)).toBe(0);
  });

  it("stays at 0 for a single-position axis", () => {
    expect(nextPlayIndex(0, 1)).toBe(0);
  });

  it("recovers from an index the tensor cannot hold", () => {
    expect(nextPlayIndex(99, 3)).toBe(0);
    expect(nextPlayIndex(-4, 3)).toBe(1);
  });
});

describe("PLAY_FRAME_MS", () => {
  it("is the configured 10 frames a second", () => {
    expect(PLAY_FRAME_MS).toBe(100);
  });
});
