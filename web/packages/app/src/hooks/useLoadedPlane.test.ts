import { describe, expect, it } from "vitest";
import { viewportHoldsPlane } from "./useLoadedPlane";

describe("viewportHoldsPlane", () => {
  it("takes a single raster, which ImageLayer reports only once it resolved", () => {
    expect(viewportHoldsPlane({ data: new Uint8Array(1) })).toBe(true);
    expect(viewportHoldsPlane(undefined)).toBe(true);
  });

  it("takes a viewport whose tiles all have content", () => {
    expect(viewportHoldsPlane([{ content: {} }, { content: {} }])).toBe(true);
  });

  it("refuses a viewport with a failed tile, which deck.gl still calls loaded", () => {
    expect(viewportHoldsPlane([{ content: {} }, { content: null }])).toBe(false);
    expect(viewportHoldsPlane([null])).toBe(false);
  });
});
