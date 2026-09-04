import { describe, expect, it } from "vitest";
import {
  MIN_CANVAS_WIDTH,
  MIN_CONTROL_WIDTH,
  MIN_SIDEBAR_WIDTH,
  clampPaneWidth,
} from "./paneWidth";

describe("clampPaneWidth", () => {
  it("passes a width that leaves room for the other pane and the canvas", () => {
    expect(clampPaneWidth(420, MIN_SIDEBAR_WIDTH, 320, 1920)).toBe(420);
  });

  it("holds the pane at its own minimum", () => {
    expect(clampPaneWidth(40, MIN_SIDEBAR_WIDTH, 320, 1920)).toBe(MIN_SIDEBAR_WIDTH);
  });

  it("stops where the canvas would start disappearing", () => {
    // 1200 wide, 320 already taken by the other pane: at most 1200-320-320.
    expect(clampPaneWidth(900, MIN_CONTROL_WIDTH, 320, 1200)).toBe(1200 - 320 - MIN_CANVAS_WIDTH);
  });

  it("keeps the pane usable on a window too narrow for all three", () => {
    // The minimum wins over an upper bound that has gone below it: a cramped
    // window should still show a pane you can read, not a 40px sliver.
    expect(clampPaneWidth(500, MIN_SIDEBAR_WIDTH, 400, 600)).toBe(MIN_SIDEBAR_WIDTH);
  });

  it("reports no opinion for a value that is not a number", () => {
    expect(clampPaneWidth(Number.NaN, MIN_SIDEBAR_WIDTH, 320, 1920)).toBe(0);
  });
});
