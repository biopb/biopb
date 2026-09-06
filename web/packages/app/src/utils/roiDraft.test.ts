import { describe, it, expect } from "vitest";
import {
  CLOSE_HANDLE_PX,
  DEFAULT_POLYLINE_WIDTH,
  MAX_POLYLINE_WIDTH,
  MIN_POLYLINE_WIDTH,
  closeDraft,
  closesOnFirstVertex,
  isCompletable,
  isTextEntryTarget,
  minimumPoints,
  placePoint,
  undoPoint,
  type RoiDraft,
} from "./roiDraft";

describe("placePoint", () => {
  it("completes a point on the first click", () => {
    const { draft, completed } = placePoint(null, "point", [3, 4]);
    expect(draft).toBeNull();
    expect(completed).toEqual({ kind: "point", at: { x: 3, y: 4 } });
  });

  it("completes a rectangle on the second", () => {
    const first = placePoint(null, "rectangle", [10, 20]);
    expect(first.completed).toBeUndefined();
    const second = placePoint(first.draft, "rectangle", [4, 30]);
    // Normalised, whichever way round the two clicks went.
    expect(second.completed).toEqual({
      kind: "rectangle",
      topLeft: { x: 4, y: 20 },
      bottomRight: { x: 10, y: 30 },
    });
    expect(second.draft).toBeNull();
  });

  it("accumulates a polygon without completing it", () => {
    let draft: RoiDraft | null = null;
    for (const at of [[0, 0], [4, 0], [4, 4], [0, 4]] as Array<[number, number]>) {
      const result = placePoint(draft, "polygon", at);
      expect(result.completed).toBeUndefined();
      draft = result.draft;
    }
    expect(draft?.points).toHaveLength(4);
  });

  it("starts over when the tool changes mid-draft", () => {
    const started = placePoint(null, "polygon", [1, 1]);
    const switched = placePoint(started.draft, "polyline", [9, 9]);
    expect(switched.draft).toEqual({ tool: "polyline", points: [[9, 9]] });
  });

  it("does nothing under the select tool", () => {
    expect(placePoint(null, "select", [1, 1])).toEqual({ draft: null });
  });
});

describe("closeDraft", () => {
  const ring: Array<[number, number]> = [[0, 0], [4, 0], [4, 4]];

  it("turns a polygon draft into geometry", () => {
    expect(closeDraft({ tool: "polygon", points: ring })).toEqual({
      kind: "polygon",
      points: [{ x: 0, y: 0 }, { x: 4, y: 0 }, { x: 4, y: 4 }],
    });
  });

  it("turns a polyline draft into geometry, at a width somebody would choose", () => {
    // Zero was the old default. Width is geometry -- the band of pixels the
    // stroke covers -- and a zero-width one renders at the layer's hairline
    // floor at every zoom, so it came out a construction line rather than
    // something traced over structure.
    expect(closeDraft({ tool: "polyline", points: ring.slice(0, 2) })).toEqual({
      kind: "polyline",
      points: [{ x: 0, y: 0 }, { x: 4, y: 0 }],
      width: DEFAULT_POLYLINE_WIDTH,
    });
    expect(DEFAULT_POLYLINE_WIDTH).toBeGreaterThan(1);
  });

  it("stores the width it is given, clamped to what the control offers", () => {
    const at = (width: number) =>
      (closeDraft({ tool: "polyline", points: ring.slice(0, 2) }, width) as { width: number }).width;
    expect(at(12)).toBe(12);
    expect(at(0)).toBe(MIN_POLYLINE_WIDTH);
    expect(at(1e6)).toBe(MAX_POLYLINE_WIDTH);
    expect(at(Number.NaN)).toBe(DEFAULT_POLYLINE_WIDTH);
  });

  it("refuses a polygon of two vertices", () => {
    // Not a lesser polygon -- a line, which the store would reject anyway.
    expect(closeDraft({ tool: "polygon", points: ring.slice(0, 2) })).toBeNull();
  });

  it("refuses a polyline of one", () => {
    expect(closeDraft({ tool: "polyline", points: ring.slice(0, 1) })).toBeNull();
  });

  it("refuses nothing at all", () => {
    expect(closeDraft(null)).toBeNull();
  });
});

describe("isCompletable / minimumPoints", () => {
  it("knows what each tool needs", () => {
    expect(minimumPoints("point")).toBe(1);
    expect(minimumPoints("rectangle")).toBe(2);
    expect(minimumPoints("polyline")).toBe(2);
    expect(minimumPoints("polygon")).toBe(3);
  });

  it("is false for a short draft and for none", () => {
    expect(isCompletable({ tool: "polygon", points: [[0, 0], [1, 1]] })).toBe(false);
    expect(isCompletable(null)).toBe(false);
  });
});

describe("undoPoint", () => {
  it("drops the last vertex", () => {
    expect(undoPoint({ tool: "polygon", points: [[0, 0], [1, 1]] })).toEqual({
      tool: "polygon",
      points: [[0, 0]],
    });
  });

  it("dropping the only vertex ends the draft", () => {
    expect(undoPoint({ tool: "polygon", points: [[0, 0]] })).toBeNull();
    expect(undoPoint(null)).toBeNull();
  });
});

describe("closesOnFirstVertex", () => {
  const draft: RoiDraft = { tool: "polygon", points: [[100, 100], [140, 100], [140, 140]] };

  it("closes on a click within the handle", () => {
    // scale = world units per screen pixel, so the handle is constant on screen.
    expect(closesOnFirstVertex(draft, [102, 102], 1)).toBe(true);
  });

  it("does not close on a click beyond it", () => {
    expect(closesOnFirstVertex(draft, [100 + CLOSE_HANDLE_PX + 2, 100], 1)).toBe(false);
  });

  it("keeps the handle the same size on screen when zoomed out", () => {
    // At 4 world units per pixel the same screen distance is 4x the world one.
    expect(closesOnFirstVertex(draft, [100 + 3 * CLOSE_HANDLE_PX, 100], 4)).toBe(true);
    expect(closesOnFirstVertex(draft, [100 + 3 * CLOSE_HANDLE_PX, 100], 1)).toBe(false);
  });

  it("never closes a draft too short to complete", () => {
    expect(closesOnFirstVertex({ tool: "polygon", points: [[0, 0]] }, [0, 0], 1)).toBe(false);
  });

  it("never closes a fixed-vertex tool", () => {
    expect(closesOnFirstVertex({ tool: "rectangle", points: [[0, 0]] }, [0, 0], 1)).toBe(false);
  });
});

describe("isTextEntryTarget", () => {
  const target = (tagName: string, isContentEditable = false) =>
    ({ tagName, isContentEditable }) as unknown as EventTarget;

  it("claims the fields a draft's window-level keys would otherwise reach", () => {
    // The source search, the chat composer, the label/set fields, the slice
    // inputs: Backspace in any of them must edit text, not take back a vertex.
    expect(isTextEntryTarget(target("INPUT"))).toBe(true);
    expect(isTextEntryTarget(target("TEXTAREA"))).toBe(true);
    expect(isTextEntryTarget(target("SELECT"))).toBe(true);
  });

  it("claims a contenteditable host whatever its tag", () => {
    expect(isTextEntryTarget(target("DIV", true))).toBe(true);
  });

  it("leaves the canvas and the document body alone", () => {
    expect(isTextEntryTarget(target("CANVAS"))).toBe(false);
    expect(isTextEntryTarget(target("BODY"))).toBe(false);
    expect(isTextEntryTarget(target("BUTTON"))).toBe(false);
  });

  it("is case-insensitive, for a target reported in lower case", () => {
    expect(isTextEntryTarget(target("input"))).toBe(true);
  });

  it("survives a target that is not an element", () => {
    expect(isTextEntryTarget(null)).toBe(false);
    expect(isTextEntryTarget({} as EventTarget)).toBe(false);
  });
});
