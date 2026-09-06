import { describe, it, expect } from "vitest";
import type { RoiAnnotation, RoiGeometry } from "@biopb/tensor-flight-client";
import { HIT_SLOP_PX, distanceToSegment, hitsRoi, pointInRing, roiAt } from "./roiHitTest";

function roi(geometry: RoiGeometry, roiId = "r"): RoiAnnotation {
  return {
    roiId,
    arrayId: "src/Image:0",
    setName: "default",
    label: "",
    geometry,
    plane: {},
    props: {},
    rev: 1,
    createdAtMs: 0,
    updatedAtMs: 0,
  };
}

const SQUARE: RoiGeometry = {
  kind: "polygon",
  points: [{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 100, y: 100 }, { x: 0, y: 100 }],
};

describe("pointInRing", () => {
  const ring: Array<[number, number]> = [[0, 0], [100, 0], [100, 100], [0, 100]];

  it("is true inside and false outside", () => {
    expect(pointInRing([50, 50], ring)).toBe(true);
    expect(pointInRing([150, 50], ring)).toBe(false);
    expect(pointInRing([50, -1], ring)).toBe(false);
  });

  it("handles a concave shape", () => {
    // A "C": the notch is outside even though it is within the bounding box.
    const c: Array<[number, number]> = [
      [0, 0], [100, 0], [100, 30], [40, 30], [40, 70], [100, 70], [100, 100], [0, 100],
    ];
    expect(pointInRing([20, 50], c)).toBe(true);
    expect(pointInRing([70, 50], c)).toBe(false);
  });
});

describe("distanceToSegment", () => {
  it("measures perpendicular distance when the foot is on the segment", () => {
    expect(distanceToSegment([50, 10], [0, 0], [100, 0])).toBeCloseTo(10);
  });

  it("clamps to an endpoint rather than the infinite line", () => {
    // Off the end: the answer is the distance to the endpoint, not to the line.
    expect(distanceToSegment([120, 0], [0, 0], [100, 0])).toBeCloseTo(20);
  });

  it("handles a degenerate segment", () => {
    expect(distanceToSegment([3, 4], [0, 0], [0, 0])).toBeCloseTo(5);
  });
});

describe("hitsRoi", () => {
  it("hits anywhere inside a filled shape", () => {
    expect(hitsRoi(roi(SQUARE), [50, 50], 1)).toBe(true);
  });

  it("hits just outside the outline, within slop", () => {
    // A thin sliver has almost no interior, so the outline has to be grabbable.
    expect(hitsRoi(roi(SQUARE), [-2, 50], 5)).toBe(true);
    expect(hitsRoi(roi(SQUARE), [-20, 50], 5)).toBe(false);
  });

  it("hits a polyline by proximity to the stroke", () => {
    const line: RoiGeometry = {
      kind: "polyline",
      points: [{ x: 0, y: 0 }, { x: 100, y: 0 }],
      width: 0,
    };
    expect(hitsRoi(roi(line), [50, 3], 5)).toBe(true);
    expect(hitsRoi(roi(line), [50, 30], 5)).toBe(false);
  });

  it("hits a point by proximity", () => {
    const point: RoiGeometry = { kind: "point", at: { x: 10, y: 10 } };
    expect(hitsRoi(roi(point), [12, 12], 5)).toBe(true);
    expect(hitsRoi(roi(point), [30, 30], 5)).toBe(false);
  });

  it("hits an ellipse through its tessellated ring", () => {
    const ellipse: RoiGeometry = {
      kind: "ellipse",
      center: { x: 50, y: 50 },
      radius: { x: 20, y: 10 },
    };
    expect(hitsRoi(roi(ellipse), [50, 50], 1)).toBe(true);
    expect(hitsRoi(roi(ellipse), [50, 45], 1)).toBe(true);
    // Inside the bounding box, outside the ellipse.
    expect(hitsRoi(roi(ellipse), [68, 58], 1)).toBe(false);
  });
});

describe("roiAt", () => {
  const rois = [roi(SQUARE, "under"), roi(SQUARE, "over")];

  it("returns the topmost hit, which is the one drawn last", () => {
    expect(roiAt(rois, [50, 50], 1)?.roiId).toBe("over");
  });

  it("returns null where nothing is", () => {
    expect(roiAt(rois, [500, 500], 1)).toBeNull();
  });

  it("scales its tolerance with the zoom", () => {
    const point = [roi({ kind: "point", at: { x: 0, y: 0 } })];
    const justOutside = HIT_SLOP_PX + 2;
    expect(roiAt(point, [justOutside, 0], 1)).toBeNull();
    // Zoomed out, the same world distance is well inside the screen tolerance.
    expect(roiAt(point, [justOutside, 0], 4)?.roiId).toBe("r");
  });
});
