import { describe, it, expect } from "vitest";
import type { RoiAnnotation, RoiGeometry, TileInfo } from "@biopb/tensor-flight-client";
import {
  ELLIPSE_SEGMENTS,
  buildRoiLayers,
  currentPlaneFor,
  roiLayerId,
  roiPath,
  roiRing,
  roiSetCounts,
  setColor,
  visibleRois,
} from "./roiLayers";

function roi(over: Partial<RoiAnnotation> & { geometry: RoiGeometry }): RoiAnnotation {
  return {
    roiId: "r",
    arrayId: "src/Image:0",
    setName: "default",
    label: "",
    plane: {},
    props: {},
    rev: 1,
    createdAtMs: 0,
    updatedAtMs: 0,
    ...over,
  };
}

const POINT: RoiGeometry = { kind: "point", at: { x: 3, y: 4 } };
const POLY: RoiGeometry = {
  kind: "polygon",
  points: [{ x: 0, y: 0 }, { x: 4, y: 0 }, { x: 4, y: 4 }],
};
const LINE: RoiGeometry = { kind: "polyline", points: [{ x: 0, y: 0 }, { x: 9, y: 3 }], width: 2 };

describe("roiLayerId", () => {
  it("carries Viv's view id, without which the layer is never drawn", () => {
    // VivViewer filters every layer through
    // `layer.id.includes(getVivId(viewport.id))`, silently.
    expect(roiLayerId("shapes")).toContain("-#detail#");
  });

  it("is distinct per layer", () => {
    expect(roiLayerId("shapes")).not.toBe(roiLayerId("points"));
  });
});

describe("roiRing", () => {
  it("passes a polygon's vertices through", () => {
    expect(roiRing(POLY)).toEqual([[0, 0], [4, 0], [4, 4]]);
  });

  it("turns a rectangle into four corners, wound consistently", () => {
    const ring = roiRing({
      kind: "rectangle",
      topLeft: { x: 1, y: 2 },
      bottomRight: { x: 5, y: 9 },
    });
    expect(ring).toEqual([[1, 2], [5, 2], [5, 9], [1, 9]]);
  });

  it("tessellates an ellipse to its own extent", () => {
    const ring = roiRing({
      kind: "ellipse",
      center: { x: 10, y: 20 },
      radius: { x: 4, y: 2 },
    })!;
    expect(ring).toHaveLength(ELLIPSE_SEGMENTS);
    const xs = ring.map((p) => p[0]);
    const ys = ring.map((p) => p[1]);
    expect(Math.min(...xs)).toBeCloseTo(6, 6);
    expect(Math.max(...xs)).toBeCloseTo(14, 6);
    expect(Math.min(...ys)).toBeCloseTo(18, 6);
    expect(Math.max(...ys)).toBeCloseTo(22, 6);
  });

  it("declines the arms that are not polygons", () => {
    expect(roiRing(POINT)).toBeNull();
    expect(roiRing(LINE)).toBeNull();
  });
});

describe("roiPath", () => {
  it("takes only the polyline", () => {
    expect(roiPath(LINE)).toEqual([[0, 0], [9, 3]]);
    expect(roiPath(POLY)).toBeNull();
  });
});

describe("setColor", () => {
  it("is stable for a name", () => {
    expect(setColor("nuclei")).toEqual(setColor("nuclei"));
  });

  it("does not depend on what else exists", () => {
    // Assigned by name, so a set keeps its colour as others come and go.
    const before = setColor("nuclei");
    setColor("membranes");
    expect(setColor("nuclei")).toEqual(before);
  });
});

describe("visibleRois", () => {
  const set = [
    roi({ roiId: "a", geometry: POLY, plane: { z: 4 } }),
    roi({ roiId: "b", geometry: POLY, plane: { z: 9 } }),
    roi({ roiId: "c", geometry: POLY, plane: {} }),
    roi({ roiId: "d", geometry: POLY, setName: "other", plane: { z: 4 } }),
  ];

  it("keeps this plane's annotations and the unpinned ones", () => {
    const ids = visibleRois(set, { z: 4 }, []).map((r) => r.roiId);
    expect(ids).toEqual(["a", "c", "d"]);
  });

  it("drops a hidden set", () => {
    const ids = visibleRois(set, { z: 4 }, ["other"]).map((r) => r.roiId);
    expect(ids).toEqual(["a", "c"]);
  });

  it("keeps an unpinned annotation on every plane", () => {
    expect(visibleRois(set, { z: 99 }, []).map((r) => r.roiId)).toEqual(["c"]);
  });
});

describe("roiSetCounts", () => {
  it("counts per set in first-seen order", () => {
    expect(
      roiSetCounts([
        roi({ geometry: POLY, setName: "b" }),
        roi({ geometry: POLY, setName: "a" }),
        roi({ geometry: POLY, setName: "b" }),
      ]),
    ).toEqual([
      { setName: "b", count: 2 },
      { setName: "a", count: 1 },
    ]);
  });
});

describe("currentPlaneFor", () => {
  const info: TileInfo = {
    array_id: "src/Image:0",
    dim_labels: ["t", "z", "y", "x"],
    shape: [10, 20, 512, 512],
    chunk_shape: [1, 1, 256, 256],
    dtype: "uint16",
    tile_size: 256,
    plane: { y: 2, x: 3, s: null },
    selectable: { t: 0, z: 1, c: null },
    sel_axes: [],
    levels: [],
  };

  it("keys the viewer's current indices by dim label", () => {
    expect(currentPlaneFor(info, { t: 3, z: 12, c: 0, axes: {} })).toEqual({ t: 3, z: 12 });
  });

  it("is empty before the grid is known", () => {
    expect(currentPlaneFor(null, { t: 3, z: 12, c: 0, axes: {} })).toEqual({});
  });

  it("reads an unnamed axis by its slider key", () => {
    const unnamed: TileInfo = {
      ...info,
      dim_labels: ["POS", "z", "y", "x"],
      shape: [5, 20, 512, 512],
      selectable: { t: null, z: 1, c: null },
    };
    // "POS" is a real label, so it can hold a pin; its index lives under `a0`.
    expect(currentPlaneFor(unnamed, { t: 0, z: 7, c: 0, axes: { a0: 2 } })).toEqual({
      POS: 2,
      z: 7,
    });
  });
});

describe("buildRoiLayers", () => {
  const set = [
    roi({ roiId: "a", geometry: POLY }),
    roi({ roiId: "b", geometry: LINE }),
    roi({ roiId: "c", geometry: POINT }),
  ];

  it("builds one layer per geometry family, all addressable by Viv", () => {
    const layers = buildRoiLayers({
      rois: set,
      currentPlane: {},
      hiddenSets: [],
      visible: true,
    }) as Array<{ id: string }>;
    expect(layers).toHaveLength(3);
    for (const layer of layers) expect(layer.id).toContain("-#detail#");
  });

  it("omits a family with nothing in it", () => {
    const layers = buildRoiLayers({
      rois: [roi({ geometry: POINT })],
      currentPlane: {},
      hiddenSets: [],
      visible: true,
    }) as Array<{ id: string }>;
    expect(layers.map((l) => l.id)).toEqual([roiLayerId("points")]);
  });

  it("draws nothing when the overlay is off", () => {
    expect(buildRoiLayers({ rois: set, currentPlane: {}, hiddenSets: [], visible: false })).toEqual(
      [],
    );
  });

  it("draws nothing when the plane filter empties the set", () => {
    const pinned = [roi({ geometry: POLY, plane: { z: 1 } })];
    expect(
      buildRoiLayers({ rois: pinned, currentPlane: { z: 2 }, hiddenSets: [], visible: true }),
    ).toEqual([]);
  });

  it("leaves every layer unpickable, so the pixel readout still answers", () => {
    // TileViewer's hover badge reads `info.sourceLayer`/`info.tile`; a pickable
    // overlay would answer the hover itself and blank the readout over an ROI.
    const layers = buildRoiLayers({
      rois: set,
      currentPlane: {},
      hiddenSets: [],
      visible: true,
    }) as Array<{ props: { pickable: boolean } }>;
    for (const layer of layers) expect(layer.props.pickable).toBe(false);
  });
});
