import { describe, it, expect } from "vitest";
import type { RoiAnnotation, RoiGeometry, TileInfo } from "@biopb/tensor-flight-client";
import {
  ELLIPSE_SEGMENTS,
  buildRoiLayers,
  buildSelectionLayers,
  currentPlaneFor,
  roiLayerId,
  planeFromSelection,
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
      rotation: 0,
    })!;
    expect(ring).toHaveLength(ELLIPSE_SEGMENTS);
    const xs = ring.map((p) => p[0]);
    const ys = ring.map((p) => p[1]);
    expect(Math.min(...xs)).toBeCloseTo(6, 6);
    expect(Math.max(...xs)).toBeCloseTo(14, 6);
    expect(Math.min(...ys)).toBeCloseTo(18, 6);
    expect(Math.max(...ys)).toBeCloseTo(22, 6);
  });

  it("turns an ellipse a quarter turn onto its other axis", () => {
    const ring = roiRing({
      kind: "ellipse",
      center: { x: 10, y: 20 },
      radius: { x: 4, y: 2 },
      rotation: Math.PI / 2,
    })!;
    const xs = ring.map((p) => p[0]);
    const ys = ring.map((p) => p[1]);
    // The extents have swapped: what was 8 wide is now 8 tall.
    expect(Math.min(...xs)).toBeCloseTo(8, 6);
    expect(Math.max(...xs)).toBeCloseTo(12, 6);
    expect(Math.min(...ys)).toBeCloseTo(16, 6);
    expect(Math.max(...ys)).toBeCloseTo(24, 6);
  });

  it("rotates about the centre rather than the origin", () => {
    const ring = roiRing({
      kind: "ellipse",
      center: { x: 100, y: 200 },
      radius: { x: 5, y: 5 },
      rotation: 0.9,
    })!;
    // A circle is rotation-invariant, so a rotation that moved the centre
    // would show up here as a displaced ring and nowhere else.
    for (const [x, y] of ring) {
      expect(Math.hypot(x - 100, y - 200)).toBeCloseTo(5, 6);
    }
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
    roi({ roiId: "a", geometry: POLY, plane: { 1: 4 } }),
    roi({ roiId: "b", geometry: POLY, plane: { 1: 9 } }),
    roi({ roiId: "c", geometry: POLY, plane: {} }),
    roi({ roiId: "d", geometry: POLY, setName: "other", plane: { 1: 4 } }),
  ];

  it("keeps this plane's annotations and the unpinned ones", () => {
    const ids = visibleRois(set, { 1: 4 }, []).map((r) => r.roiId);
    expect(ids).toEqual(["a", "c", "d"]);
  });

  it("drops a hidden set", () => {
    const ids = visibleRois(set, { 1: 4 }, ["other"]).map((r) => r.roiId);
    expect(ids).toEqual(["a", "c"]);
  });

  it("keeps an unpinned annotation on every plane", () => {
    expect(visibleRois(set, { 1: 99 }, []).map((r) => r.roiId)).toEqual(["c"]);
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
    expect(currentPlaneFor(info, { t: 3, z: 12, c: 0, axes: {} })).toEqual({ 0: 3, 1: 12 });
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
    // Its index lives under the slider key `a0`; the pin names axis 0.
    expect(currentPlaneFor(unnamed, { t: 0, z: 7, c: 0, axes: { a0: 2 } })).toEqual({
      0: 2,
      1: 7,
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
    // Four, not three: a polyline is a band plus its centreline, which is how
    // every other kind is drawn too -- a translucent area under an opaque edge.
    expect(layers).toHaveLength(4);
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
    const pinned = [roi({ geometry: POLY, plane: { 1: 1 } })];
    expect(
      buildRoiLayers({ rois: pinned, currentPlane: { 1: 2 }, hiddenSets: [], visible: true }),
    ).toEqual([]);
  });

  it("draws every kind as a translucent area under an opaque outline", () => {
    // The polyline used to be the exception: one opaque band, which hid the
    // pixels it was pointing at and left it looking nothing like the rest.
    const layers = buildRoiLayers({
      rois: set,
      currentPlane: {},
      hiddenSets: [],
      visible: true,
    }) as Array<{ id: string; props: Record<string, unknown> }>;
    const alpha = (id: string, accessor: string) => {
      const layer = layers.find((l) => l.id === roiLayerId(id));
      const get = layer?.props[accessor] as (d: unknown) => number[];
      const datum = (layer?.props.data as unknown[])[0];
      return get(datum)[3];
    };
    for (const [id, accessor] of [
      ["shapes", "getFillColor"],
      ["path-bands", "getColor"],
      ["points", "getFillColor"],
    ] as const) {
      expect(alpha(id, accessor)).toBeLessThan(128);
    }
    for (const [id, accessor] of [
      ["shapes", "getLineColor"],
      ["paths", "getColor"],
      ["points", "getLineColor"],
    ] as const) {
      expect(alpha(id, accessor)).toBeGreaterThan(200);
    }
  });

  it("builds the same layers whatever is selected", () => {
    // The point of splitting selection out: a click must not change this
    // memo's output, or deck.gl regenerates every attribute of every layer --
    // earcut included -- for the whole set.
    const opts = { rois: set, currentPlane: {}, hiddenSets: [], visible: true };
    const layers = buildRoiLayers(opts) as Array<{ id: string; props: Record<string, unknown> }>;
    for (const layer of layers) {
      for (const accessor of ["getLineColor", "getColor", "getFillColor"]) {
        const get = layer.props[accessor];
        // Not a function of the datum's id: nothing here can vary by selection.
        if (typeof get === "function") {
          const datum = (layer.props.data as unknown[])[0] as Record<string, unknown>;
          const roi = (datum.roi ?? datum) as Record<string, unknown>;
          expect((get as (d: unknown) => number[])({ ...datum, roi: { ...roi, roiId: "other" } }))
            .toEqual((get as (d: unknown) => number[])(datum));
        }
      }
    }
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

describe("buildSelectionLayers", () => {
  const line = roi({ roiId: "b", geometry: LINE });

  it("draws nothing without a selection", () => {
    expect(buildSelectionLayers(null)).toEqual([]);
  });

  it("emphasises one annotation, whatever the set holds", () => {
    // One datum, so a click costs the same at ten annotations and at the
    // server's five thousand.
    const layers = buildSelectionLayers(line) as Array<{
      id: string;
      props: Record<string, unknown>;
    }>;
    expect(layers).toHaveLength(1);
    expect(layers[0]!.id).toContain("-#detail#");
    expect((layers[0]!.props.data as unknown[])).toHaveLength(1);
    expect(layers[0]!.props.pickable).toBe(false);
  });

  it("emphasises the outline, and draws no area of its own", () => {
    // The area belongs to the layer underneath: selecting an annotation is not
    // supposed to change what it claims, only to mark which one it is.
    for (const geometry of [POLY, LINE, POINT]) {
      const layers = buildSelectionLayers(roi({ geometry })) as Array<{
        props: Record<string, unknown>;
      }>;
      const props = layers[0]!.props;
      expect(props.filled ?? false).toBe(false);
      const color = (props.getLineColor ?? props.getColor) as number[];
      expect(color).toEqual([255, 255, 255, 255]);
    }
  });

  it("outlines a selected shape more heavily than the overlay does", () => {
    const base = buildRoiLayers({
      rois: [roi({ geometry: POLY })],
      currentPlane: {},
      hiddenSets: [],
      visible: true,
    }) as Array<{ props: Record<string, unknown> }>;
    const picked = buildSelectionLayers(roi({ geometry: POLY })) as Array<{
      props: Record<string, unknown>;
    }>;
    // Wider, so it covers the outline it stands in for rather than fringing it.
    expect(picked[0]!.props.getLineWidth as number).toBeGreaterThan(
      base[0]!.props.getLineWidth as number,
    );
  });
});

describe("planeFromSelection", () => {
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

  it("translates Viv's key-addressed selection into dim-label pins", () => {
    // This is how the overlay reads the plane that is actually on screen:
    // TileViewer's `loadedKey` is a serialised selection of exactly this shape.
    expect(planeFromSelection(info, { t: 3, z: 12 })).toEqual({ 0: 3, 1: 12 });
  });

  it("agrees with currentPlaneFor for the same position", () => {
    // The two must not drift: one drives the overlay, the other the panel count.
    const slice = { t: 3, z: 12, c: 0, axes: {} };
    expect(currentPlaneFor(info, slice)).toEqual(planeFromSelection(info, { t: 3, z: 12 }));
  });

  it("reads an unnamed axis under its slider key", () => {
    const unnamed: TileInfo = {
      ...info,
      dim_labels: ["POS", "z", "y", "x"],
      shape: [5, 20, 512, 512],
      selectable: { t: null, z: 1, c: null },
    };
    expect(planeFromSelection(unnamed, { a0: 2, z: 7 })).toEqual({ 0: 2, 1: 7 });
  });

  it("is empty before the grid is known", () => {
    expect(planeFromSelection(null, { z: 3 })).toEqual({});
  });
});
