import { describe, expect, it } from "vitest";
import type { TileInfo } from "@biopb/tensor-flight-client";
import { buildLabelLayers, labelLayerId, labelSelection } from "./labelLayers";
import type { SliceIndices } from "./vivUtils";

function grid(over: Partial<TileInfo>): TileInfo {
  return {
    array_id: "src0",
    dim_labels: ["y", "x"],
    shape: [64, 64],
    chunk_shape: [64, 64],
    dtype: "uint16",
    tile_size: 64,
    plane: { y: 0, x: 1, s: null },
    selectable: { t: null, z: null, c: null },
    sel_axes: [],
    levels: [{ level: 0, scale: 1, width: 64, height: 64, cols: 1, rows: 1 }],
    ...over,
  } as TileInfo;
}

function at(over: Partial<SliceIndices> = {}): SliceIndices {
  return { t: 0, z: 0, c: 0, axes: {}, ...over };
}

/** T C Z Y X image, and the T Z Y X set that spans its non-channel extent. */
const IMAGE = grid({
  array_id: "src0",
  dim_labels: ["t", "c", "z", "y", "x"],
  shape: [50, 3, 20, 64, 64],
  selectable: { t: 0, z: 2, c: 1 },
  plane: { y: 3, x: 4, s: null },
});
const SET = grid({
  array_id: "src0/labels/nuclei",
  dim_labels: ["t", "z", "y", "x"],
  shape: [50, 20, 64, 64],
  selectable: { t: 0, z: 1, c: null },
  plane: { y: 2, x: 3, s: null },
  dtype: "uint32",
});

describe("labelLayerId", () => {
  it("carries Viv's view id, without which the layer is never drawn", () => {
    expect(labelLayerId("nuclei")).toContain("-#detail#");
  });

  it("is distinct per set, so two sets are two layers", () => {
    expect(labelLayerId("nuclei")).not.toBe(labelLayerId("cells"));
  });
});

describe("labelSelection", () => {
  it("carries the named axes across, and drops the channel", () => {
    expect(labelSelection(IMAGE, SET, at({ t: 7, z: 4, c: 2 }))).toEqual({ t: 7, z: 4 });
  });

  it("matches an unnamed axis by position, not by key", () => {
    // The image's unnamed axis is wire 0 and the set's is wire 0 too, but put
    // the channel FIRST and the keys diverge: image `a1`, set `a0`.
    const image = grid({
      dim_labels: ["c", "", "y", "x"],
      shape: [3, 155, 64, 64],
      selectable: { t: null, z: null, c: 0 },
      plane: { y: 2, x: 3, s: null },
    });
    const set = grid({
      dim_labels: ["", "y", "x"],
      shape: [155, 64, 64],
      selectable: { t: null, z: null, c: null },
      plane: { y: 1, x: 2, s: null },
    });
    // The user is on frame 40 of the image, whose slider is keyed `a1`.
    expect(labelSelection(image, set, at({ axes: { a1: 40 } }))).toEqual({ a0: 40 });
  });

  it("clamps to the set's own extent", () => {
    const short = grid({
      dim_labels: ["t", "z", "y", "x"],
      shape: [50, 1, 64, 64],
      selectable: { t: 0, z: 1, c: null },
      plane: { y: 2, x: 3, s: null },
    });
    expect(labelSelection(IMAGE, short, at({ t: 7, z: 4 }))).toEqual({ t: 7, z: 0 });
  });

  it("falls back to matching by name when the rank rule does not hold", () => {
    const odd = grid({
      dim_labels: ["z", "y", "x"],
      shape: [20, 64, 64],
      selectable: { t: null, z: 0, c: null },
      plane: { y: 1, x: 2, s: null },
    });
    // Four non-channel image axes, three in the set: nothing to align, so `z`
    // is read by its name rather than by an index that would name `t`.
    expect(labelSelection(IMAGE, odd, at({ t: 7, z: 4 }))).toEqual({ z: 4 });
  });

  it("handles an image with no channel axis at all", () => {
    expect(labelSelection(SET, SET, at({ t: 7, z: 4 }))).toEqual({ t: 7, z: 4 });
  });
});

describe("buildLabelLayers", () => {
  const sources = [{ one: true }, { two: true }];

  it("draws nothing without a set", () => {
    expect(buildLabelLayers(null)).toEqual([]);
  });

  it("draws nothing when the sources are empty", () => {
    expect(
      buildLabelLayers({ name: "nuclei", sources: [], selection: {}, opacity: 0.5 }),
    ).toEqual([]);
  });

  it("builds one layer, under an id Viv will draw", () => {
    const layers = buildLabelLayers({
      name: "nuclei",
      sources,
      selection: { t: 3 },
      opacity: 0.5,
    }) as Array<{ id: string; props: Record<string, unknown> }>;
    expect(layers).toHaveLength(1);
    expect(layers[0]!.id).toBe(labelLayerId("nuclei"));
    expect(layers[0]!.props.opacity).toBe(0.5);
    expect(layers[0]!.props.selections).toEqual([{ t: 3 }]);
    // The identity ramp: anything else renames every object.
    expect(layers[0]!.props.contrastLimits).toEqual([[0, 1]]);
    expect(layers[0]!.props.interpolation).toBe("nearest");
    expect(layers[0]!.props.pickable).toBe(false);
  });

  it("hands a single level the one source rather than the pyramid", () => {
    const layers = buildLabelLayers({
      name: "nuclei",
      sources: [sources[0]!],
      selection: {},
      opacity: 1,
    }) as Array<{ props: { loader: unknown } }>;
    expect(layers[0]!.props.loader).toBe(sources[0]);
  });

  it("keeps one extensions array, so a slider move recompiles no shader", () => {
    const one = buildLabelLayers({ name: "a", sources, selection: {}, opacity: 1 });
    const two = buildLabelLayers({ name: "a", sources, selection: {}, opacity: 0.2 });
    const props = (l: unknown) => (l as { props: { extensions: unknown } }).props.extensions;
    expect(props(one[0])).toBe(props(two[0]));
  });
});
