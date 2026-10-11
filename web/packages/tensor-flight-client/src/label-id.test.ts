import { describe, expect, it } from "vitest";
import { isReservedLabelName, labelSelection, splitLabelArrayId } from "./label-id.js";
import type { TileInfo } from "./types.js";

describe("splitLabelArrayId", () => {
  it("takes a set on a sole-tensor source apart", () => {
    expect(splitLabelArrayId("src0/@labels/nuclei")).toEqual({
      imageArrayId: "src0",
      name: "nuclei",
      level: null,
    });
  });

  it("keeps the image's own field", () => {
    expect(splitLabelArrayId("plate/A/1/@labels/nuclei")).toEqual({
      imageArrayId: "plate/A/1",
      name: "nuclei",
      level: null,
    });
  });

  it("reads a native level under the set", () => {
    expect(splitLabelArrayId("src0/@labels/nuclei/2")).toEqual({
      imageArrayId: "src0",
      name: "nuclei",
      level: "2",
    });
  });

  it("takes the LAST labels segment, so an image field may hold one", () => {
    // The server splits the same way: an image whose own field says "labels"
    // is not what makes its sets' ids ambiguous, because a name is slash-free.
    expect(splitLabelArrayId("src0/@labels/a/@labels/b")).toEqual({
      imageArrayId: "src0/@labels/a",
      name: "b",
      level: null,
    });
  });

  it("is null for an ordinary tensor", () => {
    expect(splitLabelArrayId("src0")).toBeNull();
    expect(splitLabelArrayId("src0/0")).toBeNull();
  });

  it("is null for a trailing labels segment with no name", () => {
    expect(splitLabelArrayId("src0/labels")).toBeNull();
    expect(splitLabelArrayId("src0/@labels/")).toBeNull();
  });

  it("does not read the source_id as the segment", () => {
    // `labels/nuclei` would be a source called "labels" holding a field
    // "nuclei" -- a source_id is slash-free and is never an image field.
    expect(splitLabelArrayId("labels/nuclei")).toBeNull();
  });
});

describe("isReservedLabelName", () => {
  it("marks the server's own sets", () => {
    expect(isReservedLabelName("@ome")).toBe(true);
    expect(isReservedLabelName("nuclei")).toBe(false);
  });
});

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

/** T C Z Y X image, and the T Z Y X set that spans its non-channel extent. */
const IMAGE = grid({
  dim_labels: ["t", "c", "z", "y", "x"],
  shape: [50, 3, 20, 64, 64],
  selectable: { t: 0, z: 2, c: 1 },
  plane: { y: 3, x: 4, s: null },
});
/** The set of the same image under the rule: its rank, the channel a singleton. */
const SET = grid({
  array_id: "src0/@labels/nuclei",
  dim_labels: ["t", "c", "z", "y", "x"],
  shape: [50, 1, 20, 64, 64],
  selectable: { t: 0, z: 2, c: 1 },
  plane: { y: 3, x: 4, s: null },
  dtype: "uint32",
});

/** An RGB image `T Y X S` and its set: the samples axis is left out. */
const RGB_IMAGE = grid({
  dim_labels: ["t", "y", "x", "s"],
  shape: [50, 64, 64, 3],
  selectable: { t: 0, z: null, c: null },
  plane: { y: 1, x: 2, s: 3 },
});
const RGB_SET = grid({
  array_id: "src0/@labels/nuclei",
  dim_labels: ["t", "y", "x"],
  shape: [50, 64, 64],
  selectable: { t: 0, z: null, c: null },
  plane: { y: 1, x: 2, s: null },
  dtype: "uint32",
});

describe("labelSelection", () => {
  it("lines a same-rank set up by position, the channel clamped to its one plane", () => {
    expect(labelSelection(IMAGE, SET, { t: 7, z: 4, c: 2 })).toEqual({
      t: 7,
      c: 0,
      z: 4,
    });
  });

  it("reads an RGB image's plane without a samples axis on either side", () => {
    expect(labelSelection(RGB_IMAGE, RGB_SET, { t: 7 })).toEqual({ t: 7 });
  });

  it("reads the plane it is given, not a slice position", () => {
    // The caller hands it the selection ON SCREEN, which during play is a frame
    // behind what was asked for. Nothing here may re-derive the plane, or two
    // overlays of one image could disagree about which frame they are drawing.
    expect(labelSelection(IMAGE, SET, { t: 3, z: 9, c: 1 })).toEqual({
      t: 3,
      c: 0,
      z: 9,
    });
  });

  it("matches an unnamed axis by position, not by key", () => {
    const image = grid({
      dim_labels: ["c", "", "y", "x"],
      shape: [3, 155, 64, 64],
      selectable: { t: null, z: null, c: 0 },
      plane: { y: 2, x: 3, s: null },
    });
    const set = grid({
      dim_labels: ["c", "", "y", "x"],
      shape: [1, 155, 64, 64],
      selectable: { t: null, z: null, c: 0 },
      plane: { y: 2, x: 3, s: null },
    });
    expect(labelSelection(image, set, { c: 2, a1: 40 })).toEqual({ c: 0, a1: 40 });
  });

  it("clamps to the set's own extent", () => {
    const short = grid({
      dim_labels: ["t", "c", "z", "y", "x"],
      shape: [50, 1, 1, 64, 64],
      selectable: { t: 0, z: 2, c: 1 },
      plane: { y: 3, x: 4, s: null },
    });
    expect(labelSelection(IMAGE, short, { t: 7, z: 4 })).toEqual({ t: 7, c: 0, z: 0 });
  });

  it("handles an image with no channel axis at all", () => {
    const image = grid({
      dim_labels: ["t", "z", "y", "x"],
      shape: [50, 20, 64, 64],
      selectable: { t: 0, z: 1, c: null },
      plane: { y: 2, x: 3, s: null },
    });
    expect(labelSelection(image, image, { t: 7, z: 4 })).toEqual({ t: 7, z: 4 });
  });
});
