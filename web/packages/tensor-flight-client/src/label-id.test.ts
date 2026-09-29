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
const SET = grid({
  array_id: "src0/@labels/nuclei",
  dim_labels: ["t", "z", "y", "x"],
  shape: [50, 20, 64, 64],
  selectable: { t: 0, z: 1, c: null },
  plane: { y: 2, x: 3, s: null },
  dtype: "uint32",
  image_axes: [0, 2, 3, 4],
});

describe("labelSelection", () => {
  it("carries the named axes across, and drops the channel", () => {
    expect(labelSelection(IMAGE, SET, { t: 7, z: 4, c: 2 })).toEqual({ t: 7, z: 4 });
  });

  it("reads the plane it is given, not a slice position", () => {
    // The caller hands it the selection ON SCREEN, which during play is a frame
    // behind what was asked for. Nothing here may re-derive the plane, or two
    // overlays of one image could disagree about which frame they are drawing.
    expect(labelSelection(IMAGE, SET, { t: 3, z: 9, c: 1 })).toEqual({ t: 3, z: 9 });
  });

  it("uses the mapping the server states, not one it re-derives", () => {
    // Load-bearing: the stated mapping is the whole point of the server
    // publishing it. A set whose `image_axes` says otherwise is followed.
    const swapped = grid({ ...SET, image_axes: [2, 0, 3, 4] });
    expect(labelSelection(IMAGE, swapped, { t: 7, z: 4 })).toEqual({ t: 4, z: 7 });
  });

  it("matches an unnamed axis by position, not by key", () => {
    // The image's unnamed axis is keyed `a1` and the set's `a0`: matching by
    // key would read frame 0 where frame 40 was asked for.
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
      image_axes: [1, 2, 3],
    });
    expect(labelSelection(image, set, { a1: 40 })).toEqual({ a0: 40 });
  });

  it("clamps to the set's own extent", () => {
    const short = grid({
      dim_labels: ["t", "z", "y", "x"],
      shape: [50, 1, 64, 64],
      selectable: { t: 0, z: 1, c: null },
      plane: { y: 2, x: 3, s: null },
      image_axes: [0, 2, 3, 4],
    });
    expect(labelSelection(IMAGE, short, { t: 7, z: 4 })).toEqual({ t: 7, z: 0 });
  });

  describe("against a server that states nothing", () => {
    const unstated = (over: Partial<TileInfo>) => {
      const info = grid(over);
      delete info.image_axes;
      return info;
    };

    it("re-derives the extent rule, reaching the same answer", () => {
      const set = unstated({ ...SET });
      expect(labelSelection(IMAGE, set, { t: 7, z: 4, c: 2 })).toEqual({ t: 7, z: 4 });
    });

    it("falls back to matching by key when the rank rule does not hold", () => {
      // Four non-channel image axes, three in the set: nothing to align, so `z`
      // is read by its name rather than by an index that would name `t`.
      const odd = unstated({
        dim_labels: ["z", "y", "x"],
        shape: [20, 64, 64],
        selectable: { t: null, z: 0, c: null },
        plane: { y: 1, x: 2, s: null },
      });
      expect(labelSelection(IMAGE, odd, { t: 7, z: 4 })).toEqual({ z: 4 });
    });

    it("handles an image with no channel axis at all", () => {
      const set = unstated({ ...SET });
      expect(labelSelection(set, set, { t: 7, z: 4 })).toEqual({ t: 7, z: 4 });
    });
  });

  it("ignores a stated mapping of the wrong length", () => {
    // A server and a set that disagree about rank: the statement cannot be
    // about this set, so the rule answers instead of an index out of range.
    const bad = grid({ ...SET, image_axes: [0, 2] });
    expect(labelSelection(IMAGE, bad, { t: 7, z: 4 })).toEqual({ t: 7, z: 4 });
  });
});
