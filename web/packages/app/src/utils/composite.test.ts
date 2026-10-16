import { describe, expect, it } from "vitest";
import type { TileInfo } from "@biopb/tensor-flight-client";
import {
  COMPOSITE_MAX_CHANNELS,
  compositeChannels,
  compositeColors,
  compositeSelections,
  compositeWindows,
  withChannel,
} from "./composite";

const TCZYX: TileInfo = {
  array_id: "src/Image:0",
  dim_labels: ["T", "C", "Z", "Y", "X"],
  shape: [4, 3, 16, 1024, 1024],
  chunk_shape: [1, 1, 1, 512, 512],
  dtype: "<u2",
  tile_size: 512,
  plane: { y: 3, x: 4, s: null },
  selectable: { t: 0, c: 1, z: 2 },
  sel_axes: [],
  levels: [{ level: 0, scale: 1, height: 1024, width: 1024, cols: 2, rows: 2 }],
};

const withC = (c: number): TileInfo => ({ ...TCZYX, shape: [4, c, 16, 1024, 1024] });

describe("compositeChannels", () => {
  it("lists every channel of a 2-4 channel image once asked", () => {
    expect(compositeChannels(withC(2), true)).toEqual([0, 1]);
    expect(compositeChannels(TCZYX, true)).toEqual([0, 1, 2]);
    expect(compositeChannels(withC(COMPOSITE_MAX_CHANNELS), true)).toEqual([0, 1, 2, 3]);
  });

  it("does nothing until asked", () => {
    expect(compositeChannels(TCZYX, false)).toBeNull();
  });

  it("does not apply to one channel, or to more than the cap", () => {
    expect(compositeChannels(withC(1), true)).toBeNull();
    expect(compositeChannels(withC(COMPOSITE_MAX_CHANNELS + 1), true)).toBeNull();
  });

  it("does not apply to interleaved RGB, which is already colour", () => {
    const rgb: TileInfo = {
      ...TCZYX,
      dim_labels: ["T", "C", "Z", "Y", "X", "S"],
      shape: [1, 3, 1, 1024, 1024, 3],
      plane: { y: 3, x: 4, s: 5 },
    };
    expect(compositeChannels(rgb, true)).toBeNull();
  });

  it("does not apply to an image with no channel axis, or an unnamed one", () => {
    const plain: TileInfo = { ...TCZYX, dim_labels: ["Y", "X"], shape: [64, 64], plane: { y: 0, x: 1, s: null } };
    const sequence: TileInfo = {
      ...TCZYX,
      dim_labels: ["i", "y", "x"],
      shape: [3, 64, 64],
      plane: { y: 1, x: 2, s: null },
    };
    expect(compositeChannels(plain, true)).toBeNull();
    // Three files on an opaque axis are not three channels.
    expect(compositeChannels(sequence, true)).toBeNull();
  });
});

describe("compositeSelections", () => {
  it("differs between channels only on the channel axis", () => {
    const sels = compositeSelections(TCZYX, { t: 2, z: 5, c: 1, axes: {} }, [0, 1, 2]);
    // The slider's own c (1) does not leak into any of them.
    expect(sels).toEqual([
      { t: 2, c: 0, z: 5 },
      { t: 2, c: 1, z: 5 },
      { t: 2, c: 2, z: 5 },
    ]);
  });
});

describe("withChannel", () => {
  it("moves a selection to another channel and nothing else", () => {
    const [first] = compositeSelections(TCZYX, { t: 2, z: 5, c: 0, axes: {} }, [0]);
    expect(withChannel(TCZYX, first as Record<string, number>, 2)).toEqual({ t: 2, c: 2, z: 5 });
  });

  it("clamps to the axis", () => {
    const [first] = compositeSelections(TCZYX, { t: 0, z: 0, c: 0, axes: {} }, [0]);
    expect(withChannel(TCZYX, first as Record<string, number>, 9).c).toBe(2);
  });
});

describe("compositeColors", () => {
  it("takes the conventional pairing for unnamed channels", () => {
    expect(compositeColors(["auto", "auto"], [undefined, undefined])).toEqual(["green", "magenta"]);
    expect(compositeColors(["auto", "auto", "auto"], [])).toEqual(["red", "green", "blue"]);
    expect(compositeColors(["auto", "auto", "auto", "auto"], [])).toEqual(["red", "green", "blue", "gray"]);
  });

  it("keeps a colour the user picked, and a colour a name implies", () => {
    expect(compositeColors(["#123456", "auto", "auto"], [undefined, "DAPI", "Channel 2"])).toEqual([
      "#123456",
      "blue",
      "red",
    ]);
  });

  it("never gives two unnamed channels the same colour as a named one", () => {
    // GFP takes green, so the unnamed pair skips it.
    expect(compositeColors(["auto", "auto", "auto"], ["GFP", "x", "y"])).toEqual(["green", "red", "blue"]);
  });

  it("does not let two names that imply one colour blend into it", () => {
    // Cy7 and Cy5 are both far-red, which is magenta; the second gives way.
    expect(compositeColors(["auto", "auto", "auto", "auto"], ["Cy7", "Cy5", "Cy3", "GFP"])).toEqual([
      "magenta",
      "red",
      "cyan",
      "green",
    ]);
  });

  it("lets a colour the user picked claim it before any name does", () => {
    expect(compositeColors(["auto", "magenta"], ["Cy5", undefined])).toEqual(["green", "magenta"]);
  });

  it("treats a name it does not recognise as unnamed", () => {
    expect(compositeColors(["auto", "auto"], ["Channel 0", "Channel 1"])).toEqual(["green", "magenta"]);
  });

  it("leaves an explicit grey alone", () => {
    expect(compositeColors(["gray", "auto"], [undefined, undefined])).toEqual(["gray", "green"]);
  });
});

describe("compositeWindows", () => {
  const values = (n: number) => new Float64Array(Array.from({ length: n }, (_, i) => i));

  it("windows each channel on its own samples", () => {
    const w = compositeWindows([0, 1], { 0: { values: values(101) }, 1: { values: values(1001) } }, 1);
    expect(w).toEqual([
      [1, 99],
      [10, 990],
    ]);
  });

  it("is the full range at a scale of 0", () => {
    expect(compositeWindows([0], { 0: { values: values(101) } }, 0)).toEqual([[0, 100]]);
  });

  it("leaves a channel with no samples yet null", () => {
    expect(compositeWindows([0, 1], { 0: { values: values(11) } }, 0)).toEqual([[0, 10], null]);
  });
});
