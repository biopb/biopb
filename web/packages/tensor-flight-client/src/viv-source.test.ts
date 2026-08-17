/**
 * Unit tests for viv-source.ts: the Viv PixelSource adapter over the tile API.
 *
 * The client is a stub; these check the mapping (labels, dtype, per-level shape,
 * tile addressing, selection) rather than transport, which client.test.ts covers.
 */

import { describe, it, expect, vi } from "vitest";
import {
  createTensorPixelSources,
  pixelSourcesFromInfo,
  vivDtype,
  vivLabels,
} from "./viv-source.js";
import { TensorAbortError } from "./client.js";
import type { RequestOptions, TensorHttpClient } from "./client.js";
import type { SliceRequest, TileInfo, TileRequest } from "./types.js";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** TCZYX uint16, 1024x1024 plane in 512px tiles -> two levels. */
const INFO: TileInfo = {
  array_id: "src0/Image:0",
  dim_labels: ["T", "C", "Z", "Y", "X"],
  shape: [1, 3, 16, 1024, 1024],
  chunk_shape: [1, 1, 1, 512, 512],
  dtype: "<u2",
  tile_size: 512,
  plane: { y: 3, x: 4, s: null },
  selectable: { t: 0, c: 1, z: 2 },
  pinned: [],
  levels: [
    { level: 0, scale: 1, height: 1024, width: 1024, cols: 2, rows: 2 },
    { level: 1, scale: 2, height: 512, width: 512, cols: 1, rows: 1 },
  ],
};

/** Interleaved RGB: the samples axis becomes Viv's "_c". */
const RGB_INFO: TileInfo = {
  array_id: "rgb/Image:0",
  dim_labels: ["T", "C", "Z", "Y", "X", "S"],
  shape: [1, 1, 1, 1411, 1411, 3],
  chunk_shape: [1, 1, 1, 1411, 1411, 3],
  dtype: "|u1",
  tile_size: 512,
  plane: { y: 3, x: 4, s: 5 },
  selectable: { t: 0, c: 1, z: 2 },
  pinned: [],
  levels: [
    { level: 0, scale: 1, height: 1411, width: 1411, cols: 3, rows: 3 },
    { level: 1, scale: 2, height: 706, width: 706, cols: 2, rows: 2 },
    { level: 2, scale: 4, height: 353, width: 353, cols: 1, rows: 1 },
  ],
};

function stubClient(overrides: Partial<Record<string, unknown>> = {}) {
  const tile = vi.fn(async (_req: TileRequest, _opts?: RequestOptions) => ({
    buffer: new ArrayBuffer(512 * 512 * 2),
    shape: [1, 1, 1, 512, 512],
    dtype: "uint16",
    dimLabels: ["T", "C", "Z", "Y", "X"],
    tileSize: 512,
    level: 0,
    col: 0,
    row: 0,
  }));
  const slice = vi.fn(async (_req: SliceRequest, _opts?: RequestOptions) => ({
    buffer: new ArrayBuffer(512 * 512 * 2),
    shape: [1, 1, 1, 512, 512],
    dtype: "uint16",
    dimLabels: ["T", "C", "Z", "Y", "X"],
  }));
  const tileInfo = vi.fn(async (_id: string, _tensorId?: string, _opts?: RequestOptions) => INFO);
  return { tile, slice, tileInfo, ...overrides } as unknown as TensorHttpClient & {
    tile: typeof tile; slice: typeof slice; tileInfo: typeof tileInfo;
  };
}

// ---------------------------------------------------------------------------
// dtype
// ---------------------------------------------------------------------------

describe("vivDtype", () => {
  it("accepts the descriptor spelling from tile_info", () => {
    expect(vivDtype("<u2")).toBe("Uint16");
    expect(vivDtype("|u1")).toBe("Uint8");
    expect(vivDtype("<f4")).toBe("Float32");
  });

  it("accepts the header spelling from the tile response", () => {
    // Same tensor, two spellings; both reach the adapter on different paths.
    expect(vivDtype("uint16")).toBe("Uint16");
    expect(vivDtype("float64")).toBe("Float64");
  });

  it("rejects a dtype Viv cannot sample", () => {
    expect(() => vivDtype("<c8")).toThrow(/no Viv equivalent/);
  });
});

// ---------------------------------------------------------------------------
// labels
// ---------------------------------------------------------------------------

describe("vivLabels", () => {
  it("lowercases and puts the plane last, as Viv requires", () => {
    expect(vivLabels(INFO)).toEqual(["t", "c", "z", "y", "x"]);
  });

  it("renames an interleaved samples axis to _c", () => {
    expect(vivLabels(RGB_INFO)).toEqual(["t", "c", "z", "y", "x", "_c"]);
  });

  it("rejects a tensor whose plane is not last rather than transposing silently", () => {
    const bad = { ...INFO, plane: { y: 0, x: 1, s: null } };
    expect(() => vivLabels(bad)).toThrow(/canonical/);
  });
});

// ---------------------------------------------------------------------------
// Source construction
// ---------------------------------------------------------------------------

describe("pixelSourcesFromInfo", () => {
  it("returns one source per level, index 0 full resolution", () => {
    const sources = pixelSourcesFromInfo(stubClient(), "src0", INFO);
    expect(sources).toHaveLength(2);
    expect(sources[0]!.shape[INFO.plane.x]).toBe(1024);
    expect(sources[1]!.shape[INFO.plane.x]).toBe(512);
  });

  it("shrinks only the plane across levels", () => {
    const [full, half] = pixelSourcesFromInfo(stubClient(), "src0", INFO);
    // Slider axes are the same data at every zoom.
    expect(full!.shape.slice(0, 3)).toEqual([1, 3, 16]);
    expect(half!.shape.slice(0, 3)).toEqual([1, 3, 16]);
  });

  it("carries dtype and tileSize onto every source", () => {
    for (const s of pixelSourcesFromInfo(stubClient(), "src0", INFO)) {
      expect(s.dtype).toBe("Uint16");
      expect(s.tileSize).toBe(512);
    }
  });

  it("fetches tile_info once and returns it alongside", async () => {
    const client = stubClient();
    const { data, info } = await createTensorPixelSources(client, "src0");
    expect(client.tileInfo).toHaveBeenCalledTimes(1);
    expect(info).toBe(INFO);
    expect(data).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// getTile
// ---------------------------------------------------------------------------

describe("PixelSource.getTile", () => {
  it("maps Viv's x/y to the tile grid at this source's level", async () => {
    const client = stubClient();
    const [, half] = pixelSourcesFromInfo(client, "src0", INFO);
    await half!.getTile({ x: 0, y: 0, selection: { t: 0, c: 0, z: 0 } });
    expect(client.tile.mock.calls[0]![0]).toMatchObject({
      source_id: "src0", level: 1, col: 0, row: 0,
    });
  });

  it("translates a label-keyed selection into t/z/c", async () => {
    const client = stubClient();
    const [full] = pixelSourcesFromInfo(client, "src0", INFO);
    await full!.getTile({ x: 1, y: 1, selection: { t: 0, c: 2, z: 7 } });
    expect(client.tile.mock.calls[0]![0]).toMatchObject({ c: 2, z: 7, col: 1, row: 1 });
  });

  it("returns a typed array view, not the raw buffer", async () => {
    const client = stubClient();
    const [full] = pixelSourcesFromInfo(client, "src0", INFO);
    const px = await full!.getTile({ x: 0, y: 0, selection: { t: 0, c: 0, z: 0 } });
    expect(px.data).toBeInstanceOf(Uint16Array);
    expect(px.data.length).toBe(512 * 512);
    expect([px.width, px.height]).toEqual([512, 512]);
  });

  it("reports the served size for a short edge tile", async () => {
    const client = stubClient({
      tile: vi.fn(async (_req: TileRequest, _opts?: RequestOptions) => ({
        buffer: new ArrayBuffer(387 * 387 * 3),
        shape: [1, 1, 1, 387, 387, 3],
        dtype: "uint8", dimLabels: [], tileSize: 512, level: 0, col: 2, row: 2,
      })),
    });
    const [full] = pixelSourcesFromInfo(client, "rgb", RGB_INFO);
    const px = await full!.getTile({ x: 2, y: 2, selection: { t: 0, c: 0, z: 0 } });
    expect([px.width, px.height]).toEqual([387, 387]);
  });

  it("answers an out-of-grid tile locally instead of spending a round trip", async () => {
    const client = stubClient();
    const [, half] = pixelSourcesFromInfo(client, "src0", INFO);
    // Level 1 is a single tile; deck.gl can still ask past the edge mid-settle.
    const px = await half!.getTile({ x: 5, y: 0, selection: { t: 0, c: 0, z: 0 } });
    expect(client.tile).not.toHaveBeenCalled();
    expect(px.data.every((v) => v === 0)).toBe(true);
  });

  it("forwards Viv's abort signal to the request", async () => {
    const client = stubClient();
    const [full] = pixelSourcesFromInfo(client, "src0", INFO);
    const ctrl = new AbortController();
    await full!.getTile({ x: 0, y: 0, selection: { t: 0, c: 0, z: 0 }, signal: ctrl.signal });
    expect(client.tile.mock.calls[0]![1]).toEqual({ signal: ctrl.signal });
  });

  it("refuses to select an axis the tile API cannot address", async () => {
    const info: TileInfo = {
      ...INFO,
      dim_labels: ["POS", "C", "Z", "Y", "X"],
      selectable: { t: null, c: 1, z: 2 },
    };
    const [full] = pixelSourcesFromInfo(stubClient(), "src0", info);
    // Index 0 is the correct default, so only a non-zero request is wrong.
    await expect(
      full!.getTile({ x: 0, y: 0, selection: { pos: 0, c: 0, z: 0 } }),
    ).resolves.toBeDefined();
    await expect(
      full!.getTile({ x: 0, y: 0, selection: { pos: 3, c: 0, z: 0 } }),
    ).rejects.toThrow(/cannot be selected/);
  });

  it("refuses a non-zero index on an axis the tensor does not have", async () => {
    const info: TileInfo = { ...INFO, selectable: { t: null, c: 1, z: 2 } };
    const [full] = pixelSourcesFromInfo(stubClient(), "src0", info);
    await expect(
      full!.getTile({ x: 0, y: 0, selection: { t: 4, c: 0, z: 0 } }),
    ).rejects.toThrow(/no "t" axis/);
  });
});

// ---------------------------------------------------------------------------
// getRaster
// ---------------------------------------------------------------------------

describe("PixelSource.getRaster", () => {
  it("reads the whole level in one slice at that level's scale", async () => {
    const client = stubClient();
    const [, half] = pixelSourcesFromInfo(client, "src0", INFO);
    await half!.getRaster({ selection: { t: 0, c: 1, z: 3 } });
    const req = client.slice.mock.calls[0]![0];
    expect(req.scale_hint).toEqual([1, 1, 1, 2, 2]);
    // Plane full, slider axes pinned to the selected index.
    expect(req.slice_start).toEqual([0, 1, 3, 0, 0]);
    expect(req.slice_stop).toEqual([1, 2, 4, 1024, 1024]);
  });

  it("refuses a level too large to pull in one read", async () => {
    const client = stubClient();
    const [full] = pixelSourcesFromInfo(client, "src0", INFO, { maxRasterPixels: 1000 });
    await expect(full!.getRaster({ selection: { t: 0, c: 0, z: 0 } }))
      .rejects.toThrow(/would read 1048576 pixels/);
    expect(client.slice).not.toHaveBeenCalled();
  });

  it("allows it when the caller raises the budget deliberately", async () => {
    const client = stubClient();
    const [full] = pixelSourcesFromInfo(client, "src0", INFO, { maxRasterPixels: 2e6 });
    await expect(full!.getRaster({ selection: { t: 0, c: 0, z: 0 } })).resolves.toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// onTileError
// ---------------------------------------------------------------------------

describe("PixelSource.onTileError", () => {
  it("swallows a cancelled tile", () => {
    const onTileError = vi.fn();
    const [full] = pixelSourcesFromInfo(stubClient(), "src0", INFO, { onTileError });
    full!.onTileError(new TensorAbortError("/api/tile/src0"));
    // Panning away is not a failure and must not reach the error UI.
    expect(onTileError).not.toHaveBeenCalled();
  });

  it("forwards a real failure", () => {
    const onTileError = vi.fn();
    const [full] = pixelSourcesFromInfo(stubClient(), "src0", INFO, { onTileError });
    const err = new Error("502 upstream");
    full!.onTileError(err);
    expect(onTileError).toHaveBeenCalledWith(err);
  });
});
