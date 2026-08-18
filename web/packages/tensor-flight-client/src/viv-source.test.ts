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
    const sources = pixelSourcesFromInfo(stubClient(), INFO);
    expect(sources).toHaveLength(2);
    expect(sources[0]!.shape[INFO.plane.x]).toBe(1024);
    expect(sources[1]!.shape[INFO.plane.x]).toBe(512);
  });

  it("shrinks only the plane across levels", () => {
    const [full, half] = pixelSourcesFromInfo(stubClient(), INFO);
    // Slider axes are the same data at every zoom.
    expect(full!.shape.slice(0, 3)).toEqual([1, 3, 16]);
    expect(half!.shape.slice(0, 3)).toEqual([1, 3, 16]);
  });

  it("carries dtype and tileSize onto every source", () => {
    for (const s of pixelSourcesFromInfo(stubClient(), INFO)) {
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
    const [, half] = pixelSourcesFromInfo(client, INFO);
    await half!.getTile({ x: 0, y: 0, selection: { t: 0, c: 0, z: 0 } });
    expect(client.tile.mock.calls[0]![0]).toMatchObject({
      array_id: "src0/Image:0", level: 1, col: 0, row: 0,
    });
  });

  it("translates a label-keyed selection into t/z/c", async () => {
    const client = stubClient();
    const [full] = pixelSourcesFromInfo(client, INFO);
    await full!.getTile({ x: 1, y: 1, selection: { t: 0, c: 2, z: 7 } });
    expect(client.tile.mock.calls[0]![0]).toMatchObject({ c: 2, z: 7, col: 1, row: 1 });
  });

  it("returns a typed array view, not the raw buffer", async () => {
    const client = stubClient();
    const [full] = pixelSourcesFromInfo(client, INFO);
    const px = await full!.getTile({ x: 0, y: 0, selection: { t: 0, c: 0, z: 0 } });
    expect(px.data).toBeInstanceOf(Uint16Array);
    expect(px.data.length).toBe(512 * 512);
    expect([px.width, px.height]).toEqual([512, 512]);
  });

  it("carries `meta`, which Viv dereferences unguarded for interleaved data", () => {
    // ImageLayer.renderLayers does `const { photometricInterpretation = 2 } =
    // loader.meta` on the interleaved branch. The destructuring default covers a
    // missing property, not a missing object, so a source without `meta` throws
    // inside deck.gl and renders nothing -- while the tiles fetch normally and
    // the canvas just stays blank. `meta` is optional in Viv's own type, so
    // nothing but this test holds the line.
    const client = stubClient({});
    for (const info of [RGB_INFO, INFO]) {
      for (const source of pixelSourcesFromInfo(client, info)) {
        expect(source.meta).toBeDefined();
        expect(source.meta!.photometricInterpretation).toBe(2);
      }
    }
  });

  it("reports the served size for a short edge tile", async () => {
    const client = stubClient({
      tile: vi.fn(async (_req: TileRequest, _opts?: RequestOptions) => ({
        buffer: new ArrayBuffer(387 * 387 * 3),
        shape: [1, 1, 1, 387, 387, 3],
        dtype: "uint8", dimLabels: [], tileSize: 512, level: 0, col: 2, row: 2,
      })),
    });
    const [full] = pixelSourcesFromInfo(client, RGB_INFO);
    const px = await full!.getTile({ x: 2, y: 2, selection: { t: 0, c: 0, z: 0 } });
    expect([px.width, px.height]).toEqual([387, 387]);
  });

  it("answers an out-of-grid tile locally instead of spending a round trip", async () => {
    const client = stubClient();
    const [, half] = pixelSourcesFromInfo(client, INFO);
    // Level 1 is a single tile; deck.gl can still ask past the edge mid-settle.
    const px = await half!.getTile({ x: 5, y: 0, selection: { t: 0, c: 0, z: 0 } });
    expect(client.tile).not.toHaveBeenCalled();
    expect(px.data.every((v) => v === 0)).toBe(true);
    // Zeros alone is a trivially-true assertion on an undersized buffer, so
    // pin the length too.
    expect(px.data.length).toBe(px.width * px.height);
  });

  it("sizes an out-of-grid RGB tile for its samples, not just its pixels", async () => {
    // Interleaved RGB carries 3 values per pixel. A plain width*height buffer
    // is a third of what the layer uploads, so WebGL rejects the texture or
    // reads past the end -- and only when a viewport pans off the edge.
    const client = stubClient({
      tile: vi.fn(async (_req: TileRequest, _opts?: RequestOptions) => ({
        buffer: new ArrayBuffer(512 * 512 * 3),
        shape: [1, 1, 1, 512, 512, 3],
        dtype: "uint8", dimLabels: [], tileSize: 512, level: 0, col: 0, row: 0,
      })),
    });
    const [full] = pixelSourcesFromInfo(client, RGB_INFO);
    const real = await full!.getTile({ x: 0, y: 0, selection: { t: 0, c: 0, z: 0 } });
    const oob = await full!.getTile({ x: 99, y: 0, selection: { t: 0, c: 0, z: 0 } });
    expect(oob.data.length).toBe(real.data.length);
    expect(oob.data.length).toBe(512 * 512 * 3);
    expect(oob.data).toBeInstanceOf(Uint8Array);
  });

  it("forwards Viv's abort signal to the request", async () => {
    const client = stubClient();
    const [full] = pixelSourcesFromInfo(client, INFO);
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
    const [full] = pixelSourcesFromInfo(stubClient(), info);
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
    const [full] = pixelSourcesFromInfo(stubClient(), info);
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
    const [, half] = pixelSourcesFromInfo(client, INFO);
    await half!.getRaster({ selection: { t: 0, c: 1, z: 3 } });
    const req = client.slice.mock.calls[0]![0];
    expect(req.scale_hint).toEqual([1, 1, 1, 2, 2]);
    // Plane full, slider axes pinned to the selected index.
    expect(req.slice_start).toEqual([0, 1, 3, 0, 0]);
    expect(req.slice_stop).toEqual([1, 2, 4, 1024, 1024]);
  });

  it("refuses a level too large to pull in one read", async () => {
    const client = stubClient();
    const [full] = pixelSourcesFromInfo(client, INFO, { maxRasterPixels: 1000 });
    await expect(full!.getRaster({ selection: { t: 0, c: 0, z: 0 } }))
      .rejects.toThrow(/would read 1048576 pixels/);
    expect(client.slice).not.toHaveBeenCalled();
  });

  it("allows it when the caller raises the budget deliberately", async () => {
    const client = stubClient();
    const [full] = pixelSourcesFromInfo(client, INFO, { maxRasterPixels: 2e6 });
    await expect(full!.getRaster({ selection: { t: 0, c: 0, z: 0 } })).resolves.toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// onTileError
// ---------------------------------------------------------------------------

describe("PixelSource.onTileError", () => {
  it("swallows a cancelled tile", () => {
    const onTileError = vi.fn();
    const [full] = pixelSourcesFromInfo(stubClient(), INFO, { onTileError });
    full!.onTileError(new TensorAbortError("/api/tile/src0"));
    // Panning away is not a failure and must not reach the error UI.
    expect(onTileError).not.toHaveBeenCalled();
  });

  it("forwards a real failure", () => {
    const onTileError = vi.fn();
    const [full] = pixelSourcesFromInfo(stubClient(), INFO, { onTileError });
    const err = new Error("502 upstream");
    full!.onTileError(err);
    expect(onTileError).toHaveBeenCalledWith(err);
  });
});

// ---------------------------------------------------------------------------
// getRaster coalescing (#772)
// ---------------------------------------------------------------------------

/** A `slice` that never settles on its own, so in-flight state is observable. */
function deferredClient() {
  const calls: {
    signal?: AbortSignal;
    settle: () => void;
    fail: (err: unknown) => void;
  }[] = [];
  const slice = vi.fn(
    (_req: SliceRequest, opts?: RequestOptions) =>
      new Promise((resolve, reject) => {
        opts?.signal?.addEventListener("abort", () =>
          reject(new TensorAbortError("/api/slice")),
        );
        calls.push({
          signal: opts?.signal,
          settle: () =>
            resolve({
              buffer: new ArrayBuffer(512 * 512 * 2),
              shape: [1, 1, 1, 512, 512],
              dtype: "uint16",
              dimLabels: ["T", "C", "Z", "Y", "X"],
            }),
          fail: reject,
        });
      }),
  );
  return { client: stubClient({ slice }), calls, slice };
}

/** What Viv's ImageLayer swallows; anything else becomes an unhandled rejection. */
const SIGNAL_ABORTED = "__vivSignalAborted";

const SEL_A = { t: 0, c: 0, z: 0 };
const SEL_B = { t: 0, c: 0, z: 1 };

describe("getRaster request sharing", () => {
  it("serves two callers for the same plane from one read", async () => {
    const { client, calls, slice } = deferredClient();
    const [, half] = pixelSourcesFromInfo(client, INFO);
    const a = half!.getRaster({ selection: SEL_A });
    const b = half!.getRaster({ selection: SEL_A });
    // The background ImageLayer and the contrast sampler ask for exactly this,
    // independently, on every selection change.
    expect(slice).toHaveBeenCalledTimes(1);
    calls[0]!.settle();
    expect((await a).width).toBe(512);
    expect((await b).width).toBe(512);
  });

  it("does not confuse two levels of the same plane", async () => {
    const { client, slice } = deferredClient();
    const [full, half] = pixelSourcesFromInfo(client, INFO);
    void full!.getRaster({ selection: SEL_A }).catch(() => {});
    void half!.getRaster({ selection: SEL_A }).catch(() => {});
    expect(slice).toHaveBeenCalledTimes(2);
  });

  it("starts a fresh read once the shared one has settled", async () => {
    const { client, calls, slice } = deferredClient();
    const [, half] = pixelSourcesFromInfo(client, INFO);
    const first = half!.getRaster({ selection: SEL_A });
    calls[0]!.settle();
    await first;
    // Sharing is de-duplication, not caching: a later ask still reads.
    void half!.getRaster({ selection: SEL_A }).catch(() => {});
    expect(slice).toHaveBeenCalledTimes(2);
  });
});

describe("getRaster superseding", () => {
  it("aborts every level of the plane that is no longer wanted", async () => {
    const { client, calls } = deferredClient();
    const [full, half] = pixelSourcesFromInfo(client, INFO);
    const stale = [
      full!.getRaster({ selection: SEL_A }),
      half!.getRaster({ selection: SEL_A }),
    ];
    expect(calls.map((c) => c.signal?.aborted)).toEqual([false, false]);

    const wanted = half!.getRaster({ selection: SEL_B });
    // Both of the previous selection's reads, not just the one re-asked.
    expect(calls.map((c) => c.signal?.aborted)).toEqual([true, true, false]);
    await expect(stale[0]).rejects.toBe(SIGNAL_ABORTED);
    await expect(stale[1]).rejects.toBe(SIGNAL_ABORTED);

    calls[2]!.settle();
    await expect(wanted).resolves.toBeDefined();
  });

  it("rejects with the value Viv's ImageLayer swallows", async () => {
    // It ends its chain with `catch(e => { if (e !== SIGNAL_ABORTED) throw e })`,
    // so an AbortError or a TensorAbortError here is an unhandled rejection.
    const { client, calls } = deferredClient();
    const [, half] = pixelSourcesFromInfo(client, INFO);
    const stale = half!.getRaster({ selection: SEL_A });
    void half!.getRaster({ selection: SEL_B }).catch(() => {});
    await expect(stale).rejects.toBe(SIGNAL_ABORTED);
    expect(calls[0]!.signal?.aborted).toBe(true);
  });

  it("passes a real failure through untouched", async () => {
    const { client, calls } = deferredClient();
    const [, half] = pixelSourcesFromInfo(client, INFO);
    const raster = half!.getRaster({ selection: SEL_A });
    const err = new Error("502 upstream");
    calls[0]!.fail(err);
    await expect(raster).rejects.toBe(err);
  });
});

describe("getRaster caller signals", () => {
  it("detaches one caller without cancelling the other's read", async () => {
    const { client, calls } = deferredClient();
    const [, half] = pixelSourcesFromInfo(client, INFO);
    const mine = new AbortController();
    const leaving = half!.getRaster({ selection: SEL_A, signal: mine.signal });
    const staying = half!.getRaster({ selection: SEL_A });

    mine.abort();
    await expect(leaving).rejects.toBe(SIGNAL_ABORTED);
    // The shared read belongs to whoever is still waiting on it.
    expect(calls[0]!.signal?.aborted).toBe(false);
    calls[0]!.settle();
    await expect(staying).resolves.toBeDefined();
  });

  it("cancels the read when the last waiter leaves", async () => {
    const { client, calls } = deferredClient();
    const [, half] = pixelSourcesFromInfo(client, INFO);
    const one = new AbortController();
    const two = new AbortController();
    const a = half!.getRaster({ selection: SEL_A, signal: one.signal });
    const b = half!.getRaster({ selection: SEL_A, signal: two.signal });

    one.abort();
    expect(calls[0]!.signal?.aborted).toBe(false);
    two.abort();
    // Nobody is left to receive the answer, so it stops costing a connection.
    expect(calls[0]!.signal?.aborted).toBe(true);
    await expect(a).rejects.toBe(SIGNAL_ABORTED);
    await expect(b).rejects.toBe(SIGNAL_ABORTED);
  });

  it("does not start a read for a caller that has already given up", async () => {
    const { client, calls } = deferredClient();
    const [, half] = pixelSourcesFromInfo(client, INFO);
    const dead = new AbortController();
    dead.abort();
    await expect(half!.getRaster({ selection: SEL_A, signal: dead.signal }))
      .rejects.toBe(SIGNAL_ABORTED);
    expect(calls).toHaveLength(0);
  });

  it("lets a caller that has already given up supersede nothing", async () => {
    const { client, calls } = deferredClient();
    const [, half] = pixelSourcesFromInfo(client, INFO);
    const wanted = half!.getRaster({ selection: SEL_A });
    const dead = new AbortController();
    dead.abort();
    // A different selection, so taking this one at face value would retire the
    // read somebody is still waiting on.
    await expect(half!.getRaster({ selection: SEL_B, signal: dead.signal }))
      .rejects.toBe(SIGNAL_ABORTED);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.signal?.aborted).toBe(false);
    calls[0]!.settle();
    await expect(wanted).resolves.toBeDefined();
  });
});
