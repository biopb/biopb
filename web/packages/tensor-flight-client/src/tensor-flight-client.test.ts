import { describe, expect, it, vi } from "vitest";

import { TensorFlightClient } from "./tensor-flight-client.js";
import type { DataSourceDescriptor, TileInfo, TypedNdArray } from "./types.js";

const SLICE = { data: new Uint16Array(4), shape: [2, 2], dtype: "uint16", dim_labels: ["y", "x"] };

const TILE_INFO = {
  array_id: "src0/Image:0",
  dim_labels: ["y", "x"],
  shape: [2, 2],
  chunk_shape: [2, 2],
  dtype: "uint16",
} as unknown as TileInfo;

const LISTED: DataSourceDescriptor = {
  source_id: "src0",
  source_url: "/data/src0",
  source_type: "zarr",
  is_resolved: true,
  tensors: [{ array_id: "src0", dim_labels: ["y", "x"], shape: [2, 2], dtype: "uint16" }],
};

describe("TensorFlightClient.getTensor", () => {
  it("describes an unlisted tensor itself, from its own array_id", async () => {
    const c = new TensorFlightClient("http://x", null);
    const tileInfo = vi.spyOn(c.http, "tileInfo").mockResolvedValue(TILE_INFO);
    const slice = vi.spyOn(c.http, "slice").mockResolvedValue(SLICE as unknown as TypedNdArray);

    await c.getTensor("src0/Image:0").compute();

    expect(tileInfo).toHaveBeenCalledTimes(1);
    expect(tileInfo.mock.calls[0]![0]).toBe("src0/Image:0");
    expect(slice.mock.calls[0]![0].array_id).toBe("src0/Image:0");
    expect(slice.mock.calls[0]![0].slice_stop).toEqual([2, 2]);
  });

  it("asks once however many computes are waiting", async () => {
    const c = new TensorFlightClient("http://x", null);
    const tileInfo = vi.spyOn(c.http, "tileInfo").mockResolvedValue(TILE_INFO);
    vi.spyOn(c.http, "slice").mockResolvedValue(SLICE as unknown as TypedNdArray);

    const t = c.getTensor("src0/Image:0");
    await Promise.all([t.compute(), t.compute()]);

    expect(tileInfo).toHaveBeenCalledTimes(1);
  });

  it("uses a listed source without asking the server", async () => {
    const c = new TensorFlightClient("http://x", null);
    vi.spyOn(c.http, "listSourcesPage").mockResolvedValue({ sources: [LISTED], truncated: false });
    const tileInfo = vi.spyOn(c.http, "tileInfo");
    const slice = vi.spyOn(c.http, "slice").mockResolvedValue(SLICE as unknown as TypedNdArray);

    await c.listSourcesPage(10);
    await c.getTensor("src0").compute();

    expect(tileInfo).not.toHaveBeenCalled();
    expect(slice).toHaveBeenCalledTimes(1);
  });
});
