/**
 * Unit tests for tensor-array.ts:
 *   - buildAxisMap
 *   - isAxisMapAmbiguous
 *   - computeScaleHint
 *   - TensorArray.compute() slice range assembly
 */

import { describe, it, expect, vi, type Mock } from "vitest";

import {
  buildAxisMap,
  isAxisMapAmbiguous,
  computeScaleHint,
  TensorArray,
  type AxisMap,
} from "./tensor-array.js";
import type { TensorHttpClient } from "./client.js";
import type { TensorDescriptor, TypedNdArray } from "./types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeDesc(
  dimLabels: string[],
  shape?: number[],
  dtype = "uint16",
  arrayId = "t0",
): TensorDescriptor {
  return {
    array_id: arrayId,
    dim_labels: dimLabels,
    shape: shape ?? dimLabels.map(() => 64),
    chunk_shape: dimLabels.map(() => 32),
    dtype,
  };
}

/** Minimal TensorHttpClient stub. slice() returns a resolved promise. */
function makeClient(returnVal?: TypedNdArray): TensorHttpClient {
  const stub: Partial<TensorHttpClient> = {
    slice: vi.fn().mockResolvedValue(
      returnVal ?? {
        buffer: new ArrayBuffer(0),
        shape: [],
        dtype: "uint8",
        dimLabels: [],
      },
    ),
  };
  return stub as TensorHttpClient;
}

// ---------------------------------------------------------------------------
// buildAxisMap
// ---------------------------------------------------------------------------

describe("buildAxisMap", () => {
  it("maps explicit tzcyx labels", () => {
    const m = buildAxisMap(["t", "z", "c", "y", "x"]);
    expect(m).toEqual<AxisMap>({ t: 0, z: 1, c: 2, y: 3, x: 4 });
  });

  it("maps explicit yx only (2-D)", () => {
    const m = buildAxisMap(["y", "x"]);
    expect(m.y).toBe(0);
    expect(m.x).toBe(1);
    expect(m.z).toBeNull();
    expect(m.t).toBeNull();
    expect(m.c).toBeNull();
  });

  it("maps aliases: depth → z, width → x, height → y, time → t, channel → c", () => {
    const m = buildAxisMap(["time", "depth", "channel", "height", "width"]);
    expect(m).toEqual<AxisMap>({ t: 0, z: 1, c: 2, y: 3, x: 4 });
  });

  it("does not duplicate z when labels are channel-first c,y,x", () => {
    const m = buildAxisMap(["c", "y", "x"]);
    expect(m).toEqual<AxisMap>({ t: null, z: null, c: 0, y: 1, x: 2 });
  });

  it("does not duplicate c when labels are z,y,x", () => {
    const m = buildAxisMap(["z", "y", "x"]);
    expect(m).toEqual<AxisMap>({ t: null, z: 0, c: null, y: 1, x: 2 });
  });

  it("applies positional heuristic for unknown labels (last=x, second-last=y)", () => {
    const m = buildAxisMap(["a", "b"]);
    expect(m.x).toBe(1);
    expect(m.y).toBe(0);
  });

  it("positional 5-D fallback", () => {
    const m = buildAxisMap(["a", "b", "c_unk", "d", "e"]);
    // last→x, second-last→y, third-last→z, fourth-last→c, fifth-last→t
    expect(m.x).toBe(4);
    expect(m.y).toBe(3);
    expect(m.z).toBe(2);
    // c conflicts with CHANNEL set — only if label "c_unk" is NOT in CHANNEL set
    // "c_unk" is not in CHANNEL set so positional fallback applies
    expect(m.c).toBe(1);
    expect(m.t).toBe(0);
  });

  it("is case-insensitive", () => {
    const m = buildAxisMap(["T", "Z", "C", "Y", "X"]);
    expect(m).toEqual<AxisMap>({ t: 0, z: 1, c: 2, y: 3, x: 4 });
  });

  it("trims whitespace", () => {
    const m = buildAxisMap([" y ", " x "]);
    expect(m.y).toBe(0);
    expect(m.x).toBe(1);
  });

  it("handles empty dim_labels", () => {
    const m = buildAxisMap([]);
    expect(m).toEqual<AxisMap>({ t: null, z: null, c: null, y: null, x: null });
  });
});

// ---------------------------------------------------------------------------
// isAxisMapAmbiguous
// ---------------------------------------------------------------------------

describe("isAxisMapAmbiguous", () => {
  it("returns false for fully-known labels", () => {
    expect(isAxisMapAmbiguous(["t", "z", "c", "y", "x"])).toBe(false);
  });

  it("returns true when any label is unknown", () => {
    expect(isAxisMapAmbiguous(["y", "x", "lambda"])).toBe(true);
  });

  it("returns false for alias labels", () => {
    expect(isAxisMapAmbiguous(["time", "depth", "channel", "height", "width"])).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// computeScaleHint
// ---------------------------------------------------------------------------

describe("computeScaleHint", () => {
  const axisMap: AxisMap = { t: null, z: null, c: null, y: 0, x: 1 };

  it("returns [1,1] when data fits in viewport", () => {
    // 256×256 data, 1024×1024 viewport → scale < 1 → clamped to 1
    const { factors } = computeScaleHint([256, 256], axisMap, 1024, 1024);
    expect(factors).toEqual([1, 1]);
  });

  it("returns [2,2] for 2× zoom-out (2048 data, 1024 viewport)", () => {
    const { factors } = computeScaleHint([2048, 2048], axisMap, 1024, 1024, 10_000_000);
    // rawScale = 2048/1024 = 2 → snap to power-of-two 2
    expect(factors[0]).toBe(2);
    expect(factors[1]).toBe(2);
  });

  it("returns [4,4] for 4× zoom-out (4096 data, 1024 viewport)", () => {
    const { factors } = computeScaleHint([4096, 4096], axisMap, 1024, 1024, 10_000_000);
    expect(factors[0]).toBe(4);
    expect(factors[1]).toBe(4);
  });

  it("pixel_budget tightens the scale up", () => {
    // 4096×4096 = 16M pixels, budget 100 → factor ≥ sqrt(16M/100) ≈ 400 → snap to 512
    const { factors } = computeScaleHint([4096, 4096], axisMap, 4096, 4096, 100);
    expect(factors[0]).toBeGreaterThan(1);
  });

  it("returns all-1 when y or x axis is null", () => {
    const noYX: AxisMap = { t: null, z: null, c: null, y: null, x: null };
    const { factors } = computeScaleHint([4096, 4096], noYX, 512, 512);
    expect(factors).toEqual([1, 1]);
  });

  it("preserves non-spatial axes at 1 for 4-D tensor", () => {
    const zyx: AxisMap = { t: null, z: 0, c: null, y: 1, x: 2 };
    const { factors } = computeScaleHint([10, 4096, 4096], zyx, 1024, 1024, 10_000_000);
    expect(factors[0]).toBe(1); // z axis untouched
    expect(factors[1]).toBe(factors[2]); // y and x should match
  });

  it("hysteresis: no change when at prev factor within 20%", () => {
    // Scale 2, prev was 2 → should stay at 2
    const { factors: f1 } = computeScaleHint([2048, 2048], axisMap, 1024, 1024, 10_000_000, [2, 2]);
    expect(f1).toEqual([2, 2]);
  });
});

// ---------------------------------------------------------------------------
// TensorArray.compute() — slice range assembly
// ---------------------------------------------------------------------------

describe("TensorArray.compute", () => {
  const desc = makeDesc(["z", "y", "x"], [10, 128, 256]);

  it("sends full extent when no options provided", async () => {
    const client = makeClient();
    const ta = new TensorArray(client, "src0", desc);
    await ta.compute();
    const callArg = (client.slice as Mock).mock.calls[0]![0];
    expect(callArg.slice_start).toEqual([0, 0, 0]);
    expect(callArg.slice_stop).toEqual([10, 128, 256]);
  });

  it("clamps out-of-range slice stops to shape", async () => {
    const client = makeClient();
    const ta = new TensorArray(client, "src0", desc);
    await ta.compute({ z: [0, 999], y: [0, 999], x: [0, 999] });
    const callArg = (client.slice as Mock).mock.calls[0]![0];
    expect(callArg.slice_stop).toEqual([10, 128, 256]);
  });

  it("sends scalar z as single-index range [z, z+1]", async () => {
    const client = makeClient();
    const ta = new TensorArray(client, "src0", desc);
    await ta.compute({ z: 3 });
    const callArg = (client.slice as Mock).mock.calls[0]![0];
    expect(callArg.slice_start![0]).toBe(3);
    expect(callArg.slice_stop![0]).toBe(4);
  });

  it("includes scale_hint and reduction_method in request", async () => {
    const client = makeClient();
    const ta = new TensorArray(client, "src0", desc);
    await ta.compute({ scaleHint: [1, 2, 2], reductionMethod: "area" });
    const callArg = (client.slice as Mock).mock.calls[0]![0];
    expect(callArg.scale_hint).toEqual([1, 2, 2]);
    expect(callArg.reduction_method).toBe("area");
  });

  it("sets correct source_id and tensor_id", async () => {
    const client = makeClient();
    const ta = new TensorArray(client, "my-source", desc);
    await ta.compute();
    const callArg = (client.slice as Mock).mock.calls[0]![0];
    expect(callArg.source_id).toBe("my-source");
    expect(callArg.tensor_id).toBe("t0");
  });

  it("exposes ndim, shape, dtype from descriptor", () => {
    const client = makeClient();
    const ta = new TensorArray(client, "src0", desc);
    expect(ta.ndim).toBe(3);
    expect(ta.shape).toEqual([10, 128, 256]);
    expect(ta.dtype).toBe("uint16");
  });

  it("propagates client errors", async () => {
    const client = makeClient();
    (client.slice as Mock).mockRejectedValueOnce(new Error("network error"));
    const ta = new TensorArray(client, "src0", desc);
    await expect(ta.compute()).rejects.toThrow("network error");
  });
});
