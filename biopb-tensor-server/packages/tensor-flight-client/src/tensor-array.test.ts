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

  it("returns [1,1] with default zoom (viewportZoom=1)", () => {
    // effectiveTargetScale = max(1/1, 1) = 1
    // maxScale varies by tensor size, but min(maxScale, 1) = 1
    const { factors } = computeScaleHint([4096, 4096], axisMap, 1024, 1024);
    expect(factors).toEqual([1, 1]);
  });

  it("pixel_budget determines maxScale (upper bound on scale)", () => {
    // 4096×4096 = 16M pixels, budget 100 → maxScale = sqrt(16M/100) ≈ 400
    // effectiveTargetScale = 1 (default zoom)
    // min(400, 1) = 1, snapped to 1
    const { factors } = computeScaleHint([4096, 4096], axisMap, 1024, 1024, 100);
    expect(factors[0]).toBe(1);
  });

  it("zoomed-out (viewportZoom<1) increases scale", () => {
    // viewportZoom=0.5 → effectiveTargetScale = max(1/0.5, 1) = 2
    // maxScale = 4 (for 4096×4096 with 1M budget)
    // min(4, 2) = 2
    const { factors } = computeScaleHint([4096, 4096], axisMap, 1024, 1024, 1_000_000, 0.5);
    expect(factors).toEqual([2, 2]);
  });

  it("zoomed-in (viewportZoom>1) capped at scale=1", () => {
    // viewportZoom=2 → effectiveTargetScale = max(1/2, 1) = 1
    // min(maxScale, 1) = 1 always
    const { factors } = computeScaleHint([4096, 4096], axisMap, 1024, 1024, 1_000_000, 2);
    expect(factors).toEqual([1, 1]);
  });

  it("returns all-1 when y or x axis is null", () => {
    const noYX: AxisMap = { t: null, z: null, c: null, y: null, x: null };
    const { factors } = computeScaleHint([4096, 4096], noYX, 512, 512);
    expect(factors).toEqual([1, 1]);
  });

  it("preserves non-spatial axes at 1 for 4-D tensor", () => {
    const zyx: AxisMap = { t: null, z: 0, c: null, y: 1, x: 2 };
    const { factors } = computeScaleHint([10, 4096, 4096], zyx, 1024, 1024);
    expect(factors[0]).toBe(1); // z axis untouched
    expect(factors[1]).toBe(factors[2]); // y and x should match
    expect(factors[1]).toBe(1); // default zoom gives scale=1
  });

  it("very zoomed-out can reach maxScale", () => {
    // viewportZoom=0.25 → effectiveTargetScale = max(1/0.25, 1) = 4
    // maxScale = 4 (for 4096×4096 with 1M budget)
    // min(4, 4) = 4
    const { factors } = computeScaleHint([4096, 4096], axisMap, 1024, 1024, 1_000_000, 0.25);
    expect(factors).toEqual([4, 4]);
  });

  it("zoomed-out beyond maxScale clamps to maxScale", () => {
    // viewportZoom=0.1 → effectiveTargetScale = max(1/0.1, 1) = 10
    // maxScale = 4 (for 4096×4096 with 1M budget)
    // min(4, 10) = 4
    const { factors } = computeScaleHint([4096, 4096], axisMap, 1024, 1024, 1_000_000, 0.1);
    expect(factors).toEqual([4, 4]);
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
