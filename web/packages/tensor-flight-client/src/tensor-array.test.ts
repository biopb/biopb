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
  sliderAxes,
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
    expect(m).toEqual<AxisMap>({ t: 0, z: 1, c: 2, y: 3, x: 4, s: null });
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
    expect(m).toEqual<AxisMap>({ t: 0, z: 1, c: 2, y: 3, x: 4, s: null });
  });

  it("does not duplicate z when labels are channel-first c,y,x", () => {
    const m = buildAxisMap(["c", "y", "x"]);
    expect(m).toEqual<AxisMap>({ t: null, z: null, c: 0, y: 1, x: 2, s: null });
  });

  it("does not duplicate c when labels are z,y,x", () => {
    const m = buildAxisMap(["z", "y", "x"]);
    expect(m).toEqual<AxisMap>({ t: null, z: 0, c: null, y: 1, x: 2, s: null });
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
    expect(m).toEqual<AxisMap>({ t: 0, z: 1, c: 2, y: 3, x: 4, s: null });
  });

  it("trims whitespace", () => {
    const m = buildAxisMap([" y ", " x "]);
    expect(m.y).toBe(0);
    expect(m.x).toBe(1);
  });

  it("handles empty dim_labels", () => {
    const m = buildAxisMap([]);
    expect(m).toEqual<AxisMap>({ t: null, z: null, c: null, y: null, x: null, s: null });
  });
});

// ---------------------------------------------------------------------------
// buildAxisMap — the canonical wire order (biopb/biopb#596)
// ---------------------------------------------------------------------------

describe("buildAxisMap: interleaved samples", () => {
  it("keeps the plane ahead of a trailing samples axis", () => {
    const m = buildAxisMap(["t", "c", "z", "y", "x", "s"]);
    expect(m).toEqual<AxisMap>({ t: 0, c: 1, z: 2, y: 3, x: 4, s: 5 });
  });

  it("does not let z swallow the samples axis of a bare RGB source", () => {
    // Regression: the old whole-array fallback assigned the unknown trailing
    // "s" to z, so the UI showed a Z slider over the colour components and a
    // z-crop sliced into them.
    const m = buildAxisMap(["y", "x", "s"]);
    expect(m.s).toBe(2);
    expect(m.z).toBeNull();
    expect(m).toEqual<AxisMap>({ t: null, c: null, z: null, y: 0, x: 1, s: 2 });
  });

  it("treats a samples label as known, so RGB is not flagged ambiguous", () => {
    // Every interleaved-RGB source used to trip the ambiguity warning.
    expect(isAxisMapAmbiguous(["T", "C", "Z", "Y", "X", "S"])).toBe(false);
    expect(isAxisMapAmbiguous(["y", "x", "samples"])).toBe(false);
  });

  it("needs room for y and x before honouring samples", () => {
    const m = buildAxisMap(["y", "s"]);
    expect(m.s).toBeNull();
    expect([m.y, m.x]).toEqual([0, 1]);
  });

  it("ignores a samples label the canonical order would not put last", () => {
    // Not an order the server serves; it reduces to a leading axis like any
    // other rather than being hunted down.
    const m = buildAxisMap(["s", "y", "x"]);
    expect(m.s).toBeNull();
    expect([m.y, m.x]).toEqual([1, 2]);
  });
});

describe("buildAxisMap: the plane is a position", () => {
  it("reads y/x off the tail even when the labels are unknown", () => {
    const m = buildAxisMap(["q", "w", "e"]);
    expect([m.y, m.x]).toEqual([1, 2]);
  });

  it("does not let a label on the plane become a slider axis", () => {
    // A source labelling its last two axes z/y does not thereby get a z slider
    // sitting on the axis being displayed.
    const m = buildAxisMap(["c", "z", "y"]);
    expect([m.y, m.x]).toEqual([1, 2]);
    expect(m.z).toBeNull();
    expect(m.c).toBe(0);
  });

  it("agrees with the server's plane_axes on the 6-D RGB layout", () => {
    // The coupling that matters: this side picks the crop and scale_hint, the
    // sidecar renders the block they describe. Both must call axis 3 Y.
    const m = buildAxisMap(["T", "C", "Z", "Y", "X", "S"]);
    expect([m.y, m.x, m.s]).toEqual([3, 4, 5]);
  });
});

// ---------------------------------------------------------------------------
// sliderAxes — navigation without renaming
// ---------------------------------------------------------------------------

describe("sliderAxes", () => {
  it("names the axes the labels name", () => {
    const axes = sliderAxes(["t", "c", "z", "y", "x"], [4, 3, 16, 512, 512]);
    expect(axes.map((a) => [a.axis, a.named, a.title, a.key])).toEqual([
      [0, "t", "T", "t"],
      [1, "c", "C", "c"],
      [2, "z", "Z", "z"],
    ]);
  });

  it("excludes the plane", () => {
    expect(sliderAxes(["y", "x"], [512, 512])).toEqual([]);
    expect(sliderAxes(["y", "x", "s"], [512, 512, 3])).toEqual([]);
  });

  it("does NOT rename an unnamed axis to z", () => {
    // buildAxisMap's positional fallback would call this one `z`. A TIFF
    // sequence's 155 stacked files are not depth planes, and nothing in the
    // source says they are.
    expect(buildAxisMap(["i", "y", "x"]).z).toBe(0);
    const axes = sliderAxes(["i", "y", "x"], [155, 1024, 1344]);
    expect(axes).toEqual([
      { axis: 0, named: null, title: "i", key: "a0", extent: 155 },
    ]);
  });

  it("still makes an unlabelled store navigable", () => {
    // What the fallback existed to provide, kept: every axis gets a control.
    // Only the invented semantics are gone.
    const axes = sliderAxes(["dim0", "dim1", "dim2", "dim3"], [4, 5, 512, 512]);
    expect(axes.map((a) => a.title)).toEqual(["dim0", "dim1"]);
    expect(axes.map((a) => a.key)).toEqual(["a0", "a1"]);
    expect(axes.every((a) => a.named === null)).toBe(true);
  });

  it("gives the second of two axes sharing a label a key of its own", () => {
    // Only the first `c` has a name to be addressed by; the second would
    // otherwise collapse into the same selection entry.
    const axes = sliderAxes(["c", "c", "y", "x"], [2, 3, 512, 512]);
    expect(axes.map((a) => [a.named, a.key])).toEqual([
      ["c", "c"],
      [null, "a1"],
    ]);
  });

  it("titles an empty label by position", () => {
    const axes = sliderAxes(["", "y", "x"], [7, 512, 512]);
    expect(axes[0]).toMatchObject({ title: "axis 0", key: "a0" });
  });

  it("keeps extent-1 axes, which are still part of a selection", () => {
    const axes = sliderAxes(["t", "y", "x"], [1, 512, 512]);
    expect(axes).toEqual([{ axis: 0, named: "t", title: "T", key: "t", extent: 1 }]);
  });

  it("gives every axis a unique key", () => {
    const labels = ["c", "c", "", "pos", "z", "y", "x"];
    const axes = sliderAxes(labels, [2, 3, 4, 5, 6, 512, 512]);
    expect(new Set(axes.map((a) => a.key)).size).toBe(axes.length);
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
  const axisMap: AxisMap = { t: null, z: null, c: null, y: 0, x: 1, s: null };

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
    const noYX: AxisMap = { t: null, z: null, c: null, y: null, x: null, s: null };
    const { factors } = computeScaleHint([4096, 4096], noYX, 512, 512);
    expect(factors).toEqual([1, 1]);
  });

  it("preserves non-spatial axes at 1 for 4-D tensor", () => {
    const zyx: AxisMap = { t: null, z: 0, c: null, y: 1, x: 2, s: null };
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
