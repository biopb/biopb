import { describe, expect, it } from "vitest";
import {
  DEFAULT_VIEWER_URL_STATE,
  decodeViewerState,
  encodeViewerState,
  resolveArrayId,
  type ViewerUrlState,
} from "./viewerUrl";
import type { DataSourceDescriptor, TensorDescriptor } from "@biopb/tensor-flight-client";

const TENSOR: TensorDescriptor = {
  array_id: "hpc__ome-tiff_00b764c29c31/Image:0",
  dim_labels: ["t", "c", "z", "y", "x"],
  shape: [10, 3, 40, 512, 512],
  chunk_shape: [],
  dtype: "uint16",
};

/** Two axes sharing a label, so the second is unnamed and keyed `a1`. */
const AMBIGUOUS: TensorDescriptor = {
  array_id: "seq_1",
  dim_labels: ["t", "t", "y", "x"],
  shape: [4, 7, 256, 256],
  chunk_shape: [],
  dtype: "uint8",
};

const SOURCE: DataSourceDescriptor = {
  source_id: "hpc__ome-tiff_00b764c29c31",
  source_url: "file:///data/a.ome.tiff",
  source_type: "ome-tiff",
  metadata_json: null,
  tensors: [TENSOR],
};

const defaults = (arrayId: string): ViewerUrlState => ({ arrayId, ...DEFAULT_VIEWER_URL_STATE });

const decode = (qs: string, t = TENSOR) =>
  decodeViewerState(new URLSearchParams(qs), t, defaults(t.array_id));

describe("decodeViewerState", () => {
  it("reads indices, contrast and render mode", () => {
    const s = decode("t=5&z=12&c=1&p=2.5&g=1.5&v=1&vm=additive");
    expect(s.slice).toMatchObject({ t: 5, z: 12, c: 1, percentileScale: 2.5, gamma: 1.5 });
    expect(s.render3d).toBe(true);
    expect(s.volumeRenderMode).toBe("additive");
  });

  it("falls back to defaults for absent parameters", () => {
    expect(decode("").slice).toEqual(DEFAULT_VIEWER_URL_STATE.slice);
    // Only what the link names is adopted; the rest stays at the default.
    expect(decode("t=5").slice).toEqual({ ...DEFAULT_VIEWER_URL_STATE.slice, t: 5 });
  });

  it("clamps an index past the tensor's extent", () => {
    // The whole point of requiring a descriptor: setSlice has no bound of its own.
    expect(decode("z=400").slice.z).toBe(39);
    expect(decode("t=-3").slice.t).toBe(0);
  });

  it("rounds a fractional index rather than passing it to the fetch", () => {
    expect(decode("z=7.6").slice.z).toBe(8);
  });

  it("ignores values that are not numbers", () => {
    expect(decode("z=abc&g=NaN").slice).toMatchObject({ z: 0, gamma: 1 });
  });

  it("keeps an unnamed axis under its sliderAxes key", () => {
    expect(decode("a1=5", AMBIGUOUS).slice.axes).toEqual({ a1: 5 });
  });

  it("drops an axis key the tensor in view does not have", () => {
    // Left over from the previously viewed source; carrying it would put a
    // phantom entry in the selection.
    expect(decode("a7=2").slice.axes).toEqual({});
  });

  it("clamps an unnamed axis to its own extent", () => {
    expect(decode("a1=99", AMBIGUOUS).slice.axes).toEqual({ a1: 6 });
  });

  it("lets mm win over a stale percentile, and p clear the flag", () => {
    expect(decode("mm=1").slice.useMinMax).toBe(true);
    expect(decode("mm=1&p=2").slice).toMatchObject({ useMinMax: true });
    expect(decode("p=2").slice).toMatchObject({ useMinMax: false, percentileScale: 2 });
  });

  it("clamps gamma to the slider's track", () => {
    expect(decode("g=99").slice.gamma).toBe(4);
    expect(decode("g=0").slice.gamma).toBe(0.25);
  });

  it("ignores an unknown render mode", () => {
    expect(decode("vm=bogus").volumeRenderMode).toBe("mip");
  });

  it("keeps one bad parameter from discarding the rest of the link", () => {
    expect(decode("z=9999&t=3&g=1.5").slice).toMatchObject({ t: 3, z: 39, gamma: 1.5 });
  });
});

describe("encodeViewerState", () => {
  const enc = (state: Partial<ViewerUrlState>, qs = "") =>
    encodeViewerState(
      new URLSearchParams(qs),
      { ...defaults(TENSOR.array_id), ...state },
      defaults(TENSOR.array_id),
    ).toString();

  it("writes only the id when everything is at its default", () => {
    expect(decodeURIComponent(enc({}))).toBe("id=hpc__ome-tiff_00b764c29c31/Image:0");
  });

  it("omits defaults and keeps what was set", () => {
    const out = enc({ slice: { ...DEFAULT_VIEWER_URL_STATE.slice, t: 5, gamma: 1.5 } });
    expect(out).toContain("t=5");
    expect(out).toContain("g=1.5");
    expect(out).not.toContain("z=");
    expect(out).not.toContain("c=");
  });

  it("preserves parameters it does not own", () => {
    expect(enc({}, "next=%2Fadmin")).toContain("next=%2Fadmin");
  });

  it("drops a stale axis key belonging to the previous source", () => {
    expect(enc({}, "a7=3")).not.toContain("a7");
  });

  it("writes mm instead of p when min/max is on", () => {
    const out = enc({ slice: { ...DEFAULT_VIEWER_URL_STATE.slice, useMinMax: true, percentileScale: 0 } });
    expect(out).toContain("mm=1");
    expect(out).not.toContain("p=");
  });

  it("rounds a scrubbed gamma rather than writing its full float", () => {
    const out = enc({ slice: { ...DEFAULT_VIEWER_URL_STATE.slice, gamma: 2 ** (1 / 3) } });
    expect(out).toContain("g=1.26");
  });

  it("round-trips a state through decode", () => {
    const state: ViewerUrlState = {
      arrayId: TENSOR.array_id,
      slice: { t: 5, z: 12, c: 2, axes: {}, percentileScale: 2.5, useMinMax: false, gamma: 1.5 },
      render3d: true,
      volumeRenderMode: "minip",
    };
    const qs = encodeViewerState(new URLSearchParams(), state, defaults(TENSOR.array_id));
    expect(decodeViewerState(qs, TENSOR, defaults(TENSOR.array_id))).toEqual(state);
  });
});

describe("resolveArrayId", () => {
  it("finds the tensor and its owning source", () => {
    expect(resolveArrayId([SOURCE], TENSOR.array_id)?.source.source_id).toBe(SOURCE.source_id);
  });

  it("returns null for an id the catalog no longer holds", () => {
    // A link shared before a re-index changed the content hash.
    expect(resolveArrayId([SOURCE], "hpc__ome-tiff_deadbeef/Image:0")).toBeNull();
  });

  it("does not confuse a source_id with a tensor address", () => {
    expect(resolveArrayId([SOURCE], SOURCE.source_id)).toBeNull();
  });
});
