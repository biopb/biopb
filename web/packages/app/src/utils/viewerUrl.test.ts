import { describe, expect, it } from "vitest";
import {
  DEFAULT_VIEWER_URL_STATE,
  decodeViewerState,
  encodeViewerState,
  type ViewerUrlState,
} from "./viewerUrl";
import type { TensorDescriptor } from "@biopb/tensor-flight-client";

const TENSOR: TensorDescriptor = {
  array_id: "hpc__ome-tiff_00b764c29c31/Image:0",
  dim_labels: ["t", "c", "z", "y", "x"],
  shape: [10, 3, 40, 512, 512],
  chunk_shape: [],
  dtype: "uint16",
};


const defaults = (arrayId: string): ViewerUrlState => ({ arrayId, ...DEFAULT_VIEWER_URL_STATE });

const decode = (qs: string, t = TENSOR) =>
  decodeViewerState(new URLSearchParams(qs), defaults(t.array_id));

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

  it("rounds a fractional index rather than passing it to the fetch", () => {
    expect(decode("z=7.6").slice.z).toBe(8);
  });

  it("reads an index without bounding it -- the grid does that later", () => {
    // Decoding cannot know the extent: a pinned link has no descriptor until
    // tile_info answers. clampSliceTo covers this, and is tested there.
    expect(decode("z=400").slice.z).toBe(400);
    expect(decode("t=-3").slice.t).toBe(0);
    expect(decode("a7=2").slice.axes).toEqual({ a7: 2 });
  });

  it("keeps a bad parameter from discarding the rest of the link", () => {
    expect(decode("z=abc&t=3&g=1.5").slice).toMatchObject({ t: 3, z: 0, gamma: 1.5 });
  });

  it("takes the id from the link, pinned or not", () => {
    expect(decode("id=src_a@9f1c4e2b/Image:0").arrayId).toBe("src_a@9f1c4e2b/Image:0");
  });

  it("ignores values that are not numbers", () => {
    expect(decode("z=abc&g=NaN").slice).toMatchObject({ z: 0, gamma: 1 });
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
    expect(decodeViewerState(qs, defaults(TENSOR.array_id))).toEqual(state);
  });
});
