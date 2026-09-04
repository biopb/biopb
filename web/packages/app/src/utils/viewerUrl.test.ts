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

const enc = (state: Partial<ViewerUrlState>, qs = "") =>
  encodeViewerState(
    new URLSearchParams(qs),
    { ...defaults(TENSOR.array_id), ...state },
    defaults(TENSOR.array_id),
  ).toString();

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

  it("reads the retired mm flag as the window it named", () => {
    // Links in the wild still carry it; a scale of 0 is the same 0-100 window.
    expect(decode("mm=1").slice).toMatchObject({ contrastMode: "auto", percentileScale: 0 });
    expect(decode("mm=1&p=2").slice).toMatchObject({ percentileScale: 0 });
    expect(decode("p=2").slice).toMatchObject({ contrastMode: "auto", percentileScale: 2 });
  });

  it("reads a fixed window, and ignores one that names no window", () => {
    expect(decode("cl=100,4000").slice).toMatchObject({
      contrastMode: "fixed",
      fixedLimits: [100, 4000],
    });
    expect(decode("cl=4000,100").slice).toMatchObject({ contrastMode: "auto", fixedLimits: null });
    expect(decode("cl=abc").slice).toMatchObject({ contrastMode: "auto" });
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

  it("writes the fixed window instead of the percentile it is not using", () => {
    const out = enc({
      slice: {
        ...DEFAULT_VIEWER_URL_STATE.slice,
        contrastMode: "fixed",
        fixedLimits: [100, 4000],
        percentileScale: 2,
      },
    });
    expect(out).toContain("cl=100%2C4000");
    expect(out).not.toContain("p=");
  });

  it("rounds a scrubbed gamma rather than writing its full float", () => {
    const out = enc({ slice: { ...DEFAULT_VIEWER_URL_STATE.slice, gamma: 2 ** (1 / 3) } });
    expect(out).toContain("g=1.26");
  });

  it("round-trips a state through decode", () => {
    const state: ViewerUrlState = {
      arrayId: TENSOR.array_id,
      slice: {
        t: 5,
        z: 12,
        c: 2,
        axes: {},
        contrastMode: "auto",
        percentileScale: 2.5,
        fixedLimits: null,
        gamma: 1.5,
      },
      render3d: true,
      volumeRenderMode: "minip",
      camera3d: { target: [12.5, 30, 7.2], zoom: -2.125, rotationX: 20, rotationOrbit: -45 },
      camera2d: null,
    };
    const qs = encodeViewerState(new URLSearchParams(), state, defaults(TENSOR.array_id));
    expect(decodeViewerState(qs, defaults(TENSOR.array_id))).toEqual(state);
  });
});

describe("the 3-D camera", () => {
  it("is absent until the view is orbited", () => {
    expect(decode("t=1").camera3d).toBeNull();
    expect(enc({ render3d: true })).not.toContain("tg=");
  });

  it("needs both a target and a zoom to be a camera", () => {
    // Half a camera would frame the volume somewhere nobody chose.
    expect(decode("tg=1,2,3").camera3d).toBeNull();
    expect(decode("zm=-2").camera3d).toBeNull();
    expect(decode("tg=1,2&zm=-2").camera3d).toBeNull();
    expect(decode("tg=1,2,x&zm=-2").camera3d).toBeNull();
  });

  it("defaults the rotations, since 0 is the fitted orientation", () => {
    expect(decode("tg=1,2,3&zm=-2").camera3d).toEqual({
      target: [1, 2, 3], zoom: -2, rotationX: 0, rotationOrbit: 0,
    });
  });

  it("clamps pitch to what OrbitController allows", () => {
    // Bounded at decode, unlike the indices: these limits belong to the camera,
    // not to the tensor, so there is no grid to wait for.
    expect(decode("tg=0,0,0&zm=0&rx=400").camera3d?.rotationX).toBe(90);
    expect(decode("tg=0,0,0&zm=0&rx=-400").camera3d?.rotationX).toBe(-90);
  });

  it("wraps the orbit angle instead of clamping it", () => {
    // Periodic: 370 degrees is a real bearing, where clamping would mean 180.
    expect(decode("tg=0,0,0&zm=0&ro=370").camera3d?.rotationOrbit).toBe(10);
    expect(decode("tg=0,0,0&zm=0&ro=-190").camera3d?.rotationOrbit).toBe(170);
    expect(decode("tg=0,0,0&zm=0&ro=180").camera3d?.rotationOrbit).toBe(-180);
  });

  it("rounds an orbit rather than writing its full floats", () => {
    const out = enc({
      render3d: true,
      camera3d: { target: [1.23456, 2.5, 3], zoom: 1 / 3, rotationX: 10.987, rotationOrbit: 0 },
    });
    expect(out).toContain("tg=1.2%2C2.5%2C3");
    expect(out).toContain("zm=0.333");
    expect(out).toContain("rx=11");
    // The fitted orientation, so omitted like every other default.
    expect(out).not.toContain("ro=");
  });
});

describe("the 2-D camera", () => {
  it("is told from a 3-D one by the target's arity, not by the mode", () => {
    // Two components is a plane, three is a volume. That is a property of the
    // data, so neither decoder needs to know which viewer is mounted.
    expect(decode("tg=10,20&zm=-1").camera2d).toEqual({ target: [10, 20], zoom: -1 });
    expect(decode("tg=10,20&zm=-1").camera3d).toBeNull();
    expect(decode("tg=1,2,3&zm=-1").camera2d).toBeNull();
    expect(decode("tg=1,2,3&zm=-1").camera3d).not.toBeNull();
  });

  it("needs both a target and a zoom", () => {
    expect(decode("tg=10,20").camera2d).toBeNull();
    expect(decode("zm=-1").camera2d).toBeNull();
    expect(decode("tg=10,x&zm=-1").camera2d).toBeNull();
  });

  it("writes the mounted viewer's camera and only that one", () => {
    const cam2 = { target: [10, 20] as [number, number], zoom: -1 };
    const cam3 = { target: [1, 2, 3] as [number, number, number], zoom: -2, rotationX: 0, rotationOrbit: 0 };

    // Both cameras remembered, 2-D on screen: the link carries the plane.
    const flat = enc({ render3d: false, camera2d: cam2, camera3d: cam3 });
    expect(decodeURIComponent(flat)).toContain("tg=10,20");
    expect(flat).toContain("zm=-1");

    // Same store, 3-D on screen.
    const vol = enc({ render3d: true, camera2d: cam2, camera3d: cam3 });
    expect(decodeURIComponent(vol)).toContain("tg=1,2,3");
    expect(vol).toContain("zm=-2");
  });

  it("round-trips through decode", () => {
    const state: ViewerUrlState = {
      arrayId: TENSOR.array_id,
      slice: DEFAULT_VIEWER_URL_STATE.slice,
      render3d: false,
      volumeRenderMode: DEFAULT_VIEWER_URL_STATE.volumeRenderMode,
      camera3d: null,
      camera2d: { target: [10.5, 20.25], zoom: -1.5 },
    };
    const qs = encodeViewerState(new URLSearchParams(), state, defaults(TENSOR.array_id));
    // The target rounds to a tenth, as the encoder documents.
    expect(decodeViewerState(qs, defaults(TENSOR.array_id)).camera2d).toEqual({
      target: [10.5, 20.3],
      zoom: -1.5,
    });
  });
});
