import { afterEach, describe, expect, it, vi } from "vitest";
import { TensorApiError, TensorNetworkError } from "@biopb/tensor-flight-client";
import type {
  DataSourceDescriptor,
  RoiAnnotation,
  RoiSetInfo,
  SourceJobStatus,
  TensorFlightClient,
  TileInfo,
} from "@biopb/tensor-flight-client";
import {
  selectRoiScopes,
  selectRoiSets,
  selectRois,
  selectRoisError,
  selectRoisPending,
  selectVisibleSets,
  selectDraft,
  selectSelectedRoi,
  selectContrastTrack,
  selectContrastWindow,
  selectPlaneLimits,
  selectObservedLimits,
  selectTileInfo,
  selectLabelOverlay,
  selectUrlLabelOverlay,
  selectUrlVisibleSets,
  catalogFingerprint,
  useAppStore,
  EMPTY_VIEW,
} from "./store";
import type { TensorView, ViewTarget } from "./store";
import { DEFAULT_LABEL_OPACITY } from "./utils/vivUtils";

const SOURCE: DataSourceDescriptor = {
  source_id: "listed",
  source_url: "file:///listed",
  source_type: "file",
  metadata_json: null,
  is_resolved: true,
  tensors: [],
};

const BASE_POSITION = { t: 0, z: 0, c: 0, axes: {} };

const BASE_DISPLAY = {
  contrastMode: "auto" as const,
  percentileScale: 1,
  fixedLimits: null,
  gamma: 1,
};

/** Only its identity is under test here; the extents just have to be readable. */
const TILE_INFO = {
  array_id: "first@abcd1234",
  dtype: "<f4",
  dim_labels: ["T", "C", "Z", "Y", "X"],
  shape: [1, 1, 1, 8, 8],
} as unknown as TileInfo;

/** A `target` as `openTensor` leaves it: ready when given a key, idle otherwise. */
function targetFor(key: string | null, over: Partial<ViewTarget> = {}): ViewTarget {
  return {
    epoch: 100,
    requested: key,
    linked: false,
    status: key ? "ready" : "idle",
    key,
    info: key ? ({ ...TILE_INFO, array_id: key } as TileInfo) : null,
    retrying: false,
    error: null,
    seedSets: null,
    ...over,
  };
}

/** The selection fields for `key` in view, resolved. Spread into a `setState`. */
function viewOf(key: string, over: Partial<ViewTarget> = {}) {
  return {
    activeSourceId: key.split("/", 1)[0] ?? key,
    activeTensorId: key,
    target: targetFor(key, over),
  };
}

/** Put `key` in view, resolved. */
function view(key: string) {
  useAppStore.setState(viewOf(key));
}

/** The record the store keeps for `key`, as it stands. */
const held = (key = "first") => useAppStore.getState().views[key] ?? EMPTY_VIEW;

/** Write into `key`'s record, as a landing or an earlier visit would have. */
function seedView(key: string, patch: Partial<TensorView>) {
  useAppStore.setState((s) => ({
    views: { ...s.views, [key]: { ...(s.views[key] ?? EMPTY_VIEW), ...patch } },
  }));
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

/** A client whose tile_info answers every address with itself, minus a pin. */
const echoClient = () =>
  ({
    http: {
      tileInfo: (id: string) =>
        Promise.resolve({ ...TILE_INFO, array_id: id.replace(/@[^/]*/, "") } as TileInfo),
    },
  }) as unknown as TensorFlightClient;

/** Open a link and wait for its target to resolve. */
async function openLink(query: string) {
  useAppStore.setState({ client: echoClient() });
  useAppStore.getState().applyViewerState(new URLSearchParams(query));
  await settle();
}

const client = (sources: DataSourceDescriptor[]) =>
  ({
    listSources: vi.fn().mockResolvedValue(sources),
    http: { readyz: vi.fn().mockResolvedValue({ backend_health: {} }) },
  }) as unknown as TensorFlightClient;

afterEach(() => {
  useAppStore.setState({ views: {} });
  useAppStore.getState().stopCatalogPolling();
  vi.useRealTimers();
});

describe("catalog polling", () => {
  it("keeps a URL-hydrated source that is absent from the listing", async () => {
    vi.useFakeTimers();
    useAppStore.setState({
      client: client([]),
      connectionState: "connected",
      sources: [SOURCE],
      ...viewOf("shared-source/Image:0", { linked: true }),
    });

    useAppStore.getState().startCatalogPolling();
    await vi.advanceTimersByTimeAsync(60000);

    expect(useAppStore.getState().activeSourceId).toBe("shared-source");
  });

  it("clears a clicked source that is absent from the listing", async () => {
    vi.useFakeTimers();
    useAppStore.setState({
      client: client([]),
      connectionState: "connected",
      sources: [SOURCE],
      ...viewOf(SOURCE.source_id),
    });

    useAppStore.getState().startCatalogPolling();
    await vi.advanceTimersByTimeAsync(60000);

    expect(useAppStore.getState().activeSourceId).toBeNull();
  });

  it("judges the source in view when the listing lands, not when it was asked for", async () => {
    vi.useFakeTimers();
    const other = { ...SOURCE, source_id: "other-source" };
    let land: (sources: DataSourceDescriptor[]) => void = () => {};
    const slow = {
      listSources: vi.fn().mockReturnValue(new Promise((resolve) => (land = resolve))),
      http: { readyz: vi.fn().mockResolvedValue({ backend_health: {} }) },
    } as unknown as TensorFlightClient;
    useAppStore.setState({
      client: slow,
      connectionState: "connected",
      sources: [SOURCE],
      ...viewOf(SOURCE.source_id),
    });

    useAppStore.getState().startCatalogPolling();
    await vi.advanceTimersByTimeAsync(60000);
    // The user opens another source while the listing is in flight.
    useAppStore.setState(viewOf(other.source_id));
    land([other]);
    await vi.advanceTimersByTimeAsync(0);

    expect(useAppStore.getState().activeSourceId).toBe("other-source");
  });
});

describe("viewer URL state", () => {
  it("clears cameras when a URL names none", () => {
    useAppStore.setState({
      activeTensorId: "first",
      camera3d: { target: [1, 2, 3], zoom: 1, rotationX: 10, rotationOrbit: 20 },
      camera2d: { target: [4, 5], zoom: 2 },
    });

    expect(useAppStore.getState().applyViewerState(new URLSearchParams("id=second"))).toBe(true);
    expect(useAppStore.getState().camera3d).toBeNull();
    expect(useAppStore.getState().camera2d).toBeNull();
  });

  // The id is the same tensor by every spelling, and it still does not carry a
  // camera over: only the link decides. Written down because the tempting
  // "unless it is the same tensor" exemption cannot be computed here -- see
  // applyViewerState.
  it("clears cameras even when the URL names the tensor already in view", () => {
    useAppStore.setState({
      activeTensorId: "same",
      camera3d: { target: [1, 2, 3], zoom: 1, rotationX: 10, rotationOrbit: 20 },
      camera2d: { target: [4, 5], zoom: 2 },
    });

    expect(useAppStore.getState().applyViewerState(new URLSearchParams("id=same&z=2"))).toBe(true);
    expect(useAppStore.getState().camera3d).toBeNull();
    expect(useAppStore.getState().camera2d).toBeNull();
  });

  it("takes the cameras the URL does name", () => {
    useAppStore.setState({ activeTensorId: "first", camera3d: null, camera2d: null });

    expect(useAppStore.getState().applyViewerState(new URLSearchParams("id=second&tg=1,2,3&zm=4&ro=30"))).toBe(true);
    expect(useAppStore.getState().camera3d).toEqual({
      target: [1, 2, 3],
      zoom: 4,
      rotationX: 0,
      rotationOrbit: 30,
    });
    expect(useAppStore.getState().camera2d).toBeNull();
  });

  it("clears indices and the render mode, keeping viewer preferences", () => {
    useAppStore.setState({
      activeTensorId: "first",
      render3d: true,
      position: { ...BASE_POSITION, t: 5, z: 6, c: 1, axes: { a3: 2 } },
      display: { ...BASE_DISPLAY, percentileScale: 2, gamma: 1.6 },
    });

    expect(useAppStore.getState().applyViewerState(new URLSearchParams("id=second"))).toBe(true);
    const s = useAppStore.getState();
    expect(s.render3d).toBe(false);
    expect(s.position).toMatchObject({ t: 0, z: 0, c: 0, axes: {} });
    // Preferences, not properties of the tensor -- openTensor carries these
    // across a change too.
    expect(s.display).toMatchObject({ percentileScale: 2, gamma: 1.6 });
  });

  it("keeps what the link names", () => {
    useAppStore.setState({ activeTensorId: "first", render3d: false, position: { ...BASE_POSITION, t: 5 } });

    expect(useAppStore.getState().applyViewerState(new URLSearchParams("id=second&t=9&v=1"))).toBe(true);
    expect(useAppStore.getState().position.t).toBe(9);
    expect(useAppStore.getState().render3d).toBe(true);
  });
});

describe("the levels the data has shown", () => {
  const at = (channel: number) =>
    useAppStore.setState((s) => ({ position: { ...s.position, c: channel } }));
  it("widens to cover every plane sampled, not just the last one", () => {
    view("first");
    useAppStore.getState().noteObservedLimits([12, 4000], 0);
    useAppStore.getState().noteObservedLimits([0, 3], 0);
    at(0);

    // Without the union a fixed window chosen on the bright plane could not be
    // widened again while the dark one is in view.
    expect(selectObservedLimits(useAppStore.getState())).toEqual([0, 4000]);
  });

  it("keeps each channel on its own scale", () => {
    view("first");
    useAppStore.getState().noteObservedLimits([12, 4000], 0);
    useAppStore.getState().noteObservedLimits([0, 1], 1);
    at(1);

    expect(selectObservedLimits(useAppStore.getState())).toEqual([0, 1]);
  });

  it("starts over on another tensor rather than widening across two", () => {
    view("first");
    useAppStore.getState().noteObservedLimits([12, 4000], 0);
    view("second");
    useAppStore.getState().noteObservedLimits([0, 1], 0);
    at(0);

    expect(selectObservedLimits(useAppStore.getState())).toEqual([0, 1]);
  });

  it("hides a union sampled from another tensor", () => {
    view("first");
    useAppStore.getState().noteObservedLimits([12, 4000], 0);
    view("second");
    at(0);

    expect(selectObservedLimits(useAppStore.getState())).toBeNull();
  });

  it("does not write when the plane adds nothing", () => {
    view("first");
    useAppStore.getState().noteObservedLimits([0, 4000], 0);
    const before = held().observedLimits;
    useAppStore.getState().noteObservedLimits([10, 900], 0);

    // The viewers publish this from a memo on every render; a fresh identity
    // each time would loop through the effect that writes it.
    expect(held().observedLimits).toBe(before);
  });
});

/** Put `key` in view with a grid of this dtype. */
function viewWithDtype(key: string, dtype: string) {
  useAppStore.setState({
    ...viewOf(key, { info: { ...TILE_INFO, array_id: key, dtype } as TileInfo }),
    position: BASE_POSITION,
    display: BASE_DISPLAY,
    runtime: { planeReady: false, samples: null },
  });
}

/** Sorted samples, as `contrastSamples` returns them. */
const samplesOf = (...values: number[]) => ({
  plane: {},
  values: Float64Array.from(values).sort(),
});

describe("the contrast track and window", () => {
  it("is the dtype's own range for an integer tensor, whatever was observed", () => {
    viewWithDtype("first", "<u2");
    useAppStore.getState().noteObservedLimits([10, 20], 0);

    expect(selectContrastTrack(useAppStore.getState())).toEqual([0, 65535]);
  });

  it("follows the levels the data has shown on a float tensor", () => {
    // Which is why it is derived from everything a plane has shown, not from
    // the plane in view: a window fixed on a bright plane has to stay reachable
    // from a dim one.
    viewWithDtype("first", "<f4");
    useAppStore.getState().notePlaneSamples(samplesOf(12, 4000), 0, 100);
    useAppStore.getState().notePlaneSamples(samplesOf(0, 3), 0, 100);

    expect(selectContrastTrack(useAppStore.getState())).toEqual([0, 4000]);
    expect(selectPlaneLimits(useAppStore.getState())).toEqual([0, 3]);
  });

  it("is the same answer for the shader and the panel, and needs nothing published", () => {
    // biopb/biopb#955: one derivation, so the bar cannot be drawn on a track the
    // shader is not clamping into. A window and its track come from one state.
    viewWithDtype("first", "<f4");
    useAppStore.getState().notePlaneSamples(samplesOf(0, 100, 200, 300, 400), 0, 100);
    useAppStore.setState((s) => ({ display: { ...s.display, contrastMode: "fixed", fixedLimits: [50, 9000] } }));

    const s = useAppStore.getState();
    expect(selectContrastTrack(s)).toEqual([0, 9000]);
    expect(selectContrastWindow(s)).toEqual([50, 9000]);
  });

  it("is null until the grid has landed", () => {
    useAppStore.setState({ target: targetFor(null) });

    expect(selectContrastTrack(useAppStore.getState())).toBeNull();
    expect(selectContrastWindow(useAppStore.getState())).toBeNull();
  });

  it("windows the plane's percentiles in auto mode, the whole track before a plane", () => {
    viewWithDtype("first", "<u2");
    expect(selectContrastWindow(useAppStore.getState())).toEqual([0, 65535]);

    const values = Array.from({ length: 101 }, (_, i) => i * 10);
    useAppStore.getState().notePlaneSamples(samplesOf(...values), 0, 100);
    // percentileScale 1 is the 1st-99th.
    expect(selectContrastWindow(useAppStore.getState())).toEqual([10, 990]);
  });

  it("brings a fixed window inside the track", () => {
    viewWithDtype("first", "<u2");
    useAppStore.setState((s) => ({ display: { ...s.display, contrastMode: "fixed", fixedLimits: [-5, 70000] } }));

    expect(selectContrastWindow(useAppStore.getState())).toEqual([0, 65535]);
  });

  it("drops what a viewer publishes under another epoch", () => {
    // A viewer that outlives its tensor for a commit must not leak into the next.
    viewWithDtype("first", "<f4");
    useAppStore.getState().notePlaneSamples(samplesOf(1, 2), 0, 99);
    useAppStore.getState().setPlaneReady(true, 99);

    const s = useAppStore.getState();
    expect(s.runtime.samples).toBeNull();
    expect(s.runtime.planeReady).toBe(false);
    expect(held().observedLimits).toEqual({});
  });

  it("notes a plane's extremes as levels the tensor has shown, on its channel", () => {
    viewWithDtype("first", "<f4");
    useAppStore.getState().notePlaneSamples(samplesOf(5, 9), 2, 100);

    expect(held().observedLimits[2]).toEqual([5, 9]);
    expect(held().observedLimits[0]).toBeUndefined();
  });

  it("starts a new epoch with no plane and nothing ready", () => {
    viewWithDtype("first", "<f4");
    useAppStore.getState().notePlaneSamples(samplesOf(1, 2), 0, 100);
    useAppStore.getState().setPlaneReady(true, 100);
    useAppStore.setState({ client: null });
    useAppStore.getState().openTensor("second");

    expect(useAppStore.getState().runtime).toEqual({ planeReady: false, samples: null });
  });

  it("un-readies the plane in the same write that moves the position", () => {
    // What play relies on: the next tick must not see the previous frame's
    // readiness, whatever order the viewer's own effect runs in.
    viewWithDtype("first", "<f4");
    useAppStore.getState().setPlaneReady(true, 100);
    useAppStore.getState().setPosition({ z: 3 });
    expect(useAppStore.getState().runtime.planeReady).toBe(false);

    useAppStore.getState().setPlaneReady(true, 100);
    useAppStore.getState().setAxisIndex({ named: "z", key: "z" }, 4);
    expect(useAppStore.getState().runtime.planeReady).toBe(false);

    useAppStore.getState().setPlaneReady(true, 100);
    useAppStore.getState().setPosition({ z: 4 });
    expect(useAppStore.getState().runtime.planeReady).toBe(true);
  });

  it("does not write when the plane is unchanged", () => {
    viewWithDtype("first", "<f4");
    useAppStore.getState().setPlaneReady(true, 100);
    const before = useAppStore.getState().runtime;
    useAppStore.getState().setPlaneReady(true, 100);

    expect(useAppStore.getState().runtime).toBe(before);
  });
});

describe("position and display", () => {
  it("moves one axis by name or into axes, keeping its siblings", () => {
    useAppStore.setState({ position: { ...BASE_POSITION, z: 1, axes: { a0: 4 } } });
    useAppStore.getState().setAxisIndex({ named: "z", key: "z" }, 7);
    useAppStore.getState().setAxisIndex({ named: null, key: "a3" }, 2);

    expect(useAppStore.getState().position).toEqual({ t: 0, z: 7, c: 0, axes: { a0: 4, a3: 2 } });
  });

  it("keeps the position object when an axis is set to where it is", () => {
    useAppStore.setState({ position: { ...BASE_POSITION, axes: { a0: 4 } } });
    const before = useAppStore.getState().position;
    useAppStore.getState().setAxisIndex({ named: null, key: "a0" }, 4);

    expect(useAppStore.getState().position).toBe(before);
  });

  it("keeps the position object when a write changes nothing", () => {
    // What a Viv selection is derived from: a fresh object refetches the plane.
    useAppStore.setState({ position: { ...BASE_POSITION, z: 3, axes: { a0: 1 } } });
    const before = useAppStore.getState().position;
    useAppStore.getState().setPosition({ z: 3 });
    useAppStore.getState().setPosition({ axes: { a0: 1 } });

    expect(useAppStore.getState().position).toBe(before);
  });

  it("leaves the position object alone when the display changes", () => {
    useAppStore.setState({ position: BASE_POSITION });
    const before = useAppStore.getState().position;
    useAppStore.getState().setDisplay({ gamma: 2 });
    useAppStore.getState().setDisplay({ contrastMode: "fixed", fixedLimits: [1, 2] });

    expect(useAppStore.getState().position).toBe(before);
    expect(useAppStore.getState().display).toMatchObject({ gamma: 2, fixedLimits: [1, 2] });
  });

  it("drops a fixed window with the tensor, and keeps the other display settings", () => {
    useAppStore.setState({
      display: { contrastMode: "fixed", percentileScale: 2, fixedLimits: [1, 2], gamma: 1.5 },
      client: null,
    });
    useAppStore.getState().openTensor("other");

    expect(useAppStore.getState().display).toEqual({
      contrastMode: "fixed",
      percentileScale: 2,
      fixedLimits: null,
      gamma: 1.5,
    });
  });
});

// A versioned source answers tile_info for `src/field` with `src@tok/field`
// (biopb/biopb#780), and a bare `src` with the field it binds by default.
// `openTensor` resolves that before any viewer mounts, so everything keyed to
// the tensor is keyed to the resolved `src/field`, whatever the request said.
describe("opening a tensor", () => {
  const VERSIONED = { ...TILE_INFO, array_id: "src@tok/field" } as TileInfo;

  /** A client whose tile_info answers `answer(id)`, recording what it was asked. */
  function resolver(answer: (id: string) => TileInfo | Promise<TileInfo>, asked: string[] = []) {
    return {
      http: {
        tileInfo: (id: string) => {
          asked.push(id);
          try {
            return Promise.resolve(answer(id));
          } catch (err) {
            return Promise.reject(err);
          }
        },
      },
    } as unknown as TensorFlightClient;
  }

  function fresh(client: TensorFlightClient) {
    useAppStore.setState({
      client,
      target: targetFor(null),
      position: BASE_POSITION,
      display: BASE_DISPLAY,
      views: {},
      runtime: { planeReady: false, samples: null },
    });
  }

  it("is resolving, with no grid and no key, until tile_info answers", async () => {
    fresh(resolver(() => VERSIONED));
    useAppStore.getState().openTensor("src/field");

    const before = useAppStore.getState().target;
    expect(before).toMatchObject({ status: "resolving", requested: "src/field", key: null, info: null });
    expect(selectTileInfo(useAppStore.getState())).toBeNull();

    await settle();
    expect(useAppStore.getState().target).toMatchObject({ status: "ready", key: "src/field", info: VERSIONED });
  });

  it("keys on the tensor, not on the version token tile_info adds", async () => {
    fresh(resolver(() => VERSIONED));
    useAppStore.getState().openTensor("src/field");
    await settle();

    expect(useAppStore.getState().target.key).toBe("src/field");
    expect(useAppStore.getState().target.info?.array_id).toBe("src@tok/field");
  });

  it("resolves a bare source_id to the field the server binds", async () => {
    fresh(resolver(() => VERSIONED));
    useAppStore.getState().openTensor("src");
    await settle();

    expect(useAppStore.getState().activeTensorId).toBe("src");
    expect(useAppStore.getState().target.key).toBe("src/field");
  });

  it("asks for the address it was given, pin included", async () => {
    const asked: string[] = [];
    fresh(resolver(() => VERSIONED, asked));
    useAppStore.getState().openTensor("src@pin/field");
    await settle();

    expect(asked).toEqual(["src@pin/field"]);
    expect(useAppStore.getState().target.requested).toBe("src@pin/field");
    expect(useAppStore.getState().activeTensorId).toBe("src/field");
    expect(useAppStore.getState().target.key).toBe("src/field");
  });

  it("shows the contrast state a viewer published once the grid has landed", async () => {
    fresh(resolver(() => VERSIONED));
    useAppStore.getState().openTensor("src/field");
    await settle();
    useAppStore.getState().noteObservedLimits([0, 5000], 0);

    expect(selectObservedLimits(useAppStore.getState())).toEqual([0, 5000]);
    expect(selectContrastTrack(useAppStore.getState())).toEqual([0, 5000]);
  });

  it("keeps an unpinned link's sets once the grid lands", async () => {
    fresh(resolver(() => VERSIONED));
    useAppStore.getState().applyViewerState(new URLSearchParams("id=src/field&rs=nuclei"));
    await settle();

    expect(selectVisibleSets(useAppStore.getState())).toEqual(["nuclei"]);
  });

  it("keeps a pinned link's sets once the grid lands", async () => {
    fresh(resolver(() => ({ ...TILE_INFO, array_id: "src@pin/field" }) as TileInfo));
    useAppStore.getState().applyViewerState(new URLSearchParams("id=src@pin/field&rs=nuclei"));
    await settle();

    expect(selectVisibleSets(useAppStore.getState())).toEqual(["nuclei"]);
  });

  it("writes a bare source_id's sets under the field it resolved to", async () => {
    fresh(resolver(() => VERSIONED));
    useAppStore.getState().applyViewerState(new URLSearchParams("id=src&rs=nuclei"));
    await settle();

    expect(useAppStore.getState().target.key).toBe("src/field");
    expect(selectVisibleSets(useAppStore.getState())).toEqual(["nuclei"]);
    expect(held("src/field").visibleSets).toEqual(["nuclei"]);
  });

  it("does not show a link's sets while the target is still resolving", () => {
    fresh(resolver(() => new Promise<TileInfo>(() => {})));
    useAppStore.getState().applyViewerState(new URLSearchParams("id=src&rs=nuclei"));

    expect(selectVisibleSets(useAppStore.getState())).toBeNull();
  });

  it("bounds the slice by the grid it landed", async () => {
    fresh(resolver(() => VERSIONED));
    useAppStore.getState().applyViewerState(new URLSearchParams("id=src/field&t=9&z=4"));
    await settle();

    // TILE_INFO is 1 x 1 x 1 x 8 x 8.
    expect(useAppStore.getState().position).toMatchObject({ t: 0, z: 0 });
  });

  it("fetches the ROI listing once, under the address tile_info resolved", async () => {
    const asked: string[] = [];
    const client = {
      http: {
        tileInfo: () => Promise.resolve(VERSIONED),
        listRois: (arrayId: string) => {
          asked.push(arrayId);
          return Promise.resolve({ rois: [], sets: [], truncated: false, skipped: 0 });
        },
      },
    } as unknown as TensorFlightClient;
    useAppStore.setState({ roisUnavailable: false });
    fresh(client);
    useAppStore.getState().openTensor("src");
    // TileViewer mounts only once the target is ready, so nothing asks earlier.
    await useAppStore.getState().loadRois();
    expect(asked).toEqual([]);
    await settle();
    await Promise.all([useAppStore.getState().loadRois(), useAppStore.getState().loadRois()]);

    expect(asked).toEqual(["src@tok/field"]);
    expect(selectRoiScopes(useAppStore.getState())).toEqual({ "": { truncated: false, skipped: 0 } });
  });

  it("drops a landing for a tensor the user has already left", async () => {
    let answerFirst: (info: TileInfo) => void = () => {};
    fresh(
      resolver((id) =>
        id === "first" ? new Promise<TileInfo>((r) => { answerFirst = r; }) : ({ ...TILE_INFO, array_id: id }) as TileInfo,
      ),
    );
    useAppStore.getState().openTensor("first");
    useAppStore.getState().openTensor("second");
    await settle();
    answerFirst({ ...TILE_INFO, array_id: "first" } as TileInfo);
    await settle();

    expect(useAppStore.getState().target).toMatchObject({ status: "ready", key: "second" });
  });

  it("does not let a stale landing bound the slice of the tensor now open", async () => {
    let answerFirst: (info: TileInfo) => void = () => {};
    fresh(
      resolver((id) =>
        id === "first" ? new Promise<TileInfo>((r) => { answerFirst = r; }) : new Promise<TileInfo>(() => {}),
      ),
    );
    useAppStore.getState().openTensor("first");
    useAppStore.getState().openTensor("second");
    answerFirst({ ...TILE_INFO, array_id: "first" } as TileInfo);
    await settle();

    expect(useAppStore.getState().target).toMatchObject({ status: "resolving", requested: "second", info: null });
  });

  it("reports a refused tile_info as a settled fact, and lets it be retried", async () => {
    let answer: () => TileInfo = () => {
      throw new TensorApiError(404, "no such tensor");
    };
    fresh(resolver(() => answer()));
    useAppStore.getState().openTensor("gone");
    await settle();

    expect(useAppStore.getState().target).toMatchObject({
      status: "failed",
      error: { kind: "capability" },
    });

    answer = () => ({ ...TILE_INFO, array_id: "gone" }) as TileInfo;
    useAppStore.getState().retryTarget();
    expect(useAppStore.getState().target).toMatchObject({ status: "resolving", error: null });
    await settle();
    expect(useAppStore.getState().target).toMatchObject({ status: "ready", key: "gone" });
  });

  it("re-asks once after a transport failure before giving up", async () => {
    vi.useFakeTimers();
    let calls = 0;
    fresh(
      resolver(() => {
        calls += 1;
        throw new TensorNetworkError("/api/tile_info");
      }),
    );
    useAppStore.getState().openTensor("slow");
    await vi.advanceTimersByTimeAsync(0);
    expect(useAppStore.getState().target).toMatchObject({ status: "resolving", retrying: true });

    await vi.advanceTimersByTimeAsync(600);
    expect(calls).toBe(2);
    expect(useAppStore.getState().target).toMatchObject({
      status: "failed",
      retrying: false,
      error: { kind: "transport" },
    });
  });

  it("clears a link's state when nothing is opened", () => {
    fresh(resolver(() => VERSIONED));
    useAppStore.getState().openTensor("src");
    useAppStore.getState().openTensor(null);

    expect(useAppStore.getState()).toMatchObject({ activeSourceId: null, activeTensorId: null });
    expect(useAppStore.getState().target).toMatchObject({ status: "idle", key: null, info: null });
  });
});

describe("what the URL carries while a link resolves", () => {
  const LINK = "id=src/field&rs=nuclei&rs=cells&lb=src%2Ffield%2F%40labels%2Fmasks";

  it("keeps a link's sets and overlay while tile_info is pending", () => {
    useAppStore.setState({
      client: {
        http: { tileInfo: () => new Promise<TileInfo>(() => {}) },
      } as unknown as TensorFlightClient,
    });
    useAppStore.getState().applyViewerState(new URLSearchParams(LINK));

    const s = useAppStore.getState();
    // The scoped selectors have nothing to say yet; the URL must not read that
    // as "no sets" and drop the params.
    expect(selectVisibleSets(s)).toBeNull();
    expect(selectLabelOverlay(s)).toBeNull();
    expect(selectUrlVisibleSets(s)).toEqual(["nuclei", "cells"]);
    expect(selectUrlLabelOverlay(s)).toBe("src/field/@labels/masks");
  });

  it("keeps them when tile_info fails", async () => {
    useAppStore.setState({
      client: {
        http: {
          tileInfo: () => Promise.reject(new TensorApiError(404, "no such tensor")),
        },
      } as unknown as TensorFlightClient,
    });
    useAppStore.getState().applyViewerState(new URLSearchParams(LINK));
    await settle();

    const s = useAppStore.getState();
    expect(s.target.status).toBe("failed");
    expect(selectUrlVisibleSets(s)).toEqual(["nuclei", "cells"]);
    expect(selectUrlLabelOverlay(s)).toBe("src/field/@labels/masks");
  });

  it("tells an explicit empty rs= from no rs at all", () => {
    useAppStore.setState({
      client: {
        http: { tileInfo: () => new Promise<TileInfo>(() => {}) },
      } as unknown as TensorFlightClient,
    });
    useAppStore.getState().applyViewerState(new URLSearchParams("id=src/field&rs="));
    expect(selectUrlVisibleSets(useAppStore.getState())).toEqual([]);

    useAppStore.getState().applyViewerState(new URLSearchParams("id=src/field"));
    expect(selectUrlVisibleSets(useAppStore.getState())).toBeNull();
    expect(selectUrlLabelOverlay(useAppStore.getState())).toBeNull();
  });

  it("does not carry a link's names onto a click that follows", () => {
    useAppStore.setState({
      client: {
        http: { tileInfo: () => new Promise<TileInfo>(() => {}) },
      } as unknown as TensorFlightClient,
    });
    useAppStore.getState().applyViewerState(new URLSearchParams(LINK));
    useAppStore.getState().openTensor("other");

    expect(selectUrlVisibleSets(useAppStore.getState())).toBeNull();
  });

  it("reads the resolved, scoped values once the target is ready", async () => {
    useAppStore.setState({ client: echoClient() });
    useAppStore.getState().applyViewerState(new URLSearchParams(LINK));
    await settle();

    const s = useAppStore.getState();
    expect(s.target.status).toBe("ready");
    expect(selectUrlVisibleSets(s)).toEqual(selectVisibleSets(s));
    expect(selectUrlVisibleSets(s)).toEqual(["nuclei", "cells"]);
    expect(selectUrlLabelOverlay(s)).toBe("src/field/@labels/masks");
  });
});

describe("the grid in view", () => {
  it("is the target's grid once it has landed", () => {
    view("first");
    expect(selectTileInfo(useAppStore.getState())).toMatchObject({ array_id: "first" });
  });

  it("is hidden the moment another tensor is opened, not when its own lands", () => {
    view("first");
    useAppStore.setState({ client: null });
    useAppStore.getState().openTensor("second");

    expect(selectTileInfo(useAppStore.getState())).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// ROI annotation state is scoped to the tensor in view
//
// Every way the tensor changes -- a click, a link, a late landing -- has to
// leave the previous tensor's state unreadable, so these are read through
// selectors rather than reset at each writer.
// ---------------------------------------------------------------------------

const ROI_FIXTURE = [
  {
    roiId: "a",
    arrayId: "first",
    setName: "nuclei",
    label: "",
    geometry: { kind: "point" as const, at: { x: 1, y: 1 } },
    plane: {},
    props: {},
    rev: 1,
    createdAtMs: 0,
    updatedAtMs: 0,
  },
];

/** State as it stands after a set has been fetched and used on `first`. */
function seedAnnotated() {
  useAppStore.setState(viewOf("first"));
  seedView("first", {
    rois: ROI_FIXTURE,
    roiScopes: { "": { truncated: true, skipped: 3 } },
    roisError: "boom",
    visibleSets: ["nuclei"],
  });
}

describe("ROI state across a tensor change", () => {
  it("shows a tensor its own annotations", () => {
    seedAnnotated();
    const s = useAppStore.getState();
    expect(selectRois(s)).toHaveLength(1);
    expect(selectRoiScopes(s)).toEqual({ "": { truncated: true, skipped: 3 } });
    expect(selectRoisError(s)).toBe("boom");
    expect(selectVisibleSets(s)).toEqual(["nuclei"]);
  });

  it("hides all of it once another tensor is selected", () => {
    seedAnnotated();
    useAppStore.getState().openTensor("second");
    const s = useAppStore.getState();
    expect(selectRois(s)).toEqual([]);
    expect(selectRoiScopes(s)).toEqual({});
    expect(selectRoisError(s)).toBeNull();
    expect(selectVisibleSets(s)).toBeNull();
  });

  it("hides all of it when a LINK changes the tensor", () => {
    // The path a reset in `selectSource` would miss entirely.
    seedAnnotated();
    useAppStore.getState().applyViewerState(new URLSearchParams({ id: "second/Image:0" }));
    const s = useAppStore.getState();
    expect(selectRois(s)).toEqual([]);
    expect(selectVisibleSets(s)).toBeNull();
    // And the rows are untouched, which is the point: nothing cleared them, so
    // reading them directly is what the leak was.
    expect(held("first").rois).toHaveLength(1);
    expect(held("first").roiScopes[""]?.truncated).toBe(true);
  });

  it("does not carry a warning onto a tensor it is not about", () => {
    // The worst of the leaks: "this tensor holds more than is shown" reads as a
    // fact about the image on screen.
    seedAnnotated();
    useAppStore.getState().applyViewerState(new URLSearchParams({ id: "second" }));
    const s = useAppStore.getState();
    expect(selectRoiScopes(s)).toEqual({});
    expect(selectRoisError(s)).toBeNull();
  });

  it("keeps a tensor's annotations under a content-pinned link to it", async () => {
    // A pin names which bytes to render, not a different tensor: annotations
    // carry no version, so the rows already held are this tensor's.
    seedAnnotated();
    await openLink("id=first@9f1c4e2b");
    expect(selectRois(useAppStore.getState())).toHaveLength(1);
  });

  it("hides annotations held for another tensor under a pinned link", () => {
    seedAnnotated();
    useAppStore.getState().applyViewerState(new URLSearchParams({ id: "second@9f1c4e2b" }));
    expect(selectRois(useAppStore.getState())).toEqual([]);
  });

  it("starts from the new tensor's default rather than editing another tensor's list", () => {
    seedAnnotated();
    view("second");
    useAppStore.getState().toggleSetVisible("debris");
    expect(selectVisibleSets(useAppStore.getState())).toEqual(["debris"]);
  });

  it("reports loading only for the tensor in view", () => {
    useAppStore.setState(viewOf("first"));
    seedView("second", { roisPending: [""] });
    expect(selectRoisPending(useAppStore.getState())).toEqual([]);
    seedView("first", { roisPending: [""] });
    expect(selectRoisPending(useAppStore.getState())).toEqual([""]);
  });
});

describe("per-tensor records", () => {
  it("keeps a tensor's rows and choices for when the user comes back", () => {
    seedAnnotated();
    view("second");
    expect(selectRois(useAppStore.getState())).toEqual([]);
    view("first");

    const s = useAppStore.getState();
    expect(selectRois(s)).toHaveLength(1);
    expect(selectVisibleSets(s)).toEqual(["nuclei"]);
  });

  it("keeps the rows on screen while a revisited tensor is listed again", async () => {
    seedAnnotated();
    view("second");
    useAppStore.setState({ client: echoClient() });
    useAppStore.getState().openTensor("first");
    await settle();

    const s = useAppStore.getState();
    expect(selectRois(s)).toHaveLength(1);
    // Nothing counts as landed, so the next loadRois asks the server.
    expect(selectRoiScopes(s)).toEqual({});
  });

  it("keeps a link's set choice out of the record until the target lands", async () => {
    seedAnnotated();
    await openLink("id=first&rs=cells");
    expect(selectVisibleSets(useAppStore.getState())).toEqual(["cells"]);

    // A link naming none opens on the default, not the last choice.
    await openLink("id=first");
    expect(selectVisibleSets(useAppStore.getState())).toBeNull();
  });

  it("evicts the least recently written record past the limit", () => {
    view("t0");
    for (let i = 0; i < 10; i++) {
      useAppStore.setState(viewOf(`t${i}`));
      useAppStore.getState().toggleSetVisible("x");
    }
    useAppStore.setState(viewOf("t0"));
    // Only the current tensor is protected: t0 was written first and is oldest.
    const keys = Object.keys(useAppStore.getState().views);
    expect(keys).toHaveLength(8);
    expect(keys).not.toContain("t1");
    expect(keys).toContain("t9");
  });
});

describe("the sets on screen", () => {
  const SETS = [
    { setName: "nuclei", count: 1, reserved: false },
    { setName: "cells", count: 2, reserved: false },
    { setName: "@ome", count: 120, reserved: true },
  ];

  function seedListed() {
    useAppStore.setState(viewOf("first"));
    seedView("first", {
      rois: ROI_FIXTURE,
      roiSets: SETS,
      roiScopes: { "": { truncated: false, skipped: 0 } },
      visibleSets: null,
    });
  }

  it("materialises the default before the first toggle", () => {
    // `null` is "this tensor's default", which is not "none": switching one set
    // off must leave the other client-owned sets on.
    seedListed();
    useAppStore.getState().toggleSetVisible("cells");
    expect(selectVisibleSets(useAppStore.getState())).toEqual(["nuclei"]);
  });

  it("switches a server-owned set on by name", () => {
    seedListed();
    useAppStore.getState().toggleSetVisible("@ome");
    expect(selectVisibleSets(useAppStore.getState())).toEqual(["nuclei", "cells", "@ome"]);
  });

  it("adopts the sets a link names, and turns the overlay on for them", async () => {
    seedListed();
    useAppStore.setState({ showRois: false });
    await openLink("id=first&rs=@ome");
    const s = useAppStore.getState();
    expect(selectVisibleSets(s)).toEqual(["@ome"]);
    expect(s.showRois).toBe(true);
  });

  it("leaves the overlay toggle alone when a link names no sets", () => {
    seedListed();
    useAppStore.setState({ showRois: false });
    useAppStore.getState().applyViewerState(new URLSearchParams("id=first"));
    expect(useAppStore.getState().showRois).toBe(false);
    // A preference, so it outlives this test too.
    useAppStore.setState({ showRois: true });
  });
});

describe("loadRois", () => {
  type Listing = { rois?: RoiAnnotation[]; sets?: RoiSetInfo[]; truncated?: boolean; skipped?: number };

  /**
   * A client whose listRois records what it was asked for, as `id` or
   * `id?set`, and answers with `respond(setName)` -- `setName` undefined for
   * the unqualified listing.
   */
  function stubClient(asked: string[], respond: (setName?: string) => Listing = () => ({})) {
    return {
      http: {
        listRois: (arrayId: string, setName?: string) => {
          asked.push(setName ? `${arrayId}?${setName}` : arrayId);
          const r = respond(setName);
          return Promise.resolve({
            rois: r.rois ?? [],
            sets: r.sets ?? [],
            truncated: r.truncated ?? false,
            skipped: r.skipped ?? 0,
          });
        },
      },
    } as unknown as TensorFlightClient;
  }

  const OME_ROW = { ...ROI_FIXTURE[0]!, roiId: "o1", setName: "@ome" };

  /** Forget the client-owned listing landed, so the next loadRois fetches it again. */
  function unlandListing() {
    const scopes = { ...held().roiScopes };
    delete scopes[""];
    seedView("first", { roiScopes: scopes });
  }

  function fresh(
    client: TensorFlightClient,
    over: { target?: ViewTarget; visibleSets?: string[] | null } = {},
  ) {
    const { visibleSets, ...top } = over;
    useAppStore.setState({ client, ...viewOf("first"), views: {}, roisUnavailable: false, ...top });
    if (visibleSets !== undefined) seedView("first", { visibleSets });
  }

  it("fetches the tensor in view", async () => {
    const asked: string[] = [];
    fresh(stubClient(asked));
    await useAppStore.getState().loadRois();
    expect(asked).toEqual(["first"]);
    expect(held("first").roiScopes).not.toEqual({});
    expect(selectRoiScopes(useAppStore.getState())).toEqual({ "": { truncated: false, skipped: 0 } });
  });

  it("waits for the target to resolve", async () => {
    // Nothing could be shown yet, and there is no address to ask about.
    const asked: string[] = [];
    fresh(stubClient(asked), { target: targetFor(null) });
    await useAppStore.getState().loadRois();
    expect(asked).toEqual([]);
  });

  it("asks under the address tile_info gave, and holds the rows under the key", async () => {
    const asked: string[] = [];
    fresh(stubClient(asked), {
      target: targetFor("first", {
        info: { ...TILE_INFO, array_id: "first@9f1c4e2b" } as TileInfo,
      }),
    });
    await useAppStore.getState().loadRois();
    expect(asked).toEqual(["first@9f1c4e2b"]);
    expect(held("first").roiScopes).not.toEqual({});
  });

  it("asks once for a set it already holds", async () => {
    const asked: string[] = [];
    fresh(stubClient(asked));
    await useAppStore.getState().loadRois();
    await useAppStore.getState().loadRois();
    expect(asked).toEqual(["first"]);
  });

  it("does not fetch a server-owned set until it is on screen", async () => {
    // The lazy half: the listing names it, the rows wait for the toggle.
    const asked: string[] = [];
    fresh(stubClient(asked, () => ({ sets: [{ setName: "@ome", count: 120, reserved: true }] })));
    await useAppStore.getState().loadRois();
    expect(asked).toEqual(["first"]);
    useAppStore.getState().toggleSetVisible("@ome");
    await useAppStore.getState().loadRois();
    expect(asked).toEqual(["first", "first?@ome"]);
    expect(Object.keys(selectRoiScopes(useAppStore.getState()))).toEqual(["", "@ome"]);
  });

  it("fetches a server-owned set a link names alongside the listing, not after it", async () => {
    // Before any listing has landed the prefix says the set needs its own
    // fetch, so a link straight to it does not wait a round trip to find out.
    const asked: string[] = [];
    fresh(stubClient(asked), { visibleSets: ["@ome"] });
    await useAppStore.getState().loadRois();
    expect(asked).toEqual(["first", "first?@ome"]);
  });

  it("does not fetch a named set the listing says is not there", async () => {
    const asked: string[] = [];
    fresh(stubClient(asked, () => ({ sets: [{ setName: "@ome", count: 1, reserved: true }] })));
    await useAppStore.getState().loadRois();
    seedView("first", { visibleSets: ["@gone"] });
    await useAppStore.getState().loadRois();
    expect(asked).toEqual(["first"]);
  });

  it("holds a named set's rows beside the listing's, and replaces only its own on a refetch", async () => {
    const client = stubClient([], (setName) => ({
      rois: setName ? [OME_ROW] : [{ ...ROI_FIXTURE[0]!, roiId: "n1" }],
      sets: [
        { setName: "nuclei", count: 1, reserved: false },
        { setName: "@ome", count: 1, reserved: true },
      ],
    }));
    fresh(client, { visibleSets: ["nuclei", "@ome"] });
    await useAppStore.getState().loadRois();
    expect(held().rois.map((r) => r.roiId).sort()).toEqual(["n1", "o1"]);
    expect(selectRoiSets(useAppStore.getState())).toHaveLength(2);
  });

  it("un-lands a set whose stored count no longer matches what it holds", async () => {
    // The server rebuilds a reserved set on every registration. A later
    // listing carries the new count, which is how the stale rows are noticed
    // -- and the next loadRois fetches them again.
    const asked: string[] = [];
    let omeCount = 1;
    const client = stubClient(asked, (setName) => ({
      rois: setName ? [OME_ROW] : [],
      sets: [{ setName: "@ome", count: omeCount, reserved: true }],
    }));
    fresh(client, { visibleSets: ["@ome"] });
    await useAppStore.getState().loadRois();
    expect(Object.keys(selectRoiScopes(useAppStore.getState())).sort()).toEqual(["", "@ome"]);

    // Something re-registered the source; the listing is fetched again for
    // whatever reason and now says two rows.
    omeCount = 2;
    unlandListing();
    await useAppStore.getState().loadRois();
    expect(selectRoiScopes(useAppStore.getState())["@ome"]).toBeUndefined();
    await useAppStore.getState().loadRois();
    expect(asked.filter((a) => a.endsWith("?@ome"))).toHaveLength(2);
  });

  it("counts a row it could not read as held, rather than as someone else's change", async () => {
    const client = stubClient([], (setName) => ({
      rois: setName ? [OME_ROW] : [],
      sets: [{ setName: "@ome", count: 2, reserved: true }],
      skipped: setName ? 1 : 0,
    }));
    fresh(client, { visibleSets: ["@ome"] });
    await useAppStore.getState().loadRois();
    unlandListing();
    await useAppStore.getState().loadRois();
    expect(selectRoiScopes(useAppStore.getState())["@ome"]).toEqual({ truncated: false, skipped: 1 });
  });

  it("does not un-land a set the cap clipped, which disagrees by construction", async () => {
    const client = stubClient([], (setName) => ({
      rois: setName ? [OME_ROW] : [],
      sets: [{ setName: "@ome", count: 5000, reserved: true }],
      truncated: setName !== undefined,
    }));
    fresh(client, { visibleSets: ["@ome"] });
    await useAppStore.getState().loadRois();
    unlandListing();
    await useAppStore.getState().loadRois();
    expect(selectRoiScopes(useAppStore.getState())["@ome"]).toEqual({ truncated: true, skipped: 0 });
  });

  it("lands a response for a tensor that moved on, out of view", async () => {
    // Held under its own tensor, where the selectors hide it -- and where a
    // switch back finds it without another round trip.
    let resolve: (v: unknown) => void = () => {};
    const client = {
      http: {
        listRois: () => new Promise((r) => { resolve = r; }),
      },
    } as unknown as TensorFlightClient;
    fresh(client);
    const load = useAppStore.getState().loadRois();
    view("second");
    resolve({ rois: [ROI_FIXTURE[0]], sets: [], truncated: false, skipped: 0 });
    await load;
    expect(held("first").roiScopes).not.toEqual({});
    expect(selectRois(useAppStore.getState())).toEqual([]);
    view("first");
    expect(selectRois(useAppStore.getState())).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Authoring
// ---------------------------------------------------------------------------

describe("authoring state", () => {
  const GEOM = { kind: "point" as const, at: { x: 1, y: 2 } };

  function stored(over: Record<string, unknown> = {}) {
    return {
      roiId: "srv1",
      arrayId: "first",
      setName: "default",
      label: "",
      geometry: GEOM,
      plane: {},
      props: {},
      rev: 1,
      createdAtMs: 0,
      updatedAtMs: 0,
      ...over,
    };
  }

  function seedFor(client: unknown) {
    useAppStore.setState({
      client: client as TensorFlightClient,
      ...viewOf("first"),
      views: {},
      selectedRoiId: null,
      roiWriteError: null,
      newLabel: "",
      newSetName: "",
      render3d: false,
    });
  }

  it("sends the label, set and pin, and adopts what the server stored", async () => {
    const calls: unknown[] = [];
    seedFor({
      http: {
        putRois: (arrayId: string, rois: unknown[], opts: unknown) => {
          calls.push({ arrayId, rois, opts });
          return Promise.resolve({ stored: [stored({ label: "cell" })], conflicts: [], skipped: 0 });
        },
      },
    });
    useAppStore.setState({ newLabel: "cell", newSetName: "nuclei" });
    await useAppStore.getState().createRoi(GEOM, { 2: 12 });

    expect(calls).toEqual([
      {
        arrayId: "first",
        rois: [{ geometry: GEOM, plane: { 2: 12 }, label: "cell", setName: "nuclei" }],
        opts: { checkRev: true },
      },
    ]);
    // The server assigns the id, so the local copy has to be its answer, not
    // the shape that was sent.
    expect(held().rois.map((r) => r.roiId)).toEqual(["srv1"]);
    expect(useAppStore.getState().selectedRoiId).toBe("srv1");
  });

  it("targets the tensor tile_info resolved a bare source_id to, not the bare id", async () => {
    // biopb/biopb#1155: clicking a multi-tensor source's row (rather than one
    // of its tensors) selects the bare source_id, which the Flight server
    // resolves once tile_info answers. Annotations have to follow that
    // resolution or they land on a different tensor than the one on screen.
    const calls: unknown[] = [];
    seedFor({
      http: {
        putRois: (arrayId: string, rois: unknown[], opts: unknown) => {
          calls.push({ arrayId, rois, opts });
          return Promise.resolve({ stored: [stored()], conflicts: [], skipped: 0 });
        },
      },
    });
    view("scratch/tensorA");

    await useAppStore.getState().createRoi(GEOM, {});

    expect((calls[0] as { arrayId: string }).arrayId).toBe("scratch/tensorA");
  });

  it("draws the annotation before the server has stored it", async () => {
    // The click that finishes a shape also clears the draft, so without this
    // the shape vanishes for a round trip. Nothing about the click should wait
    // on the network.
    let seen: RoiAnnotation[] = [];
    let release = () => {};
    const landed = new Promise<void>((resolve) => (release = resolve));
    seedFor({
      http: {
        putRois: async () => {
          seen = held().rois;
          await landed;
          return { stored: [stored({ label: "cell" })], conflicts: [], skipped: 0 };
        },
      },
    });
    const writing = useAppStore.getState().createRoi(GEOM, { 2: 12 });
    // Drawn already, with the geometry that was traced.
    expect(seen).toHaveLength(1);
    expect(seen[0]?.geometry).toEqual(GEOM);
    expect(seen[0]?.plane).toEqual({ 2: 12 });
    // Not selected yet: the id is this client's invention until the server
    // answers with its own.
    expect(useAppStore.getState().selectedRoiId).toBeNull();

    release();
    await writing;
    // Swapped in place for the stored row, not appended beside it.
    expect(held().rois.map((r) => r.roiId)).toEqual(["srv1"]);
    expect(useAppStore.getState().selectedRoiId).toBe("srv1");
  });

  it("takes the provisional row back out when the write fails", async () => {
    seedFor({ http: { putRois: () => Promise.reject(new Error("422 rejected")) } });
    await useAppStore.getState().createRoi(GEOM, {});
    expect(held().rois).toEqual([]);
    expect(useAppStore.getState().roiWriteError).toContain("422 rejected");
  });

  it("puts a provisional row in the set its stored form will land in", async () => {
    // The server substitutes "default" for an empty set name, so a blank one
    // here would show a nameless set row for the length of a round trip.
    let seen: RoiAnnotation[] = [];
    seedFor({
      http: {
        putRois: () => {
          seen = held().rois;
          return Promise.resolve({ stored: [stored()], conflicts: [], skipped: 0 });
        },
      },
    });
    await useAppStore.getState().createRoi(GEOM, {});
    expect(seen[0]?.setName).toBe("default");
  });

  it("leaves a provisional row alone when Delete reaches it mid-write", async () => {
    // Its id is one this client invented; the server has never heard of it.
    let asked = false;
    seedFor({ http: { deleteRois: () => ((asked = true), Promise.resolve([])) } });
    const provisional = { ...stored(), roiId: "pending,1" };
    seedView("first", { rois: [provisional] });
    await useAppStore.getState().deleteRoi("pending,1");
    expect(asked).toBe(false);
    expect(held().rois).toHaveLength(1);
  });

  it("removes a deleted annotation on the keystroke, not on the answer", async () => {
    let seen: RoiAnnotation[] = [];
    seedFor({
      http: {
        deleteRois: () => {
          seen = held().rois;
          return Promise.resolve(["srv1"]);
        },
      },
    });
    seedView("first", { rois: [stored()] });
    useAppStore.setState({ selectedRoiId: "srv1" });
    await useAppStore.getState().deleteRoi("srv1");
    // Gone before the request was made, not after it came back.
    expect(seen).toEqual([]);
  });

  it("puts a row back where it was when the delete fails", async () => {
    // Not at the end: the overlay draws later annotations over earlier ones, so
    // a failed delete must not reorder what covers what.
    seedFor({ http: { deleteRois: () => Promise.reject(new Error("503 upstream")) } });
    seedView("first", { rois: [stored({ roiId: "a" }), stored({ roiId: "b" }), stored({ roiId: "c" })] });
    await useAppStore.getState().deleteRoi("b");
    expect(held().rois.map((r) => r.roiId)).toEqual(["a", "b", "c"]);
    expect(useAppStore.getState().roiWriteError).toContain("503 upstream");
  });

  it("restores a failed delete into the tensor it came out of, not the one in view", async () => {
    seedFor({
      http: {
        deleteRois: () => {
          useAppStore.setState(viewOf("second"));
          return Promise.reject(new Error("503 upstream"));
        },
      },
    });
    seedView("first", { rois: [stored()] });
    await useAppStore.getState().deleteRoi("srv1");
    expect(held("first").rois).toHaveLength(1);
    expect(held("second").rois).toEqual([]);
  });

  it("lands a write in its own tensor's record after the user moved on", async () => {
    // The annotation is stored; it belongs to the image it was drawn on, and
    // is neither drawn over the new one nor selected there.
    seedFor({
      http: {
        putRois: () => {
          useAppStore.setState(viewOf("second"));
          return Promise.resolve({ stored: [stored()], conflicts: [], skipped: 0 });
        },
      },
    });
    await useAppStore.getState().createRoi(GEOM, {});
    expect(held("first").rois.map((r) => r.roiId)).toEqual(["srv1"]);
    expect(selectRois(useAppStore.getState())).toEqual([]);
    expect(useAppStore.getState().selectedRoiId).toBeNull();
  });

  it("reports a failed write instead of pretending it landed", async () => {
    seedFor({ http: { putRois: () => Promise.reject(new Error("422 rejected")) } });
    await useAppStore.getState().createRoi(GEOM, {});
    expect(useAppStore.getState().roiWriteError).toContain("422 rejected");
    expect(held().rois).toEqual([]);
  });

  it("removes a deleted annotation and clears the selection with it", async () => {
    seedFor({ http: { deleteRois: () => Promise.resolve(["srv1"]) } });
    seedView("first", { rois: [stored()] });
    useAppStore.setState({ selectedRoiId: "srv1" });
    await useAppStore.getState().deleteRoi("srv1");
    expect(held().rois).toEqual([]);
    expect(useAppStore.getState().selectedRoiId).toBeNull();
  });

  it("drops a row the server says it never had, rather than leaving it stuck", async () => {
    seedFor({ http: { deleteRois: () => Promise.resolve([]) } });
    seedView("first", { rois: [stored()] });
    useAppStore.setState({ selectedRoiId: "srv1" });
    await useAppStore.getState().deleteRoi("srv1");
    expect(held().rois).toEqual([]);
  });

  it("clears a whole set by name, leaving the other sets alone", async () => {
    const calls: unknown[] = [];
    seedFor({
      http: {
        deleteRois: (arrayId: string, roiIds: unknown, opts: unknown) => {
          calls.push({ arrayId, roiIds, opts });
          // Deliberately fewer ids than the set holds: the cap means this
          // client may never have seen every row it just deleted.
          return Promise.resolve(["srv1"]);
        },
      },
    });
    seedView("first", {
      rois: [
        stored({ roiId: "srv1", setName: "nuclei" }),
        stored({ roiId: "srv2", setName: "nuclei" }),
        stored({ roiId: "srv3", setName: "cells" }),
      ],
    });
    await useAppStore.getState().clearRoiSet("nuclei");

    // No ids: the server drops the set in one transaction rather than the
    // subset this client happens to hold.
    expect(calls).toEqual([{ arrayId: "first", roiIds: undefined, opts: { setName: "nuclei" } }]);
    expect(held().rois.map((r) => r.roiId)).toEqual(["srv3"]);
  });

  it("drops a selection that was in the cleared set, and the set from the chosen list", async () => {
    // The name means nothing once the set is gone, and leaving it on the list
    // would pin a later set of the same name to this one's toggle.
    seedFor({ http: { deleteRois: () => Promise.resolve(["srv1"]) } });
    seedView("first", {
      rois: [stored({ roiId: "srv1", setName: "nuclei" }), stored({ roiId: "srv2", setName: "cells" })],
      roiSets: [
        { setName: "nuclei", count: 1, reserved: false },
        { setName: "cells", count: 1, reserved: false },
      ],
      visibleSets: ["nuclei", "cells"],
    });
    useAppStore.setState({ selectedRoiId: "srv1" });
    await useAppStore.getState().clearRoiSet("nuclei");
    expect(useAppStore.getState().selectedRoiId).toBeNull();
    expect(held().visibleSets).toEqual(["cells"]);
    expect(held().roiSets.map((set) => set.setName)).toEqual(["cells"]);
  });

  it("puts a new annotation's set on screen when the chosen list would hide it", async () => {
    // A materialised list is exactly the sets shown, so a set switched off --
    // or one that did not exist a moment ago -- would swallow the shape the
    // user just traced.
    seedFor({ http: { putRois: () => Promise.resolve({ stored: [stored({ setName: "fresh" })], conflicts: [], skipped: 0 }) } });
    useAppStore.setState({ newSetName: "fresh" });
    seedView("first", { roiSets: [], visibleSets: ["nuclei"] });
    await useAppStore.getState().createRoi(GEOM, {});
    expect(held().visibleSets).toEqual(["nuclei", "fresh"]);
    // And the listing counts it, so a later listing does not read the row as
    // someone else's change.
    expect(held().roiSets).toEqual([{ setName: "fresh", count: 1, reserved: false }]);
  });

  it("moves the listing's count with a delete, and back when it fails", async () => {
    seedFor({ http: { deleteRois: () => Promise.reject(new Error("500 upstream")) } });
    seedView("first", {
      rois: [stored({ roiId: "srv1", setName: "nuclei" })],
      roiSets: [{ setName: "nuclei", count: 1, reserved: false }],
    });
    const done = useAppStore.getState().deleteRoi("srv1");
    expect(held().roiSets[0]?.count).toBe(0);
    await done;
    expect(held().roiSets[0]?.count).toBe(1);
  });

  it("keeps a selection that was in another set", async () => {
    seedFor({ http: { deleteRois: () => Promise.resolve([]) } });
    seedView("first", {
      rois: [stored({ roiId: "srv1", setName: "nuclei" }), stored({ roiId: "srv2", setName: "cells" })],
    });
    useAppStore.setState({ selectedRoiId: "srv2" });
    await useAppStore.getState().clearRoiSet("nuclei");
    expect(useAppStore.getState().selectedRoiId).toBe("srv2");
  });

  it("reports a failed clear instead of emptying the panel", async () => {
    seedFor({ http: { deleteRois: () => Promise.reject(new Error("500 upstream")) } });
    seedView("first", { rois: [stored({ setName: "nuclei" })] });
    await useAppStore.getState().clearRoiSet("nuclei");
    expect(useAppStore.getState().roiWriteError).toContain("500 upstream");
    expect(held().rois).toHaveLength(1);
  });

  it("applies a clear to its own tensor after the user moved on", async () => {
    seedFor({
      http: {
        deleteRois: () => {
          useAppStore.setState(viewOf("second"));
          return Promise.resolve(["srv1"]);
        },
      },
    });
    seedView("first", { rois: [stored({ setName: "nuclei" })] });
    seedView("second", { rois: [stored({ roiId: "other", setName: "nuclei" })] });
    await useAppStore.getState().clearRoiSet("nuclei");
    // The set was deleted from the tensor it was asked of, and only from there.
    expect(held("first").rois).toEqual([]);
    expect(held("second").rois).toHaveLength(1);
  });

  it("abandons the draft when the tool changes", () => {
    useAppStore.setState({ tool: "polygon", ...viewOf("first") });
    useAppStore.getState().setDraft({ tool: "polygon", points: [[0, 0]] });
    expect(selectDraft(useAppStore.getState())).not.toBeNull();
    useAppStore.getState().setTool("rectangle");
    expect(selectDraft(useAppStore.getState())).toBeNull();
    expect(held().draft).toBeNull();
  });

  it("materialises the default before the first broadcast toggle", () => {
    // Otherwise toggling one axis would silently also pin whatever the default
    // was broadcasting.
    seedFor({ http: {} });
    useAppStore.getState().toggleBroadcastAxis(2, [1]);
    expect(held().broadcastAxes?.sort()).toEqual([1, 2]);
  });

  it("hides the draft in 3-D and on another tensor", () => {
    seedFor({ http: {} });
    useAppStore.getState().setDraft({ tool: "polygon", points: [[0, 0]] });
    expect(selectDraft(useAppStore.getState())).not.toBeNull();

    useAppStore.setState({ render3d: true });
    expect(selectDraft(useAppStore.getState())).toBeNull();

    useAppStore.setState({ render3d: false, ...viewOf("second") });
    expect(selectDraft(useAppStore.getState())).toBeNull();
  });

  it("resolves the selection against the set in view, so it cannot go stale", () => {
    seedFor({ http: {} });
    seedView("first", { rois: [stored()] });
    useAppStore.setState({ selectedRoiId: "srv1" });
    expect(selectSelectedRoi(useAppStore.getState())?.roiId).toBe("srv1");

    // The annotation is gone from the set: no cleanup needed anywhere.
    seedView("first", { rois: [] });
    expect(selectSelectedRoi(useAppStore.getState())).toBeNull();
  });
});

describe("a draft does not survive a plane change", () => {
  function startDraft() {
    useAppStore.setState({
      ...viewOf("first"),
      render3d: false,
      position: { ...BASE_POSITION, z: 12 },
    });
    useAppStore.getState().setDraft({ tool: "polygon", points: [[0, 0], [4, 0]] });
  }

  it("is there while the plane holds still", () => {
    startDraft();
    expect(selectDraft(useAppStore.getState())).not.toBeNull();
  });

  it("goes when the slider moves", () => {
    // The vertices were traced against another plane's pixels; finishing here
    // would pin the shape to a plane it was not drawn on.
    startDraft();
    useAppStore.getState().setPosition({ z: 13 });
    expect(selectDraft(useAppStore.getState())).toBeNull();
  });

  it("goes when play steps any axis", () => {
    startDraft();
    useAppStore.getState().setPosition({ t: 1 });
    expect(selectDraft(useAppStore.getState())).toBeNull();
  });

  it("goes when an unnamed axis moves", () => {
    startDraft();
    useAppStore.getState().setPosition({ axes: { a0: 2 } });
    expect(selectDraft(useAppStore.getState())).toBeNull();
  });

  it("survives a contrast or gamma change", () => {
    // Display settings do not move the plane, so they do not invalidate a shape.
    startDraft();
    useAppStore.getState().setDisplay({ gamma: 2.2 });
    useAppStore.getState().setDisplay({ contrastMode: "fixed" });
    useAppStore.getState().setDisplay({ percentileScale: 2 });
    expect(selectDraft(useAppStore.getState())).not.toBeNull();
  });

  it("comes back if the plane comes back", () => {
    // A stray scroll costs nothing; the draft is hidden, not destroyed.
    startDraft();
    useAppStore.getState().setPosition({ z: 13 });
    useAppStore.getState().setPosition({ z: 12 });
    expect(selectDraft(useAppStore.getState())).not.toBeNull();
  });

  it("a click after the plane moved starts a fresh draft, not a continuation", () => {
    // TileViewer places through `selectDraft`, so the stale vertices are not
    // extended -- this is the behaviour the hidden-not-destroyed choice rests on.
    startDraft();
    useAppStore.getState().setPosition({ z: 13 });
    const seen = selectDraft(useAppStore.getState());
    expect(seen).toBeNull();
  });
});

describe("the overlay toggle governs the whole annotation surface", () => {
  function drafting() {
    useAppStore.setState({
      ...viewOf("first"),
      render3d: false,
      showRois: true,
      position: { ...BASE_POSITION, z: 12 },
    });
    useAppStore.getState().setDraft({ tool: "polygon", points: [[0, 0], [4, 0]] });
  }

  it("hides an in-progress draft, not just the stored shapes", () => {
    // Otherwise a draft keeps drawing over an overlay the user switched off,
    // and finishing writes an annotation they cannot see.
    drafting();
    expect(selectDraft(useAppStore.getState())).not.toBeNull();
    useAppStore.getState().setShowRois(false);
    expect(selectDraft(useAppStore.getState())).toBeNull();
  });

  it("gives the draft back when the overlay comes back", () => {
    drafting();
    useAppStore.getState().setShowRois(false);
    useAppStore.getState().setShowRois(true);
    expect(selectDraft(useAppStore.getState())).not.toBeNull();
  });
});

describe("recent sources", () => {
  const recentClient = (
    sources: DataSourceDescriptor[],
    tileInfo: (id: string) => Promise<TileInfo>,
  ) =>
    ({
      listSources: vi.fn().mockResolvedValue(sources),
      http: {
        readyz: vi.fn().mockResolvedValue({ backend_health: {} }),
        tileInfo: vi.fn(tileInfo),
      },
    }) as unknown as TensorFlightClient;

  const UPLOAD_INFO = {
    array_id: "upload_7f3@abcd1234",
    dim_labels: ["Y", "X"],
    shape: [1024, 1024],
    chunk_shape: [256, 256],
    dtype: "uint16",
  } as unknown as TileInfo;

  const apiError = (status: number) =>
    new TensorApiError(status, status === 404 ? "Not Found" : "Bad Gateway", "");

  it("rebuilds an unlisted id from tile_info, which the catalog cannot answer", async () => {
    // The whole point: a cache:// upload is deliberately never in the catalog
    // (biopb/biopb#265), so /api/sources 404s it and only the registry-backed
    // tile_info can describe it.
    const client = recentClient([], async () => UPLOAD_INFO);
    useAppStore.setState({ client, sources: [], recentIds: ["upload_7f3"] });

    await useAppStore.getState().hydrateRecents();

    const [row] = useAppStore.getState().recentSources;
    expect(row?.source_id).toBe("upload_7f3");
    expect(row?.tensors[0]?.shape).toEqual([1024, 1024]);
  });

  it("reuses the catalog descriptor instead of a round trip", async () => {
    const tileInfo = vi.fn(async () => UPLOAD_INFO);
    const client = recentClient([SOURCE], tileInfo);
    useAppStore.setState({ client, sources: [SOURCE], recentIds: [SOURCE.source_id] });

    await useAppStore.getState().hydrateRecents();

    expect(client.http.tileInfo).not.toHaveBeenCalled();
    expect(useAppStore.getState().recentSources[0]).toBe(SOURCE);
  });

  it("drops an id the server says is gone", async () => {
    const client = recentClient([], async () => {
      throw apiError(404);
    });
    useAppStore.setState({ client, sources: [], recentIds: ["evicted"] });

    await useAppStore.getState().hydrateRecents();

    expect(useAppStore.getState().recentIds).toEqual([]);
    expect(useAppStore.getState().recentSources).toEqual([]);
  });

  it("keeps an id the server merely failed to answer for", async () => {
    // A 502 says nothing about the id. Pruning on it would empty the list
    // exactly when the server is down -- when it is most worth keeping.
    const client = recentClient([], async () => {
      throw apiError(502);
    });
    useAppStore.setState({ client, sources: [], recentIds: ["upload_7f3"] });

    await useAppStore.getState().hydrateRecents();

    expect(useAppStore.getState().recentIds).toEqual(["upload_7f3"]);
    expect(useAppStore.getState().recentSources).toEqual([]);
  });

  it("keeps recency order across both resolution paths", async () => {
    const client = recentClient([SOURCE], async () => UPLOAD_INFO);
    useAppStore.setState({
      client,
      sources: [SOURCE],
      recentIds: ["upload_7f3", SOURCE.source_id],
    });

    await useAppStore.getState().hydrateRecents();

    expect(useAppStore.getState().recentSources.map((s) => s.source_id)).toEqual([
      "upload_7f3",
      SOURCE.source_id,
    ]);
  });

  it("records a source opened from a link", async () => {
    // The case the list exists for: an upload is reachable only by the id its
    // DoPut returned, which arrives as a URL.
    useAppStore.setState({ recentIds: [] });
    useAppStore.getState().applyViewerState(new URLSearchParams("id=upload_7f3"));
    expect(useAppStore.getState().recentIds).toEqual(["upload_7f3"]);
  });

  it("records the source, not the field, when a link names a tensor", () => {
    useAppStore.setState({ recentIds: [] });
    useAppStore.getState().applyViewerState(new URLSearchParams("id=multi/Image:2"));
    expect(useAppStore.getState().recentIds).toEqual(["multi"]);
  });

  it("does not reorder on a repeat visit to what is already newest", () => {
    // applyViewerState re-runs on every slider move; a fresh list each time
    // would be a localStorage write per frame.
    useAppStore.setState({ recentIds: ["a", "b"] });
    const before = useAppStore.getState().recentIds;
    useAppStore.getState().applyViewerState(new URLSearchParams("id=a"));
    expect(useAppStore.getState().recentIds).toBe(before);
  });

  it("adopts another tab's list without writing it back", () => {
    useAppStore.setState({ recentIds: ["a"] });
    useAppStore.getState().syncRecents(["b", "a"]);
    expect(useAppStore.getState().recentIds).toEqual(["b", "a"]);
  });
});

describe("catalogFingerprint", () => {
  const tensor = (over = {}) => ({
    array_id: "listed",
    dim_labels: ["y", "x"],
    shape: [4, 4],
    chunk_shape: [],
    dtype: "uint16",
    ...over,
  });

  it("is stable across separately-built but equal listings", () => {
    expect(catalogFingerprint([{ ...SOURCE }])).toBe(
      catalogFingerprint([{ ...SOURCE }]),
    );
  });

  it("changes when a source resolves in place", () => {
    // The url set is identical here -- the whole point. Before this keyed on
    // more than urls, a resolve that completed was invisible to the poll.
    const before = catalogFingerprint([
      { ...SOURCE, is_resolved: false, tensors: [] },
    ]);
    const after = catalogFingerprint([
      { ...SOURCE, is_resolved: true, tensors: [tensor()] },
    ]);
    expect(before).not.toBe(after);
  });

  it("changes on a field it was never told about", () => {
    // The point of stringifying the whole descriptor: a field added to
    // DataSourceDescriptor later is covered without anyone remembering to
    // extend this, which is how the url-only check went blind.
    const before = catalogFingerprint([{ ...SOURCE, source_type: "zarr" }]);
    const after = catalogFingerprint([{ ...SOURCE, source_type: "nd2" }]);
    expect(before).not.toBe(after);
  });

  it("changes when a tensor's shape grows", () => {
    const small = catalogFingerprint([{ ...SOURCE, tensors: [tensor()] }]);
    const grown = catalogFingerprint([
      { ...SOURCE, tensors: [tensor({ shape: [8, 4, 4] })] },
    ]);
    expect(small).not.toBe(grown);
  });

  it("changes when a source gains a second tensor", () => {
    const one = catalogFingerprint([{ ...SOURCE, tensors: [tensor()] }]);
    const two = catalogFingerprint([
      { ...SOURCE, tensors: [tensor(), tensor({ array_id: "listed/b" })] },
    ]);
    expect(one).not.toBe(two);
  });

  it("does not collide across a field boundary", () => {
    // Both of these concatenated to "abRD" under the hand-rolled version this
    // replaced, so a poll could miss a change outright. JSON's own framing is
    // what makes them distinguishable.
    expect(
      catalogFingerprint([{ ...SOURCE, source_id: "a", source_url: "b" }]),
    ).not.toBe(
      catalogFingerprint([{ ...SOURCE, source_id: "ab", source_url: "" }]),
    );
  });

  it("does not collide across a source boundary", () => {
    // Same failure one level up: two sources ran together into one string.
    expect(
      catalogFingerprint([
        { ...SOURCE, source_id: "x", source_url: "" },
        { ...SOURCE, source_id: "y", source_url: "" },
      ]),
    ).not.toBe(
      catalogFingerprint([{ ...SOURCE, source_id: "xRDy", source_url: "" }]),
    );
  });
});

describe("resolve jobs", () => {
  const status = (over: Partial<SourceJobStatus> = {}): SourceJobStatus => ({
    kind: "resolve",
    source_id: "cloud0",
    state: "running",
    progress: {},
    error: null,
    elapsed_seconds: 0,
    cancel_requested: false,
    ...over,
  });

  const stubClient = (http: Record<string, unknown>) =>
    ({ http, listSources: vi.fn().mockResolvedValue([]) }) as unknown as TensorFlightClient;

  afterEach(() => {
    useAppStore.getState().stopJobPolling();
    useAppStore.setState({ sourceJobs: {}, client: null });
  });

  it("records the job the server hands back", async () => {
    useAppStore.setState({
      client: stubClient({
        startResolve: vi.fn().mockResolvedValue(status({ started: true })),
      }),
    });
    await useAppStore.getState().startResolve("cloud0");
    expect(useAppStore.getState().sourceJobs["resolve:cloud0"]?.state).toBe("running");
  });

  it("surfaces a resolve that never started", async () => {
    // The modal reads sourceJobs, so a rejected POST has to leave an entry
    // behind or the failure is invisible.
    useAppStore.setState({
      client: stubClient({
        startResolve: vi.fn().mockRejectedValue(new Error("host unreachable")),
      }),
    });
    await useAppStore.getState().startResolve("cloud0");
    const job = useAppStore.getState().sourceJobs["resolve:cloud0"];
    expect(job?.state).toBe("error");
    expect(job?.error).toContain("host unreachable");
  });

  it("dismisses only the job named", async () => {
    useAppStore.setState({
      sourceJobs: {
        "resolve:cloud0": status({ state: "error" }),
        "resolve:cloud1": status({ source_id: "cloud1", state: "done" }),
      },
    });
    useAppStore.getState().dismissSourceJob("resolve", "cloud0");
    expect(Object.keys(useAppStore.getState().sourceJobs)).toEqual(["resolve:cloud1"]);
  });

  it("a failed cancel leaves the job alone", async () => {
    // Only the server can stop the recall, and it re-reports the flag on the
    // next poll -- a failed cancel means the button has not taken yet, which is
    // not something to tell the user about.
    useAppStore.setState({
      client: stubClient({ cancelJob: vi.fn().mockRejectedValue(new Error("nope")) }),
      sourceJobs: { "resolve:cloud0": status() },
    });
    await useAppStore.getState().cancelSourceJob("resolve", "cloud0");
    expect(useAppStore.getState().sourceJobs["resolve:cloud0"]?.state).toBe("running");
  });

  it("re-reads the catalog when a resolve lands", async () => {
    // The row still lists the pre-resolve tensors the moment a resolve lands.
    const listSources = vi.fn().mockResolvedValue([]);
    const client = {
      listSources,
      http: {
        startResolve: vi.fn().mockResolvedValue(status()),
        jobStatus: vi.fn().mockResolvedValue(status({ state: "done" })),
      },
    } as unknown as TensorFlightClient;
    useAppStore.setState({ client });

    await useAppStore.getState().startResolve("cloud0");
    await vi.waitFor(
      () =>
        expect(useAppStore.getState().sourceJobs["resolve:cloud0"]?.state).toBe(
          "done",
        ),
      { timeout: 3000 },
    );
    expect(listSources).toHaveBeenCalled();
  });

  describe("opening the source a resolve finished", () => {
    const realOpenTensor = useAppStore.getState().openTensor;
    afterEach(() => useAppStore.setState({ openTensor: realOpenTensor }));

    const resolveClient = (final: SourceJobStatus, first = status()) =>
      ({
        listSources: vi.fn().mockResolvedValue([]),
        http: {
          startResolve: vi.fn().mockResolvedValue(first),
          jobStatus: vi.fn().mockResolvedValue(final),
        },
      }) as unknown as TensorFlightClient;

    const landed = () =>
      vi.waitFor(
        () =>
          expect(useAppStore.getState().sourceJobs["resolve:cloud0"]?.state).toBe(
            "done",
          ),
        { timeout: 3000 },
      );

    it("opens the bare source when nothing else was opened meanwhile", async () => {
      const openTensor = vi.fn();
      useAppStore.setState({ openTensor, client: resolveClient(status({ state: "done" })) });
      await useAppStore.getState().startResolve("cloud0");
      await landed();
      expect(openTensor).toHaveBeenCalledWith("cloud0");
    });

    it("leaves the user's newer selection alone", async () => {
      const openTensor = vi.fn();
      useAppStore.setState({ openTensor, client: resolveClient(status({ state: "done" })) });
      await useAppStore.getState().startResolve("cloud0");
      // Something else is opened while the resolve runs: the epoch moves on.
      const { target } = useAppStore.getState();
      useAppStore.setState({ target: { ...target, epoch: target.epoch + 1 } });
      await landed();
      expect(openTensor).not.toHaveBeenCalled();
    });

    it("opens nothing for a resolve that failed", async () => {
      const openTensor = vi.fn();
      useAppStore.setState({
        openTensor,
        client: resolveClient(status({ state: "error", error: "boom" })),
      });
      await useAppStore.getState().startResolve("cloud0");
      await vi.waitFor(
        () =>
          expect(useAppStore.getState().sourceJobs["resolve:cloud0"]?.state).toBe(
            "error",
          ),
        { timeout: 3000 },
      );
      expect(openTensor).not.toHaveBeenCalled();
    });

    it("settles a resolve that was already done when it started", async () => {
      const openTensor = vi.fn();
      const done = status({ state: "done", started: true });
      useAppStore.setState({ openTensor, client: resolveClient(done, done) });
      await useAppStore.getState().startResolve("cloud0");
      expect(openTensor).toHaveBeenCalledWith("cloud0");
    });

    it("counts an open made while the start request was in flight", async () => {
      const openTensor = vi.fn();
      const client = resolveClient(status({ state: "done" }));
      (client.http.startResolve as ReturnType<typeof vi.fn>).mockImplementation(() => {
        const { target } = useAppStore.getState();
        useAppStore.setState({ target: { ...target, epoch: target.epoch + 1 } });
        return Promise.resolve(status());
      });
      useAppStore.setState({ openTensor, client });
      await useAppStore.getState().startResolve("cloud0");
      await landed();
      expect(openTensor).not.toHaveBeenCalled();
    });

    it("does not open for a resolve it only joined", async () => {
      const openTensor = vi.fn();
      useAppStore.setState({
        openTensor,
        client: resolveClient(status({ state: "done" }), status({ started: false })),
      });
      await useAppStore.getState().startResolve("cloud0");
      await landed();
      expect(openTensor).not.toHaveBeenCalled();
    });
  });
});

describe("the label overlay", () => {
  it("is drawn only when it belongs to the tensor in view", () => {
    useAppStore.setState({
      ...viewOf("src0"),
      labelOverlay: "src0/@labels/nuclei",
    });
    expect(selectLabelOverlay(useAppStore.getState())).toBe("src0/@labels/nuclei");

    useAppStore.setState({ ...viewOf("src1"), labelOverlay: "src0/@labels/nuclei" });
    expect(selectLabelOverlay(useAppStore.getState())).toBeNull();
  });

  it("survives a round trip through another tensor", () => {
    // Not reset on selection, for the reason the annotation state is not: the
    // id names its own image, so the scoping selector is enough.
    useAppStore.setState({ ...viewOf("src0"), labelOverlay: "src0/@labels/nuclei" });
    useAppStore.setState({ client: null });
    useAppStore.getState().openTensor("src1");
    expect(selectLabelOverlay(useAppStore.getState())).toBeNull();
    // A click leaves the overlay's id alone; a link replaces it.
    expect(useAppStore.getState().labelOverlay).toBe("src0/@labels/nuclei");
    view("src0");
    expect(selectLabelOverlay(useAppStore.getState())).toBe("src0/@labels/nuclei");
  });

  it("matches a content-pinned link to the set of its stable id", () => {
    useAppStore.setState({
      ...viewOf("src0", { requested: "src0@abcd1234" }),
      labelOverlay: "src0/@labels/nuclei",
    });
    expect(selectLabelOverlay(useAppStore.getState())).toBe("src0/@labels/nuclei");
  });

  it("refuses an id that names no set at all", () => {
    useAppStore.setState({
      ...viewOf("src0"),
      labelOverlay: "src0",
    });
    expect(selectLabelOverlay(useAppStore.getState())).toBeNull();
  });

  it("clamps the opacity to what the shader can take", () => {
    const { setLabelOpacity } = useAppStore.getState();
    setLabelOpacity(2);
    expect(useAppStore.getState().labelOpacity).toBe(1);
    setLabelOpacity(-1);
    expect(useAppStore.getState().labelOpacity).toBe(0);
    setLabelOpacity(Number.NaN);
    expect(useAppStore.getState().labelOpacity).toBe(DEFAULT_LABEL_OPACITY);
  });

  it("opens a link's overlay, and clears one the link does not name", () => {
    useAppStore.setState({ labelOverlay: "old/@labels/x", labelOpacity: 0.5 });
    useAppStore.getState().applyViewerState(
      new URLSearchParams("id=src0&lb=src0%2F%40labels%2Fnuclei&lo=0.25"),
    );
    expect(useAppStore.getState().labelOverlay).toBe("src0/@labels/nuclei");
    expect(useAppStore.getState().labelOpacity).toBe(0.25);

    useAppStore.getState().applyViewerState(new URLSearchParams("id=src0"));
    expect(useAppStore.getState().labelOverlay).toBeNull();
    // The alpha is a preference about the overlay rather than about a set, so
    // it carries across the way gamma and the percentile window do.
    expect(useAppStore.getState().labelOpacity).toBe(0.25);
  });
});
