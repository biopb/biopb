import { afterEach, describe, expect, it, vi } from "vitest";
import type { DataSourceDescriptor, TensorFlightClient, TileInfo } from "@biopb/tensor-flight-client";
import {
  selectHiddenSets,
  selectRois,
  selectRoisError,
  selectRoisLoading,
  selectRoisSkipped,
  selectRoisTruncated,
  selectDraft,
  selectSelectedRoi,
  selectTileInfo,
  useAppStore,
} from "./store";

const SOURCE: DataSourceDescriptor = {
  source_id: "listed",
  source_url: "file:///listed",
  source_type: "file",
  metadata_json: null,
  tensors: [],
};

const BASE_SLICE = {
  t: 0,
  z: 0,
  c: 0,
  axes: {},
  contrastMode: "auto" as const,
  percentileScale: 1,
  fixedLimits: null,
  gamma: 1,
};

/** Only its identity is under test here; the extents just have to be readable. */
const TILE_INFO = {
  array_id: "first@abcd1234",
  dim_labels: ["T", "C", "Z", "Y", "X"],
  shape: [1, 1, 1, 8, 8],
} as unknown as TileInfo;

const client = (sources: DataSourceDescriptor[]) =>
  ({
    listSources: vi.fn().mockResolvedValue(sources),
    http: { readyz: vi.fn().mockResolvedValue({ backend_health: {} }) },
  }) as unknown as TensorFlightClient;

afterEach(() => {
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
      activeSourceId: "shared-source",
      activeTensorId: "shared-source/Image:0",
      requestedArrayId: "shared-source/Image:0",
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
      activeSourceId: SOURCE.source_id,
      activeTensorId: SOURCE.source_id,
      requestedArrayId: null,
    });

    useAppStore.getState().startCatalogPolling();
    await vi.advanceTimersByTimeAsync(60000);

    expect(useAppStore.getState().activeSourceId).toBeNull();
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
      slice: { ...BASE_SLICE, t: 5, z: 6, c: 1, axes: { a3: 2 }, percentileScale: 2, gamma: 1.6 },
    });

    expect(useAppStore.getState().applyViewerState(new URLSearchParams("id=second"))).toBe(true);
    const s = useAppStore.getState();
    expect(s.render3d).toBe(false);
    expect(s.slice).toMatchObject({ t: 0, z: 0, c: 0, axes: {} });
    // Preferences, not properties of the tensor -- selectSource carries these
    // across a change too.
    expect(s.slice).toMatchObject({ percentileScale: 2, gamma: 1.6 });
  });

  it("keeps what the link names", () => {
    useAppStore.setState({ activeTensorId: "first", render3d: false, slice: { ...BASE_SLICE, t: 5 } });

    expect(useAppStore.getState().applyViewerState(new URLSearchParams("id=second&t=9&v=1"))).toBe(true);
    expect(useAppStore.getState().slice.t).toBe(9);
    expect(useAppStore.getState().render3d).toBe(true);
  });
});

describe("the grid in view", () => {
  it("hides a grid fetched for another id", () => {
    useAppStore.getState().setTileInfo(TILE_INFO, "first");
    useAppStore.setState({ activeTensorId: "second", requestedArrayId: null });

    expect(selectTileInfo(useAppStore.getState())).toBeNull();
  });

  // The point of pairing on the *requested* id rather than the one tile_info
  // answers with: those differ by design (a version token, and the field a bare
  // source_id resolves to), so comparing the answer would never match.
  it("keeps a grid whose id was asked for, however it answers", () => {
    useAppStore.getState().setTileInfo(TILE_INFO, "first");
    useAppStore.setState({ activeTensorId: "first", requestedArrayId: null });

    expect(selectTileInfo(useAppStore.getState())).toBe(TILE_INFO);
  });

  it("pairs against the requested address when a link pinned one", () => {
    useAppStore.getState().setTileInfo(TILE_INFO, "first@abcd1234");
    useAppStore.setState({ activeTensorId: "first", requestedArrayId: "first@abcd1234" });

    expect(selectTileInfo(useAppStore.getState())).toBe(TILE_INFO);
  });
});

// ---------------------------------------------------------------------------
// ROI annotation state is scoped to the tensor in view
//
// `selectSource` is not the only way that tensor changes -- `applyViewerState`
// does it straight from a URL, without going through it -- so these are read
// through selectors rather than reset at each writer. Everything below is a
// leak that a reset written in `selectSource` alone would not have caught.
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
  useAppStore.setState({
    activeSourceId: "first",
    activeTensorId: "first",
    requestedArrayId: null,
    rois: ROI_FIXTURE,
    roisFor: "first",
    roisTruncated: true,
    roisSkipped: 3,
    roisError: "boom",
    roisErrorFor: "first",
    hiddenSets: ["nuclei"],
    hiddenSetsFor: "first",
  });
}

describe("ROI state across a tensor change", () => {
  it("shows a tensor its own annotations", () => {
    seedAnnotated();
    const s = useAppStore.getState();
    expect(selectRois(s)).toHaveLength(1);
    expect(selectRoisTruncated(s)).toBe(true);
    expect(selectRoisSkipped(s)).toBe(3);
    expect(selectRoisError(s)).toBe("boom");
    expect(selectHiddenSets(s)).toEqual(["nuclei"]);
  });

  it("hides all of it once another tensor is selected", () => {
    seedAnnotated();
    useAppStore.getState().selectSource("second");
    const s = useAppStore.getState();
    expect(selectRois(s)).toEqual([]);
    expect(selectRoisTruncated(s)).toBe(false);
    expect(selectRoisSkipped(s)).toBe(0);
    expect(selectRoisError(s)).toBeNull();
    expect(selectHiddenSets(s)).toEqual([]);
  });

  it("hides all of it when a LINK changes the tensor", () => {
    // The path a reset in `selectSource` would miss entirely.
    seedAnnotated();
    useAppStore.getState().applyViewerState(new URLSearchParams({ id: "second/Image:0" }));
    const s = useAppStore.getState();
    expect(selectRois(s)).toEqual([]);
    expect(selectHiddenSets(s)).toEqual([]);
    // And the raw fields are untouched, which is the point: nothing cleared
    // them, so reading them directly is what the leak was.
    expect(s.hiddenSets).toEqual(["nuclei"]);
    expect(s.roisTruncated).toBe(true);
  });

  it("does not carry a warning onto a tensor it is not about", () => {
    // The worst of the leaks: "this tensor holds more than is shown" reads as a
    // fact about the image on screen.
    seedAnnotated();
    useAppStore.getState().applyViewerState(new URLSearchParams({ id: "second" }));
    const s = useAppStore.getState();
    expect(selectRoisTruncated(s)).toBe(false);
    expect(selectRoisSkipped(s)).toBe(0);
    expect(selectRoisError(s)).toBeNull();
  });

  it("follows a content-pinned link to the same source", () => {
    // `requestedArrayId` is the address in view when a link pins one, so the
    // guard has to use it -- not `activeTensorId`.
    seedAnnotated();
    useAppStore.getState().applyViewerState(new URLSearchParams({ id: "first@9f1c4e2b" }));
    expect(selectRois(useAppStore.getState())).toEqual([]);
    useAppStore.setState({ rois: ROI_FIXTURE, roisFor: "first@9f1c4e2b" });
    expect(selectRois(useAppStore.getState())).toHaveLength(1);
  });

  it("starts a fresh hidden list rather than editing another tensor's", () => {
    seedAnnotated();
    useAppStore.getState().selectSource("second");
    useAppStore.getState().toggleSetHidden("debris");
    expect(selectHiddenSets(useAppStore.getState())).toEqual(["debris"]);
  });

  it("reports loading only for the tensor in view", () => {
    useAppStore.setState({
      activeTensorId: "first",
      requestedArrayId: null,
      roisPending: "second",
    });
    expect(selectRoisLoading(useAppStore.getState())).toBe(false);
    useAppStore.setState({ roisPending: "first" });
    expect(selectRoisLoading(useAppStore.getState())).toBe(true);
  });
});

describe("loadRois", () => {
  /** A client whose listRois records what it was asked for. */
  function stubClient(asked: string[]) {
    return {
      http: {
        listRois: (arrayId: string) => {
          asked.push(arrayId);
          return Promise.resolve({ rois: [], truncated: false, skipped: 0 });
        },
      },
    } as unknown as TensorFlightClient;
  }

  it("fetches the tensor in view", async () => {
    const asked: string[] = [];
    useAppStore.setState({
      client: stubClient(asked),
      activeTensorId: "first",
      requestedArrayId: null,
      roisFor: null,
      roisPending: null,
      roisUnavailable: false,
    });
    await useAppStore.getState().loadRois("first");
    expect(asked).toEqual(["first"]);
    expect(useAppStore.getState().roisFor).toBe("first");
  });

  it("refuses a tensor that is not in view", async () => {
    // Nothing could display it: every selector hides a set whose tensor is not
    // the current one, so the round trip would be pure waste.
    const asked: string[] = [];
    useAppStore.setState({
      client: stubClient(asked),
      activeTensorId: "first",
      requestedArrayId: null,
      roisFor: null,
      roisPending: null,
      roisUnavailable: false,
    });
    await useAppStore.getState().loadRois("second");
    expect(asked).toEqual([]);
  });

  it("follows the pinned address when a link named one", async () => {
    const asked: string[] = [];
    useAppStore.setState({
      client: stubClient(asked),
      activeTensorId: "first",
      requestedArrayId: "first@9f1c4e2b",
      roisFor: null,
      roisPending: null,
      roisUnavailable: false,
    });
    await useAppStore.getState().loadRois("first");
    expect(asked).toEqual([]);
    await useAppStore.getState().loadRois("first@9f1c4e2b");
    expect(asked).toEqual(["first@9f1c4e2b"]);
  });

  it("asks once for a set it already holds", async () => {
    const asked: string[] = [];
    useAppStore.setState({
      client: stubClient(asked),
      activeTensorId: "first",
      requestedArrayId: null,
      roisFor: null,
      roisPending: null,
      roisUnavailable: false,
    });
    await useAppStore.getState().loadRois("first");
    await useAppStore.getState().loadRois("first");
    expect(asked).toEqual(["first"]);
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
      activeTensorId: "first",
      requestedArrayId: null,
      rois: [],
      roisFor: "first",
      selectedRoiId: null,
      roiWriteError: null,
      newLabel: "",
      newSetName: "",
      draft: null,
      draftFor: null,
      render3d: false,
      broadcastAxes: null,
      broadcastAxesFor: null,
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
    expect(useAppStore.getState().rois.map((r) => r.roiId)).toEqual(["srv1"]);
    expect(useAppStore.getState().selectedRoiId).toBe("srv1");
  });

  it("drops a write that landed after the tensor moved on", async () => {
    // The annotation is stored; appending it would draw it over another image.
    seedFor({
      http: {
        putRois: () => {
          useAppStore.setState({ activeTensorId: "second", roisFor: "second", rois: [] });
          return Promise.resolve({ stored: [stored()], conflicts: [], skipped: 0 });
        },
      },
    });
    await useAppStore.getState().createRoi(GEOM, {});
    expect(useAppStore.getState().rois).toEqual([]);
  });

  it("reports a failed write instead of pretending it landed", async () => {
    seedFor({ http: { putRois: () => Promise.reject(new Error("422 rejected")) } });
    await useAppStore.getState().createRoi(GEOM, {});
    expect(useAppStore.getState().roiWriteError).toContain("422 rejected");
    expect(useAppStore.getState().rois).toEqual([]);
  });

  it("removes a deleted annotation and clears the selection with it", async () => {
    seedFor({ http: { deleteRois: () => Promise.resolve(["srv1"]) } });
    useAppStore.setState({ rois: [stored()], selectedRoiId: "srv1" });
    await useAppStore.getState().deleteRoi("srv1");
    expect(useAppStore.getState().rois).toEqual([]);
    expect(useAppStore.getState().selectedRoiId).toBeNull();
  });

  it("drops a row the server says it never had, rather than leaving it stuck", async () => {
    seedFor({ http: { deleteRois: () => Promise.resolve([]) } });
    useAppStore.setState({ rois: [stored()], selectedRoiId: "srv1" });
    await useAppStore.getState().deleteRoi("srv1");
    expect(useAppStore.getState().rois).toEqual([]);
  });

  it("abandons the draft when the tool changes", () => {
    useAppStore.setState({ tool: "polygon", draft: { tool: "polygon", points: [[0, 0]] } });
    useAppStore.getState().setTool("rectangle");
    expect(useAppStore.getState().draft).toBeNull();
  });

  it("materialises the default before the first broadcast toggle", () => {
    // Otherwise toggling one axis would silently also pin whatever the default
    // was broadcasting.
    seedFor({ http: {} });
    useAppStore.getState().toggleBroadcastAxis(2, [1]);
    expect(useAppStore.getState().broadcastAxes?.sort()).toEqual([1, 2]);
  });

  it("hides the draft in 3-D and on another tensor", () => {
    seedFor({ http: {} });
    useAppStore.getState().setDraft({ tool: "polygon", points: [[0, 0]] });
    expect(selectDraft(useAppStore.getState())).not.toBeNull();

    useAppStore.setState({ render3d: true });
    expect(selectDraft(useAppStore.getState())).toBeNull();

    useAppStore.setState({ render3d: false, activeTensorId: "second" });
    expect(selectDraft(useAppStore.getState())).toBeNull();
  });

  it("resolves the selection against the set in view, so it cannot go stale", () => {
    seedFor({ http: {} });
    useAppStore.setState({ rois: [stored()], selectedRoiId: "srv1" });
    expect(selectSelectedRoi(useAppStore.getState())?.roiId).toBe("srv1");

    // The annotation is gone from the set: no cleanup needed anywhere.
    useAppStore.setState({ rois: [] });
    expect(selectSelectedRoi(useAppStore.getState())).toBeNull();
  });
});
