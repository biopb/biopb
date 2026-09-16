import { afterEach, describe, expect, it, vi } from "vitest";
import { TensorApiError } from "@biopb/tensor-flight-client";
import type {
  DataSourceDescriptor,
  RoiAnnotation,
  RoiSetInfo,
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
  selectObservedLimits,
  selectTileInfo,
  catalogFingerprint,
  useAppStore,
} from "./store";

const SOURCE: DataSourceDescriptor = {
  source_id: "listed",
  source_url: "file:///listed",
  source_type: "file",
  metadata_json: null,
  data_resident: true,
  is_resolved: true,
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

describe("the levels the data has shown", () => {
  const at = (channel: number) =>
    useAppStore.setState({ slice: { ...BASE_SLICE, c: channel } });

  it("widens to cover every plane sampled, not just the last one", () => {
    useAppStore.setState({ activeTensorId: "first", requestedArrayId: null });
    useAppStore.getState().noteObservedLimits([12, 4000], "first", 0);
    useAppStore.getState().noteObservedLimits([0, 3], "first", 0);
    at(0);

    // Without the union a fixed window chosen on the bright plane could not be
    // widened again while the dark one is in view.
    expect(selectObservedLimits(useAppStore.getState())).toEqual([0, 4000]);
  });

  it("keeps each channel on its own scale", () => {
    useAppStore.setState({ activeTensorId: "first", requestedArrayId: null });
    useAppStore.getState().noteObservedLimits([12, 4000], "first", 0);
    useAppStore.getState().noteObservedLimits([0, 1], "first", 1);
    at(1);

    expect(selectObservedLimits(useAppStore.getState())).toEqual([0, 1]);
  });

  it("starts over on another tensor rather than widening across two", () => {
    useAppStore.getState().noteObservedLimits([12, 4000], "first", 0);
    useAppStore.getState().noteObservedLimits([0, 1], "second", 0);
    useAppStore.setState({ activeTensorId: "second", requestedArrayId: null });
    at(0);

    expect(selectObservedLimits(useAppStore.getState())).toEqual([0, 1]);
  });

  it("hides a union sampled from another tensor", () => {
    useAppStore.getState().noteObservedLimits([12, 4000], "first", 0);
    useAppStore.setState({ activeTensorId: "second", requestedArrayId: null });
    at(0);

    expect(selectObservedLimits(useAppStore.getState())).toBeNull();
  });

  it("does not write when the plane adds nothing", () => {
    useAppStore.setState({ activeTensorId: "first", requestedArrayId: null });
    useAppStore.getState().noteObservedLimits([0, 4000], "first", 0);
    const before = useAppStore.getState().observedLimits;
    useAppStore.getState().noteObservedLimits([10, 900], "first", 0);

    // The viewers publish this from a memo on every render; a fresh identity
    // each time would loop through the effect that writes it.
    expect(useAppStore.getState().observedLimits).toBe(before);
  });
});

describe("the contrast track", () => {
  it("hides a track published for another tensor", () => {
    // The reason it is guarded rather than reset: a viewer that errored out
    // after publishing leaves its track behind, and a stale one would draw the
    // panel's bar on the previous tensor's grey levels.
    useAppStore.getState().setContrastTrack([0, 4000], "first");
    useAppStore.setState({ activeTensorId: "second", requestedArrayId: null });

    expect(selectContrastTrack(useAppStore.getState())).toBeNull();
  });

  it("hands the panel the track the viewer derived", () => {
    // The whole point of publishing it (biopb/biopb#955): one deriver, so the
    // bar cannot be drawn on a track the shader is not clamping into.
    useAppStore.setState({ activeTensorId: "first", requestedArrayId: null });
    useAppStore.getState().setContrastTrack([12.5, 4000], "first");

    expect(selectContrastTrack(useAppStore.getState())).toEqual([12.5, 4000]);
  });

  it("does not write when the track is unchanged", () => {
    useAppStore.setState({ activeTensorId: "first", requestedArrayId: null });
    useAppStore.getState().setContrastTrack([0, 4000], "first");
    const before = useAppStore.getState().contrastTrack;
    useAppStore.getState().setContrastTrack([0, 4000], "first");

    // The viewer derives this in a memo on every render; a fresh identity each
    // time would loop through the effect that publishes it.
    expect(useAppStore.getState().contrastTrack).toBe(before);
  });

  it("writes an equal track published for a different tensor", () => {
    // Two tensors can share a dtype and so a track. The id has to move anyway,
    // or the selector would go on hiding it.
    useAppStore.getState().setContrastTrack([0, 65535], "first");
    useAppStore.getState().setContrastTrack([0, 65535], "second");
    useAppStore.setState({ activeTensorId: "second", requestedArrayId: null });

    expect(selectContrastTrack(useAppStore.getState())).toEqual([0, 65535]);
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
    roiScopes: { "": { truncated: true, skipped: 3 } },
    roisError: "boom",
    roisErrorFor: "first",
    visibleSets: ["nuclei"],
    visibleSetsFor: "first",
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
    useAppStore.getState().selectSource("second");
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
    expect(s.rois).toHaveLength(1);
    expect(s.roiScopes[""]?.truncated).toBe(true);
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

  it("follows a content-pinned link to the same source", () => {
    // `requestedArrayId` is the address in view when a link pins one, so the
    // guard has to use it -- not `activeTensorId`.
    seedAnnotated();
    useAppStore.getState().applyViewerState(new URLSearchParams({ id: "first@9f1c4e2b" }));
    expect(selectRois(useAppStore.getState())).toEqual([]);
    useAppStore.setState({ rois: ROI_FIXTURE, roisFor: "first@9f1c4e2b" });
    expect(selectRois(useAppStore.getState())).toHaveLength(1);
  });

  it("starts from the new tensor's default rather than editing another tensor's list", () => {
    seedAnnotated();
    useAppStore.getState().selectSource("second");
    useAppStore.getState().toggleSetVisible("debris");
    expect(selectVisibleSets(useAppStore.getState())).toEqual(["debris"]);
  });

  it("reports loading only for the tensor in view", () => {
    useAppStore.setState({
      activeTensorId: "first",
      requestedArrayId: null,
      roisPendingFor: "second",
      roisPending: [""],
    });
    expect(selectRoisPending(useAppStore.getState())).toEqual([]);
    useAppStore.setState({ roisPendingFor: "first" });
    expect(selectRoisPending(useAppStore.getState())).toEqual([""]);
  });
});

describe("the sets on screen", () => {
  const SETS = [
    { setName: "nuclei", count: 1, reserved: false },
    { setName: "cells", count: 2, reserved: false },
    { setName: "@ome", count: 120, reserved: true },
  ];

  function seedListed() {
    useAppStore.setState({
      activeTensorId: "first",
      requestedArrayId: null,
      rois: ROI_FIXTURE,
      roiSets: SETS,
      roisFor: "first",
      roiScopes: { "": { truncated: false, skipped: 0 } },
      visibleSets: null,
      visibleSetsFor: null,
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

  it("adopts the sets a link names, and turns the overlay on for them", () => {
    seedListed();
    useAppStore.setState({ showRois: false });
    useAppStore.getState().applyViewerState(new URLSearchParams("id=first&rs=@ome"));
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
    useAppStore.setState((s) => {
      const scopes = { ...s.roiScopes };
      delete scopes[""];
      return { roiScopes: scopes };
    });
  }

  function fresh(client: TensorFlightClient, over: Record<string, unknown> = {}) {
    useAppStore.setState({
      client,
      activeTensorId: "first",
      requestedArrayId: null,
      rois: [],
      roiSets: [],
      roisFor: null,
      roiScopes: {},
      roisPending: [],
      roisPendingFor: null,
      roisUnavailable: false,
      visibleSets: null,
      visibleSetsFor: null,
      ...over,
    });
  }

  it("fetches the tensor in view", async () => {
    const asked: string[] = [];
    fresh(stubClient(asked));
    await useAppStore.getState().loadRois("first");
    expect(asked).toEqual(["first"]);
    expect(useAppStore.getState().roisFor).toBe("first");
    expect(selectRoiScopes(useAppStore.getState())).toEqual({ "": { truncated: false, skipped: 0 } });
  });

  it("refuses a tensor that is not in view", async () => {
    // Nothing could display it: every selector hides a set whose tensor is not
    // the current one, so the round trip would be pure waste.
    const asked: string[] = [];
    fresh(stubClient(asked));
    await useAppStore.getState().loadRois("second");
    expect(asked).toEqual([]);
  });

  it("follows the pinned address when a link named one", async () => {
    const asked: string[] = [];
    fresh(stubClient(asked), { requestedArrayId: "first@9f1c4e2b" });
    await useAppStore.getState().loadRois("first");
    expect(asked).toEqual([]);
    await useAppStore.getState().loadRois("first@9f1c4e2b");
    expect(asked).toEqual(["first@9f1c4e2b"]);
  });

  it("asks once for a set it already holds", async () => {
    const asked: string[] = [];
    fresh(stubClient(asked));
    await useAppStore.getState().loadRois("first");
    await useAppStore.getState().loadRois("first");
    expect(asked).toEqual(["first"]);
  });

  it("does not fetch a server-owned set until it is on screen", async () => {
    // The lazy half: the listing names it, the rows wait for the toggle.
    const asked: string[] = [];
    fresh(stubClient(asked, () => ({ sets: [{ setName: "@ome", count: 120, reserved: true }] })));
    await useAppStore.getState().loadRois("first");
    expect(asked).toEqual(["first"]);
    useAppStore.getState().toggleSetVisible("@ome");
    await useAppStore.getState().loadRois("first");
    expect(asked).toEqual(["first", "first?@ome"]);
    expect(Object.keys(selectRoiScopes(useAppStore.getState()))).toEqual(["", "@ome"]);
  });

  it("fetches a server-owned set a link names alongside the listing, not after it", async () => {
    // Before any listing has landed the prefix says the set needs its own
    // fetch, so a link straight to it does not wait a round trip to find out.
    const asked: string[] = [];
    fresh(stubClient(asked), { visibleSets: ["@ome"], visibleSetsFor: "first" });
    await useAppStore.getState().loadRois("first");
    expect(asked).toEqual(["first", "first?@ome"]);
  });

  it("does not fetch a named set the listing says is not there", async () => {
    const asked: string[] = [];
    fresh(stubClient(asked, () => ({ sets: [{ setName: "@ome", count: 1, reserved: true }] })));
    await useAppStore.getState().loadRois("first");
    useAppStore.setState({ visibleSets: ["@gone"], visibleSetsFor: "first" });
    await useAppStore.getState().loadRois("first");
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
    fresh(client, { visibleSets: ["nuclei", "@ome"], visibleSetsFor: "first" });
    await useAppStore.getState().loadRois("first");
    expect(useAppStore.getState().rois.map((r) => r.roiId).sort()).toEqual(["n1", "o1"]);
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
    fresh(client, { visibleSets: ["@ome"], visibleSetsFor: "first" });
    await useAppStore.getState().loadRois("first");
    expect(Object.keys(selectRoiScopes(useAppStore.getState())).sort()).toEqual(["", "@ome"]);

    // Something re-registered the source; the listing is fetched again for
    // whatever reason and now says two rows.
    omeCount = 2;
    unlandListing();
    await useAppStore.getState().loadRois("first");
    expect(selectRoiScopes(useAppStore.getState())["@ome"]).toBeUndefined();
    await useAppStore.getState().loadRois("first");
    expect(asked.filter((a) => a.endsWith("?@ome"))).toHaveLength(2);
  });

  it("counts a row it could not read as held, rather than as someone else's change", async () => {
    const client = stubClient([], (setName) => ({
      rois: setName ? [OME_ROW] : [],
      sets: [{ setName: "@ome", count: 2, reserved: true }],
      skipped: setName ? 1 : 0,
    }));
    fresh(client, { visibleSets: ["@ome"], visibleSetsFor: "first" });
    await useAppStore.getState().loadRois("first");
    unlandListing();
    await useAppStore.getState().loadRois("first");
    expect(selectRoiScopes(useAppStore.getState())["@ome"]).toEqual({ truncated: false, skipped: 1 });
  });

  it("does not un-land a set the cap clipped, which disagrees by construction", async () => {
    const client = stubClient([], (setName) => ({
      rois: setName ? [OME_ROW] : [],
      sets: [{ setName: "@ome", count: 5000, reserved: true }],
      truncated: setName !== undefined,
    }));
    fresh(client, { visibleSets: ["@ome"], visibleSetsFor: "first" });
    await useAppStore.getState().loadRois("first");
    unlandListing();
    await useAppStore.getState().loadRois("first");
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
    const load = useAppStore.getState().loadRois("first");
    useAppStore.getState().selectSource("second");
    resolve({ rois: [ROI_FIXTURE[0]], sets: [], truncated: false, skipped: 0 });
    await load;
    expect(useAppStore.getState().roisFor).toBe("first");
    expect(selectRois(useAppStore.getState())).toEqual([]);
    useAppStore.getState().selectSource("first");
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
          seen = useAppStore.getState().rois;
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
    expect(useAppStore.getState().rois.map((r) => r.roiId)).toEqual(["srv1"]);
    expect(useAppStore.getState().selectedRoiId).toBe("srv1");
  });

  it("takes the provisional row back out when the write fails", async () => {
    seedFor({ http: { putRois: () => Promise.reject(new Error("422 rejected")) } });
    await useAppStore.getState().createRoi(GEOM, {});
    expect(useAppStore.getState().rois).toEqual([]);
    expect(useAppStore.getState().roiWriteError).toContain("422 rejected");
  });

  it("puts a provisional row in the set its stored form will land in", async () => {
    // The server substitutes "default" for an empty set name, so a blank one
    // here would show a nameless set row for the length of a round trip.
    let seen: RoiAnnotation[] = [];
    seedFor({
      http: {
        putRois: () => {
          seen = useAppStore.getState().rois;
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
    useAppStore.setState({ rois: [provisional] });
    await useAppStore.getState().deleteRoi("pending,1");
    expect(asked).toBe(false);
    expect(useAppStore.getState().rois).toHaveLength(1);
  });

  it("removes a deleted annotation on the keystroke, not on the answer", async () => {
    let seen: RoiAnnotation[] = [];
    seedFor({
      http: {
        deleteRois: () => {
          seen = useAppStore.getState().rois;
          return Promise.resolve(["srv1"]);
        },
      },
    });
    useAppStore.setState({ rois: [stored()], selectedRoiId: "srv1" });
    await useAppStore.getState().deleteRoi("srv1");
    // Gone before the request was made, not after it came back.
    expect(seen).toEqual([]);
  });

  it("puts a row back where it was when the delete fails", async () => {
    // Not at the end: the overlay draws later annotations over earlier ones, so
    // a failed delete must not reorder what covers what.
    seedFor({ http: { deleteRois: () => Promise.reject(new Error("503 upstream")) } });
    useAppStore.setState({
      rois: [stored({ roiId: "a" }), stored({ roiId: "b" }), stored({ roiId: "c" })],
    });
    await useAppStore.getState().deleteRoi("b");
    expect(useAppStore.getState().rois.map((r) => r.roiId)).toEqual(["a", "b", "c"]);
    expect(useAppStore.getState().roiWriteError).toContain("503 upstream");
  });

  it("does not restore a failed delete into another tensor's list", async () => {
    seedFor({
      http: {
        deleteRois: () => {
          useAppStore.setState({ activeTensorId: "second", roisFor: "second", rois: [] });
          return Promise.reject(new Error("503 upstream"));
        },
      },
    });
    useAppStore.setState({ rois: [stored()] });
    await useAppStore.getState().deleteRoi("srv1");
    expect(useAppStore.getState().rois).toEqual([]);
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
    useAppStore.setState({
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
    expect(useAppStore.getState().rois.map((r) => r.roiId)).toEqual(["srv3"]);
  });

  it("drops a selection that was in the cleared set, and the set from the chosen list", async () => {
    // The name means nothing once the set is gone, and leaving it on the list
    // would pin a later set of the same name to this one's toggle.
    seedFor({ http: { deleteRois: () => Promise.resolve(["srv1"]) } });
    useAppStore.setState({
      rois: [stored({ roiId: "srv1", setName: "nuclei" }), stored({ roiId: "srv2", setName: "cells" })],
      roiSets: [
        { setName: "nuclei", count: 1, reserved: false },
        { setName: "cells", count: 1, reserved: false },
      ],
      selectedRoiId: "srv1",
      visibleSets: ["nuclei", "cells"],
      visibleSetsFor: "first",
    });
    await useAppStore.getState().clearRoiSet("nuclei");
    expect(useAppStore.getState().selectedRoiId).toBeNull();
    expect(useAppStore.getState().visibleSets).toEqual(["cells"]);
    expect(useAppStore.getState().roiSets.map((set) => set.setName)).toEqual(["cells"]);
  });

  it("puts a new annotation's set on screen when the chosen list would hide it", async () => {
    // A materialised list is exactly the sets shown, so a set switched off --
    // or one that did not exist a moment ago -- would swallow the shape the
    // user just traced.
    seedFor({ http: { putRois: () => Promise.resolve({ stored: [stored({ setName: "fresh" })], conflicts: [], skipped: 0 }) } });
    useAppStore.setState({
      newSetName: "fresh",
      roiSets: [],
      visibleSets: ["nuclei"],
      visibleSetsFor: "first",
    });
    await useAppStore.getState().createRoi(GEOM, {});
    expect(useAppStore.getState().visibleSets).toEqual(["nuclei", "fresh"]);
    // And the listing counts it, so a later listing does not read the row as
    // someone else's change.
    expect(useAppStore.getState().roiSets).toEqual([{ setName: "fresh", count: 1, reserved: false }]);
  });

  it("moves the listing's count with a delete, and back when it fails", async () => {
    seedFor({ http: { deleteRois: () => Promise.reject(new Error("500 upstream")) } });
    useAppStore.setState({
      rois: [stored({ roiId: "srv1", setName: "nuclei" })],
      roiSets: [{ setName: "nuclei", count: 1, reserved: false }],
    });
    const done = useAppStore.getState().deleteRoi("srv1");
    expect(useAppStore.getState().roiSets[0]?.count).toBe(0);
    await done;
    expect(useAppStore.getState().roiSets[0]?.count).toBe(1);
  });

  it("keeps a selection that was in another set", async () => {
    seedFor({ http: { deleteRois: () => Promise.resolve([]) } });
    useAppStore.setState({
      rois: [stored({ roiId: "srv1", setName: "nuclei" }), stored({ roiId: "srv2", setName: "cells" })],
      selectedRoiId: "srv2",
    });
    await useAppStore.getState().clearRoiSet("nuclei");
    expect(useAppStore.getState().selectedRoiId).toBe("srv2");
  });

  it("reports a failed clear instead of emptying the panel", async () => {
    seedFor({ http: { deleteRois: () => Promise.reject(new Error("500 upstream")) } });
    useAppStore.setState({ rois: [stored({ setName: "nuclei" })] });
    await useAppStore.getState().clearRoiSet("nuclei");
    expect(useAppStore.getState().roiWriteError).toContain("500 upstream");
    expect(useAppStore.getState().rois).toHaveLength(1);
  });

  it("drops a clear that landed after the tensor moved on", async () => {
    seedFor({
      http: {
        deleteRois: () => {
          useAppStore.setState({ activeTensorId: "second", roisFor: "second" });
          return Promise.resolve(["srv1"]);
        },
      },
    });
    useAppStore.setState({ rois: [stored({ setName: "nuclei" })] });
    await useAppStore.getState().clearRoiSet("nuclei");
    // The rows belong to another tensor now; emptying them here would clear a
    // panel describing an image this delete never touched.
    expect(useAppStore.getState().rois).toHaveLength(1);
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

describe("a draft does not survive a plane change", () => {
  function startDraft() {
    useAppStore.setState({
      activeTensorId: "first",
      requestedArrayId: null,
      render3d: false,
      slice: { ...BASE_SLICE, z: 12 },
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
    useAppStore.getState().setSlice({ z: 13 });
    expect(selectDraft(useAppStore.getState())).toBeNull();
  });

  it("goes when play steps any axis", () => {
    startDraft();
    useAppStore.getState().setSlice({ t: 1 });
    expect(selectDraft(useAppStore.getState())).toBeNull();
  });

  it("goes when an unnamed axis moves", () => {
    startDraft();
    useAppStore.getState().setSlice({ axes: { a0: 2 } });
    expect(selectDraft(useAppStore.getState())).toBeNull();
  });

  it("survives a contrast or gamma change", () => {
    // Those ride SliceState too, and neither invalidates a shape being drawn.
    startDraft();
    useAppStore.getState().setSlice({ gamma: 2.2 });
    useAppStore.getState().setSlice({ contrastMode: "fixed" });
    useAppStore.getState().setSlice({ percentileScale: 2 });
    expect(selectDraft(useAppStore.getState())).not.toBeNull();
  });

  it("comes back if the plane comes back", () => {
    // A stray scroll costs nothing; the draft is hidden, not destroyed.
    startDraft();
    useAppStore.getState().setSlice({ z: 13 });
    useAppStore.getState().setSlice({ z: 12 });
    expect(selectDraft(useAppStore.getState())).not.toBeNull();
  });

  it("a click after the plane moved starts a fresh draft, not a continuation", () => {
    // TileViewer places through `selectDraft`, so the stale vertices are not
    // extended -- this is the behaviour the hidden-not-destroyed choice rests on.
    startDraft();
    useAppStore.getState().setSlice({ z: 13 });
    const seen = selectDraft(useAppStore.getState());
    expect(seen).toBeNull();
  });
});

describe("the overlay toggle governs the whole annotation surface", () => {
  function drafting() {
    useAppStore.setState({
      activeTensorId: "first",
      requestedArrayId: null,
      render3d: false,
      showRois: true,
      slice: { ...BASE_SLICE, z: 12 },
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

  it("changes when residency flips either way", () => {
    const resident = catalogFingerprint([{ ...SOURCE, data_resident: true }]);
    const evicted = catalogFingerprint([{ ...SOURCE, data_resident: false }]);
    expect(resident).not.toBe(evicted);
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
