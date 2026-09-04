import { afterEach, describe, expect, it, vi } from "vitest";
import type { DataSourceDescriptor, TensorFlightClient, TileInfo } from "@biopb/tensor-flight-client";
import { selectTileInfo, useAppStore } from "./store";

const SOURCE: DataSourceDescriptor = {
  source_id: "listed",
  source_url: "file:///listed",
  source_type: "file",
  metadata_json: null,
  tensors: [],
};

const BASE_SLICE = { t: 0, z: 0, c: 0, axes: {}, percentileScale: 1, useMinMax: false, gamma: 1 };

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
      slice: { t: 5, z: 6, c: 1, axes: { a3: 2 }, percentileScale: 2, useMinMax: false, gamma: 1.6 },
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
