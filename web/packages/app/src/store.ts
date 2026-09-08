import { create } from "zustand";
import { TensorFlightClient } from "@biopb/tensor-flight-client";
import type {
  DataSourceDescriptor,
  QuerySourcesResult,
  RoiAnnotation,
  RoiGeometry,
  TileInfo,
} from "@biopb/tensor-flight-client";
import { DEFAULT_POLYLINE_WIDTH, clampPolylineWidth } from "./utils/roiDraft";
import type { RoiDraft, RoiTool } from "./utils/roiDraft";
import { TensorApiError } from "@biopb/tensor-flight-client";
import { withBase } from "./base";
import { DEFAULT_VIEWER_URL_STATE, decodeViewerState } from "./utils/viewerUrl";
import { type ColorValue, extractChannelNames } from "./utils/colorUtils";
import { clampSliceTo } from "./utils/vivUtils";
import { splitArrayVersion } from "@biopb/tensor-flight-client";
import {
  DEFAULT_VOLUME_RENDER_MODE,
  type VolumeRenderMode,
} from "./utils/volumeUtils";

export type ConnectionState = "idle" | "connecting" | "connected" | "error";

export interface SliceState {
  t: number;
  z: number;
  c: number;
  /**
   * Index chosen on each axis `t`/`z`/`c` cannot name, keyed by `SliderAxis.key`
   * (`a0`, `a3`, ...).
   *
   * A TIFF sequence's `i`, a plate's `POS`, the second of two axes sharing a
   * label: navigable, but with no semantic name to hold them under. Reset with
   * t/z/c on a source change, and for the same reason — a key means "axis 0 of
   * the tensor in view", so it does not survive one.
   */
  axes: Record<string, number>;
  /**
   * How the contrast window is chosen: from the plane's own histogram, or from
   * two grey levels the user fixed.
   *
   * Fixed is what makes two planes comparable -- an automatic window rescales
   * per plane, so a channel that dims over a timelapse looks constant under it.
   */
  contrastMode: "auto" | "fixed";
  /** Width of the automatic percentile window: 0 = min-max, 1 = 1-99, 2 = 2-98. */
  percentileScale: number;
  /**
   * The window in `fixed` mode, in raw grey levels, or null for "not chosen".
   *
   * Null seeds from whatever is on screen when the mode is turned on, so
   * switching to fixed does not change the image. Dropped on a source change:
   * a level is a value of *this* tensor's dtype, and a uint16 window carried
   * onto a uint8 image is a white frame.
   */
  fixedLimits: [number, number] | null;
  // Display-only exponent applied to the normalized intensity, after the
  // contrast window and before the channel color. 1 leaves the ramp linear;
  // below 1 lifts the dim end, above 1 pushes it down.
  gamma: number;
}

/**
 * The 3-D camera, in `OrbitView`'s own terms.
 *
 * A mirror, not the source of truth: deck.gl owns the camera while the volume
 * is mounted and this trails it on a debounce, which is what keeps an orbit
 * smooth. It is read back only to seed the next mount -- see `VolumeViewer`.
 */
/**
 * The 2-D camera, in `DetailView`'s terms.
 *
 * A mirror of Viv's own view state, on the same terms as {@link Camera3DState}:
 * `VivViewer` keeps driving the viewport and this trails it.
 *
 * The target is carried as `[x, y]`, not the `[x, y, z]` deck.gl reports. An
 * orthographic view's third component is structurally zero
 * (`getDefaultInitialViewState` builds it that way), so carrying it would put a
 * constant in every shared link -- and its absence is what tells a 2-D camera
 * from a 3-D one in the URL.
 */
export interface Camera2DState {
  target: [number, number];
  /** log2(pixels per world unit), as Viv's own initial view state computes. */
  zoom: number;
}

export interface Camera3DState {
  /** Orbit centre, in the scaled world space `volumeCentre` computes. */
  target: [number, number, number];
  /** log2(pixels per world unit), as `volumeZoom` returns. */
  zoom: number;
  /** Pitch, in degrees; `OrbitController` holds it within +/-90. */
  rotationX: number;
  /** Bearing, in degrees, reported wrapped into [-180, 180). */
  rotationOrbit: number;
}

export interface AppState {
  // Client
  client: TensorFlightClient | null;
  connectionState: ConnectionState;
  connectionError: string | null;
  devMode: boolean;
  apiBase: string;

  // Data sources
  sources: DataSourceDescriptor[];
  sourcesLoading: boolean;
  // Progressive discovery: the server is SERVING but its catalog scan is still
  // running. Lets the source list show "Indexing…" instead of "No sources" when
  // the catalog is briefly empty at startup. Refreshed from /readyz.
  scanning: boolean;

  // Active selection
  activeSourceId: string | null;
  /**
   * The selection, always the *stable* address -- what the tree highlights and
   * what `selectSource` sets. Never carries a version token.
   */
  activeTensorId: string | null;
  /**
   * The exact address a link asked for, which may be content-pinned
   * (`id@token`), or null when the selection came from a click.
   *
   * Separate from `activeTensorId` because the two answer different questions:
   * this is what the render path fetches, that is what the catalog UI compares
   * against. Folding them together would either break the tree's highlight (it
   * matches catalog ids, which are never pinned) or force a click to resolve a
   * token before it could select anything.
   *
   * Cleared by `selectSource`: a click supersedes whatever version a link named.
   */
  requestedArrayId: string | null;

  // Slice controls
  slice: SliceState;

  /**
   * The transfer grid of the tensor in view, published by whichever viewer
   * mounted it.
   *
   * The catalog's descriptor is not a substitute: `/api/sources` is a listing,
   * refreshed only when the set of source *urls* changes, so a source that
   * gains a tensor or a timelapse whose `T` grows keeps its old `shape` there
   * until a reload. Bounding a slider on that means a control that cannot reach
   * frames the tensor has. `tile_info` is fetch-per-call and answers for the
   * tensor as it is now.
   *
   * Null while nothing is loaded, and cleared on a source change so a stale
   * grid can never bound the next tensor.
   *
   * Read it through `selectTileInfo`, not directly: a viewer keeps its previous
   * grid until its next fetch answers, so this slot alone cannot say which
   * tensor the grid in it describes.
   */
  tileInfo: TileInfo | null;
  /**
   * The `array_id` the grid above was fetched for -- the viewer's own `arrayId`
   * prop, published back with it. See `selectTileInfo`.
   */
  tileInfoFor: string | null;

  // --- ROI annotations (docs/roi-annotations-ui.md) -----------------------
  /** The tensor's whole annotation set. Filtered to the plane at render time. */
  rois: RoiAnnotation[];
  /**
   * The `array_id` `rois` was fetched for, as `tileInfoFor` is for the grid.
   *
   * This is what makes `loadRois` idempotent, and that is load-bearing rather
   * than an optimisation: switching to the 3-D viewer and back remounts the
   * whole 2-D subtree (`ViewerPane` keys on the render mode), so without it a
   * round trip through 3-D would refetch a set that can reach megabytes.
   */
  roisFor: string | null;
  /** The fetch in flight, so a superseded response cannot overwrite a newer one. */
  roisPending: string | null;
  roisError: string | null;
  /** The tensor `roisError` is about. See `selectRoisError`. */
  roisErrorFor: string | null;
  /** The per-tensor cap clipped the set: what is shown is not all there is. */
  roisTruncated: boolean;
  /** Rows whose geometry this client could not read -- see `decodeRoiListResult`. */
  roisSkipped: number;
  /**
   * The server does not offer annotations at all (501: disabled, or no metadata
   * DB). Distinct from an error, because it is a fact about the deployment
   * rather than a failure -- the UI hides itself instead of reporting a fault.
   */
  roisUnavailable: boolean;
  /** Overlay on/off. A view preference, so it outlives a tensor change. */
  showRois: boolean;
  /**
   * Sets switched off by name. Absence means visible, so a set that appears
   * later shows up rather than starting hidden.
   *
   * Read through `selectHiddenSets`: names mean nothing outside the tensor they
   * were hidden in, and `hiddenSetsFor` is what scopes them to it.
   */
  hiddenSets: string[];
  /** The tensor `hiddenSets` names sets of. See `selectHiddenSets`. */
  hiddenSetsFor: string | null;

  // --- authoring ----------------------------------------------------------
  /** Which tool the pointer carries. A preference: it outlives a tensor change. */
  tool: RoiTool;
  /** The shape being placed. Read through `selectDraft`. */
  draft: RoiDraft | null;
  /** The tensor the draft is being drawn on. See `selectDraft`. */
  draftFor: string | null;
  /** The slice position it is being drawn at. See `selectDraft`. */
  draftSliceKey: string | null;
  /** Selected annotation. Read through `selectSelectedRoi`, which validates it. */
  selectedRoiId: string | null;
  /** Label and set the next new annotation gets. Preferences, kept across tensors. */
  newLabel: string;
  newSetName: string;
  /**
   * Width a new polyline gets, in image pixels. A preference like the two
   * above, and geometry rather than styling -- it is the band of pixels the
   * stroke claims, so it is stored on the annotation and scales with the image.
   */
  newPolylineWidth: number;
  /**
   * Axes a new annotation should NOT pin, i.e. broadcast across. `null` means
   * "the default for this tensor" (see `selectBroadcastAxes`), which is not the
   * same as "none" -- an empty array is a deliberate choice to pin everything.
   */
  broadcastAxes: number[] | null;
  /** The tensor `broadcastAxes` names axes of. */
  broadcastAxesFor: string | null;
  /** A write that failed or lost a conditional put, for the panel to report. */
  roiWriteError: string | null;

  /**
   * The axis being scrubbed automatically (a `SliderAxis.key`), or null.
   *
   * Not part of `SliceState`: play changes which index is asked for over time,
   * not what a frame looks like, and folding it in would put a transient of the
   * UI into the object the viewer diffs its refetches on.
   */
  playAxis: string | null;
  /**
   * The contrast window actually in use, published by whichever viewer is
   * mounted -- automatic or fixed, whichever the slice asked for.
   *
   * Read only to seed `fixedLimits` when the user turns fixed on: without it
   * the panel would have to re-derive the histogram the viewer already has,
   * and the image would jump the moment the mode changed.
   */
  appliedLimits: [number, number] | null;
  /**
   * The sampled min and max grey level of the plane on screen, or null before
   * one has been sampled.
   *
   * What the automatic window would be with neither tail trimmed, published
   * separately because in fixed mode `appliedLimits` is the user's window and
   * no longer says anything about the data. Read to reset a fixed window onto
   * the image actually in view.
   */
  planeLimits: [number, number] | null;
  /**
   * The widest window the data has shown so far, per channel, and the tensor it
   * was sampled from.
   *
   * A float dtype names no range of its own, so the track a fixed window is
   * chosen on is the levels that have actually appeared. Taking that from the
   * plane in view alone would shrink the track on a dim plane -- exactly when a
   * fixed window needs to reach the levels of a bright one. Per channel because
   * two channels of one tensor need not share a scale.
   *
   * Read through `selectObservedLimits`, which hides a union sampled from
   * another tensor rather than trusting a reset, for the reason `tileInfo` is
   * read through `selectTileInfo`.
   */
  observedLimits: Record<number, [number, number]>;
  observedLimitsFor: string | null;
  /**
   * Whether what is on the canvas is the slice that was last asked for.
   *
   * Published by whichever viewer is mounted. Play reads it to pace itself to
   * the data plane rather than to a timer, and the tiled viewer reads its own
   * copy of the same fact to cover a stale plane.
   */
  planeReady: boolean;

  // UI options
  showAdvancedOptions: boolean;
  /**
   * Render the active tensor as a volume rather than as a plane.
   *
   * Not part of `SliceState`: that object's identity drives the tiled viewer's
   * refetches, and the render mode changes which viewer is mounted rather than
   * which pixels it wants. Reset on a source change — the next tensor may have
   * no volume to render, and a toggle stuck on would open it on an error.
   */
  render3d: boolean;
  /**
   * How the 3-D ray-cast combines voxels. A viewing preference rather than a
   * property of the tensor, so unlike `render3d` it survives a source change —
   * someone who wants additive wants it for the next stack too.
   */
  volumeRenderMode: VolumeRenderMode;
  /**
   * Where the 3-D camera is, or null for "wherever the volume fits".
   *
   * Null rather than a computed default because the fitted camera depends on
   * the volume and the pane size, neither of which the store knows; only the
   * viewer can work it out, so the store says "unset" and lets it.
   */
  camera3d: Camera3DState | null;
  /** Where the 2-D camera is, or null for "wherever the plane fits". */
  camera2d: Camera2DState | null;

  // Channel colors (sourceId -> channelIdx -> color)
  channelColors: Record<string, Record<number, ColorValue>>;
  // Channel names (sourceId -> channel names array)
  channelNames: Record<string, string[]>;

  // Catalog polling
  pollingInterval: number;

  // Actions
  initClient: (apiBase: string, token: string | null, devMode: boolean) => void;
  loadSources: () => Promise<void>;
  querySources: (sql: string) => Promise<QuerySourcesResult>;
  selectSource: (sourceId: string | null, tensorId?: string) => void;
  setSlice: (partial: Partial<SliceState>) => void;
  setTileInfo: (value: TileInfo | null, forArrayId: string) => void;
  loadRois: (arrayId: string) => Promise<void>;
  setShowRois: (value: boolean) => void;
  toggleSetHidden: (setName: string) => void;
  setTool: (tool: RoiTool) => void;
  setDraft: (draft: RoiDraft | null) => void;
  setSelectedRoi: (roiId: string | null) => void;
  setNewLabel: (label: string) => void;
  setNewSetName: (setName: string) => void;
  setNewPolylineWidth: (width: number) => void;
  toggleBroadcastAxis: (axis: number, defaults: number[]) => void;
  createRoi: (geometry: RoiGeometry, plane: Record<number, number>) => Promise<void>;
  deleteRoi: (roiId: string) => Promise<void>;
  clearRoiSet: (setName: string) => Promise<void>;
  /**
   * Adopt a whole viewing state at once, as decoded from the URL.
   *
   * One `set`, not a `selectSource` followed by a `setSlice`: `selectSource`
   * resets the slice by design, so the two-call form would need the caller to
   * know the order and would still flash the reset state through the viewer.
   * Returns false when the catalog holds no such tensor -- a link to a source
   * that has since been re-indexed -- which leaves the store untouched so the
   * viewer opens empty rather than on a guess.
   */
  applyViewerState: (params: URLSearchParams) => boolean;
  setPlayAxis: (key: string | null) => void;
  setAppliedLimits: (value: [number, number]) => void;
  setPlaneLimits: (value: [number, number]) => void;
  noteObservedLimits: (
    value: [number, number],
    forArrayId: string,
    channel: number,
  ) => void;
  setPlaneReady: (value: boolean) => void;
  setShowAdvancedOptions: (value: boolean) => void;
  setRender3d: (value: boolean) => void;
  setVolumeRenderMode: (value: VolumeRenderMode) => void;
  setCamera3d: (value: Camera3DState | null) => void;
  setCamera2d: (value: Camera2DState | null) => void;
  getChannelColor: (sourceId: string, channelIdx: number) => ColorValue;
  setChannelColor: (sourceId: string, channelIdx: number, color: ColorValue) => void;
  loadChannelNames: (sourceId: string) => Promise<void>;
  clearSession: () => void;
  startCatalogPolling: () => void;
  stopCatalogPolling: () => void;
}

// Internal timer storage (non-reactive, module-level)
let _pollingTimerId: ReturnType<typeof setInterval> | undefined;

// LocalStorage key for channel color persistence
const CHANNEL_COLORS_STORAGE_KEY = "biopb_channel_colors";

function loadColorsFromStorage(): Record<string, Record<number, ColorValue>> {
  try {
    const stored = localStorage.getItem(CHANNEL_COLORS_STORAGE_KEY);
    if (stored) {
      return JSON.parse(stored) as Record<string, Record<number, ColorValue>>;
    }
  } catch {
    // Ignore parse errors
  }
  return {};
}

function saveColorsToStorage(colors: Record<string, Record<number, ColorValue>>) {
  try {
    localStorage.setItem(CHANNEL_COLORS_STORAGE_KEY, JSON.stringify(colors));
  } catch {
    // Ignore storage errors
  }
}

export const useAppStore = create<AppState>((set, get) => ({
  client: null,
  connectionState: "idle",
  connectionError: null,
  devMode: false,
  apiBase: withBase("/data_plane"),

  sources: [],
  sourcesLoading: false,
  scanning: false,

  activeSourceId: null,
  activeTensorId: null,
  requestedArrayId: null,

  slice: {
    t: 0,
    z: 0,
    c: 0,
    axes: {},
    contrastMode: "auto",
    percentileScale: 1,  // Default 1-99 percentile
    fixedLimits: null,
    gamma: 1,
  },

  tileInfo: null,
  tileInfoFor: null,

  rois: [],
  roisFor: null,
  roisPending: null,
  roisError: null,
  roisErrorFor: null,
  roisTruncated: false,
  roisSkipped: 0,
  roisUnavailable: false,
  showRois: true,
  hiddenSets: [],
  hiddenSetsFor: null,

  tool: "select",
  draft: null,
  draftFor: null,
  draftSliceKey: null,
  selectedRoiId: null,
  newLabel: "",
  newSetName: "",
  newPolylineWidth: DEFAULT_POLYLINE_WIDTH,
  broadcastAxes: null,
  broadcastAxesFor: null,
  roiWriteError: null,

  playAxis: null,
  planeReady: false,
  appliedLimits: null,
  planeLimits: null,
  observedLimits: {},
  observedLimitsFor: null,

  showAdvancedOptions: false,
  render3d: false,
  volumeRenderMode: DEFAULT_VOLUME_RENDER_MODE,
  camera3d: null,
  camera2d: null,

  // Load persisted colors from localStorage on initialization
  channelColors: loadColorsFromStorage(),
  channelNames: {},

  pollingInterval: 60000,

  initClient(apiBase, token, devMode) {
    set({
      client: new TensorFlightClient(apiBase, token),
      connectionState: "connecting",
      connectionError: null,
      devMode,
      apiBase,
    });
  },

  async loadSources() {
    const { client } = get();
    if (!client) return;
    set({ sourcesLoading: true });
    try {
      const sources = await client.listSources();
      // Sort sources by source_url for consistent display and comparison
      const sorted = sources.sort((a, b) => a.source_url.localeCompare(b.source_url));
      set({ sources: sorted, sourcesLoading: false, connectionState: "connected" });
    } catch (err) {
      set({
        sourcesLoading: false,
        connectionState: "error",
        connectionError: err instanceof Error ? err.message : String(err),
      });
    }
  },

  async querySources(sql: string): Promise<QuerySourcesResult> {
    const { client } = get();
    if (!client) {
      return { rows: [], totalSources: 0, returnedSources: 0, truncated: false };
    }
    return client.http.querySources(sql);
  },

  selectSource(sourceId, tensorId) {
    if (!sourceId) {
      set({ activeSourceId: null, activeTensorId: null, requestedArrayId: null });
      return;
    }
    // No catalog lookup, and no `tensors[0]` guess. A bare source_id *is* a
    // valid array_id (the identity policy in descriptor.proto), and the Flight
    // server resolves it to whatever it binds as that source's default tensor.
    // Guessing the first entry here is what biopb/biopb#75 was about: two
    // derivations of one identity that can disagree, where the geometry came
    // from tensors[0] and the read went somewhere else.
    const tid = tensorId ?? sourceId;
    set({
      activeSourceId: sourceId,
      activeTensorId: tid,
      requestedArrayId: null,
      render3d: false,
      // camera3d goes with render3d: it is in the previous volume's world
      // space, so carrying it over would frame the next stack from an
      // arbitrary point.
      camera3d: null,
      camera2d: null,
      tileInfo: null,
      tileInfoFor: null,
      // An axis key means "axis of the tensor in view", so a play in progress
      // does not survive one either.
      playAxis: null,
      planeReady: false,
      appliedLimits: null,
      planeLimits: null,
      // Annotation state is deliberately NOT reset here. Every piece of it
      // carries the tensor it belongs to and is read through a selector that
      // hides it otherwise -- the same treatment `tileInfo` gets, and for a
      // sharper reason: `applyViewerState` changes the tensor without coming
      // through this function at all, so a reset written here would be missed
      // by every link the app opens.
    });
    // `fixedLimits` goes with the tensor for the reason the indices do -- a
    // grey level is a value of its dtype. The mode is a preference and stays,
    // seeding itself from the next tensor's own window.
    set((s) => ({ slice: { ...s.slice, t: 0, z: 0, c: 0, axes: {}, fixedLimits: null } }));
  },

  setSlice(partial) {
    set((s) => ({ slice: { ...s.slice, ...partial } }));
  },

  setPlayAxis(key) {
    set({ playAxis: key });
  },

  setAppliedLimits(value) {
    // Compared by content: the viewers recompute this array every render, and
    // storing a fresh identity each time would loop through their effect.
    set((s) =>
      s.appliedLimits && s.appliedLimits[0] === value[0] && s.appliedLimits[1] === value[1]
        ? s
        : { appliedLimits: value },
    );
  },

  setPlaneLimits(value) {
    set((s) =>
      s.planeLimits && s.planeLimits[0] === value[0] && s.planeLimits[1] === value[1]
        ? s
        : { planeLimits: value },
    );
  },

  noteObservedLimits(value, forArrayId, channel) {
    set((s) => {
      // A union carried over from another tensor is not narrowed back by a
      // union, so the switch starts one rather than widening the old one.
      const carried = s.observedLimitsFor === forArrayId;
      const prev = carried ? s.observedLimits[channel] : undefined;
      if (prev && prev[0] <= value[0] && prev[1] >= value[1]) return s;
      const next: [number, number] = prev
        ? [Math.min(prev[0], value[0]), Math.max(prev[1], value[1])]
        : [value[0], value[1]];
      return {
        observedLimits: carried
          ? { ...s.observedLimits, [channel]: next }
          : { [channel]: next },
        observedLimitsFor: forArrayId,
      };
    });
  },

  setPlaneReady(value) {
    set((s) => (s.planeReady === value ? s : { planeReady: value }));
  },

  setTileInfo(value, forArrayId) {
    // The grid is the first thing that can say what an index may be, so the
    // slice is bounded here rather than where it was read -- see clampSliceTo.
    set((s) => ({
      tileInfo: value,
      tileInfoFor: forArrayId,
      slice: clampSliceTo(s.slice, value),
    }));
  },

  async loadRois(arrayId) {
    const { client } = get();
    if (!client) return;
    // Only the tensor in view is worth asking about. Every selector above hides
    // a set belonging to another tensor, so a fetch for anything else could
    // never be shown -- enforced here rather than trusted to the callers, the
    // same way the reads are guarded rather than the writers.
    if (currentArrayId(get()) !== arrayId) return;
    // Idempotent: already held, or already asked for. Callers fire this from a
    // mount effect, and the 2-D subtree remounts on every render-mode flip.
    if (get().roisFor === arrayId || get().roisPending === arrayId) return;
    if (get().roisUnavailable) return;
    set({ roisPending: arrayId, roisError: null });
    try {
      const result = await client.http.listRois(arrayId);
      // A response for a tensor that is no longer the one being asked about.
      if (get().roisPending !== arrayId) return;
      set({
        rois: result.rois,
        roisFor: arrayId,
        roisPending: null,
        roisTruncated: result.truncated,
        roisSkipped: result.skipped,
      });
    } catch (err) {
      if (get().roisPending !== arrayId) return;
      // 501 is the server saying it does not do annotations. Latched for the
      // session so every later tensor skips the round trip.
      if (err instanceof TensorApiError && err.status === 501) {
        set({ roisPending: null, roisUnavailable: true });
        return;
      }
      // `roisFor` deliberately stays null so a remount or a tensor switch can
      // try again; the effect's own deps keep that from becoming a retry loop.
      set({
        roisPending: null,
        roisErrorFor: arrayId,
        roisError: err instanceof Error ? err.message : String(err),
      });
    }
  },

  setTool(tool) {
    // A draft belongs to the tool that started it; switching abandons it rather
    // than reinterpreting placed vertices under different rules.
    set((s) =>
      s.tool === tool ? s : { tool, draft: null, draftFor: null, draftSliceKey: null },
    );
  },

  setDraft(draft) {
    set({
      draft,
      draftFor: draft ? currentArrayId(get()) : null,
      draftSliceKey: draft ? sliceKey(get().slice) : null,
    });
  },

  setSelectedRoi(roiId) {
    set({ selectedRoiId: roiId });
  },

  setNewLabel(label) {
    set({ newLabel: label });
  },

  setNewSetName(setName) {
    set({ newSetName: setName });
  },

  setNewPolylineWidth(width) {
    set({ newPolylineWidth: clampPolylineWidth(width) });
  },

  toggleBroadcastAxis(axis, defaults) {
    set((s) => {
      const arrayId = currentArrayId(s);
      // `null` means "this tensor's default", so materialise that before
      // editing -- otherwise the first toggle would silently also pin whatever
      // the default was broadcasting.
      const current =
        s.broadcastAxesFor === arrayId && s.broadcastAxes !== null ? s.broadcastAxes : defaults;
      return {
        broadcastAxesFor: arrayId,
        broadcastAxes: current.includes(axis)
          ? current.filter((a) => a !== axis)
          : [...current, axis],
      };
    });
  },

  async createRoi(geometry, plane) {
    const { client, newLabel, newSetName } = get();
    const arrayId = currentArrayId(get());
    if (!client || !arrayId) return;

    // Drawn before it is stored. The click that finishes a shape also clears
    // the draft, so without this the shape the user just traced disappears for
    // the length of a round trip and comes back when the server answers --
    // which is the whole of the felt latency of drawing.
    //
    // Not selected yet: the id is this client's invention until the server
    // answers with its own, and selecting a row whose identity is about to
    // change puts that invention in front of the user.
    const provisional: RoiAnnotation = {
      roiId: provisionalRoiId(),
      arrayId,
      setName: newSetName || DEFAULT_ROI_SET,
      label: newLabel,
      geometry,
      plane,
      props: {},
      rev: 0,
      createdAtMs: Date.now(),
      updatedAtMs: Date.now(),
    };
    set((s) => ({
      roiWriteError: null,
      rois: s.roisFor === arrayId ? [...s.rois, provisional] : s.rois,
    }));

    /** Take the provisional row back out, wherever the list has moved on to. */
    const withoutProvisional = (rois: RoiAnnotation[]) =>
      rois.filter((roi) => roi.roiId !== provisional.roiId);

    try {
      const result = await client.http.putRois(
        arrayId,
        [{ geometry, plane, label: newLabel, setName: newSetName || undefined }],
        { checkRev: true },
      );
      // Dropped if the tensor moved on mid-write: the annotation is stored, and
      // appending it to a list that now describes another tensor would put it
      // on screen over the wrong image. The provisional row goes with it.
      if (currentArrayId(get()) !== arrayId || get().roisFor !== arrayId) {
        set((s) => ({ rois: withoutProvisional(s.rois) }));
        return;
      }
      const stored = result.stored[0];
      if (!stored) {
        set((s) => ({
          rois: withoutProvisional(s.rois),
          roiWriteError: "The server stored no annotation",
        }));
        return;
      }
      // Swapped in place rather than appended, so the annotation does not jump
      // to the end of a list it was already drawn in.
      set((s) => ({
        rois: s.rois.map((roi) => (roi.roiId === provisional.roiId ? stored : roi)),
        selectedRoiId: stored.roiId,
      }));
    } catch (err) {
      set((s) => ({
        rois: withoutProvisional(s.rois),
        roiWriteError: err instanceof Error ? err.message : String(err),
      }));
    }
  },

  async deleteRoi(roiId) {
    const { client } = get();
    const arrayId = currentArrayId(get());
    if (!client || !arrayId) return;
    // Nothing to ask the server about: this row is a local placeholder for a
    // write still in flight, and its id is one this client invented. Left for
    // the write to resolve, which is a round trip away.
    if (isProvisionalRoiId(roiId)) return;

    // Gone on the keystroke, not on the answer. The row is under the pointer
    // when Delete is pressed, so leaving it there for a round trip reads as the
    // key having missed -- and the far commoner outcome by far is that the
    // delete succeeds.
    const index = get().rois.findIndex((roi) => roi.roiId === roiId);
    const removed = get().rois[index];
    set((s) => ({
      roiWriteError: null,
      rois: s.rois.filter((roi) => roi.roiId !== roiId),
      selectedRoiId: s.selectedRoiId === roiId ? null : s.selectedRoiId,
    }));

    try {
      await client.http.deleteRois(arrayId, [roiId]);
      // An id the server did not have is already gone as far as the UI is
      // concerned, so an empty answer is not a failure: the row stays removed
      // rather than coming back as one that cannot be got rid of.
    } catch (err) {
      set((s) => {
        const failed = { roiWriteError: err instanceof Error ? err.message : String(err) };
        // Only back into the list it came out of. After a tensor change the
        // list describes another image, and restoring there would draw this
        // annotation over it.
        if (!removed || currentArrayId(get()) !== arrayId || s.roisFor !== arrayId) return failed;
        const rois = [...s.rois];
        rois.splice(Math.min(index, rois.length), 0, removed);
        return { ...failed, rois };
      });
    }
  },

  async clearRoiSet(setName) {
    const { client } = get();
    const arrayId = currentArrayId(get());
    if (!client || !arrayId) return;
    set({ roiWriteError: null });
    try {
      // No ids: the server drops the whole set in one transaction, so this is
      // not limited to the rows the cap let this client see.
      await client.http.deleteRois(arrayId, undefined, { setName });
      if (currentArrayId(get()) !== arrayId || get().roisFor !== arrayId) return;
      // Filtered by name rather than by the ids that came back, for the same
      // reason: what was deleted is the set, and the response enumerates only
      // what the server chose to list.
      set((s) => ({
        rois: s.rois.filter((roi) => roi.setName !== setName),
        selectedRoiId:
          s.rois.find((roi) => roi.roiId === s.selectedRoiId)?.setName === setName
            ? null
            : s.selectedRoiId,
        // The name means nothing once the set is gone, and leaving it behind
        // would start a later set of the same name hidden.
        hiddenSets: s.hiddenSets.filter((name) => name !== setName),
      }));
    } catch (err) {
      set({ roiWriteError: err instanceof Error ? err.message : String(err) });
    }
  },

  setShowRois(value) {
    set((s) => (s.showRois === value ? s : { showRois: value }));
  },

  toggleSetHidden(setName) {
    set((s) => {
      // Scoped to the tensor in view: a list carried over from another one is
      // not "nothing hidden here", it is a list of names that mean nothing here.
      const arrayId = currentArrayId(s);
      const current = s.hiddenSetsFor === arrayId ? s.hiddenSets : [];
      return {
        hiddenSetsFor: arrayId,
        hiddenSets: current.includes(setName)
          ? current.filter((name) => name !== setName)
          : [...current, setName],
      };
    });
  },

  applyViewerState(params) {
    const requested = params.get("id");
    if (!requested) return false;
    // The link may name a pinned address; the selection is always the stable
    // one, and `source_id` is the prefix before the first "/" by the identity
    // policy -- so both come out of the id itself with no catalog lookup. That
    // is what lets a shared link open while the catalog is capped, still
    // scanning, or missing the source entirely; an id that names nothing then
    // fails at the fetch, which can say so, rather than here in silence.
    const { arrayId: stable } = splitArrayVersion(requested);
    const s = get();
    // Nothing about the tensor in view is inherited. Indices are in its grid
    // and a camera is in its world space, so letting either survive into a link
    // that does not name one frames whatever opens next from a point nobody
    // chose.
    //
    // There is deliberately no "unless it is the same tensor" exemption,
    // because that cannot be decided here. A bare `source_id` and the
    // `source_id/field` it resolves to are two spellings of one identity, and
    // only the Flight server knows which field it binds as a source's default
    // -- so an id comparison reports a tensor change for the catalog's own
    // bare-source click, and reports it in one direction only, which is worse
    // than not asking.
    //
    // Nothing is lost to that. `encodeViewerState` omits a field only when it
    // is at the default used here, so a link this app wrote round-trips
    // exactly; inheriting could only ever change what an *incomplete*
    // hand-written link opens at, which is the case the previous tensor's
    // framing is wrong for.
    //
    // The percentile window, gamma and the render mode do carry across: they
    // are preferences that belong to the viewer rather than to any one tensor,
    // which is how `selectSource` treats them too.
    const next = decodeViewerState(params, {
      ...DEFAULT_VIEWER_URL_STATE,
      arrayId: stable,
      slice: {
        ...DEFAULT_VIEWER_URL_STATE.slice,
        contrastMode: s.slice.contrastMode,
        percentileScale: s.slice.percentileScale,
        gamma: s.slice.gamma,
      },
      volumeRenderMode: s.volumeRenderMode,
    });
    set({
      activeSourceId: stable.split("/", 1)[0] ?? null,
      activeTensorId: stable,
      requestedArrayId: requested,
      slice: next.slice,
      render3d: next.render3d,
      volumeRenderMode: next.volumeRenderMode,
      camera3d: next.camera3d,
      camera2d: next.camera2d,
      // `tileInfo` is left alone: `selectTileInfo` already hides a grid fetched
      // for another id, and clearing it here would depend on a viewer mounting
      // afterwards to put one back.
    });
    return true;
  },

  setShowAdvancedOptions(value) {
    set({ showAdvancedOptions: value });
  },

  setRender3d(value) {
    set({ render3d: value });
  },

  setVolumeRenderMode(value) {
    set({ volumeRenderMode: value });
  },

  setCamera3d(value) {
    set({ camera3d: value });
  },

  setCamera2d(value) {
    set({ camera2d: value });
  },

  getChannelColor(sourceId, channelIdx) {
    const { channelColors } = get();
    const sourceColors = channelColors[sourceId];
    if (sourceColors && sourceColors[channelIdx]) {
      return sourceColors[channelIdx];
    }
    // No persisted color - return "auto" to use guessed default
    return "auto";
  },

  setChannelColor(sourceId, channelIdx, color) {
    const { channelColors } = get();
    const newColors = {
      ...channelColors,
      [sourceId]: {
        ...channelColors[sourceId],
        [channelIdx]: color,
      },
    };
    set({ channelColors: newColors });
    saveColorsToStorage(newColors);
  },

  async loadChannelNames(sourceId) {
    const { client } = get();
    if (!client) return;
    try {
      const metadata = await client.getSourceMetadata(sourceId);
      const names = extractChannelNames(metadata);
      if (names.length > 0) {
        set((s) => ({ channelNames: { ...s.channelNames, [sourceId]: names } }));
      }
    } catch {
      // Ignore errors - channel names are optional
    }
  },

  clearSession() {
    sessionStorage.removeItem("biopb_token");
    window.location.href = withBase("/unlock");
  },

  startCatalogPolling() {
    const pollingTimerId = setInterval(async () => {
      const { client, sources, activeSourceId, requestedArrayId, selectSource } = get();
      if (!client || get().connectionState !== "connected") return;

      try {
        const newSources = await client.listSources();
        const sorted = newSources.sort((a, b) => a.source_url.localeCompare(b.source_url));

        // Refresh the scan-in-progress flag so the "Indexing…" hint clears once
        // the background catalog scan finishes (best-effort; a readyz blip just
        // leaves the previous value).
        try {
          const readyz = await client.http.readyz();
          set({ scanning: !!readyz.backend_health?.full_scan_in_progress });
        } catch {
          // ignore transient readyz errors
        }

        // Compare source_urls to detect changes
        const oldUrls = sources.map((s) => s.source_url).join(",");
        const newUrls = sorted.map((s) => s.source_url).join(",");

        if (oldUrls !== newUrls) {
          set({ sources: sorted });

          // A catalog response is a listing, not proof that an unlisted source
          // is gone: it may be capped, still scanning, or temporarily failed.
          // In particular, retain a source selected from a shared URL, which
          // deliberately does not need to be present in the listing.
          if (
            activeSourceId &&
            !requestedArrayId &&
            !sorted.find((s) => s.source_id === activeSourceId)
          ) {
            selectSource(null);
          }
        }
      } catch (err) {
        // Silent failure - don't change connection state for transient errors
        console.warn("Catalog polling error:", err);
      }
    }, get().pollingInterval);

    // Store timer ID for cleanup
    _pollingTimerId = pollingTimerId;
  },

  stopCatalogPolling() {
    if (_pollingTimerId) {
      clearInterval(_pollingTimerId);
      _pollingTimerId = undefined;
    }
  },
}));

/**
 * The grid for what is currently addressed, or null while none has landed.
 *
 * `tileInfo` is whichever viewer last published one, and a viewer holds its
 * previous grid until its next fetch answers -- so between a new selection and
 * that answer the slot describes the tensor that just left. Pairing it with the
 * id it was fetched for makes that window invisible instead of wrong: the
 * sliders fall back to the catalog, and the URL write-back falls back to the id
 * it was asked for rather than stamping the previous tensor's into the bar.
 *
 * The comparison is safe where the one in `applyViewerState` was not, and for a
 * concrete reason: both sides are copies of a single string -- the `arrayId`
 * prop the viewer was mounted with -- rather than two spellings of one
 * identity, so no canonicalization only the server can do is involved.
 */
/**
 * The address the viewer is actually rendering: the exact one a link asked for,
 * which may be content-pinned, else the selection.
 *
 * Single-sourced because every "is this state about the tensor in view?" guard
 * has to agree, and two spellings of that question would disagree exactly when
 * a pinned link is open.
 */
export function currentArrayId(s: AppState): string | null {
  return s.requestedArrayId ?? s.activeTensorId;
}

export function selectTileInfo(s: AppState): TileInfo | null {
  return s.tileInfoFor === currentArrayId(s) ? s.tileInfo : null;
}

/** The levels this tensor's current channel has shown, or null if none yet. */
export function selectObservedLimits(s: AppState): [number, number] | null {
  if (s.observedLimitsFor !== currentArrayId(s)) return null;
  return s.observedLimits[s.slice.c] ?? null;
}

// --- ROI annotations -------------------------------------------------------
//
// Every one of these hides state belonging to another tensor rather than
// relying on someone having reset it. `selectSource` is not the only way the
// tensor in view changes -- `applyViewerState` does it too, straight from a URL
// -- so a reset would have to be written at every such site and kept in step
// with the next one. Answering the question at the read instead makes that
// impossible to get wrong, which is the same reason `tileInfo` is left alone.
//
// Stable empty constants: a selector returning a fresh [] on every call would
// re-render its subscriber on every unrelated store write.
/**
 * Prefix for a row drawn before the server has stored it.
 *
 * Namespaced with a character the wire cannot produce, so it can never collide
 * with a server id: the delete route splits ids on "," and the Flight action
 * rejects one containing it, which is the same reasoning that lets ids be
 * joined into a query string.
 */
const PROVISIONAL_PREFIX = "pending,";
let provisionalSeq = 0;

function provisionalRoiId(): string {
  return `${PROVISIONAL_PREFIX}${++provisionalSeq}`;
}

/** A row that exists only in this client, because its write is still in flight. */
export function isProvisionalRoiId(roiId: string): boolean {
  return roiId.startsWith(PROVISIONAL_PREFIX);
}

/**
 * What the server substitutes for an empty set name before storing.
 *
 * Repeated here so a provisional row lands in the set row its stored form will,
 * rather than appearing under a blank name for the length of a round trip.
 */
const DEFAULT_ROI_SET = "default";

const NO_ROIS: RoiAnnotation[] = [];
const NO_SETS: string[] = [];

/** This tensor's annotations, or none while another tensor's are still held. */
export function selectRois(s: AppState): RoiAnnotation[] {
  return s.roisFor === currentArrayId(s) ? s.rois : NO_ROIS;
}

/** A fetch is in flight for the tensor in view. */
export function selectRoisLoading(s: AppState): boolean {
  return s.roisPending !== null && s.roisPending === currentArrayId(s);
}

/** The cap clipped THIS tensor's set -- not one looked at earlier. */
export function selectRoisTruncated(s: AppState): boolean {
  return s.roisFor === currentArrayId(s) && s.roisTruncated;
}

export function selectRoisSkipped(s: AppState): number {
  return s.roisFor === currentArrayId(s) ? s.roisSkipped : 0;
}

export function selectRoisError(s: AppState): string | null {
  return s.roisErrorFor === currentArrayId(s) ? s.roisError : null;
}

/**
 * A stable key for the slice position, for spotting that it has moved.
 *
 * Only the indices: contrast, gamma and the percentile window ride `SliceState`
 * too, and none of them invalidate a shape being drawn.
 */
export function sliceKey(slice: SliceState): string {
  return `${slice.t}|${slice.z}|${slice.c}|${JSON.stringify(slice.axes)}`;
}

/** Sets hidden in the tensor in view. Names do not carry across tensors. */
export function selectHiddenSets(s: AppState): string[] {
  return s.hiddenSetsFor === currentArrayId(s) ? s.hiddenSets : NO_SETS;
}

/**
 * The shape being placed, if it belongs here.
 *
 * Scoped to the tensor AND to 2-D: the volume viewer has no drawing surface, and
 * a half-placed polygon reappearing after a trip through 3-D is only confusing.
 * Both conditions answered at the read, so no writer has to remember either.
 */
export function selectDraft(s: AppState): RoiDraft | null {
  // `showRois` off means the annotation surface is off, not just that stored
  // shapes are hidden -- otherwise a draft keeps drawing over an overlay the
  // user switched off, and finishing writes something they cannot see. To thin
  // clutter while drawing, hide the noisy set instead; that is what the per-set
  // toggles are for.
  if (!s.showRois) return null;
  if (s.draftFor !== currentArrayId(s) || s.render3d) return null;
  // Vertices were traced against the pixels of one plane. Navigating away --
  // the slider, a keyboard scroll, or play stepping an axis -- makes them a
  // shape drawn on an image nobody is looking at any more, and finishing there
  // would pin it to the plane it was NOT drawn on. Answered at the read, so
  // every route that moves the slice is covered without naming any of them.
  return s.draftSliceKey === sliceKey(s.slice) ? s.draft : null;
}

/**
 * The selected annotation, resolved against the set actually in view.
 *
 * Resolving rather than storing it means a selection cannot outlive the
 * annotation: deleting it, switching tensor, or a refetch that no longer
 * contains it all leave this null with nothing to clean up.
 */
export function selectSelectedRoi(s: AppState): RoiAnnotation | null {
  if (!s.selectedRoiId) return null;
  return selectRois(s).find((roi) => roi.roiId === s.selectedRoiId) ?? null;
}

/**
 * The selected annotation's id, or null when the selection is not in view.
 *
 * A primitive, so a subscriber re-renders on a change of selection rather than
 * on every change to the set it lives in.
 */
export function selectSelectedRoiId(s: AppState): string | null {
  return selectSelectedRoi(s)?.roiId ?? null;
}

/**
 * Axes a new annotation will broadcast across, given this tensor's defaults.
 *
 * `defaults` is computed from the grid by the caller (channel, normally -- see
 * `defaultBroadcastAxes`), because the store does not read `TileInfo`.
 */
export function selectBroadcastAxes(s: AppState, defaults: number[]): number[] {
  return s.broadcastAxesFor === currentArrayId(s) && s.broadcastAxes !== null
    ? s.broadcastAxes
    : defaults;
}
