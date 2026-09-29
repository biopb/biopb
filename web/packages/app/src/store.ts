import { create } from "zustand";
import { TensorFlightClient } from "@biopb/tensor-flight-client";
import type {
  DataSourceDescriptor,
  QuerySourcesResult,
  RoiAnnotation,
  RoiGeometry,
  RoiListResult,
  RoiSetInfo,
  SliderAxis,
  SourceJobStatus,
  TileInfo,
} from "@biopb/tensor-flight-client";
import { DEFAULT_POLYLINE_WIDTH, clampPolylineWidth } from "./utils/roiDraft";
import type { RoiDraft, RoiTool } from "./utils/roiDraft";
import { roiSetCounts } from "./utils/roiSets";
import {
  TensorAbortError,
  TensorApiError,
  isReservedSetName,
  isTransportError,
  vivDtype,
} from "@biopb/tensor-flight-client";
import { withBase } from "./base";
import { DEFAULT_VIEWER_URL_STATE, decodeViewerState } from "./utils/viewerUrl";
import type { ViewerUrlState } from "./utils/viewerUrl";
import { type ColorValue, extractChannelNames } from "./utils/colorUtils";
import {
  DEFAULT_LABEL_OPACITY,
  clampContrastLimits,
  clampLabelOpacity,
  clampSliceTo,
  contrastLimitsFrom,
  contrastTrack,
  percentileBounds,
} from "./utils/vivUtils";
import {
  descriptorFromTileInfo,
  forget as forgetRecents,
  readRecents,
  remember as rememberRecent,
  writeRecents,
} from "./utils/recentSources";
import { splitArrayVersion, splitLabelArrayId } from "@biopb/tensor-flight-client";
import {
  DEFAULT_VOLUME_RENDER_MODE,
  type VolumeRenderMode,
} from "./utils/volumeUtils";

export type ConnectionState = "idle" | "connecting" | "connected" | "error";

/**
 * Where in the tensor's grid the view is. Per tensor: an index means "index of
 * the tensor in view", so `openTensor` resets it.
 *
 * Separate from {@link DisplayState} because the two change for different
 * reasons and cost differently: a new position asks the server for other
 * pixels, a new display setting re-shades the same ones. Keeping the objects
 * apart is what lets a contrast drag leave `position` -- and everything keyed
 * on its identity: the Viv selection, the volume request, a draft's plane --
 * untouched.
 */
export interface PositionState {
  t: number;
  z: number;
  c: number;
  /**
   * Index chosen on each axis `t`/`z`/`c` cannot name, keyed by `SliderAxis.key`
   * (`a0`, `a3`, ...).
   *
   * A TIFF sequence's `i`, a plate's `POS`, the second of two axes sharing a
   * label: navigable, but with no semantic name to hold them under. Reset with
   * t/z/c on a source change, and for the same reason -- a key means "axis 0 of
   * the tensor in view", so it does not survive one.
   */
  axes: Record<string, number>;
}

/**
 * How the image is shaded. A preference that outlives a tensor, except
 * `fixedLimits`, which is a value of one tensor's dtype.
 */
export interface DisplayState {
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
 * Position and display as one record. Only the URL codec uses it: a link names
 * both in one flat set of parameters.
 */
export type SliceState = PositionState & DisplayState;

const pickPosition = ({ t, z, c, axes }: SliceState): PositionState => ({ t, z, c, axes });
const pickDisplay = ({ contrastMode, percentileScale, fixedLimits, gamma }: SliceState): DisplayState => ({
  contrastMode,
  percentileScale,
  fixedLimits,
  gamma,
});

/** `next`, unless it holds the same indices as `current`, in which case `current`. */
function samePositionOr(current: PositionState, next: PositionState): PositionState {
  return samePosition(current, next) ? current : next;
}

/** Same indices, so a write that changes nothing can keep the object. */
function samePosition(a: PositionState, b: PositionState): boolean {
  if (a.t !== b.t || a.z !== b.z || a.c !== b.c) return false;
  const keys = Object.keys(a.axes);
  return keys.length === Object.keys(b.axes).length && keys.every((k) => a.axes[k] === b.axes[k]);
}

/**
 * Why a viewer could not start.
 *
 * `"capability"` is a settled fact about this browser or tensor, not worth
 * re-testing. `"transport"` is anything that might go the other way next time,
 * so it gets a retry.
 */
export type ViewerErrorKind = "capability" | "transport";

/** What a link carries beyond the address: the whole viewing state it names. */
export type ViewInit = Omit<ViewerUrlState, "arrayId">;

/**
 * The tensor being opened, and the single answer to "which tensor is on screen".
 *
 * Resolved here, before any viewer mounts: `openTensor` fetches `tile_info`
 * and only then does `key` exist, so nothing renders against a guess and the
 * key never changes spelling. A viewer is handed `info` rather than fetching.
 */
export interface ViewTarget {
  /** Bumped by every open. An async landing for an older epoch is dropped. */
  epoch: number;
  /** What the click or link named: bare, pinned, or a specific field. */
  requested: string | null;
  /**
   * Opened from a link, which names its own source and so is not evicted when
   * the catalog listing does not contain it.
   */
  linked: boolean;
  status: "idle" | "resolving" | "ready" | "failed";
  /** The resolved stable address, `src/field`: never bare, never pinned. Set when ready. */
  key: string | null;
  /** The one `tile_info`, shared by both viewers, the panels and the URL. */
  info: TileInfo | null;
  /** The first attempt failed with a transport error and another is coming. */
  retrying: boolean;
  error: { reason: string; kind: ViewerErrorKind } | null;
  /** A link's `rs=` names, written under `key` when the target lands. */
  seedSets: string[] | null;
}

/** The sampled grey levels of one plane. */
export interface PlaneSamples {
  /** The plane they were read for, by identity: the Viv selection, or the volume. */
  plane: object;
  /** Sorted ascending, as `contrastSamples` returns them. */
  values: Float64Array;
}

/** Facts only a mounted viewer can observe. See {@link AppState.runtime}. */
export interface Runtime {
  epoch: number;
  /**
   * Whether what is on the canvas is the plane that was last asked for. Play
   * reads it to pace itself to the data plane rather than to a timer, and the
   * tiled viewer reads its own copy of the same fact to cover a stale plane.
   */
  planeReady: boolean;
  /**
   * The last plane sampled, deliberately kept across a plane change so the
   * contrast does not flash while the next read is in flight. Null before the
   * first.
   */
  samples: PlaneSamples | null;
}

/**
 * What belongs to one tensor, held under its `ViewTarget.key`.
 */
export interface TensorView {
  /**
   * The rows held, across every scope that has landed. Filtered to the plane
   * and to the visible sets at render time.
   */
  rois: RoiAnnotation[];
  /**
   * Every set on the tensor, server-owned ones included, with its stored row
   * count. The discovery half of the listing: `rois` holds a server-owned set
   * only once it was asked for by name.
   */
  roiSets: RoiSetInfo[];
  /**
   * The fetch scopes that have landed: `""` for the unqualified listing (the
   * client-owned sets, all at once), a set name for a server-owned set fetched
   * on its own. Presence is what makes `loadRois` idempotent, and that is
   * load-bearing rather than an optimisation: switching to the 3-D viewer and
   * back remounts the whole 2-D subtree (`ViewerPane` keys on the render mode),
   * so without it a round trip through 3-D would refetch a set that can reach
   * megabytes.
   *
   * A scope is dropped again when a later listing's counts disagree with the
   * rows held for it -- the server rebuilds a reserved set on every
   * registration -- so the next `loadRois` fetches it afresh. Opening the
   * tensor drops them all, keeping the rows on screen while they are re-listed:
   * someone else may have annotated it in the meantime.
   */
  roiScopes: Record<string, RoiScopeState>;
  /** Scopes in flight. */
  roisPending: string[];
  roisError: string | null;
  /**
   * The sets on screen, by name, or `null` for this tensor's default: the
   * client-owned sets shown, the server-owned ones not.
   *
   * A positive list rather than a hidden one because the default is not "all":
   * a link has to be able to say *show* `@ome`, and a hidden list could only
   * say the opposite. The default is materialised on the first toggle, as
   * `broadcastAxes` is. A server-owned set on the list is what `loadRois`
   * fetches by name -- visibility is what drives the lazy fetch.
   */
  visibleSets: string[] | null;
  /**
   * Axes a new annotation should NOT pin, i.e. broadcast across. `null` means
   * "the default for this tensor" (see `selectBroadcastAxes`), which is not the
   * same as "none" -- an empty array is a deliberate choice to pin everything.
   */
  broadcastAxes: number[] | null;
  /** The shape being placed, and the slice position it is being drawn at. */
  draft: { shape: RoiDraft; sliceKey: string } | null;
  /**
   * The widest window the data has shown so far, per channel.
   *
   * A float dtype names no range of its own, so the track a fixed window is
   * chosen on is the levels that have actually appeared. Taking that from the
   * plane in view alone would shrink the track on a dim plane -- exactly when a
   * fixed window needs to reach the levels of a bright one. Per channel because
   * two channels of one tensor need not share a scale.
   */
  observedLimits: Record<number, [number, number]>;
  /** Logical clock of the last write, for eviction. */
  used: number;
}

/** What a tensor with no record yet reads as. Stable, so selectors do not churn. */
export const EMPTY_VIEW: TensorView = {
  rois: [],
  roiSets: [],
  roiScopes: {},
  roisPending: [],
  roisError: null,
  visibleSets: null,
  broadcastAxes: null,
  draft: null,
  observedLimits: {},
  used: 0,
};

/** Tensors whose record is kept. The one in view is never evicted. */
const VIEW_LIMIT = 8;
let viewClock = 0;

/**
 * A patch to `views` that updates `key`'s record, or nothing when there is no
 * tensor to write to. Evicts the least recently written record past the limit.
 */
function withView(
  s: AppState,
  key: string | null,
  patch: Partial<TensorView> | ((v: TensorView) => Partial<TensorView>),
): Partial<AppState> {
  if (!key) return {};
  const current = s.views[key] ?? EMPTY_VIEW;
  const views = {
    ...s.views,
    [key]: { ...current, ...(typeof patch === "function" ? patch(current) : patch), used: ++viewClock },
  };
  const keys = Object.keys(views);
  if (keys.length > VIEW_LIMIT) {
    const victim = keys
      .filter((k) => k !== key && k !== s.target.key)
      .sort((a, b) => (views[a]?.used ?? 0) - (views[b]?.used ?? 0))[0];
    if (victim !== undefined) delete views[victim];
  }
  return { views };
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

/** What one landed fetch scope said about itself. */
export interface RoiScopeState {
  /** The server's per-tensor cap clipped this fetch: it holds more than was returned. */
  truncated: boolean;
  /** Rows whose geometry this client could not read -- see `decodeRoiListResult`. */
  skipped: number;
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

  /**
   * source_ids this browser has opened, most recent first. Persisted; see
   * `utils/recentSources`.
   */
  recentIds: readonly string[];
  /**
   * `recentIds` resolved to something renderable, same order.
   *
   * Shorter than `recentIds` whenever an id did not resolve this pass. A listed
   * source is taken straight from `sources`; an unlisted one -- a `cache://`
   * upload, or a source past the catalog cap -- is rebuilt from its `tile_info`,
   * which resolves against the Flight server's registry rather than the catalog
   * and so answers for both.
   */
  recentSources: DataSourceDescriptor[];

  // Active selection
  activeSourceId: string | null;
  /**
   * The selection, always the *stable* address -- what the tree highlights and
   * what `openTensor` sets. Never carries a version token.
   */
  activeTensorId: string | null;
  /**
   * The tensor being opened: what was asked for, whether it has resolved, and
   * the one `tile_info` that answers for it. See {@link ViewTarget}.
   */
  target: ViewTarget;

  // Slice controls
  position: PositionState;
  display: DisplayState;

  // --- per-tensor state ---------------------------------------------------
  /**
   * Everything that belongs to "the tensor in view", one record per tensor, keyed
   * by `target.key`. A small LRU: leaving a tensor keeps its record, so coming
   * back finds its annotation rows warm.
   *
   * An async writer captures the key when it starts and writes into that
   * record, so a landing for a tensor the user has left is harmless rather than
   * something to guard against. Read through the selectors below, which resolve
   * the tensor in view -- never `views` directly.
   */
  views: Record<string, TensorView>;
  /**
   * The server does not offer annotations at all (501: disabled, or no metadata
   * DB). Distinct from an error, because it is a fact about the deployment
   * rather than a failure -- the UI hides itself instead of reporting a fault.
   */
  roisUnavailable: boolean;
  /** Overlay on/off. A view preference, so it outlives a tensor change. */
  showRois: boolean;

  // --- authoring ----------------------------------------------------------
  /** Which tool the pointer carries. A preference: it outlives a tensor change. */
  tool: RoiTool;
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
  /** A write that failed or lost a conditional put, for the panel to report. */
  roiWriteError: string | null;

  // --- label overlay ------------------------------------------------------
  /**
   * The label set drawn over the image, as its whole `array_id`, or null.
   *
   * One at a time. Two categorical overlays stacked would be two palettes
   * fighting for the same pixel, with no blend of them that says which object
   * is which -- so picking a second set replaces the first.
   *
   * Scoped like the annotation state rather than reset on a source change: the
   * id names its own image, so `selectLabelOverlay` can tell whether the set
   * belongs to the tensor in view, and coming back to that tensor finds its
   * overlay still on.
   */
  labelOverlay: string | null;
  /**
   * The overlay's alpha, 0-1.
   *
   * Never 1 by default: the point of an overlay is the image under it, and a
   * fully opaque one is just a different image.
   */
  labelOpacity: number;

  /**
   * The axis being scrubbed automatically (a `SliderAxis.key`), or null.
   *
   * Not part of `SliceState`: play changes which index is asked for over time,
   * not what a frame looks like, and folding it in would put a transient of the
   * UI into the object the viewer diffs its refetches on.
   */
  playAxis: string | null;
  /**
   * What only the mounted viewer can observe, tagged with the `target.epoch` it
   * was observed under. A write carrying another epoch is dropped, so a viewer
   * that outlives its tensor for a commit cannot leak into the next one.
   *
   * Everything else the contrast controls need is derived from this and from
   * `target.info`, `views[key].observedLimits` and `display`, by
   * {@link selectContrastTrack} and {@link selectContrastWindow}: the shader and
   * the panel call the same selectors (biopb/biopb#955), so they cannot disagree
   * and nothing derived is stored.
   */
  runtime: Runtime;

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
  /**
   * The only way the tensor in view changes: a tree click, a recent, a link, or
   * (as `null`) the catalog poll dropping it.
   *
   * Bumps the epoch, resets everything that belongs to the previous tensor,
   * and resolves `tile_info` in the background; `target` turns `ready` when it
   * lands. `init` is a link's viewing state -- its presence is what marks the
   * open as a link's. A click passes none, which resets the indices and
   * cameras to their defaults; preferences carry across either way.
   */
  openTensor: (address: string | null, init?: ViewInit) => void;
  /** Re-resolve a failed target, keeping everything else as it was. */
  retryTarget: () => void;
  /** Record a source as just opened, and persist the list. */
  noteRecent: (sourceId: string) => void;
  /** Adopt a list written by another tab. Does not write it back. */
  syncRecents: (ids: readonly string[]) => void;
  /** Resolve `recentIds` into `recentSources`, dropping ids the server 404s. */
  hydrateRecents: () => Promise<void>;
  /** Move within the grid. A move to where the view already is keeps the object. */
  setPosition: (partial: Partial<PositionState>) => void;
  setDisplay: (partial: Partial<DisplayState>) => void;
  /**
   * Fetch what the tensor in view should hold and does not yet: the client-owned
   * sets, and every server-owned set that is visible. Idempotent on what has
   * landed and what is in flight, so callers fire it freely -- on mount, and
   * whenever the visible sets change.
   */
  loadRois: () => Promise<void>;
  setShowRois: (value: boolean) => void;
  toggleSetVisible: (setName: string) => void;
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
  /** Draw this label set over its image, or nothing. See `labelOverlay`. */
  setLabelOverlay: (arrayId: string | null) => void;
  setLabelOpacity: (value: number) => void;
  /**
   * Decode a link and `openTensor` it. Returns false when the link names no
   * tensor, leaving the store untouched.
   */
  applyViewerState: (params: URLSearchParams) => boolean;
  setPlayAxis: (key: string | null) => void;
  /**
   * Move one slider axis to `value`, under its name or into `axes`.
   *
   * Reads the state it writes into rather than a caller's copy: these writes are
   * debounced and, under play, fired from a timer -- a stale `axes` map would
   * silently drop a sibling axis's index.
   */
  setAxisIndex: (axis: Pick<SliderAxis, "named" | "key">, value: number) => void;
  /** Widen the tensor in view's observed levels on `channel`. */
  noteObservedLimits: (value: [number, number], channel: number) => void;
  /**
   * A viewer sampled a plane: keep its values for the contrast window, and note
   * their extremes as levels the tensor has shown, on `channel`.
   */
  notePlaneSamples: (samples: PlaneSamples, channel: number, epoch: number) => void;
  setPlaneReady: (value: boolean, epoch: number) => void;
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

  /**
   * Resolve/warm jobs in flight or recently settled, keyed `"<kind>:<id>"`.
   *
   * Server-owned: these mirror `/api/sources/{id}/{kind}/status` and are
   * re-read on a poll, never advanced locally. A recall survives a reload and
   * a second tab, so the browser's copy is a view of it, not the record of it.
   */
  sourceJobs: Record<string, SourceJobStatus>;
  startResolve: (sourceId: string) => Promise<void>;
  startWarm: (sourceId: string) => Promise<void>;
  cancelSourceJob: (kind: SourceJobKind, sourceId: string) => Promise<void>;
  dismissSourceJob: (kind: SourceJobKind, sourceId: string) => void;
  stopJobPolling: () => void;
}

// Internal timer storage (non-reactive, module-level)
let _pollingTimerId: ReturnType<typeof setInterval> | undefined;

export type SourceJobKind = "resolve" | "warm";

/** Composite key for {@link AppState.sourceJobs}. */
export function jobKey(kind: SourceJobKind, sourceId: string): string {
  return `${kind}:${sourceId}`;
}

/**
 * How often an in-flight resolve/warm is re-read.
 *
 * Far tighter than the 60s catalog poll because this one is only running while
 * the user is watching a progress bar they asked for, and it stops the moment
 * nothing is in flight.
 */
const JOB_POLL_MS = 1000;

/**
 * Hydrate-ahead after a resolve, off.
 *
 * The server's chunk cache serves its segments by mmap, so warming a source
 * larger than RAM walks the whole page-cache LRU and evicts the segments
 * serving every *other* source -- for bytes warm never even uses, since the
 * read only exists to make the sync client write to disk. It does not keep its
 * own coarse levels either, and nothing portable would: there is no
 * `posix_fadvise` on Windows, which is where the synced-folder sources this
 * serves live (biopb/biopb#1043). Flip back once warm has a retention policy.
 *
 * This is the SPA's only warm trigger, so while it is false `WarmTray` never
 * appears. Both stay wired and tested, ready for the flip.
 */
const AUTO_WARM_AFTER_RESOLVE: boolean = false;

let _jobPollTimerId: ReturnType<typeof setInterval> | undefined;

/** A job that has stopped moving, whatever the reason. */
function isSettled(job: SourceJobStatus): boolean {
  return job.state !== "running";
}

/**
 * Everything about a catalog listing the tree renders off, as one string.
 *
 * The poll used to diff the `source_url` set alone, which is blind to every
 * in-place change: a cloud source that resolves was already listed under that
 * url, it just gained its tensors and flipped its flags, so the tree kept
 * showing it as unresolved until a manual reload (biopb/biopb#1030). Warming
 * and eviction are invisible the same way. `JSON.stringify` covers every
 * field of `DataSourceDescriptor` by construction, so a field added later
 * can't go silently blind to the poll the way the url-only check did.
 *
 * It is order-sensitive, in both array and key order, and that is safe: the
 * worst an ordering change can do is a false *positive* -- one extra repaint.
 * A false negative is impossible, because two listings stringify alike only
 * when every field of every source already matches.
 *
 * Order is deterministic anyway, resting on three things worth naming since
 * nothing else states them: the server lists `ORDER BY source_id` (unique, so
 * a total order); `.sort()` is stable, so the `source_url` ties -- real, every
 * upload sorts as `""` -- keep that order; and both sides are `JSON.parse` of
 * the same endpoint, whose key order is fixed by `_source_row_to_dict`. Feed
 * `sources` from a hand-built descriptor instead and the third stops holding.
 */
export function catalogFingerprint(sources: DataSourceDescriptor[]): string {
  return JSON.stringify(sources);
}

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

const IDLE_TARGET: ViewTarget = {
  epoch: 0,
  requested: null,
  linked: false,
  status: "idle",
  key: null,
  info: null,
  retrying: false,
  error: null,
  seedSets: null,
};

export const useAppStore = create<AppState>((set, get) => ({
  client: null,
  connectionState: "idle",
  connectionError: null,
  devMode: false,
  apiBase: withBase("/data_plane"),

  sources: [],
  sourcesLoading: false,
  scanning: false,

  // Seeded at module init, like the persisted channel colors below, so the node
  // is populated on the first render rather than after an effect. `readRecents`
  // answers [] where there is no storage, which covers node.
  recentIds: readRecents(),
  recentSources: [],

  activeSourceId: null,
  activeTensorId: null,
  target: IDLE_TARGET,

  position: { t: 0, z: 0, c: 0, axes: {} },
  display: {
    contrastMode: "auto",
    percentileScale: 1, // Default 1-99 percentile
    fixedLimits: null,
    gamma: 1,
  },

  views: {},
  roisUnavailable: false,
  showRois: true,

  tool: "select",
  selectedRoiId: null,
  newLabel: "",
  newSetName: "",
  newPolylineWidth: DEFAULT_POLYLINE_WIDTH,
  roiWriteError: null,

  labelOverlay: null,
  labelOpacity: DEFAULT_LABEL_OPACITY,

  playAxis: null,
  runtime: { epoch: 0, planeReady: false, samples: null },

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

  noteRecent(sourceId) {
    const next = rememberRecent(get().recentIds, sourceId);
    // `remember` hands back the same array when the id is already the head, so
    // the repeat calls from `applyViewerState` -- one per slider move -- cost a
    // comparison rather than a write.
    if (next === get().recentIds) return;
    set({ recentIds: next });
    writeRecents(next);
  },

  syncRecents(ids) {
    // No write back: this is another tab's list arriving, and echoing it would
    // have the two tabs writing over each other on every change. Skip the set
    // when the value hasn't actually moved -- this is also called on mount with
    // a fresh read of the same storage the initial state already loaded, and a
    // same-content array would otherwise re-trigger hydrateRecents for nothing.
    const current = get().recentIds;
    if (ids.length === current.length && ids.every((id, i) => id === current[i])) {
      return;
    }
    set({ recentIds: ids });
  },

  async hydrateRecents() {
    const { client, recentIds, sources, recentSources } = get();
    if (!client) return;
    const listed = new Map(sources.map((s) => [s.source_id, s]));
    // Already resolved last pass: an unlisted id (a `cache://` upload) never
    // starts appearing in `listed`, so without this a hydrate triggered by an
    // unrelated catalog change, or by this same hydrate pruning a sibling id,
    // would re-fetch tile_info for it every time.
    const known = new Map(recentSources.map((s) => [s.source_id, s]));

    const results = await Promise.all(
      recentIds.map(async (id) => {
        const desc = listed.get(id) ?? known.get(id);
        if (desc) return { id, desc, evicted: false };
        try {
          // Registry-backed, unlike /api/sources, which reads the catalog and so
          // 404s every `cache://` upload by construction. This is the same call
          // the render path makes, so an id that resolves here is one that will
          // open.
          return {
            id,
            desc: descriptorFromTileInfo(id, await client.http.tileInfo(id)),
            evicted: false,
          };
        } catch (err) {
          // A 404 is the server saying the id names nothing: evicted, or lost to
          // a restart. Anything else -- unreachable, 5xx, a timeout -- says
          // nothing about the id, so the entry stays and simply does not render
          // this pass.
          return { id, desc: null, evicted: err instanceof TensorApiError && err.status === 404 };
        }
      }),
    );

    const gone = results.filter((r) => r.evicted).map((r) => r.id);
    const kept = forgetRecents(get().recentIds, gone);
    if (kept !== get().recentIds) writeRecents(kept);
    set({
      recentIds: kept,
      recentSources: results.flatMap((r) => (r.desc ? [r.desc] : [])),
    });
  },

  openTensor(address, init) {
    const epoch = get().target.epoch + 1;
    resolveController?.abort();
    resolveController = null;
    if (!address) {
      set({ activeSourceId: null, activeTensorId: null, target: { ...IDLE_TARGET, epoch } });
      return;
    }
    // The link may name a pinned address; the selection is always the stable
    // one, and `source_id` is the prefix before the first "/" by the identity
    // policy. No catalog lookup and no `tensors[0]` guess: a bare source_id *is*
    // a valid array_id, and the Flight server resolves it to whatever it binds
    // as the default (biopb/biopb#75). It is also what lets a shared link open
    // while the catalog is capped, still scanning, or missing the source; an id
    // that names nothing fails at the fetch, which can say so.
    const { arrayId: stable } = splitArrayVersion(address);
    const sourceId = stable.split("/", 1)[0] ?? stable;
    // A link counts as opening the source: reaching a `cache://` upload by its
    // returned id is the case the list exists for. An id that names nothing is
    // recorded too, then dropped by the first hydrate that 404s it.
    get().noteRecent(sourceId);
    // The one reset list. Nothing about the previous tensor is inherited:
    // indices are in its grid and a camera is in its world space, so letting
    // either survive frames the next tensor from a point nobody chose, and a
    // `fixedLimits` grey level is a value of its dtype. What a link names
    // replaces the default; without one it is the default. Contrast mode,
    // percentile window, gamma, render mode and label opacity are preferences
    // and carry across.
    set((s) => ({
      activeSourceId: sourceId,
      activeTensorId: stable,
      target: {
        ...IDLE_TARGET,
        epoch,
        requested: address,
        linked: init !== undefined,
        status: "resolving",
        seedSets: init?.visibleSets ?? null,
      },
      position: init ? pickPosition(init.slice) : { t: 0, z: 0, c: 0, axes: {} },
      display: init ? pickDisplay(init.slice) : { ...s.display, fixedLimits: null },
      render3d: init?.render3d ?? false,
      volumeRenderMode: init?.volumeRenderMode ?? s.volumeRenderMode,
      camera3d: init?.camera3d ?? null,
      camera2d: init?.camera2d ?? null,
      labelOpacity: init?.labelOpacity ?? s.labelOpacity,
      // A click leaves the overlay alone: its id names its image, so it is
      // hidden where it does not apply and is still there on a return. A link
      // replaces it, and the sets it names are written under the key when the
      // target lands.
      ...(init
        ? { labelOverlay: init.labelOverlay }
        : {}),
      // A link to particular sets is a link to see them; with the overlay off
      // it would be a link to nothing.
      ...(init?.visibleSets ? { showRois: true } : {}),
      // An axis key means "axis of the tensor in view", so a play in progress
      // does not survive one either.
      playAxis: null,
      runtime: { epoch, planeReady: false, samples: null },
    }));
    void resolveTarget(get, set, epoch);
  },

  retryTarget() {
    const { target } = get();
    if (target.status !== "failed" || !target.requested) return;
    const epoch = target.epoch + 1;
    set({ target: { ...target, epoch, status: "resolving", error: null, retrying: false } });
    void resolveTarget(get, set, epoch);
  },

  setPosition(partial) {
    set((s) => {
      const next = { ...s.position, ...partial };
      return samePosition(s.position, next) ? s : { position: next };
    });
  },

  setDisplay(partial) {
    set((s) => ({ display: { ...s.display, ...partial } }));
  },

  setAxisIndex(axis, value) {
    set((s) => {
      const next: PositionState = axis.named
        ? { ...s.position, [axis.named]: value }
        : { ...s.position, axes: { ...s.position.axes, [axis.key]: value } };
      return samePosition(s.position, next) ? s : { position: next };
    });
  },

  setPlayAxis(key) {
    set({ playAxis: key });
  },

  noteObservedLimits(value, channel) {
    const key = get().target.key;
    set((s) => widenObserved(s, key, value, channel));
  },

  notePlaneSamples(samples, channel, epoch) {
    const key = get().target.key;
    set((s) => {
      if (s.runtime.epoch !== epoch) return s;
      const { values } = samples;
      return {
        runtime: { ...s.runtime, samples },
        ...(values.length > 0
          ? widenObserved(s, key, [values[0] as number, values[values.length - 1] as number], channel)
          : {}),
      };
    });
  },

  setPlaneReady(value, epoch) {
    set((s) =>
      s.runtime.epoch !== epoch || s.runtime.planeReady === value
        ? s
        : { runtime: { ...s.runtime, planeReady: value } },
    );
  },

  async loadRois() {
    const { client, target } = get();
    if (!client || target.status !== "ready" || !target.info || !target.key) return;
    if (get().roisUnavailable) return;
    // Held under the token-free key, fetched under the exact address, which
    // carries the server's current version token.
    const key = target.key;
    const arrayId = target.info.array_id;

    const { roiScopes: landed, roiSets, roisPending: pending } = get().views[key] ?? EMPTY_VIEW;
    // Any scope landing brings the whole listing of set names with it.
    const listed = Object.keys(landed).length > 0;
    // The unqualified listing always; a server-owned set only while it is on
    // screen, which is what makes a set that can reach megabytes lazy. Once a
    // listing has landed it says which names are server-owned, so a link
    // naming a set that is not there costs nothing; before that, the prefix
    // is the only guess there is.
    const wanted = [CLIENT_OWNED_SCOPE];
    for (const name of selectVisibleSets(get()) ?? []) {
      const reserved = listed
        ? roiSets.find((set) => set.setName === name)?.reserved
        : isReservedSetName(name);
      if (reserved) wanted.push(name);
    }
    // Idempotent: already held, or already asked for. Callers fire this from a
    // mount effect, and the 2-D subtree remounts on every render-mode flip.
    const scopes = wanted.filter((scope) => !(scope in landed) && !pending.includes(scope));
    if (scopes.length === 0) return;
    set((s) =>
      withView(s, key, (v) => ({ roisPending: [...v.roisPending, ...scopes], roisError: null })),
    );

    const fetchScope = async (scope: string) => {
      try {
        const result = await client.http.listRois(arrayId, scope || undefined);
        set((s) => withView(s, key, (v) => landRoiScope(v, scope, result)));
      } catch (err) {
        // 501 is the server saying it does not do annotations. Latched for the
        // session so every later tensor skips the round trip.
        if (err instanceof TensorApiError && err.status === 501) {
          set((s) => ({ roisUnavailable: true, ...withView(s, key, { roisPending: [] }) }));
          return;
        }
        // The scope deliberately stays un-landed so a remount or a tensor switch
        // can try again; the effect's own deps keep that from becoming a retry
        // loop.
        set((s) =>
          withView(s, key, (v) => ({
            roisPending: v.roisPending.filter((p) => p !== scope),
            roisError: err instanceof Error ? err.message : String(err),
          })),
        );
      }
    };
    await Promise.all(scopes.map(fetchScope));
  },

  setTool(tool) {
    // A draft belongs to the tool that started it; switching abandons it rather
    // than reinterpreting placed vertices under different rules.
    set((s) =>
      s.tool === tool
        ? s
        : {
            tool,
            views: Object.fromEntries(
              Object.entries(s.views).map(([k, v]) => [k, v.draft ? { ...v, draft: null } : v]),
            ),
          },
    );
  },

  setDraft(draft) {
    set((s) =>
      withView(s, s.target.key, {
        draft: draft ? { shape: draft, sliceKey: sliceKey(s.position) } : null,
      }),
    );
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
      // `null` means "this tensor's default", so materialise that before
      // editing -- otherwise the first toggle would silently also pin whatever
      // the default was broadcasting.
      const current = selectBroadcastAxes(s, defaults);
      return withView(s, s.target.key, {
        broadcastAxes: current.includes(axis)
          ? current.filter((a) => a !== axis)
          : [...current, axis],
      });
    });
  },

  async createRoi(geometry, plane) {
    const { client, newLabel, newSetName } = get();
    const arrayId = get().target.info?.array_id;
    const key = get().target.key;
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
      ...withView(s, key, (v) => ({ rois: [...v.rois, provisional] })),
    }));

    /** Take the provisional row back out of the list it was added to. */
    const withoutProvisional = (rois: RoiAnnotation[]) =>
      rois.filter((roi) => roi.roiId !== provisional.roiId);

    try {
      const result = await client.http.putRois(
        arrayId,
        [{ geometry, plane, label: newLabel, setName: newSetName || undefined }],
        { checkRev: true },
      );
      const stored = result.stored[0];
      if (!stored) {
        set((s) => ({
          roiWriteError: "The server stored no annotation",
          ...withView(s, key, (v) => ({ rois: withoutProvisional(v.rois) })),
        }));
        return;
      }
      // Swapped in place rather than appended, so the annotation does not jump
      // to the end of a list it was already drawn in. Written into this tensor's
      // record even if the user has left it: the annotation is stored, and its
      // own record is the right place for it.
      set((s) => ({
        // Only selected while its tensor is the one in view.
        ...(s.target.key === key ? { selectedRoiId: stored.roiId } : {}),
        ...withView(s, key, (v) => {
          const visible = v.visibleSets;
          return {
            rois: v.rois.map((roi) => (roi.roiId === provisional.roiId ? stored : roi)),
            // The listing's count follows the write, so a later listing does not
            // read this row as someone else's change (see landRoiScope).
            roiSets: withSetCount(v.roiSets, stored.setName, 1),
            // Onto the screen it was drawn on. A materialised list is exactly
            // the sets shown, so a new name -- or one switched off -- would
            // otherwise swallow the shape the user just traced.
            ...(visible !== null && !visible.includes(stored.setName)
              ? { visibleSets: [...visible, stored.setName] }
              : {}),
          };
        }),
      }));
    } catch (err) {
      set((s) => ({
        roiWriteError: err instanceof Error ? err.message : String(err),
        ...withView(s, key, (v) => ({ rois: withoutProvisional(v.rois) })),
      }));
    }
  },

  async deleteRoi(roiId) {
    const { client } = get();
    const arrayId = get().target.info?.array_id;
    const key = get().target.key;
    if (!client || !arrayId) return;
    // Nothing to ask the server about: this row is a local placeholder for a
    // write still in flight, and its id is one this client invented. Left for
    // the write to resolve, which is a round trip away.
    if (isProvisionalRoiId(roiId)) return;

    // Gone on the keystroke, not on the answer. The row is under the pointer
    // when Delete is pressed, so leaving it there for a round trip reads as the
    // key having missed -- and the far commoner outcome by far is that the
    // delete succeeds.
    const held = (get().views[key ?? ""] ?? EMPTY_VIEW).rois;
    const index = held.findIndex((roi) => roi.roiId === roiId);
    const removed = held[index];
    set((s) => ({
      roiWriteError: null,
      selectedRoiId: s.selectedRoiId === roiId ? null : s.selectedRoiId,
      ...withView(s, key, (v) => ({
        rois: v.rois.filter((roi) => roi.roiId !== roiId),
        roiSets: removed ? withSetCount(v.roiSets, removed.setName, -1) : v.roiSets,
      })),
    }));

    try {
      await client.http.deleteRois(arrayId, [roiId]);
      // An id the server did not have is already gone as far as the UI is
      // concerned, so an empty answer is not a failure: the row stays removed
      // rather than coming back as one that cannot be got rid of.
    } catch (err) {
      set((s) => {
        const failed = { roiWriteError: err instanceof Error ? err.message : String(err) };
        if (!removed) return failed;
        return {
          ...failed,
          ...withView(s, key, (v) => {
            const rois = [...v.rois];
            rois.splice(Math.min(index, rois.length), 0, removed);
            return { rois, roiSets: withSetCount(v.roiSets, removed.setName, 1) };
          }),
        };
      });
    }
  },

  async clearRoiSet(setName) {
    const { client } = get();
    const arrayId = get().target.info?.array_id;
    const key = get().target.key;
    if (!client || !arrayId) return;
    set({ roiWriteError: null });
    try {
      // No ids: the server drops the whole set in one transaction, so this is
      // not limited to the rows the cap let this client see.
      await client.http.deleteRois(arrayId, undefined, { setName });
      // Filtered by name rather than by the ids that came back, for the same
      // reason: what was deleted is the set, and the response enumerates only
      // what the server chose to list.
      set((s) => {
        const held = (s.views[key ?? ""] ?? EMPTY_VIEW).rois;
        return {
          selectedRoiId:
            held.find((roi) => roi.roiId === s.selectedRoiId)?.setName === setName
              ? null
              : s.selectedRoiId,
          ...withView(s, key, (v) => ({
            rois: v.rois.filter((roi) => roi.setName !== setName),
            // Gone from the listing too, as the server's own next listing would
            // have it: it enumerates stored rows, and there are none.
            roiSets: v.roiSets.filter((set) => set.setName !== setName),
            // The name means nothing once the set is gone, and leaving it on the
            // list would pin a later set of the same name to whatever this one
            // was toggled to.
            ...(v.visibleSets?.includes(setName)
              ? { visibleSets: v.visibleSets.filter((name) => name !== setName) }
              : {}),
          })),
        };
      });
    } catch (err) {
      set({ roiWriteError: err instanceof Error ? err.message : String(err) });
    }
  },

  setShowRois(value) {
    set((s) => (s.showRois === value ? s : { showRois: value }));
  },

  setLabelOverlay(arrayId) {
    set((s) => (s.labelOverlay === arrayId ? s : { labelOverlay: arrayId }));
  },

  setLabelOpacity(value) {
    const opacity = clampLabelOpacity(value);
    set((s) => (s.labelOpacity === opacity ? s : { labelOpacity: opacity }));
  },

  toggleSetVisible(setName) {
    set((s) => {
      // `null` means the default, so materialise that before editing --
      // otherwise the first toggle would silently also hide every other
      // client-owned set.
      const current = selectVisibleSets(s) ?? defaultVisibleSets(s);
      return withView(s, s.target.key, {
        visibleSets: current.includes(setName)
          ? current.filter((name) => name !== setName)
          : [...current, setName],
      });
    });
  },

  applyViewerState(params) {
    const requested = params.get("id");
    if (!requested) return false;
    const s = get();
    // Everything the link does not name is the default, not whatever the
    // previous tensor left: `openTensor` resets, and this only supplies what
    // the link chose. Encoding omits a field only when it is at the default
    // used here, so a link this app wrote round-trips exactly.
    //
    // The percentile window and gamma are preferences rather than properties of
    // the tensor, so they are the fallback for a link that names none, as are
    // the render mode and the label overlay's opacity.
    const init = decodeViewerState(params, {
      ...DEFAULT_VIEWER_URL_STATE,
      arrayId: requested,
      slice: {
        ...DEFAULT_VIEWER_URL_STATE.slice,
        contrastMode: s.display.contrastMode,
        percentileScale: s.display.percentileScale,
        gamma: s.display.gamma,
      },
      volumeRenderMode: s.volumeRenderMode,
      labelOpacity: s.labelOpacity,
    });
    get().openTensor(requested, init);
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
      const { client, sources, activeSourceId, target, openTensor } = get();
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

        if (catalogFingerprint(sources) !== catalogFingerprint(sorted)) {
          set({ sources: sorted });

          // A catalog response is a listing, not proof that an unlisted source
          // is gone: it may be capped, still scanning, or temporarily failed.
          // In particular, retain a source selected from a shared URL, which
          // deliberately does not need to be present in the listing.
          if (
            activeSourceId &&
            !target.linked &&
            !sorted.find((s) => s.source_id === activeSourceId)
          ) {
            openTensor(null);
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

  sourceJobs: {},

  async startResolve(sourceId: string) {
    await startSourceJob(get, set, "resolve", sourceId);
  },

  async startWarm(sourceId: string) {
    await startSourceJob(get, set, "warm", sourceId);
  },

  async cancelSourceJob(kind: SourceJobKind, sourceId: string) {
    const { client } = get();
    if (!client) return;
    try {
      const status = await client.http.cancelJob(kind, sourceId);
      putJob(set, status);
    } catch {
      // The server is the only thing that can actually stop the recall, and it
      // re-reports the flag on the next poll. A failed cancel is worth nothing
      // to say here -- the button simply hasn't taken yet.
    }
  },

  dismissSourceJob(kind: SourceJobKind, sourceId: string) {
    set((s) => {
      const next = { ...s.sourceJobs };
      delete next[jobKey(kind, sourceId)];
      return { sourceJobs: next };
    });
  },

  stopJobPolling() {
    if (_jobPollTimerId) {
      clearInterval(_jobPollTimerId);
      _jobPollTimerId = undefined;
    }
  },
}));

type Get = () => AppState;
type Set = (partial: Partial<AppState> | ((s: AppState) => Partial<AppState>)) => void;

function putJob(set: Set, status: SourceJobStatus): void {
  set((s) => ({
    sourceJobs: {
      ...s.sourceJobs,
      [jobKey(status.kind, status.source_id)]: status,
    },
  }));
}

async function startSourceJob(
  get: Get,
  set: Set,
  kind: SourceJobKind,
  sourceId: string,
): Promise<void> {
  const { client } = get();
  if (!client) return;
  try {
    const status =
      kind === "resolve"
        ? await client.http.startResolve(sourceId)
        : await client.http.startWarm(sourceId);
    putJob(set, status);
    ensureJobPolling(get, set);
  } catch (err) {
    // Synthesised rather than swallowed: a resolve that never started is the
    // one failure the user most needs told about, and the surface that shows
    // it reads `sourceJobs` -- so there has to be an entry for it to read.
    putJob(set, {
      kind,
      source_id: sourceId,
      state: "error",
      progress: {},
      error: err instanceof Error ? err.message : String(err),
      elapsed_seconds: 0,
      cancel_requested: false,
    });
  }
}

function ensureJobPolling(get: Get, set: Set): void {
  if (_jobPollTimerId) return;
  _jobPollTimerId = setInterval(() => {
    void pollSourceJobs(get, set);
  }, JOB_POLL_MS);
}

async function pollSourceJobs(get: Get, set: Set): Promise<void> {
  const { client, sourceJobs } = get();
  const running = Object.values(sourceJobs).filter((j) => !isSettled(j));
  if (!client || running.length === 0) {
    get().stopJobPolling();
    return;
  }

  // In parallel, not one at a time: each is an independent HTTP request, and a
  // sequential await-in-loop would make a tick's cost grow with job count and
  // risk overlapping the next tick if it ran long.
  const results = await Promise.allSettled(
    running.map((j) => client.http.jobStatus(j.kind, j.source_id)),
  );
  for (const result of results) {
    if (result.status === "rejected") {
      // A blip leaves the previous status in place; the next tick retries. Not
      // marked failed: the recall is server-side and unaffected by our poll.
      continue;
    }
    const after = result.value;
    putJob(set, after);
    if (isSettled(after)) {
      await onJobSettled(get, after);
    }
  }
}

/**
 * What happens when a job stops.
 *
 * A finished resolve leaves the catalog row stale -- the source is hydrated but
 * the tree still has the listing from before -- so the list is re-read.
 *
 * The warm that used to follow it is gated off; see AUTO_WARM_AFTER_RESOLVE.
 * When on it is unconditional, with no check for whether the source is
 * multi-file, because the server answers that structurally -- a single-file
 * source's warm finishes at once with `files_total === 0` and the tray never
 * shows a bar for it. Keeping a list of multi-file source types on this side
 * would be a copy that drifts.
 */
async function onJobSettled(get: Get, job: SourceJobStatus): Promise<void> {
  if (job.kind !== "resolve" || job.state !== "done") return;
  if (!AUTO_WARM_AFTER_RESOLVE) {
    await get().loadSources();
    return;
  }
  // Independent: the catalog reload and starting the warm hit different
  // endpoints and different store slices, so there is nothing for one to wait
  // on from the other.
  await Promise.all([get().loadSources(), get().startWarm(job.source_id)]);
}

/** The epoch's in-flight `tile_info`, aborted when a newer open supersedes it. */
let resolveController: AbortController | null = null;

/**
 * One slow response, from a cold catalog or a moment of load, must not cost the
 * tensor its viewer -- but a server that blew an 8 s budget twice is not
 * rescued by a third ask, and every attempt holds the pane empty.
 */
const TILE_INFO_RETRY_MS = [500];

/**
 * Fetch `tile_info` for the open target and land it, unless a newer open
 * superseded this one.
 */
async function resolveTarget(
  get: () => AppState,
  set: (partial: Partial<AppState> | ((s: AppState) => Partial<AppState>)) => void,
  epoch: number,
): Promise<void> {
  const current = () => get().target.epoch === epoch;
  const { client, target } = get();
  const requested = target.requested;
  if (!requested) return;
  const fail = (reason: string, kind: ViewerErrorKind) => {
    if (!current()) return;
    set((s) => ({
      target: { ...s.target, status: "failed", retrying: false, error: { reason, kind } },
    }));
  };
  if (!client) {
    fail("not connected to a server", "transport");
    return;
  }
  const controller = new AbortController();
  resolveController = controller;
  for (let attempt = 0; ; attempt += 1) {
    try {
      const info = await client.http.tileInfo(requested, { signal: controller.signal });
      if (!current()) return;
      const key = splitArrayVersion(info.array_id).arrayId;
      set((s) => ({
        target: {
          ...s.target,
          status: "ready",
          key,
          info,
          retrying: false,
          error: null,
          seedSets: null,
        },
        // The grid is the first thing that can say what an index may be, so the
        // slice is bounded here rather than where it was read -- see clampSliceTo.
        position: samePositionOr(s.position, clampSliceTo(s.position, info)),
        // A link's set names belong to the tensor it resolved to, so they are
        // written under its key rather than under whatever the link spelled.
        ...withView(s, key, {
          // Which sets to draw belongs to the tensor, so a link that names none
          // opens on the tensor's default rather than on the last choice.
          ...(s.target.linked ? { visibleSets: s.target.seedSets } : {}),
          // Rows kept from an earlier visit stay on screen, but are listed again.
          ...(s.views[key] ? { roiScopes: {} } : {}),
        }),
      }));
      return;
    } catch (err) {
      if (!current() || err instanceof TensorAbortError) return;
      const transport = isTransportError(err);
      const delay = transport ? TILE_INFO_RETRY_MS[attempt] : undefined;
      if (delay === undefined) {
        fail(err instanceof Error ? err.message : String(err), transport ? "transport" : "capability");
        return;
      }
      set((s) => ({ target: { ...s.target, retrying: true } }));
      await new Promise((resolve) => setTimeout(resolve, delay));
      if (!current()) return;
    }
  }
}

/**
 * The grid for the tensor in view, or null until it has resolved.
 *
 * Nothing to pair against the request any more: the target resets it on every
 * open, and a viewer is only mounted once it is there.
 */
export function selectTileInfo(s: AppState): TileInfo | null {
  return s.target.info;
}

/**
 * The label set to draw over the tensor in view, or null.
 *
 * Scoped by the id itself rather than by a per-tensor record: a set's
 * `array_id` names its image, so the question "is this overlay about what is on
 * screen?" is answered by the two ids and nothing has to be kept in step.
 *
 * Compared against the *stable* address on both sides, because a pinned link
 * (`id@token`) is the same tensor as the id the tree offered sets for.
 */
export function selectLabelOverlay(s: AppState): string | null {
  const shown = s.target.key;
  if (!shown || !s.labelOverlay) return null;
  const address = splitLabelArrayId(s.labelOverlay);
  if (!address) return null;
  return address.imageArrayId === shown ? s.labelOverlay : null;
}

/** A patch widening `key`'s observed union on `channel` to cover `value`, or none. */
function widenObserved(
  s: AppState,
  key: string | null,
  value: [number, number],
  channel: number,
): Partial<AppState> {
  const prev = key ? s.views[key]?.observedLimits[channel] : undefined;
  if (prev && prev[0] <= value[0] && prev[1] >= value[1]) return {};
  const next: [number, number] = prev
    ? [Math.min(prev[0], value[0]), Math.max(prev[1], value[1])]
    : [value[0], value[1]];
  return withView(s, key, (cur) => ({
    observedLimits: { ...cur.observedLimits, [channel]: next },
  }));
}

/**
 * The record of the tensor in view, or an empty one while none has resolved.
 *
 * Every per-tensor read goes through here, so nothing compares an id: a tensor
 * that is not in view simply is not the one this returns.
 */
export function selectView(s: AppState): TensorView {
  return (s.target.key ? s.views[s.target.key] : undefined) ?? EMPTY_VIEW;
}

/** The levels this tensor's current channel has shown, or null if none yet. */
export function selectObservedLimits(s: AppState): [number, number] | null {
  return selectView(s).observedLimits[s.position.c] ?? null;
}

/**
 * The sampled min and max grey level of the plane last sampled, or null before
 * one has been. What the automatic window would be with neither tail trimmed;
 * in fixed mode the window no longer says anything about the data, so this is
 * what a fixed window is reset onto.
 *
 * Returns a fresh array: read it with `useShallow`.
 */
export function selectPlaneLimits(s: AppState): [number, number] | null {
  const samples = s.runtime.samples;
  return samples ? contrastLimitsFrom(samples.values, 0, 100) : null;
}

/**
 * The track a contrast window is chosen on: the dtype's own range, or on a
 * float tensor the levels the data has shown. Null until the grid is there --
 * the panel then falls back to what the catalog says of the dtype.
 *
 * Derived, not published (biopb/biopb#955): the viewer's shader and the panel's
 * bar both call this, so a plane change cannot leave the bar on a track the
 * shader is not clamping into. Returns a fresh array: read it with `useShallow`.
 */
export function selectContrastTrack(s: AppState): [number, number] | null {
  const info = s.target.info;
  if (!info) return null;
  return contrastTrack(
    vivDtype(info.dtype),
    selectObservedLimits(s),
    selectPlaneLimits(s),
    s.display.fixedLimits,
  );
}

/**
 * The contrast window the shader uses, and the panel seeds a fixed window from:
 * the user's fixed levels brought inside the track, or the percentile window of
 * the plane last sampled -- or the whole track while none has been. Null until
 * the grid is there. Returns a fresh array: read it with `useShallow`.
 */
export function selectContrastWindow(s: AppState): [number, number] | null {
  const info = s.target.info;
  const track = selectContrastTrack(s);
  if (!info || !track) return null;
  const { contrastMode, fixedLimits, percentileScale } = s.display;
  // A fixed window is the user's, not the plane's: it is not re-derived per
  // plane, only brought inside the track it is being applied to.
  if (contrastMode === "fixed") {
    return fixedLimits ? clampContrastLimits(fixedLimits, track, vivDtype(info.dtype)) : track;
  }
  const samples = s.runtime.samples;
  if (!samples) return track;
  const [lo, hi] = percentileBounds(percentileScale);
  return contrastLimitsFrom(samples.values, lo, hi);
}

// --- ROI annotations -------------------------------------------------------
//
// Every one of these hides state belonging to another tensor rather than
// relying on someone having reset it: an async writer can land after the tensor
// changed, and a reset cannot catch that.
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

/** The scope of the unqualified listing: every client-owned set at once. */
export const CLIENT_OWNED_SCOPE = "";

export function selectRois(s: AppState): RoiAnnotation[] {
  return selectView(s).rois;
}

/** Every set on the tensor in view, server-owned ones included. */
export function selectRoiSets(s: AppState): RoiSetInfo[] {
  return selectView(s).roiSets;
}

/** The scopes landed for the tensor in view, with what each fetch said. */
export function selectRoiScopes(s: AppState): Record<string, RoiScopeState> {
  return selectView(s).roiScopes;
}

/** The scopes in flight for the tensor in view -- not one looked at earlier. */
export function selectRoisPending(s: AppState): string[] {
  return selectView(s).roisPending;
}

export function selectRoisError(s: AppState): string | null {
  return selectView(s).roisError;
}

/**
 * The sets this tensor shows by default, spelled out: every client-owned one
 * the listing knows or the rows hold. What a toggle edits when nothing has
 * been chosen yet.
 */
function defaultVisibleSets(s: AppState): string[] {
  const names = [
    ...selectRoiSets(s).filter((set) => !set.reserved).map((set) => set.setName),
    ...roiSetCounts(selectRois(s)).map((set) => set.setName),
  ];
  return [...new Set(names)].filter((name) => !isReservedSetName(name));
}

/** `roiSets` with one set's stored count moved by `delta`, added if it is new. */
function withSetCount(sets: RoiSetInfo[], setName: string, delta: number): RoiSetInfo[] {
  if (!sets.some((set) => set.setName === setName)) {
    return delta > 0
      ? [...sets, { setName, count: delta, reserved: isReservedSetName(setName) }]
      : sets;
  }
  return sets.map((set) =>
    set.setName === setName ? { ...set, count: Math.max(0, set.count + delta) } : set,
  );
}

/** The rows a scope's fetch answers for. */
function inScope(roi: RoiAnnotation, scope: string): boolean {
  return scope === CLIENT_OWNED_SCOPE ? !isReservedSetName(roi.setName) : roi.setName === scope;
}

/**
 * Fold a landed fetch into the store.
 *
 * The first landing for a tensor replaces everything held for the last one --
 * the tensor is the unit of eviction -- and a later one merges: its rows
 * replace the rows of its own scope, and a provisional row is kept wherever it
 * is, since its write is still in flight.
 *
 * Every response enumerates the whole tensor's sets with their stored counts,
 * so it also says whether the rows held for *other* scopes are still current.
 * A scope whose counts disagree is un-landed, and the next `loadRois` fetches
 * it again -- which is how a reserved set the server rebuilt on a
 * re-registration, or a set another client wrote to, catches up without a
 * reload. A clipped scope is exempt: it disagrees by construction.
 *
 * Counts are a proxy: a per-set version in `RoiSetInfo` would make this an
 * exact comparison and let the writes stop shadowing the counts.
 */
function landRoiScope(
  v: TensorView,
  scope: string,
  result: RoiListResult,
): Partial<TensorView> {
  const kept = v.rois.filter((roi) => !inScope(roi, scope) || isProvisionalRoiId(roi.roiId));
  const rois = [...kept, ...result.rois];
  const scopes: Record<string, RoiScopeState> = { ...v.roiScopes };
  scopes[scope] = { truncated: result.truncated, skipped: result.skipped };

  const stored = new Map(result.sets.map((set) => [set.setName, set.count]));
  const held = new Map(
    roiSetCounts(rois.filter((roi) => !isProvisionalRoiId(roi.roiId))).map((set) => [
      set.setName,
      set.count,
    ]),
  );
  const total = (counts: Map<string, number>, names: string[]) =>
    names.reduce((sum, name) => sum + (counts.get(name) ?? 0), 0);
  for (const [other, state] of Object.entries(scopes)) {
    if (other === scope || state.truncated) continue;
    const names =
      other === CLIENT_OWNED_SCOPE
        ? [...new Set([...stored.keys(), ...held.keys()])].filter((name) => !isReservedSetName(name))
        : [other];
    if (total(stored, names) !== total(held, names) + state.skipped) delete scopes[other];
  }

  return {
    rois,
    roiSets: result.sets,
    roiScopes: scopes,
    roisPending: v.roisPending.filter((p) => p !== scope),
  };
}

/**
 * A stable key for the slice position, for spotting that it has moved.
 *
 * Only the indices: contrast, gamma and the percentile window ride `SliceState`
 * too, and none of them invalidate a shape being drawn.
 */
export function sliceKey(slice: PositionState): string {
  return `${slice.t}|${slice.z}|${slice.c}|${JSON.stringify(slice.axes)}`;
}

/**
 * A link's own set names while its target is not ready, or the resolved ones.
 *
 * For the URL writer. The scoped selectors answer null both for "the tensor's
 * default" and for "no tensor resolved yet", and the writer reads null as
 * "drop the param" -- so a link whose `tile_info` is slow or failed would have
 * its `rs=` rewritten away, and a reload could no longer retry it. Until the
 * target is ready, a linked open still says what the link named (`[]` being an
 * explicit "none", null "no value").
 */
export function selectUrlVisibleSets(s: AppState): string[] | null {
  const { target } = s;
  return target.linked && target.status !== "ready" ? target.seedSets : selectVisibleSets(s);
}

/** A link's own label overlay while its target is not ready. See {@link selectUrlVisibleSets}. */
export function selectUrlLabelOverlay(s: AppState): string | null {
  const { target } = s;
  return target.linked && target.status !== "ready" ? s.labelOverlay : selectLabelOverlay(s);
}

/**
 * The sets chosen for the tensor in view, or null for its default. Names do not
 * carry across tensors.
 */
export function selectVisibleSets(s: AppState): string[] | null {
  return selectView(s).visibleSets;
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
  if (s.render3d) return null;
  const draft = selectView(s).draft;
  // Vertices were traced against the pixels of one plane. Navigating away --
  // the slider, a keyboard scroll, or play stepping an axis -- makes them a
  // shape drawn on an image nobody is looking at any more, and finishing there
  // would pin it to the plane it was NOT drawn on. Answered at the read, so
  // every route that moves the slice is covered without naming any of them.
  return draft && draft.sliceKey === sliceKey(s.position) ? draft.shape : null;
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
  return selectView(s).broadcastAxes ?? defaults;
}
