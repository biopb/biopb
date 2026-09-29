import type { StateCreator } from "zustand";
import type { SliderAxis, TileInfo } from "@biopb/tensor-flight-client";
import {
  TensorAbortError,
  isTransportError,
  splitArrayVersion,
} from "@biopb/tensor-flight-client";
import { DEFAULT_VIEWER_URL_STATE, decodeViewerState } from "../utils/viewerUrl";
import type { ViewerUrlState } from "../utils/viewerUrl";
import { clampSliceTo } from "../utils/vivUtils";
import {
  type DisplayState,
  type PositionState,
  moved,
  pickDisplay,
  pickPosition,
  samePositionOr,
} from "./slice";
import { INITIAL_RUNTIME } from "./runtime";
import { withView } from "./tensorView";
import type { AppState } from "./types";

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

/**
 * The 3-D camera, in `OrbitView`'s own terms.
 *
 * A mirror, not the source of truth: deck.gl owns the camera while the volume
 * is mounted and this trails it on a debounce, which is what keeps an orbit
 * smooth. It is read back only to seed the next mount -- see `VolumeViewer`.
 */
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

/**
 * Which tensor is in view and where in it: the target, the position and display
 * settings, the view mode and the cameras. Everything `openTensor` resets.
 */
export interface TargetSlice {
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
  /**
   * The axis being scrubbed automatically (a `SliderAxis.key`), or null.
   *
   * Not part of `SliceState`: play changes which index is asked for over time,
   * not what a frame looks like, and folding it in would put a transient of the
   * UI into the object the viewer diffs its refetches on.
   */
  playAxis: string | null;
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
   * Where the 3-D camera is, or null for "wherever the volume fits".
   *
   * Null rather than a computed default because the fitted camera depends on
   * the volume and the pane size, neither of which the store knows; only the
   * viewer can work it out, so the store says "unset" and lets it.
   */
  camera3d: Camera3DState | null;
  /** Where the 2-D camera is, or null for "wherever the plane fits". */
  camera2d: Camera2DState | null;
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
  /** Move within the grid. A move to where the view already is keeps the object. */
  setPosition: (partial: Partial<PositionState>) => void;
  setDisplay: (partial: Partial<DisplayState>) => void;
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
  setRender3d: (value: boolean) => void;
  setCamera3d: (value: Camera3DState | null) => void;
  setCamera2d: (value: Camera2DState | null) => void;
}

/** The epoch's in-flight `tile_info`, aborted when a newer open supersedes it. */
let resolveController: AbortController | null = null;

export const createTargetSlice: StateCreator<AppState, [], [], TargetSlice> = (set, get) => ({
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

  playAxis: null,
  render3d: false,
  camera3d: null,
  camera2d: null,

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
      runtime: INITIAL_RUNTIME,
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
      return moved(s, next);
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
      return moved(s, next);
    });
  },

  setPlayAxis(key) {
    set({ playAxis: key });
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

  setRender3d(value) {
    set({ render3d: value });
  },

  setCamera3d(value) {
    set({ camera3d: value });
  },

  setCamera2d(value) {
    set({ camera2d: value });
  },
});

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
