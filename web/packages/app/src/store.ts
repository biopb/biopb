import { create } from "zustand";
import { TensorFlightClient } from "@biopb/tensor-flight-client";
import type { DataSourceDescriptor, QuerySourcesResult, TileInfo } from "@biopb/tensor-flight-client";
import { withBase } from "./base";
import { decodeViewerState } from "./utils/viewerUrl";
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
  percentileScale: number;  // 0 = min-max, 1 = 1-99 percentile, 2 = 2-98 percentile
  useMinMax: boolean;  // When true, use full min-max range (0-100 percentile)
  // Display-only exponent applied to the normalized intensity, after the
  // contrast window and before the channel color. 1 leaves the ramp linear;
  // below 1 lifts the dim end, above 1 pushes it down.
  gamma: number;
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
   */
  tileInfo: TileInfo | null;

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
  setTileInfo: (value: TileInfo | null) => void;
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
  setShowAdvancedOptions: (value: boolean) => void;
  setRender3d: (value: boolean) => void;
  setVolumeRenderMode: (value: VolumeRenderMode) => void;
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
    percentileScale: 1,  // Default 1-99 percentile
    useMinMax: false,
    gamma: 1,
  },

  tileInfo: null,

  showAdvancedOptions: false,
  render3d: false,
  volumeRenderMode: DEFAULT_VOLUME_RENDER_MODE,

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
      tileInfo: null,
    });
    set((s) => ({ slice: { ...s.slice, t: 0, z: 0, c: 0, axes: {} } }));
  },

  setSlice(partial) {
    set((s) => ({ slice: { ...s.slice, ...partial } }));
  },

  setTileInfo(value) {
    // The grid is the first thing that can say what an index may be, so the
    // slice is bounded here rather than where it was read -- see clampSliceTo.
    set((s) => ({ tileInfo: value, slice: clampSliceTo(s.slice, value) }));
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
    const next = decodeViewerState(params, {
      arrayId: stable,
      slice: s.slice,
      render3d: s.render3d,
      volumeRenderMode: s.volumeRenderMode,
    });
    set({
      activeSourceId: stable.split("/", 1)[0] ?? null,
      activeTensorId: stable,
      requestedArrayId: requested,
      slice: next.slice,
      render3d: next.render3d,
      volumeRenderMode: next.volumeRenderMode,
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
      const { client, sources, activeSourceId, selectSource } = get();
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

          // Clear selection if active source was removed
          if (activeSourceId && !sorted.find((s) => s.source_id === activeSourceId)) {
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
