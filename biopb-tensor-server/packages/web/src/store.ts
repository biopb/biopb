import { create } from "zustand";
import { TensorFlightClient } from "@biopb/tensor-flight-client";
import type { DataSourceDescriptor, QuerySourcesResult } from "@biopb/tensor-flight-client";
import { type ColorValue, extractChannelNames } from "./utils/colorUtils";

export type ConnectionState = "idle" | "connecting" | "connected" | "error";

export interface SliceState {
  t: number;
  z: number;
  c: number;
  reductionMethod: string;
  percentileScale: number;  // 0 = min-max, 1 = 1-99 percentile, 2 = 2-98 percentile
  useMinMax: boolean;  // When true, use full min-max range (0-100 percentile)
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
  activeTensorId: string | null;

  // Slice controls
  slice: SliceState;

  // UI options
  showAdvancedOptions: boolean;

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
  setShowAdvancedOptions: (value: boolean) => void;
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
  apiBase: "http://localhost:8814",

  sources: [],
  sourcesLoading: false,
  scanning: false,

  activeSourceId: null,
  activeTensorId: null,

  slice: {
    t: 0,
    z: 0,
    c: 0,
    reductionMethod: "nearest",
    percentileScale: 1,  // Default 1-99 percentile
    useMinMax: false,
  },

  showAdvancedOptions: false,

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
      set({ activeSourceId: null, activeTensorId: null });
      return;
    }
    const { sources } = get();
    const src = sources.find((s) => s.source_id === sourceId);
    const tid = tensorId ?? src?.tensors[0]?.array_id ?? null;
    set({ activeSourceId: sourceId, activeTensorId: tid });
    set((s) => ({ slice: { ...s.slice, t: 0, z: 0, c: 0 } }));
  },

  setSlice(partial) {
    set((s) => ({ slice: { ...s.slice, ...partial } }));
  },

  setShowAdvancedOptions(value) {
    set({ showAdvancedOptions: value });
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
    window.location.href = "/unlock";
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
