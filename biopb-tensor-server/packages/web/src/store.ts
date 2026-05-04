import { create } from "zustand";
import { TensorFlightClient } from "@biopb/tensor-flight-client";
import type { DataSourceDescriptor } from "@biopb/tensor-flight-client";

export type ConnectionState = "idle" | "connecting" | "connected" | "error";

export interface SliceState {
  t: number;
  z: number;
  c: number;
  reductionMethod: string;
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

  // Active selection
  activeSourceId: string | null;
  activeTensorId: string | null;

  // Slice controls
  slice: SliceState;

  // Catalog polling
  pollingInterval: number;

  // Actions
  initClient: (apiBase: string, token: string | null, devMode: boolean) => void;
  loadSources: () => Promise<void>;
  selectSource: (sourceId: string | null, tensorId?: string) => void;
  setSlice: (partial: Partial<SliceState>) => void;
  clearSession: () => void;
  startCatalogPolling: () => void;
  stopCatalogPolling: () => void;
}

// Internal timer storage (non-reactive, module-level)
let _pollingTimerId: ReturnType<typeof setInterval> | undefined;

export const useAppStore = create<AppState>((set, get) => ({
  client: null,
  connectionState: "idle",
  connectionError: null,
  devMode: false,
  apiBase: "http://localhost:8816",

  sources: [],
  sourcesLoading: false,

  activeSourceId: null,
  activeTensorId: null,

  slice: {
    t: 0,
    z: 0,
    c: 0,
    reductionMethod: "nearest",
  },

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
