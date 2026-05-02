/**
 * Global app state store (Zustand).
 *
 * Holds:
 *  - TensorHttpClient singleton (initialised once after /api/token fetch)
 *  - Source list
 *  - Active selection (sourceId, tensorId)
 *  - Slice state (t, z, c)
 *  - Connection status
 */

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

  // Actions
  initClient: (apiBase: string, token: string | null, devMode: boolean) => void;
  loadSources: () => Promise<void>;
  selectSource: (sourceId: string, tensorId?: string) => void;
  setSlice: (partial: Partial<SliceState>) => void;
  clearSession: () => void;
}

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
      set({ sources, sourcesLoading: false, connectionState: "connected" });
    } catch (err) {
      set({
        sourcesLoading: false,
        connectionState: "error",
        connectionError: err instanceof Error ? err.message : String(err),
      });
    }
  },

  selectSource(sourceId, tensorId) {
    const { sources } = get();
    const src = sources.find((s) => s.source_id === sourceId);
    const tid = tensorId ?? src?.tensors[0]?.array_id ?? null;
    set({ activeSourceId: sourceId, activeTensorId: tid });
    // Reset slice to zero when switching source
    set((s) => ({ slice: { ...s.slice, t: 0, z: 0, c: 0 } }));
  },

  setSlice(partial) {
    set((s) => ({ slice: { ...s.slice, ...partial } }));
  },

  clearSession() {
    sessionStorage.removeItem("biopb_token");
    window.location.href = "/unlock";
  },
}));
