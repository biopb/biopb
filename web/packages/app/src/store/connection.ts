import type { StateCreator } from "zustand";
import { TensorFlightClient } from "@biopb/tensor-flight-client";
import type { QuerySourcesResult } from "@biopb/tensor-flight-client";
import { withBase } from "../base";
import { catalogIsFilling } from "../utils/catalogHealth";
import type { AppState } from "./types";

export type ConnectionState = "idle" | "connecting" | "connected" | "error";

/** Delays between catalog polls: a quiet catalog is asked about less and less. */
export const POLL_BACKOFF_MS = [30_000, 60_000, 120_000, 300_000];

/** The backoff step after a poll: back to the start while the catalog is
 *  changing or still being indexed, one further out while it is not. */
export function nextPollStep(step: number, active: boolean): number {
  return active ? 0 : Math.min(step + 1, POLL_BACKOFF_MS.length - 1);
}

// Internal timer storage (non-reactive, module-level). The generation tells a
// poll still in flight that the loop it belonged to was stopped.
let _pollingTimerId: ReturnType<typeof setTimeout> | undefined;
let _pollingGeneration = 0;

/** The client, and the connection's health. */
export interface ConnectionSlice {
  // Client
  client: TensorFlightClient | null;
  connectionState: ConnectionState;
  connectionError: string | null;
  devMode: boolean;
  apiBase: string;

  // The top of the catalog is being read for the first time.
  sourcesLoading: boolean;
  // Progressive discovery: the server is SERVING but its catalog scan is still
  // running. Lets the source list show "Indexing…" instead of "No sources" when
  // the catalog is briefly empty at startup. Refreshed from /readyz.
  scanning: boolean;
  initClient: (apiBase: string, token: string | null, devMode: boolean) => void;
  loadCatalog: () => Promise<void>;
  querySources: (sql: string) => Promise<QuerySourcesResult>;
  clearSession: () => void;
  startCatalogPolling: () => void;
  stopCatalogPolling: () => void;
}

export const createConnectionSlice: StateCreator<AppState, [], [], ConnectionSlice> = (set, get) => ({
  client: null,
  connectionState: "idle",
  connectionError: null,
  devMode: false,
  apiBase: withBase("/data_plane"),

  sourcesLoading: false,
  scanning: false,

  initClient(apiBase, token, devMode) {
    get().resetTree();
    set({
      client: new TensorFlightClient(apiBase, token),
      connectionState: "connecting",
      connectionError: null,
      devMode,
      apiBase,
    });
  },

  async loadCatalog() {
    const { client } = get();
    if (!client) return;
    set({ sourcesLoading: true });
    try {
      await get().loadLevel("");
      set({ sourcesLoading: false, connectionState: "connected" });
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

  clearSession() {
    sessionStorage.removeItem("biopb_token");
    window.location.href = withBase("/unlock");
  },

  startCatalogPolling() {
    get().stopCatalogPolling();
    const generation = ++_pollingGeneration;
    let step = 0;

    // One poll. True when the next one should come soon: the tree moved, the
    // catalog is still being indexed, or no answer was had (not connected, or a
    // failure -- a backoff must not widen while the server is the thing that is
    // wrong).
    const poll = async (): Promise<boolean> => {
      const { client } = get();
      if (!client || get().connectionState !== "connected") return true;

      // The tree and the scan-in-progress flag are independent. The flag clears
      // the "Indexing…" hint once the background catalog scan finishes
      // (best-effort; a readyz blip just leaves the previous value).
      const [changed, readyz] = await Promise.all([
        get().refreshTree(),
        client.http.readyz().catch(() => null),
      ]);
      // Stopped, or pointed at another server, while this was in flight.
      if (generation !== _pollingGeneration || get().client !== client) return true;
      if (readyz) set({ scanning: catalogIsFilling(readyz.backend_health) });
      return changed || get().scanning;
    };

    const tick = async () => {
      let active = true;
      try {
        active = await poll();
      } catch (err) {
        // Silent failure - don't change connection state for transient errors
        console.warn("Catalog polling error:", err);
      }
      if (generation !== _pollingGeneration) return;
      step = nextPollStep(step, active);
      _pollingTimerId = setTimeout(tick, POLL_BACKOFF_MS[step]);
    };
    _pollingTimerId = setTimeout(tick, POLL_BACKOFF_MS[0]);
  },

  stopCatalogPolling() {
    _pollingGeneration++;
    if (_pollingTimerId) {
      clearTimeout(_pollingTimerId);
      _pollingTimerId = undefined;
    }
  },
});
