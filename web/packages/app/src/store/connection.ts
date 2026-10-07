import type { StateCreator } from "zustand";
import { TensorFlightClient } from "@biopb/tensor-flight-client";
import type { DataSourceDescriptor, QuerySourcesResult } from "@biopb/tensor-flight-client";
import { withBase } from "../base";
import { catalogIsFilling } from "../utils/catalogHealth";
import type { AppState } from "./types";

export type ConnectionState = "idle" | "connecting" | "connected" | "error";

// Internal timer storage (non-reactive, module-level)
let _pollingTimerId: ReturnType<typeof setInterval> | undefined;

/**
 * Everything about a catalog listing the tree renders off, as one string.
 *
 * The poll used to diff the `source_url` set alone, which is blind to every
 * in-place change: a cloud source that resolves was already listed under that
 * url, it just gained its tensors and flipped its flags, so the tree kept
 * showing it as unresolved until a manual reload (biopb/biopb#1030). Eviction
 * is invisible the same way. `JSON.stringify` covers every
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

/** The client, the connection's health, and the catalog it lists. */
export interface ConnectionSlice {
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
  // Catalog polling
  pollingInterval: number;
  initClient: (apiBase: string, token: string | null, devMode: boolean) => void;
  loadSources: () => Promise<void>;
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

  sources: [],
  sourcesLoading: false,
  scanning: false,

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

  clearSession() {
    sessionStorage.removeItem("biopb_token");
    window.location.href = withBase("/unlock");
  },

  startCatalogPolling() {
    const pollingTimerId = setInterval(async () => {
      const { client } = get();
      if (!client || get().connectionState !== "connected") return;

      try {
        const newSources = await client.listSources();
        const sorted = newSources.sort((a, b) => a.source_url.localeCompare(b.source_url));

        // Refresh the scan-in-progress flag so the "Indexing…" hint clears once
        // the background catalog scan finishes (best-effort; a readyz blip just
        // leaves the previous value).
        try {
          const readyz = await client.http.readyz();
          set({ scanning: catalogIsFilling(readyz.backend_health) });
        } catch {
          // ignore transient readyz errors
        }

        // Read after the awaits: a tensor opened while the listing was in
        // flight is the one to protect, not the one that was open when it began.
        const { sources, activeSourceId, target, openTensor } = get();
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
});
