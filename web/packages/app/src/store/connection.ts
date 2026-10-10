import type { StateCreator } from "zustand";
import { TensorFlightClient } from "@biopb/tensor-flight-client";
import type { DataSourceDescriptor, QuerySourcesResult } from "@biopb/tensor-flight-client";
import { withBase } from "../base";
import { catalogIsFilling } from "../utils/catalogHealth";
import type { AppState } from "./types";

export type ConnectionState = "idle" | "connecting" | "connected" | "error";

/** Most sources one listing carries. A tree of more is slow to draw and a poll
 *  of more slow to parse; the server says when the catalog is longer. */
export const CATALOG_LIMIT = 20_000;

/** Delays between catalog polls: a quiet catalog is asked about less and less. */
export const POLL_BACKOFF_MS = [30_000, 60_000, 120_000, 300_000];

/** The backoff step after a poll: back to the start while the catalog is
 *  changing or still being indexed, one further out while it is not. */
export function nextPollStep(step: number, active: boolean): number {
  return active ? 0 : Math.min(step + 1, POLL_BACKOFF_MS.length - 1);
}

/** One listing, ordered by `source_url` for display and comparison. */
async function fetchCatalog(client: TensorFlightClient) {
  const { sources, truncated } = await client.listSourcesPage(CATALOG_LIMIT);
  return { sorted: sources.sort((a, b) => a.source_url.localeCompare(b.source_url)), truncated };
}

// Internal timer storage (non-reactive, module-level). The generation tells a
// poll still in flight that the loop it belonged to was stopped.
let _pollingTimerId: ReturnType<typeof setTimeout> | undefined;
let _pollingGeneration = 0;

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
  // The server held back sources past `CATALOG_LIMIT`: `sources` is a prefix of
  // the catalog, not all of it.
  catalogTruncated: boolean;
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
  catalogTruncated: false,

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
      const { sorted, truncated } = await fetchCatalog(client);
      set({
        sources: sorted,
        catalogTruncated: truncated,
        sourcesLoading: false,
        connectionState: "connected",
      });
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

    // One poll. True when the catalog moved or is still being indexed, which is
    // when the next one should come soon.
    const poll = async (): Promise<boolean> => {
      const { client } = get();
      if (!client || get().connectionState !== "connected") return false;

      // The listing and the scan-in-progress flag are independent. The flag
      // clears the "Indexing…" hint once the background catalog scan finishes
      // (best-effort; a readyz blip just leaves the previous value).
      const [{ sorted, truncated }, readyz] = await Promise.all([
        fetchCatalog(client),
        client.http.readyz().catch(() => null),
      ]);
      if (readyz) set({ scanning: catalogIsFilling(readyz.backend_health) });

      // Read after the awaits: a tensor opened while the listing was in
      // flight is the one to protect, not the one that was open when it began.
      const { sources, catalogTruncated, scanning, activeSourceId, target, openTensor } = get();
      const changed =
        truncated !== catalogTruncated || catalogFingerprint(sources) !== catalogFingerprint(sorted);
      if (changed) {
        set({ sources: sorted, catalogTruncated: truncated });

        // A catalog response is a listing, not proof that an unlisted source
        // is gone: it may be cut at the limit, still scanning, or temporarily
        // failed. In particular, retain a source selected from a shared URL,
        // which deliberately does not need to be present in the listing.
        if (
          activeSourceId &&
          !truncated &&
          !target.linked &&
          !sorted.find((s) => s.source_id === activeSourceId)
        ) {
          openTensor(null);
        }
      }
      return changed || scanning;
    };

    const tick = async () => {
      let active = false;
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
