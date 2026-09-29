import type { StateCreator } from "zustand";
import type { DataSourceDescriptor } from "@biopb/tensor-flight-client";
import { TensorApiError } from "@biopb/tensor-flight-client";
import {
  descriptorFromTileInfo,
  forget as forgetRecents,
  readRecents,
  remember as rememberRecent,
  writeRecents,
} from "../utils/recentSources";
import type { AppState } from "./types";

/** The sources this browser has opened, most recent first. */
export interface RecentsSlice {
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
  /** Record a source as just opened, and persist the list. */
  noteRecent: (sourceId: string) => void;
  /** Adopt a list written by another tab. Does not write it back. */
  syncRecents: (ids: readonly string[]) => void;
  /** Resolve `recentIds` into `recentSources`, dropping ids the server 404s. */
  hydrateRecents: () => Promise<void>;
}

export const createRecentsSlice: StateCreator<AppState, [], [], RecentsSlice> = (set, get) => ({
  // Seeded at module init, like the persisted channel colors below, so the node
  // is populated on the first render rather than after an effect. `readRecents`
  // answers [] where there is no storage, which covers node.
  recentIds: readRecents(),
  recentSources: [],

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
});
