import type { StateCreator } from "zustand";
import { TensorApiError } from "@biopb/tensor-flight-client";
import type { DataSourceDescriptor, TreeLevel, TreeLevelResult } from "@biopb/tensor-flight-client";
import type { AppState } from "./types";

/** Most children one folder asks for. The server says when a folder holds more. */
export const LEVEL_LIMIT = 2000;

/**
 * A level as last read, with the validator the server gave it. A level written
 * from a reveal or a resolved row has none: it counts as changed at the next
 * poll, which stores the server's.
 */
export interface LoadedLevel {
  level: TreeLevel;
  etag: string | null;
}

/** Whether two reads of one level are the same: only equal etags say so. */
function sameLevel(a: LoadedLevel, b: LoadedLevel): boolean {
  return a.etag !== null && a.etag === b.etag;
}

const loaded = ({ level, etag }: TreeLevelResult): LoadedLevel => ({ level, etag });

/** Whether some level holds `sourceId`. */
function holds(levels: Record<string, LoadedLevel>, sourceId: string): boolean {
  return Object.values(levels).some((l) =>
    l.level.sources.some((x) => x.descriptor.source_id === sourceId),
  );
}

/** `ids` with `id` removed if it is there, else added. */
export function toggled(ids: ReadonlySet<string>, id: string): Set<string> {
  const next = new Set(ids);
  if (!next.delete(id)) next.add(id);
  return next;
}

/**
 * The levels, status and index after reading `entries` (folder key, level).
 *
 * The index follows the levels: what a level held before and no longer holds
 * leaves it, so a source that is gone from the catalog is not still "listed".
 * A level that was read is no longer loading or failed.
 */
function withLevels(
  s: AppState,
  entries: Array<[string, LoadedLevel]>,
): Pick<AppState, "treeLevels" | "levelStatus" | "sourceIndex"> {
  const treeLevels = { ...s.treeLevels };
  const levelStatus = { ...s.levelStatus };
  const sourceIndex = { ...s.sourceIndex };
  for (const [key, entry] of entries) {
    for (const old of s.treeLevels[key]?.level.sources ?? []) {
      delete sourceIndex[old.descriptor.source_id];
    }
    for (const src of entry.level.sources) sourceIndex[src.descriptor.source_id] = src.descriptor;
    treeLevels[key] = entry;
    delete levelStatus[key];
  }
  return { treeLevels, levelStatus, sourceIndex };
}

/**
 * The catalog as a tree read a folder at a time: only the levels the viewer has
 * opened are held, so what is fetched, kept and drawn follows what is on screen,
 * not the size of the catalog.
 */
export interface TreeSlice {
  /** Loaded levels by folder key; "" is the top. */
  treeLevels: Record<string, LoadedLevel>;
  /** Folders drawn open (the top is always shown). */
  openFolders: Set<string>;
  /** Levels being read, or that failed, by folder key. A level read fine is absent. */
  levelStatus: Record<string, "loading" | "error">;
  /**
   * Every descriptor the viewer has been shown -- loaded levels, a reveal, a
   * resolved row -- by source id. A source is looked up here, never searched for
   * in a list; the catalog as a whole is not held.
   */
  sourceIndex: Record<string, DataSourceDescriptor>;
  /** Read a folder's level, unless one is already being read. */
  loadLevel: (key: string) => Promise<void>;
  /** Open a folder (reading it if need be), or close it. */
  toggleFolder: (key: string) => void;
  /**
   * Open the folders down to `sourceId` and load their levels, in one request.
   * False when the server does not show that source.
   */
  revealSource: (sourceId: string) => Promise<boolean>;
  /** Re-read the top and the open folders. True when any changed or could not be read. */
  refreshTree: () => Promise<boolean>;
  /** Swap a source's row into the levels and the index. */
  applySourceRow: (row: DataSourceDescriptor) => void;
  /** Forget everything read: another server is being talked to. */
  resetTree: () => void;
}

export const createTreeSlice: StateCreator<AppState, [], [], TreeSlice> = (set, get) => ({
  treeLevels: {},
  openFolders: new Set(),
  levelStatus: {},
  sourceIndex: {},

  async loadLevel(key) {
    const { client } = get();
    if (!client || get().levelStatus[key] === "loading") return;
    set((s) => ({ levelStatus: { ...s.levelStatus, [key]: "loading" } }));
    try {
      const result = loaded(await client.http.treeLevel(key, LEVEL_LIMIT));
      // Another server was connected to while this was in flight.
      if (get().client !== client) return;
      set((s) => withLevels(s, [[key, result]]));
    } catch (err) {
      if (get().client !== client) return;
      console.warn("Tree level failed:", err);
      set((s) => ({ levelStatus: { ...s.levelStatus, [key]: "error" } }));
      throw err;
    }
  },

  toggleFolder(key) {
    const { openFolders, treeLevels, loadLevel } = get();
    const next = toggled(openFolders, key);
    set({ openFolders: next });
    // A level that failed is read again by opening it again.
    if (next.has(key) && !treeLevels[key]) void loadLevel(key).catch(() => {});
  },

  async revealSource(sourceId) {
    const { client } = get();
    if (!client) return false;
    let reveal;
    try {
      reveal = await client.http.treeReveal(sourceId);
    } catch (err) {
      if (err instanceof TensorApiError && err.status === 404) return false;
      throw err;
    }
    if (get().client !== client) return false;
    set((s) => {
      const merged = withLevels(
        s,
        reveal.levels.map((level): [string, LoadedLevel] => [level.key, { level, etag: null }]),
      );
      return {
        ...merged,
        openFolders: new Set([...s.openFolders, ...reveal.levels.map((l) => l.key).filter(Boolean)]),
        sourceIndex: { ...merged.sourceIndex, [reveal.descriptor.source_id]: reveal.descriptor },
      };
    });
    return true;
  },

  async refreshTree() {
    const { client } = get();
    if (!client) return false;
    const before = get().treeLevels;
    let failed = false;
    // The top first among the keys: it is what a first poll is for.
    const keys = ["", ...[...get().openFolders].filter((k) => k !== "")];
    const results = await Promise.all(
      keys.map((key) =>
        client.http.treeLevel(key, LEVEL_LIMIT).then(
          (r) => ({ key, result: loaded(r) }),
          // A level that fails keeps its last read: nothing says it changed.
          (err: unknown) => {
            console.warn("Tree level refresh failed:", err);
            failed = true;
            return null;
          },
        ),
      ),
    );
    // Pointed at another server while this was in flight.
    if (get().client !== client) return true;

    const changed = results.filter(
      (r): r is { key: string; result: LoadedLevel } =>
        r !== null && !(r.key in before && sameLevel(before[r.key]!, r.result)),
    );
    if (changed.length === 0) return failed;

    // A source that was on screen and is gone from its level is gone from the
    // catalog -- unless a level is cut short, which says nothing about what it
    // left out, or the source came from a shared link, which does not need to
    // be listed.
    const { activeSourceId, target, openTensor } = get();
    const watching = activeSourceId !== null && !target.linked && holds(get().treeLevels, activeSourceId);
    set((s) => withLevels(s, changed.map((c): [string, LoadedLevel] => [c.key, c.result])));
    const after = get().treeLevels;
    if (
      watching &&
      !holds(after, activeSourceId) &&
      !Object.values(after).some((l) => l.level.truncated)
    ) {
      openTensor(null);
    }
    return true;
  },

  applySourceRow(row) {
    set((s) => {
      const levels: Record<string, LoadedLevel> = {};
      for (const [key, entry] of Object.entries(s.treeLevels)) {
        levels[key] = entry.level.sources.some((x) => x.descriptor.source_id === row.source_id)
          ? {
              etag: null,
              level: {
                ...entry.level,
                sources: entry.level.sources.map((x) =>
                  x.descriptor.source_id === row.source_id ? { ...x, descriptor: row } : x,
                ),
              },
            }
          : entry;
      }
      return {
        treeLevels: levels,
        sourceIndex: { ...s.sourceIndex, [row.source_id]: row },
      };
    });
  },

  resetTree() {
    set({ treeLevels: {}, openFolders: new Set(), levelStatus: {}, sourceIndex: {} });
  },
});
