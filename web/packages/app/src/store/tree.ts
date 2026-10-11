import type { StateCreator } from "zustand";
import { TensorApiError } from "@biopb/tensor-flight-client";
import type { DataSourceDescriptor, TreeLevel, TreeLevelResult } from "@biopb/tensor-flight-client";
import type { AppState } from "./types";

/** Most children one folder asks for. The server says when a folder holds more. */
export const LEVEL_LIMIT = 2000;

/** A level as last read, with the validator the server gave it. */
export interface LoadedLevel {
  level: TreeLevel;
  etag: string | null;
}

/** Whether two reads of one level are the same. Etags decide when both have one. */
function sameLevel(a: LoadedLevel, b: LoadedLevel): boolean {
  if (a.etag !== null && b.etag !== null) return a.etag === b.etag;
  return JSON.stringify(a.level) === JSON.stringify(b.level);
}

const loaded = ({ level, etag }: TreeLevelResult): LoadedLevel => ({ level, etag });

/** The source ids a set of levels holds. */
function idsIn(levels: Record<string, LoadedLevel>): Set<string> {
  const ids = new Set<string>();
  for (const { level } of Object.values(levels)) {
    for (const s of level.sources) ids.add(s.descriptor.source_id);
  }
  return ids;
}

/** `index` plus the descriptors the levels carry. */
function indexed(
  index: Record<string, DataSourceDescriptor>,
  levels: Iterable<TreeLevel>,
): Record<string, DataSourceDescriptor> {
  const next = { ...index };
  for (const level of levels) {
    for (const s of level.sources) next[s.descriptor.source_id] = s.descriptor;
  }
  return next;
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
      set((s) => {
        const status = { ...s.levelStatus };
        delete status[key];
        return {
          treeLevels: { ...s.treeLevels, [key]: result },
          levelStatus: status,
          sourceIndex: indexed(s.sourceIndex, [result.level]),
        };
      });
    } catch (err) {
      if (get().client !== client) return;
      console.warn("Tree level failed:", err);
      set((s) => ({ levelStatus: { ...s.levelStatus, [key]: "error" } }));
      throw err;
    }
  },

  toggleFolder(key) {
    const { openFolders, treeLevels, levelStatus, loadLevel } = get();
    const next = new Set(openFolders);
    if (next.delete(key)) {
      set({ openFolders: next });
      return;
    }
    next.add(key);
    set({ openFolders: next });
    // An error level is read again by opening it again.
    if (!treeLevels[key] && levelStatus[key] !== "loading") void loadLevel(key).catch(() => {});
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
      const open = new Set(s.openFolders);
      const levels = { ...s.treeLevels };
      for (const level of reveal.levels) {
        levels[level.key] = { level, etag: null };
        if (level.key !== "") open.add(level.key);
      }
      return {
        treeLevels: levels,
        openFolders: open,
        sourceIndex: {
          ...indexed(s.sourceIndex, reveal.levels),
          [reveal.descriptor.source_id]: reveal.descriptor,
        },
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
        r !== null && (!(r.key in before) || !sameLevel(before[r.key]!, r.result)),
    );
    if (changed.length === 0) return failed;

    const { activeSourceId, target, openTensor } = get();
    const idsBefore = idsIn(get().treeLevels);
    set((s) => {
      const levels = { ...s.treeLevels };
      for (const { key, result } of changed) levels[key] = result;
      return {
        treeLevels: levels,
        sourceIndex: indexed(
          s.sourceIndex,
          changed.map((c) => c.result.level),
        ),
      };
    });

    // A source that was on screen and is gone from its level is gone from the
    // catalog -- unless a level is cut short, which says nothing about what it
    // left out, or the source came from a shared link, which does not need to
    // be listed.
    const after = get().treeLevels;
    if (
      activeSourceId &&
      !target.linked &&
      idsBefore.has(activeSourceId) &&
      !idsIn(after).has(activeSourceId) &&
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
