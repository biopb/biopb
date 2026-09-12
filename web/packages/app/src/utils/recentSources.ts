// The sources this browser has opened, most recent first.
//
// localStorage, not a cookie and not the server. A cookie rides every request to
// the origin -- including each /api/tile GET while panning -- to carry state no
// handler reads. And "what have I been looking at" is this browser's state, not
// something the server should be made to hold per client. It is also the only
// one of the three that fits both scopes this has to satisfy: shared by every
// tab (sessionStorage is per-tab, so two tabs of one link would disagree about
// the list) and free to outlive a browser restart.
//
// Ids only, never descriptors. The point of the list is to reach `cache://`
// uploads, which the catalog deliberately never lists (biopb/biopb#265) and
// which can be evicted or lost to a server restart -- so a descriptor
// snapshotted here would render a row that cannot be opened. Re-reading each id
// from the server is what keeps the list honest, and the re-read doubles as the
// liveness check: see `hydrateRecents` in store.ts.
//
// The list algebra below is pure and tested. The three functions that touch
// `localStorage` are thin and untested, for the reason sessionFetch's are: this
// suite runs in node, with no DOM.

import type {
  DataSourceDescriptor,
  TileInfo,
} from "@biopb/tensor-flight-client";

const STORAGE_KEY = "biopb.recentSources";

/** Long enough to cover a session's hopping, short enough that the node stays a
 * shortcut rather than a second catalog -- on a laptop-height sidebar a longer
 * list is the whole pane, and the catalog it sits above has to be scrolled to
 * at all. Sorting by name removed the recency cue that would have made a longer
 * list skimmable. */
export const MAX_RECENT = 15;

/**
 * `id` at the front, every other entry in order, nothing twice, capped.
 *
 * Returns the input array *by identity* when `id` is already the head, so a
 * caller can skip the write. That matters: this is called from
 * `applyViewerState`, which re-runs on every slider move and camera nudge, and
 * a fresh array each time would be a localStorage write per frame.
 */
export function remember(ids: readonly string[], id: string): readonly string[] {
  if (!id || ids[0] === id) return ids;
  return [id, ...ids.filter((prev) => prev !== id)].slice(0, MAX_RECENT);
}

/**
 * Drop ids the server has said are gone.
 *
 * Only for a definite answer (a 404). An id that merely failed to fetch stays:
 * a server that is down, restarting, or briefly unreachable would otherwise
 * empty the list, and that is exactly when it is worth keeping.
 */
export function forget(
  ids: readonly string[],
  gone: readonly string[],
): readonly string[] {
  if (!gone.length) return ids;
  const dead = new Set(gone);
  const kept = ids.filter((id) => !dead.has(id));
  return kept.length === ids.length ? ids : kept;
}

/**
 * The stored list, or an empty one for anything that is not a list of strings.
 *
 * Separate from `readRecents` so the corrupt-value paths are testable: this is
 * shared storage under a stable key, so a stale shape from an older build -- or
 * a hand-edit -- has to degrade to "no recents" rather than throw on boot.
 */
export function parseRecents(raw: string | null): string[] {
  if (!raw) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return [];
  }
  if (!Array.isArray(parsed)) return [];
  const ids = parsed.filter(
    (entry): entry is string => typeof entry === "string" && entry.length > 0,
  );
  // Deduped on the way in as well as the way out: the cap is only a bound if
  // the stored list is one entry per source.
  return [...new Set(ids)].slice(0, MAX_RECENT);
}

/**
 * A one-tensor `DataSourceDescriptor` for a recent id the catalog does not list.
 *
 * `source_url` is deliberately empty rather than guessed. `tile_info` describes
 * a *tensor* and does not carry the source's url, and the two kinds that reach
 * here want opposite guesses -- a `cache://` upload's url is derivable from its
 * id, a source past the catalog cap has a real filesystem path that is not. An
 * empty url says "no path known", which the tree already renders by falling back
 * to the id; inventing `cache://` for both would mislabel the second kind.
 *
 * The id given is the one to address it by, not `info.array_id`: that one is
 * content-versioned (`id@token`), and pinning a row to a token means it stops
 * resolving the moment the upload is rewritten.
 */
export function descriptorFromTileInfo(
  sourceId: string,
  info: TileInfo,
): DataSourceDescriptor {
  return {
    source_id: sourceId,
    source_url: "",
    source_type: "",
    metadata_json: null,
    tensors: [
      {
        array_id: sourceId,
        dim_labels: info.dim_labels,
        shape: info.shape,
        // Empty for the reason a listing's entries are: the transfer grid is
        // answered per resolved tensor, and this descriptor is a listing entry.
        chunk_shape: [],
        dtype: info.dtype,
      },
    ],
  };
}

/** The stored list. Empty when storage is unavailable (private mode, disabled
 * site data) -- the feature degrades to "no recents", never to a boot failure. */
export function readRecents(): string[] {
  try {
    return parseRecents(localStorage.getItem(STORAGE_KEY));
  } catch {
    return [];
  }
}

/** Persist the list. Best-effort for the reason `readRecents` is, plus one more:
 * a full quota throws here, and losing a shortcut is not worth an error. */
export function writeRecents(ids: readonly string[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
  } catch {
    // ignore
  }
}

/**
 * Call `onChange` when another tab changes the list.
 *
 * This is the half of the cross-tab requirement that localStorage does not give
 * for free: the storage is shared, but a write in one tab does not re-render
 * another. `storage` fires only in the *other* tabs, which is exactly the gap --
 * the writing tab updates its own state directly.
 *
 * A `null` key is a whole-store `clear()`, which has to count as a change.
 */
export function subscribeRecents(
  onChange: (ids: string[]) => void,
): () => void {
  const handler = (event: StorageEvent) => {
    if (event.key !== null && event.key !== STORAGE_KEY) return;
    onChange(readRecents());
  };
  window.addEventListener("storage", handler);
  return () => window.removeEventListener("storage", handler);
}
