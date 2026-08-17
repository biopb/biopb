/**
 * Shaping rules for the metadata panel's JSON tree.
 *
 * A source's metadata is whatever the backend recorded, and that spans three
 * orders of magnitude: a 16-key OME header, or a MicroManager per-frame blob
 * with 2,001 top-level keys and ~485k nodes. The panel used to render the
 * second one in full — ~248k rows, ~744k React elements, mounted and unmounted
 * synchronously on every source change, which is why the canvas could not
 * repaint for seconds. Nobody reads 2,001 keys at once, so the tree is bounded
 * here rather than trusted to be small.
 */

/** Rows a node draws before the rest go behind a "… N more". */
export const MAX_ROWS = 10;

/**
 * Widest node still worth expanding on sight.
 *
 * Above a 16-key OME header and far below a per-frame blob: the point is to
 * keep the common case fully readable without opening anything, while a wide
 * node waits for a click.
 */
export const AUTO_EXPAND_MAX_ENTRIES = 32;

const emptinessCache = new WeakMap<object, boolean>();

/**
 * Whether a value carries nothing worth a row: null, or a container whose every
 * leaf is null all the way down.
 *
 * Memoised on object identity because the same subtree is asked about
 * repeatedly — once when its parent filters rows, again when it filters its
 * own, and once more per level below that. Rendering two levels of the
 * MicroManager blob took 253k walks; the cache makes it one walk per node.
 * Keys are objects from a single `JSON.parse`, so entries die with the metadata
 * they describe.
 */
export function isEmptyForDisplay(value: unknown): boolean {
  if (value === null || value === undefined) return true;
  if (typeof value !== "object") return false;

  const cached = emptinessCache.get(value);
  if (cached !== undefined) return cached;

  const children: unknown[] = Array.isArray(value)
    ? value
    : Object.values(value as Record<string, unknown>);
  const empty = children.length === 0 || children.every(isEmptyForDisplay);
  emptinessCache.set(value, empty);
  return empty;
}

/** The entries of an object that earn a row. */
export function visibleEntries(value: Record<string, unknown>): [string, unknown][] {
  return Object.entries(value).filter(([, v]) => !isEmptyForDisplay(v));
}

/**
 * Whether any entry earns a row, without working out which.
 *
 * A collapsed node only has to choose between `{...}` and `{}`, and `some`
 * stops at the first non-empty entry — on a wide node the full filter walks the
 * entire subtree for an answer it then discards.
 */
export function hasVisibleEntries(value: Record<string, unknown>): boolean {
  return Object.values(value).some((v) => !isEmptyForDisplay(v));
}

/** How many rows a node would draw with nothing capped. */
export function childCount(value: unknown): number {
  if (Array.isArray(value)) return value.length;
  if (value !== null && typeof value === "object") return Object.keys(value).length;
  return 0;
}

/**
 * Whether a node starts expanded.
 *
 * Depth alone is the wrong test. Two levels is right for a 16-key header and
 * pathological for a 2,001-key one, where it costs a quarter of a million rows
 * before the user has clicked anything — so width decides as well.
 */
export function autoExpanded(value: unknown, depth: number): boolean {
  return depth < 2 && childCount(value) <= AUTO_EXPAND_MAX_ENTRIES;
}
