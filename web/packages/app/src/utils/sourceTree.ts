// The parts of the source tree that are a rule rather than a rendering.
//
// Split out of SourceTree.tsx for the reason paneWidth is split out of HomePage:
// a component file that also exports helpers loses fast refresh, and a rule left
// inline in a component is a rule nothing can check. What a url's path parts are,
// and what belongs under "Recent", both have answers worth pinning.

import { splitLabelArrayId } from "@biopb/tensor-flight-client";
import type { DataSourceDescriptor, TensorDescriptor } from "@biopb/tensor-flight-client";

// Origin scheme the tensor server stamps on a drag-dropped source's source_url
// (server-side DND_URL_PREFIX). Display-only marker of drop provenance; the tree
// strips it so a dropped source renders under a clean root, identical to a
// scheme-less re-root. Keep in sync with the server constant and the napari
// plugin's _get_path_parts.
const DND_URL_PREFIX = "dnd://";

// The "Recent" node's own folder id. Space-prefixed so it cannot collide with a
// path folder, which is always a "/"-joined url fragment.
export const RECENT_FOLDER_ID = " recent";

// Row ids under "Recent" are namespaced away from the catalog row for the same
// source, which is a different node holding the same descriptor. Sharing an id
// would make `folderPathTo` reveal whichever copy it reached first, and the
// reveal is meant to find the source in its real place in the hierarchy.
const RECENT_ROW_PREFIX = `${RECENT_FOLDER_ID}:`;

export interface TreeNode {
  id: string;           // unique id (path for folders, source_id for sources)
  name: string;         // display name
  type: "folder" | "source";
  children: TreeNode[];
  source?: DataSourceDescriptor;  // only for source nodes
  depth: number;
}

export function getPathParts(url: string): string[] {
  if (!url) return [];
  if (url.startsWith(DND_URL_PREFIX)) {
    // Drag-dropped source: strip the origin scheme and split the re-rooted
    // remainder as a plain path. String-strip (not URL parse) avoids host/port
    // misparsing of a basename like "exp:2.zarr".
    return url.slice(DND_URL_PREFIX.length).split(/[\\/]+/).filter(Boolean);
  }
  try {
    const parsed = new URL(url);
    const path = parsed.pathname.split("/").filter(Boolean);
    // Authority URLs (remote tensor-server mirrors "grpc://host:port/remote/path",
    // "s3://bucket/key", …) surface the endpoint "<protocol>//<host>" as the root,
    // so mirrored sources nest by their remote filepath under an endpoint node
    // instead of collapsing into a flat "grpc:" node (biopb/biopb#297). Local
    // file:// has an empty host, so it is unchanged (just its path). Mirror of the
    // napari plugin's _get_path_parts — keep the two behaviorally in lockstep.
    if (parsed.host) {
      return [`${parsed.protocol}//${parsed.host}`, ...path];
    }
    return path;
  } catch {
    return url.split("/").filter(Boolean);
  }
}

/** Whether a source matches a (already-lowercased, trimmed) search query,
 * against the same fields the tree search box and the catalog SQL fallback
 * both check. */
export function matchesQuery(src: DataSourceDescriptor, q: string): boolean {
  return `${src.source_id} ${src.source_url} ${src.source_type}`
    .toLowerCase()
    .includes(q);
}

/** What to call a source in the tree: its leaf path segment where a url is
 * known, the id otherwise -- which is the only name an upload has. Shared by
 * `buildTree`'s catalog nodes and `recentNode`'s, so the two trees never
 * disagree on what a source is called. */
export function sourceLabel(src: DataSourceDescriptor): string {
  const parts = getPathParts(src.source_url);
  return parts[parts.length - 1] ?? src.source_id;
}

/** The last path segment of a bare id or url, falling back to the whole
 * string -- for a short display label where only that string is known, not a
 * full {@link DataSourceDescriptor} (e.g. a resolve/warm job's `source_id`). */
export function shortId(idOrUrl: string): string {
  const parts = getPathParts(idOrUrl);
  return parts[parts.length - 1] ?? idOrUrl;
}

/**
 * Glyph marking a source the server has not hydrated yet. Matches the napari
 * `tensor_browser`'s own indicator so the two browsers read the same.
 */
export const UNRESOLVED_GLYPH = "\u2601";

/** Hover copy for {@link UNRESOLVED_GLYPH}. */
export const UNRESOLVED_TOOLTIP =
  "Not resolved \u2014 this source's content is remote and has not been " +
  "downloaded yet, so its tensors are unknown. It has to be resolved before " +
  "it can be opened. (Cloud / remote source support is experimental.)";

/**
 * Has the server got a real, hydrated adapter behind this source?
 *
 * The catalog field, not `tensors.length === 0` -- napari infers it that way
 * (`_is_unresolved`) and the inference is wrong in both directions: a resolved
 * source can legitimately carry no tensor, and an unresolved one is not
 * *defined* by the empty list, it just happens to have one. `is_resolved` is
 * the server's own answer, and it is monotonic, so a listing can only ever lag
 * in the harmless direction (biopb/biopb#1030).
 *
 * Defaults to resolved for a descriptor from a server predating the field --
 * the right reading for every source that existed before it.
 */
export function isUnresolved(src: DataSourceDescriptor): boolean {
  return src.is_resolved === false;
}

/**
 * The "Recent" folder, or null when nothing is in it.
 *
 * Built beside `buildTree` and prepended rather than merged into it because a
 * recent source often has no path to hang under -- an upload has no path at
 * all.
 *
 * **Recency decides membership, not order.** `recents` arrives newest first and
 * that is what the cap evicts by, but displaying it that way moved a row to the
 * top the moment it was clicked, re-sorting the list under the pointer and
 * leaving the next click somewhere else. Sorted by name the rows stay put, and
 * a list short enough to be a shortcut does not need to advertise which entry
 * was most recent.
 *
 * The tiebreak is load-bearing rather than tidiness: two sources can share a
 * leaf name (`plate1.zarr` under different folders), `Array.sort` is stable, and
 * without it those two would keep their incoming *recency* order -- reproducing
 * the jump this exists to remove, in exactly the case where two rows look alike
 * and a swap is hardest to notice.
 */
export function recentNode(recents: DataSourceDescriptor[]): TreeNode | null {
  if (recents.length === 0) return null;
  recents = [...recents].sort(
    (a, b) =>
      sourceLabel(a).localeCompare(sourceLabel(b)) ||
      a.source_id.localeCompare(b.source_id),
  );
  return {
    id: RECENT_FOLDER_ID,
    name: "Recent",
    type: "folder",
    depth: 1,
    children: recents.map((src) => ({
      id: RECENT_ROW_PREFIX + src.source_id,
      name: sourceLabel(src),
      type: "source" as const,
      children: [],
      source: src,
      depth: 2,
    })),
  };
}

/** An image tensor and the label sets addressed under it. */
export interface TensorGroup {
  image: TensorDescriptor;
  /** Its sets, by name. Empty for an ordinary tensor. */
  labelSets: TensorDescriptor[];
}

/**
 * A source's tensors, with each label set filed under the image it annotates.
 *
 * The catalog lists a set as an ordinary tensor of the source -- there is no
 * `role` column, by decision (biopb/biopb#1059) -- so the path is what says a
 * tensor is one, and this is where the tree reads it.
 *
 * A set whose image the source does not list comes back as a group of its own
 * rather than being dropped. That should not happen (the server registers a set
 * on its parent), but a tensor the catalog lists and the tree silently hides is
 * the worse failure of the two.
 */
export function groupTensors(tensors: TensorDescriptor[]): TensorGroup[] {
  const groups = new Map<string, TensorGroup>();
  const order: string[] = [];
  const sets: Array<{ tensor: TensorDescriptor; imageArrayId: string; name: string }> = [];

  for (const tensor of tensors) {
    const address = splitLabelArrayId(tensor.array_id);
    if (address) {
      sets.push({ tensor, imageArrayId: address.imageArrayId, name: address.name });
      continue;
    }
    if (groups.has(tensor.array_id)) continue;
    groups.set(tensor.array_id, { image: tensor, labelSets: [] });
    order.push(tensor.array_id);
  }

  for (const set of sets) {
    const group = groups.get(set.imageArrayId);
    if (group) {
      group.labelSets.push(set.tensor);
      continue;
    }
    groups.set(set.tensor.array_id, { image: set.tensor, labelSets: [] });
    order.push(set.tensor.array_id);
  }

  // Sets sorted by name, images left in the order the server listed them: it
  // puts image tensors first on purpose, and the catalog's scalar `dtype` and
  // `shape_summary` describe `tensors[0]`.
  for (const group of groups.values()) {
    group.labelSets.sort((a, b) => a.array_id.localeCompare(b.array_id));
  }
  return order.map((id) => groups.get(id) as TensorGroup);
}
