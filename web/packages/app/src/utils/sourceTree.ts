// The parts of the source tree that are a rule rather than a rendering.
//
// Split out of SourceTree.tsx for the reason paneWidth is split out of HomePage:
// a component file that also exports helpers loses fast refresh, and a rule left
// inline in a component is a rule nothing can check. What a url's path parts are,
// and what belongs under "Recent", both have answers worth pinning.

import type { DataSourceDescriptor } from "@biopb/tensor-flight-client";

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
const RECENT_ROW_PREFIX = " recent:";

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

/** What to call a recent row: its leaf name where a url is known, the id
 * otherwise -- which is the only name an upload has. */
export function recentLabel(src: DataSourceDescriptor): string {
  const parts = getPathParts(src.source_url);
  return parts[parts.length - 1] ?? src.source_id;
}

/**
 * The "Recent" folder, or null when nothing is in it.
 *
 * Built beside `buildTree` and prepended rather than merged into it, for two
 * reasons: `buildTree` sorts alphabetically and recency order is the whole point
 * of this node, and a recent source often has no path to hang under -- an upload
 * has no path at all.
 */
export function recentNode(recents: DataSourceDescriptor[]): TreeNode | null {
  if (recents.length === 0) return null;
  return {
    id: RECENT_FOLDER_ID,
    name: "Recent",
    type: "folder",
    depth: 1,
    children: recents.map((src) => ({
      id: RECENT_ROW_PREFIX + src.source_id,
      name: recentLabel(src),
      type: "source" as const,
      children: [],
      source: src,
      depth: 2,
    })),
  };
}
