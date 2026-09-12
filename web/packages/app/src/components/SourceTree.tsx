"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { selectTileInfo, useAppStore } from "../store";
import type { DataSourceDescriptor } from "@biopb/tensor-flight-client";
import { splitArrayVersion } from "@biopb/tensor-flight-client";
import { readRecents, subscribeRecents } from "../utils/recentSources";
import {
  RECENT_FOLDER_ID,
  type TreeNode,
  getPathParts,
  matchesQuery,
  recentNode,
  sourceLabel,
} from "../utils/sourceTree";

// Threshold for switching to server-side SQL query
const SERVER_QUERY_THRESHOLD = 1000;

function tensorShortName(arrayId: string): string {
  const parts = arrayId.split("/").filter(Boolean);
  return parts[parts.length - 1] || arrayId;
}

function formatShape(shape: number[]): string {
  return shape.join("×");
}

function buildTree(sources: DataSourceDescriptor[]): TreeNode {
  const root: TreeNode = { id: "", name: "", type: "folder", children: [], depth: 0 };

  // Build initial tree from sources
  for (const src of sources) {
    const parts = getPathParts(src.source_url);
    if (parts.length === 0) {
      // No path parts, add directly to root
      root.children.push({
        id: src.source_id,
        name: sourceLabel(src),
        type: "source",
        children: [],
        source: src,
        depth: 1,
      });
      continue;
    }

    // Navigate/create folder path
    let current = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const part = parts[i]!;
      let child = current.children.find((c) => c.type === "folder" && c.name === part);
      if (!child) {
        child = {
          id: current.id + "/" + part,
          name: part,
          type: "folder",
          children: [],
          depth: current.depth + 1,
        };
        current.children.push(child);
      }
      current = child;
    }

    // Add source as leaf
    current.children.push({
      id: src.source_id,
      name: sourceLabel(src),
      type: "source",
      children: [],
      source: src,
      depth: current.depth + 1,
    });
  }

  // Sort children: folders first, then sources, both alphabetically
  function sortChildren(node: TreeNode) {
    node.children.sort((a, b) => {
      if (a.type !== b.type) return a.type === "folder" ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    for (const child of node.children) {
      sortChildren(child);
    }
  }
  sortChildren(root);

  // Flatten paths: merge folders that have only one folder child
  function flattenPaths(node: TreeNode): void {
    for (const child of node.children) {
      if (child.type === "folder") {
        // Recursively flatten first
        flattenPaths(child);

        // Check if this folder should be flattened
        // Condition: exactly one child, and that child is a folder
        while (
          child.children.length === 1 &&
          child.children[0]?.type === "folder"
        ) {
          const grandchild = child.children[0];
          // Merge: append grandchild name to child name
          child.name = child.name + "/" + grandchild.name;
          child.id = grandchild.id;
          child.children = grandchild.children;
        }

        // Continue flattening in case new structure allows more flattening
        flattenPaths(child);
      }
    }
  }
  flattenPaths(root);

  // Flattening rewires parent/child links but leaves stale depths behind (a
  // merged node's deeper descendants keep their pre-merge level), which shows up
  // as a subtree indented one extra step. Recompute every depth from the final
  // tree level in one pass so indentation is exactly the nesting depth.
  function recomputeDepths(node: TreeNode, depth: number): void {
    node.depth = depth;
    for (const child of node.children) {
      recomputeDepths(child, depth + 1);
    }
  }
  recomputeDepths(root, 0);

  return root;
}

// Filter tree to show only matching sources, auto-expand folders with matches
function filterTree(
  node: TreeNode,
  matchingSourceIds: Set<string>,
  expandedFolders: Set<string>
): TreeNode | null {
  if (node.type === "source") {
    if (matchingSourceIds.has(node.id)) {
      return node;
    }
    return null;
  }

  // Folder: filter children
  const filteredChildren: TreeNode[] = [];
  for (const child of node.children) {
    const filtered = filterTree(child, matchingSourceIds, expandedFolders);
    if (filtered) {
      filteredChildren.push(filtered);
      // Auto-expand folders containing matches
      if (filtered.type === "source" || filtered.children.length > 0) {
        expandedFolders.add(node.id);
      }
    }
  }

  if (filteredChildren.length === 0) {
    return null;
  }

  return { ...node, children: filteredChildren };
}

/**
 * The folders between the root and `sourceId`, outermost first, or null when no
 * source node answers to it.
 *
 * Walked rather than derived from `source_url`: `flattenPaths` merges a
 * single-child chain into one node and takes the *grandchild's* id, so a folder
 * id is not a prefix of the path it displays. The built tree is the only thing
 * that knows which ids survived.
 */
function folderPathTo(node: TreeNode, sourceId: string): string[] | null {
  for (const child of node.children) {
    if (child.type === "source") {
      if (child.id === sourceId) return [];
      continue;
    }
    const below = folderPathTo(child, sourceId);
    if (below) return [child.id, ...below];
  }
  return null;
}

function Chevron({ expanded }: { expanded: boolean }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: 16,
        fontSize: 10,
        transition: "transform 0.15s",
        transform: expanded ? "rotate(90deg)" : "rotate(0deg)",
        opacity: 0.6,
      }}
    >
      ▶
    </span>
  );
}

// Empty gutter the width of a Chevron, so leaf (source/tensor) rows keep their
// labels in the same column as folder labels — otherwise a folder's children
// align left of the folder's own label and the indent steps unevenly.
function ChevronSlot() {
  return (
    <span
      aria-hidden="true"
      style={{ display: "inline-block", width: 16, flexShrink: 0 }}
    />
  );
}

export interface TreeRowProps {
  node: TreeNode;
  activeSourceId: string | null;
  activeTensorId: string | null;
  expandedFolders: Set<string>;
  toggleFolder: (id: string) => void;
  selectSource: (sourceId: string, tensorId?: string) => void;
}

export function TreeRow({
  node,
  activeSourceId,
  activeTensorId,
  expandedFolders,
  toggleFolder,
  selectSource,
}: TreeRowProps) {
  const indent = node.depth * 12 + 12;

  if (node.type === "folder") {
    const expanded = expandedFolders.has(node.id);
    return (
      <>
        <button
          className="tree-item tree-folder"
          style={{
            width: "100%",
            textAlign: "left",
            display: "flex",
            alignItems: "center",
            paddingLeft: indent,
          }}
          onClick={() => toggleFolder(node.id)}
        >
          <Chevron expanded={expanded} />
          <span style={{ marginLeft: 4 }}>{node.name}</span>
        </button>
        {expanded &&
          node.children.map((child) => (
            <TreeRow
              key={child.id}
              node={child}
              activeSourceId={activeSourceId}
              activeTensorId={activeTensorId}
              expandedFolders={expandedFolders}
              toggleFolder={toggleFolder}
              selectSource={selectSource}
            />
          ))}
      </>
    );
  }

  // Source node
  const src = node.source!;
  const isActive = src.source_id === activeSourceId;
  const hasMultipleTensors = src.tensors.length > 1;
  const firstTensor = src.tensors[0];

  return (
    <>
      <button
        className={`tree-item ${isActive ? "active" : ""}`}
        style={{
          width: "100%",
          textAlign: "left",
          display: "flex",
          alignItems: "center",
          paddingLeft: indent,
        }}
        // Only the catalog row carries this: it is what the reveal effect
        // scrolls to, and the "Recent" copy of the same source -- same
        // descriptor, different node -- comes first in document order and would
        // otherwise shadow it.
        data-source-id={node.id === src.source_id ? src.source_id : undefined}
        onClick={() => {
          if (src.tensors.length === 1) {
            selectSource(src.source_id, src.tensors[0]?.array_id);
          } else {
            selectSource(src.source_id);
          }
        }}
        title={src.source_url}
      >
        <ChevronSlot />
        <span style={{ flex: 1, marginLeft: 4 }}>{node.name}</span>
        {hasMultipleTensors ? (
          <span className="tensor-pill" style={{ marginLeft: 8 }}>
            {src.tensors.length}
          </span>
        ) : firstTensor ? (
          <span
            className="dim-badge"
            style={{ marginLeft: 8 }}
            title={formatShape(firstTensor.shape)}
          >
            {formatShape(firstTensor.shape)}
          </span>
        ) : null}
      </button>

      {/* Nested tensors when source is active and has multiple tensors */}
      {isActive && hasMultipleTensors &&
        src.tensors.map((t) => {
          const tActive = t.array_id === activeTensorId;
          const tName = tensorShortName(t.array_id);
          return (
            <button
              key={`tensor:${src.source_id}:${t.array_id}`}
              className={`tree-item tensor-item ${tActive ? "active" : ""}`}
              style={{
                width: "100%",
                textAlign: "left",
                paddingLeft: indent + 12,
                display: "flex",
                alignItems: "center",
                fontSize: 12,
              }}
              onClick={() => selectSource(src.source_id, t.array_id)}
              title={`${t.array_id}\nShape: ${formatShape(t.shape)}\nDtype: ${t.dtype}`}
            >
              <ChevronSlot />
              <span style={{ flex: 1, marginLeft: 4 }}>{tName}</span>
            </button>
          );
        })}
    </>
  );
}

export function SourceTree() {
  const sources = useAppStore((s) => s.sources);
  const sourcesLoading = useAppStore((s) => s.sourcesLoading);
  const scanning = useAppStore((s) => s.scanning);
  const activeSourceId = useAppStore((s) => s.activeSourceId);
  // Which tensor row to mark, in the catalog's own spelling. `activeTensorId`
  // may be a bare source_id -- from a link, or from clicking a source rather
  // than one of its tensors -- and no row is named that way. Only the Flight
  // server knows which field it binds as a source's default, and `tile_info` is
  // where it says so, so the grid the render path already fetched is what
  // resolves it; until that lands there is nothing better than the id itself.
  const activeTensorId = useAppStore((s) => {
    const info = selectTileInfo(s);
    return info ? splitArrayVersion(info.array_id).arrayId : s.activeTensorId;
  });
  const selectSource = useAppStore((s) => s.selectSource);
  const querySources = useAppStore((s) => s.querySources);
  const recentIds = useAppStore((s) => s.recentIds);
  const recentSources = useAppStore((s) => s.recentSources);
  const syncRecents = useAppStore((s) => s.syncRecents);
  const hydrateRecents = useAppStore((s) => s.hydrateRecents);

  const [query, setQuery] = useState("");
  const [serverFilteredIds, setServerFilteredIds] = useState<Set<string> | null>(null);
  const [serverQueryLoading, setServerQueryLoading] = useState(false);
  // "Recent" starts open: a node nobody opens is a node nobody finds, and it is
  // the one place an uploaded source can be reached at all.
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(
    () => new Set([RECENT_FOLDER_ID]),
  );

  // Another tab's writes. localStorage is shared but does not re-render, so
  // without this two tabs of one link would drift apart -- which is the reason
  // the list is not in sessionStorage.
  useEffect(() => {
    syncRecents(readRecents());
    return subscribeRecents(syncRecents);
  }, [syncRecents]);

  // Re-resolved when the list changes or the catalog does: a recent id that the
  // catalog has just listed should stop costing a `tile_info` round trip, and
  // one the server has dropped should leave the node.
  useEffect(() => {
    void hydrateRecents();
  }, [hydrateRecents, recentIds, sources]);

  // Determine if we should use server-side queries
  const useServerQuery = sources.length > SERVER_QUERY_THRESHOLD;

  // Debounce server queries
  const [debouncedQuery, setDebouncedQuery] = useState("");
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query), 300);
    return () => clearTimeout(timer);
  }, [query]);

  // Server-side filtering
  useEffect(() => {
    if (!useServerQuery || !debouncedQuery.trim()) {
      setServerFilteredIds(null);
      return;
    }

    const q = debouncedQuery.trim().toLowerCase();
    const escaped = q.replace(/'/g, "''").replace(/%/g, "\\%").replace(/_/g, "\\_");
    const sql = `SELECT source_id FROM sources WHERE
      LOWER(source_id) LIKE '%${escaped}%' OR
      LOWER(source_url) LIKE '%${escaped}%' OR
      LOWER(source_type) LIKE '%${escaped}%'`;

    setServerQueryLoading(true);
    querySources(sql)
      .then((result) => {
        const ids = new Set(result.rows.map((r) => r.source_id as string));
        setServerFilteredIds(ids);
        setServerQueryLoading(false);
      })
      .catch((err) => {
        console.warn("Server query failed:", err);
        setServerFilteredIds(null);
        setServerQueryLoading(false);
      });
  }, [debouncedQuery, useServerQuery, querySources]);

  // Client-side filter
  const filteredSources = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return sources;

    if (serverFilteredIds) {
      return sources.filter((s) => serverFilteredIds.has(s.source_id));
    }

    return sources.filter((s) => matchesQuery(s, q));
  }, [query, sources, serverFilteredIds]);

  // Build tree from filtered sources
  const tree = useMemo(() => buildTree(filteredSources), [filteredSources]);

  // Recents filter client-side even when the catalog has switched to a
  // server-side query: that query is SQL against the catalog, which by
  // construction does not hold the uploads this node exists to show.
  const recent = useMemo(() => {
    const q = query.trim().toLowerCase();
    const matching = q ? recentSources.filter((s) => matchesQuery(s, q)) : recentSources;
    return recentNode(matching);
  }, [recentSources, query]);

  // Filter tree when search is active (client-side only)
  const displayTree = useMemo(() => {
    if (!query.trim() || serverFilteredIds) {
      return tree;
    }

    const q = query.trim().toLowerCase();
    const matchingIds = new Set(
      filteredSources
        .filter((s) => `${s.source_id} ${s.source_url} ${s.source_type}`.toLowerCase().includes(q))
        .map((s) => s.source_id)
    );

    const newExpanded = new Set(expandedFolders);
    const filtered = filterTree(tree, matchingIds, newExpanded);
    if (filtered && newExpanded.size !== expandedFolders.size) {
      setExpandedFolders(newExpanded);
    }
    return filtered ?? tree;
  }, [tree, query, filteredSources, serverFilteredIds, expandedFolders]);

  const listRef = useRef<HTMLDivElement | null>(null);
  // The selection this has already revealed. Latched so a later catalog poll --
  // which rebuilds the tree every 60s -- cannot re-open a folder the user just
  // collapsed, while a selection the catalog does not hold *yet* stays unlatched
  // and is revealed by whichever poll first brings it in.
  const revealed = useRef<string | null>(null);
  const pendingScroll = useRef<string | null>(null);

  useEffect(() => {
    if (!activeSourceId) {
      revealed.current = null;
      return;
    }
    if (revealed.current === activeSourceId) return;
    // A link can select a source that is filtered out by the search box, or one
    // the catalog has not listed (still scanning, or past its cap). Nothing to
    // reveal then, and nothing latched, so clearing the search brings it in.
    const folders = folderPathTo(tree, activeSourceId);
    if (!folders) return;
    revealed.current = activeSourceId;
    pendingScroll.current = activeSourceId;
    if (folders.length > 0) {
      setExpandedFolders((prev) => {
        if (folders.every((id) => prev.has(id))) return prev;
        const next = new Set(prev);
        for (const id of folders) next.add(id);
        return next;
      });
    }
  }, [activeSourceId, tree]);

  // Deliberately every commit: the row this wants may not exist until the
  // expansion above has rendered, and it is one ref read until it does.
  useEffect(() => {
    const id = pendingScroll.current;
    if (id === null) return;
    const row = listRef.current?.querySelector(`[data-source-id="${CSS.escape(id)}"]`);
    if (!row) return;
    pendingScroll.current = null;
    row.scrollIntoView({ block: "nearest" });
  });

  const toggleFolder = useCallback((id: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  /**
   * Selecting from "Recent", which must leave the catalog exactly where it is.
   *
   * The reveal effect above keys on `activeSourceId` alone and so cannot tell a
   * catalog click from a shortcut; without this, picking a row near the top of
   * the pane expands a folder chain further down and scrolls the tree to it --
   * a jump to somewhere the reader did not click. Recording the origin here is
   * what the effect cannot work out for itself.
   *
   * Marking it *revealed* rather than adding a flag to skip: the two say the
   * same thing (this selection needs no reveal) and the latch already means
   * "shown", so a later catalog poll does not reopen the question either. A
   * catalog click and a shared link leave it unlatched and still reveal.
   */
  const selectFromRecent = useCallback(
    (sourceId: string, tensorId?: string) => {
      revealed.current = sourceId;
      selectSource(sourceId, tensorId);
    },
    [selectSource],
  );

  return (
    <section style={{ display: "grid", gridTemplateRows: "auto 1fr", height: "100%" }}>
      <div style={{ padding: "0.5rem 1rem" }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={useServerQuery ? "Search (SQL filter)" : "Search sources"}
          aria-label="Search sources"
          style={{ width: "100%" }}
        />
        {useServerQuery && (
          <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
            {sources.length.toLocaleString()} sources • Server-side filter
          </div>
        )}
      </div>

      <div ref={listRef} style={{ overflow: "auto" }}>
        {sourcesLoading || serverQueryLoading ? (
          <div style={{ padding: "0.5rem 1rem", opacity: 0.8 }}>
            {serverQueryLoading ? "Searching..." : "Loading sources..."}
          </div>
        ) : (
          <>
            {recent && (
              <TreeRow
                node={recent}
                activeSourceId={activeSourceId}
                activeTensorId={activeTensorId}
                expandedFolders={expandedFolders}
                toggleFolder={toggleFolder}
                selectSource={selectFromRecent}
              />
            )}
            {/* The empty notice is about the catalog, so it is suppressed while
                "Recent" has rows -- "No sources" above a list of sources reads
                as a bug. */}
            {filteredSources.length === 0 && !recent ? (
              <div style={{ padding: "0.5rem 1rem", opacity: 0.8 }}>
                {scanning && sources.length === 0
                  ? "Indexing data folder… (sources will appear as they are found)"
                  : "No sources"}
              </div>
            ) : (
              displayTree.children.map((child) => (
                <TreeRow
                  key={child.id}
                  node={child}
                  activeSourceId={activeSourceId}
                  activeTensorId={activeTensorId}
                  expandedFolders={expandedFolders}
                  toggleFolder={toggleFolder}
                  selectSource={selectSource}
                />
              ))
            )}
          </>
        )}
      </div>
    </section>
  );
}
