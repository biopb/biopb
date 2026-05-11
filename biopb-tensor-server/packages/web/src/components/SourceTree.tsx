"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useAppStore } from "../store";
import type { DataSourceDescriptor } from "@biopb/tensor-flight-client";

// Threshold for switching to server-side SQL query
const SERVER_QUERY_THRESHOLD = 1000;

interface TreeNode {
  id: string;           // unique id (path for folders, source_id for sources)
  name: string;         // display name
  type: "folder" | "source";
  children: TreeNode[];
  source?: DataSourceDescriptor;  // only for source nodes
  depth: number;
}

function getPathParts(url: string): string[] {
  if (!url) return [];
  try {
    const parsed = new URL(url);
    return parsed.pathname.split("/").filter(Boolean);
  } catch {
    return url.split("/").filter(Boolean);
  }
}

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
        name: src.source_id,
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
    const sourceName = parts[parts.length - 1]!;
    current.children.push({
      id: src.source_id,
      name: sourceName,
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
          // Adjust depths of merged children
          for (const gc of child.children) {
            gc.depth = child.depth + 1;
          }
        }

        // Continue flattening in case new structure allows more flattening
        flattenPaths(child);
      }
    }
  }
  flattenPaths(root);

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

interface TreeRowProps {
  node: TreeNode;
  activeSourceId: string | null;
  activeTensorId: string | null;
  expandedFolders: Set<string>;
  toggleFolder: (id: string) => void;
  selectSource: (sourceId: string, tensorId?: string) => void;
}

function TreeRow({
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
          gap: 8,
          paddingLeft: indent,
        }}
        onClick={() => {
          if (src.tensors.length === 1) {
            selectSource(src.source_id, src.tensors[0]?.array_id);
          } else {
            selectSource(src.source_id);
          }
        }}
        title={src.source_url}
      >
        <span style={{ flex: 1 }}>{node.name}</span>
        {hasMultipleTensors ? (
          <span className="tensor-pill">{src.tensors.length}</span>
        ) : firstTensor ? (
          <span className="dim-badge" title={formatShape(firstTensor.shape)}>
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
                gap: 8,
                fontSize: 12,
              }}
              onClick={() => selectSource(src.source_id, t.array_id)}
              title={`${t.array_id}\nShape: ${formatShape(t.shape)}\nDtype: ${t.dtype}`}
            >
              <span style={{ flex: 1 }}>{tName}</span>
            </button>
          );
        })}
    </>
  );
}

export function SourceTree() {
  const sources = useAppStore((s) => s.sources);
  const sourcesLoading = useAppStore((s) => s.sourcesLoading);
  const activeSourceId = useAppStore((s) => s.activeSourceId);
  const activeTensorId = useAppStore((s) => s.activeTensorId);
  const selectSource = useAppStore((s) => s.selectSource);
  const querySources = useAppStore((s) => s.querySources);

  const [query, setQuery] = useState("");
  const [serverFilteredIds, setServerFilteredIds] = useState<Set<string> | null>(null);
  const [serverQueryLoading, setServerQueryLoading] = useState(false);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());

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

    return sources.filter((s) => {
      const hay = `${s.source_id} ${s.source_url} ${s.source_type}`.toLowerCase();
      return hay.includes(q);
    });
  }, [query, sources, serverFilteredIds]);

  // Build tree from filtered sources
  const tree = useMemo(() => buildTree(filteredSources), [filteredSources]);

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

      <div style={{ overflow: "auto" }}>
        {sourcesLoading || serverQueryLoading ? (
          <div style={{ padding: "0.5rem 1rem", opacity: 0.8 }}>
            {serverQueryLoading ? "Searching..." : "Loading sources..."}
          </div>
        ) : filteredSources.length === 0 ? (
          <div style={{ padding: "0.5rem 1rem", opacity: 0.8 }}>No sources</div>
        ) : displayTree ? (
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
        ) : null}
      </div>
    </section>
  );
}