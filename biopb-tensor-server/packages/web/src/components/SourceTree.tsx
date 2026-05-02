"use client";

import { useMemo, useState } from "react";
import { useAppStore } from "../store";

interface TreeNode {
  name: string;
  path: string;
  children: Map<string, TreeNode>;
  sourceIds: string[];
}

function createNode(name: string, path: string): TreeNode {
  return { name, path, children: new Map(), sourceIds: [] };
}

function sourceLabel(sourceUrl: string): string {
  try {
    const url = new URL(sourceUrl);
    const parts = url.pathname.split("/").filter(Boolean);
    return parts[parts.length - 1] || url.hostname || sourceUrl;
  } catch {
    const parts = sourceUrl.split("/").filter(Boolean);
    return parts[parts.length - 1] || sourceUrl;
  }
}

export function SourceTree() {
  const sources = useAppStore((s) => s.sources);
  const sourcesLoading = useAppStore((s) => s.sourcesLoading);
  const activeSourceId = useAppStore((s) => s.activeSourceId);
  const activeTensorId = useAppStore((s) => s.activeTensorId);
  const selectSource = useAppStore((s) => s.selectSource);

  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return sources;
    return sources.filter((s) => {
      const hay = `${s.source_id} ${s.source_url} ${s.source_type}`.toLowerCase();
      return hay.includes(q);
    });
  }, [query, sources]);

  const tree = useMemo(() => {
    const root = createNode("/", "");
    for (const src of filtered) {
      let current = root;
      const parts = src.source_url
        .replace(/^\w+:\/\//, "")
        .split("/")
        .filter(Boolean);
      const pathParts = parts.length > 0 ? parts : [src.source_id];
      for (const part of pathParts) {
        const nextPath = current.path ? `${current.path}/${part}` : part;
        if (!current.children.has(part)) {
          current.children.set(part, createNode(part, nextPath));
        }
        current = current.children.get(part) as TreeNode;
      }
      current.sourceIds.push(src.source_id);
    }
    return root;
  }, [filtered]);

  const byId = useMemo(() => new Map(sources.map((s) => [s.source_id, s])), [sources]);

  const renderNode = (node: TreeNode, depth: number): JSX.Element[] => {
    const rows: JSX.Element[] = [];

    const childNodes = [...node.children.values()].sort((a, b) => a.name.localeCompare(b.name));
    for (const child of childNodes) {
      rows.push(
        <div key={`dir:${child.path}`} className="tree-item" style={{ paddingLeft: 10 + depth * 12 }}>
          <span style={{ opacity: 0.7 }}>{child.name}</span>
        </div>,
      );
      rows.push(...renderNode(child, depth + 1));
    }

    for (const sourceId of node.sourceIds) {
      const src = byId.get(sourceId);
      if (!src) continue;
      const isActive = src.source_id === activeSourceId;

      rows.push(
        <button
          key={`src:${src.source_id}`}
          className={`tree-item ${isActive ? "active" : ""}`}
          style={{ width: "100%", textAlign: "left", paddingLeft: 10 + depth * 12 }}
          onClick={() => selectSource(src.source_id)}
          title={src.source_url}
        >
          <span>{sourceLabel(src.source_url)}</span>
          <span className="tensor-pill" style={{ marginLeft: "auto" }}>
            {src.tensors.length}
          </span>
        </button>,
      );

      if (isActive) {
        for (const t of src.tensors) {
          const tActive = t.array_id === activeTensorId;
          rows.push(
            <button
              key={`tensor:${src.source_id}:${t.array_id}`}
              className={`tree-item ${tActive ? "active" : ""}`}
              style={{ width: "100%", textAlign: "left", paddingLeft: 24 + depth * 12, opacity: 0.9 }}
              onClick={() => selectSource(src.source_id, t.array_id)}
              title={`${t.array_id} (${t.dtype})`}
            >
              {t.array_id}
            </button>,
          );
        }
      }
    }

    return rows;
  };

  return (
    <section style={{ display: "grid", gridTemplateRows: "auto 1fr", height: "100%" }}>
      <div style={{ display: "grid", gap: 8 }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search sources"
          aria-label="Search sources"
        />
      </div>

      <div style={{ overflow: "auto", marginTop: 10 }}>
        {sourcesLoading ? (
          <div style={{ opacity: 0.8 }}>Loading sources...</div>
        ) : filtered.length === 0 ? (
          <div style={{ opacity: 0.8 }}>No sources</div>
        ) : (
          renderNode(tree, 0)
        )}
      </div>
    </section>
  );
}
