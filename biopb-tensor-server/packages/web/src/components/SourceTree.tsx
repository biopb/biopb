"use client";

import { useMemo, useState } from "react";
import { useAppStore } from "../store";

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

function computeDisplayNames(allSources: { source_id: string; source_url: string }[]): Map<string, string> {
  const result = new Map<string, string>();

  // Get path parts for each source
  const pathPartsMap = new Map<string, string[]>();
  for (const src of allSources) {
    pathPartsMap.set(src.source_id, getPathParts(src.source_url));
  }

  // Check if any names collide at the base level (last part only)
  const baseNames = new Map<string, string[]>();
  for (const [id, parts] of pathPartsMap) {
    const base = parts[parts.length - 1] || id;
    if (!baseNames.has(base)) baseNames.set(base, []);
    baseNames.get(base)!.push(id);
  }

  // If no collisions, use base names directly
  const hasCollision = Array.from(baseNames.values()).some(ids => ids.length > 1);

  if (!hasCollision) {
    for (const [id, parts] of pathPartsMap) {
      result.set(id, parts[parts.length - 1] || id);
    }
    return result;
  }

  // Find the minimum depth needed to make all names unique
  // Start with 2 parts (parent/filename), increase if still colliding
  let depth = 2;
  const maxDepth = Math.max(...Array.from(pathPartsMap.values()).map(p => p.length));

  while (depth <= maxDepth) {
    const namesAtDepth = new Map<string, string[]>();
    for (const [id, parts] of pathPartsMap) {
      // Take last `depth` parts
      const nameParts = parts.slice(-depth);
      const name = nameParts.join("/") || id;
      if (!namesAtDepth.has(name)) namesAtDepth.set(name, []);
      namesAtDepth.get(name)!.push(id);
    }

    // Check if all unique at this depth
    const allUnique = Array.from(namesAtDepth.values()).every(ids => ids.length === 1);
    if (allUnique) {
      for (const [id, parts] of pathPartsMap) {
        const nameParts = parts.slice(-depth);
        result.set(id, nameParts.join("/") || id);
      }
      return result;
    }

    depth++;
  }

  // Fallback: use full paths
  for (const [id, parts] of pathPartsMap) {
    result.set(id, parts.join("/") || id);
  }
  return result;
}

function formatShape(shape: number[]): string {
  return shape.join("×");
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

  const displayNames = useMemo(() => computeDisplayNames(filtered), [filtered]);

  const rows: JSX.Element[] = useMemo(() => {
    const result: JSX.Element[] = [];

    for (const src of filtered) {
      const isActive = src.source_id === activeSourceId;
      const displayName = displayNames.get(src.source_id) || src.source_id;
      const hasMultipleTensors = src.tensors.length > 1;
      const firstTensor = src.tensors[0];

      result.push(
        <button
          key={`src:${src.source_id}`}
          className={`tree-item ${isActive ? "active" : ""}`}
          style={{
            width: "100%",
            textAlign: "left",
            display: "flex",
            alignItems: "center",
            gap: 8,
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
          <span style={{ flex: 1 }}>{displayName}</span>
          {hasMultipleTensors ? (
            <span className="tensor-pill">{src.tensors.length}</span>
          ) : firstTensor ? (
            <span className="dim-badge" title={formatShape(firstTensor.shape)}>
              {formatShape(firstTensor.shape)}
            </span>
          ) : null}
        </button>
      );

      if (isActive && hasMultipleTensors) {
        for (const t of src.tensors) {
          const tActive = t.array_id === activeTensorId;
          const tName = tensorShortName(t.array_id);
          result.push(
            <button
              key={`tensor:${src.source_id}:${t.array_id}`}
              className={`tree-item tensor-item ${tActive ? "active" : ""}`}
              style={{
                width: "100%",
                textAlign: "left",
                paddingLeft: 24,
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
        }
      }
    }

    return result;
  }, [filtered, activeSourceId, activeTensorId, displayNames, selectSource]);

  return (
    <section style={{ display: "grid", gridTemplateRows: "auto 1fr", height: "100%" }}>
      <div style={{ padding: "0.5rem 1rem" }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search sources"
          aria-label="Search sources"
          style={{ width: "100%" }}
        />
      </div>

      <div style={{ overflow: "auto" }}>
        {sourcesLoading ? (
          <div style={{ padding: "0.5rem 1rem", opacity: 0.8 }}>Loading sources...</div>
        ) : filtered.length === 0 ? (
          <div style={{ padding: "0.5rem 1rem", opacity: 0.8 }}>No sources</div>
        ) : (
          rows
        )}
      </div>
    </section>
  );
}