"use client";

import { useEffect, useState } from "react";
import { useAppStore } from "../store";

interface MetaPanelProps {
  sourceId: string;
}

function isEmptyForDisplay(value: unknown): boolean {
  if (value === null || value === undefined) return true;
  if (Array.isArray(value)) {
    if (value.length === 0) return true;
    return value.every(isEmptyForDisplay);
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) return true;
    return entries.every(([, v]) => isEmptyForDisplay(v));
  }
  return false;
}

function JsonNode({ value, depth = 0 }: { value: unknown; depth?: number }) {
  const [expanded, setExpanded] = useState(depth < 2);

  if (value === null) return <span style={{ color: "#f87171" }}>null</span>;
  if (value === undefined) return <span style={{ color: "#f87171" }}>undefined</span>;
  if (typeof value === "boolean") return <span style={{ color: "#60a5fa" }}>{value ? "true" : "false"}</span>;
  if (typeof value === "number") return <span style={{ color: "#a78bfa" }}>{value}</span>;
  if (typeof value === "string") {
    if (value.length > 50) {
      return (
        <span style={{ color: "#34d399" }} title={value}>
          "{value.slice(0, 50)}..."
        </span>
      );
    }
    return <span style={{ color: "#34d399" }}>"{value}"</span>;
  }

  if (Array.isArray(value)) {
    if (value.length === 0) return <span style={{ color: "#64748b" }}>[]</span>;
    if (!expanded) {
      return (
        <span
          style={{ color: "#64748b", cursor: "pointer" }}
          onClick={() => setExpanded(true)}
        >
          [...{value.length}]
        </span>
      );
    }
    return (
      <span>
        <span style={{ color: "#64748b", cursor: "pointer" }} onClick={() => setExpanded(false)}>
          [
        </span>
        <div style={{ marginLeft: 12 }}>
          {value.slice(0, 10).map((v, i) => (
            <div key={i}>
              <span style={{ color: "#64748b" }}>{i}: </span>
              <JsonNode value={v} depth={depth + 1} />
            </div>
          ))}
          {value.length > 10 && <span style={{ color: "#64748b" }}>... {value.length - 10} more</span>}
        </div>
        <span style={{ color: "#64748b" }}>]</span>
      </span>
    );
  }

  if (typeof value === "object") {
    const allEntries = Object.entries(value as Record<string, unknown>);
    const entries = allEntries.filter(([, v]) => !isEmptyForDisplay(v));
    if (entries.length === 0) return <span style={{ color: "#64748b" }}>{}</span>;
    if (!expanded) {
      return (
        <span
          style={{ color: "#64748b", cursor: "pointer" }}
          onClick={() => setExpanded(true)}
        >
          {"{...}"}
        </span>
      );
    }
    return (
      <span>
        <span style={{ color: "#64748b", cursor: "pointer" }} onClick={() => setExpanded(false)}>
          {"{"}
        </span>
        <div style={{ marginLeft: 12 }}>
          {entries.map(([k, v]) => (
            <div key={k}>
              <span style={{ color: "#fbbf24" }}>{k}: </span>
              <JsonNode value={v} depth={depth + 1} />
            </div>
          ))}
        </div>
        <span style={{ color: "#64748b" }}>{"}"}</span>
      </span>
    );
  }

  return <span>{String(value)}</span>;
}

export function MetaPanel({ sourceId }: MetaPanelProps) {
  const client = useAppStore((s) => s.client);
  const sources = useAppStore((s) => s.sources);
  const activeTensorId = useAppStore((s) => s.activeTensorId);
  const [metadata, setMetadata] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const source = sources.find((s) => s.source_id === sourceId);
  const tensor = source?.tensors.find((t) => t.array_id === activeTensorId);

  useEffect(() => {
    let cancelled = false;
    if (!client || !sourceId) {
      setMetadata(null);
      return;
    }

    setLoading(true);
    setError(null);

    client
      .getSourceMetadata(sourceId)
      .then((m) => {
        if (!cancelled) setMetadata(m);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [client, sourceId]);

  return (
    <section className="meta-panel">
      {/* Key info section */}
      {source && (
        <div style={{ marginBottom: 12, padding: 8, background: "#1e2435", borderRadius: 4 }}>
          <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>Source URL</div>
          <div style={{ fontSize: 12, wordBreak: "break-all" }}>{source.source_url || source.source_id}</div>

          {tensor && (
            <>
              <div style={{ fontSize: 11, color: "#64748b", marginTop: 8 }}>Tensor Shape</div>
              <div style={{ fontSize: 12 }}>{tensor.shape.join(" × ")}</div>
              <div style={{ fontSize: 11, color: "#64748b", marginTop: 8 }}>Data Type</div>
              <div style={{ fontSize: 12 }}>{tensor.dtype}</div>
            </>
          )}
        </div>
      )}

      {/* Metadata section */}
      <div style={{ marginBottom: 8 }}>
        <span style={{ fontSize: 11, color: "#64748b" }}>Metadata</span>
      </div>

      {loading && <div style={{ color: "#64748b" }}>Loading...</div>}
      {error && <div style={{ color: "#f87171" }}>{error}</div>}
      {!loading && !error && metadata && <JsonNode value={metadata} />}
    </section>
  );
}