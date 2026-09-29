"use client";

import { memo, useEffect, useState } from "react";
import type { DataSourceDescriptor } from "@biopb/tensor-flight-client";
import { selectTileInfo, useAppStore } from "../store";
import {
  MAX_ROWS,
  autoExpanded,
  hasVisibleEntries,
  visibleEntries,
} from "../utils/jsonTree";

interface MetaPanelProps {
  sourceId: string;
}

/** The affordance for the rows a node is holding back. */
function MoreRow({ count, onClick }: { count: number; onClick: () => void }) {
  return (
    <div
      style={{ color: "#64748b", cursor: "pointer" }}
      onClick={onClick}
      title="Show the remaining entries"
    >
      ... {count.toLocaleString()} more
    </div>
  );
}

/**
 * Memoised because it is recursive: without this, any re-render of the panel
 * re-renders every row below it, and the rows are the expensive part.
 */
const JsonNode = memo(function JsonNode({
  value,
  depth = 0,
  revealed = false,
}: {
  value: unknown;
  depth?: number;
  /**
   * This row came from a "… N more", so it starts shut whatever its depth says.
   *
   * Otherwise revealing 1,991 held-back keys also opens each one: measured at
   * 27k DOM nodes for the MicroManager blob, when the point of the click was to
   * see the key list. Asking for the rest is not asking to expand the rest.
   */
  revealed?: boolean;
}) {
  const [expanded, setExpanded] = useState(() => !revealed && autoExpanded(value, depth));
  const [showAll, setShowAll] = useState(false);

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
          {(showAll ? value : value.slice(0, MAX_ROWS)).map((v, i) => (
            <div key={i}>
              <span style={{ color: "#64748b" }}>{i}: </span>
              <JsonNode value={v} depth={depth + 1} revealed={i >= MAX_ROWS} />
            </div>
          ))}
          {!showAll && value.length > MAX_ROWS && (
            <MoreRow count={value.length - MAX_ROWS} onClick={() => setShowAll(true)} />
          )}
        </div>
        <span style={{ color: "#64748b" }}>]</span>
      </span>
    );
  }

  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    if (!expanded) {
      // Answered without filtering: a collapsed node only chooses between the
      // two placeholders, and the filter is a whole-subtree walk.
      if (!hasVisibleEntries(obj)) return <span style={{ color: "#64748b" }}>{"{}"}</span>;
      return (
        <span
          style={{ color: "#64748b", cursor: "pointer" }}
          onClick={() => setExpanded(true)}
        >
          {`{...${Object.keys(obj).length}}`}
        </span>
      );
    }
    const entries = visibleEntries(obj);
    if (entries.length === 0) return <span style={{ color: "#64748b" }}>{"{}"}</span>;
    const shown = showAll ? entries : entries.slice(0, MAX_ROWS);
    return (
      <span>
        <span style={{ color: "#64748b", cursor: "pointer" }} onClick={() => setExpanded(false)}>
          {"{"}
        </span>
        <div style={{ marginLeft: 12 }}>
          {shown.map(([k, v], i) => (
            <div key={k}>
              <span style={{ color: "#fbbf24" }}>{k}: </span>
              <JsonNode value={v} depth={depth + 1} revealed={i >= MAX_ROWS} />
            </div>
          ))}
          {shown.length < entries.length && (
            <MoreRow
              count={entries.length - shown.length}
              onClick={() => setShowAll(true)}
            />
          )}
        </div>
        <span style={{ color: "#64748b" }}>{"}"}</span>
      </span>
    );
  }

  return <span>{String(value)}</span>;
});

export function MetaPanel({ sourceId }: MetaPanelProps) {
  const client = useAppStore((s) => s.client);
  const activeTensorId = useAppStore((s) => s.activeTensorId);
  // The live grid, so the shape shown is the tensor's now rather than the
  // catalog scan's. See `sliderGrid` for why those can differ.
  const tileInfo = useAppStore(selectTileInfo);
  const [metadata, setMetadata] = useState<Record<string, unknown> | null>(null);
  const [source, setSource] = useState<DataSourceDescriptor | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Addressed by id rather than found in `sources`: the catalog listing is
  // capped and may still be scanning, and this panel describes one source that
  // is already open -- there is no reason for it to depend on the whole list.
  useEffect(() => {
    let cancelled = false;
    if (!client || !sourceId) {
      setSource(null);
      return;
    }
    client.http
      .getSource(sourceId)
      .then((d) => {
        if (!cancelled) setSource(d);
      })
      .catch(() => {
        // The metadata request below reports for both; a missing descriptor
        // just leaves the key-info block out.
        if (!cancelled) setSource(null);
      });
    return () => {
      cancelled = true;
    };
  }, [client, sourceId]);

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
          <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>Array</div>
          <div style={{ fontSize: 12, wordBreak: "break-all" }}>
            {/* `array_id` is the whole address already -- prefixing the
                source_id again produced "src/src/field" on a multi-tensor
                source. */}
            {activeTensorId ?? source.source_id}
          </div>
          <div style={{ fontSize: 11, color: "#64748b", marginTop: 8, marginBottom: 4 }}>Source URL</div>
          <div style={{ fontSize: 12, wordBreak: "break-all" }}>{source.source_url || source.source_id}</div>

          {tileInfo && (
            <>
              <div style={{ fontSize: 11, color: "#64748b", marginTop: 8 }}>Tensor Shape</div>
              <div style={{ fontSize: 12 }}>{tileInfo.shape.join(" × ")}</div>
              <div style={{ fontSize: 11, color: "#64748b", marginTop: 8 }}>Data Type</div>
              <div style={{ fontSize: 12 }}>{tileInfo.dtype}</div>
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
