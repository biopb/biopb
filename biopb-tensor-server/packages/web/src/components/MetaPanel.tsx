"use client";

import { useEffect, useState } from "react";
import { useAppStore } from "../store";

interface MetaPanelProps {
  sourceId: string;
}

export function MetaPanel({ sourceId }: MetaPanelProps) {
  const client = useAppStore((s) => s.client);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<Record<string, unknown> | null>(null);

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
      <button className="icon-btn" onClick={() => setOpen((v) => !v)}>
        {open ? "Hide metadata" : "Show metadata"}
      </button>

      {open && (
        <div style={{ marginTop: 8 }}>
          {loading && <div>Loading metadata...</div>}
          {error && <div className="error-toast" style={{ position: "static" }}>{error}</div>}
          {!loading && !error && (
            <pre style={{ whiteSpace: "pre-wrap", overflow: "auto", maxHeight: 280 }}>
              {JSON.stringify(metadata ?? {}, null, 2)}
            </pre>
          )}
        </div>
      )}
    </section>
  );
}
