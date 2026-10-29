import { useCallback, useEffect, useRef, useState } from "react";
import type { AddSourceJobStatus } from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";
import { addCaution, describeAdd, runAdd } from "../utils/addSource";
import { FileBrowser } from "./admin/FileBrowser";

const NOTE: React.CSSProperties = { fontSize: 11, color: "#64748b", marginTop: 4 };

/**
 * "Add path…" for the tree: pick a file or folder on the *server*, and it is
 * served from then until the server restarts (it is not written to the config).
 * The same thing a drag-and-drop does from napari, reached from the browser.
 *
 * A folder is walked and can take a while, so the tree is refreshed as sources
 * land rather than at the end, and the walk can be cancelled (what it has
 * registered stays).
 */
export function AddSourceControl() {
  const client = useAppStore((s) => s.client);
  const refreshTree = useAppStore((s) => s.refreshTree);
  const [browsing, setBrowsing] = useState(false);
  const [job, setJob] = useState<AddSourceJobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const alive = useRef(true);
  const shown = useRef(0);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  // Stable for FileBrowser, whose load effect depends on it.
  const browse = useCallback((p?: string) => client!.http.browse(p), [client]);
  const running = job?.state === "running";

  const add = async (path: string) => {
    if (!client) return;
    setBrowsing(false);
    setError(null);
    setJob(null);
    shown.current = 0;
    const reread = () => void refreshTree().catch(() => {});
    try {
      const last = await runAdd(
        client.http,
        path,
        (s) => {
          if (!alive.current) return;
          setJob(s);
          // New sources are on the server's tree as they register.
          const n = s.progress.added_count ?? 0;
          if (n !== shown.current) {
            shown.current = n;
            reread();
          }
        },
        undefined,
        () => !alive.current,
      );
      if (alive.current) setJob(last);
    } catch (e) {
      if (alive.current) setError(e instanceof Error ? e.message : String(e));
    }
    reread();
  };

  const cancel = () => {
    if (!client || !job) return;
    client.http.cancelAddSource(job.source_id).catch(() => {});
  };

  const result = job?.result;
  const caution = result ? addCaution(result) : null;

  return (
    <>
      <button
        type="button"
        className="icon-btn"
        disabled={!client || running}
        onClick={() => setBrowsing(true)}
        title="Serve a file or folder on the server until it restarts"
        style={{ flexShrink: 0, whiteSpace: "nowrap" }}
      >
        + Add path…
      </button>

      {browsing && (
        <FileBrowser browse={browse} onPick={add} onClose={() => setBrowsing(false)} />
      )}

      {/* Under the search box: sits in the header's column, not in the row. */}
      <div style={{ gridColumn: "1 / -1" }}>
        {running && job && (
          <div style={NOTE}>
            {job.cancel_requested ? "Cancelling" : "Adding"} {job.source_id}…{" "}
            {(job.progress.added_count ?? 0).toLocaleString()} found{" "}
            <button type="button" className="icon-btn" onClick={cancel} disabled={job.cancel_requested}>
              Cancel
            </button>
          </div>
        )}
        {!running && job?.state === "error" && (
          <div style={{ ...NOTE, color: "#fca5a5" }}>
            Could not add {job.source_id}: {job.error}{" "}
            <button type="button" className="icon-btn" onClick={() => setJob(null)}>
              Dismiss
            </button>
          </div>
        )}
        {!running && result && job?.state !== "error" && (
          <div style={NOTE}>
            {job?.state === "cancelled" ? "Stopped. " : ""}
            {describeAdd(result)}
            {caution && <> — {caution}</>}{" "}
            <button type="button" className="icon-btn" onClick={() => setJob(null)}>
              Dismiss
            </button>
            {result.failed.slice(0, 3).map((f) => (
              <div key={f.path} style={{ color: "#fca5a5" }} title={f.reason}>
                {f.path}: {f.reason}
              </div>
            ))}
          </div>
        )}
        {error && <div style={{ ...NOTE, color: "#fca5a5" }}>{error}</div>}
      </div>
    </>
  );
}
