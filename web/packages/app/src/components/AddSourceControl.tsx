import { useCallback, useEffect, useRef, useState } from "react";
import type { AddSourceJobStatus } from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";
import { addNotice, runAdd } from "../utils/addSource";
import { useNotices } from "../utils/notices";
import { FileBrowser } from "./admin/FileBrowser";

const NOTE: React.CSSProperties = { fontSize: 11, color: "#64748b", marginTop: 4 };

/** Fewest ms between tree re-lists while a long add runs. */
const REFRESH_MS = 5000;

/**
 * "Add path…" for the tree: pick a file or folder on the *server*, and it is
 * served from then until the server restarts (it is not written to the config).
 * The same thing a drag-and-drop does from napari, reached from the browser.
 *
 * A folder is walked and can take a while, so the tree is refreshed every few
 * seconds as sources land rather than only at the end, and the walk can be
 * cancelled (what it has registered stays).
 */
export function AddSourceControl() {
  const client = useAppStore((s) => s.client);
  const refreshTree = useAppStore((s) => s.refreshTree);
  const [browsing, setBrowsing] = useState(false);
  // The add under way; its outcome goes to a notice, not here.
  const [job, setJob] = useState<AddSourceJobStatus | null>(null);
  const alive = useRef(true);
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
    setJob(null);
    const { notify } = useNotices.getState();
    // Sources show up in the tree as they register, but re-listing the catalog
    // is a request per open folder: at most one at a time, and not every poll.
    let shown = 0;
    let lastRead = 0;
    let reading = false;
    const reread = () => {
      reading = true;
      lastRead = Date.now();
      refreshTree()
        .catch(() => {})
        .finally(() => {
          reading = false;
        });
    };
    try {
      const last = await runAdd(
        client.http,
        path,
        (s) => {
          if (!alive.current) return;
          setJob(s);
          const n = s.progress.added_count ?? 0;
          if (n !== shown && !reading && Date.now() - lastRead >= REFRESH_MS) {
            shown = n;
            reread();
          }
        },
        { stopped: () => !alive.current },
      );
      const outcome = addNotice(last);
      if (outcome) notify(outcome.kind, outcome.text, outcome.details);
    } catch (e) {
      notify("error", `Could not add ${path}: ${e instanceof Error ? e.message : String(e)}`);
    }
    if (alive.current) setJob(null);
    reread();
  };

  const cancel = () => {
    if (!client || !job) return;
    client.http.cancelAddSource(job.source_id).catch(() => {});
  };

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
      {running && job && (
        <div style={{ ...NOTE, gridColumn: "1 / -1" }}>
          {job.cancel_requested ? "Cancelling" : "Adding"} {job.source_id}…{" "}
          {(job.progress.added_count ?? 0).toLocaleString()} found{" "}
          <button type="button" className="icon-btn" onClick={cancel} disabled={job.cancel_requested}>
            Cancel
          </button>
        </div>
      )}
    </>
  );
}
