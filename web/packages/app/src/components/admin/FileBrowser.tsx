import { useCallback, useEffect, useState } from "react";
import type { BrowseResponse } from "@biopb/tensor-flight-client";
import { Modal } from "../Modal";

/**
 * Server-side file/directory chooser (biopb/biopb#244).
 *
 * Navigates the *server's* filesystem via `GET /api/admin/browse`, so the user
 * picks a real server path (`/data/microscopy`) instead of typing it. Only
 * mounted in **local mode** — where the server's filesystem is the user's own
 * machine — so it never discloses a remote host's files; the caller gates it on
 * `AdminStatus.local`.
 *
 * The user can navigate into directories, go up, pick a directory ("Use this
 * folder"), or pick a file (click it). Path joining infers the separator from
 * the server-returned absolute path, so it works against a Windows server too.
 */

interface FileBrowserProps {
  /** Bound `client.http.browse`. */
  browse: (path?: string) => Promise<BrowseResponse>;
  /** Where to start (the row's current value); blank → server home dir. */
  initialPath?: string;
  onPick: (path: string) => void;
  onClose: () => void;
}

function joinPath(dir: string, name: string): string {
  const sep = dir.includes("\\") && !dir.includes("/") ? "\\" : "/";
  return dir.endsWith(sep) ? `${dir}${name}` : `${dir}${sep}${name}`;
}

export function FileBrowser({ browse, initialPath, onPick, onClose }: FileBrowserProps) {
  const [data, setData] = useState<BrowseResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (path?: string) => {
      setLoading(true);
      setError(null);
      try {
        setData(await browse(path));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [browse],
  );

  useEffect(() => {
    load(initialPath || undefined);
  }, [load, initialPath]);

  const dir = data?.path ?? "";

  return (
    <Modal
      title="Choose a file or folder"
      onClose={onClose}
      className="wide"
      labelId="file-browser-title"
    >
      <div className="file-browser">
        <div className="file-browser-bar">
          <button
            type="button"
            className="icon-btn"
            disabled={loading || !data?.parent}
            onClick={() => data?.parent && load(data.parent)}
            title="Up one level"
          >
            ↑ Up
          </button>
          <code className="file-browser-path" title={dir}>
            {dir || "…"}
          </code>
        </div>

        {error ? (
          <div className="admin-banner error">{error}</div>
        ) : (
          <ul className="file-browser-list">
            {loading && <li className="file-browser-empty">Loading…</li>}
            {!loading && data?.entries.length === 0 && (
              <li className="file-browser-empty">Empty folder.</li>
            )}
            {!loading &&
              data?.entries.map((e) => {
                const full = joinPath(dir, e.name);
                return (
                  <li key={e.name}>
                    <button
                      type="button"
                      className={`file-browser-entry${e.is_dir ? " is-dir" : ""}`}
                      onClick={() => (e.is_dir ? load(full) : onPick(full))}
                    >
                      <span className="file-browser-icon" aria-hidden="true">
                        {e.is_dir ? "📁" : "📄"}
                      </span>
                      <span className="file-browser-name">{e.name}</span>
                    </button>
                  </li>
                );
              })}
          </ul>
        )}

        {data?.truncated && (
          <p className="file-browser-note">
            Showing the first entries only — navigate into a subfolder to narrow
            down.
          </p>
        )}

        <div className="admin-modal-actions">
          <button type="button" className="icon-btn" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="submit-btn"
            disabled={!dir}
            onClick={() => onPick(dir)}
          >
            Use this folder
          </button>
        </div>
      </div>
    </Modal>
  );
}
