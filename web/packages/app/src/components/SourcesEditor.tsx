import { useState } from "react";

/**
 * Structured editor for the config `sources` array (the admin page's headline).
 *
 * Pure-controlled: it never holds its own copy of the data — every edit calls
 * `onChange` with the next sources array, which AdminPage merges into the one
 * canonical `config` object (so the raw-JSON view stays in sync). Mirrors the
 * read-only browse rows in SourceTree.tsx visually.
 */

export type SourceEntry = Record<string, unknown>;

interface SourcesEditorProps {
  sources: SourceEntry[];
  onChange: (next: SourceEntry[]) => void;
  /** Per-source-index validation messages from a rejected PUT (422). */
  errorsByIndex: Record<number, string[]>;
  disabled?: boolean;
  /** Open the server-side file chooser for local row `i`, seeded from its current
   * URL. Provided only in local mode (biopb/biopb#244); when absent the row keeps
   * the typed-path input alone. */
  onBrowse?: (i: number) => void;
}

// Storage types accepted by SourceConfig.type (config.py). "" = auto-detect.
const LOCAL_TYPES = [
  "",
  "zarr",
  "hdf5",
  "ome-tiff",
  "ome-tiff-multifile",
  "ome-zarr",
  "ome-zarr-hcs",
  "aics",
];

function str(v: unknown): string {
  return typeof v === "string" ? v : v == null ? "" : String(v);
}

export function SourcesEditor({
  sources,
  onChange,
  errorsByIndex,
  disabled,
  onBrowse,
}: SourcesEditorProps) {
  const [addOpen, setAddOpen] = useState(false);

  function update(i: number, patch: SourceEntry) {
    const next = sources.map((s, idx) => (idx === i ? { ...s, ...patch } : s));
    onChange(next);
  }

  function remove(i: number) {
    onChange(sources.filter((_, idx) => idx !== i));
  }

  function addLocal() {
    onChange([...sources, { url: "", monitor: false }]);
    setAddOpen(false);
  }

  function addRemote() {
    // A remote tensor server is identified by a grpc:// url + an alias.
    onChange([...sources, { url: "grpc://", alias: "", monitor: false }]);
    setAddOpen(false);
  }

  const isRemote = (s: SourceEntry) =>
    str(s.url).startsWith("grpc://") || s.alias != null;

  // Canonical per-entry display title (e.g. "Local Source 1" / "Remote Source 2"),
  // numbered within its kind — sources have no single name field, so this is the
  // group heading shown above each entry's fields.
  let localN = 0;
  let remoteN = 0;
  const titles = sources.map((s) =>
    isRemote(s) ? `Remote Source ${++remoteN}` : `Local Source ${++localN}`,
  );

  return (
    <section className="sources-editor">
      <div className="sources-editor-head">
        <div className="add-menu">
          <button
            type="button"
            className="icon-btn"
            disabled={disabled}
            onClick={() => setAddOpen((v) => !v)}
          >
            + Add ▾
          </button>
          {addOpen && (
            <div className="add-menu-list">
              <button type="button" onClick={addLocal}>
                Add local folder / file
              </button>
              <button type="button" onClick={addRemote}>
                Add remote tensor server…
              </button>
            </div>
          )}
        </div>
      </div>

      {sources.length === 0 ? (
        <div className="sources-empty">
          <p>No data sources yet.</p>
          <button type="button" className="submit-btn" disabled={disabled} onClick={addLocal}>
            Add a data folder
          </button>
        </div>
      ) : (
        <ul className="source-rows">
          {sources.map((s, i) => {
            const remote = isRemote(s);
            const errs = errorsByIndex[i] ?? [];
            return (
              <li key={i} className={`entry-row${errs.length ? " has-error" : ""}`}>
                <div className="entry-head">
                  <span className="entry-title">{titles[i]}</span>
                  <button
                    type="button"
                    className="entry-delete"
                    disabled={disabled}
                    onClick={() => remove(i)}
                  >
                    Delete
                  </button>
                </div>
                <div className="entry-fields">
                <div className="source-row-grid">
                  <label>
                    {remote ? "URL (grpc://host:port)" : "Path / URL"}
                    {!remote && onBrowse ? (
                      <span className="source-row-path">
                        <input
                          type="text"
                          value={str(s.url)}
                          disabled={disabled}
                          placeholder="/data/microscopy"
                          onChange={(e) => update(i, { url: e.target.value })}
                        />
                        <button
                          type="button"
                          className="icon-btn"
                          disabled={disabled}
                          onClick={() => onBrowse(i)}
                        >
                          Browse…
                        </button>
                      </span>
                    ) : (
                      <input
                        type="text"
                        value={str(s.url)}
                        disabled={disabled}
                        placeholder={remote ? "grpc://lab-nas:8815" : "/data/microscopy"}
                        onChange={(e) => update(i, { url: e.target.value })}
                      />
                    )}
                  </label>

                  {remote ? (
                    <>
                      <label>
                        Alias
                        <input
                          type="text"
                          value={str(s.alias)}
                          disabled={disabled}
                          placeholder="nas"
                          onChange={(e) => update(i, { alias: e.target.value })}
                        />
                      </label>
                      <label>
                        Credentials profile
                        <input
                          type="text"
                          value={str(s.credentials_profile)}
                          disabled={disabled}
                          onChange={(e) =>
                            update(i, { credentials_profile: e.target.value })
                          }
                        />
                      </label>
                    </>
                  ) : (
                    <>
                      <label>
                        Type
                        <select
                          value={str(s.type)}
                          disabled={disabled}
                          onChange={(e) =>
                            update(i, { type: e.target.value || undefined })
                          }
                        >
                          {LOCAL_TYPES.map((t) => (
                            <option key={t || "auto"} value={t}>
                              {t || "auto-detect"}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Source ID
                        <input
                          type="text"
                          value={str(s.source_id)}
                          disabled={disabled}
                          placeholder="(auto)"
                          onChange={(e) =>
                            update(i, { source_id: e.target.value || undefined })
                          }
                        />
                      </label>
                    </>
                  )}

                  <label className="source-row-monitor">
                    <input
                      type="checkbox"
                      checked={s.monitor === true}
                      disabled={disabled}
                      onChange={(e) => update(i, { monitor: e.target.checked })}
                    />
                    Monitor for changes
                  </label>

                  {"path" in s && (
                    <label className="source-row-deprecated">
                      <span className="adv-field-name">
                        path <span className="deprecated-tag">deprecated</span>
                      </span>
                      <input
                        type="text"
                        value={str(s.path)}
                        disabled={disabled}
                        onChange={(e) =>
                          update(i, { path: e.target.value || undefined })
                        }
                      />
                      <span className="adv-field-help">
                        Deprecated alias for <code>url</code> — set the URL field
                        above and clear this.
                      </span>
                    </label>
                  )}
                </div>

                {errs.length > 0 && (
                  <ul className="source-row-errors">
                    {errs.map((m, k) => (
                      <li key={k}>{m}</li>
                    ))}
                  </ul>
                )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
