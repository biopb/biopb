import { useState } from "react";
import {
  sourceItemProperties,
  type ConfigSchema,
  type SchemaProp,
} from "@biopb/tensor-flight-client";
import { SchemaField } from "./admin/SchemaField";

/**
 * Structured editor for the config `sources` array (the admin page's headline).
 *
 * Schema-driven like the other editors: each field's label, control, enum,
 * bounds, and help text come from the `sources` item schema
 * (`sourceItemProperties(schema)`) via the shared `<SchemaField>`; this component
 * only curates *which* fields a local vs. remote source shows (and adds the
 * server-side "Browse…" chooser to a local path). Any other key already set on a
 * source is still rendered so nothing is hidden.
 *
 * Pure-controlled: every edit calls `onChange` with the next sources array, which
 * AdminPage merges into the one canonical `config` object.
 */

export type SourceEntry = Record<string, unknown>;

interface SourcesEditorProps {
  sources: SourceEntry[];
  onChange: (next: SourceEntry[]) => void;
  /** The config JSON Schema (drives field labels/types/help). */
  schema: ConfigSchema | null;
  /** Per-source-index validation messages from a rejected PUT (422). */
  errorsByIndex: Record<number, string[]>;
  disabled?: boolean;
  /** Open the server-side file chooser for local row `i`, seeded from its current
   * URL. Provided only in local mode (biopb/biopb#244). */
  onBrowse?: (i: number) => void;
}

// Curated field order per source kind; every other *set* scalar key on a source
// (source_id, cloud, dataset, path, …) is appended so nothing is hidden.
const LOCAL_KEYS = ["url", "type", "monitor"];
const REMOTE_KEYS = ["url", "alias", "credentials_profile", "monitor"];

function str(v: unknown): string {
  return typeof v === "string" ? v : v == null ? "" : String(v);
}

export function SourcesEditor({
  sources,
  onChange,
  schema,
  errorsByIndex,
  disabled,
  onBrowse,
}: SourcesEditorProps) {
  const [addOpen, setAddOpen] = useState(false);
  const sourceProps = sourceItemProperties(schema);

  function update(i: number, key: string, value: unknown) {
    const next = sources.map((s, idx) => {
      if (idx !== i) return s;
      const copy = { ...s };
      if (value === undefined) delete copy[key];
      else copy[key] = value;
      return copy;
    });
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

  // A remote (tensor-server proxy) source is identified by its grpc:// URL or an
  // explicit tensor-server type -- NOT by `alias`, which is also valid on a local
  // source (there it is the catalog tree root), so keying on it mislabels a local
  // dir as remote.
  const isRemote = (s: SourceEntry) =>
    str(s.url).startsWith("grpc://") || str(s.type) === "tensor-server";

  // Canonical per-entry display title, numbered within its kind.
  let localN = 0;
  let remoteN = 0;
  const titles = sources.map((s) =>
    isRemote(s) ? `Remote Source ${++remoteN}` : `Local Source ${++localN}`,
  );

  function keysFor(s: SourceEntry, remote: boolean): string[] {
    const base = remote ? REMOTE_KEYS : LOCAL_KEYS;
    const extra = Object.keys(s).filter(
      (k) =>
        !k.startsWith("_") &&
        !base.includes(k) &&
        !Array.isArray(s[k]) &&
        s[k] != null &&
        s[k] !== "",
    );
    return [...base, ...extra];
  }

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
                    {keysFor(s, remote).map((key) => {
                      const prop: SchemaProp = sourceProps[key] ?? { type: "string" };
                      const browseable = key === "url" && !remote && !!onBrowse;
                      return (
                        <SchemaField
                          key={key}
                          fieldKey={key}
                          prop={prop}
                          value={s[key]}
                          errs={[]}
                          disabled={disabled}
                          unsetLabel={key === "type" ? "auto-detect" : undefined}
                          placeholder={
                            key === "url"
                              ? remote
                                ? "grpc://lab-nas:8815"
                                : "/data/microscopy"
                              : undefined
                          }
                          append={
                            browseable ? (
                              <button
                                type="button"
                                className="icon-btn"
                                disabled={disabled}
                                onClick={() => onBrowse!(i)}
                              >
                                Browse…
                              </button>
                            ) : undefined
                          }
                          onChange={(v) => update(i, key, v)}
                        />
                      );
                    })}
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
