import { useState } from "react";
import {
  credentialProfileProperties,
  fieldErrors,
  isSecretProfileKey,
  REDACTED_SENTINEL,
  type ConfigError,
  type ConfigSchema,
} from "@biopb/tensor-flight-client";
import { SchemaField } from "./admin/SchemaField";

/**
 * Structured editor for the `credentials` block (remote-storage profiles that
 * `credentials_profile` on a source points at). Its own settings panel — a
 * nested *array of profiles* with masked secrets, rendered directly (the panel
 * shell owns the heading), no collapse.
 *
 * Secret fields (`key`/`secret`/`token`) arrive from `GET /api/config` as the
 * REDACTED_SENTINEL and are rendered as password inputs seeded with that mask;
 * `PUT /api/config` restores the stored secret only when the sentinel round-trips
 * verbatim, so an untouched secret is preserved and typing over it replaces it.
 *
 * A profile's `name` is set once, in the add-bar (typed before "+ Add profile"),
 * and then shown as the block's static title — there is no rename field. This
 * mirrors the Sources editor's fixed titles and sidesteps the masked-secret
 * rename hazard (the server keys stored secrets by profile name, so a rename
 * would silently drop them).
 */

type Config = Record<string, unknown>;
type Profile = Record<string, unknown>;

interface CredentialsEditorProps {
  config: Config;
  schema: ConfigSchema | null;
  errors: ConfigError[];
  disabled?: boolean;
  onChange: (next: Config) => void;
}

// Field render order within a profile row (schema is the source of truth for
// which keys exist; this only fixes a sensible display order).
const FIELD_ORDER = [
  "name",
  "storage_type",
  "region",
  "endpoint_url",
  "key",
  "secret",
  "token",
];

function str(v: unknown): string {
  return typeof v === "string" ? v : v == null ? "" : String(v);
}

function credentials(config: Config): Config {
  const c = config.credentials;
  return c && typeof c === "object" && !Array.isArray(c) ? (c as Config) : {};
}

function profilesOf(config: Config): Profile[] {
  const p = credentials(config).profiles;
  return Array.isArray(p) ? (p as Profile[]) : [];
}

export function CredentialsEditor({
  config,
  schema,
  errors,
  disabled,
  onChange,
}: CredentialsEditorProps) {
  // Name for the next profile, typed in the add-bar; a profile's name is fixed at
  // creation (the block title is a static display), so there is no rename path —
  // which also sidesteps the masked-secret rename hazard.
  const [newName, setNewName] = useState("");
  const props = credentialProfileProperties(schema);
  const profiles = profilesOf(config);
  if (!schema) return null;

  const orderedKeys = [
    ...FIELD_ORDER.filter((k) => k in props),
    ...Object.keys(props).filter((k) => !FIELD_ORDER.includes(k)),
  ];

  function commit(nextProfiles: Profile[], defaultProfile?: string) {
    const creds: Config = { ...credentials(config) };
    creds.profiles = nextProfiles;
    if (defaultProfile !== undefined) {
      if (defaultProfile === "") delete creds.default_profile;
      else creds.default_profile = defaultProfile;
    }
    onChange({ ...config, credentials: creds });
  }

  function updateProfile(i: number, key: string, value: unknown) {
    const next = profiles.map((p, idx) => {
      if (idx !== i) return p;
      const copy = { ...p };
      if (value === undefined) delete copy[key];
      else copy[key] = value;
      return copy;
    });
    // Renaming the profile that `default_profile` points at must follow the
    // rename, or the reference dangles (a blanked name clears it).
    let nextDefault: string | undefined;
    if (key === "name") {
      const oldName = str(profiles[i]?.name);
      if (oldName && oldName === str(credentials(config).default_profile)) {
        nextDefault = value == null ? "" : str(value);
      }
    }
    commit(next, nextDefault);
  }

  function addProfile() {
    const name = newName.trim();
    if (!name) return;
    commit([...profiles, { name, storage_type: "s3" }]);
    setNewName("");
  }

  function removeProfile(i: number) {
    // If the removed profile was the default, clear the dangling reference.
    const removedName = str(profiles[i]?.name);
    const clearsDefault = removedName && removedName === str(credentials(config).default_profile);
    commit(
      profiles.filter((_, idx) => idx !== i),
      clearsDefault ? "" : undefined,
    );
  }

  const currentDefault = str(credentials(config).default_profile);

  return (
    <div className="creds-panel">
      <div className="creds-body">
          <div className="creds-default">
            <label className="adv-field-label">
              <span className="adv-field-name">
                <code>default_profile</code>
              </span>
              <select
                value={currentDefault}
                disabled={disabled}
                onChange={(e) => commit(profiles, e.target.value)}
              >
                <option value="">(none)</option>
                {profiles
                  .map((p) => str(p.name))
                  .filter(Boolean)
                  .map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
              </select>
            </label>
          </div>

          <div className="creds-add-bar">
            <input
              className="creds-add-name"
              type="text"
              value={newName}
              placeholder="profile name"
              disabled={disabled}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  addProfile();
                }
              }}
            />
            <button
              type="button"
              className="icon-btn"
              disabled={disabled || !newName.trim()}
              onClick={addProfile}
            >
              + Add profile
            </button>
          </div>

          {profiles.length === 0 ? (
            <p className="adv-field-help">
              No credential profiles. Add one to give a remote source (S3 / GCS /
              Azure) access keys.
            </p>
          ) : (
            <ul className="creds-rows">
              {profiles.map((p, i) => {
                return (
                <li key={i} className="entry-row">
                  <div className="entry-head">
                    <span className="entry-title">
                      {str(p.name) || "Unnamed profile"}
                    </span>
                    <button
                      type="button"
                      className="entry-delete"
                      disabled={disabled}
                      onClick={() => removeProfile(i)}
                    >
                      Delete
                    </button>
                  </div>
                  <div className="entry-fields">
                  <div className="creds-row-grid">
                    {orderedKeys
                      .filter((k) => k !== "name")
                      .map((key) => {
                        const secret = isSecretProfileKey(props[key]);
                        const masked = secret && str(p[key]) === REDACTED_SENTINEL;
                        return (
                          <SchemaField
                            key={key}
                            fieldKey={key}
                            prop={props[key] ?? { type: "string" }}
                            value={p[key]}
                            errs={fieldErrors(errors, [
                              "credentials",
                              "profiles",
                              i,
                              key,
                            ])}
                            disabled={disabled}
                            secret={secret}
                            placeholder={
                              secret ? "(stored — leave to keep)" : undefined
                            }
                            note={
                              masked
                                ? "Stored secret hidden — leave as-is to keep it."
                                : undefined
                            }
                            onChange={(v) => updateProfile(i, key, v)}
                          />
                        );
                      })}
                  </div>
                  </div>
                </li>
                );
              })}
            </ul>
          )}
      </div>
    </div>
  );
}
