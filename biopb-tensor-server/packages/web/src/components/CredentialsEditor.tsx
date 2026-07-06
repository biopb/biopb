import { useEffect, useState } from "react";
import {
  credentialProfileProperties,
  enumValues,
  fieldErrors,
  isSecretProfileKey,
  REDACTED_SENTINEL,
  type ConfigError,
  type ConfigSchema,
} from "@biopb/tensor-flight-client";

/**
 * Structured editor for the `credentials` block (remote-storage profiles that
 * `credentials_profile` on a source points at). Kept separate from the flat
 * AdvancedSections because credentials are a nested *array of profiles* with
 * masked secrets.
 *
 * Secret fields (`key`/`secret`/`token`) arrive from `GET /api/config` as the
 * REDACTED_SENTINEL and are rendered as password inputs seeded with that mask;
 * `PUT /api/config` restores the stored secret only when the sentinel round-trips
 * verbatim, so an untouched secret is preserved and typing over it replaces it.
 * The whole section is collapsible and starts collapsed (advanced surface).
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
  const props = credentialProfileProperties(schema);
  const profiles = profilesOf(config);
  // Section carries an error → force open (mirrors AdvancedSections).
  const hasError = errors.some((e) => e.path[0] === "credentials");
  const [userOpen, setUserOpen] = useState(false);
  // Latch open on error so fixing it doesn't collapse the section mid-edit.
  useEffect(() => {
    if (hasError) setUserOpen(true);
  }, [hasError]);
  if (!schema) return null;
  const open = userOpen || hasError;

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
    setUserOpen(true);
    commit([...profiles, { name: "", storage_type: "s3" }]);
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
    <div className={`adv-section${hasError ? " has-error" : ""}`}>
      <button
        type="button"
        className="adv-section-header"
        aria-expanded={open}
        onClick={() => setUserOpen(!open)}
      >
        <span className="adv-caret">{open ? "▾" : "▸"}</span>
        <span className="adv-section-title">Credentials</span>
        <span className="adv-section-hint">
          {profiles.length} profile{profiles.length === 1 ? "" : "s"} for remote
          storage
        </span>
        {hasError && <span className="adv-section-errdot" title="Has errors" />}
      </button>

      {open && (
        <div className="adv-section-body creds-body">
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

          {profiles.length === 0 ? (
            <p className="adv-field-help">
              No credential profiles. Add one to give a remote source (S3 / GCS /
              Azure) access keys.
            </p>
          ) : (
            <ul className="creds-rows">
              {profiles.map((p, i) => {
                // The server restores masked secrets by matching profile *name*
                // (restore_redacted_secrets), so renaming a profile that still
                // carries a masked secret would silently drop it. Lock the name
                // until the secrets are re-entered (or the profile has none).
                const hasMaskedSecret = orderedKeys.some(
                  (k) => isSecretProfileKey(k) && str(p[k]) === REDACTED_SENTINEL,
                );
                return (
                <li key={i} className="creds-row">
                  <div className="creds-row-grid">
                    {orderedKeys.map((key) => {
                      const secret = isSecretProfileKey(key);
                      const locked = key === "name" && hasMaskedSecret;
                      const errs = fieldErrors(errors, [
                        "credentials",
                        "profiles",
                        i,
                        key,
                      ]);
                      const invalid = errs.length > 0;
                      const hardEnum = enumValues(props[key]);
                      const value = str(p[key]);
                      return (
                        <label key={key} className="adv-field-label">
                          <span className="adv-field-name">
                            <code>{key}</code>
                            {secret && <span className="secret-tag">secret</span>}
                          </span>
                          {hardEnum ? (
                            <select
                              value={value}
                              disabled={disabled}
                              className={invalid ? "invalid" : undefined}
                              onChange={(e) =>
                                updateProfile(
                                  i,
                                  key,
                                  e.target.value === "" ? undefined : e.target.value,
                                )
                              }
                            >
                              <option value="">(unset)</option>
                              {hardEnum
                                .filter((v): v is string => typeof v === "string")
                                .map((v) => (
                                  <option key={v} value={v}>
                                    {v}
                                  </option>
                                ))}
                            </select>
                          ) : (
                            <input
                              type={secret ? "password" : "text"}
                              value={value}
                              disabled={disabled || locked}
                              autoComplete={secret ? "new-password" : "off"}
                              placeholder={
                                secret ? "(stored — leave to keep)" : undefined
                              }
                              className={invalid ? "invalid" : undefined}
                              onChange={(e) =>
                                updateProfile(
                                  i,
                                  key,
                                  e.target.value === "" ? undefined : e.target.value,
                                )
                              }
                            />
                          )}
                          {locked && (
                            <span className="adv-field-help">
                              Rename disabled — stored secrets are keyed by name.
                              Clear a secret field to rename.
                            </span>
                          )}
                          {secret && value === REDACTED_SENTINEL && (
                            <span className="adv-field-help">
                              Stored secret hidden — leave as-is to keep it.
                            </span>
                          )}
                          {errs.map((m, k) => (
                            <span key={k} className="adv-field-error">
                              {m}
                            </span>
                          ))}
                        </label>
                      );
                    })}
                  </div>
                  <button
                    type="button"
                    className="source-row-remove"
                    title="Remove profile"
                    disabled={disabled}
                    onClick={() => removeProfile(i)}
                  >
                    ✕
                  </button>
                </li>
                );
              })}
            </ul>
          )}

          <button
            type="button"
            className="icon-btn"
            disabled={disabled}
            onClick={addProfile}
          >
            + Add credential profile
          </button>
        </div>
      )}
    </div>
  );
}
