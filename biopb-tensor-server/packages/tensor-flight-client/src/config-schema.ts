/**
 * Client-side helpers over the tensor-server config JSON Schema served by
 * `GET /api/config` (`build_config_schema()` in config_schema.py).
 *
 * The admin page uses these to (a) render structured "advanced" sections from
 * the schema instead of a raw-JSON blob, and (b) validate fields *inline before
 * Save*, mirroring what the server enforces on `PUT /api/config`.
 *
 * Validation intentionally mirrors the server's leniency: the published schema
 * keeps `additionalProperties: true` and case-insensitive enums (`log_level`,
 * `reduction_method`) emit **no** `enum` (only a `description`), so this checker
 * flags exactly what the server's `Draft202012Validator` would — required keys,
 * hard (case-sensitive) enums, numeric bounds, and gross type mismatches — and
 * leaves everything else to the server. The accepted set for the soft enums is
 * surfaced as helper text (from the property `description`), not as a hard error.
 */

import type { AdminConfigError } from "./types.js";

/** A minimal view of a JSON Schema (Draft 2020-12) node we care about. */
export interface SchemaProp {
  type?: string | string[];
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  description?: string;
  deprecated?: boolean;
  required?: string[];
  properties?: Record<string, SchemaProp>;
  items?: SchemaProp;
  additionalProperties?: boolean;
}

/** The whole config schema (top-level object with per-section properties). */
export type ConfigSchema = SchemaProp;

/** A field-level validation error, shaped like the server's 422 entries. */
export type ConfigError = AdminConfigError;

/**
 * Marker the server substitutes for each stored credential secret in
 * `GET /api/config` (`REDACTED_SENTINEL` in config.py). `PUT /api/config`
 * restores the real value only when this exact string round-trips back, so a
 * credentials editor must resubmit an untouched secret verbatim — dropping the
 * field would silently delete the stored secret.
 */
export const REDACTED_SENTINEL = "***REDACTED***";

/** Secret profile keys that arrive masked and must round-trip to be preserved. */
export const SECRET_PROFILE_KEYS = ["key", "secret", "token"] as const;

/** The object-typed config sections we render structured editors for, in order. */
export const ADVANCED_SECTIONS = [
  "server",
  "compute",
  "cache",
  "pyramid",
  "precache",
  "metadata_db",
] as const;

/** The `properties` map for one config section (e.g. `server`), or `{}`. */
export function sectionProperties(
  schema: ConfigSchema | null | undefined,
  section: string,
): Record<string, SchemaProp> {
  const spec = schema?.properties?.[section];
  if (!spec || spec.type !== "object") return {};
  return spec.properties ?? {};
}

/** The per-item `properties` map for the `sources` array, or `{}`. */
export function sourceItemProperties(
  schema: ConfigSchema | null | undefined,
): Record<string, SchemaProp> {
  return schema?.properties?.sources?.items?.properties ?? {};
}

/** The keys `[[sources]]` marks required (`["url"]`). */
export function sourceRequiredKeys(
  schema: ConfigSchema | null | undefined,
): string[] {
  return schema?.properties?.sources?.items?.required ?? [];
}

/** The per-profile `properties` map for `credentials.profiles`, or `{}`. */
export function credentialProfileProperties(
  schema: ConfigSchema | null | undefined,
): Record<string, SchemaProp> {
  return (
    schema?.properties?.credentials?.properties?.profiles?.items?.properties ?? {}
  );
}

/** The keys a `credentials.profiles` item marks required (`["name"]`). */
export function credentialProfileRequired(
  schema: ConfigSchema | null | undefined,
): string[] {
  return schema?.properties?.credentials?.properties?.profiles?.items?.required ?? [];
}

export function isSecretProfileKey(key: string): boolean {
  return (SECRET_PROFILE_KEYS as readonly string[]).includes(key);
}

export function isDeprecated(prop: SchemaProp | undefined): boolean {
  return prop?.deprecated === true;
}

/** The hard (case-sensitive) enum values for a field, or `null` if none. */
export function enumValues(prop: SchemaProp | undefined): unknown[] | null {
  return Array.isArray(prop?.enum) ? prop!.enum : null;
}

/** The primitive JSON type used to pick a form control (first non-null type). */
export function primaryType(prop: SchemaProp | undefined): string {
  const t = prop?.type;
  if (Array.isArray(t)) return t.find((x) => x !== "null") ?? t[0] ?? "string";
  return t ?? "string";
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

function fmt(v: unknown): string {
  return v === null ? "null" : String(v);
}

/** JSON-schema type names a JS value satisfies (integer implies number too). */
function jsonTypesOf(value: unknown): string[] {
  if (value === null) return ["null"];
  if (typeof value === "boolean") return ["boolean"];
  if (typeof value === "number") {
    return Number.isInteger(value) ? ["integer", "number"] : ["number"];
  }
  if (typeof value === "string") return ["string"];
  if (Array.isArray(value)) return ["array"];
  if (typeof value === "object") return ["object"];
  return [];
}

function checkValue(
  path: (string | number)[],
  value: unknown,
  prop: SchemaProp,
  out: ConfigError[],
): void {
  // Hard enum (case-sensitive) first — a bad enum is the clearest failure.
  const hard = enumValues(prop);
  if (hard) {
    if (!hard.some((e) => e === value)) {
      out.push({ path, message: `must be one of: ${hard.map(fmt).join(", ")}` });
    }
    return;
  }
  // Type: only flag a gross mismatch (the schema's declared types).
  const allowed = Array.isArray(prop.type)
    ? prop.type
    : prop.type
      ? [prop.type]
      : [];
  if (allowed.length) {
    const got = jsonTypesOf(value);
    if (!got.some((t) => allowed.includes(t))) {
      out.push({ path, message: `expected ${allowed.join(" or ")}` });
      return;
    }
  }
  // Numeric bounds.
  if (typeof value === "number") {
    if (prop.minimum != null && value < prop.minimum) {
      out.push({ path, message: `must be ≥ ${prop.minimum}` });
    }
    if (prop.maximum != null && value > prop.maximum) {
      out.push({ path, message: `must be ≤ ${prop.maximum}` });
    }
  }
}

function isBlank(value: unknown): boolean {
  return value == null || value === "";
}

/**
 * Validate a config dict against the schema, returning field-level errors
 * (same shape as the server's 422 body) so the admin form can render them
 * inline before Save.
 */
export function validateConfig(
  config: Record<string, unknown> | null | undefined,
  schema: ConfigSchema | null | undefined,
): ConfigError[] {
  const out: ConfigError[] = [];
  if (!config || !schema?.properties) return out;

  for (const [section, spec] of Object.entries(schema.properties)) {
    if (section === "sources") continue; // handled below
    if (section === "credentials") continue; // secrets masked; skip deep checks
    if (spec.type !== "object" || !spec.properties) continue;
    const val = config[section];
    if (val == null || typeof val !== "object" || Array.isArray(val)) continue;
    const obj = val as Record<string, unknown>;
    for (const [key, fieldSpec] of Object.entries(spec.properties)) {
      if (!(key in obj) || isBlank(obj[key])) continue;
      checkValue([section, key], obj[key], fieldSpec, out);
    }
  }

  // Sources: required keys + per-field checks.
  const sources = config.sources;
  if (Array.isArray(sources)) {
    const itemProps = sourceItemProperties(schema);
    const required = sourceRequiredKeys(schema);
    sources.forEach((src, i) => {
      if (src == null || typeof src !== "object" || Array.isArray(src)) return;
      const obj = src as Record<string, unknown>;
      for (const req of required) {
        // The deprecated `path` is a back-compat alias for `url` (the server
        // reads `url or path`), so a legacy `path`-only source is valid and must
        // not be flagged — otherwise Save is greyed out on load for a config the
        // server would happily accept.
        if (req === "url" && !isBlank(obj.path)) continue;
        if (isBlank(obj[req])) {
          out.push({ path: ["sources", i, req], message: `${req} is required` });
        }
      }
      for (const [key, fieldSpec] of Object.entries(itemProps)) {
        if (required.includes(key)) continue; // covered above
        if (!(key in obj) || isBlank(obj[key])) continue;
        checkValue(["sources", i, key], obj[key], fieldSpec, out);
      }
    });
  }

  // Credentials: only the non-secret required key (`name`) is validated; secret
  // values arrive masked (REDACTED_SENTINEL) so they are never checked here.
  const creds = config.credentials as { profiles?: unknown } | undefined;
  if (creds && typeof creds === "object" && Array.isArray(creds.profiles)) {
    const required = credentialProfileRequired(schema);
    creds.profiles.forEach((p, i) => {
      if (p == null || typeof p !== "object" || Array.isArray(p)) return;
      const obj = p as Record<string, unknown>;
      for (const req of required) {
        if (isBlank(obj[req])) {
          out.push({
            path: ["credentials", "profiles", i, req],
            message: `${req} is required`,
          });
        }
      }
    });
  }

  return out;
}

/**
 * Migrate deprecated keys the *runtime* parser accepts but the published JSON
 * Schema does not, so a save the server would 422 is normalized first. Today
 * that is the source `path` alias: the parser reads `url or path`, but the
 * schema marks source items `required: ["url"]`, so a `path`-only source must be
 * rewritten to `url` before `PUT /api/config`. Returns the same object when
 * nothing needed changing (referentially stable).
 */
export function normalizeConfigForSave(
  config: Record<string, unknown> | null | undefined,
): Record<string, unknown> | null | undefined {
  if (!config) return config;
  const sources = config.sources;
  if (!Array.isArray(sources)) return config;
  let touched = false;
  const nextSources = sources.map((s) => {
    if (s == null || typeof s !== "object" || Array.isArray(s)) return s;
    const obj = s as Record<string, unknown>;
    if (!("path" in obj)) return s;
    touched = true;
    const next = { ...obj };
    if (isBlank(next.url) && !isBlank(next.path)) next.url = next.path;
    delete next.path; // resolved into `url`; the deprecated alias is dropped
    return next;
  });
  return touched ? { ...config, sources: nextSources } : config;
}

/** Messages whose path exactly equals `path` (string-compared element-wise). */
export function fieldErrors(
  errors: ConfigError[],
  path: (string | number)[],
): string[] {
  return errors
    .filter(
      (e) =>
        e.path.length === path.length &&
        e.path.every((p, i) => String(p) === String(path[i])),
    )
    .map((e) => e.message);
}
