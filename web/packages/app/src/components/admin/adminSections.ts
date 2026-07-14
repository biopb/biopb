/**
 * Fixed, curated navigation model for the admin settings page.
 *
 * The old admin surface auto-generated collapsible sections straight from the
 * config JSON Schema — flexible, but the user saw raw section/field names with
 * no ordering or explanation. This replaces that with a **fixed left-nav
 * layout** (chrome://settings / GitHub-settings style): a hand-ordered flat list
 * of sections, each with a human title + one-line description.
 *
 * The schema stays the source of truth for *field* types, defaults, and
 * validation (see config-schema.ts) — this module only fixes the *presentation*:
 * which sections exist, their order, their prose, and which of a section's schema
 * fields are "common" (shown directly) vs "advanced" (behind a disclosure).
 * A schema field not listed as common still renders — it just lands under the
 * Advanced disclosure — so a new server field is never silently dropped.
 */

/** How a nav item's content panel is rendered. */
export type AdminNavKind = "sources" | "credentials" | "fields" | "raw";

export interface AdminNavItem {
  /** Stable id (also the URL-ish key). For `fields` items this equals the config
   * section key, so an error `path[0]` maps straight onto the owning nav item. */
  id: string;
  /** Nav label and content-panel heading. */
  label: string;
  /** One-line panel subtitle explaining what the section controls. */
  description: string;
  kind: AdminNavKind;
  /** Config section key for `kind: "fields"` (server / cache / …). */
  section?: string;
  /** Ordered "common" field keys shown directly; the rest of the section's
   * schema fields fall under the Advanced disclosure. */
  commonFields?: string[];
}

/**
 * The flat, ordered nav. Sources first (the headline), then credentials, the
 * performance knobs, the server/runtime settings, and finally the raw-JSON
 * escape hatch.
 */
export const ADMIN_NAV: AdminNavItem[] = [
  {
    id: "sources",
    label: "Data Sources",
    description:
      "Folders, files, and remote servers this data plane catalogs and serves. " +
      "Add a local path to index everything under it, or a remote tensor server " +
      "to federate its catalog.",
    kind: "sources",
  },
  {
    id: "credentials",
    label: "Credentials",
    description:
      "Access keys for remote object storage (S3 / GCS / Azure). A source points " +
      "at a profile by name via its credentials profile.",
    kind: "credentials",
  },
  {
    id: "cache",
    label: "Cache",
    description:
      "In-memory and on-disk caching of decoded image chunks. Larger limits keep " +
      "more data warm at the cost of RAM and disk.",
    kind: "fields",
    section: "cache",
    // On-disk keys (config-schema maps the dataclass's memory_max_bytes →
    // cache.max_bytes); commonFields must match the schema, not the dataclass.
    commonFields: ["backend", "max_bytes", "file_max_total_gb"],
  },
  {
    id: "pyramid",
    label: "Pyramid",
    description:
      "How the server builds reduced-resolution levels so large images render " +
      "quickly when zoomed out.",
    kind: "fields",
    section: "pyramid",
    commonFields: ["reduction_method", "threshold"],
  },
  {
    id: "precache",
    label: "Precache",
    description:
      "Background warming of the coarsest pyramid level while the server is idle, " +
      "so the first view of a source is fast.",
    kind: "fields",
    section: "precache",
    commonFields: ["enabled"],
  },
  {
    id: "server",
    label: "Server",
    description:
      "Network binding, logging, and how the server watches source folders for " +
      "changes.",
    kind: "fields",
    section: "server",
    commonFields: ["host", "port", "log_level", "writable"],
  },
  {
    id: "metadata_db",
    label: "Metadata DB",
    description: "Limits on catalog queries served from the metadata database.",
    kind: "fields",
    section: "metadata_db",
    commonFields: ["max_query_results", "query_timeout_ms"],
  },
  {
    id: "raw",
    label: "Raw JSON",
    description:
      "Edit the entire configuration as raw JSON — for fields not surfaced in the " +
      "sections above, or bulk edits. Malformed JSON is rejected before it touches " +
      "the config.",
    kind: "raw",
  },
];

/** The default (first) nav section id. */
export const DEFAULT_ADMIN_NAV_ID = ADMIN_NAV[0]!.id;

/** The nav item for an id, falling back to the first section for an unknown id. */
export function navItemById(id: string): AdminNavItem {
  return ADMIN_NAV.find((n) => n.id === id) ?? ADMIN_NAV[0]!;
}

/** Nav ids that correspond to an error-addressable config section (`path[0]`). */
const SECTION_NAV_IDS = new Set(
  ADMIN_NAV.filter((n) => n.kind !== "raw").map((n) => n.id),
);

/**
 * The nav item an error belongs to, from its config path. Errors are addressed
 * by `path[0]` (the config section: `sources` / `credentials` / `server` / …),
 * which is exactly a nav id here, so the mapping is a lookup. Returns null for a
 * path that doesn't target a nav section (never surfaced in a section badge).
 */
export function navIdForErrorPath(
  path: (string | number)[],
): string | null {
  const head = path.length ? String(path[0]) : "";
  return SECTION_NAV_IDS.has(head) ? head : null;
}

const UNIT_WORDS: Record<string, string> = {
  mb: "MB",
  gb: "GB",
  kb: "KB",
  ms: "ms",
  db: "DB",
  id: "ID",
  url: "URL",
  dir: "dir",
};

/**
 * A human label for a snake_case config key: words title-cased, with a few
 * unit/acronym tokens (MB/GB/ms/DB/ID/URL) fixed up. The raw key is still shown
 * beside it in the UI, so this only needs to read well, not round-trip.
 */
export function humanizeKey(key: string): string {
  const words = key.split("_").filter(Boolean);
  return words
    .map((w, i) => {
      const unit = UNIT_WORDS[w];
      if (unit) return unit;
      if (i === 0) return w.charAt(0).toUpperCase() + w.slice(1);
      return w;
    })
    .join(" ");
}
