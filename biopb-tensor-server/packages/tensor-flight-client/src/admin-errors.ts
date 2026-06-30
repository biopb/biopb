/**
 * Group the flat `422` validation errors from `PUT /api/config` into per-source
 * messages + a general summary, for the admin form's inline display.
 *
 * The backend stringifies every JSON-path element (`[str(x) for x in
 * e.absolute_path]`), so a source index arrives as the *string* `"0"`, not the
 * number `0` — this coerces it back rather than relying on `typeof === "number"`
 * (which would never match the wire form). See biopb/biopb#237.
 */

import type { AdminConfigError } from "./types.js";

export interface SplitConfigErrors {
  /** Per-source-index formatted messages (key is the numeric `sources` index). */
  byIndex: Record<number, string[]>;
  /** Everything not attributable to a single source row. */
  general: string[];
}

/** Return true if `v` is an integer index in string or number form ("0" / 0). */
function asIndex(v: unknown): number | null {
  const n = Number(v);
  return Number.isInteger(n) && String(n) === String(v) ? n : null;
}

export function splitConfigErrors(errors: AdminConfigError[]): SplitConfigErrors {
  const byIndex: Record<number, string[]> = {};
  const general: string[] = [];
  for (const e of errors) {
    const idx = e.path[0] === "sources" ? asIndex(e.path[1]) : null;
    if (idx !== null) {
      (byIndex[idx] ??= []).push(`${e.path.slice(2).join(".") || "source"}: ${e.message}`);
    } else {
      const where = e.path.length ? e.path.join(".") + ": " : "";
      general.push(where + e.message);
    }
  }
  return { byIndex, general };
}
