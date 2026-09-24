/**
 * Navigation for the **biopb-mcp** settings page — the sibling of
 * adminSections.ts, but for the agent client's own config
 * (`~/.config/biopb/mcp-config.json`, served by the control at
 * `GET/PUT /api/mcp_config`).
 *
 * The nav is read off the served schema: each config section with a `title` is
 * an item, in schema order, with the section's `description` as its prose
 * (both set on `McpConfig` in biopb_mcp/_config.py). A section without a title
 * stays editable through the Raw JSON panel only.
 */

import type { ConfigSchema } from "@biopb/tensor-flight-client";

export interface McpNavItem {
  /** Stable id; for a config section this equals the section key so an error
   * `path[0]` maps straight onto the owning nav item. */
  id: string;
  label: string;
  description: string;
  /** A config section, or "raw" for the raw-JSON escape hatch. */
  kind: "fields" | "raw";
  section?: string;
}

const RAW_ITEM: McpNavItem = {
  id: "raw",
  label: "Raw JSON",
  description:
    "Edit the entire biopb-mcp configuration as raw JSON — for bulk edits or " +
    "fields not surfaced above. Malformed JSON is rejected before it is applied.",
  kind: "raw",
};

/** The nav for *schema*: its titled sections, then Raw JSON. */
export function mcpNav(schema: ConfigSchema | null | undefined): McpNavItem[] {
  const sections = Object.entries(schema?.properties ?? {})
    .filter(([, spec]) => spec.type === "object" && spec.title)
    .map(([key, spec]) => ({
      id: key,
      label: spec.title!,
      description: spec.description ?? "",
      kind: "fields" as const,
      section: key,
    }));
  return [...sections, RAW_ITEM];
}

/** The item *id* names, else the first. */
export function mcpNavItemById(nav: McpNavItem[], id: string): McpNavItem {
  return nav.find((n) => n.id === id) ?? nav[0]!;
}

/** The nav item an error belongs to, from its config path (`path[0]` == section). */
export function mcpNavIdForErrorPath(
  nav: McpNavItem[],
  path: (string | number)[],
): string | null {
  const head = path.length ? String(path[0]) : "";
  return nav.some((n) => n.kind === "fields" && n.id === head) ? head : null;
}
