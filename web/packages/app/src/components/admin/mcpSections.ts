/**
 * Fixed navigation model for the **biopb-mcp** settings page — the sibling of
 * adminSections.ts, but for the agent client's own config
 * (`~/.config/biopb/mcp-config.json`, served by the control at
 * `GET/PUT /api/mcp_config`).
 *
 * Nothing merges with the tensor config: this is a separate file + schema. Only
 * the *machinery* is shared — the same schema-driven `SectionFields` / `SchemaField`
 * render these sections, and the schema stays the source of truth for field
 * types/defaults/help. This module fixes the presentation: which sections exist,
 * their order, and their prose.
 */

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

/**
 * Flat, ordered nav, grouped by concern in reading order: the data-plane knobs
 * the headless kernel uses, the compute-plane knobs, then the MCP server runtime.
 * The `id` equals the config section key. The demo-widget sections
 * (`widget` / `detection` / `grid`, the experimental image_processing/ widgets)
 * are deliberately omitted from the nav — they stay in the config and remain
 * editable via the Raw JSON panel, just not surfaced as first-class settings.
 */
export const MCP_NAV: McpNavItem[] = [
  {
    id: "tensor_browser",
    label: "Data Plane",
    description: "The Arrow Flight tensor-server the kernel and browser widget read image data from.",
    kind: "fields",
    section: "tensor_browser",
  },
  {
    id: "pyramid",
    label: "Pyramid",
    description: "How multiscale pyramids are built for large tensors added to the viewer.",
    kind: "fields",
    section: "pyramid",
  },
  {
    id: "services",
    label: "Services",
    description: "ProcessImage algorithm servers wired into the kernel as `ops`, and the skills catalog.",
    kind: "fields",
    section: "services",
  },
  {
    id: "timeout",
    label: "Timeouts",
    description: "Per-call gRPC timeouts for the compute plane.",
    kind: "fields",
    section: "timeout",
  },
  {
    id: "grpc",
    label: "gRPC",
    description: "gRPC channel limits for the compute plane.",
    kind: "fields",
    section: "grpc",
  },
  {
    id: "memory",
    label: "Memory",
    description: "Chunk-size guardrails for eager transfers.",
    kind: "fields",
    section: "memory",
  },
  {
    id: "transport",
    label: "Transport",
    description: "The MCP server's front-end transport (stdio / http) and its network guards.",
    kind: "fields",
    section: "transport",
  },
  {
    id: "kernel",
    label: "Kernel",
    description: "The child Jupyter kernel that runs agent code: bring-up, timeouts, and the orphan watchdog.",
    kind: "fields",
    section: "kernel",
  },
  {
    id: "dask",
    label: "Dask",
    description: "The dask scheduler / cluster the kernel computes on.",
    kind: "fields",
    section: "dask",
  },
  {
    id: "tensor",
    label: "Catalog Watcher",
    description: "The background source-catalog watcher's backoff bounds.",
    kind: "fields",
    section: "tensor",
  },
  {
    id: "viewer",
    label: "Viewer",
    description: "How the napari viewer fetches image slices.",
    kind: "fields",
    section: "viewer",
  },
  {
    id: "observe",
    label: "Observe",
    description: "The loopback web UI for watching execute_code job history (http transport only).",
    kind: "fields",
    section: "observe",
  },
  {
    id: "update",
    label: "Updates",
    description: "The kernel-start auto-updater that offers to re-run the installer on a newer release.",
    kind: "fields",
    section: "update",
  },
  {
    id: "raw",
    label: "Raw JSON",
    description:
      "Edit the entire biopb-mcp configuration as raw JSON — for bulk edits or " +
      "fields not surfaced above. Malformed JSON is rejected before it is applied.",
    kind: "raw",
  },
];

export const MCP_DEFAULT_NAV_ID = MCP_NAV[0]!.id;

export function mcpNavItemById(id: string): McpNavItem {
  return MCP_NAV.find((n) => n.id === id) ?? MCP_NAV[0]!;
}

const MCP_SECTION_NAV_IDS = new Set(
  MCP_NAV.filter((n) => n.kind !== "raw").map((n) => n.id),
);

/** The nav item an error belongs to, from its config path (`path[0]` == section). */
export function mcpNavIdForErrorPath(path: (string | number)[]): string | null {
  const head = path.length ? String(path[0]) : "";
  return MCP_SECTION_NAV_IDS.has(head) ? head : null;
}
