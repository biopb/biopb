import type { DataSourceDescriptor } from "@biopb/tensor-flight-client";

/** The `sources` columns a descriptor is made of, as a SELECT list. */
export const SOURCE_COLUMNS = "source_id, source_url, source_type, is_resolved, tensors";

/** The column a server newer than the others adds: why a source is unresolved. */
export const REASON_COLUMN = "unresolved_reason";

/**
 * A `sources` row from a catalog query, as the descriptor the listing returns
 * for the same source. A query hands back the columns as stored, so this only
 * names them; there is nothing to compute.
 */
export function descriptorFromRow(row: Record<string, unknown>): DataSourceDescriptor {
  const tensors = (row.tensors as Array<Record<string, unknown>> | null) ?? [];
  const reason = row.unresolved_reason as DataSourceDescriptor["unresolved_reason"];
  return {
    source_id: String(row.source_id),
    source_url: String(row.source_url ?? ""),
    source_type: String(row.source_type ?? ""),
    // A server predating the column has no key; the default reads as resolved,
    // the right answer for every source it ever listed.
    is_resolved: row.is_resolved === undefined ? true : Boolean(row.is_resolved),
    ...(reason ? { unresolved_reason: reason } : {}),
    tensors: tensors.map((t) => ({
      array_id: String(t.array_id),
      dim_labels: ((t.dim_labels as string[] | null) ?? []).map(String),
      shape: ((t.shape as number[] | null) ?? []).map(Number),
      dtype: String(t.dtype ?? ""),
    })),
  };
}
