import type { BackendHealth } from "@biopb/tensor-flight-client";

/**
 * Is the catalog still filling in?
 *
 * Two things fill it: the scan that finds sources (`full_scan_in_progress`) and
 * the registration that reads them afterwards, which can outlast the scan by
 * hours on a large site (`registration_pending`: sources found, not yet read).
 * A UI that stopped watching at the end of the scan would leave those rows
 * showing as pending until the page was reloaded. A server that predates
 * `registration_pending` has only the first.
 */
export function catalogIsFilling(
  health: Pick<BackendHealth, "full_scan_in_progress" | "registration_pending"> | null | undefined,
): boolean {
  if (!health) return false;
  return !!health.full_scan_in_progress || (health.registration_pending ?? 0) > 0;
}
