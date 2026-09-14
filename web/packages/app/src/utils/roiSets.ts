import { isReservedSetName } from "@biopb/tensor-flight-client";
import type { RoiAnnotation } from "@biopb/tensor-flight-client";

/**
 * Whether a row of `setName` is on screen.
 *
 * `visibleSets` is the chosen list, or null for the tensor's default: the
 * client-owned sets shown, the server-owned ones not. See `AppState.visibleSets`.
 */
export function isSetShown(setName: string, visibleSets: string[] | null): boolean {
  return visibleSets === null ? !isReservedSetName(setName) : visibleSets.includes(setName);
}

/** Sets present in a collection, with their counts, in first-seen order. */
export function roiSetCounts(rois: RoiAnnotation[]): Array<{ setName: string; count: number }> {
  const counts = new Map<string, number>();
  for (const roi of rois) {
    counts.set(roi.setName, (counts.get(roi.setName) ?? 0) + 1);
  }
  return [...counts].map(([setName, count]) => ({ setName, count }));
}
