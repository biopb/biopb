import { isReservedSetName } from "@biopb/tensor-flight-client";

/**
 * Whether a row of `setName` is on screen.
 *
 * `visibleSets` is the chosen list, or null for the tensor's default: the
 * client-owned sets shown, the server-owned ones not. See `AppState.visibleSets`.
 */
export function isSetShown(setName: string, visibleSets: string[] | null): boolean {
  return visibleSets === null ? !isReservedSetName(setName) : visibleSets.includes(setName);
}
