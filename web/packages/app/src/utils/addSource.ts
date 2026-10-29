import type {
  AddSourceJobStatus,
  AddSourceResult,
  TensorHttpClient,
} from "@biopb/tensor-flight-client";

/** Wait between polls of a running add. */
export const ADD_POLL_MS = 1000;

type AddApi = Pick<TensorHttpClient, "addSource" | "addSourceStatus">;

/**
 * Add a server path and follow it to the end, reporting each status read.
 *
 * Starting is idempotent per path on the server, so a path already being added
 * is joined, not started twice. Resolves with the last status read; a failed
 * poll rejects, and the add itself carries on server-side regardless.
 */
export async function runAdd(
  api: AddApi,
  path: string,
  onUpdate: (status: AddSourceJobStatus) => void,
  wait: (ms: number) => Promise<void> = (ms) => new Promise((r) => setTimeout(r, ms)),
  stopped: () => boolean = () => false,
): Promise<AddSourceJobStatus> {
  let status = await api.addSource(path);
  onUpdate(status);
  while (status.state === "running" && !stopped()) {
    await wait(ADD_POLL_MS);
    status = await api.addSourceStatus(path);
    onUpdate(status);
  }
  return status;
}

function count(n: number, noun: string): string {
  return `${n.toLocaleString()} ${noun}${n === 1 ? "" : "s"}`;
}

/** One line for what an add came to, e.g. "Added 12 sources, 3 already served". */
export function describeAdd(result: AddSourceResult): string {
  const parts: string[] = [];
  if (result.added) parts.push(`Added ${count(result.added, "source")}`);
  if (result.refreshed) parts.push(`${result.refreshed.toLocaleString()} rebuilt`);
  const stale = result.already_present - result.refreshed;
  if (stale > 0) parts.push(`${stale.toLocaleString()} already served`);
  if (result.removed) parts.push(`${result.removed.toLocaleString()} gone from disk`);
  if (result.failed_total) parts.push(`${result.failed_total.toLocaleString()} failed`);
  if (parts.length === 0) return "Nothing found to add";
  // "Added" leads the first clause only when there was something added.
  return parts.join(", ");
}

/** The caution an incomplete import deserves, or null when it was complete. */
export function addCaution(result: AddSourceResult): string | null {
  if (result.skipped_offline || result.skipped_cloud_dirs) {
    return "Some cloud files were left out (not downloaded to this machine).";
  }
  return null;
}
