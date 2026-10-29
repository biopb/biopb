import type {
  AddSourceJobStatus,
  AddSourceResult,
  RemoveSourceResult,
  TensorHttpClient,
} from "@biopb/tensor-flight-client";
import type { NoticeDraft } from "./notices";

/** Wait between polls of a running add. */
export const ADD_POLL_MS = 1000;

type AddApi = Pick<TensorHttpClient, "addSource" | "addSourceStatus">;

interface RunAddOptions {
  /** Pause between polls; replaced in tests. */
  wait?: (ms: number) => Promise<void>;
  /** True to stop following (the add itself carries on server-side). */
  stopped?: () => boolean;
}

/**
 * Add a server path and follow it to the end, reporting each status read.
 *
 * Starting is idempotent per path on the server, so a path already being added
 * is joined, not started twice. The server names the job by the path it
 * normalised, so the poll asks by that name, not the one typed. Resolves with
 * the last status read; a failed poll rejects, and the add carries on regardless.
 */
export async function runAdd(
  api: AddApi,
  path: string,
  onUpdate: (status: AddSourceJobStatus) => void,
  { wait = (ms) => new Promise((r) => setTimeout(r, ms)), stopped = () => false }: RunAddOptions = {},
): Promise<AddSourceJobStatus> {
  let status = await api.addSource(path);
  const job = status.source_id;
  onUpdate(status);
  while (status.state === "running" && !stopped()) {
    await wait(ADD_POLL_MS);
    status = await api.addSourceStatus(job);
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
  return parts.join(", ");
}

/** How many of an add's failed paths its notice lists. */
const FAILURES_LISTED = 3;

/**
 * The notice an add earns once it has ended, or null while it still runs.
 *
 * An error when the add itself failed, or when it came to nothing but failures;
 * otherwise info, with any failed paths listed under it.
 */
export function addNotice(status: AddSourceJobStatus): NoticeDraft | null {
  if (status.state === "running") return null;
  if (status.state === "error" || !status.result) {
    return {
      kind: "error",
      text: `Could not add ${status.source_id}: ${status.error ?? "no result"}`,
      details: [],
    };
  }
  const r = status.result;
  const caution = addCaution(r);
  const text =
    (status.state === "cancelled" ? "Stopped. " : "") +
    describeAdd(r) +
    (caution ? ` — ${caution}` : "");
  const nothingWorked = r.added + r.already_present + r.refreshed + r.removed === 0;
  return {
    kind: r.failed_total > 0 && nothingWorked ? "error" : "info",
    text,
    details: r.failed.slice(0, FAILURES_LISTED).map((f) => `${f.path}: ${f.reason}`),
  };
}

/** The notice for a finished removal of the dropped root `name`. */
export function removeNotice(name: string, result: RemoveSourceResult): NoticeDraft {
  if (result.failed.length === 0) {
    return { kind: "info", text: `Removed ${name} (${count(result.removed, "source")})`, details: [] };
  }
  return {
    kind: "error",
    text: `Removed ${result.removed.toLocaleString()} of ${name}; ${result.failed.length.toLocaleString()} failed`,
    details: result.failed.slice(0, FAILURES_LISTED).map((f) => `${f.path}: ${f.reason}`),
  };
}

/** The caution an incomplete import deserves, or null when it was complete. */
export function addCaution(result: AddSourceResult): string | null {
  if (result.skipped_offline || result.skipped_cloud_dirs) {
    return "Some cloud files were left out (not downloaded to this machine).";
  }
  return null;
}
