import type { StateCreator } from "zustand";
import type { DataSourceDescriptor, SourceJobStatus } from "@biopb/tensor-flight-client";
import type { AppState, Get, Set } from "./types";

export type SourceJobKind = "resolve" | "warm";

/** Composite key for {@link AppState.sourceJobs}. */
export function jobKey(kind: SourceJobKind, sourceId: string): string {
  return `${kind}:${sourceId}`;
}

/**
 * How often an in-flight resolve/warm is re-read.
 *
 * Far tighter than the 60s catalog poll because this one is only running while
 * the user is watching a progress bar they asked for, and it stops the moment
 * nothing is in flight.
 */
const JOB_POLL_MS = 1000;

/**
 * Hydrate-ahead after a resolve, off.
 *
 * The server's chunk cache serves its segments by mmap, so warming a source
 * larger than RAM walks the whole page-cache LRU and evicts the segments
 * serving every *other* source -- for bytes warm never even uses, since the
 * read only exists to make the sync client write to disk. It does not keep its
 * own coarse levels either, and nothing portable would: there is no
 * `posix_fadvise` on Windows, which is where the synced-folder sources this
 * serves live (biopb/biopb#1043). Flip back once warm has a retention policy.
 *
 * This is the SPA's only warm trigger, so while it is false `WarmTray` never
 * appears. Both stay wired and tested, ready for the flip.
 */
const AUTO_WARM_AFTER_RESOLVE: boolean = false;

let _jobPollTimerId: ReturnType<typeof setInterval> | undefined;

/** A job that has stopped moving, whatever the reason. */
function isSettled(job: SourceJobStatus): boolean {
  return job.state !== "running";
}

export interface JobsSlice {
  /**
   * Resolve/warm jobs in flight or recently settled, keyed `"<kind>:<id>"`.
   *
   * Server-owned: these mirror `/api/sources/{id}/{kind}/status` and are
   * re-read on a poll, never advanced locally. A recall survives a reload and
   * a second tab, so the browser's copy is a view of it, not the record of it.
   */
  sourceJobs: Record<string, SourceJobStatus>;
  startResolve: (sourceId: string) => Promise<void>;
  startWarm: (sourceId: string) => Promise<void>;
  cancelSourceJob: (kind: SourceJobKind, sourceId: string) => Promise<void>;
  dismissSourceJob: (kind: SourceJobKind, sourceId: string) => void;
  stopJobPolling: () => void;
}

export const createJobsSlice: StateCreator<AppState, [], [], JobsSlice> = (set, get) => ({
  sourceJobs: {},

  async startResolve(sourceId: string) {
    await startSourceJob(get, set, "resolve", sourceId);
  },

  async startWarm(sourceId: string) {
    await startSourceJob(get, set, "warm", sourceId);
  },

  async cancelSourceJob(kind: SourceJobKind, sourceId: string) {
    const { client } = get();
    if (!client) return;
    try {
      const status = await client.http.cancelJob(kind, sourceId);
      putJob(set, status);
    } catch {
      // The server is the only thing that can actually stop the recall, and it
      // re-reports the flag on the next poll. A failed cancel is worth nothing
      // to say here -- the button simply hasn't taken yet.
    }
  },

  dismissSourceJob(kind: SourceJobKind, sourceId: string) {
    set((s) => {
      const next = { ...s.sourceJobs };
      delete next[jobKey(kind, sourceId)];
      return { sourceJobs: next };
    });
  },

  stopJobPolling() {
    if (_jobPollTimerId) {
      clearInterval(_jobPollTimerId);
      _jobPollTimerId = undefined;
    }
  },
});

function putJob(set: Set, status: SourceJobStatus): void {
  set((s) => ({
    sourceJobs: {
      ...s.sourceJobs,
      [jobKey(status.kind, status.source_id)]: status,
    },
  }));
}

async function startSourceJob(
  get: Get,
  set: Set,
  kind: SourceJobKind,
  sourceId: string,
): Promise<void> {
  const { client } = get();
  if (!client) return;
  try {
    const status =
      kind === "resolve"
        ? await client.http.startResolve(sourceId)
        : await client.http.startWarm(sourceId);
    putJob(set, status);
    ensureJobPolling(get, set);
  } catch (err) {
    // Synthesised rather than swallowed: a resolve that never started is the
    // one failure the user most needs told about, and the surface that shows
    // it reads `sourceJobs` -- so there has to be an entry for it to read.
    putJob(set, {
      kind,
      source_id: sourceId,
      state: "error",
      progress: {},
      error: err instanceof Error ? err.message : String(err),
      elapsed_seconds: 0,
      cancel_requested: false,
    });
  }
}

function ensureJobPolling(get: Get, set: Set): void {
  if (_jobPollTimerId) return;
  _jobPollTimerId = setInterval(() => {
    void pollSourceJobs(get, set);
  }, JOB_POLL_MS);
}

async function pollSourceJobs(get: Get, set: Set): Promise<void> {
  const { client, sourceJobs } = get();
  const running = Object.values(sourceJobs).filter((j) => !isSettled(j));
  if (!client || running.length === 0) {
    get().stopJobPolling();
    return;
  }

  // In parallel, not one at a time: each is an independent HTTP request, and a
  // sequential await-in-loop would make a tick's cost grow with job count and
  // risk overlapping the next tick if it ran long.
  const results = await Promise.allSettled(
    running.map((j) => client.http.jobStatus(j.kind, j.source_id)),
  );
  for (const result of results) {
    if (result.status === "rejected") {
      // A blip leaves the previous status in place; the next tick retries. Not
      // marked failed: the recall is server-side and unaffected by our poll.
      continue;
    }
    const after = result.value;
    putJob(set, after);
    if (isSettled(after)) {
      await onJobSettled(get, set, after);
    }
  }
}

/**
 * What happens when a job stops.
 *
 * A finished resolve leaves the catalog row stale -- the source is hydrated but
 * the tree still has the listing from before. The job carries the new row, which
 * replaces the stale one; the list is re-read only if it did not.
 *
 * The warm that used to follow it is gated off; see AUTO_WARM_AFTER_RESOLVE.
 * When on it is unconditional, with no check for whether the source is
 * multi-file, because the server answers that structurally -- a single-file
 * source's warm finishes at once with `files_total === 0` and the tray never
 * shows a bar for it. Keeping a list of multi-file source types on this side
 * would be a copy that drifts.
 */
async function onJobSettled(get: Get, set: Set, job: SourceJobStatus): Promise<void> {
  if (job.kind !== "resolve" || job.state !== "done") return;
  const row = job.source;
  const refresh = row ? Promise.resolve(applySourceRow(set, row)) : get().loadSources();
  if (!AUTO_WARM_AFTER_RESOLVE) {
    await refresh;
    return;
  }
  // Independent: the catalog reload and starting the warm hit different
  // endpoints and different store slices, so there is nothing for one to wait
  // on from the other.
  await Promise.all([refresh, get().startWarm(job.source_id)]);
}

/** Swap one source's row into the catalog copy, keeping its order. */
function applySourceRow(set: Set, row: DataSourceDescriptor): void {
  set((s) => ({
    sources: s.sources.map((src) => (src.source_id === row.source_id ? row : src)),
  }));
}
