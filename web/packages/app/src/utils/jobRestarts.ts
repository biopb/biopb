/** Where the kernel restarted in the observe list.
 *
 * A restart does not renumber jobs -- ids are never reused -- but the jobs above
 * it ran in a namespace that is gone, so the list draws a line there, as a
 * notebook prints "Kernel restarted". The server gives each restart as the last
 * job number issued before it (`after`), not as a place in the retained rows: a
 * job that has aged out of the list must not take the line with it.
 *
 * Its own function because the placement is easy to get off by one, and the
 * poll it feeds is not reachable from a test without a DOM. */

export interface Restart {
  /** The last job number issued before the restart. */
  after: number;
  /** Wall-clock seconds. */
  at: number;
}

export type Entry<T> =
  | { kind: "row"; job: T }
  | { kind: "restart"; at: number };

/** The N of `job-N`; 0 for any other id. */
export function jobNumber(id: string): number {
  const n = Number.parseInt(id.slice(id.lastIndexOf("-") + 1), 10);
  return Number.isNaN(n) ? 0 : n;
}

/** *rowsNewestFirst* with a `restart` entry after every row newer than it (so
 * below the jobs that ran after the restart, above those before). A restart
 * older than every row that is left goes last. */
export function withRestarts<T extends { job_id: string }>(
  rowsNewestFirst: T[],
  restarts: Restart[],
): Entry<T>[] {
  const marks = [...restarts].sort((a, b) => b.after - a.after || b.at - a.at);
  const out: Entry<T>[] = [];
  let m = 0;
  for (const job of rowsNewestFirst) {
    const n = jobNumber(job.job_id);
    while (m < marks.length && marks[m]!.after >= n) {
      out.push({ kind: "restart", at: marks[m]!.at });
      m++;
    }
    out.push({ kind: "row", job });
  }
  for (; m < marks.length; m++) out.push({ kind: "restart", at: marks[m]!.at });
  return out;
}
