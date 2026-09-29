/** Where the kernel restarted in the observe list.
 *
 * A restart does not renumber jobs -- ids are never reused -- but the jobs above
 * it ran in a namespace that is gone, so the list draws a line there, as a
 * notebook prints "Kernel restarted". The server gives each restart as the `seq`
 * of the last job admitted before it (`after`), not as a place in the retained
 * rows: a job that has aged out of the list must not take the line with it.
 * Placed by `seq`, never by parsing the id: a task's is `task-<random hex>`.
 *
 * Its own function because the placement is easy to get off by one, and the
 * poll it feeds is not reachable from a test without a DOM. */

export interface Restart {
  /** The `seq` of the last job admitted before the restart. */
  after: number;
  /** Wall-clock seconds. */
  at: number;
}

export type Entry<T> =
  | { kind: "row"; job: T }
  | { kind: "restart"; at: number; after: number };

/** *rowsNewestFirst* with a `restart` entry after every row newer than it (so
 * below the jobs that ran after the restart, above those before). A restart
 * older than every row that is left goes last. */
export function withRestarts<T extends { seq?: number }>(
  rowsNewestFirst: T[],
  restarts: Restart[],
): Entry<T>[] {
  const marks = [...restarts].sort((a, b) => b.after - a.after || b.at - a.at);
  const out: Entry<T>[] = [];
  let m = 0;
  for (const job of rowsNewestFirst) {
    while (m < marks.length && marks[m]!.after >= (job.seq ?? 0)) {
      out.push({ kind: "restart", ...marks[m]! });
      m++;
    }
    out.push({ kind: "row", job });
  }
  for (; m < marks.length; m++) out.push({ kind: "restart", ...marks[m]! });
  return out;
}
