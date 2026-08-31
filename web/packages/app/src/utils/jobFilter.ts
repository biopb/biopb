/** Telling a verification re-run from the session's own work.
 *
 * `verify_workflow` replays a workflow the session already ran, in a scratch
 * namespace, through the same job runner. Those cells therefore arrive in the
 * observe list looking exactly like ordinary code cells that happen to repeat
 * earlier ones -- and after a retry, twice over. The server marks them; this is
 * what the page does with the mark.
 */

/** What a job row says about being a verification run. */
export interface VerificationMark {
  title?: string;
  cells: number;
  status: string; // ok | error | running
}

/** Which cells the list is showing. Verification runs get their own view rather
 * than being hidden: they are real work the kernel did, and an audit that
 * silently drops rows is worse than one that repeats them -- so "all" stays the
 * default and remains complete. */
export type JobFilter = "all" | "work" | "verify";

/** The rows one view shows, in the order given.
 *
 * Keyed on the structural mark, never on the intent prose: the server's intent
 * for a verification reads "verify workflow: <title>", but a cell whose author
 * wrote "check the verify step" is ordinary work and must stay in the work view.
 */
export function filterJobs<T extends { verify?: VerificationMark | null }>(
  jobs: T[],
  filter: JobFilter,
): T[] {
  if (filter === "all") return jobs;
  const wantVerify = filter === "verify";
  return jobs.filter((j) => !!j.verify === wantVerify);
}
