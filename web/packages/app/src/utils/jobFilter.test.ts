import { describe, expect, it } from "vitest";
import { filterJobs, type VerificationMark } from "./jobFilter";

interface Row {
  job_id: string;
  verify?: VerificationMark | null;
}
const work: Row = { job_id: "j-1" };
const verified: Row = {
  job_id: "j-2",
  verify: { title: "count nuclei", cells: 2, status: "ok" },
};
const rows: Row[] = [work, verified];

describe("filterJobs", () => {
  it("shows everything under 'all'", () => {
    // The audit view stays complete: a verification is real work the kernel
    // did, and the default must not quietly drop it.
    expect(filterJobs(rows, "all")).toEqual(rows);
  });

  it("splits the two views with no row in neither", () => {
    expect(filterJobs(rows, "work")).toEqual([work]);
    expect(filterJobs(rows, "verify")).toEqual([verified]);
  });

  it("treats an explicit null verify as ordinary work", () => {
    // The server sends null, not an absent key, for a non-verification job.
    const nulled: Row = { job_id: "j-3", verify: null };
    expect(filterJobs([nulled], "work")).toEqual([nulled]);
    expect(filterJobs([nulled], "verify")).toEqual([]);
  });

  it("keeps order, so newest-first still means newest-first", () => {
    const many: Row[] = [work, verified, { job_id: "j-4" }];
    expect(filterJobs(many, "work").map((j) => j.job_id)).toEqual([
      "j-1",
      "j-4",
    ]);
  });
});
