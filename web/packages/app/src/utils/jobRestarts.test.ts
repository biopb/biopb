import { describe, expect, it } from "vitest";
import { jobNumber, withRestarts } from "./jobRestarts";

const rows = (...ns: number[]) => ns.map((n) => ({ job_id: `job-${n}` }));
const shape = (es: ReturnType<typeof withRestarts<{ job_id: string }>>) =>
  es.map((e) => (e.kind === "row" ? e.job.job_id : `|${e.at}`));

describe("jobNumber", () => {
  it("reads job-N", () => expect(jobNumber("job-12")).toBe(12));
  it("is 0 for anything else", () => expect(jobNumber("verify")).toBe(0));
});

describe("withRestarts", () => {
  it("draws the line between the job before the restart and the one after", () => {
    // job-3 was the last before the restart; the list is newest first.
    expect(shape(withRestarts(rows(5, 4, 3, 2), [{ after: 3, at: 100 }]))).toEqual(
      ["job-5", "job-4", "|100", "job-3", "job-2"],
    );
  });

  it("puts a restart with no later job on top", () => {
    expect(shape(withRestarts(rows(3, 2), [{ after: 3, at: 100 }]))).toEqual([
      "|100",
      "job-3",
      "job-2",
    ]);
  });

  it("keeps a restart whose jobs have aged out at the bottom", () => {
    expect(shape(withRestarts(rows(9, 8), [{ after: 4, at: 100 }]))).toEqual([
      "job-9",
      "job-8",
      "|100",
    ]);
  });

  it("orders several restarts, newest first", () => {
    const out = withRestarts(rows(6, 4, 2), [
      { after: 2, at: 10 },
      { after: 4, at: 20 },
    ]);
    expect(shape(out)).toEqual(["job-6", "|20", "job-4", "|10", "job-2"]);
  });

  it("is the rows alone with no restarts", () => {
    expect(shape(withRestarts(rows(2, 1), []))).toEqual(["job-2", "job-1"]);
  });
});
