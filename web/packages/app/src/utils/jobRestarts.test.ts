import { describe, expect, it } from "vitest";
import { withRestarts } from "./jobRestarts";

// `id` only names a row in the assertions; placement is by `seq`.
const rows = (...seqs: number[]) => seqs.map((seq) => ({ seq, id: `#${seq}` }));
const shape = (es: ReturnType<typeof withRestarts<{ seq: number; id: string }>>) =>
  es.map((e) => (e.kind === "row" ? e.job.id : `|${e.at}`));

describe("withRestarts", () => {
  it("draws the line between the job before the restart and the one after", () => {
    // Seq 3 was the last admitted before the restart; the list is newest first.
    expect(shape(withRestarts(rows(5, 4, 3, 2), [{ after: 3, at: 100 }]))).toEqual(
      ["#5", "#4", "|100", "#3", "#2"],
    );
  });

  it("places a task among the jobs by seq, whatever its id says", () => {
    const list = [
      { seq: 4, id: "job-9" },
      { seq: 3, id: "task-3fa2c1" }, // a hex id that parses as nothing useful
      { seq: 2, id: "task-123456" }, // one that parses as a large number
      { seq: 1, id: "job-1" },
    ];
    expect(shape(withRestarts(list, [{ after: 2, at: 100 }]))).toEqual([
      "job-9",
      "task-3fa2c1",
      "|100",
      "task-123456",
      "job-1",
    ]);
  });

  it("puts a restart with no later job on top", () => {
    expect(shape(withRestarts(rows(3, 2), [{ after: 3, at: 100 }]))).toEqual([
      "|100",
      "#3",
      "#2",
    ]);
  });

  it("keeps a restart whose jobs have aged out at the bottom", () => {
    expect(shape(withRestarts(rows(9, 8), [{ after: 4, at: 100 }]))).toEqual([
      "#9",
      "#8",
      "|100",
    ]);
  });

  it("orders several restarts, newest first", () => {
    const out = withRestarts(rows(6, 4, 2), [
      { after: 2, at: 10 },
      { after: 4, at: 20 },
    ]);
    expect(shape(out)).toEqual(["#6", "|20", "#4", "|10", "#2"]);
  });

  it("is the rows alone with no restarts", () => {
    expect(shape(withRestarts(rows(2, 1), []))).toEqual(["#2", "#1"]);
  });
});
