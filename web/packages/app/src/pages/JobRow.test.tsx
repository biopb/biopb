import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { JobRow } from "./ObservePage";

// Server-rendered, because the workspace has no DOM test environment. That
// covers what the row *offers*, which is the decision here; the click handler's
// stopPropagation is not reachable from this renderer.

const job = (over: Record<string, unknown> = {}) =>
  ({
    job_id: "j-1",
    status: "ok",
    elapsed: 3,
    code_preview: "print(1)",
    origin: "mcp",
    ...over,
  }) as Parameters<typeof JobRow>[0]["job"];

const render = (
  j: ReturnType<typeof job>,
  over: Partial<Parameters<typeof JobRow>[0]> = {},
) =>
  renderToStaticMarkup(
    <JobRow
      job={j}
      open={false}
      detail={undefined}
      onToggle={() => {}}
      onInterrupt={() => {}}
      {...over}
    />,
  );

describe("JobRow", () => {
  it("says why the cell was run, when whoever ran it said", () => {
    const html = render(job({ intent_preview: "isolate the nuclei channel" }));
    expect(html).toContain("isolate the nuclei channel");
    // Why *instead of* what: the row has one line, and the code is in the
    // detail this row expands to.
    expect(html).not.toContain("print(1)");
  });

  it("falls back to the code line when nobody said", () => {
    // The user console submits no intent, and neither does an older child.
    for (const j of [job(), job({ intent_preview: "" })]) {
      expect(render(j)).toContain("print(1)");
    }
  });

  it("offers interrupt on the cell that is running", () => {
    // Matched on the class, not the word: an `interrupted` job's own status
    // badge says "interrupted", which is a substring of it.
    expect(render(job({ status: "running" }))).toContain("job-stop");
  });

  it("marks the row the kernel is busy with", () => {
    // The status word is already there to be read; the class is what lets the
    // card be found without reading any of them.
    expect(render(job({ status: "running" }))).toContain("job busy");
  });

  it("plays its entrance only on the row that has just arrived", () => {
    // A row already on screen replays nothing on the next poll -- which is
    // every poll, so getting this wrong is a list that twitches every 3s.
    expect(render(job(), { fresh: true })).toContain("enter");
    expect(render(job())).not.toContain("enter");
  });

  it("keeps what it fetched after it is closed", () => {
    // The close animates, and a row that unmounted its content on close would
    // have an empty box to fold away.
    const html = render(job(), { open: false, detail: { code: "arr.mean()" } });
    expect(html).toContain("arr.mean()");
  });

  it("offers it nowhere else", () => {
    // The kernel runs one cell at a time, so a row that is not running has
    // nothing to interrupt -- and a button that cannot act is how the old
    // header placement earned its "No running job." dialog.
    for (const status of ["ok", "error", "interrupted"]) {
      expect(render(job({ status }))).not.toContain("job-stop");
    }
  });
});
