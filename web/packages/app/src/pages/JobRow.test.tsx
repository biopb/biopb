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

const render = (j: ReturnType<typeof job>) =>
  renderToStaticMarkup(
    <JobRow
      job={j}
      open={false}
      detail={undefined}
      onToggle={() => {}}
      onInterrupt={() => {}}
    />,
  );

describe("JobRow", () => {
  it("offers interrupt on the cell that is running", () => {
    // Matched on the class, not the word: an `interrupted` job's own status
    // badge says "interrupted", which is a substring of it.
    expect(render(job({ status: "running" }))).toContain("job-stop");
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
