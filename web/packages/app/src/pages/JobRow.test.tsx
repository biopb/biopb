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

  it("marks a verification run and names the workflow, not the intent", () => {
    // The server's intent for these reads "verify workflow: count nuclei",
    // which would spend the row's one line restating the badge beside it.
    const html = render(
      job({
        intent_preview: "verify workflow: count nuclei",
        verify: { title: "count nuclei", cells: 3, status: "ok" },
      }),
    );
    expect(html).toContain("badge verify");
    expect(html).toContain("count nuclei");
    expect(html).not.toContain("verify workflow: count nuclei");
  });

  it("leaves ordinary work unmarked", () => {
    // The badge is the filter's ground truth, so it must not appear on a cell
    // that merely mentions verification in its intent.
    const html = render(job({ intent_preview: "check the verify step" }));
    expect(html).not.toContain("badge verify");
  });

  it("names an untitled workflow rather than showing a blank line", () => {
    const html = render(job({ verify: { title: "", cells: 1, status: "ok" } }));
    expect(html).toContain("(untitled workflow)");
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
