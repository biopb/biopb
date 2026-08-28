import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ConsolePanel } from "./ObservePage";

// Server-rendered, for the reason JobRow.test.tsx gives: the workspace has no
// DOM test environment. That covers what the panel *offers*, which is the
// decision here; the typing and the clear-on-success are not reachable from
// this renderer.

type Running = Parameters<typeof ConsolePanel>[0]["running"];

const render = (running: Running = null) =>
  renderToStaticMarkup(
    <ConsolePanel running={running} onRun={async () => null} />,
  );

describe("ConsolePanel", () => {
  it("offers a reason beside the cell", () => {
    // The point of the field: a human's cell can now state why it ran, which
    // is what the notebook export could previously say about every writer
    // except the person at the machine.
    const html = render();
    expect(html).toContain("console-why");
    expect(html).toContain("why? (optional)");
  });

  it("says the reason is optional rather than asking for one", () => {
    // Someone running one line to look at a variable owes no explanation, and
    // a field that reads as required would make the console feel like paperwork.
    expect(render()).not.toContain("required");
  });

  it("still explains what the panel is, now in the label's tooltip", () => {
    // The field took the space the prose had, so the prose has to survive
    // somewhere -- otherwise the panel stops saying which kernel it runs in.
    expect(render()).toContain("serialized against the agent");
  });

  it("still renders a busy kernel as state, naming the holder", () => {
    // Regression guard: the busy label shares the row the field was added to.
    const html = render({
      job_id: "job-3",
      status: "running",
      elapsed: 1,
      origin: "mcp",
    } as NonNullable<Running>);
    expect(html).toContain("kernel busy");
    expect(html).toContain("job-3");
  });
});
