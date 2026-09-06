import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RoiToolStripView, type RoiToolStripViewProps } from "./RoiToolStrip";

const render = (over: Partial<RoiToolStripViewProps> = {}) =>
  renderToStaticMarkup(
    <RoiToolStripView
      tool="select"
      draft={null}
      onSetTool={() => {}}
      onFinish={() => {}}
      onCancel={() => {}}
      {...over}
    />,
  );

describe("RoiToolStripView", () => {
  it("marks the active tool, and only it", () => {
    const html = render({ tool: "polygon" });
    expect((html.match(/aria-pressed="true"/g) ?? [])).toHaveLength(1);
  });

  it("says nothing about a draft when none is open", () => {
    expect(render()).not.toContain("Finish");
  });

  it("counts toward the minimum while a shape is too short", () => {
    const html = render({
      tool: "polygon",
      draft: { tool: "polygon", points: [[0, 0], [1, 1]] },
    });
    expect(html).toContain("2 of 3 points");
  });

  it("offers Finish only once the shape can close", () => {
    const short = render({ draft: { tool: "polygon", points: [[0, 0], [1, 1]] } });
    expect(short).toContain("disabled");
    const ready = render({ draft: { tool: "polygon", points: [[0, 0], [1, 1], [2, 2]] } });
    expect(ready).toContain("Finish");
    expect(ready).not.toContain("disabled");
  });

  it("always offers Cancel while drawing", () => {
    expect(render({ draft: { tool: "polygon", points: [[0, 0]] } })).toContain("Cancel");
  });
});
