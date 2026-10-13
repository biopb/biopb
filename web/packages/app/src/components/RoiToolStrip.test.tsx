import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RoiToolStripView, type RoiToolStripViewProps } from "./RoiToolStrip";

const render = (over: Partial<RoiToolStripViewProps> = {}) =>
  renderToStaticMarkup(
    <RoiToolStripView
      tool="select"
      draft={null}
      polylineWidth={4}
      onSetTool={() => {}}
      onSetPolylineWidth={() => {}}
      onSetDraftText={() => {}}
      onFinish={() => {}}
      onCancel={() => {}}
      {...over}
    />,
  );

describe("RoiToolStripView", () => {
  it("offers the width control with the polyline tool, and only then", () => {
    // A polyline's width is geometry, so it is decided before the first click
    // -- but it means nothing to any other tool.
    expect(render({ tool: "polyline" })).toContain("Width");
    expect(render({ tool: "polygon" })).not.toContain("Width");
    expect(render({ tool: "select" })).not.toContain("Width");
  });

  it("shows the width it will store", () => {
    const html = render({ tool: "polyline", polylineWidth: 12 });
    expect(html).toContain('value="12"');
    expect(html).toContain(">12<");
  });

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

describe("the text tool", () => {
  it("offers itself, and says where to click while nothing is placed", () => {
    expect(render()).toContain('title="Text (click where it goes, then type)"');
    expect(render({ tool: "text" })).toContain("Click the image where the text goes.");
    expect(render({ tool: "polygon" })).not.toContain("where the text goes");
  });

  it("asks for the text once the anchor is placed, and shows what is typed", () => {
    const html = render({
      tool: "text",
      draft: { tool: "text", points: [[1, 1]], text: "late\nstage" },
    });
    expect(html).toContain("<textarea");
    expect(html).toContain("late\nstage");
    expect(html).toContain('maxLength="2000"');
    // No point count: a caption has one anchor and the count would only confuse.
    expect(html).not.toContain("points");
  });

  it("holds Finish until there is text", () => {
    const blank = render({ tool: "text", draft: { tool: "text", points: [[1, 1]], text: "" } });
    expect(blank).toMatch(/<button[^>]*disabled=""[^>]*>Finish/);
    const typed = render({ tool: "text", draft: { tool: "text", points: [[1, 1]], text: "x" } });
    expect(typed).not.toMatch(/<button[^>]*disabled=""[^>]*>Finish/);
  });

  it("shows no text box for the other tools", () => {
    const html = render({
      tool: "polygon",
      draft: { tool: "polygon", points: [[0, 0], [1, 1]] },
    });
    expect(html).not.toContain("<textarea");
  });
});
