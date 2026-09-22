import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LabelPanelView } from "./LabelPanel";

const render = (arrayId: string | null, opacity = 0.5) =>
  renderToStaticMarkup(
    <LabelPanelView
      arrayId={arrayId}
      opacity={opacity}
      onOpacity={() => {}}
      onHide={() => {}}
    />,
  );

describe("LabelPanelView", () => {
  it("says nothing when no set is drawn", () => {
    // A panel over an image with no overlay is a control over nothing -- the
    // same rule RoiPanel follows for a server without annotations.
    expect(render(null)).toBe("");
  });

  it("names the set, not the whole address", () => {
    const html = render("zarr_b1/@labels/nuclei");
    expect(html).toContain("nuclei");
    // The address is still reachable, as the row's title.
    expect(html).toContain('title="zarr_b1/@labels/nuclei"');
  });

  it("falls back to the id when it names no set", () => {
    expect(render("zarr_b1")).toContain("zarr_b1");
  });

  it("reports the opacity as a percentage", () => {
    expect(render("zarr_b1/@labels/nuclei", 0.35)).toContain("35%");
    expect(render("zarr_b1/@labels/nuclei", 0)).toContain("0%");
  });

  it("offers the way back to no overlay", () => {
    expect(render("zarr_b1/@labels/nuclei")).toContain("Hide");
  });
});
