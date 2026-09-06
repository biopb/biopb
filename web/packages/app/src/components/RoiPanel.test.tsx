import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { RoiAnnotation, RoiGeometry } from "@biopb/tensor-flight-client";
import { RoiPanelView, type RoiPanelViewProps } from "./RoiPanel";

// Server-rendered, as JobRow is: the workspace has no DOM environment, and this
// covers what the panel *says*, which is the decision here. The view takes props
// rather than reading the store precisely so this works -- zustand's server
// snapshot is `getInitialState()`, so a store-reading component would render its
// defaults here no matter what the test set.

const POLY: RoiGeometry = {
  kind: "polygon",
  points: [{ x: 0, y: 0 }, { x: 4, y: 0 }, { x: 4, y: 4 }],
};

function roi(over: Partial<RoiAnnotation> = {}): RoiAnnotation {
  return {
    roiId: "r",
    arrayId: "src/Image:0",
    setName: "default",
    label: "",
    geometry: POLY,
    plane: {},
    props: {},
    rev: 1,
    createdAtMs: 0,
    updatedAtMs: 0,
    ...over,
  };
}

const render = (over: Partial<RoiPanelViewProps> = {}) =>
  renderToStaticMarkup(
    <RoiPanelView
      rois={[]}
      currentPlane={{}}
      hiddenSets={[]}
      showRois
      loading={false}
      error={null}
      truncated={false}
      skipped={0}
      unavailable={false}
      onToggleOverlay={() => {}}
      onToggleSet={() => {}}
      {...over}
    />,
  );

describe("RoiPanelView", () => {
  it("says nothing at all when the server does not offer annotations", () => {
    // A deployment fact, not a fault: a permanently empty panel is worse than
    // no panel.
    expect(render({ unavailable: true })).toBe("");
  });

  it("counts what is on this plane against the whole set", () => {
    const html = render({
      rois: [
        roi({ roiId: "a", plane: { z: 4 } }),
        roi({ roiId: "b", plane: { z: 9 } }),
        roi({ roiId: "c" }),
      ],
      currentPlane: { z: 4 },
    });
    // The z=4 one and the unpinned one, out of three.
    expect(html).toContain("2 of 3 here");
  });

  it("lists each set with its count", () => {
    const html = render({
      rois: [roi({ setName: "nuclei" }), roi({ setName: "nuclei" }), roi({ setName: "debris" })],
    });
    expect(html).toContain("nuclei");
    expect(html).toContain("debris");
  });

  it("says none rather than a count when the tensor has no annotations", () => {
    expect(render()).toContain("none");
  });

  it("distinguishes a first load from an empty tensor", () => {
    expect(render({ loading: true })).toContain("loading");
    // Not while a set is already on screen: a refetch should not blank a count.
    expect(render({ loading: true, rois: [roi()] })).toContain("1 of 1 here");
  });

  it("says the set was clipped by the server's cap", () => {
    expect(render({ rois: [roi()], truncated: true })).toContain("per-tensor maximum");
  });

  it("owns up to rows it could not read", () => {
    const html = render({ rois: [roi()], skipped: 2 });
    expect(html).toContain("2 annotations could not be read");
  });

  it("reports a load failure", () => {
    expect(render({ error: "Timeout after 8000ms" })).toContain("Timeout after 8000ms");
  });

  it("shows a hidden set as unchecked but still listed", () => {
    // Still listed, because the row is how it gets switched back on.
    const html = render({ rois: [roi({ setName: "nuclei" })], hiddenSets: ["nuclei"] });
    expect(html).toContain("nuclei");
    expect(html).not.toContain('checked=""/> <span');
  });
});
