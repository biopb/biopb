import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { RoiAnnotation, RoiGeometry } from "@biopb/tensor-flight-client";
import {
  RoiAuthorView,
  RoiPanelView,
  type RoiAuthorViewProps,
  type RoiPanelViewProps,
} from "./RoiPanel";

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
      onClearSet={() => {}}
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
        roi({ roiId: "a", plane: { 1: 4 } }),
        roi({ roiId: "b", plane: { 1: 9 } }),
        roi({ roiId: "c" }),
      ],
      currentPlane: { 1: 4 },
    });
    // The axis-1=4 one and the unpinned one, out of three.
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

  it("offers a clear button per set, disarmed", () => {
    const html = render({ rois: [roi({ setName: "nuclei" }), roi({ setName: "cells" })] });
    // Two rows, two buttons, neither armed: the word that means the next click
    // deletes something only appears after the first one.
    expect(html.match(/roi-set-clear/g) ?? []).toHaveLength(2);
    expect(html).not.toContain("Sure?");
    expect(html).toContain("Delete every annotation in nuclei");
  });

  it("counts what a clear would delete, in the set it would delete from", () => {
    const html = render({
      rois: [roi({ setName: "nuclei" }), roi({ setName: "nuclei" }), roi({ setName: "cells" })],
    });
    expect(html).toContain("Delete all 2 annotations in nuclei");
    expect(html).toContain("Delete all 1 annotation in cells");
  });
});

// ---------------------------------------------------------------------------
// Authoring
// ---------------------------------------------------------------------------

const AXES = [
  { axis: 0, named: "t" as const, title: "T", key: "t", extent: 10 },
  { axis: 1, named: "c" as const, title: "C", key: "c", extent: 3 },
  { axis: 2, named: "z" as const, title: "Z", key: "z", extent: 20 },
];

const renderAuthor = (over: Partial<RoiAuthorViewProps> = {}) =>
  renderToStaticMarkup(
    <RoiAuthorView
      newLabel=""
      newSetName=""
      axes={AXES}
      broadcastAxes={[1]}
      currentPlane={{ 0: 3, 1: 1, 2: 12 }}
      selected={null}
      writeError={null}
      onSetNewLabel={() => {}}
      onSetNewSetName={() => {}}
      onToggleBroadcast={() => {}}
      {...over}
    />,
  );

describe("RoiAuthorView", () => {
  it("shows every pinnable axis with the index it will pin", () => {
    const html = renderAuthor();
    expect(html).toContain(">T<");
    expect(html).toContain(">3<");
    expect(html).toContain(">12<");
  });

  it("shows a broadcast axis as `all`, not as an index", () => {
    // Channel broadcasts by default: an annotation names an object, not a
    // channel, so it must not vanish when the user switches channel.
    const html = renderAuthor();
    expect(html).toContain(">all<");
    expect((html.match(/aria-pressed="true"/g) ?? [])).toHaveLength(1);
  });

  it("pins everything when nothing broadcasts", () => {
    const html = renderAuthor({ broadcastAxes: [] });
    expect(html).not.toContain(">all<");
  });

  it("says a selected annotation's pin in words, not axis numbers", () => {
    const html = renderAuthor({
      selected: {
        roiId: "r1",
        arrayId: "src",
        setName: "nuclei",
        label: "cell",
        geometry: { kind: "point", at: { x: 1, y: 1 } },
        plane: { 2: 12 },
        props: {},
        rev: 1,
        createdAtMs: 0,
        updatedAtMs: 0,
      },
    });
    expect(html).toContain("cell");
    expect(html).toContain("nuclei");
    expect(html).toContain("Z 12");
  });

  it("says what a selection is on one line", () => {
    // It elides rather than wrapping, so everything about the selection has to
    // fit in a single run of text.
    const html = renderAuthor({
      selected: {
        roiId: "r1",
        arrayId: "src",
        setName: "nuclei",
        label: "cell",
        geometry: { kind: "polygon", points: [] },
        plane: { 2: 12 },
        props: {},
        rev: 1,
        createdAtMs: 0,
        updatedAtMs: 0,
      },
      currentPlane: { 2: 12 },
    });
    expect(html).toContain("Selected: cell — polygon in nuclei, Z 12");
  });

  it("says when a selected annotation is on every plane", () => {
    // Otherwise indistinguishable from a pinned one on the plane in view.
    const html = renderAuthor({
      selected: {
        roiId: "r1",
        arrayId: "src",
        setName: "default",
        label: "",
        geometry: { kind: "point", at: { x: 1, y: 1 } },
        plane: {},
        props: {},
        rev: 1,
        createdAtMs: 0,
        updatedAtMs: 0,
      },
    });
    expect(html).toContain("no label — point in default, every plane");
  });

  it("reports a failed write", () => {
    expect(renderAuthor({ writeError: "422 rejected" })).toContain("422 rejected");
  });
});

describe("RoiAuthorView selection and the plane", () => {
  const pinned = {
    roiId: "r1",
    arrayId: "src",
    setName: "nuclei",
    label: "cell",
    geometry: { kind: "point" as const, at: { x: 1, y: 1 } },
    plane: { 2: 12 },
    props: {},
    rev: 1,
    createdAtMs: 0,
    updatedAtMs: 0,
  };

  it("names a selection on this plane", () => {
    expect(renderAuthor({ selected: pinned, currentPlane: { 2: 12 } })).toContain("Selected:");
  });

  it("withholds it once the plane moves off the selection", () => {
    // What the row says is deletable -- the viewer binds Delete on the same
    // condition -- so a shape the user cannot see must drop out of it. The
    // selection itself survives, so scrubbing back brings it into reach.
    const html = renderAuthor({ selected: pinned, currentPlane: { 2: 13 } });
    expect(html).not.toContain("Selected:");
    // The set name is unique to the selected block; "cell" is also the Label
    // input's placeholder, so it is in the markup either way.
    expect(html).not.toContain("nuclei");
  });

  it("keeps an unpinned selection reachable on every plane", () => {
    const html = renderAuthor({ selected: { ...pinned, plane: {} }, currentPlane: { 2: 99 } });
    expect(html).toContain("Selected:");
  });
});
