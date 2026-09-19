import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { DataSourceDescriptor } from "@biopb/tensor-flight-client";
import { TreeRow } from "./SourceTree";
import {
  UNRESOLVED_GLYPH,
  type TreeNode,
  recentNode,
} from "../utils/sourceTree";

// `TreeRow` takes props, so it server-renders. `SourceTree` itself does not:
// zustand's server snapshot is `getInitialState()`, so a store-reading component
// draws its defaults no matter what a test sets -- the same reason RoiPanel is
// split into a view. What the node *contains* is covered in utils/sourceTree.

const LISTED: DataSourceDescriptor = {
  source_id: "zarr_a3f2",
  source_url: "file:///data/experiment/plate1.zarr",
  source_type: "zarr",
  metadata_json: null,
  is_resolved: true,
  tensors: [],
};

const UPLOAD: DataSourceDescriptor = {
  source_id: "upload_7f3",
  source_url: "",
  source_type: "",
  metadata_json: null,
  is_resolved: true,
  tensors: [
    {
      array_id: "upload_7f3",
      dim_labels: ["Y", "X"],
      shape: [1024, 1024],
      chunk_shape: [],
      dtype: "uint16",
    },
  ],
};

const render = (node: TreeNode, expanded: string[] = [node.id]) =>
  renderToStaticMarkup(
    <TreeRow
      node={node}
      activeSourceId={null}
      activeTensorId={null}
      expandedFolders={new Set(expanded)}
      toggleFolder={() => {}}
      selectSource={() => {}}
    />,
  );

describe("TreeRow under Recent", () => {
  it("shows an uploaded source, which the catalog cannot list at all", () => {
    // The feature in one line: cache:// uploads are deliberately absent from the
    // catalog (biopb/biopb#265), so without this node there is no route to one
    // from the tree.
    const html = render(recentNode([UPLOAD])!);
    expect(html).toContain("Recent");
    expect(html).toContain("upload_7f3");
  });

  it("leaves the catalog row as the only scroll target", () => {
    // Both copies of a source render, but only the catalog's carries
    // `data-source-id`: the reveal effect scrolls to the first match in document
    // order and "Recent" comes first, so a shared attribute would strand the
    // real row off-screen.
    expect(render(recentNode([LISTED])!)).not.toContain("data-source-id");
  });

  it("still marks the catalog's own row", () => {
    const html = render({
      id: LISTED.source_id,
      name: "plate1.zarr",
      type: "source",
      children: [],
      source: LISTED,
      depth: 1,
    });
    expect(html).toContain(`data-source-id="${LISTED.source_id}"`);
  });
});

describe("TreeRow for an unresolved source", () => {
  const CLOUD: DataSourceDescriptor = {
    source_id: "onedrive_9c1",
    source_url: "file:///data/cloud/timelapse.zarr",
    source_type: "zarr",
    metadata_json: null,
    is_resolved: false,
    tensors: [],
  };

  const sourceNode = (src: DataSourceDescriptor): TreeNode => ({
    id: src.source_id,
    name: "timelapse.zarr",
    type: "source",
    children: [],
    source: src,
    depth: 1,
  });

  it("marks it with the glyph napari uses and dims the row", () => {
    const html = render(sourceNode(CLOUD));
    expect(html).toContain(UNRESOLVED_GLYPH);
    expect(html).toContain("unresolved");
  });

  it("is not openable: selecting it would fetch a tile that cannot exist", () => {
    // The row is a div, not a button, so there is nothing to activate. That
    // also keeps the Resolve button below legal -- interactive content cannot
    // nest inside a button.
    const html = render(sourceNode(CLOUD));
    expect(html).toContain("<div");
    expect(html).not.toContain("<button");
  });

  it("offers Resolve, the only control on the row", () => {
    const html = renderToStaticMarkup(
      <TreeRow
        node={sourceNode(CLOUD)}
        activeSourceId={null}
        activeTensorId={null}
        expandedFolders={new Set(["onedrive_9c1"])}
        toggleFolder={() => {}}
        selectSource={() => {}}
        startResolve={() => {}}
        resolving={new Set()}
      />,
    );
    expect(html).toContain("Resolve");
    expect(html).toContain("resolve-btn");
  });

  it("says so instead of re-offering while a resolve is under way", () => {
    const html = renderToStaticMarkup(
      <TreeRow
        node={sourceNode(CLOUD)}
        activeSourceId={null}
        activeTensorId={null}
        expandedFolders={new Set(["onedrive_9c1"])}
        toggleFolder={() => {}}
        selectSource={() => {}}
        startResolve={() => {}}
        resolving={new Set(["onedrive_9c1"])}
      />,
    );
    expect(html).toContain("Resolving");
    expect(html).toContain("disabled");
  });

  it("explains itself on hover, keeping the url", () => {
    const html = render(sourceNode(CLOUD));
    expect(html).toContain("Not resolved");
    expect(html).toContain("timelapse.zarr");
  });

  it("leaves a resolved source untouched", () => {
    // The guard is `is_resolved`, not "has no tensors" -- a resolved source
    // that happens to list none must not be dimmed or disabled.
    const html = render(sourceNode({ ...CLOUD, is_resolved: true }));
    expect(html).not.toContain(UNRESOLVED_GLYPH);
    expect(html).toContain("<button");
  });

  it("shows no shape badge, having nothing to report yet", () => {
    const withTensors = { ...CLOUD, tensors: UPLOAD.tensors, is_resolved: true };
    expect(render(sourceNode(withTensors))).toContain("1024×1024");
    expect(render(sourceNode({ ...withTensors, is_resolved: false }))).not.toContain(
      "1024×1024",
    );
  });
});

describe("TreeRow with label sets", () => {
  const WITH_LABELS: DataSourceDescriptor = {
    source_id: "zarr_b1",
    source_url: "file:///data/experiment/cells.zarr",
    source_type: "zarr",
    metadata_json: null,
    is_resolved: true,
    tensors: [
      {
        array_id: "zarr_b1",
        dim_labels: ["t", "c", "y", "x"],
        shape: [50, 3, 512, 512],
        chunk_shape: [],
        dtype: "uint16",
      },
      {
        array_id: "zarr_b1/labels/nuclei",
        dim_labels: ["t", "y", "x"],
        shape: [50, 512, 512],
        chunk_shape: [],
        dtype: "uint32",
      },
    ],
  };

  const node: TreeNode = {
    id: WITH_LABELS.source_id,
    name: "cells.zarr",
    type: "source",
    children: [],
    source: WITH_LABELS,
    depth: 1,
  };

  const open = (labelOverlay: string | null = null) =>
    renderToStaticMarkup(
      <TreeRow
        node={node}
        activeSourceId={WITH_LABELS.source_id}
        activeTensorId="zarr_b1"
        expandedFolders={new Set()}
        toggleFolder={() => {}}
        selectSource={() => {}}
        labelOverlay={labelOverlay}
        setLabelOverlay={() => {}}
      />,
    );

  it("lists the set under its image once the source is open", () => {
    const html = open();
    expect(html).toContain("label-item");
    expect(html).toContain("nuclei");
  });

  it("marks the set that is drawn, and only that one", () => {
    expect(open("zarr_b1/labels/nuclei")).toContain('aria-pressed="true"');
    expect(open(null)).toContain('aria-pressed="false"');
    // A set of a different image never appears among these rows, so the mark
    // follows the id and needs no scoping of its own.
    expect(open("other/labels/nuclei")).toContain('aria-pressed="false"');
  });

  it("counts images in the pill, not tensors", () => {
    // A set is not an alternative to the image: a "2" here would say this
    // source holds two pictures, and it holds one.
    expect(open()).not.toContain("tensor-pill");
  });

  it("renders standalone, without the overlay props", () => {
    // `TreeRow` is a props component, and the row must not require the store.
    expect(() =>
      renderToStaticMarkup(
        <TreeRow
          node={node}
          activeSourceId={WITH_LABELS.source_id}
          activeTensorId="zarr_b1"
          expandedFolders={new Set()}
          toggleFolder={() => {}}
          selectSource={() => {}}
        />,
      ),
    ).not.toThrow();
  });
});
