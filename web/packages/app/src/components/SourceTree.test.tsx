import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { DataSourceDescriptor } from "@biopb/tensor-flight-client";
import { TreeRow } from "./SourceTree";
import { type TreeNode, recentNode } from "../utils/sourceTree";

// `TreeRow` takes props, so it server-renders. `SourceTree` itself does not:
// zustand's server snapshot is `getInitialState()`, so a store-reading component
// draws its defaults no matter what a test sets -- the same reason RoiPanel is
// split into a view. What the node *contains* is covered in utils/sourceTree.

const LISTED: DataSourceDescriptor = {
  source_id: "zarr_a3f2",
  source_url: "file:///data/experiment/plate1.zarr",
  source_type: "zarr",
  metadata_json: null,
  tensors: [],
};

const UPLOAD: DataSourceDescriptor = {
  source_id: "upload_7f3",
  source_url: "",
  source_type: "",
  metadata_json: null,
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
