import { describe, expect, it } from "vitest";
import type { DataSourceDescriptor, TensorDescriptor } from "@biopb/tensor-flight-client";
import {
  getPathParts,
  groupTensors,
  isUnresolved,
  recentNode,
  sourceLabel,
} from "./sourceTree";

const source = (over: Partial<DataSourceDescriptor> = {}): DataSourceDescriptor => ({
  source_id: "zarr_a3f2",
  source_url: "file:///data/experiment/plate1.zarr",
  source_type: "zarr",
  metadata_json: null,
  is_resolved: true,
  tensors: [],
  ...over,
});

/** What `hydrateRecents` builds for an id the catalog does not list: no url,
 * because tile_info describes a tensor and does not carry one. */
const UPLOAD = source({ source_id: "upload_7f3", source_url: "", source_type: "" });

describe("getPathParts", () => {
  it("splits a local file url into its path", () => {
    expect(getPathParts("file:///data/experiment/plate1.zarr")).toEqual([
      "data",
      "experiment",
      "plate1.zarr",
    ]);
  });

  it("roots a remote mirror at its endpoint", () => {
    // Otherwise every mirror collapses into one flat "grpc:" node (#297).
    expect(getPathParts("grpc://host:8815/remote/plate1.zarr")).toEqual([
      "grpc://host:8815",
      "remote",
      "plate1.zarr",
    ]);
  });

  it("strips the drop scheme without parsing it as a url", () => {
    // String-strip, not URL parse: a basename like "exp:2.zarr" would otherwise
    // misparse as a host and port.
    expect(getPathParts("dnd://scratch/exp:2.zarr")).toEqual(["scratch", "exp:2.zarr"]);
  });

  it("falls back to a plain split on something that is not a url", () => {
    expect(getPathParts("just/a/path")).toEqual(["just", "a", "path"]);
  });

  it("has nothing to say about an empty url", () => {
    expect(getPathParts("")).toEqual([]);
  });
});

describe("sourceLabel", () => {
  it("names a listed source by its leaf, not its whole path", () => {
    expect(sourceLabel(source())).toBe("plate1.zarr");
  });

  it("names an upload by its id, the only name it has", () => {
    expect(sourceLabel(UPLOAD)).toBe("upload_7f3");
  });
});

describe("recentNode", () => {
  it("is nothing at all when there are no recents", () => {
    expect(recentNode([])).toBeNull();
  });

  it("sorts by name, so a click cannot re-order the list under the pointer", () => {
    // `recents` arrives newest first. Rendering it that way moved the row just
    // clicked to the top, putting the next row somewhere else.
    const node = recentNode([UPLOAD, source()]);
    expect(node?.children.map((c) => c.name)).toEqual(["plate1.zarr", "upload_7f3"]);
  });

  it("orders the same however recency shuffles", () => {
    const byName = (n: ReturnType<typeof recentNode>) =>
      n?.children.map((c) => c.source?.source_id);
    const a = source({ source_id: "s1", source_url: "file:///d/a.zarr" });
    const b = source({ source_id: "s2", source_url: "file:///d/b.zarr" });
    expect(byName(recentNode([a, b]))).toEqual(byName(recentNode([b, a])));
  });

  it("breaks a name tie on source_id rather than leaving it to recency", () => {
    // Two folders, one basename. `Array.sort` is stable, so without the
    // tiebreak these two keep their incoming recency order and still swap on a
    // click -- the case where the rows look alike and the jump is least
    // visible.
    const a = source({ source_id: "aaa", source_url: "file:///one/plate1.zarr" });
    const b = source({ source_id: "bbb", source_url: "file:///two/plate1.zarr" });
    expect(recentNode([b, a])?.children.map((c) => c.source?.source_id)).toEqual([
      "aaa",
      "bbb",
    ]);
  });

  it("does not reorder the caller's array", () => {
    // `recentIds`/`recentSources` stay newest-first: that is what the cap
    // evicts by, and sorting in place would quietly change which entry is
    // dropped next.
    const recents = [UPLOAD, source()];
    recentNode(recents);
    expect(recents.map((s) => s.source_id)).toEqual(["upload_7f3", "zarr_a3f2"]);
  });

  it("namespaces its row ids away from the catalog's", () => {
    // The same source can appear twice, as two nodes holding one descriptor. A
    // shared node id would make `folderPathTo` reveal whichever copy it found
    // first, and the reveal is meant to find the source in its real place.
    expect(recentNode([source()])?.children[0]?.id).not.toBe("zarr_a3f2");
  });

  it("carries the descriptor through, so a row can be opened", () => {
    expect(recentNode([UPLOAD])?.children[0]?.source).toBe(UPLOAD);
  });
});

describe("isUnresolved", () => {
  const tensor = {
    array_id: "a",
    dim_labels: ["y", "x"],
    shape: [4, 4],
    chunk_shape: [],
    dtype: "uint16",
  };

  it("reads the server's own answer", () => {
    expect(isUnresolved(source({ is_resolved: false }))).toBe(true);
    expect(isUnresolved(source({ is_resolved: true }))).toBe(false);
  });

  it("does not infer from an empty tensor list, the way napari does", () => {
    // `_is_unresolved(src) = len(src.tensors) == 0` is wrong in both
    // directions; these two cases are what a catalog field buys us.
    expect(isUnresolved(source({ is_resolved: true, tensors: [] }))).toBe(false);
    expect(isUnresolved(source({ is_resolved: false, tensors: [tensor] }))).toBe(
      true,
    );
  });

  it("treats a descriptor from a server predating the field as resolved", () => {
    const legacy = source();
    delete (legacy as Partial<DataSourceDescriptor>).is_resolved;
    expect(isUnresolved(legacy)).toBe(false);
  });
});

function labelTensor(arrayId: string): TensorDescriptor {
  return {
    array_id: arrayId,
    dim_labels: ["y", "x"],
    shape: [64, 64],
    chunk_shape: [],
    dtype: "uint16",
  };
}

const ids = (tensors: TensorDescriptor[]) => tensors.map((t) => t.array_id);

describe("groupTensors", () => {
  it("leaves an ordinary source as one group per tensor", () => {
    const groups = groupTensors([labelTensor("src/A/1"), labelTensor("src/A/2")]);
    expect(groups.map((g) => g.image.array_id)).toEqual(["src/A/1", "src/A/2"]);
    expect(groups.every((g) => g.labelSets.length === 0)).toBe(true);
  });

  it("files a set under its image rather than beside it", () => {
    const groups = groupTensors([labelTensor("src0"), labelTensor("src0/@labels/nuclei")]);
    expect(groups).toHaveLength(1);
    expect(groups[0]!.image.array_id).toBe("src0");
    expect(ids(groups[0]!.labelSets)).toEqual(["src0/@labels/nuclei"]);
  });

  it("files each set under the image it names, not under the first one", () => {
    const groups = groupTensors([
      labelTensor("plate/A/1"),
      labelTensor("plate/A/2"),
      labelTensor("plate/A/2/@labels/nuclei"),
    ]);
    expect(groups.map((g) => g.image.array_id)).toEqual(["plate/A/1", "plate/A/2"]);
    expect(groups[0]!.labelSets).toEqual([]);
    expect(ids(groups[1]!.labelSets)).toEqual(["plate/A/2/@labels/nuclei"]);
  });

  it("keeps the server's tensor order, which puts images first", () => {
    // `tensors[0]` is the source's picture by the server's ordering, so
    // re-sorting the images here would disagree with the row above them.
    const groups = groupTensors([labelTensor("src/z"), labelTensor("src/a")]);
    expect(groups.map((g) => g.image.array_id)).toEqual(["src/z", "src/a"]);
  });

  it("sorts the sets by name, which the listing does not promise", () => {
    const groups = groupTensors([
      labelTensor("src0"),
      labelTensor("src0/@labels/nuclei"),
      labelTensor("src0/@labels/@ome"),
      labelTensor("src0/@labels/cells"),
    ]);
    expect(ids(groups[0]!.labelSets)).toEqual([
      "src0/@labels/@ome",
      "src0/@labels/cells",
      "src0/@labels/nuclei",
    ]);
  });

  it("shows an orphan set rather than hiding it", () => {
    // Should not happen -- the server registers a set on its parent -- but a
    // tensor the catalog lists and the tree silently drops is the worse failure.
    const groups = groupTensors([labelTensor("src/A/1"), labelTensor("src/A/9/@labels/nuclei")]);
    expect(groups.map((g) => g.image.array_id)).toEqual([
      "src/A/1",
      "src/A/9/@labels/nuclei",
    ]);
  });

  it("is empty for a source with no tensors", () => {
    expect(groupTensors([])).toEqual([]);
  });
});
