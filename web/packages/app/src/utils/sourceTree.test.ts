import { describe, expect, it } from "vitest";
import type { DataSourceDescriptor } from "@biopb/tensor-flight-client";
import { getPathParts, recentNode, sourceLabel } from "./sourceTree";

const source = (over: Partial<DataSourceDescriptor> = {}): DataSourceDescriptor => ({
  source_id: "zarr_a3f2",
  source_url: "file:///data/experiment/plate1.zarr",
  source_type: "zarr",
  metadata_json: null,
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
