import { describe, expect, it } from "vitest";
import type { TileInfo } from "@biopb/tensor-flight-client";
import {
  MAX_RECENT,
  descriptorFromTileInfo,
  forget,
  parseRecents,
  remember,
} from "./recentSources";

const TILE_INFO = {
  array_id: "upload_7f3@abcd1234",
  dim_labels: ["Y", "X"],
  shape: [1024, 1024],
  chunk_shape: [256, 256],
  dtype: "uint16",
} as unknown as TileInfo;

describe("remember", () => {
  it("puts the newest first", () => {
    expect(remember(["a", "b"], "c")).toEqual(["c", "a", "b"]);
  });

  it("moves a revisit to the front rather than stacking it", () => {
    expect(remember(["a", "b", "c"], "c")).toEqual(["c", "a", "b"]);
  });

  it("hands back the same array when the id is already the head", () => {
    // Identity, not equality: `noteRecent` skips the localStorage write on this,
    // and `applyViewerState` calls it on every slider move.
    const ids = ["a", "b"];
    expect(remember(ids, "a")).toBe(ids);
  });

  it("caps the list, dropping the oldest", () => {
    const full = Array.from({ length: MAX_RECENT }, (_, i) => `s${i}`);
    const next = remember(full, "new");
    expect(next).toHaveLength(MAX_RECENT);
    expect(next[0]).toBe("new");
    expect(next).not.toContain(`s${MAX_RECENT - 1}`);
  });

  it("ignores an empty id", () => {
    const ids = ["a"];
    expect(remember(ids, "")).toBe(ids);
  });
});

describe("forget", () => {
  it("drops only the ids named", () => {
    expect(forget(["a", "b", "c"], ["b"])).toEqual(["a", "c"]);
  });

  it("keeps identity when nothing is dropped, so no needless write", () => {
    const ids = ["a", "b"];
    expect(forget(ids, [])).toBe(ids);
    expect(forget(ids, ["zz"])).toBe(ids);
  });
});

describe("parseRecents", () => {
  it("reads a stored list", () => {
    expect(parseRecents('["a","b"]')).toEqual(["a", "b"]);
  });

  it.each([
    ["missing", null],
    ["empty", ""],
    ["not json", "{"],
    ["not a list", '{"a":1}'],
  ])("degrades to no recents on a %s value", (_label, raw) => {
    expect(parseRecents(raw)).toEqual([]);
  });

  it("drops entries that are not non-empty strings", () => {
    expect(parseRecents('["a",1,null,"",{"b":2},"c"]')).toEqual(["a", "c"]);
  });

  it("dedupes, so the cap is a real bound", () => {
    expect(parseRecents('["a","b","a"]')).toEqual(["a", "b"]);
  });

  it("caps a list that was written longer", () => {
    const raw = JSON.stringify(
      Array.from({ length: MAX_RECENT + 5 }, (_, i) => `s${i}`),
    );
    expect(parseRecents(raw)).toHaveLength(MAX_RECENT);
  });
});

describe("descriptorFromTileInfo", () => {
  it("addresses the row by the id asked for, not the versioned array_id", () => {
    // A content-pinned id stops resolving the moment the upload is rewritten,
    // so a row pinned to one would go dead on the next DoPut.
    const desc = descriptorFromTileInfo("upload_7f3", TILE_INFO);
    expect(desc.source_id).toBe("upload_7f3");
    expect(desc.tensors[0]?.array_id).toBe("upload_7f3");
  });

  it("leaves the url empty rather than guessing one", () => {
    expect(descriptorFromTileInfo("upload_7f3", TILE_INFO).source_url).toBe("");
  });

  it("carries the structure the tree renders", () => {
    const tensor = descriptorFromTileInfo("upload_7f3", TILE_INFO).tensors[0];
    expect(tensor?.shape).toEqual([1024, 1024]);
    expect(tensor?.dtype).toBe("uint16");
    expect(tensor?.dim_labels).toEqual(["Y", "X"]);
    // Empty for the reason a listing's entries are: the grid is answered per
    // resolved tensor, by tile_info itself.
    expect(tensor?.chunk_shape).toEqual([]);
  });
});
