import { describe, expect, it } from "vitest";
import { isReservedLabelName, splitLabelArrayId } from "./label-id.js";

describe("splitLabelArrayId", () => {
  it("takes a set on a sole-tensor source apart", () => {
    expect(splitLabelArrayId("src0/labels/nuclei")).toEqual({
      imageArrayId: "src0",
      name: "nuclei",
      level: null,
    });
  });

  it("keeps the image's own field", () => {
    expect(splitLabelArrayId("plate/A/1/labels/nuclei")).toEqual({
      imageArrayId: "plate/A/1",
      name: "nuclei",
      level: null,
    });
  });

  it("reads a native level under the set", () => {
    expect(splitLabelArrayId("src0/labels/nuclei/2")).toEqual({
      imageArrayId: "src0",
      name: "nuclei",
      level: "2",
    });
  });

  it("takes the LAST labels segment, so an image field may hold one", () => {
    // The server splits the same way: an image whose own field says "labels"
    // is not what makes its sets' ids ambiguous, because a name is slash-free.
    expect(splitLabelArrayId("src0/labels/a/labels/b")).toEqual({
      imageArrayId: "src0/labels/a",
      name: "b",
      level: null,
    });
  });

  it("is null for an ordinary tensor", () => {
    expect(splitLabelArrayId("src0")).toBeNull();
    expect(splitLabelArrayId("src0/0")).toBeNull();
  });

  it("is null for a trailing labels segment with no name", () => {
    expect(splitLabelArrayId("src0/labels")).toBeNull();
    expect(splitLabelArrayId("src0/labels/")).toBeNull();
  });

  it("does not read the source_id as the segment", () => {
    // `labels/nuclei` would be a source called "labels" holding a field
    // "nuclei" -- a source_id is slash-free and is never an image field.
    expect(splitLabelArrayId("labels/nuclei")).toBeNull();
  });
});

describe("isReservedLabelName", () => {
  it("marks the server's own sets", () => {
    expect(isReservedLabelName("@ome")).toBe(true);
    expect(isReservedLabelName("nuclei")).toBe(false);
  });
});
