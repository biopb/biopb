import { describe, expect, it } from "vitest";
import { descriptorFromRow } from "./catalogRow";

const ROW = {
  source_id: "s1",
  source_url: "file:///data/a.tif",
  source_type: "tiff",
  is_resolved: true,
  tensors: [{ array_id: "s1", dim_labels: ["Y", "X"], shape: [4, 8], dtype: "uint16" }],
};

describe("descriptorFromRow", () => {
  it("names the columns of a sources row, and nothing more", () => {
    expect(descriptorFromRow(ROW)).toEqual(ROW);
  });

  it("carries the reason when the row has one, and not otherwise", () => {
    const pending = descriptorFromRow({ ...ROW, is_resolved: false, unresolved_reason: "pending" });
    expect(pending.unresolved_reason).toBe("pending");
    expect(descriptorFromRow({ ...ROW, unresolved_reason: null })).not.toHaveProperty(
      "unresolved_reason",
    );
  });

  it("reads a row from a server predating is_resolved as resolved", () => {
    const { is_resolved: _drop, ...old } = ROW;
    expect(descriptorFromRow(old).is_resolved).toBe(true);
  });

  it("tolerates a source with no tensors yet", () => {
    expect(descriptorFromRow({ ...ROW, tensors: null }).tensors).toEqual([]);
  });
});
