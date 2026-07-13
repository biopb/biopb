import { describe, it, expect } from "vitest";
import { splitConfigErrors } from "./admin-errors.js";
import type { AdminConfigError } from "./types.js";

describe("splitConfigErrors", () => {
  it("attributes a sources[i] error to its row using the wire's STRING index", () => {
    // The backend stringifies path elements, so the index arrives as "0", not 0.
    // This is the exact shape that regressed the typeof===number check.
    const errors: AdminConfigError[] = [
      { path: ["sources", "0", "url"], message: "'url' is a required property" },
    ];
    const { byIndex, general } = splitConfigErrors(errors);
    expect(byIndex[0]).toEqual(["url: 'url' is a required property"]);
    expect(general).toEqual([]);
  });

  it("also accepts a native numeric index (defensive)", () => {
    const errors: AdminConfigError[] = [
      { path: ["sources", 1, "type"], message: "not a valid enum" },
    ];
    expect(splitConfigErrors(errors).byIndex[1]).toEqual(["type: not a valid enum"]);
  });

  it("labels a whole-source error (no field) as 'source'", () => {
    const errors: AdminConfigError[] = [
      { path: ["sources", "2"], message: "is not of type object" },
    ];
    expect(splitConfigErrors(errors).byIndex[2]).toEqual(["source: is not of type object"]);
  });

  it("routes non-source errors to the general summary with a dotted path", () => {
    const errors: AdminConfigError[] = [
      { path: ["pyramid", "downscale_factor"], message: "1 is less than the minimum of 2" },
      { path: [], message: "top-level problem" },
    ];
    const { byIndex, general } = splitConfigErrors(errors);
    expect(byIndex).toEqual({});
    expect(general).toEqual([
      "pyramid.downscale_factor: 1 is less than the minimum of 2",
      "top-level problem",
    ]);
  });

  it("does not treat a non-integer second element as a row index", () => {
    const errors: AdminConfigError[] = [
      { path: ["sources", "all"], message: "must be an array" },
    ];
    const { byIndex, general } = splitConfigErrors(errors);
    expect(byIndex).toEqual({});
    expect(general).toEqual(["sources.all: must be an array"]);
  });

  it("groups multiple errors on the same row", () => {
    const errors: AdminConfigError[] = [
      { path: ["sources", "0", "url"], message: "required" },
      { path: ["sources", "0", "type"], message: "bad enum" },
    ];
    expect(splitConfigErrors(errors).byIndex[0]).toEqual([
      "url: required",
      "type: bad enum",
    ]);
  });
});
