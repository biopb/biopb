import { describe, expect, it } from "vitest";
import { checkablePath } from "./useMissingPaths";

describe("checkablePath", () => {
  it("names a server path to look up", () => {
    expect(checkablePath("/data/microscopy")).toBe("/data/microscopy");
    expect(checkablePath("  ~/images ")).toBe("~/images");
    expect(checkablePath("C:\\scans\\run1")).toBe("C:\\scans\\run1");
  });

  it("skips what is not a path on the server", () => {
    expect(checkablePath("")).toBeNull();
    expect(checkablePath("   ")).toBeNull();
    expect(checkablePath(undefined)).toBeNull();
    expect(checkablePath("grpc://lab:8815")).toBeNull();
    expect(checkablePath("s3://bucket/key")).toBeNull();
    expect(checkablePath("dnd://scratch")).toBeNull();
  });
});
