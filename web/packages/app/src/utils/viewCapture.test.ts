import { describe, expect, it } from "vitest";
import { fitWithin, readiness, viewParams, waitFor } from "./viewCapture";

describe("fitWithin", () => {
  it("scales the longest edge down to the limit and keeps the aspect", () => {
    expect(fitWithin({ width: 2000, height: 1000 }, 1000)).toEqual({ width: 1000, height: 500 });
  });
  it("never scales up", () => {
    expect(fitWithin({ width: 300, height: 150 }, 1024)).toEqual({ width: 300, height: 150 });
  });
  it("ignores a nonsense limit", () => {
    expect(fitWithin({ width: 300, height: 150 }, 0)).toEqual({ width: 300, height: 150 });
  });
});

describe("viewParams", () => {
  it("takes a bare query string, a leading ?, or a whole address", () => {
    for (const v of ["id=a&z=2", "?id=a&z=2", "http://h:8813/viewer?id=a&z=2"]) {
      expect(viewParams(v).get("id")).toBe("a");
      expect(viewParams(v).get("z")).toBe("2");
    }
  });
});

describe("readiness", () => {
  const base = { status: "ready", error: null, planeReady: true, roisPending: 0 };
  it("is ready only when the target, the plane and the overlays have landed", () => {
    expect(readiness(base).kind).toBe("ready");
    expect(readiness({ ...base, planeReady: false }).kind).toBe("waiting");
    expect(readiness({ ...base, roisPending: 1 }).kind).toBe("waiting");
    expect(readiness({ ...base, status: "resolving" }).kind).toBe("waiting");
  });
  it("reports a failed target with its error", () => {
    expect(readiness({ ...base, status: "failed", error: "404" })).toEqual({
      kind: "failed",
      error: "404",
    });
  });
});

describe("waitFor", () => {
  it("answers as soon as the check does", async () => {
    let n = 0;
    const r = await waitFor(() => (++n > 2 ? { kind: "ready" } : { kind: "waiting" }), 1000, () => false, 1);
    expect(r.kind).toBe("ready");
  });
  it("times out, and honours cancellation", async () => {
    expect((await waitFor(() => ({ kind: "waiting" }), 20, () => false, 5)).kind).toBe("timeout");
    expect((await waitFor(() => ({ kind: "waiting" }), 1000, () => true, 5)).kind).toBe("cancelled");
  });
});
