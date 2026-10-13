import { afterEach, describe, expect, it, vi } from "vitest";
import { copyText } from "./clipboard";

afterEach(() => vi.unstubAllGlobals());

describe("copyText", () => {
  it("uses the async clipboard where there is one", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    expect(await copyText("x")).toBe(true);
    expect(writeText).toHaveBeenCalledWith("x");
  });

  it("reports failure rather than throwing when nothing can copy", async () => {
    // No clipboard API and no DOM to select in: the plain-http, no-document case.
    vi.stubGlobal("navigator", {});
    expect(await copyText("x")).toBe(false);
    // A refused write falls through to the selection route, which also fails here.
    vi.stubGlobal("navigator", {
      clipboard: { writeText: vi.fn().mockRejectedValue(new Error("denied")) },
    });
    expect(await copyText("x")).toBe(false);
  });
});
