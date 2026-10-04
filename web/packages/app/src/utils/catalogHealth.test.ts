import { describe, expect, it } from "vitest";
import { catalogIsFilling } from "./catalogHealth";

describe("catalogIsFilling", () => {
  it("is true while the scan runs", () => {
    expect(catalogIsFilling({ full_scan_in_progress: true })).toBe(true);
  });

  it("stays true after the scan while sources await registration", () => {
    expect(
      catalogIsFilling({ full_scan_in_progress: false, registration_pending: 12 }),
    ).toBe(true);
  });

  it("is false once both are done", () => {
    expect(
      catalogIsFilling({ full_scan_in_progress: false, registration_pending: 0 }),
    ).toBe(false);
  });

  it("reads a server that predates the count as the scan alone", () => {
    expect(catalogIsFilling({ full_scan_in_progress: false })).toBe(false);
    expect(catalogIsFilling({})).toBe(false);
  });

  it("is false when there is no health to read", () => {
    expect(catalogIsFilling(null)).toBe(false);
    expect(catalogIsFilling(undefined)).toBe(false);
  });
});
