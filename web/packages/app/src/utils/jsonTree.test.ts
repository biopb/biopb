import { describe, expect, it } from "vitest";
import {
  AUTO_EXPAND_MAX_ENTRIES,
  MAX_ROWS,
  autoExpanded,
  childCount,
  hasVisibleEntries,
  isEmptyForDisplay,
  visibleEntries,
} from "./jsonTree";

/** A stand-in for the MicroManager per-frame blob: wide, and nested. */
function wideBlob(keys: number): Record<string, unknown> {
  const blob: Record<string, unknown> = {};
  for (let i = 0; i < keys; i++) {
    blob[`FrameKey-${i}`] = { Frame: i, Channel: "STORM", Extra: null };
  }
  return blob;
}

describe("isEmptyForDisplay", () => {
  it("treats null, undefined and empty containers as nothing to show", () => {
    expect(isEmptyForDisplay(null)).toBe(true);
    expect(isEmptyForDisplay(undefined)).toBe(true);
    expect(isEmptyForDisplay([])).toBe(true);
    expect(isEmptyForDisplay({})).toBe(true);
  });

  it("keeps falsy leaves that are actual values", () => {
    expect(isEmptyForDisplay(0)).toBe(false);
    expect(isEmptyForDisplay("")).toBe(false);
    expect(isEmptyForDisplay(false)).toBe(false);
  });

  it("recurses: a container of nothing is nothing", () => {
    expect(isEmptyForDisplay({ a: { b: [null, null] } })).toBe(true);
    expect(isEmptyForDisplay({ a: { b: [null, 0] } })).toBe(false);
  });

  it("answers the same subtree from cache rather than walking it again", () => {
    // The rendering path asks about each node once per expanded level above it,
    // so the second walk has to be free.
    const shared = { deep: { deeper: Array.from({ length: 50_000 }, () => null) } };
    const blob = { a: shared, b: shared };

    isEmptyForDisplay(blob);
    const start = performance.now();
    for (let i = 0; i < 1000; i++) isEmptyForDisplay(shared);
    expect(performance.now() - start).toBeLessThan(50);
  });
});

describe("visibleEntries", () => {
  it("drops the entries that carry nothing", () => {
    expect(visibleEntries({ a: 1, b: null, c: {}, d: { e: [null] } })).toEqual([["a", 1]]);
  });
});

describe("hasVisibleEntries", () => {
  it("agrees with visibleEntries on both answers", () => {
    expect(hasVisibleEntries({ a: null, b: {} })).toBe(false);
    expect(hasVisibleEntries({ a: null, b: 2 })).toBe(true);
  });

  it("stops at the first entry worth showing", () => {
    // A collapsed wide node must not pay for a full filter to draw "{...}".
    // Each sibling reports when its own contents are read, so this asserts the
    // work skipped rather than elapsed time -- a shared CI runner cannot be
    // held to a millisecond bound, and the blob below allocates enough to put
    // a GC pause inside the measured window.
    let inspected = 0;
    const blob: Record<string, unknown> = { first: 1 };
    for (let i = 0; i < 2000; i++) {
      blob[`k${i}`] = {
        get deep() {
          inspected++;
          return Array.from({ length: 200 }, () => null);
        },
      };
    }

    expect(hasVisibleEntries(blob)).toBe(true);
    expect(inspected).toBe(0);
  });
});

describe("childCount", () => {
  it("counts what a node would draw, and nothing for a leaf", () => {
    expect(childCount([1, 2, 3])).toBe(3);
    expect(childCount({ a: 1, b: 2 })).toBe(2);
    expect(childCount("Channel 0")).toBe(0);
    expect(childCount(null)).toBe(0);
  });
});

describe("autoExpanded", () => {
  it("opens an ordinary header on sight", () => {
    const ome = Object.fromEntries(Array.from({ length: 16 }, (_, i) => [`k${i}`, i]));
    expect(autoExpanded(ome, 0)).toBe(true);
    expect(autoExpanded(ome, 1)).toBe(true);
  });

  it("leaves a wide node closed at any depth", () => {
    expect(autoExpanded(wideBlob(2001), 0)).toBe(false);
    expect(autoExpanded(wideBlob(AUTO_EXPAND_MAX_ENTRIES + 1), 0)).toBe(false);
    expect(autoExpanded(wideBlob(AUTO_EXPAND_MAX_ENTRIES), 0)).toBe(true);
  });

  it("still stops at two levels for a narrow one", () => {
    expect(autoExpanded({ a: 1 }, 2)).toBe(false);
  });
});

describe("row budget", () => {
  it("bounds what a source switch mounts, whatever the metadata holds", () => {
    // The regression this file exists for: two default-open levels over 2,001
    // top-level keys was ~248k rows, mounted and unmounted synchronously.
    const blob = wideBlob(2001);

    const topRows = autoExpanded(blob, 0) ? Math.min(childCount(blob), MAX_ROWS) : 0;
    let rows = topRows;
    for (const [, v] of visibleEntries(blob).slice(0, topRows)) {
      rows += autoExpanded(v, 1) ? Math.min(childCount(v), MAX_ROWS) : 0;
    }

    expect(rows).toBeLessThanOrEqual(MAX_ROWS + MAX_ROWS * MAX_ROWS);
  });
});
