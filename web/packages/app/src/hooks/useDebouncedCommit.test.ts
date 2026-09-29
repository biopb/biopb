import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createDebouncer } from "./useDebouncedCommit";

describe("createDebouncer", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("commits once the quiet period passes, with the last value for a key", () => {
    const d = createDebouncer(150);
    const seen: number[] = [];
    d.schedule("gamma", () => seen.push(1));
    vi.advanceTimersByTime(100);
    d.schedule("gamma", () => seen.push(2));
    vi.advanceTimersByTime(100);
    expect(seen).toEqual([]);
    vi.advanceTimersByTime(60);
    expect(seen).toEqual([2]);
  });

  it("does not let a second slider drop the first one's commit", () => {
    // What one shared timer did: moving the percentile slider 100 ms after the
    // axis slider cancelled the axis commit.
    const d = createDebouncer(150);
    const seen: string[] = [];
    d.schedule("axis:a0", () => seen.push("axis"));
    vi.advanceTimersByTime(100);
    d.schedule("percentile", () => seen.push("percentile"));
    vi.advanceTimersByTime(200);
    expect(seen.sort()).toEqual(["axis", "percentile"]);
  });

  it("drops what is pending on cancelAll", () => {
    const d = createDebouncer(150);
    const seen: string[] = [];
    d.schedule("a", () => seen.push("a"));
    d.schedule("b", () => seen.push("b"));
    d.cancelAll();
    vi.advanceTimersByTime(500);
    expect(seen).toEqual([]);
  });
});
