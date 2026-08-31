import { describe, expect, it } from "vitest";
import { TIPS, eligibleTips, nextTip, type Tip, type TipContext } from "./tips";

const ctx = (over: Partial<TipContext> = {}): TipContext => ({
  sourceCount: 12,
  hasSelection: false,
  scanning: false,
  ...over,
});

describe("TIPS", () => {
  it("has no duplicate ids", () => {
    // The id is the rotation cursor: a duplicate would make the rotation
    // jump back to the first of the pair and never reach the rest.
    expect(new Set(TIPS.map((t) => t.id)).size).toBe(TIPS.length);
  });

  it("always has something to say", () => {
    // A context with nothing selected and nothing scanning is the emptiest one
    // the viewer can be in; the bar must not be blank there.
    expect(eligibleTips(ctx()).length).toBeGreaterThan(0);
  });
});

describe("eligibleTips", () => {
  it("holds back the tips whose context is not on screen", () => {
    const withoutSelection = eligibleTips(ctx()).length;
    const withSelection = eligibleTips(ctx({ hasSelection: true })).length;
    expect(withSelection).toBeGreaterThan(withoutSelection);
  });

  it("offers the server-filter tip only for a catalog big enough to need it", () => {
    const ids = (n: number) => eligibleTips(ctx({ sourceCount: n })).map((t) => t.id);
    expect(ids(999)).not.toContain("server-filter");
    expect(ids(5000)).toContain("server-filter");
  });

  it("offers the indexing tip only while the scan is running", () => {
    expect(eligibleTips(ctx()).map((t) => t.id)).not.toContain("indexing");
    expect(eligibleTips(ctx({ scanning: true })).map((t) => t.id)).toContain("indexing");
  });

  it("keeps declaration order, so the rotation is not reshuffled by a state change", () => {
    const all = eligibleTips(ctx({ hasSelection: true, scanning: true, sourceCount: 5000 }));
    expect(all.map((t) => t.id)).toEqual(TIPS.map((t) => t.id));
  });
});

describe("nextTip", () => {
  const list: Tip[] = [{ id: "a", text: "A" }, { id: "b", text: "B" }, { id: "c", text: "C" }];

  it("walks forward and wraps", () => {
    expect(nextTip("a", list)?.id).toBe("b");
    expect(nextTip("c", list)?.id).toBe("a");
  });

  it("starts at the front from nothing", () => {
    expect(nextTip(null, list)?.id).toBe("a");
  });

  it("starts at the front when the current tip has dropped out of the list", () => {
    // Deselecting a source removes most tips. The cursor then names a tip that
    // is describing something no longer on screen, so the rotation restarts
    // rather than resuming at whatever now sits at that index.
    expect(nextTip("gone", list)?.id).toBe("a");
  });

  it("has nothing to offer from an empty list", () => {
    expect(nextTip("a", [])).toBeNull();
  });
});
