import { describe, expect, it } from "vitest";
import {
  advanceViewerKey,
  isResolutionUpgrade,
  sameStableAddress,
  type ViewerKeyState,
} from "./tensorIdentity";

describe("sameStableAddress", () => {
  it("ignores a token on either side", () => {
    expect(sameStableAddress("scratch@srvtok/tensorA", "scratch/tensorA")).toBe(true);
    expect(sameStableAddress("scratch/tensorA", "scratch@srvtok/tensorA")).toBe(true);
  });

  it("is false for two different tensors", () => {
    expect(sameStableAddress("scratch/tensorA", "scratch/tensorB")).toBe(false);
  });
});

// ViewerPane remounts its viewer whenever the key it derives from the tensor
// id changes -- deliberately, since the two hold state (a camera, contrast
// samples, a tile cache) that a new tensor should not inherit. `setTileInfo`
// in the store snaps a bare source_id to the specific field it resolved to,
// which changes that id's spelling without changing what it addresses, so
// this is what tells the pane not to remount over that upgrade specifically.
describe("isResolutionUpgrade", () => {
  it("is true when a bare source_id resolves to one of its fields", () => {
    expect(isResolutionUpgrade("scratch", "scratch/tensorA")).toBe(true);
  });

  it("is true across a content-pinned link, token and all", () => {
    expect(isResolutionUpgrade("scratch@abcd1234", "scratch@abcd1234/tensorA")).toBe(
      true,
    );
  });

  it("is false for a switch to an unrelated source", () => {
    expect(isResolutionUpgrade("scratch", "other")).toBe(false);
  });

  it("is false switching between two fields of an already-resolved source", () => {
    expect(isResolutionUpgrade("scratch/tensorA", "scratch/tensorB")).toBe(false);
  });

  it("is false when nothing changed", () => {
    expect(isResolutionUpgrade("scratch", "scratch")).toBe(false);
    expect(isResolutionUpgrade("scratch/tensorA", "scratch/tensorA")).toBe(false);
  });
});

/** Folds a sequence of tensor id props through `advanceViewerKey`, the way
 * ViewerPane's ref advances render over render, and returns each step's
 * resulting `keyedTensorId` -- what the remount key is actually built from. */
function fold(first: string, ...rest: string[]): string[] {
  let state: ViewerKeyState = { keyedTensorId: first, lastTensorId: first };
  return [first, ...rest].map((id) => {
    state = advanceViewerKey(state, id);
    return state.keyedTensorId;
  });
}

describe("advanceViewerKey", () => {
  it("does not remount across a resolve, but does remount a real switch after it", () => {
    // The exact sequence a bare-source click produces: the source itself
    // resolves to its default field, then the user picks a different one.
    // `keyedTensorId` must track which without conflating the two -- a
    // version of this that leaves `keyedTensorId` at the bare id forever
    // once resolved would read the second step as *also* a resolve of
    // "scratch", not a switch to a field other than the one already shown.
    expect(fold("scratch", "scratch/tensorA", "scratch/tensorB")).toEqual([
      "scratch",
      "scratch", // same key: the resolve does not remount
      "scratch/tensorB", // different key: the switch does
    ]);
  });

  it("remounts a switch straight to a field with no bare hop in front of it", () => {
    expect(fold("scratch/tensorA", "scratch/tensorB")).toEqual([
      "scratch/tensorA",
      "scratch/tensorB",
    ]);
  });

  it("remounts a switch to an unrelated source after a resolve", () => {
    expect(fold("scratch", "scratch/tensorA", "other")).toEqual([
      "scratch",
      "scratch",
      "other",
    ]);
  });

  it("keeps the same key across repeated renders with nothing new", () => {
    expect(fold("scratch", "scratch", "scratch")).toEqual([
      "scratch",
      "scratch",
      "scratch",
    ]);
  });

  it("keeps the resolved key stable across repeated resolved renders", () => {
    expect(fold("scratch", "scratch/tensorA", "scratch/tensorA")).toEqual([
      "scratch",
      "scratch",
      "scratch",
    ]);
  });
});
