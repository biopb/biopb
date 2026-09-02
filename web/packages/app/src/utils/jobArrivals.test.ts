import { describe, expect, it } from "vitest";
import { arrivals } from "./jobArrivals";

describe("arrivals", () => {
  const ids = (...v: string[]) => new Set(v);

  it("marks nothing on the first load", () => {
    // Every row is new then and none of them arrived; a whole list animating
    // at once says nothing about any of it.
    expect(arrivals(null, ids("j-1", "j-2", "j-3"))).toEqual([]);
  });

  it("marks only what was not there last poll", () => {
    expect(arrivals(ids("j-1", "j-2"), ids("j-1", "j-2", "j-3"))).toEqual([
      "j-3",
    ]);
  });

  it("marks nothing when the list did not change", () => {
    // Which is every poll but the few that matter -- 3s apart, for the life of
    // the session. Anything but [] here is a list that twitches.
    expect(arrivals(ids("j-1", "j-2"), ids("j-1", "j-2"))).toEqual([]);
  });

  it("does not re-announce a row when the kernel prunes an older one", () => {
    // The kernel keeps the newest 200 jobs, so a long session drops rows off
    // the far end of a list nothing was added to.
    expect(arrivals(ids("j-1", "j-2"), ids("j-2"))).toEqual([]);
  });

  it("counts a row that comes back after a restart", () => {
    // Restart empties the list and clears the history, so the next first poll
    // is a first load again -- and the rows after it did arrive.
    expect(arrivals(null, ids("j-1"))).toEqual([]);
    expect(arrivals(ids(), ids("j-1"))).toEqual(["j-1"]);
  });
});
