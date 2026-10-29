import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { INFO_MS, useNotices } from "./notices";

const { notify, dismiss } = useNotices.getState();
const texts = () => useNotices.getState().notices.map((n) => n.text);

describe("notices", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useNotices.setState({ notices: [] });
  });
  afterEach(() => vi.useRealTimers());

  it("lets an info notice go by itself", () => {
    notify("info", "Added 3 sources");
    expect(texts()).toEqual(["Added 3 sources"]);
    vi.advanceTimersByTime(INFO_MS - 1);
    expect(texts()).toEqual(["Added 3 sources"]);
    vi.advanceTimersByTime(2);
    expect(texts()).toEqual([]);
  });

  it("keeps an error until it is dismissed", () => {
    const id = notify("error", "Could not add /x");
    vi.advanceTimersByTime(INFO_MS * 10);
    expect(texts()).toEqual(["Could not add /x"]);
    dismiss(id);
    expect(texts()).toEqual([]);
  });

  it("carries detail lines, and drops only the one dismissed", () => {
    const a = notify("error", "a", ["/p: unreadable"]);
    notify("error", "b");
    expect(useNotices.getState().notices[0]!.details).toEqual(["/p: unreadable"]);
    dismiss(a);
    expect(texts()).toEqual(["b"]);
  });

  it("holds a few at a time, the oldest making room", () => {
    for (const t of ["1", "2", "3", "4", "5"]) notify("error", t);
    expect(texts()).toEqual(["2", "3", "4", "5"]);
  });
});
