import { beforeEach, describe, expect, it, vi } from "vitest";

// Same reason as sessionFetch.test.ts: neither exists in the node environment,
// and what this module does with them is not what is under test here.
vi.mock("../auth", () => ({
  authHeaders: (extra?: Record<string, string>) => ({ ...(extra || {}) }),
  redirectToUnlock: vi.fn(),
}));

import { fetchChatStatus, fetchHistory } from "./chatClient";

const answering = (body: unknown) =>
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })),
  );

beforeEach(() => {
  vi.unstubAllGlobals();
});

// These parsers build their result field by field rather than returning the
// body, so a key the server sends and the type declares is still absent unless
// it is read here. Both of these are read by something that renders.

describe("fetchChatStatus", () => {
  it("keeps the compacted count", async () => {
    // The pane shows every message whether or not the model still sees it in
    // full, so this number is the only sign a compaction happened.
    answering({ enabled: true, ready: true, model: "m", compacted: 12 });
    expect((await fetchChatStatus("/s"))!.compacted).toBe(12);
  });

  it("reads a child that does not send one as having folded nothing", async () => {
    answering({ enabled: true, ready: true, model: "m" });
    expect((await fetchChatStatus("/s"))!.compacted).toBe(0);
  });
});

describe("fetchHistory", () => {
  it("keeps whether the page is the whole thread", async () => {
    // Without it a view cannot tell a reset from a delta, and appends the new
    // conversation to the cleared one.
    answering({ messages: [], busy: false, full: true });
    expect((await fetchHistory("/s", "m-1"))!.full).toBe(true);
  });

  it("treats a child that does not say as sending a delta", async () => {
    answering({ messages: [], busy: false });
    expect((await fetchHistory("/s", "m-1"))!.full).toBe(false);
  });
});
