import { beforeEach, describe, expect, it, vi } from "vitest";

// Same reason as sessionFetch.test.ts: neither exists in the node environment,
// and what this module does with them is not what is under test here.
vi.mock("../auth", () => ({
  authHeaders: (extra?: Record<string, string>) => ({ ...(extra || {}) }),
  redirectToUnlock: vi.fn(),
}));

import {
  fetchChatStatus,
  fetchHistory,
  fetchModels,
  setModel,
} from "./chatClient";

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

  it("is null when the read fails, so the pane keeps its thread", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("nope", { status: 404 })),
    );
    expect(await fetchHistory("/s", null)).toBe(null);
  });
});

describe("fetchModels", () => {
  it("takes the choices, and names one that came without a name", () => {
    answering({
      model: "openai/gpt-5.5",
      choices: [{ value: "openai/gpt-5.5", name: "GPT-5.5" }, { value: "x/y" }],
    });
    return fetchModels("/s").then((m) => {
      expect(m).toEqual({
        model: "openai/gpt-5.5",
        choices: [
          { value: "openai/gpt-5.5", name: "GPT-5.5" },
          { value: "x/y", name: "x/y" },
        ],
      });
    });
  });

  it("drops a choice with no value rather than offering a blank row", async () => {
    answering({ model: "m", choices: [{ name: "no value" }, null, { value: "ok" }] });
    expect((await fetchModels("/s"))!.choices).toEqual([
      { value: "ok", name: "ok" },
    ]);
  });

  it("reads a provider with no list as having none, not as unreachable", async () => {
    // Null is a failed read and means keep what you have; an empty list is an
    // answer -- `GET /models` is optional in the OpenAI-compatible shape.
    answering({ model: "test-model", choices: [] });
    expect((await fetchModels("/s"))!.choices).toEqual([]);
  });
});

describe("setModel", () => {
  const refusing = (status: number, body: unknown) =>
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(body), { status })),
    );

  it("passes on what the provider said", async () => {
    // The refusal has to say what to type instead, or it sends the reader to
    // the config file to find out.
    refusing(400, { error: "no such model 'gpt-6'. Offered: x, y" });
    expect(await setModel("/s", "gpt-6")).toContain("Offered: x, y");
  });

  it("reads a busy session as state rather than as a failure", async () => {
    refusing(409, {});
    expect(await setModel("/s", "x")).toContain("turn is running");
  });

  it("is null when it took", async () => {
    answering({ model: "x" });
    expect(await setModel("/s", "x")).toBe(null);
  });
});
