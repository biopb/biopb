import { describe, expect, it } from "vitest";
import {
  applyLiveOutput,
  fromChatHistory,
  groupThread,
  latestLine,
  mergeHistory,
  toolText,
  type ChatMessage,
  type ToolCallItem,
} from "./chatThread";

const user = (id: string, content: string): ChatMessage => ({
  id,
  role: "user",
  content,
});

const assistant = (
  id: string,
  content: string,
  calls: { id: string; name: string }[] = [],
): ChatMessage => ({
  id,
  role: "assistant",
  content,
  ...(calls.length
    ? {
        tool_calls: calls.map((c) => ({
          id: c.id,
          function: { name: c.name, arguments: "{}" },
        })),
      }
    : {}),
});

const toolResult = (
  id: string,
  callId: string,
  name: string,
  content: string,
  error = false,
): ChatMessage => ({
  id,
  role: "tool",
  content,
  tool_call_id: callId,
  name,
  ...(error ? { error: true } : {}),
});

const imageCarrier = (id: string, name: string, data: string): ChatMessage => ({
  id,
  role: "user",
  content: `(image returned by ${name})`,
  image: data,
  mime: "image/png",
});

const calls = (items: ReturnType<typeof fromChatHistory>): ToolCallItem[] =>
  items.filter((i): i is ToolCallItem => i.kind === "tool_call");

describe("mergeHistory", () => {
  it("appends a delta", () => {
    const merged = mergeHistory([user("m-1", "hi")], [assistant("m-2", "hello")]);
    expect(merged.map((m) => m.id)).toEqual(["m-1", "m-2"]);
  });

  it("replaces when the response repeats something we hold", () => {
    // An unknown `after` returns the whole conversation rather than an error,
    // and the request cannot tell the two apart. Appending it would duplicate
    // every message on screen.
    const held = [user("m-1", "hi"), assistant("m-2", "hello")];
    const replay = [...held, user("m-3", "again")];
    expect(mergeHistory(held, replay).map((m) => m.id)).toEqual([
      "m-1",
      "m-2",
      "m-3",
    ]);
  });

  it("replaces after a reset, whose ids start over at m-1", () => {
    const held = [user("m-1", "old"), assistant("m-2", "older")];
    const fresh = [user("m-1", "new")];
    expect(mergeHistory(held, fresh)).toEqual(fresh);
  });

  it("does not truncate the thread when two polls overlap", () => {
    // A fetch that outlives the interval leaves two in flight against the same
    // cursor, and the second returns what the first has already appended. That
    // repetition is not a replay: reading it as one would drop everything
    // before the overlap.
    const held = [user("m-1", "hi"), assistant("m-2", "a"), assistant("m-3", "b")];
    const late = [held[1]!, held[2]!]; // the slower poll, same delta
    expect(mergeHistory(held, late).map((m) => m.id)).toEqual([
      "m-1",
      "m-2",
      "m-3",
    ]);
  });

  it("appends only the part of a delta it has not seen", () => {
    const held = [user("m-1", "hi"), assistant("m-2", "a")];
    const overlapping = [assistant("m-2", "a"), assistant("m-3", "b")];
    expect(mergeHistory(held, overlapping).map((m) => m.id)).toEqual([
      "m-1",
      "m-2",
      "m-3",
    ]);
  });

  it("takes the first page whole", () => {
    const page = [user("m-1", "hi")];
    expect(mergeHistory([], page)).toEqual(page);
  });

  it("keeps what it has when the page is empty", () => {
    const held = [user("m-1", "hi")];
    expect(mergeHistory(held, [])).toEqual(held);
  });
});

describe("fromChatHistory", () => {
  it("keeps the user's own words as a user message", () => {
    const items = fromChatHistory([user("m-1", "count the nuclei")]);
    expect(items).toEqual([
      {
        kind: "message",
        id: "m-1",
        role: "user",
        blocks: [{ type: "text", text: "count the nuclei" }],
      },
    ]);
  });

  it("renders nothing for the assistant message that only carries calls", () => {
    // Empty content is the chat-completions carrier for tool_calls, not silence
    // the reader should see a bubble for.
    const items = fromChatHistory([
      assistant("m-2", "", [{ id: "c1", name: "execute_code" }]),
    ]);
    expect(items.filter((i) => i.kind === "message")).toEqual([]);
    expect(calls(items)).toHaveLength(1);
  });

  it("keeps what the model said before calling a tool", () => {
    const items = fromChatHistory([
      assistant("m-2", "Let me look.", [{ id: "c1", name: "execute_code" }]),
    ]);
    expect(items[0]).toMatchObject({ kind: "message", role: "assistant" });
    expect(items[1]).toMatchObject({ kind: "tool_call", title: "execute_code" });
  });

  it("resolves a call with its result, by id", () => {
    const items = fromChatHistory([
      assistant("m-2", "", [{ id: "c1", name: "execute_code" }]),
      toolResult("m-3", "c1", "execute_code", "42"),
    ]);
    expect(calls(items)).toHaveLength(1);
    expect(calls(items)[0]).toMatchObject({
      id: "c1",
      status: "completed",
      title: "execute_code",
    });
    expect(toolText(calls(items)[0]!)).toBe("42");
  });

  it("marks a failed tool as failed rather than as an answer", () => {
    const items = fromChatHistory([
      assistant("m-2", "", [{ id: "c1", name: "nope" }]),
      toolResult("m-3", "c1", "nope", "Error: unknown tool", true),
    ]);
    expect(calls(items)[0]!.status).toBe("failed");
  });

  it("shows an unanswered call as running only while a turn is going", () => {
    const thread = [assistant("m-2", "", [{ id: "c1", name: "execute_code" }])];
    expect(calls(fromChatHistory(thread, true))[0]!.status).toBe("in_progress");
    expect(calls(fromChatHistory(thread, false))[0]!.status).toBe("pending");
  });

  it("keeps a result whose call it never saw", () => {
    // Evidence with nothing to attach to is still evidence; dropping it would
    // silently lose a tool's output.
    const items = fromChatHistory([
      toolResult("m-9", "gone", "execute_code", "orphan"),
    ]);
    expect(calls(items)).toHaveLength(1);
    expect(toolText(calls(items)[0]!)).toBe("orphan");
  });

  it("gives an image to the call that produced it, not to the user", () => {
    // The image rides the thread as role "user" because that is the only role
    // that carries one back to the provider. Rendering it as something the user
    // said would attribute a screenshot to the wrong party.
    const items = fromChatHistory([
      user("m-1", "show me"),
      assistant("m-2", "", [{ id: "c1", name: "take_screenshot" }]),
      toolResult("m-3", "c1", "take_screenshot", "captured"),
      imageCarrier("m-4", "take_screenshot", "AAAA"),
    ]);
    const userMessages = items.filter(
      (i) => i.kind === "message" && i.role === "user",
    );
    expect(userMessages).toHaveLength(1); // only "show me"
    expect(calls(items)[0]!.blocks).toEqual([
      { type: "text", text: "captured" },
      { type: "image", data: "AAAA", mime: "image/png" },
    ]);
  });

  it("gives parallel screenshots one image each", () => {
    const items = fromChatHistory([
      assistant("m-2", "", [
        { id: "c1", name: "take_screenshot" },
        { id: "c2", name: "take_screenshot" },
      ]),
      toolResult("m-3", "c1", "take_screenshot", "one"),
      toolResult("m-4", "c2", "take_screenshot", "two"),
      imageCarrier("m-5", "take_screenshot", "AAAA"),
      imageCarrier("m-6", "take_screenshot", "BBBB"),
    ]);
    const images = calls(items).map((c) =>
      c.blocks.filter((b) => b.type === "image"),
    );
    expect(images.map((g) => g.length)).toEqual([1, 1]);
  });
});

describe("groupThread", () => {
  it("collapses a whole round into one group", () => {
    const groups = groupThread(
      fromChatHistory([
        user("m-1", "go"),
        assistant("m-2", "", [
          { id: "c1", name: "find_skills" },
          { id: "c2", name: "execute_code" },
        ]),
        toolResult("m-3", "c1", "find_skills", "a"),
        toolResult("m-4", "c2", "execute_code", "b"),
        assistant("m-5", "done"),
      ]),
    );
    expect(groups.map((g) => g.kind)).toEqual(["message", "tools", "message"]);
    expect(groups[1]!.kind === "tools" && groups[1]!.calls).toHaveLength(2);
  });

  it("keeps consecutive rounds together when the model says nothing between", () => {
    // The common shape: twelve silent rounds should read as one line, not
    // twelve.
    const groups = groupThread(
      fromChatHistory([
        assistant("m-1", "", [{ id: "c1", name: "execute_code" }]),
        toolResult("m-2", "c1", "execute_code", "a"),
        assistant("m-3", "", [{ id: "c2", name: "execute_code" }]),
        toolResult("m-4", "c2", "execute_code", "b"),
      ]),
    );
    expect(groups).toHaveLength(1);
    expect(groups[0]!.kind === "tools" && groups[0]!.calls).toHaveLength(2);
  });

  it("splits a round when the model speaks in between", () => {
    const groups = groupThread(
      fromChatHistory([
        assistant("m-1", "first", [{ id: "c1", name: "execute_code" }]),
        toolResult("m-2", "c1", "execute_code", "a"),
        assistant("m-3", "second", [{ id: "c2", name: "execute_code" }]),
        toolResult("m-4", "c2", "execute_code", "b"),
      ]),
    );
    expect(groups.map((g) => g.kind)).toEqual([
      "message",
      "tools",
      "message",
      "tools",
    ]);
  });

  it("lifts images out of the fold", () => {
    // Collapsing hides walls of tool text; a screenshot is the answer, so it
    // stays visible with the group collapsed.
    const groups = groupThread(
      fromChatHistory([
        assistant("m-1", "", [{ id: "c1", name: "take_screenshot" }]),
        toolResult("m-2", "c1", "take_screenshot", "captured"),
        imageCarrier("m-3", "take_screenshot", "AAAA"),
      ]),
    );
    expect(groups[0]!.kind === "tools" && groups[0]!.images).toEqual([
      { type: "image", data: "AAAA", mime: "image/png" },
    ]);
  });

  it("leaves a plain conversation alone", () => {
    const groups = groupThread(
      fromChatHistory([user("m-1", "hi"), assistant("m-2", "hello")]),
    );
    expect(groups.map((g) => g.kind)).toEqual(["message", "message"]);
  });
});


describe("applyLiveOutput", () => {
  const running = (busy = true) =>
    fromChatHistory(
      [assistant("m-1", "", [{ id: "c1", name: "execute_code" }])],
      busy,
    );

  const live = (stdout: string) => ({
    jobId: "job-1",
    stdout,
    truncated: false,
  });

  it("shows a running cell's output on the call that is running it", () => {
    // The reason the promote window was dropped at all: a long cell has to say
    // something while it runs, or the turn is the stalled cursor it replaced.
    const items = applyLiveOutput(running(), live("step 1\n"));
    expect(toolText(calls(items)[0]!)).toBe("step 1\n");
    expect(calls(items)[0]!.live).toBe(true);
  });

  it("replaces rather than accumulates", () => {
    // `partial` is the whole buffer on every poll, not a delta. Appending would
    // repeat the output once per poll -- twice a second.
    let items = applyLiveOutput(running(), live("a\n"));
    items = applyLiveOutput(items, live("a\nb\n"));
    expect(toolText(calls(items)[0]!)).toBe("a\nb\n");
  });

  it("gives the output to the running cell, not the one queued behind it", () => {
    // A round's calls are dispatched one at a time and each result is appended
    // only when its own dispatch returns, so while the first cell runs neither
    // call has an answer and both read as in_progress. The kernel is running
    // the earlier one. Taking the last match would put a cell's output on a
    // call that has not started, and the output would jump rows when the first
    // finished.
    const items = fromChatHistory(
      [
        assistant("m-1", "", [
          { id: "c1", name: "execute_code" },
          { id: "c2", name: "execute_code" },
        ]),
      ],
      true,
    );
    applyLiveOutput(items, live("from the first cell\n"));
    const [first, second] = calls(items);
    expect(toolText(first!)).toBe("from the first cell\n");
    expect(first!.live).toBe(true);
    expect(toolText(second!)).toBe("");
    expect(second!.live).toBeUndefined();
  });

  it("moves on to the next cell once the first has answered", () => {
    // The second call is running now, and its predecessor's output is its own
    // result rather than something still streaming.
    const items = fromChatHistory(
      [
        assistant("m-1", "", [
          { id: "c1", name: "execute_code" },
          { id: "c2", name: "execute_code" },
        ]),
        toolResult("m-2", "c1", "execute_code", "first done"),
      ],
      true,
    );
    applyLiveOutput(items, live("from the second cell\n"));
    const [first, second] = calls(items);
    expect(toolText(first!)).toBe("first done");
    expect(toolText(second!)).toBe("from the second cell\n");
  });

  it("leaves a finished call alone", () => {
    // Once the cell ends, its output is the tool's own result. Attaching the
    // stale buffer too would show it twice, and mark a result as still running.
    const items = fromChatHistory([
      assistant("m-1", "", [{ id: "c1", name: "execute_code" }]),
      toolResult("m-2", "c1", "execute_code", "final"),
    ]);
    applyLiveOutput(items, live("partial"));
    expect(toolText(calls(items)[0]!)).toBe("final");
    expect(calls(items)[0]!.live).toBeUndefined();
  });

  it("ignores a call that cannot be running a cell", () => {
    // Only execute_code submits one; anything else in progress does not print.
    const items = fromChatHistory(
      [assistant("m-1", "", [{ id: "c1", name: "find_skills" }])],
      true,
    );
    applyLiveOutput(items, live("noise"));
    expect(toolText(calls(items)[0]!)).toBe("");
  });

  it("does nothing when no cell is running", () => {
    const items = applyLiveOutput(running(), null);
    expect(toolText(calls(items)[0]!)).toBe("");
    expect(calls(items)[0]!.live).toBeUndefined();
  });

  it("does not mark an empty buffer as live output", () => {
    // A cell that has printed nothing yet should read as running, not as a
    // tool that answered with silence.
    const items = applyLiveOutput(running(), live(""));
    expect(calls(items)[0]!.live).toBeUndefined();
  });
});

describe("latestLine", () => {
  it("takes the newest line", () => {
    expect(latestLine("a\nb\nc\n")).toBe("c");
  });

  it("looks past the blank line a print leaves", () => {
    expect(latestLine("working\n\n\n")).toBe("working");
  });

  it("has nothing to say about no output", () => {
    expect(latestLine("")).toBe("");
    expect(latestLine("\n\n")).toBe("");
  });
});
