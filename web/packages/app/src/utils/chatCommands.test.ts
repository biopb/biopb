import { describe, expect, it } from "vitest";
import {
  COMMANDS,
  contextReport,
  matchCommands,
  parseCommand,
} from "./chatCommands";
import type { ChatMessage } from "./chatThread";

const msg = (over: Partial<ChatMessage> = {}): ChatMessage => ({
  id: "m-1",
  role: "user",
  content: "",
  ...over,
});

describe("parseCommand", () => {
  it("recognises each command, and the alias", () => {
    expect(parseCommand("/new")).toEqual({ kind: "command", name: "new" });
    expect(parseCommand("/clear")).toEqual({ kind: "command", name: "new" });
    expect(parseCommand("/compact")).toEqual({ kind: "command", name: "compact" });
    expect(parseCommand("/context")).toEqual({ kind: "command", name: "context" });
  });

  it("ignores case and surrounding space", () => {
    expect(parseCommand("  /Compact  ")).toEqual({
      kind: "command",
      name: "compact",
    });
  });

  it("sends a message that merely starts with a path", () => {
    // The case this parser is narrow for. A path is a plausible opening for a
    // real question, and reading one as a command -- known or unknown -- takes
    // the message away from the person who typed it.
    const text = "/data/run3/stack.tif is the one I mean";
    expect(parseCommand(text)).toEqual({ kind: "send", text });
    expect(parseCommand("/data/run3/stack.tif")).toEqual({
      kind: "send",
      text: "/data/run3/stack.tif",
    });
  });

  it("names the alternatives when the command does not exist", () => {
    const p = parseCommand("/compct");
    expect(p.kind).toBe("reject");
    if (p.kind !== "reject") return;
    expect(p.message).toContain("/compct");
    expect(p.message).toContain("/compact");
    expect(p.message).toContain("/clear");
  });

  it("refuses arguments rather than dropping them", () => {
    // Silently ignoring the rest of the line is how `/compact keep the notes`
    // becomes a compaction that did not keep them.
    const p = parseCommand("/compact keep the segmentation notes");
    expect(p.kind).toBe("reject");
    if (p.kind !== "reject") return;
    expect(p.message).toContain("no arguments");
  });

  it("treats ordinary prose as a message", () => {
    expect(parseCommand("what shape is the stack?")).toEqual({
      kind: "send",
      text: "what shape is the stack?",
    });
  });
});

describe("matchCommands", () => {
  it("offers the whole list on a bare slash", () => {
    expect(matchCommands("/")).toHaveLength(COMMANDS.length);
  });

  it("narrows by prefix, and matches an alias", () => {
    expect(matchCommands("/co").map((c) => c.name)).toEqual([
      "compact",
      "context",
    ]);
    expect(matchCommands("/cl").map((c) => c.name)).toEqual(["new"]);
  });

  it("offers nothing for a message", () => {
    // Otherwise the list flickers into view on the first character of one.
    expect(matchCommands("what shape is it?")).toEqual([]);
    expect(matchCommands("/data/run3")).toEqual([]);
    expect(matchCommands("")).toEqual([]);
  });
});

describe("contextReport", () => {
  it("counts only what is still sent in full", () => {
    const messages = [
      msg({ id: "m-1", content: "aaaa" }),
      msg({ id: "m-2", content: "bbbb" }),
      msg({ id: "m-3", content: "cc" }),
    ];
    const out = contextReport(messages, 2, "claude-sonnet-5");
    expect(out).toContain("3 messages");
    expect(out).toContain("2 of them folded");
    expect(out).toContain("1 message,");
    expect(out).toContain("2 characters"); // m-3 alone
  });

  it("counts a call's arguments, which ride back with it", () => {
    // Often the largest single thing in a turn: a cell of code.
    const messages = [
      msg({
        id: "m-1",
        role: "assistant",
        tool_calls: [{ function: { name: "run_code", arguments: "0123456789" } }],
      }),
    ];
    expect(contextReport(messages, 0, "m")).toContain("10 characters");
  });

  it("reports images apart from text, and omits them when there are none", () => {
    const messages = [msg({ id: "m-1", image: "a".repeat(4096), mime: "image/png" })];
    expect(contextReport(messages, 0, "m")).toContain("1 image (3 kB)");
    expect(contextReport([msg()], 0, "m")).not.toContain("image");
  });

  it("says so when nothing is folded", () => {
    expect(contextReport([msg()], 0, "m")).toContain("none folded");
  });

  it("does not present a token total it cannot know", () => {
    // The system prompt and tool schemas are not in the thread, so any total
    // would be wrong by an unknown constant. Said out loud instead.
    const out = contextReport([msg()], 0, "claude-sonnet-5");
    expect(out).toContain("system prompt");
    expect(out).not.toMatch(/token/i);
  });
});
