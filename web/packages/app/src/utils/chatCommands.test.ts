import { describe, expect, it } from "vitest";
import {
  COMMANDS,
  contextReport,
  matchCommands,
  modelReport,
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
    const cmd = (name: string) => ({ kind: "command", name, arg: "" });
    expect(parseCommand("/new")).toEqual(cmd("new"));
    expect(parseCommand("/clear")).toEqual(cmd("new"));
    expect(parseCommand("/compact")).toEqual(cmd("compact"));
    expect(parseCommand("/context")).toEqual(cmd("context"));
    expect(parseCommand("/model")).toEqual(cmd("model"));
  });

  it("ignores case and surrounding space", () => {
    expect(parseCommand("  /Compact  ")).toEqual({
      kind: "command",
      name: "compact",
      arg: "",
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

  it("hands the rest of the line to the command that asked for one", () => {
    expect(parseCommand("/model openai/gpt-5.5")).toEqual({
      kind: "command",
      name: "model",
      arg: "openai/gpt-5.5",
    });
    // Rejoined from the split, so extra space between the two is not a
    // different model id.
    expect(parseCommand("/model   openai/gpt-5.5")).toEqual({
      kind: "command",
      name: "model",
      arg: "openai/gpt-5.5",
    });
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

describe("modelReport", () => {
  const choice = (value: string) => ({ value, name: value });

  it("names the model and what else there is", () => {
    const out = modelReport("openai/gpt-5.5", [
      choice("openai/gpt-5.5"),
      choice("anthropic/claude-sonnet-5"),
    ]);
    expect(out).toContain("openai/gpt-5.5");
    expect(out).toContain("anthropic/claude-sonnet-5");
    expect(out).toContain("/model <name>");
  });

  it("stops short of a wall of text, and says how much it left out", () => {
    // A provider fronting several families publishes dozens, and the column is
    // narrow. Silently truncating would read as the whole list.
    const many = Array.from({ length: 30 }, (_, i) => choice(`m-${i}`));
    const out = modelReport("m-0", many);
    expect(out).not.toContain("m-29");
    expect(out).toContain("18 more");
  });

  it("does not read an unpublished list as a single-model provider", () => {
    // `GET /models` is optional in the OpenAI-compatible shape: an endpoint
    // that does not answer it still serves completions.
    const out = modelReport("test-model", []);
    expect(out).toContain("test-model");
    expect(out).toContain("publishes no list");
  });

  it("reports an unset model as unset rather than as blank", () => {
    expect(modelReport("", [])).toContain("No model is set.");
  });
});
