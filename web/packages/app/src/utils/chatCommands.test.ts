import { describe, expect, it } from "vitest";
import {
  COMMANDS,
  acpContextReport,
  contextReport,
  localCommands,
  matchCommands,
  modelReport,
  parseCommand,
} from "./chatCommands";
import type { AgentCommand, ContextUsage } from "./chatClient";
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

const advertised: AgentCommand[] = [
  { name: "review", description: "Review the diff", hint: "path" },
  { name: "init", description: "Write an agent file", hint: "" },
];

describe("the two command namespaces", () => {
  it("drops /compact but keeps /context under a hosted harness", () => {
    // /compact folds the built-in loop's projection of the thread, which the
    // harness never reads -- and ACP has no compaction method, so there is
    // nothing to forward it to either. /context survives because the question
    // is still the user's to ask and the agent answers it itself, and /model
    // because both engines can be moved off the model they started on.
    expect(localCommands("acp").map((c) => c.typed)).toEqual([
      "/new",
      "/context",
      "/model",
    ]);
    expect(localCommands("builtin")).toEqual(COMMANDS);
  });

  it("sends an advertised command through as text, arguments and all", () => {
    // ACP has no method for invoking one: the agent parses its own prefix, and
    // its `input.hint` exists precisely because these take arguments.
    expect(parseCommand("/review src/x.ts", "acp", advertised)).toEqual({
      kind: "send",
      text: "/review src/x.ts",
    });
  });

  it("still refuses a name in neither namespace", () => {
    // What keeps a typo'd /conect from silently becoming a prompt.
    const parsed = parseCommand("/conect", "acp", advertised);
    expect(parsed.kind).toBe("reject");
    if (parsed.kind === "reject") expect(parsed.message).toContain("/review");
  });

  it("resolves a collision to the local command", () => {
    const withNew: AgentCommand[] = [
      { name: "new", description: "the agent's own", hint: "" },
    ];
    expect(parseCommand("/new", "acp", withNew)).toEqual({
      kind: "command",
      name: "new",
      arg: "",
    });
  });

  it("does not offer the agent's commands at all", () => {
    // Accepted but not advertised. biopb gives the harness an empty throwaway
    // cwd, and a coding agent's commands are about a project: /review has no
    // repo, /init writes a file that dies with the temp dir. Listing them would
    // advertise no-ops.
    expect(matchCommands("/rev", "acp")).toEqual([]);
    expect(matchCommands("/", "acp").map((c) => c.typed)).toEqual([
      "/new",
      "/context",
      "/model",
    ]);
  });

  it("leaves a path that starts with a slash as prose", () => {
    expect(parseCommand("/data/run3/stack.tif is the one", "acp", advertised)).toEqual(
      { kind: "send", text: "/data/run3/stack.tif is the one" },
    );
  });
});

describe("acpContextReport", () => {
  const usage = (over: Partial<ContextUsage> = {}): ContextUsage => ({
    used: 12000,
    size: 200000,
    cost: 0,
    ...over,
  });

  it("reports the agent's own numbers, not an estimate", () => {
    const out = acpContextReport(usage(), "opencode");
    expect(out).toContain("12,000 of 200,000 tokens (6%)");
  });

  it("says what to do about a full context, and it is not /compact", () => {
    const out = acpContextReport(usage(), "opencode");
    expect(out).toContain("/new");
    expect(out).not.toContain("/compact");
  });

  it("shows a genuine zero cost rather than hiding it", () => {
    // A subscription model really does report 0; hiding it would read as "not
    // measured", which is a different claim.
    expect(acpContextReport(usage({ cost: 0 }), "opencode")).toContain("$0.0000");
  });

  it("omits cost the agent did not price", () => {
    expect(acpContextReport(usage({ cost: null }), "opencode")).not.toContain("$");
  });

  it("copes with a size the agent did not give", () => {
    const out = acpContextReport(usage({ size: null }), "opencode");
    expect(out).toContain("12,000 tokens");
    expect(out).not.toContain("%");
  });

  it("says so before the agent has reported anything", () => {
    // Rather than rendering "0 tokens", which is a measurement, not a silence.
    expect(acpContextReport(null, "opencode")).toContain("not reported");
    expect(acpContextReport(usage({ used: null }), "opencode")).toContain(
      "not reported",
    );
  });
});

describe("modelReport", () => {
  const choice = (value: string) => ({ value, name: value });

  it("names the model and what else there is", () => {
    const out = modelReport("openai/gpt-5.5", [
      choice("openai/gpt-5.5"),
      choice("anthropic/claude-sonnet-5"),
    ], "acp");
    expect(out).toContain("openai/gpt-5.5");
    expect(out).toContain("anthropic/claude-sonnet-5");
    expect(out).toContain("/model <name>");
  });

  it("stops short of a wall of text, and says how much it left out", () => {
    // A harness fronting several providers advertises dozens, and the column
    // is narrow. Silently truncating would read as the whole list.
    const many = Array.from({ length: 30 }, (_, i) => choice(`m-${i}`));
    const out = modelReport("m-0", many, "acp");
    expect(out).not.toContain("m-29");
    expect(out).toContain("18 more");
  });

  it("does not read an unpublished list as a single-model provider", () => {
    // `GET /models` is optional in the OpenAI-compatible shape: an endpoint
    // that does not answer it still serves completions.
    const out = modelReport("test-model", [], "builtin");
    expect(out).toContain("test-model");
    expect(out).toContain("publishes no list");
  });

  it("says a harness has not listed yet, rather than that it has none", () => {
    // Empty here is almost always "not started": the agent states its models
    // when a session opens, and biopb opens one only when it is needed.
    expect(modelReport("m", [], "acp")).toContain("once it is running");
  });

  it("reports an unset model as unset rather than as blank", () => {
    expect(modelReport("", [], "builtin")).toContain("No model is set.");
  });
});
