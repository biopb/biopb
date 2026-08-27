// The slash commands the chat pane answers by itself.
//
// The head already has buttons for two of these. They exist anyway because the
// hands are on the keyboard: a conversation is typed, and reaching for a control
// in the corner to fold it is a break in the same motion. The third, `/context`,
// has no button because it has nothing to act on -- it is a read of what the
// model is being sent, which is the question a person asks just before deciding
// whether to compact.
//
// Parsed here rather than in the pane, for the reason `chatKeys.ts` gives: the
// workspace has no DOM test environment, so anything left inline in the
// component is untestable by construction. This one also owns its own wording,
// so the sentence a mistyped command produces is pinned by a test instead of
// living in a JSX branch nothing reaches.

import type { AgentCommand, ChatEngine } from "./chatClient";
import type { ChatMessage } from "./chatThread";

export type CommandName = "new" | "compact" | "context";

export interface CommandSpec {
  name: CommandName;
  /** How it is typed, canonically. */
  typed: string;
  /** Other spellings that mean the same thing. */
  aliases: string[];
  /** One line, for the completion row. */
  help: string;
}


export const COMMANDS: CommandSpec[] = [
  {
    name: "new",
    typed: "/new",
    // `/clear` because that is the muscle memory from every other agent CLI,
    // and the cost of honouring it is one string.
    aliases: ["/clear"],
    help: "start a new conversation",
  },
  {
    name: "compact",
    typed: "/compact",
    help: "fold the older messages into a summary",
    aliases: [],
  },
  {
    name: "context",
    typed: "/context",
    help: "what the model is being sent",
    aliases: [],
  },
];

export type Parsed =
  /** Not a command, or one only the agent knows. Send it. */
  | { kind: "send"; text: string }
  | { kind: "command"; name: CommandName }
  /** Meant as a command and is not one; *message* is what to show. */
  | { kind: "reject"; message: string };

/** The local commands under *engine*.
 *
 * Under ACP the set shrinks to `/new`. `/compact` and `/context` act on the
 * built-in loop's projection of the thread -- the summary it stands behind, and
 * a count of what it is about to send. A hosted harness manages its own context
 * and never shows us its budget, so both would be answering about a thread
 * nobody reads. `/new` survives because clearing the conversation is still
 * something biopb does: it ends the agent's session as well as the transcript.
 */
export function localCommands(engine: ChatEngine): CommandSpec[] {
  if (engine !== "acp") return COMMANDS;
  return COMMANDS.filter((c) => c.name === "new");
}

/** A bare `/word`, which is the only thing treated as a command attempt.
 *
 * The narrowness is the point. A message may legitimately start with a slash --
 * `/data/run3/stack.tif is the one I mean` -- and swallowing that as a bad
 * command, or worse as a good one, is the failure worth designing against. A
 * lone token of letters after a slash is not a path and is not prose; nothing
 * else is claimed. */
const COMMANDISH = /^\/[a-zA-Z]+$/;

const listing = (specs: CommandSpec[], agent: AgentCommand[]) =>
  specs
    .map((c) =>
      c.aliases.length ? `${c.typed} (or ${c.aliases.join(", ")})` : c.typed,
    )
    .concat(agent.map((c) => "/" + c.name))
    .join(", ");

function lookup(token: string, specs: CommandSpec[]): CommandSpec | undefined {
  const t = token.toLowerCase();
  return specs.find((c) => c.typed === t || c.aliases.includes(t));
}

/**
 * What pressing Enter on *input* should do.
 *
 * Two namespaces meet here. The local commands act on this pane and are handled
 * without a round trip; *agent* holds what a hosted harness advertised, which
 * biopb neither defines nor runs -- ACP has no method for invoking one, so the
 * command goes to the agent as ordinary prompt text and the agent parses its own
 * prefix. Local wins a collision, and there are only ever a couple of local
 * names to collide with.
 *
 * What survives from the closed set is the rejection. A name in neither
 * namespace is still refused rather than sent, because that is what keeps
 * `/data/run3/stack.tif is the one I mean` prose and stops a typo'd `/conect`
 * from becoming a prompt nobody meant to write.
 */
export function parseCommand(
  input: string,
  engine: ChatEngine = "builtin",
  agent: AgentCommand[] = [],
): Parsed {
  const text = input.trim();
  const [first, ...rest] = text.split(/\s+/);
  if (!first || !COMMANDISH.test(first)) return { kind: "send", text };
  const specs = localCommands(engine);
  // Only a hosted harness has commands of its own, and only it can parse one.
  // Gated here rather than left to the caller to pass nothing, so the answer
  // does not depend on a list happening to be empty.
  if (engine !== "acp") agent = [];
  const spec = lookup(first, specs);
  if (!spec) {
    if (agent.some((c) => "/" + c.name.toLowerCase() === first.toLowerCase())) {
      // The agent's to interpret, arguments and all: its `input.hint` exists
      // precisely because these take them.
      return { kind: "send", text };
    }
    return {
      kind: "reject",
      message: `Unknown command ${first}. Available: ${listing(specs, agent)}.`,
    };
  }
  // Rejected rather than ignored. None of the *local* ones take an argument, and
  // quietly dropping the rest of the line is how `/compact keep the segmentation
  // notes` becomes a compaction that did not keep them.
  if (rest.length) {
    return { kind: "reject", message: `${spec.typed} takes no arguments.` };
  }
  return { kind: "command", name: spec.name };
}

/** The commands *input* could still become, for the completion list.
 *
 * Empty unless the input is a command attempt, so the list is absent for every
 * ordinary message rather than flickering on the first character of one.
 *
 * One character looser than `COMMANDISH`: a bare slash matches here and not
 * there. That is the moment discovery has to happen -- someone types `/` to
 * find out what there is -- and it is still not a command, so `parseCommand`
 * goes on treating it as text. */
const PARTIAL = /^\/[a-zA-Z]*$/;

export function matchCommands(
  input: string,
  engine: ChatEngine = "builtin",
): CommandSpec[] {
  const text = input.trim();
  if (!PARTIAL.test(text)) return [];
  const t = text.toLowerCase();
  // Local commands only. The agent's own are *accepted* -- `parseCommand` sends
  // an advertised name straight through -- but not offered, because offering
  // one is a promise the pane cannot keep: biopb gives the harness an empty,
  // throwaway working directory, and a coding agent's commands are about a
  // project. opencode's three are the illustration: `/review` has no repo to
  // review, `/init` writes an AGENTS.md that dies with the temp dir, and
  // `/customize-opencode` edits config that biopb pins or that the temp dir
  // takes with it. Listing them would advertise three no-ops.
  //
  // Worth revisiting if the harness is ever given a real, persistent workspace;
  // at that point they would start meaning something and listing them would be
  // honest.
  return localCommands(engine).filter(
    (c) => c.typed.startsWith(t) || c.aliases.some((a) => a.startsWith(t)),
  );
}

/** How big the conversation has become, from what the pane already holds.
 *
 * Client-side because the pane holds the same messages the child stores, and
 * `compacted` says where the projection starts -- so the part that *grows* can
 * be measured here exactly, with no round trip and no new endpoint.
 *
 * What it deliberately does not do is guess at tokens. The system prompt and
 * the tool schemas ride every call and are not in the thread, so any total
 * would be wrong by an unknown constant; and characters-per-token is a rule of
 * thumb, not a measurement. Reported instead is the quantity that is both exact
 * and the one that moves: how much of the thread is still sent in full.
 */
export function contextReport(
  messages: ChatMessage[],
  compacted: number,
  model: string,
): string {
  const live = messages.slice(compacted);
  let chars = 0;
  let images = 0;
  let bytes = 0;
  for (const m of live) {
    chars += m.content ? m.content.length : 0;
    // Arguments are projected back with the call that carries them; a cell of
    // code is often the largest single thing in a turn.
    for (const c of m.tool_calls || []) {
      chars += c.function?.arguments?.length || 0;
    }
    if (m.image) {
      images += 1;
      bytes += Math.round((m.image.length * 3) / 4); // base64 -> bytes
    }
  }
  const folded = compacted
    ? `${compacted} of them folded into a summary`
    : "none folded";
  const pic = images ? `, ${images} ${images === 1 ? "image" : "images"} (${size(bytes)})` : "";
  return [
    `model: ${model || "unknown"}`,
    `thread: ${count(messages.length, "message")}, ${folded}`,
    `sent each turn: ${count(live.length, "message")}, ${chars.toLocaleString()} characters${pic}`,
    "The system prompt and the tool schemas add a fixed amount on top of this.",
  ].join("\n");
}

function count(n: number, noun: string): string {
  return `${n} ${noun}${n === 1 ? "" : "s"}`;
}

function size(bytes: number): string {
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} kB`;
}
