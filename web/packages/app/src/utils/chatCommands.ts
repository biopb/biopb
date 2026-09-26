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

import type { ModelChoice } from "./chatClient";
import type { ChatMessage } from "./chatThread";

export type CommandName = "new" | "compact" | "context" | "model";

export interface CommandSpec {
  name: CommandName;
  /** How it is typed, canonically. */
  typed: string;
  /** Other spellings that mean the same thing. */
  aliases: string[];
  /** One line, for the completion row. */
  help: string;
  /** Whether the rest of the line is the command's, rather than a mistake.
   *
   * Off by default, and the default is the careful one: a command that quietly
   * drops what follows it is how `/compact keep the segmentation notes` becomes
   * a compaction that did not keep them.
   */
  argument?: boolean;
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
  {
    name: "model",
    typed: "/model",
    help: "show or change the model",
    aliases: [],
    // Bare it reports and lists; with a name it switches. One command rather
    // than two because the list is what tells you what to type next.
    argument: true,
  },
];

export type Parsed =
  /** Not a command. Send it. */
  | { kind: "send"; text: string }
  | { kind: "command"; name: CommandName; arg: string }
  /** Meant as a command and is not one; *message* is what to show. */
  | { kind: "reject"; message: string };

/** What a bare `/model` answers: what is answering, and what else could.
 *
 * The list is never ours: an OpenAI-compatible provider publishes `GET
 * /models`, and it is an optional route. Empty therefore means "this provider
 * does not publish a catalogue", never "there is one model".
 */
export function modelReport(current: string, choices: ModelChoice[]): string {
  const lines = [current ? `model: ${current}` : "No model is set."];
  if (choices.length) {
    const names = choices.map((c) => c.value);
    const shown = names.slice(0, MODEL_LIST_MAX);
    lines.push(`offered: ${shown.join(", ")}`);
    if (names.length > shown.length) {
      lines.push(`...and ${names.length - shown.length} more.`);
    }
  } else {
    lines.push("This provider publishes no list; any id it knows will do.");
  }
  lines.push("/model <name> switches, and keeps the conversation.");
  return lines.join("\n");
}

/** Enough to choose from, short of a wall of text in a narrow column. A
 * provider fronting several families publishes dozens. */
const MODEL_LIST_MAX = 12;

/** A bare `/word`, which is the only thing treated as a command attempt.
 *
 * The narrowness is the point. A message may legitimately start with a slash --
 * `/data/run3/stack.tif is the one I mean` -- and swallowing that as a bad
 * command, or worse as a good one, is the failure worth designing against. A
 * lone token of letters after a slash is not a path and is not prose; nothing
 * else is claimed. */
const COMMANDISH = /^\/[a-zA-Z]+$/;

const listing = (specs: CommandSpec[]) =>
  specs
    .map((c) =>
      c.aliases.length ? `${c.typed} (or ${c.aliases.join(", ")})` : c.typed,
    )
    .join(", ");

function lookup(token: string, specs: CommandSpec[]): CommandSpec | undefined {
  const t = token.toLowerCase();
  return specs.find((c) => c.typed === t || c.aliases.includes(t));
}

/**
 * What pressing Enter on *input* should do.
 *
 * The commands act on this pane and are handled without a round trip. A name
 * that is not one of them is refused rather than sent, because that is what
 * keeps `/data/run3/stack.tif is the one I mean` prose and stops a typo'd
 * `/conect` from becoming a prompt nobody meant to write.
 */
export function parseCommand(input: string): Parsed {
  const text = input.trim();
  const [first, ...rest] = text.split(/\s+/);
  if (!first || !COMMANDISH.test(first)) return { kind: "send", text };
  const spec = lookup(first, COMMANDS);
  if (!spec) {
    return {
      kind: "reject",
      message: `Unknown command ${first}. Available: ${listing(COMMANDS)}.`,
    };
  }
  // Rejected rather than ignored, for every command that does not declare one:
  // quietly dropping the rest of the line is how `/compact keep the
  // segmentation notes` becomes a compaction that did not keep them.
  if (rest.length && !spec.argument) {
    return { kind: "reject", message: `${spec.typed} takes no arguments.` };
  }
  // Rejoined from the split rather than sliced out of the input, so `/model
  // openai/gpt-5` and `/model   openai/gpt-5` are the same command.
  return { kind: "command", name: spec.name, arg: rest.join(" ") };
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

export function matchCommands(input: string): CommandSpec[] {
  const text = input.trim();
  if (!PARTIAL.test(text)) return [];
  const t = text.toLowerCase();
  return COMMANDS.filter(
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
