// The composer's memory of what has already been asked.
//
// A bounded list of past prompts and a cursor into it, kept pure for the same
// reason chatKeys.ts is: the pane has no DOM test environment, so a rule left
// inline in an `onKeyDown` is a rule nothing can check. Which arrow presses
// mean "walk" rather than "move the caret" lives there; this is what the walk
// lands on.

import type { Block, ThreadItem } from "./chatThread";

/** How far back the buffer remembers. Well past what anyone reaches by arrow
 * key, and small enough that the cost never needs thinking about. */
export const MAX_PROMPTS = 100;

export interface PromptHistory {
  /** Oldest first. */
  entries: string[];
  /** How far back the composer is showing: 0 is the newest entry, `null` is the
   * draft -- nothing is being recalled. */
  at: number | null;
  /** What was in the composer when the walk started. Walking forward off the
   * newest entry gives it back, so starting a walk cannot cost a half-typed
   * message. */
  draft: string;
}

export const noHistory: PromptHistory = { entries: [], at: null, draft: "" };

/** Record a prompt that has just gone out, and end any walk in progress.
 *
 * A repeat of the newest entry is dropped rather than stacked: re-running one
 * prompt several times is common -- `/context` between turns is the reason --
 * and a buffer that keeps all of them makes the next one further back, not
 * closer. Consecutive repeats only; an older duplicate stays where it was said,
 * so a walk reaches the same prompts in the order they were used.
 */
export function remember(h: PromptHistory, prompt: string): PromptHistory {
  const body = prompt.trim();
  if (!body) return { entries: h.entries, at: null, draft: "" };
  const repeat = h.entries[h.entries.length - 1] === body;
  return {
    entries: repeat ? h.entries : [...h.entries, body].slice(-MAX_PROMPTS),
    at: null,
    draft: "",
  };
}

/** Fill an untouched buffer from prompts sent before it existed.
 *
 * Only while it is still empty, which is what makes this safe to call on every
 * keystroke that starts a walk: once anything has been remembered the thread is
 * already folded in, and seeding again would put the same prompts in twice.
 */
export function seed(h: PromptHistory, prompts: string[]): PromptHistory {
  if (h.entries.length) return h;
  return prompts.reduce((acc, p) => remember(acc, p), h);
}

export type Recall = "older" | "newer";

/** One step of a walk, or null when the keystroke has nothing to show: an empty
 * buffer, the oldest entry already on screen, or a forward step while the draft
 * is what is showing. Null means leave the composer exactly as it is, which is
 * also what lets the caller hand the keystroke back to the textarea.
 *
 * `current` is the text on screen, stashed as the draft when a walk starts.
 */
export function recall(
  h: PromptHistory,
  dir: Recall,
  current: string,
): { history: PromptHistory; text: string } | null {
  const n = h.entries.length;
  const entry = (at: number) => h.entries[n - 1 - at]!;
  if (dir === "older") {
    if (!n) return null;
    if (h.at === null)
      return { history: { ...h, at: 0, draft: current }, text: entry(0) };
    const at = h.at + 1;
    if (at >= n) return null;
    return { history: { ...h, at }, text: entry(at) };
  }
  if (h.at === null) return null;
  const at = h.at - 1;
  if (at < 0) return { history: { ...h, at: null, draft: "" }, text: h.draft };
  return { history: { ...h, at }, text: entry(at) };
}

/** Typing ends the walk: what is in the composer is the reader's own text now,
 * so the next backward step starts again from the newest entry rather than from
 * wherever the walk had got to. The stashed draft goes with it -- it was the
 * thing the walk replaced, and the edit is what replaced it. */
export function abandon(h: PromptHistory): PromptHistory {
  return h.at === null ? h : { ...h, at: null, draft: "" };
}

/** The prompts a thread already shows, oldest first. Both engines reduce to the
 * same `ThreadItem[]`, so this reads either one. */
export function pastPrompts(thread: ThreadItem[]): string[] {
  const out: string[] = [];
  for (const item of thread) {
    if (item.kind !== "message" || item.role !== "user") continue;
    const text = item.blocks
      .flatMap((b: Block) => (b.type === "text" ? [b.text] : []))
      .join("")
      .trim();
    if (text) out.push(text);
  }
  return out;
}
