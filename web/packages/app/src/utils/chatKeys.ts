// What a keystroke means in the chat pane.
//
// Extracted because the pane has no test environment — vitest runs in `node`,
// with no DOM — so anything left inline in a `onKeyDown` or a window listener
// is untestable by construction. These are the two decisions worth being sure
// about: one is the only way to send a message, and the other is the only way
// to stop a turn.

/** Whether this keydown in the composer should send.
 *
 * Enter sends, Shift+Enter is a newline — the opposite of the console below,
 * deliberately: that one is code, where a newline is the common keystroke and
 * running is the rare one.
 *
 * `isComposing` is the one that is easy to miss. An IME (Chinese, Japanese,
 * Korean) uses Enter to *commit the candidate* it is showing, and that keydown
 * reaches the textarea like any other. Without this guard the first Enter of
 * every composed phrase sends the message instead — half a sentence, with the
 * part still in the IME buffer lost.
 */
export function sendsOnEnter(e: {
  key: string;
  shiftKey: boolean;
  isComposing?: boolean;
}): boolean {
  return e.key === "Enter" && !e.shiftKey && !e.isComposing;
}

/** What Escape does, given what is on screen.
 *
 * Escape is bound on the window rather than the composer: a reader who clicked
 * a job row to watch its output would otherwise find the key silently stops
 * working, which is worse than never having had it.
 *
 * The order is dismiss-innermost-first, with two things Escape is already
 * spoken for:
 *
 * - **Composing.** An IME candidate window takes Escape to dismiss itself. It
 *   never reaches us as a cancel, and treating it as one would kill a turn
 *   because someone changed their mind about a word.
 * - **The console.** Escape means something editor-ish to anyone with the
 *   muscle memory, and spending it on a chat turn in the *other* column is a
 *   surprise. The console does not bind it today; this leaves it free to.
 */
export type EscAction =
  | "close-image"
  | "refuse-permission"
  | "cancel-turn"
  | "none";

export function escAction(state: {
  composing: boolean;
  imageOpen: boolean;
  inConsole: boolean;
  busy: boolean;
  permissionOpen?: boolean;
}): EscAction {
  if (state.composing) return "none";
  if (state.imageOpen) return "close-image";
  if (state.inConsole) return "none";
  // Ahead of the cancel, and the reason is the dismiss-innermost rule rather
  // than an exception to it: a question the agent is blocked on is the
  // innermost thing on screen. It is also the kinder reading of the keypress --
  // "no, don't do that" refuses one action, where a cancel throws away the
  // whole turn that led to it.
  if (state.permissionOpen) return "refuse-permission";
  return state.busy ? "cancel-turn" : "none";
}

/** Whether this keydown in the composer walks the prompt buffer, and which way.
 *
 * The composer is a textarea, so the arrows already mean something: move the
 * caret. Taking them outright would make a two-line draft uneditable. So the
 * walk gets them only where the caret motion is a no-op anyway -- Up on the
 * first line, Down on the last -- which for a one-line prompt, the shape almost
 * every prompt has, is everywhere in it.
 *
 * That rule is also what keeps a walk through multi-line entries sane. Setting
 * the composer's value leaves the caret at the end, so a recalled one-line
 * prompt is still on both its first and last line and the next press keeps
 * walking; a recalled multi-line one puts the caret on its last line, where Up
 * edits it instead of walking past it.
 *
 * Three things are already spoken for:
 *
 * - **Modifiers.** Shift extends a selection, Alt/Ctrl/Cmd are word- and
 *   line-wise motions in every text field on the platform.
 * - **A selection.** An arrow with text selected collapses it, and the reader
 *   holding a selection is working on the draft, not leaving it.
 * - **Composing.** An IME candidate list is driven by the arrow keys. Walking
 *   there would replace the whole composer mid-word.
 */
export type RecallKey = "older" | "newer" | "none";

export function recallAction(e: {
  key: string;
  shiftKey: boolean;
  altKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  isComposing?: boolean;
  value: string;
  selectionStart: number;
  selectionEnd: number;
}): RecallKey {
  if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return "none";
  if (e.shiftKey || e.altKey || e.ctrlKey || e.metaKey) return "none";
  if (e.isComposing) return "none";
  if (e.selectionStart !== e.selectionEnd) return "none";
  if (e.key === "ArrowUp")
    return e.value.lastIndexOf("\n", e.selectionStart - 1) === -1
      ? "older"
      : "none";
  return e.value.indexOf("\n", e.selectionStart) === -1 ? "newer" : "none";
}
