// Every network call the chat pane makes, in one place.
//
// Not because there are many, but because the transport is the part that
// changes: the pane polls the session child today. Kept out of the component
// so that a swap is this file, not a re-wiring of the view.
//
// Two roots, following the child's split: the reads are `/api/*`, which the
// control always proxies, and the writes are `/chat/*`, which it proxies only
// when it is loopback-bound. A write therefore 404s on a control that will not
// serve it, which is why the pane gates on `chatProxied()` as well.
//
// Everything goes through `sessionFetch`: both roots are behind the control's
// auth gate, and a token is optional rather than absent on a loopback control
// (biopb#468). Calling bare is what left the observe page inert under `--token`
// for as long as it did (biopb#730), and it failed silently there because a 401
// body parses as JSON -- which is exactly how these readers would fail too.

import { sessionFetch } from "./sessionFetch";
import type { ChatMessage, LiveOutput } from "./chatThread";

export interface ChatStatus {
  enabled: boolean;
  ready: boolean;
  /** Why chat cannot run, when it cannot — an unset API key, typically. */
  reason: string | null;
  /** Who is answering: the model id the loop sends the provider. */
  model: string;
  /** How many leading messages the model now sees only as a summary. The pane
   * renders all of them regardless, so this is the only sign compaction
   * happened. Zero on an older child, which never folds anything. */
  compacted: number;
}

export interface HistoryPage {
  messages: ChatMessage[];
  /** Who is answering, as of this read. Carried here rather than on the
   * once-probed status because `/model` is session state: it is not persisted,
   * it reaches every window, and this is the only read the pane repeats. */
  model: string;
  /** Whether this page is the whole thread rather than a delta — a cursor the
   * child did not recognise, which after a reset is every other window's. */
  full: boolean;
  busy: boolean;
  /** The cell being polled right now, and what it has printed. */
  live: LiveOutput | null;
}

/**
 * Whether chat is configured on this session child, or null if unreachable.
 *
 * Null rather than a default, because the caller must not read "unreachable" as
 * "off": the pane would unmount and take a half-typed message with it. Static
 * for the life of the process, so it is probed once.
 */
export async function fetchChatStatus(base: string): Promise<ChatStatus | null> {
  try {
    const r = await sessionFetch(base + "/api/chat/status");
    if (!r.ok) return null;
    const j = await r.json();
    return {
      enabled: !!j.enabled,
      ready: !!j.ready,
      reason: typeof j.reason === "string" ? j.reason : null,
      model: typeof j.model === "string" ? j.model : "",
      // Read here or it does not exist: this builds the status field by field
      // rather than returning the body, so a key the type declares and the
      // parser drops is invisible on both sides of the seam.
      compacted: typeof j.compacted === "number" ? j.compacted : 0,
    };
  } catch {
    return null;
  }
}

/** One model the provider offers. */
export interface ModelChoice {
  value: string;
  name: string;
}

/** What the loop can be pointed at, and what it is pointed at now.
 *
 * Read when `/model` is typed rather than polled: the list moves only when the
 * session does. An empty `choices` is a provider that publishes no catalogue --
 * not a failed read, which is null.
 */
export async function fetchModels(
  base: string,
): Promise<{ model: string; choices: ModelChoice[] } | null> {
  try {
    const r = await sessionFetch(base + "/api/chat/models");
    if (!r.ok) return null;
    const j = await r.json();
    const raw = Array.isArray(j.choices) ? j.choices : [];
    return {
      model: typeof j.model === "string" ? j.model : "",
      choices: raw
        .filter((c: unknown) => c && typeof (c as ModelChoice).value === "string")
        .map((c: ModelChoice) => ({ value: c.value, name: c.name || c.value })),
    };
  } catch {
    return null;
  }
}

/** Point the loop at *model*. Returns an error to show, or null. */
export async function setModel(
  base: string,
  model: string,
): Promise<string | null> {
  let r: Response;
  try {
    r = await sessionFetch(base + "/chat/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    });
  } catch (e) {
    return String(e);
  }
  if (r.ok) return null;
  const d = await r.json().catch(() => ({}) as Record<string, unknown>);
  if (r.status === 409) return "A turn is running. Wait for it, or cancel it.";
  return String(d.error || `could not switch model (${r.status})`);
}

/** The conversation after *cursor*, or all of it when the child does not
 * recognise it. Null on a failed read, so the pane keeps what it has.
 *
 * *cursor* is the last-seen message id. */
export async function fetchHistory(
  base: string,
  cursor: string | null,
): Promise<HistoryPage | null> {
  const q = cursor ? "?after=" + encodeURIComponent(cursor) : "";
  try {
    const r = await sessionFetch(base + "/api/chat/history" + q);
    if (!r.ok) return null;
    const j = await r.json();
    return {
      messages: Array.isArray(j.messages) ? j.messages : [],
      // Absent on an older child, where every page was effectively a delta.
      full: !!j.full,
      busy: !!j.busy,
      // Empty on an older child, which the pane reads as "keep what you have"
      // rather than as "no model is set".
      model: typeof j.model === "string" ? j.model : "",
      live: readLive(j.partial),
    };
  } catch {
    return null;
  }
}

/** `partial` off the history read, or null when no cell is running.
 *
 * The child sends `null` between cells and omits nothing, but this is parsed
 * defensively like the rest: a degraded payload must read as "nothing running"
 * rather than throw inside a poll the pane depends on. */
function readLive(raw: unknown): LiveOutput | null {
  if (!raw || typeof raw !== "object") return null;
  const p = raw as Record<string, unknown>;
  if (typeof p.job_id !== "string" || typeof p.stdout !== "string") return null;
  return {
    jobId: p.job_id,
    stdout: p.stdout,
    truncated: !!p.truncated,
    // What the cell has printed in total, which is more than `stdout` once the
    // buffer is tail-capped. Defaulted rather than required: an older child
    // sends no such field, and that must read as "nothing was dropped".
    stdoutLen: typeof p.stdout_len === "number" ? p.stdout_len : p.stdout.length,
  };
}

/** Start a turn. Returns an error to show, or null when it was accepted.
 *
 * A 409 is state, not a failed action, so it comes back as prose about waiting
 * rather than about retrying. */
export async function sendTurn(
  base: string,
  text: string,
): Promise<string | null> {
  let r: Response;
  try {
    r = await sessionFetch(base + "/chat/turn", {
      method: "POST",
      // Required by the child on this root: a JSON content-type is one a
      // cross-site form POST cannot set, and this route reaches a kernel.
      // `sessionFetch` adds the bearer token alongside it.
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  } catch (e) {
    return String(e);
  }
  if (r.ok || r.status === 202) return null;
  const d = await r.json().catch(() => ({}) as Record<string, unknown>);
  if (r.status === 409) return "A turn is already running. Wait for it, or cancel it.";
  return String(d.error || `send failed (${r.status})`);
}

/** Fold the older part of the thread into a summary. Returns an error, or null.
 *
 * Projection only: the pane still renders every message. What changes is what
 * the model is given, which is why the result is reported through `compacted`
 * on the status read rather than by anything appearing in the thread.
 */
export async function compactThread(base: string): Promise<string | null> {
  let r: Response;
  try {
    r = await sessionFetch(base + "/chat/summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
  } catch (e) {
    return String(e);
  }
  if (r.ok) return null;
  if (r.status === 409) return "A turn is running. Wait for it, or cancel it.";
  const d = await r.json().catch(() => ({}) as Record<string, unknown>);
  return String(d.error || `compact failed (${r.status})`);
}

/** Start a new conversation. Returns an error string, or null.
 *
 * Refused with 409 while a turn is in flight: a cleared thread that the running
 * turn then appends the rest of its round into is an assistant turn whose calls
 * have no history behind them, which fails at the provider on every later turn.
 */
export async function resetThread(base: string): Promise<string | null> {
  let r: Response;
  try {
    r = await sessionFetch(base + "/chat/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
  } catch (e) {
    return String(e);
  }
  if (r.ok) return null;
  if (r.status === 409) return "A turn is running. Cancel it first.";
  const d = await r.json().catch(() => ({}) as Record<string, unknown>);
  return String(d.error || `reset failed (${r.status})`);
}

/** Stop the running turn. Nothing to report: cancelling nothing is a success,
 * and what actually happened arrives in the thread on the next poll. */
export async function cancelTurn(base: string): Promise<void> {
  try {
    await sessionFetch(base + "/chat/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    /* the next poll shows whether the turn is still running */
  }
}
