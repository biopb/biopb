// Every network call the chat pane makes, in one place.
//
// Not because there are many, but because the transport is the part that
// changes: the pane polls the session child today and is meant to hold a
// notification stream from an ACP agent later. Kept out of the component so
// that swap is this file, not a re-wiring of the view.
//
// Two roots, following the child's split: the reads are `/api/*`, which the
// control always proxies, and the writes are `/chat/*`, which it proxies only
// when it is loopback-bound. A write therefore 404s on a control that will not
// serve it, which is why the pane gates on `localRootsProxied()` as well.
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
  model: string;
}

export interface HistoryPage {
  messages: ChatMessage[];
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
    };
  } catch {
    return null;
  }
}

/** The conversation after *after*, or all of it when *after* is unknown to the
 * child. Null on a failed read, so the pane keeps what it has. */
export async function fetchHistory(
  base: string,
  after: string | null,
): Promise<HistoryPage | null> {
  const q = after ? "?after=" + encodeURIComponent(after) : "";
  try {
    const r = await sessionFetch(base + "/api/chat/history" + q);
    if (!r.ok) return null;
    const j = await r.json();
    return {
      messages: Array.isArray(j.messages) ? j.messages : [],
      busy: !!j.busy,
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
  return { jobId: p.job_id, stdout: p.stdout, truncated: !!p.truncated };
}

/** Start a turn. Returns an error to show, or null when it was accepted.
 *
 * A 409 is state, not a failed action — the same way the console reports a busy
 * kernel — so it comes back as prose about waiting rather than about retrying. */
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
