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
import type { AcpItem, ChatMessage, LiveOutput } from "./chatThread";

/** Which agent drives the pane. `builtin` is the in-process loop; `acp` is a
 * coding harness the user already runs, hosted over the Agent Client Protocol. */
export type ChatEngine = "builtin" | "acp";

/** One engine's availability, for the switcher. A view offering the choice has
 * to be able to grey out the half that cannot run and say why. */
export interface EngineRow {
  engine: ChatEngine;
  ready: boolean;
  reason: string | null;
}

export interface ChatStatus {
  enabled: boolean;
  ready: boolean;
  /** Why chat cannot run, when it cannot — an unset API key, typically. */
  reason: string | null;
  engine: ChatEngine;
  engines: EngineRow[];
  /** Who is answering: a model id under `builtin`, a harness name under `acp`.
   * One field because to a reader they are one fact. */
  model: string;
  /** How many leading messages the model now sees only as a summary. The pane
   * renders all of them regardless, so this is the only sign compaction
   * happened. Zero on an older child, which never folds anything. */
  compacted: number;
}

/** A slash command the agent advertises, which biopb neither defines nor runs.
 *
 * They arrive by notification and can change mid-session, which is why they
 * ride the polled history read rather than the once-probed status. */
export interface AgentCommand {
  name: string;
  description: string;
  hint: string;
}

/** What the agent says its context holds. ACP's `usage_update`, which is the
 * agent's own accounting rather than anything the pane could estimate. */
export interface ContextUsage {
  used: number | null;
  size: number | null;
  cost: number | null;
}

export interface HistoryPage {
  /** The built-in loop's thread. Empty under the ACP engine. */
  messages: ChatMessage[];
  /** The ACP thread. Null unless the child is running that engine — which is
   * how a reader tells the two apart without asking, and why the child sends
   * one key or the other rather than both. */
  items: AcpItem[] | null;
  /** The revision watermark to poll from next, under the ACP engine. */
  rev: number | null;
  /** Whether this page is the whole thread rather than a delta — a cursor the
   * child did not recognise, which after a reset is every other window's. */
  full: boolean;
  busy: boolean;
  /** What the agent says it can be asked to do. Empty under `builtin`. */
  commands: AgentCommand[];
  /** The agent's own context accounting. Null under `builtin`, and until the
   * agent has reported once. */
  usage: ContextUsage | null;
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
      // Defaulted to the built-in loop, not to nothing: an older child that
      // does not send this field is one that only has that engine.
      engine: j.engine === "acp" ? "acp" : "builtin",
      engines: Array.isArray(j.engines) ? j.engines.map(readEngineRow) : [],
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

function readEngineRow(raw: unknown): EngineRow {
  const r = (raw ?? {}) as Record<string, unknown>;
  return {
    engine: r.engine === "acp" ? "acp" : "builtin",
    ready: !!r.ready,
    reason: typeof r.reason === "string" ? r.reason : null,
  };
}

/** Who is driving the pane right now, or null when the read failed.
 *
 * Read before every history read rather than taken from the once-probed status:
 * the engine is session state, and the window that switched it is not
 * necessarily this one. A pane that missed the switch renders the outgoing
 * engine's thread forever -- it holds both, and picks by an `engine` its own
 * click is the only thing that ever moved.
 */
export async function fetchEngine(
  base: string,
): Promise<{ engine: ChatEngine; model: string } | null> {
  try {
    const r = await sessionFetch(base + "/api/chat/engine");
    if (!r.ok) return null;
    const j = await r.json();
    return {
      engine: j.engine === "acp" ? "acp" : "builtin",
      model: typeof j.model === "string" ? j.model : "",
    };
  } catch {
    return null;
  }
}

/** One model the engine offers. */
export interface ModelChoice {
  value: string;
  name: string;
}

/** What the engine can be pointed at, and what it is pointed at now.
 *
 * Read when `/model` is typed rather than polled: the list moves only when the
 * session does. An empty `choices` is the built-in loop, which has no list to
 * offer -- not a failed read, which is null.
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

/** Point the engine at *model*. Returns an error to show, or null.
 *
 * The interesting failure is a 400 naming what the harness does offer: the
 * check happens against the list the agent advertised, so a typo is answered
 * here rather than by a turn that fails at the provider.
 */
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
 * *cursor* is a last-seen message id under the built-in loop and a revision
 * watermark under ACP — two engines, two ways of saying "what I already have",
 * because only one of them can express an item that changed in place. The pane
 * holds whichever the child last gave it and passes it back unread. */
export async function fetchHistory(
  base: string,
  cursor: string | number | null,
): Promise<HistoryPage | null> {
  const q =
    cursor === null || cursor === ""
      ? ""
      : typeof cursor === "number"
        ? "?since=" + cursor
        : "?after=" + encodeURIComponent(cursor);
  try {
    const r = await sessionFetch(base + "/api/chat/history" + q);
    if (!r.ok) return null;
    const j = await r.json();
    return {
      messages: Array.isArray(j.messages) ? j.messages : [],
      items: Array.isArray(j.items) ? j.items : null,
      rev: typeof j.rev === "number" ? j.rev : null,
      // Absent on an older child, where every page was effectively a delta.
      full: !!j.full,
      busy: !!j.busy,
      commands: Array.isArray(j.commands) ? j.commands.map(readCommand) : [],
      usage: readUsage(j.usage),
      live: readLive(j.partial),
    };
  } catch {
    return null;
  }
}

function readUsage(raw: unknown): ContextUsage | null {
  if (!raw || typeof raw !== "object") return null;
  const u = raw as Record<string, unknown>;
  const num = (v: unknown) => (typeof v === "number" ? v : null);
  const usage = { used: num(u.used), size: num(u.size), cost: num(u.cost) };
  // An empty object is what the child sends before the agent has reported
  // anything, and "nothing yet" must not render as "zero tokens".
  return usage.used === null && usage.size === null ? null : usage;
}

function readCommand(raw: unknown): AgentCommand {
  const c = (raw ?? {}) as Record<string, unknown>;
  return {
    name: typeof c.name === "string" ? c.name : "",
    description: typeof c.description === "string" ? c.description : "",
    hint: typeof c.hint === "string" ? c.hint : "",
  };
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

/** Answer the question the agent is blocked on. Returns an error, or null.
 *
 * A null *optionId* is a deliberate refusal — the person dismissed the question
 * rather than choosing from it — which the child forwards as ACP's `cancelled`
 * outcome. A 409 means someone else already answered: two windows watch one
 * conversation, and being second is not a mistake, so it reports as news rather
 * than as a failure.
 */
export async function answerPermission(
  base: string,
  requestId: string,
  optionId: string | null,
): Promise<string | null> {
  let r: Response;
  try {
    r = await sessionFetch(base + "/chat/permission", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: requestId, option_id: optionId }),
    });
  } catch (e) {
    return String(e);
  }
  if (r.ok) return null;
  if (r.status === 409) return "That question was already answered.";
  const d = await r.json().catch(() => ({}) as Record<string, unknown>);
  return String(d.error || `could not answer (${r.status})`);
}

/** Switch which agent drives the pane. Returns an error to show, or null.
 *
 * The interesting failure is a 409 naming the client that holds the kernel: one
 * agent runs code in a session, the claim is only released by a kernel restart,
 * and a switch made anyway would produce a pane that answers questions and then
 * refuses every cell. The child's message says so and names the way out.
 */
export async function setEngine(
  base: string,
  engine: ChatEngine,
): Promise<string | null> {
  let r: Response;
  try {
    r = await sessionFetch(base + "/chat/engine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ engine }),
    });
  } catch (e) {
    return String(e);
  }
  if (r.ok) return null;
  const d = await r.json().catch(() => ({}) as Record<string, unknown>);
  return String(d.error || `could not switch (${r.status})`);
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
