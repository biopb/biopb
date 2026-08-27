// The chat thread as the pane renders it, and the adapter that produces it.
//
// Deliberately *not* the wire format. Today the only source is the session
// child's `/api/chat/history`, which is an OpenAI chat-completions thread; the
// planned one is an external agent over ACP, which sends normalized items and
// then *updates* them in place. Both meet here, so the pane renders one shape
// and a second source is another adapter rather than a rewrite.
//
// That is also why a tool call is an item with an id and a status rather than
// something the renderer counts: ACP moves a call through pending -> in_progress
// -> completed/failed by addressing its id, and there has to be something for
// such an update to land on. It is where a running cell's partial output will go
// too, once the loop's `on_progress` is wired through the HTTP layer.

/** One piece of renderable content. ACP's content blocks, narrowed to what the
 * session child can currently produce. */
export type Block =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mime: string };

/** The blocks the pane can actually show as a picture. */
export type ImageBlock = Extract<Block, { type: "image" }>;

export type ToolStatus = "pending" | "in_progress" | "completed" | "failed";

export interface MessageItem {
  kind: "message";
  id: string;
  role: "user" | "assistant";
  blocks: Block[];
  /** The turn failed, or a tool did; rendered as a failure, not as speech. */
  error?: boolean;
  /** The turn was stopped on purpose. */
  cancelled?: boolean;
}

export interface ToolCallItem {
  kind: "tool_call";
  id: string;
  /** The tool's name today; ACP sends a human-readable title. */
  title: string;
  status: ToolStatus;
  /** The result, once there is one. */
  blocks: Block[];
  /** These blocks are a running cell's output so far, not its result. */
  live?: boolean;
}

/** A question the agent is blocked on, waiting for someone to answer.
 *
 * Only the ACP engine produces these: the built-in loop calls biopb's own tools
 * and asks nobody. The options are the *agent's*, verbatim — it decides what
 * they mean, so the pane offers exactly what it was given and invents nothing. */
export interface PermissionItem {
  kind: "permission";
  id: string;
  /** What the agent wants to do, in its own words. */
  title: string;
  /** ACP's classification — "edit", "execute", … — or "" when it gave none.
   * An edit's title is a bare path, so this is what says what will happen. */
  toolKind: string;
  options: { id: string; name: string; kind: string }[];
  /** The id the answer is addressed to. */
  requestId: string;
  /** The option chosen, `"cancelled"` if it was refused, null while open. */
  outcome: string | null;
}

export type ThreadItem = MessageItem | ToolCallItem | PermissionItem;

/** What the cell being polled right now has printed so far.
 *
 * Not part of the thread: the loop keeps it on the transport because
 * `_llm_messages` re-projects every stored message on every later turn, so
 * streamed stdout appended to the conversation would go back to the provider
 * again and again. It reaches the pane on the history read and is attached to
 * the running call here, which is where the finished result lands too. */
export interface LiveOutput {
  jobId: string;
  stdout: string;
  /** Older output was dropped; `stdout` is the tail. */
  truncated: boolean;
}

/** The wire shape of one `/api/chat/history` message. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  ts?: number;
  tool_calls?: {
    id?: string;
    function?: { name?: string; arguments?: string };
  }[];
  tool_call_id?: string;
  name?: string;
  error?: boolean;
  cancelled?: boolean;
  image?: string;
  mime?: string;
}

/** `(image returned by <tool>)` — the placeholder the loop gives an image
 * carrier. The only thing tying that image to the call it came from. */
const IMAGE_CARRIER = /^\(image returned by (.+)\)$/;

function textBlocks(content: string): Block[] {
  return content ? [{ type: "text", text: content }] : [];
}

/**
 * Merge a polled `?after=<id>` response into what the pane already holds.
 *
 * The request cannot say which kind of response it got: an unknown `after`
 * returns the whole conversation rather than an error, which is right for a view
 * that has just loaded or has fallen behind a reset, but leaves a replay and a
 * delta indistinguishable at the call site.
 *
 * So the child says which it sent, in `full`, and nothing here guesses. The
 * guess that used to live here — a replay starts at the conversation's first
 * message, so matching first ids is a replay — held only while a reset restarted
 * message ids. Ids are monotone across one now, precisely so a stale cursor
 * cannot match a message this view has never seen.
 *
 * A delta is deduplicated rather than trusted: two polls overlap whenever a
 * fetch outlives the interval, and the second returns messages the first has
 * already appended. Appending that repetition would show the thread twice.
 */
export function mergeHistory(
  existing: ChatMessage[],
  incoming: ChatMessage[],
  full = false,
): ChatMessage[] {
  // The child says this page is the whole thread, not a delta. Replace, and
  // replace with nothing when the thread is empty: after a reset that is
  // exactly what every other window is told, and reading an empty page as "no
  // news" is what left the cleared conversation on their screen with the new
  // one appended to it. Ids are monotone across a reset, so no id-based guess
  // can stand in for this.
  if (full) return incoming;
  if (!existing.length) return incoming;
  if (!incoming.length) return existing;
  const have = new Set(existing.map((m) => m.id));
  const fresh = incoming.filter((m) => !have.has(m.id));
  return fresh.length ? existing.concat(fresh) : existing;
}

/**
 * Project a chat-completions thread onto the items the pane renders.
 *
 * *busy* is the transport's state, not the thread's: an unanswered call is
 * running if a turn is still going and merely abandoned if it is not.
 */
export function fromChatHistory(
  messages: ChatMessage[],
  busy = false,
): ThreadItem[] {
  const items: ThreadItem[] = [];
  const byCallId = new Map<string, ToolCallItem>();

  for (const m of messages) {
    // An image rides the thread as a *user* message because that is the only
    // role that carries one back to the provider — a protocol artifact, not
    // something the user said. It belongs to the call that produced it, which
    // only the placeholder text records; ACP will carry the association itself.
    if (m.image) {
      const block: Block = {
        type: "image",
        data: m.image,
        mime: m.mime || "image/png",
      };
      const name = IMAGE_CARRIER.exec(m.content)?.[1];
      const owner = lastToolCall(items, name);
      if (owner) owner.blocks.push(block);
      else
        items.push({
          kind: "message",
          id: m.id,
          role: "assistant",
          blocks: [block],
        });
      continue;
    }

    if (m.role === "tool") {
      const call = m.tool_call_id ? byCallId.get(m.tool_call_id) : undefined;
      if (call) {
        // Prepended: an image may already have landed on this call, and the
        // text it accompanies reads first.
        call.blocks = textBlocks(m.content).concat(call.blocks);
        call.status = m.error ? "failed" : "completed";
        continue;
      }
      // No declaring assistant message to attach to. Shown on its own rather
      // than dropped: a result nobody asked for is still evidence.
      items.push({
        kind: "tool_call",
        id: m.id,
        title: m.name || "tool",
        status: m.error ? "failed" : "completed",
        blocks: textBlocks(m.content),
      });
      continue;
    }

    if (m.role === "assistant") {
      // A model that speaks before calling tools said that to the user; an
      // empty content is just the carrier for the calls and renders as nothing.
      if (m.content) {
        items.push({
          kind: "message",
          id: m.id,
          role: "assistant",
          blocks: textBlocks(m.content),
          ...(m.error ? { error: true } : {}),
          ...(m.cancelled ? { cancelled: true } : {}),
        });
      }
      for (const [i, call] of (m.tool_calls || []).entries()) {
        const item: ToolCallItem = {
          kind: "tool_call",
          id: call.id || `${m.id}-${i}`,
          title: call.function?.name || "tool",
          status: busy ? "in_progress" : "pending",
          blocks: [],
        };
        if (call.id) byCallId.set(call.id, item);
        items.push(item);
      }
      continue;
    }

    items.push({
      kind: "message",
      id: m.id,
      role: "user",
      blocks: textBlocks(m.content),
    });
  }

  return items;
}

/** The most recent call named *name* that has no image yet, else the most recent
 * call at all. Parallel calls to one tool are the case this exists for. */
function lastToolCall(items: ThreadItem[], name?: string): ToolCallItem | null {
  let fallback: ToolCallItem | null = null;
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i]!;
    if (item.kind !== "tool_call") continue;
    if (!fallback) fallback = item;
    if (!name) break;
    if (item.title === name && !item.blocks.some((b) => b.type === "image"))
      return item;
  }
  return fallback;
}

/**
 * Show a running cell's output on the call that is running it.
 *
 * The turn's whole reason for dropping `execute_code`'s promote window is that
 * a long cell should say something while it runs; a thread that goes quiet for
 * three minutes is the stalled cursor that was being avoided. The finished
 * result replaces this on the next poll, arriving as the tool's own message.
 *
 * **The first matching call, not the last.** A round's calls are dispatched
 * one at a time and each result is appended only when its own dispatch returns,
 * so while the first cell runs neither it nor the calls behind it have an
 * answer yet -- and an unanswered call reads as `in_progress`. With two
 * parallel `execute_code` calls that makes both of them in progress, while the
 * kernel is running only the earlier one. Scanning from the end would put a
 * cell's output on the call that has not started.
 */
export function applyLiveOutput(
  items: ThreadItem[],
  live: LiveOutput | null,
): ThreadItem[] {
  if (!live || !live.stdout) return items;
  for (const item of items) {
    if (item.kind !== "tool_call") continue;
    if (item.status !== "in_progress" || item.title !== "execute_code") continue;
    // Replaces the text rather than adding to it: this is the whole buffer on
    // every poll, not a delta, and the call has no result of its own until it
    // finishes. Written so re-applying it is a no-op -- the render path rebuilds
    // the items each time, but a function that only works once because of that
    // is one bug away from repeating a cell's output twice a second.
    item.blocks = [
      { type: "text", text: live.stdout },
      ...item.blocks.filter((b) => b.type !== "text"),
    ];
    item.live = true;
    return items;
  }
  return items;
}

/** The wire shape of one `/api/chat/history` item under the ACP engine.
 *
 * Already the pane's shape, because the child does the translating: it holds
 * the protocol connection, so it is the only side that ever sees ACP's own
 * spelling. What arrives here is a thread, not a protocol. */
export interface AcpItem {
  id: string;
  kind: "message" | "tool_call" | "permission";
  rev: number;
  role?: "user" | "assistant";
  blocks?: Block[];
  error?: boolean;
  cancelled?: boolean;
  title?: string;
  status?: ToolStatus;
  request_id?: string;
  tool_kind?: string;
  options?: { id: string; name: string; kind: string }[];
  outcome?: string | null;
}

/**
 * Merge a polled `?since=<rev>` page into what the pane already holds.
 *
 * Not `mergeHistory`: that one deduplicates, because the built-in loop's thread
 * only ever grows and a repeat is an overlap between two polls. Here a repeat is
 * the *point* — a tool call moving to `completed` arrives as the same id again —
 * so an item in the page replaces the one it names and appends only if it is new.
 */
export function mergeAcpItems(
  existing: AcpItem[],
  incoming: AcpItem[],
  full = false,
): AcpItem[] {
  if (full) return incoming;
  if (!incoming.length) return existing;
  const out = existing.slice();
  const at = new Map(out.map((item, i) => [item.id, i]));
  for (const item of incoming) {
    const i = at.get(item.id);
    if (i === undefined) {
      at.set(item.id, out.length);
      out.push(item);
    } else {
      out[i] = item;
    }
  }
  return out;
}

/**
 * Project an ACP thread onto the items the pane renders.
 *
 * *busy* does the same job it does for the built-in loop: a call the agent
 * never reported finishing is running while the turn is, and abandoned once it
 * is not. The agent is not obliged to close every call it opens — a cancelled
 * turn leaves them open by design — so this is the only thing that stops a
 * spinner from spinning forever.
 */
export function fromAcpItems(items: AcpItem[], busy = false): ThreadItem[] {
  const out: ThreadItem[] = [];
  for (const item of items) {
    if (item.kind === "message") {
      out.push({
        kind: "message",
        id: item.id,
        role: item.role === "user" ? "user" : "assistant",
        blocks: item.blocks ?? [],
        ...(item.error ? { error: true } : {}),
        ...(item.cancelled ? { cancelled: true } : {}),
      });
    } else if (item.kind === "tool_call") {
      const status = item.status ?? "pending";
      out.push({
        kind: "tool_call",
        id: item.id,
        title: item.title || "(tool)",
        status:
          !busy && (status === "pending" || status === "in_progress")
            ? "failed"
            : status,
        blocks: item.blocks ?? [],
      });
    } else if (item.kind === "permission") {
      out.push({
        kind: "permission",
        id: item.id,
        title: item.title || "run something",
        toolKind: item.tool_kind ?? "",
        options: item.options ?? [],
        requestId: item.request_id ?? "",
        outcome: item.outcome ?? null,
      });
    }
  }
  return out;
}

/** The question the pane should be asking right now, if any.
 *
 * At most one: the agent blocks on each in turn, so a second open question
 * means the first was answered and the child has not been polled since. */
export function openPermission(items: ThreadItem[]): PermissionItem | null {
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i]!;
    if (item.kind === "permission" && item.outcome === null) return item;
  }
  return null;
}

/** The newest line worth showing on one line — what a collapsed group reports
 * while its cell runs. Trailing blank lines are what a `print()` leaves. */
export function latestLine(text: string): string {
  const lines = text.split("\n");
  while (lines.length && !lines[lines.length - 1]!.trim()) lines.pop();
  return lines.length ? lines[lines.length - 1]!.trim() : "";
}

/** A run of consecutive tool calls, or a single message — what the pane draws.
 *
 * Grouping is a render concern, so the items stay individually addressable (an
 * ACP `tool_call_update` names one id) while the pane still collapses a whole
 * round to one line. */
export type ThreadGroup =
  | { kind: "message"; id: string; item: MessageItem }
  | { kind: "permission"; id: string; item: PermissionItem }
  | { kind: "tools"; id: string; calls: ToolCallItem[]; images: ImageBlock[] };

export function groupThread(items: ThreadItem[]): ThreadGroup[] {
  const groups: ThreadGroup[] = [];
  for (const item of items) {
    if (item.kind === "message") {
      groups.push({ kind: "message", id: item.id, item });
      continue;
    }
    // Never folded into a tool group, even sitting among tool calls, which is
    // exactly where one arrives: it is the only item in the thread that is
    // *asking* the reader something, and a question inside a collapsed round is
    // a question nobody answers.
    if (item.kind === "permission") {
      groups.push({ kind: "permission", id: item.id, item });
      continue;
    }
    const last = groups[groups.length - 1];
    if (last && last.kind === "tools") last.calls.push(item);
    else groups.push({ kind: "tools", id: item.id, calls: [item], images: [] });
  }
  // Images are hoisted out of the fold. Collapsing exists to hide walls of tool
  // text; a screenshot is the answer to the question, and folding it away would
  // defeat the tool that produced it.
  for (const g of groups) {
    if (g.kind !== "tools") continue;
    g.images = g.calls.flatMap((c) =>
      c.blocks.filter((b): b is ImageBlock => b.type === "image"),
    );
  }
  return groups;
}

/** Text of a tool call's result, for the expanded view. */
export function toolText(call: ToolCallItem): string {
  return call.blocks
    .filter((b): b is Extract<Block, { type: "text" }> => b.type === "text")
    .map((b) => b.text)
    .join("\n");
}
