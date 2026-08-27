import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  answerPermission,
  cancelTurn,
  compactThread,
  fetchHistory,
  resetThread,
  sendTurn,
  setEngine,
  type AgentCommand,
  type ChatStatus,
} from "../utils/chatClient";
import {
  applyLiveOutput,
  fromAcpItems,
  fromChatHistory,
  groupThread,
  latestLine,
  mergeAcpItems,
  mergeHistory,
  openPermission,
  toolText,
  type AcpItem,
  type ChatMessage,
  type ImageBlock,
  type LiveOutput,
  type PermissionItem,
  type ToolCallItem,
} from "../utils/chatThread";
import { escAction, sendsOnEnter } from "../utils/chatKeys";
import {
  contextReport,
  matchCommands,
  parseCommand,
} from "../utils/chatCommands";

// The built-in agent, beside the job list it drives.
//
// Same page as the jobs on purpose: a chat cell appears in that list as
// `origin: "chat"`, and the message a cancelled turn leaves behind says to
// interrupt the cell "from the job list". Split across two routes, that
// sentence stops being true. It is also what makes the missing partial-output
// stream tolerable for now — the running cell's stdout is live in the row next
// to the thread.

export default function ChatPane({
  base,
  status,
  pollMs,
}: {
  base: string;
  status: ChatStatus;
  pollMs: number;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  // The ACP engine's thread, and what the harness says it can be asked to do.
  // Held beside `messages` rather than in place of it: the engine can be
  // switched mid-session, and the other thread is still there to come back to.
  const [items, setItems] = useState<AcpItem[]>([]);
  const [commands, setCommands] = useState<AgentCommand[]>([]);
  // The engine in force *now*. Seeded from the status the page probed once at
  // mount, and then owned here, because switching is a thing this pane does and
  // re-probing the whole page to learn the result of its own click would be a
  // round trip to find out something it already knows.
  const [engine, setEngineState] = useState(status.engine);
  useEffect(() => setEngineState(status.engine), [status.engine]);
  const acp = engine === "acp";
  const [busy, setBusy] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [zoom, setZoom] = useState<ImageBlock | null>(null);
  const [live, setLive] = useState<LiveOutput | null>(null);
  // The pane answering the pane, rather than the model answering the user.
  // Kept out of `messages` on purpose: everything in there is the child's and
  // has an id the poll merges against, and a locally invented message would
  // either collide with one or survive a reset. So it is one line, held below
  // the thread, replaced by the next command and cleared by the next turn --
  // which is also the moment it stops being true.
  const [notice, setNotice] = useState<string | null>(null);

  // The cursor into the conversation. A ref, not state: it is read by the poll
  // and must never be a render's worth of steps behind it.
  //
  // Two engines spell it differently -- a last-seen message id for the built-in
  // loop, a revision watermark for ACP -- and the pane holds whichever it was
  // last given rather than deciding. Only one of the two can say "an item you
  // already have has changed", which is the thing ACP does constantly.
  const after = useRef<string | number | null>(null);
  // One read at a time. A fetch slower than the interval would otherwise leave
  // several in flight against the same cursor, each fetching what the last has
  // already appended -- worst on exactly the slow link where it costs most.
  // `mergeHistory` survives the overlap; this keeps it from happening.
  const reading = useRef(false);

  const poll = useCallback(async () => {
    if (reading.current) return;
    reading.current = true;
    let page;
    try {
      page = await fetchHistory(base, after.current);
    } finally {
      reading.current = false;
    }
    if (!page) return; // unreachable; keep what is on screen
    setBusy(page.busy);
    setLive(page.live);
    // No early return above this: while a cell runs the thread gains no
    // messages at all, so a poll with an empty page is exactly the poll whose
    // live output matters. Skipping the rest on `!messages.length` would have
    // frozen the output for the entire cell -- so the skip is expressed as the
    // guard it actually is, and cannot grow to cover anything else.
    // A full page is acted on even when it is empty -- that is a reset, seen
    // from a window that did not ask for one.
    if (page.items !== null) {
      // The ACP engine. Every page is acted on, including an empty one: a
      // watermark only moves forward, and holding it back on a quiet poll would
      // re-fetch the whole thread on the next one.
      const fresh = page.items;
      setItems((prev) => mergeAcpItems(prev, fresh, page.full));
      setCommands(page.commands);
      after.current = page.rev;
      return;
    }
    if (page.full || page.messages.length) {
      setMessages((prev) => mergeHistory(prev, page.messages, page.full));
      after.current = page.messages.length
        ? page.messages[page.messages.length - 1]!.id
        : null;
    }
  }, [base]);

  // Faster while a turn runs. The page's own interval is tuned for a job list;
  // a conversation updating every three seconds reads as a stall.
  const interval = busy ? 500 : pollMs;
  useEffect(() => {
    poll();
    const t = setInterval(poll, interval);
    return () => clearInterval(t);
  }, [poll, interval]);

  const submit = useCallback(async () => {
    const body = text.trim();
    if (!body || busy || sending) return;
    setSending(true);
    setError(null);
    const err = await sendTurn(base, body);
    setSending(false);
    if (err) {
      setError(err);
      return;
    }
    setText(""); // cleared only once it was accepted, so nothing is lost
    setNotice(null); // a context report describes the thread as it was
    setBusy(true); // poll faster immediately rather than after one slow tick
    poll();
  }, [base, text, busy, sending, poll]);

  // One shape, two sources: whichever engine is running produces the same
  // `ThreadItem[]`, so everything below this line renders one thread and knows
  // nothing about where it came from.
  const thread = acp
    ? fromAcpItems(items, busy)
    : applyLiveOutput(fromChatHistory(messages, busy), live);
  const groups = groupThread(thread);
  // At most one, and it is what Escape means while it is up.
  const asking = acp ? openPermission(thread) : null;

  // Switching hands the pane to the other agent. The thread does not travel
  // with it: the two hold different conversations, and showing one engine's
  // transcript above another engine's answers would invent a continuity that
  // does not exist.
  const switchEngine = useCallback(
    async (next: "builtin" | "acp") => {
      if (next === engine) return;
      const err = await setEngine(base, next);
      if (err) {
        setError(err);
        return;
      }
      setEngineState(next);
      setError(null);
      setNotice(null);
      setMessages([]);
      setItems([]);
      setCommands([]);
      after.current = null;
      poll();
    },
    [base, engine, poll],
  );

  const stop = useCallback(async () => {
    await cancelTurn(base);
    poll();
  }, [base, poll]);

  // Answering is a write and then a poll, like every other action here: the
  // outcome is server state, so the thread shows what happened rather than this
  // guessing on its behalf.
  const answer = useCallback(
    async (item: PermissionItem, optionId: string | null) => {
      const err = await answerPermission(base, item.requestId, optionId);
      if (err) setError(err);
      poll();
    },
    [base, poll],
  );

  // Escape, bound on the window rather than the composer: a reader who clicked
  // a job row to watch its output would otherwise find the key silently stops
  // working. `escAction` holds the ordering and the two cases Escape is already
  // spoken for; see chatKeys.ts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      const action = escAction({
        composing: e.isComposing,
        imageOpen: zoom !== null,
        inConsole: !!document.activeElement?.closest(".console"),
        busy,
        permissionOpen: !!asking,
      });
      if (action === "none") return;
      e.preventDefault();
      if (action === "close-image") setZoom(null);
      else if (action === "refuse-permission") answer(asking!, null);
      else stop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [zoom, busy, stop, asking, answer]);
  // The thread's only bound. `_llm_messages` re-projects every stored message
  // on every turn, so a conversation that outgrows the provider's context fails
  // -- and records the failure in the thread, so it fails the same way forever.
  // Confirmed because it cannot be undone; the cells it ran are untouched, which
  // is what the wording has to get across.
  const startNew = useCallback(async () => {
    if (!confirm("Start a new conversation? This clears the chat. Cells it ran stay in the job list."))
      return;
    const err = await resetThread(base);
    if (err) {
      setError(err);
      return;
    }
    setMessages([]);
    after.current = null;
    setExpanded(new Set());
    setLive(null);
    setNotice(null);
    setBusy(false);
    setError(null);
    poll();
  }, [base, poll]);

  // The gentler half of the same problem: a reset gives up what was said, this
  // keeps it and folds it. Not confirmed -- nothing is lost from the thread, so
  // the worst case is a wasted provider call.
  const compact = useCallback(async () => {
    setError(null);
    const err = await compactThread(base);
    if (err) setError(err);
    else poll();
  }, [base, poll]);

  // Enter, and the button beside it. Every path in goes through here so a
  // command cannot be reachable one way and not the other -- which is what a
  // send button disabled during a turn would do to `/context`, the one command
  // whose whole point is answering "should I stop and compact?" mid-turn.
  const onEnter = useCallback(async () => {
    const parsed = parseCommand(text, engine, commands);
    if (parsed.kind === "send") {
      submit();
      return;
    }
    if (parsed.kind === "reject") {
      // Beside the composer, where the typo is, and the text is left alone so
      // it can be corrected rather than retyped.
      setError(parsed.message);
      return;
    }
    setText("");
    setError(null);
    if (parsed.name === "context") {
      // Answered from what the pane already holds. Nothing is sent, so this
      // works during a turn and costs the conversation nothing -- which matters
      // for the one command a person runs *because* they are worried about size.
      setNotice(contextReport(messages, status.compacted, status.model));
      return;
    }
    setNotice(null);
    if (parsed.name === "new") await startNew();
    else await compact();
  }, [
    text,
    submit,
    messages,
    status.compacted,
    status.model,
    engine,
    commands,
    startNew,
    compact,
  ]);

  const toggle = useCallback((id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // Both read the typed text the same way, so the button is enabled exactly
  // when Enter would do something.
  const matches = matchCommands(text, engine);
  const isCommand = parseCommand(text, engine, commands).kind === "command";


  // Stick to the newest message unless the reader has scrolled up to read.
  const scroller = useRef<HTMLDivElement | null>(null);
  const atBottom = useRef(true);
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && atBottom.current) el.scrollTop = el.scrollHeight;
  }, [messages, items, expanded, live, notice]);

  return (
    <section className="chat">
      <div className="chat-head">
        <span className="chat-title">chat</span>
        {/* Offered only when there is a choice to make. One engine configured
            is the common install, and a select with a single option is a
            control that cannot act. */}
        {status.engines.filter((e) => e.ready).length > 1 ? (
          <select
            className="chat-engine"
            value={engine}
            onChange={(e) => switchEngine(e.target.value as "builtin" | "acp")}
            title="Which agent drives this pane"
          >
            {status.engines.map((e) => (
              <option key={e.engine} value={e.engine} disabled={!e.ready}>
                {e.engine === "acp" ? "harness" : "built-in"}
              </option>
            ))}
          </select>
        ) : null}
        <span
          className="chat-model"
          title={acp ? "an ACP harness, running as you" : "the built-in loop"}
        >
          {status.model}
        </span>
        {/* Shown only once there is something to fold: a control that cannot
            act is how the observe page's Interrupt earned its "No running job."
            dialog. */}
        {status.compacted ? (
          <span className="chat-folded" title="Messages the model now sees only as a summary">
            {status.compacted} folded
          </span>
        ) : null}
        {acp ? null : (
          <button
            className="chat-new"
            onClick={compact}
            title="Summarise the older part of the conversation for the model. The thread is not changed."
          >
            compact
          </button>
        )}
        <button className="chat-new" onClick={startNew} title="Start a new conversation">
          new
        </button>
      </div>

      <div
        className="chat-thread"
        ref={scroller}
        onScroll={() => {
          const el = scroller.current;
          if (!el) return;
          atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 8;
        }}
      >
        {/* Said once, at the top of an empty thread, rather than after someone
            types a message and has it rejected. */}
        {!status.ready ? (
          <div className="chat-note">{status.reason || "chat is unavailable"}</div>
        ) : groups.length === 0 ? (
          <div className="chat-note">
            Ask about the data in this session. The agent drives the same kernel
            and viewer you do.
          </div>
        ) : null}

        {groups.map((g) =>
          g.kind === "message" ? (
            <div
              key={g.id}
              className={
                "msg " +
                g.item.role +
                (g.item.error ? " err" : "") +
                (g.item.cancelled ? " cancelled" : "")
              }
            >
              {g.item.blocks.map((b, i) =>
                b.type === "text" ? (
                  <span key={i}>{b.text}</span>
                ) : (
                  <Thumb key={i} block={b} onOpen={setZoom} />
                ),
              )}
            </div>
          ) : g.kind === "permission" ? (
            <Permission key={g.id} item={g.item} onAnswer={answer} />
          ) : (
            <ToolGroup
              key={g.id}
              calls={g.calls}
              images={g.images}
              open={expanded.has(g.id)}
              onToggle={() => toggle(g.id)}
              onOpenImage={setZoom}
            />
          ),
        )}

        {notice ? <div className="chat-report">{notice}</div> : null}
      </div>

      <div className="chat-compose">
        {/* How anyone finds out these exist. A slash on its own lists them all,
            and a click completes rather than runs -- Enter stays the thing that
            acts, including for the one that clears the conversation. */}
        {matches.length ? (
          <div className="chat-cmds">
            {matches.map((c) => (
              <button
                key={c.typed}
                className="chat-cmd"
                title={c.help}
                onClick={() => setText(c.typed)}
              >
                <span className="chat-cmd-name">{c.typed}</span>
                {c.aliases.length ? (
                  <span className="chat-cmd-alias">{c.aliases.join(", ")}</span>
                ) : null}
                <span className="chat-cmd-help">{c.help}</span>
              </button>
            ))}
          </div>
        ) : null}

        <div className="chat-input">
          <textarea
            value={text}
            disabled={!status.ready}
            placeholder={
              status.ready ? "Ask about this session…" : "chat is unavailable"
            }
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (!sendsOnEnter(e)) return;
              e.preventDefault();
              onEnter();
            }}
          />
          {/* In the corner of the box rather than on a row of its own. Enter is
              the way to send and the hint below says so, but a submit control
              has to exist: it is the only pointer path, and the only thing a
              screen reader can find. */}
          <button
            className="chat-send"
            aria-label="Send message"
            title="Send (Enter)"
            disabled={!status.ready || sending || (busy && !isCommand) || !text.trim()}
            onClick={onEnter}
          >
            ↩
          </button>
        </div>

        <div className="chat-bar">
          {busy ? (
            // Where the Cancel button used to be, and where the eye already is
            // during a turn. A button that replaces Send mid-turn changes what
            // the control under the cursor means while you are looking at it.
            <span className="chat-busy">
              working… · <kbd>esc</kbd> to cancel
            </span>
          ) : (
            <span className="chat-hint">
              <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a
              newline · <kbd>/</kbd> for commands
            </span>
          )}
          {error ? <span className="chat-err">{error}</span> : null}
        </div>
      </div>

      {zoom ? (
        <div className="chat-zoom" onClick={() => setZoom(null)}>
          <img src={dataUrl(zoom)} alt="" />
        </div>
      ) : null}
      <style>{CHAT_CSS}</style>
    </section>
  );
}

function dataUrl(b: ImageBlock): string {
  return `data:${b.mime};base64,${b.data}`;
}

function Thumb({
  block,
  onOpen,
}: {
  block: ImageBlock;
  onOpen: (b: ImageBlock) => void;
}) {
  return (
    <img
      className="chat-thumb"
      src={dataUrl(block)}
      alt="tool output"
      onClick={() => onOpen(block)}
    />
  );
}

/** A round of tool calls as one line, with the images it produced kept out of
 * the fold. */
/** The agent is blocked, asking whether it may do something.
 *
 * The buttons are the agent's own options, in its own words and its own order.
 * Nothing is relabelled and nothing is added: it decides what "allow always"
 * scopes to, and a pane that renamed the choices would be answering a different
 * question from the one it was asked.
 *
 * Answered, it stays in the thread showing what was chosen. A question that
 * vanishes leaves the reader unable to say later what they agreed to -- which
 * is exactly the record worth keeping about an agent with its own shell.
 */
function Permission({
  item,
  onAnswer,
}: {
  item: PermissionItem;
  onAnswer: (item: PermissionItem, optionId: string | null) => void;
}) {
  const chosen = item.outcome
    ? item.options.find((o) => o.id === item.outcome)
    : undefined;
  return (
    <div className={"chat-ask" + (item.outcome ? " done" : "")}>
      <div className="chat-ask-title">
        {item.toolKind ? (
          <span className="chat-ask-kind">{item.toolKind}</span>
        ) : null}
        {item.title}
      </div>
      {item.outcome ? (
        <div className="chat-ask-outcome">
          {item.outcome === "cancelled" ? "refused" : (chosen?.name ?? item.outcome)}
        </div>
      ) : (
        <div className="chat-ask-options">
          {item.options.map((o) => (
            <button
              key={o.id}
              className={
                "chat-ask-btn" +
                (o.kind.startsWith("allow") ? " allow" : " reject")
              }
              onClick={() => onAnswer(item, o.id)}
            >
              {o.name}
            </button>
          ))}
          {/* Escape does this too; the button is here because a question with
              no visible way out reads as a trap. */}
          <button
            className="chat-ask-btn reject"
            onClick={() => onAnswer(item, null)}
            title="Refuse without choosing (Esc)"
          >
            dismiss
          </button>
        </div>
      )}
    </div>
  );
}

function ToolGroup({
  calls,
  images,
  open,
  onToggle,
  onOpenImage,
}: {
  calls: ToolCallItem[];
  images: ImageBlock[];
  open: boolean;
  onToggle: () => void;
  onOpenImage: (b: ImageBlock) => void;
}) {
  const failed = calls.filter((c) => c.status === "failed").length;
  const running = calls.some((c) => c.status === "in_progress");
  const label =
    calls.length === 1 ? "1 tool call" : `${calls.length} tool calls`;
  // Collapsing hides tool *detail*; a running cell's newest line is progress,
  // and withholding it restores the silence the streaming work removed. One
  // line, only while it runs, replaced by the folded result when it finishes.
  const streaming = calls.find((c) => c.live);
  const tail = streaming ? latestLine(toolText(streaming)) : "";

  return (
    <div className={"tools" + (open ? " open" : "")}>
      <div className="tools-row" onClick={onToggle}>
        <span className="caret">{open ? "▾" : "▸"}</span>
        <span className="tools-label">{label}</span>
        {running ? <span className="tools-run">running</span> : null}
        {failed ? <span className="tools-fail">{failed} failed</span> : null}
        <span className="tools-names">
          {calls.map((c) => c.title).join(", ")}
        </span>
      </div>
      {open ? (
        <div className="tools-detail">
          {calls.map((c) => (
            <div key={c.id} className="tool">
              <div className={"tool-name " + c.status}>
                {c.title}
                <span className="tool-status">
                  {c.live ? "running — output so far" : c.status}
                </span>
              </div>
              <pre>{toolText(c) || "(no output)"}</pre>
            </div>
          ))}
        </div>
      ) : null}
      {!open && tail ? <div className="tools-tail">{tail}</div> : null}
      {images.length ? (
        <div className="tools-images">
          {images.map((b, i) => (
            <Thumb key={i} block={b} onOpen={onOpenImage} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

const CHAT_CSS = `
.chat-engine {
  font: inherit;
  font-size: 11px;
  background: transparent;
  color: inherit;
  border: 1px solid var(--border, #444);
  border-radius: 4px;
  padding: 0 2px;
}

.chat-ask {
  margin: 6px 0;
  padding: 8px 10px;
  border: 1px solid var(--warn, #b58900);
  border-radius: 6px;
  background: rgba(181, 137, 0, 0.08);
  font-size: 12px;
}
.chat-ask.done { opacity: 0.6; border-style: dashed; }
.chat-ask-title { font-weight: 600; margin-bottom: 6px; word-break: break-word; }
.chat-ask-kind { font-weight: 400; opacity: 0.7; margin-right: 6px;
                 font-family: ui-monospace, Menlo, monospace; font-size: 11px; }
.chat-ask-options { display: flex; flex-wrap: wrap; gap: 6px; }
.chat-ask-btn {
  padding: 3px 8px;
  border-radius: 4px;
  border: 1px solid currentColor;
  background: transparent;
  cursor: pointer;
  font: inherit;
}
.chat-ask-btn.allow { color: var(--ok, #2a7); }
.chat-ask-btn.reject { color: var(--muted, #888); }
.chat-ask-outcome { font-style: italic; }

  .chat { display: flex; flex-direction: column; min-height: 0;
          border: 1px solid #333; border-radius: 5px; background: #161616; }
  .chat-head { display: flex; align-items: baseline; gap: 10px; padding: 8px 12px;
               border-bottom: 1px solid #333; }
  .chat-title { color: #6a8; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }
  .chat-model { color: #777; font-size: 11px; margin-left: auto;
                font-family: ui-monospace, Menlo, monospace; }
  .chat-thread { flex: 1; overflow: auto; padding: 10px 12px; min-height: 0; }
  .chat-note { color: #777; font-size: 12px; padding: 8px 0 12px; }
  .chat .msg { margin: 0 0 10px; white-space: pre-wrap; word-break: break-word; }
  .chat .msg.user { background: #1d2430; border-left: 2px solid #47f;
                    padding: 6px 10px; border-radius: 4px; }
  .chat .msg.assistant { padding: 2px 0; }
  .chat .msg.err { color: #f99; }
  .chat .msg.cancelled { color: #999; font-style: italic; }
  .chat .tools { margin: 0 0 10px; }
  .chat .tools-row { display: flex; align-items: center; gap: 8px; cursor: pointer;
                     color: #8a8; font-size: 12px; padding: 3px 0; }
  .chat .tools-row:hover { color: #beb; }
  .chat .caret { width: 10px; }
  .chat .tools-label { font-weight: 600; }
  .chat .tools-run { color: #7e7; }
  .chat .tools-fail { color: #f99; }
  .chat .tools-names { color: #666; font-family: ui-monospace, Menlo, monospace;
                       font-size: 11px; white-space: nowrap; overflow: hidden;
                       text-overflow: ellipsis; flex: 1; min-width: 0; }
  .chat .tools-detail { border-left: 1px solid #333; margin: 4px 0 0 5px; padding-left: 10px; }
  .chat .tool { margin-bottom: 8px; }
  .chat .tool-name { font-family: ui-monospace, Menlo, monospace; font-size: 11px;
                     color: #8a8; display: flex; gap: 8px; margin-bottom: 2px; }
  .chat .tool-name.failed { color: #f99; }
  .chat .tool-status { color: #666; }
  .chat .tools pre { white-space: pre-wrap; word-break: break-word; margin: 0;
                     background: #0c0c0c; padding: 6px 8px; border-radius: 4px;
                     max-height: 32vh; overflow: auto;
                     font-family: ui-monospace, Menlo, monospace; font-size: 11px; }
  .chat .tools-tail { color: #7a8a7a; font-family: ui-monospace, Menlo, monospace;
                      font-size: 11px; margin: 2px 0 0 15px;
                      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .chat .tools-images { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
  .chat-thumb { max-width: 120px; max-height: 120px; border: 1px solid #333;
                border-radius: 4px; cursor: zoom-in; display: block; }
  .chat-compose { border-top: 1px solid #333; padding: 8px 10px; }
  .chat-input { position: relative; }
  .chat-compose textarea { width: 100%; box-sizing: border-box; min-height: 56px;
                resize: vertical; background: #0c0c0c; color: #ddd;
                border: 1px solid #333; border-radius: 4px;
                /* room for the send button in the corner */
                padding: 8px 34px 8px 8px;
                font: inherit; }
  .chat-compose textarea:focus { outline: none; border-color: #2a5; }
  .chat-compose textarea:disabled { opacity: .5; }
  .chat-send { position: absolute; right: 6px; bottom: 6px; width: 22px;
               height: 22px; padding: 0; line-height: 1; border-radius: 4px;
               border: 1px solid #333; background: #1a2a1a; color: #7e7;
               cursor: pointer; font-size: 12px; }
  .chat-send:hover:not(:disabled) { background: #244; }
  .chat-send:disabled { opacity: .35; cursor: default; background: #181818;
                        color: #666; }
  .chat-bar { display: flex; align-items: baseline; gap: 10px; margin-top: 6px;
              min-height: 15px; }
  .chat-busy { color: #7e7; font-size: 12px; }
  .chat-hint { color: #666; font-size: 12px; }
  .chat-bar kbd { font-family: ui-monospace, Menlo, monospace; font-size: 11px;
                  border: 1px solid #3a3a3a; border-radius: 3px; padding: 0 3px;
                  color: #999; }
  .chat-err { color: #f99; font-size: 12px; }
  .chat-report { white-space: pre-wrap; color: #8a8; font-size: 12px;
                 font-family: ui-monospace, Menlo, monospace;
                 border-left: 2px solid #3a3a3a; padding: 4px 0 4px 10px;
                 margin: 0 0 10px; }
  .chat-cmds { display: flex; flex-direction: column; margin-bottom: 6px;
               border: 1px solid #333; border-radius: 4px; overflow: hidden; }
  .chat-cmd { display: flex; align-items: baseline; gap: 8px; text-align: left;
              padding: 4px 8px; border: 0; background: #121212; color: #999;
              cursor: pointer; font: inherit; font-size: 12px; }
  .chat-cmd:hover { background: #1d2a1d; }
  .chat-cmd-name { color: #7e7; font-family: ui-monospace, Menlo, monospace; }
  .chat-cmd-alias { color: #666; font-family: ui-monospace, Menlo, monospace;
                    font-size: 11px; }
  /* Ellipsised, not wrapped: an agent's description can be a paragraph, and a
     completion list whose rows grow to fit one is unreadable. min-width:0 is
     what lets a flex child shrink below its content and actually clip. */
  .chat-cmd-help { color: #777; margin-left: auto; min-width: 0;
                   white-space: nowrap; overflow: hidden;
                   text-overflow: ellipsis; }
  /* No margin-left:auto here -- .chat-model already has one, and a second
     would share the free space between them instead of pinning both right. */
  .chat-new { font-size: 11px; padding: 1px 8px;
              border-radius: 10px; border: 1px solid #333; background: #181818;
              color: #999; cursor: pointer; }
  .chat-new:hover { background: #222; color: #ccc; }
  .chat-folded { color: #777; font-size: 11px; }
  .chat-zoom { position: fixed; inset: 0; background: rgba(0,0,0,.85); z-index: 50;
               display: flex; align-items: center; justify-content: center;
               cursor: zoom-out; padding: 24px; }
  .chat-zoom img { max-width: 100%; max-height: 100%; }
`;
