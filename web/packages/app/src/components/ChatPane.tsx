import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  cancelTurn,
  fetchHistory,
  sendTurn,
  type ChatStatus,
} from "../utils/chatClient";
import {
  applyLiveOutput,
  fromChatHistory,
  groupThread,
  latestLine,
  mergeHistory,
  toolText,
  type ChatMessage,
  type ImageBlock,
  type LiveOutput,
  type ToolCallItem,
} from "../utils/chatThread";

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
  const [busy, setBusy] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [zoom, setZoom] = useState<ImageBlock | null>(null);
  const [live, setLive] = useState<LiveOutput | null>(null);

  // The cursor into the conversation. A ref, not state: it is read by the poll
  // and must never be a render's worth of steps behind it.
  const after = useRef<string | null>(null);
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
    if (page.messages.length) {
      setMessages((prev) => mergeHistory(prev, page.messages));
      after.current = page.messages[page.messages.length - 1]!.id;
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
    setBusy(true); // poll faster immediately rather than after one slow tick
    poll();
  }, [base, text, busy, sending, poll]);

  const stop = useCallback(async () => {
    await cancelTurn(base);
    poll();
  }, [base, poll]);

  const toggle = useCallback((id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const groups = groupThread(
    applyLiveOutput(fromChatHistory(messages, busy), live),
  );

  // Stick to the newest message unless the reader has scrolled up to read.
  const scroller = useRef<HTMLDivElement | null>(null);
  const atBottom = useRef(true);
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && atBottom.current) el.scrollTop = el.scrollHeight;
  }, [messages, expanded, live]);

  return (
    <section className="chat">
      <div className="chat-head">
        <span className="chat-title">chat</span>
        <span className="chat-model">{status.model}</span>
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
      </div>

      <div className="chat-compose">
        <textarea
          value={text}
          disabled={!status.ready}
          placeholder={
            status.ready ? "Ask about this session…" : "chat is unavailable"
          }
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            // Enter sends, Shift+Enter is a newline — the opposite of the
            // console below, deliberately: that one is code, where a newline is
            // the common keystroke and running is the rare one.
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <div className="chat-bar">
          {busy ? (
            <button onClick={stop}>Cancel turn</button>
          ) : (
            <button
              className="primary"
              disabled={!status.ready || sending || !text.trim()}
              onClick={submit}
            >
              {sending ? "sending…" : "Send  (Enter)"}
            </button>
          )}
          {busy ? <span className="chat-busy">working…</span> : null}
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
  .chat-compose textarea { width: 100%; box-sizing: border-box; min-height: 56px;
                resize: vertical; background: #0c0c0c; color: #ddd;
                border: 1px solid #333; border-radius: 4px; padding: 8px;
                font: inherit; }
  .chat-compose textarea:focus { outline: none; border-color: #2a5; }
  .chat-compose textarea:disabled { opacity: .5; }
  .chat-bar { display: flex; align-items: center; gap: 10px; margin-top: 6px; }
  .chat-bar button:disabled { opacity: .55; cursor: default; background: #222; }
  .chat-busy { color: #7e7; font-size: 12px; }
  .chat-err { color: #f99; font-size: 12px; }
  .chat-zoom { position: fixed; inset: 0; background: rgba(0,0,0,.85); z-index: 50;
               display: flex; align-items: center; justify-content: center;
               cursor: zoom-out; padding: 24px; }
  .chat-zoom img { max-width: 100%; max-height: 100%; }
`;
