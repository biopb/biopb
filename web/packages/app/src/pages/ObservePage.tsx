import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { useParams } from "react-router-dom";
import { localRootsProxied } from "../auth";
import ChatPane from "../components/ChatPane";
import { fetchChatStatus, type ChatStatus } from "../utils/chatClient";
import { sessionFetch, sessionVerdict } from "../utils/sessionFetch";
import {
  clampChatWidth,
  defaultChatWidth,
} from "../utils/chatPaneWidth";
import { useDocumentTitle } from "../hooks/useDocumentTitle";
import { withBase } from "../base";

// Per-session observe UI, ported from the buildless _OBSERVE_HTML that each MCP
// session child used to serve at /observe. The child now serves only /api/*; the
// control front serves this SPA shell at /session/<id>/observe and proxies
// /session/<id>/api/* to the child. The API base is therefore the session prefix.

/** What to call the writer of a cell, in prose the reader is part of.
 *
 * Every site that needed this used to test `=== "user"` and call the other
 * branch "agent", which was fine while there were two writers and mislabelled
 * a chat cell the moment there were three. One mapping, so a fourth writer
 * shows its own name rather than someone else's.
 */
function writerName(origin?: string): string {
  if (origin === "user") return "you";
  if (origin === "chat") return "chat";
  if (origin === "mcp") return "the MCP client";
  return origin || "another writer";
}

const CHAT_WIDTH_KEY = "biopb.observe.chatWidth";

interface JobSummary {
  job_id: string;
  status: string; // running | ok | error | interrupted
  origin?: string; // mcp | user | chat — which surface submitted the cell
  elapsed: number;
  code_preview?: string;
  /** Why the cell was run, when whoever ran it said why. Absent on an older
   * child, and empty for a cell nobody explained (the console's, typically).
   * On a verification row it is the workflow's title: the cells arrive as bare
   * code and there is no per-cell intent to show. */
  intent_preview?: string;
  /** A verification, which ran in a scratch kernel rather than this session's. */
  verify?: boolean;
}
/** The verified workflow this session has, if any — a clean program an agent
 * rewrote from the transcript and proved by running in a scratch namespace.
 * Absent on an older child, and null until something is verified. */
interface WorkflowSummary {
  title?: string;
  cells: number;
  created?: number;
}
/** Which document to download. Not a flag inside one: the audit export is every
 * job that ran, the workflow export is the verified program someone rewrote
 * from it. Different questions, so the reader chooses. */
type NotebookKind = "audit" | "workflow";
/** Which kernel the work column is showing.
 *
 * Two kernels, one at a time. The session kernel holds the user's variables,
 * layers and viewer; a verification runs in a scratch kernel built for that run
 * and thrown away after it. Its jobs are therefore not this session's history,
 * and merging the two lists said they were. */
type Pane = "session" | "verify";
interface JobDetail {
  code?: string;
  intent?: string;
  truncated?: boolean;
  stdout_len?: number;
  elapsed?: number;
  window_alive?: boolean;
  stdout?: string;
  result_text?: string;
  error_text?: string;
}

async function jpost(url: string): Promise<{ [k: string]: unknown }> {
  try {
    const r = await sessionFetch(url, { method: "POST" });
    return await r.json().catch(() => ({}));
  } catch (e) {
    return { error: String(e) };
  }
}

export default function ObservePage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const base = withBase(`/session/${sessionId}`);
  // Terminal, once true: the id does not resolve and never will again, so the
  // page stops polling and stops offering anything that acts on the kernel.
  const [ended, setEnded] = useState(false);
  useDocumentTitle(
    `BioPB mcp - ${ended ? "ended" : "observe"}${sessionId ? ` · ${sessionId}` : ""}`,
  );

  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [verifyJobs, setVerifyJobs] = useState<JobSummary[]>([]);
  const [pane, setPane] = useState<Pane>("session");
  const [workflow, setWorkflow] = useState<WorkflowSummary | null>(null);
  const [details, setDetails] = useState<Record<string, JobDetail>>({});
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [status, setStatus] = useState("…");
  const [pollMs, setPollMs] = useState(3000);
  // The console is offered only when BOTH the control will proxy it (it is
  // loopback-bound) and this session child serves it (observe.console_enabled).
  // Either half false means every submit would 404, so render no editor at all.
  const [childConsole, setChildConsole] = useState(false);
  // The control's half of the answer for both local roots: whether it is
  // loopback-bound, and so whether it will proxy /console/* and /chat/* at all.
  const [controlLocal, setControlLocal] = useState(false);
  // Null until probed, and null again means unreachable rather than off — the
  // same distinction `console_enabled` draws below, and for the same reason.
  const [chatStatus, setChatStatus] = useState<ChatStatus | null>(null);
  // Both also require a session to run in: an editor and a composer on a dead
  // session are two more surfaces that look live and answer 404 on submit.
  const showConsole = childConsole && controlLocal && !ended;
  const showChat = !!chatStatus?.enabled && controlLocal && !ended;

  // The chat/work split, in pixels, remembered per browser. A preference, not
  // state anyone else needs, so localStorage rather than the server -- and every
  // access is guarded because a private window or blocked site data throws on
  // the accessor itself rather than returning null.
  const [chatWidth, setChatWidth] = useState<number | null>(() => {
    try {
      const stored = localStorage.getItem(CHAT_WIDTH_KEY);
      return stored === null
        ? null
        : clampChatWidth(Number(stored), window.innerWidth) || null;
    } catch {
      return null;
    }
  });
  const mainRef = useRef<HTMLElement | null>(null);
  const dragging = useRef(false);

  useEffect(() => {
    if (chatWidth === null) return; // nothing chosen yet; leave the default alone
    try {
      localStorage.setItem(CHAT_WIDTH_KEY, String(chatWidth));
    } catch {
      /* a preference that cannot be saved is still a working session */
    }
  }, [chatWidth]);

  const resizeTo = useCallback((clientX: number) => {
    const main = mainRef.current;
    if (!main) return;
    const width = clampChatWidth(
      clientX - main.getBoundingClientRect().left,
      window.innerWidth,
    );
    if (width) setChatWidth(width);
  }, []);

  const lastNewest = useRef<Record<Pane, string | null>>({
    session: null,
    verify: null,
  });
  // Latest expanded set + details for the poll closure (which fetches details for
  // open jobs) so poll stays stable — reading these through refs keeps the poll
  // interval from resubscribing on every toggle / detail update.
  const expandedRef = useRef(expanded);
  expandedRef.current = expanded;
  const detailsRef = useRef(details);
  detailsRef.current = details;

  const fetchDetail = useCallback(
    async (id: string) => {
      try {
        const r = await sessionFetch(
          base + "/api/jobs/" + encodeURIComponent(id),
        );
        if (!r.ok) return;
        const d: JobDetail = await r.json();
        setDetails((m) => ({ ...m, [id]: d }));
      } catch {
        /* keep last */
      }
    },
    [base],
  );

  const poll = useCallback(async () => {
    let r: Response;
    try {
      r = await sessionFetch(base + "/api/jobs");
    } catch {
      setStatus("unreachable");
      return;
    }
    // pollStatus below owns the diagnosis; all this has to do is not overwrite
    // a good job list with the empty one an error body parses as.
    if (sessionVerdict(r.status) !== "live") return;
    const data: {
      busy?: boolean;
      jobs?: JobSummary[];
      verify_jobs?: JobSummary[];
      workflow?: WorkflowSummary | null;
    } = await r.json().catch(() => ({}));
    if (data.busy) return; // transient; keep current render
    setWorkflow(data.workflow ?? null);
    const list = data.jobs || [];
    // Absent on an older child, which is not the same as "none ran": an empty
    // list is the honest render either way, and the pane says so itself.
    const verifyList = data.verify_jobs || [];
    setJobs(list);
    setVerifyJobs(verifyList);
    // Autocollapse, per list: a new job opens itself and closes its siblings,
    // and leaves the other kernel's open row alone. The panes do not share a
    // reading position, so a verification starting must not collapse the
    // session job someone is reading.
    let openSet = expandedRef.current;
    let changed = false;
    for (const [key, rows] of [
      ["session", list],
      ["verify", verifyList],
    ] as [Pane, JobSummary[]][]) {
      const newest = rows.length ? rows[rows.length - 1]!.job_id : null;
      if (newest === lastNewest.current[key]) continue;
      lastNewest.current[key] = newest;
      const mine = new Set(rows.map((j) => j.job_id));
      openSet = new Set([...openSet].filter((id) => !mine.has(id)));
      if (newest) openSet.add(newest);
      changed = true;
    }
    if (changed) setExpanded(openSet);
    // Refresh details for open jobs: running ones each poll, others once.
    for (const j of [...list, ...verifyList]) {
      if (!openSet.has(j.job_id)) continue;
      if (j.status === "running" || detailsRef.current[j.job_id] === undefined) {
        fetchDetail(j.job_id);
      }
    }
  }, [base, fetchDetail]);

  const pollStatus = useCallback(async () => {
    let r: Response;
    try {
      r = await sessionFetch(base + "/api/status");
    } catch {
      setStatus("unreachable");
      return;
    }
    const verdict = sessionVerdict(r.status);
    if (verdict !== "live") {
      if (verdict === "ended") {
        setEnded(true);
        setStatus("session ended");
      } else {
        setStatus("unreachable");
      }
      return;
    }
    try {
      const s = await r.json();
      if (typeof s.poll_interval_ms === "number") setPollMs(s.poll_interval_ms);
      // Only when the field is actually there. A degraded status payload (the
      // child's 503 with no kernel host, the proxy's 502 on a wedged session)
      // parses fine and carries no `console_enabled` — reading it as `false`
      // would unmount the editor and throw away a half-typed cell over a blip.
      // This is static config, not state: absent means unknown, not off.
      if (typeof s.console_enabled === "boolean")
        setChildConsole(s.console_enabled);
      const bits = [s.alive ? "alive" : "dead"];
      if (s.busy) bits.push("busy");
      if (!s.ready) bits.push("starting");
      setStatus("kernel: " + bits.join(" · "));
    } catch {
      setStatus("unreachable");
    }
  }, [base]);

  // Both halves of the local-root answer are config, fixed for the life of the
  // page — the control's follows its bind, the child's follows its config
  // file — so probe once rather than on every poll.
  useEffect(() => {
    let live = true;
    localRootsProxied().then((on) => {
      if (live) setControlLocal(on);
    });
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    let live = true;
    fetchChatStatus(base).then((s) => {
      if (live && s) setChatStatus(s);
    });
    return () => {
      live = false;
    };
  }, [base]);

  useEffect(() => {
    if (ended) return; // the record is pruned; there is nothing left to ask
    poll();
    pollStatus();
    const a = setInterval(poll, pollMs);
    const b = setInterval(pollStatus, pollMs);
    return () => {
      clearInterval(a);
      clearInterval(b);
    };
  }, [poll, pollStatus, pollMs, ended]);

  const toggle = useCallback(
    (id: string) => {
      setExpanded((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else {
          next.add(id);
          fetchDetail(id); // show detail immediately, don't wait for next poll
        }
        return next;
      });
    },
    [fetchDetail],
  );

  const saveNotebook = useCallback(async (kind: NotebookKind = "audit") => {
    let r: Response;
    try {
      r = await sessionFetch(
        base + "/api/notebook" + (kind === "workflow" ? "?workflow=1" : ""),
      );
    } catch (e) {
      alert("Save failed: " + e);
      return;
    }
    if (!r.ok) {
      alert("Save failed (" + r.status + ")");
      return;
    }
    const blob = await r.blob();
    const name = r.headers.get("X-Filename") || "biopb-mcp-session.ipynb";
    // Chromium (secure context; 127.0.0.1 counts): native Save-As picker.
    // Firefox/Safari lack it -> prompt for a name and save to Downloads.
    const picker = (
      window as unknown as {
        showSaveFilePicker?: (opts: unknown) => Promise<{
          createWritable: () => Promise<{
            write: (b: Blob) => Promise<void>;
            close: () => Promise<void>;
          }>;
        }>;
      }
    ).showSaveFilePicker;
    if (picker) {
      let handle;
      try {
        handle = await picker({
          suggestedName: name,
          types: [
            {
              description: "Jupyter notebook",
              accept: { "application/x-ipynb+json": [".ipynb"] },
            },
          ],
        });
      } catch (e) {
        if ((e as DOMException).name === "AbortError") return; // cancelled
      }
      if (handle) {
        const w = await handle.createWritable();
        await w.write(blob);
        await w.close();
        return;
      }
    }
    const chosen = prompt("Save notebook as:", name);
    if (chosen === null) return; // cancelled
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = chosen || name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, [base]);

  // The job holding the kernel, if any. Drives the Run button's disabled state,
  // so a collision is shown *before* the click rather than as a failed action:
  // one job runs at a time, and there is no preemption or queue.
  //
  // A verification counts, even though it runs elsewhere: the one job slot is
  // held in the session child and spans both kernels, so a console cell
  // submitted during one is refused.
  const verifyRunning = verifyJobs.find((j) => j.status === "running") ?? null;
  const running =
    (jobs?.find((j) => j.status === "running") ?? null) || verifyRunning;

  const runCell = useCallback(
    async (code: string): Promise<string | null> => {
      let r: Response;
      try {
        r = await sessionFetch(base + "/console/execute", {
          method: "POST",
          // Not decoration: a JSON content-type is one a cross-site form POST
          // cannot set, and the child requires it on this route for exactly
          // that reason. `sessionFetch` adds the bearer token alongside it.
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ code }),
        });
      } catch (e) {
        return String(e);
      }
      const d = await r.json().catch(() => ({}) as Record<string, unknown>);
      if (r.status === 409) {
        // Reachable despite the disabled button: `poll()` below is not awaited,
        // so a fast second click lands before the jobs list reports the cell as
        // running — and that collision is with the user's *own* job, which is
        // the branch whose wording has to agree with "you".
        const who =
          d.running_job_origin === "user"
            ? "you already have"
            : `${writerName(d.running_job_origin)} already has`;
        return `${who} a cell running (${d.running_job_id}). Wait for it, or interrupt it from its row above.`;
      }
      if (!r.ok) return String(d.error || `submit failed (${r.status})`);
      poll(); // show the new job immediately
      return null;
    },
    [base, poll],
  );

  // Offered on the running job's row rather than the header, because that is
  // what it does: the kernel runs one cell at a time, so interrupting *it* and
  // interrupting *that job* are the same act -- and on the row, the thing being
  // stopped is named, with its origin and its output beside it.
  //
  // No "nothing was running" dialog any more. The button exists only while a
  // row says running, so the one way to reach it idle is a job that finished
  // between the paint and the click -- and the row turning `ok` says so better
  // than an alert that reads as a mistake.
  const interrupt = useCallback(
    async (target: Pane) => {
      // Named, not inferred: with both kernels listed, "the running job" is
      // two different jobs depending on which list the row came from.
      await jpost(base + "/api/kernel/interrupt?target=" + target);
      poll();
    },
    [base, poll],
  );

  const restart = useCallback(async () => {
    if (!confirm("Hard-restart the kernel? All variables and layers are lost."))
      return;
    await jpost(base + "/api/kernel/restart");
    setJobs([]);
    setDetails({});
    setExpanded(new Set());
    lastNewest.current = { session: null, verify: null };
    poll();
  }, [base, poll]);

  // `jobs` is null until the first poll lands; the verification list is not,
  // because "none" is its ordinary state and is worth saying at once.
  const rows: JobSummary[] | null = pane === "session" ? jobs : verifyJobs;

  return (
    <div className="obs-page">
      <header>
        <img
          className="topbar-logo"
          src={withBase("/biopb-logo.png")}
          alt=""
          aria-hidden="true"
        />
        <h1>BioPB mcp - observe</h1>
        <span id="status">{status}</span>
        {/* Both act on the child, so both 404 once it is gone. A dead button is
            how the page told the user nothing was wrong. */}
        {ended ? null : (
          <>
            {/* The pane governs what the header offers, rather than the header
                offering both kernels' actions at once and leaving the reader to
                work out which kernel each one means. */}
            <div
              className="pane-toggle"
              role="tablist"
              aria-label="Which kernel to show"
            >
              <button
                role="tab"
                aria-selected={pane === "session"}
                className={pane === "session" ? "on" : ""}
                onClick={() => setPane("session")}
              >
                Session
              </button>
              <button
                role="tab"
                aria-selected={pane === "verify"}
                className={pane === "verify" ? "on" : ""}
                onClick={() => setPane("verify")}
                title="Workflow verifications, each in its own scratch kernel"
              >
                Verification
                {/* The one thing the session pane can no longer say for itself:
                    that the kernel is busy with something on the other side. */}
                {verifyRunning ? (
                  <span className="live" aria-label="running" />
                ) : verifyJobs.length ? (
                  <span className="count">{verifyJobs.length}</span>
                ) : null}
              </button>
            </div>
            {pane === "verify" ? (
              <button
                className="primary"
                disabled={!workflow}
                title={
                  workflow
                    ? `${workflow.title || "Verified workflow"} — ${
                        workflow.cells
                      } cell(s), each one proved by running it`
                    : "Nothing has passed verification in this session yet"
                }
                onClick={() => saveNotebook("workflow")}
              >
                ⤓ Save workflow
              </button>
            ) : (
              <button
                className="primary"
                title="Every cell this session ran, in order, as an audit trail"
                onClick={() => saveNotebook("audit")}
              >
                ⤓ Save notebook
              </button>
            )}
            {/* Session-side only. It destroys the user's variables and layers,
                and on a pane about throwaway kernels it would read as the
                harmless thing it is not. */}
            {pane === "session" ? (
              <button className="danger" onClick={restart}>
                Restart kernel
              </button>
            ) : null}
          </>
        )}
      </header>
      <main
        ref={mainRef}
        className={showChat ? "with-chat" : ""}
        style={
          chatWidth === null
            ? undefined
            : ({ ["--chat-w" as string]: `${chatWidth}px` } as React.CSSProperties)
        }
      >
        {ended ? (
          <div className="ended" role="status">
            <strong>This session has ended.</strong> Its kernel and viewer are
            gone — anything below is the last state this page saw, not live.{" "}
            <a href={withBase("/")}>Back to the dashboard</a>
          </div>
        ) : null}
        {showChat && chatStatus ? (
          <ChatPane base={base} status={chatStatus} pollMs={pollMs} />
        ) : null}
        {showChat && chatStatus ? (
          // Pointer events rather than mouse: capture keeps the drag alive when
          // the cursor outruns the handle, which it will on a fast drag.
          <div
            className="splitter"
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize the chat pane"
            tabIndex={0}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              dragging.current = true;
            }}
            onPointerMove={(e) => {
              if (dragging.current) resizeTo(e.clientX);
            }}
            onPointerUp={(e) => {
              dragging.current = false;
              e.currentTarget.releasePointerCapture(e.pointerId);
            }}
            onKeyDown={(e) => {
              // Usable without a pointer, and the only way to nudge it exactly.
              const step = e.key === "ArrowLeft" ? -16 : e.key === "ArrowRight" ? 16 : 0;
              if (!step) return;
              e.preventDefault();
              const current = chatWidth ?? defaultChatWidth(window.innerWidth);
              const width = clampChatWidth(current + step, window.innerWidth);
              if (width) setChatWidth(width);
            }}
          />
        ) : null}
        <div className="work">
          {showConsole && pane === "session" ? (
            <ConsolePanel running={running} onRun={runCell} />
          ) : null}
          {pane === "verify" ? (
            <div className="pane-note">
              Each row ran in its own scratch kernel — a headless one built for
              the run and thrown away after it. Nothing here touched your
              variables, layers or viewer.
            </div>
          ) : null}
          <div id="jobs">
            {rows == null ? (
              // Never fetched anything, and on an ended session never will:
              // "loading…" for ever is the same lie in miniature.
              ended ? null : <div className="empty">loading…</div>
            ) : rows.length === 0 ? (
              <div className="empty">
                {pane === "session"
                  ? "no jobs yet"
                  : "nothing verified in this session yet"}
              </div>
            ) : (
              // newest-first
              [...rows].reverse().map((j) => (
                <JobRow
                  key={j.job_id}
                  job={j}
                  open={expanded.has(j.job_id)}
                  detail={details[j.job_id]}
                  onToggle={() => toggle(j.job_id)}
                  onInterrupt={() => interrupt(pane)}
                />
              ))
            )}
          </div>
        </div>
      </main>
      <style>{OBS_CSS}</style>
    </div>
  );
}

/** The user's own cell, run in the same kernel the agent drives.
 *
 * Busy is rendered as *state* — a disabled button naming who holds the kernel —
 * not as an error after the click. A rejected cell is the serialization rule
 * working, and showing it as a red failure would train the user to reach for
 * Interrupt reflexively. */
function ConsolePanel({
  running,
  onRun,
}: {
  running: JobSummary | null;
  onRun: (code: string) => Promise<string | null>;
}) {
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const busy = running !== null;

  const submit = useCallback(async () => {
    if (!code.trim() || busy || submitting) return;
    setSubmitting(true);
    setError(await onRun(code));
    setSubmitting(false);
  }, [code, busy, submitting, onRun]);

  const label = busy
    ? `kernel busy · ${running!.job_id} (${writerName(running!.origin)})`
    : submitting
      ? "running…"
      : "▶ Run";

  return (
    <div className="console">
      {/* The Run button rides the label row rather than a bar of its own: that
          bar cost a whole line of a column that is now a fixed height, and the
          line it cost came off the job list. The error sits between them, where
          there was nothing but empty space. */}
      <div className="console-head">
        <span className="label">your cell — runs in this session&apos;s kernel</span>
        <span className="console-err">{error || ""}</span>
        <button
          className="primary"
          disabled={busy || submitting || !code.trim()}
          onClick={submit}
        >
          {label}
        </button>
      </div>
      <textarea
        className="console-input"
        value={code}
        spellCheck={false}
        placeholder="viewer.layers"
        onChange={(e) => setCode(e.target.value)}
        title="Ctrl+Enter to run"
        onKeyDown={(e) => {
          // Ctrl/Cmd+Enter submits; plain Enter stays a newline (this is code).
          if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            submit();
          }
        }}
      />
    </div>
  );
}

export function JobRow({
  job,
  open,
  detail,
  onToggle,
  onInterrupt,
}: {
  job: JobSummary;
  open: boolean;
  detail: JobDetail | undefined;
  onToggle: () => void;
  onInterrupt: () => void;
}) {
  const outRef = useRef<HTMLPreElement | null>(null);
  // Whether the user is parked at the bottom of the output; a live job then keeps
  // the tail visible, but scrolling up to read is not yanked back.
  const atBottom = useRef(true);

  const text =
    ((detail?.stdout || "") +
      (detail?.result_text ? "\n" + detail.result_text : "") +
      (detail?.error_text ? "\n" + detail.error_text : "")) ||
    "(no output)";

  useLayoutEffect(() => {
    const pre = outRef.current;
    if (!pre) return;
    if (job.status === "running" && atBottom.current) {
      pre.scrollTop = pre.scrollHeight;
    }
  }, [text, job.status]);

  const note = detail?.truncated
    ? "stdout truncated to last of " + detail.stdout_len + " chars · "
    : "";
  const meta =
    detail == null
      ? ""
      : note +
        detail.elapsed +
        "s" +
        (detail.window_alive === false ? " · viewer window closed" : "");

  return (
    <div className={"job" + (open ? " open" : "")}>
      <div className="row" onClick={onToggle}>
        <span className="jid">{job.job_id}</span>
        {/* Provenance is worth showing for anything that is not the MCP
            client: "mcp" is the norm here and a badge on every row would be
            noise, but a cell run by anyone else must not read as its. */}
        {job.origin && job.origin !== "mcp" ? (
          <span className="badge you">{writerName(job.origin)}</span>
        ) : null}
        <span className={"badge " + job.status}>{job.status}</span>
        {/* Why over what: `arr = arr[..., 1]` is a fact about the code, and
            "isolate the nuclei channel" is a fact about the reader's data. The
            source is one click away either way, so the row spends its one line
            on the half that is not already reconstructable from the other. */}
        {job.intent_preview ? (
          <span className="intent" title={job.intent_preview}>
            {job.intent_preview}
          </span>
        ) : (
          <span className="preview">{job.code_preview || ""}</span>
        )}
        {/* A verification's size is a fact about the run, not about its code,
            so it sits with the elapsed time rather than displacing the title. */}
        {job.verify && job.intent_preview ? (
          <span className="preview">{job.code_preview || ""}</span>
        ) : null}
        <span className="elapsed">{job.elapsed}s</span>
        {job.status === "running" ? (
          // The whole row toggles the detail, so this has to keep its click:
          // reaching for Interrupt and collapsing the output you were reading
          // is the one mistake the placement makes possible.
          <button
            className="job-stop"
            title="Interrupt this cell"
            onClick={(e) => {
              e.stopPropagation();
              onInterrupt();
            }}
          >
            interrupt
          </button>
        ) : null}
      </div>
      <div className="detail">
        {open && detail ? (
          <>
            {detail.intent ? (
              // In full here, because the row caps it at one line.
              <>
                <div className="label">intent</div>
                <div className="intent-full">{detail.intent}</div>
              </>
            ) : null}
            {detail.code ? (
              <>
                <div className="label">code</div>
                <pre className="code">{detail.code}</pre>
              </>
            ) : null}
            <div className="label">output</div>
            <div className="meta">{meta}</div>
            <pre
              className="out"
              ref={outRef}
              onScroll={() => {
                const pre = outRef.current;
                if (!pre) return;
                atBottom.current =
                  pre.scrollHeight - pre.scrollTop - pre.clientHeight < 4;
              }}
            >
              {text}
            </pre>
          </>
        ) : null}
      </div>
    </div>
  );
}

const OBS_CSS = `
  .obs-page { min-height: 100vh; background: #111; color: #ddd;
              font: 14px/1.5 system-ui, sans-serif; }
  .obs-page header { padding: 10px 16px; background: #1b1b1b; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 12px; position: sticky; top: 0; }
  .obs-page h1 { font-size: 15px; margin: 0; font-weight: 600; }
  .obs-page #status { font-size: 12px; color: #9aa; margin-right: auto; }
  .obs-page button { font: inherit; padding: 4px 10px; border: 1px solid #444; border-radius: 4px;
           background: #222; color: #ddd; cursor: pointer; }
  .obs-page button:hover { background: #2c2c2c; }
  .obs-page button.danger { border-color: #844; }
  .obs-page button.primary { background: #1d6b3f; border-color: #2a5; color: #eafff0;
                   font-weight: 600; margin-right: 6px; }
  .obs-page button.primary:hover { background: #25804b; }
  .obs-page button.primary:disabled { opacity: .5; cursor: default; background: #1d6b3f; }
  .obs-page .pane-toggle { display: flex; margin-right: 12px; }
  .obs-page .pane-toggle button { border-radius: 0; display: flex; align-items: center;
           gap: 6px; color: #999; }
  .obs-page .pane-toggle button:first-child { border-radius: 4px 0 0 4px; }
  .obs-page .pane-toggle button:last-child { border-radius: 0 4px 4px 0; border-left: none; }
  .obs-page .pane-toggle button.on { background: #333; color: #eee; }
  .obs-page .pane-toggle .live { width: 7px; height: 7px; border-radius: 50%;
           background: #4c9; animation: obs-pulse 1.4s ease-in-out infinite; }
  @keyframes obs-pulse { 50% { opacity: .25; } }
  .obs-page .pane-toggle .count { font-size: 11px; color: #888; }
  .obs-page .pane-note { color: #8a8a8a; font-size: 12px; padding: 8px 4px 12px; }
  .obs-page main { padding: 12px 16px; }
  /* The thread beside the jobs it drives: a chat cell shows up in that list,
     and its live stdout is what stands in for the thread's missing stream. */
  /* Both columns are their own scroll region, the height of the viewport, so
     the page itself never scrolls: the console stays put while the job list
     moves under it, and the composer stays put while the thread moves. A
     single page scroll took the console off screen exactly when a running job
     made the list long -- which is when you want to type the next cell. */
  .obs-page main.with-chat { display: flex; gap: 0; align-items: flex-start;
             height: calc(100vh - 58px); box-sizing: border-box; overflow: hidden; }
  /* The fallback is the original rule, so an untouched pane is sized exactly as
     it was; --chat-w exists only once the splitter has been dragged. */
  /* min-width:0 because a flex item's auto minimum is its content's min-content
     width: without it a wide line of chat output widens the pane mid-turn, and
     the width stops being the person's to set. Overflow belongs to the thread,
     which scrolls. */
  .obs-page main.with-chat .chat { flex: 0 0 var(--chat-w, clamp(340px, 34%, 520px));
             min-width: 0; height: 100%; }
  .obs-page main.with-chat .work { flex: 1; min-width: 0; height: 100%;
             display: flex; flex-direction: column; }
  /* The console keeps its natural height; only the job list scrolls. */
  .obs-page main.with-chat .work .console { flex: 0 0 auto; }
  .obs-page main.with-chat #jobs { flex: 1; min-height: 0; overflow-y: auto;
             padding-right: 2px; }
  .obs-page .splitter { flex: 0 0 10px; align-self: stretch; cursor: col-resize;
             background: transparent; border: none; position: relative; }
  .obs-page .splitter::after { content: ""; position: absolute; top: 0; bottom: 0;
             left: 4px; width: 2px; background: #2a2a2a; }
  .obs-page .splitter:hover::after, .obs-page .splitter:focus-visible::after {
             background: #2a5; }
  .obs-page .splitter:focus-visible { outline: none; }
  @media (max-width: 900px) {
    /* Stacked: one page scroll again, and nothing to drag. */
    .obs-page main.with-chat { display: block; height: auto; overflow: visible; }
    .obs-page main.with-chat .chat { height: 60vh; margin-bottom: 12px; }
    .obs-page main.with-chat .work { height: auto; display: block; }
    .obs-page main.with-chat #jobs { overflow-y: visible; }
    .obs-page .splitter { display: none; }
  }
  .obs-page .job { border: 1px solid #333; border-radius: 5px; margin-bottom: 8px; overflow: hidden; }
  .obs-page .row { display: flex; gap: 10px; align-items: center; padding: 8px 12px; cursor: pointer; }
  .obs-page .row:hover { background: #1a1a1a; }
  .obs-page .jid { font-weight: 600; }
  .obs-page .badge { font-size: 11px; padding: 1px 7px; border-radius: 10px; text-transform: uppercase; }
  .obs-page .you { background: #34305a; color: #b9b0ff; }
  .obs-page .running { background: #243; color: #7e7; }
  .obs-page .ok { background: #234; color: #8bf; }
  .obs-page .error { background: #422; color: #f99; }
  .obs-page .interrupted { background: #324; color: #c9f; }
  .obs-page .preview { color: #8a8; font-family: ui-monospace, Menlo, monospace; font-size: 12px;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; min-width: 0; }
  .obs-page .intent { color: #bcd; flex: 1; min-width: 0;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .obs-page .intent-full { color: #bcd; }
  .obs-page .elapsed { color: #888; font-size: 12px; margin-left: auto; }
  /* Beside the elapsed time, which margin-left:auto has already pushed to
     the right edge -- so the row reads left to right as what ran, how long it
     has been running, and the way to stop it. */
  .obs-page .job-stop { font-size: 11px; padding: 1px 8px; border-radius: 10px;
                        border: 1px solid #533; background: #2a1a1a;
                        color: #f99; cursor: pointer; }
  .obs-page .job-stop:hover { background: #3a2020; border-color: #744; }
  .obs-page .detail { border-top: 1px solid #333; padding: 10px 12px; display: none; }
  .obs-page .job.open .detail { display: block; }
  .obs-page .label { color: #6a8; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; margin: 8px 0 2px; }
  .obs-page .label:first-child { margin-top: 0; }
  .obs-page pre { white-space: pre-wrap; word-break: break-word; margin: 0;
        background: #0c0c0c; padding: 8px; border-radius: 4px; max-height: 50vh; overflow: auto;
        font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
  .obs-page pre.code { background: #0a0d0a; border-left: 2px solid #2a5; max-height: 30vh; }
  .obs-page .meta { color: #888; font-size: 12px; margin-bottom: 4px; }
  .obs-page .empty { color: #777; padding: 20px; text-align: center; }
  .obs-page .console { border: 1px solid #333; border-radius: 5px; padding: 10px 12px;
             margin-bottom: 12px; background: #161616; }
  .obs-page .console-input { width: 100%; box-sizing: border-box; min-height: 68px;
             resize: vertical; background: #0c0c0c; color: #ddd; border: 1px solid #333;
             border-radius: 4px; padding: 8px;
             font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
  .obs-page .console-input:focus { outline: none; border-color: #2a5; }
  .obs-page .console-head { display: flex; align-items: center; gap: 10px;
             margin-bottom: 6px; }
  .obs-page .console-head .label { margin: 0; flex: 0 0 auto; }
  .obs-page .console-head .console-err { flex: 1; min-width: 0; text-align: right;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .obs-page .console button:disabled { opacity: .55; cursor: default; background: #222; }
  .obs-page .console-err { color: #f99; font-size: 12px; }
  .obs-page .ended { border: 1px solid #744; background: #241a1a; color: #fbb;
             border-radius: 5px; padding: 10px 12px; margin-bottom: 12px; }
  .obs-page .ended strong { color: #fdd; }
  .obs-page .ended a { color: #fbb; }
`;
