import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { useParams } from "react-router-dom";
import { useDocumentTitle } from "../hooks/useDocumentTitle";

// Per-session observe UI, ported from the buildless _OBSERVE_HTML that each MCP
// session child used to serve at /observe. The child now serves only /api/*; the
// control front serves this SPA shell at /session/<id>/observe and proxies
// /session/<id>/api/* to the child. The API base is therefore the session prefix.

interface JobSummary {
  job_id: string;
  status: string; // running | ok | error | cancelled | interrupted
  elapsed: number;
  code_preview?: string;
}
interface JobDetail {
  code?: string;
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
    const r = await fetch(url, { method: "POST" });
    return await r.json().catch(() => ({}));
  } catch (e) {
    return { error: String(e) };
  }
}

export default function ObservePage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const base = `/session/${sessionId}`;
  useDocumentTitle(
    `BioPB mcp - observe${sessionId ? ` · ${sessionId}` : ""}`,
  );

  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [details, setDetails] = useState<Record<string, JobDetail>>({});
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [status, setStatus] = useState("…");
  const [pollMs, setPollMs] = useState(3000);

  const lastNewest = useRef<string | null>(null);
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
        const r = await fetch(base + "/api/jobs/" + encodeURIComponent(id));
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
    let data: { busy?: boolean; jobs?: JobSummary[] };
    try {
      data = await (await fetch(base + "/api/jobs")).json();
    } catch {
      setStatus("unreachable");
      return;
    }
    if (data.busy) return; // transient; keep current render
    const list = data.jobs || [];
    if (!list.length) {
      setJobs([]);
      setExpanded(new Set());
      lastNewest.current = null;
      return;
    }
    const newest = list[list.length - 1]!.job_id;
    let openSet = expandedRef.current;
    if (newest !== lastNewest.current) {
      // autocollapse all but the newest when a new job appears
      openSet = new Set([newest]);
      setExpanded(openSet);
      lastNewest.current = newest;
    }
    setJobs(list);
    // Refresh details for open jobs: running ones each poll, others once.
    for (const j of list) {
      if (!openSet.has(j.job_id)) continue;
      if (j.status === "running" || detailsRef.current[j.job_id] === undefined) {
        fetchDetail(j.job_id);
      }
    }
  }, [base, fetchDetail]);

  const pollStatus = useCallback(async () => {
    try {
      const s = await (await fetch(base + "/api/status")).json();
      if (typeof s.poll_interval_ms === "number") setPollMs(s.poll_interval_ms);
      const bits = [s.alive ? "alive" : "dead"];
      if (s.headless) bits.push("headless");
      if (s.busy) bits.push("busy");
      if (!s.ready) bits.push("starting");
      setStatus("kernel: " + bits.join(" · "));
    } catch {
      setStatus("unreachable");
    }
  }, [base]);

  useEffect(() => {
    poll();
    pollStatus();
    const a = setInterval(poll, pollMs);
    const b = setInterval(pollStatus, pollMs);
    return () => {
      clearInterval(a);
      clearInterval(b);
    };
  }, [poll, pollStatus, pollMs]);

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

  const saveNotebook = useCallback(async () => {
    let r: Response;
    try {
      r = await fetch(base + "/api/notebook");
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

  const interrupt = useCallback(async () => {
    const d = await jpost(base + "/api/kernel/interrupt");
    if (d && d.interrupted === false && d.status === "idle")
      alert("No running job.");
    poll();
  }, [base, poll]);

  const restart = useCallback(async () => {
    if (!confirm("Hard-restart the kernel? All variables and layers are lost."))
      return;
    await jpost(base + "/api/kernel/restart");
    setJobs([]);
    setDetails({});
    setExpanded(new Set());
    lastNewest.current = null;
    poll();
  }, [base, poll]);

  return (
    <div className="obs-page">
      <header>
        <img
          className="topbar-logo"
          src={`${import.meta.env.BASE_URL}biopb-logo.png`}
          alt=""
          aria-hidden="true"
        />
        <h1>BioPB mcp - observe</h1>
        <span id="status">{status}</span>
        <button className="primary" onClick={saveNotebook}>
          ⤓ Save notebook
        </button>
        <button onClick={interrupt}>Interrupt</button>
        <button className="danger" onClick={restart}>
          Restart kernel
        </button>
      </header>
      <main>
        <div id="jobs">
          {jobs == null ? (
            <div className="empty">loading…</div>
          ) : jobs.length === 0 ? (
            <div className="empty">no jobs yet</div>
          ) : (
            // newest-first
            [...jobs].reverse().map((j) => (
              <JobRow
                key={j.job_id}
                job={j}
                open={expanded.has(j.job_id)}
                detail={details[j.job_id]}
                onToggle={() => toggle(j.job_id)}
              />
            ))
          )}
        </div>
      </main>
      <style>{OBS_CSS}</style>
    </div>
  );
}

function JobRow({
  job,
  open,
  detail,
  onToggle,
}: {
  job: JobSummary;
  open: boolean;
  detail: JobDetail | undefined;
  onToggle: () => void;
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
        <span className={"badge " + job.status}>{job.status}</span>
        <span className="preview">{job.code_preview || ""}</span>
        <span className="elapsed">{job.elapsed}s</span>
      </div>
      <div className="detail">
        {open && detail ? (
          <>
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
  .obs-page main { padding: 12px 16px; }
  .obs-page .job { border: 1px solid #333; border-radius: 5px; margin-bottom: 8px; overflow: hidden; }
  .obs-page .row { display: flex; gap: 10px; align-items: center; padding: 8px 12px; cursor: pointer; }
  .obs-page .row:hover { background: #1a1a1a; }
  .obs-page .jid { font-weight: 600; }
  .obs-page .badge { font-size: 11px; padding: 1px 7px; border-radius: 10px; text-transform: uppercase; }
  .obs-page .running { background: #243; color: #7e7; }
  .obs-page .ok { background: #234; color: #8bf; }
  .obs-page .error { background: #422; color: #f99; }
  .obs-page .cancelled { background: #432; color: #fc9; }
  .obs-page .interrupted { background: #324; color: #c9f; }
  .obs-page .preview { color: #8a8; font-family: ui-monospace, Menlo, monospace; font-size: 12px;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; min-width: 0; }
  .obs-page .elapsed { color: #888; font-size: 12px; margin-left: auto; }
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
`;
