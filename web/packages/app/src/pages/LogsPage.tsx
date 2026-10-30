import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useDocumentTitle } from "../hooks/useDocumentTitle";
import { captureUrlToken } from "../auth";
import { SessionLocked, sessionFetch } from "../utils/sessionFetch";
import { withBase } from "../base";

// A dedicated data-plane log monitor for the control dashboard. Polls the
// control's own `GET /api/data_plane/logs` (the tail of the tensor server's
// stdout/stderr log the supervisor writes) and renders it as a follow-the-tail
// console. Data-plane only by design — the control owns that log path via the
// supervisor; the control's own log and per-session mcp logs are out of scope.
//
// Periodic tail (not a live stream): the endpoint returns the last N lines and
// this page re-fetches on an interval while "Follow" is on, matching how the rest
// of the dashboard already polls. No streaming lifecycle to manage.

const POLL_MS = 3000;
const LINE_CHOICES = [200, 500, 1000, 2000] as const;

interface LogsResponse {
  path: string | null;
  exists: boolean;
  size?: number;
  lines: string[];
  truncated: boolean;
  note?: string;
  error?: string;
}

function formatBytes(n?: number): string {
  if (n == null) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function LogsPage() {
  // `?source=` picks whose log: the last agent-client launch, the last session
  // the dashboard launched, an algorithm server's (`?name=` the entry), or
  // (default) the data plane.
  const params = useSearchParams()[0];
  const source = params.get("source") ?? "data_plane";
  const what =
    source === "agents"
      ? "Agent client"
      : source === "sessions"
        ? "Session"
        : source === "algorithms"
          ? "Algorithm"
          : "Data-plane";
  useDocumentTitle(`BioPB control - ${what.toLowerCase()} logs`);

  // Algorithm logs are per entry: the script entries the control runs (a url
  // entry's logs belong to whoever runs it), and the one being read.
  const [entries, setEntries] = useState<string[] | null>(null);
  const [entry, setEntry] = useState<string | null>(params.get("name"));

  const [data, setData] = useState<LogsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lines, setLines] = useState<number>(500);
  const [follow, setFollow] = useState(true);
  const [loading, setLoading] = useState(false);

  const bodyRef = useRef<HTMLPreElement | null>(null);
  // When following, keep the view pinned to the newest line after each refresh.
  const followRef = useRef(follow);
  followRef.current = follow;

  const load = useCallback(
    async (n: number) => {
    const url =
      source === "algorithms"
        ? entry && `/api/algorithms/logs?name=${encodeURIComponent(entry)}&lines=${n}`
        : `/api/${source}/logs?lines=${n}`;
    if (!url) return;
    setLoading(true);
    try {
      const r = await sessionFetch(withBase(url), { cache: "no-store" });
      if (!r.ok) {
        const body = (await r.json().catch(() => ({}))) as Partial<LogsResponse>;
        throw new Error(body?.error || `Could not load logs (HTTP ${r.status}).`);
      }
      const body = (await r.json()) as LogsResponse;
      setData(body);
      setError(null);
    } catch (err) {
      // Already navigating to the unlock page: nothing to report.
      if (err instanceof SessionLocked) return;
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  },
    [source, entry],
  );

  useEffect(() => {
    if (source !== "algorithms") return;
    sessionFetch(withBase("/api/algorithms"))
      .then((r) => r.json())
      .then((body) => {
        const names: string[] = (body?.servers ?? [])
          .filter((s: { kind: string }) => s.kind === "script")
          .map((s: { name: string }) => s.name);
        setEntries(names);
        setEntry((cur) => (cur && names.includes(cur) ? cur : (names[0] ?? null)));
      })
      .catch((err) => {
        if (!(err instanceof SessionLocked)) setError(String(err));
      });
  }, [source]);

  // Capture a ?token= handed over by the one-time access URL before the first
  // fetch, so a remote-mode deep link to /logs authenticates rather than bouncing.
  useEffect(() => {
    captureUrlToken();
  }, []);

  // Initial load + re-load whenever the line count changes.
  useEffect(() => {
    load(lines);
  }, [load, lines]);

  // Poll on the interval only while following.
  useEffect(() => {
    if (!follow) return;
    const id = setInterval(() => load(lines), POLL_MS);
    return () => clearInterval(id);
  }, [follow, lines, load]);

  // Pin to the bottom after a refresh while following.
  useEffect(() => {
    if (followRef.current && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [data]);

  const text = data?.lines?.length ? data.lines.join("\n") : "";

  return (
    <div className="ctrl-logs">
      <header>
        <img
          className="topbar-logo"
          src={withBase("/biopb-logo.png")}
          alt=""
          aria-hidden="true"
        />
        <h1>{what} logs</h1>
        {source === "algorithms" && entries && entries.length ? (
          <label className="lines-sel">
            Entry
            <select value={entry ?? ""} onChange={(e) => setEntry(e.target.value)}>
              {entries.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <span className="conn">
          {data?.exists
            ? `${data.lines.length} line(s)${
                data.size != null ? " · " + formatBytes(data.size) : ""
              }`
            : loading
              ? "loading…"
              : "no log"}
        </span>
        <label className="lines-sel">
          Lines
          <select
            value={lines}
            onChange={(e) => setLines(Number(e.target.value))}
          >
            {LINE_CHOICES.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <button
          className={follow ? "active" : ""}
          onClick={() => setFollow((f) => !f)}
          title="Auto-refresh and stick to the newest line"
        >
          {follow ? "Following" : "Follow"}
        </button>
        <button onClick={() => load(lines)} disabled={loading}>
          Refresh
        </button>
        <Link className="hdr-link" to="/">
          Dashboard
        </Link>
      </header>

      <main>
        {data?.path ? (
          <div className="logpath">
            <code>{data.path}</code>
            {data.truncated ? (
              <span className="trunc" title="Older lines exist above this window">
                older lines truncated
              </span>
            ) : null}
          </div>
        ) : null}

        {error ? (
          <div className="banner error">Could not load logs: {error}</div>
        ) : null}

        {source === "algorithms" && entries && !entries.length ? (
          <div className="banner empty">
            No script entries: a server someone else runs keeps its own logs.
          </div>
        ) : null}

        {data && !data.exists ? (
          <div className="banner empty">
            {data.note ||
              (data.path
                ? `The ${what.toLowerCase()} log file does not exist yet.`
                : `No ${what.toLowerCase()} log file is configured.`)}
          </div>
        ) : null}

        {data?.exists ? (
          text ? (
            <pre className="console" ref={bodyRef}>
              {text}
            </pre>
          ) : (
            <div className="banner empty">The log is empty.</div>
          )
        ) : null}
      </main>
      <style>{LOGS_CSS}</style>
    </div>
  );
}

const LOGS_CSS = `
  .ctrl-logs { min-height: 100vh; display: flex; flex-direction: column;
               background: #111; color: #ddd; font: 14px/1.5 system-ui, sans-serif; }
  .ctrl-logs header { padding: 10px 16px; background: #1b1b1b; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 12px; position: sticky; top: 0; }
  .ctrl-logs h1 { font-size: 15px; margin: 0; font-weight: 600; }
  .ctrl-logs .conn { font-size: 12px; color: #9aa; margin-left: auto; }
  .ctrl-logs .lines-sel { font-size: 12px; color: #9aa; display: flex; align-items: center; gap: 6px; }
  .ctrl-logs select { font: inherit; font-size: 12px; padding: 2px 4px; background: #222;
           color: #ddd; border: 1px solid #444; border-radius: 4px; }
  .ctrl-logs button { font: inherit; padding: 4px 12px; border: 1px solid #444; border-radius: 4px;
           background: #222; color: #ddd; cursor: pointer; }
  .ctrl-logs button:hover:not(:disabled) { background: #2c2c2c; }
  .ctrl-logs button:disabled { opacity: .45; cursor: default; }
  .ctrl-logs button.active { border-color: #2a6; color: #7e7; }
  .ctrl-logs a.hdr-link { font-size: 12px; color: #8bf; text-decoration: none; }
  .ctrl-logs a.hdr-link:hover { text-decoration: underline; }
  .ctrl-logs main { flex: 1; display: flex; flex-direction: column; min-height: 0; padding: 12px 16px; gap: 10px; }
  .ctrl-logs .logpath { font-size: 12px; color: #778; display: flex; align-items: center; gap: 12px; }
  .ctrl-logs .logpath code { font-family: ui-monospace, Menlo, monospace; word-break: break-all; }
  .ctrl-logs .trunc { color: #c96; font-size: 11px; border: 1px solid #543; border-radius: 3px; padding: 0 6px; }
  .ctrl-logs .banner { padding: 10px 12px; border-radius: 6px; font-size: 13px; }
  .ctrl-logs .banner.error { background: #2a1616; border: 1px solid #633; color: #f99; }
  .ctrl-logs .banner.empty { background: #161616; border: 1px solid #333; color: #889; }
  .ctrl-logs .console { flex: 1; min-height: 0; margin: 0; overflow: auto; background: #0c0c0c;
           border: 1px solid #262626; border-radius: 6px; padding: 12px 14px; white-space: pre-wrap;
           word-break: break-word; font: 12px/1.5 ui-monospace, Menlo, monospace; color: #cdd; }
  @media (max-width: 640px) {
    .ctrl-logs header { flex-wrap: wrap; row-gap: 6px; }
    .ctrl-logs main { padding: 10px; }
  }
`;
