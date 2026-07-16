import { useCallback, useEffect, useState } from "react";
import { useDocumentTitle } from "../hooks/useDocumentTitle";
import {
  authHeaders,
  authRequired,
  captureUrlToken,
  clearToken,
  getToken,
  redirectToUnlock,
} from "../auth";

// The control's own root dashboard, ported from the buildless _DASHBOARD_HTML
// that _control.py used to serve. Talks to the control's own token-gated /api/*
// at this single origin: /api/status + /api/sessions (polled), /api/agents +
// /api/algorithms (on load / manual refresh), and POSTs the data-plane verbs.

const POLL_MS = 3000;

interface DataPlane {
  state?: string;
  grpc_url?: string;
  web_url?: string;
  pid?: number | null;
  restarts?: number | null;
  last_error?: string | null;
}
interface SessionRec {
  session_id: string;
  port: number | string;
  started_at?: number | null;
  observe_url: string;
  // Best-effort kernel state probed by the control (none|starting|ready|busy|
  // error|unknown); decorative, may be absent on an older control.
  kernel?: string;
}
interface AgentRec {
  id: string;
  name: string;
  state: string; // "registered" | "installed" | "not_installed"
  drifted?: boolean;
}
interface AlgoRec {
  target: string;
  state: string; // "serving" | "down" | "unknown" | "invalid"
  scheme?: string;
  single_op?: boolean;
  op_count?: number;
  ops?: string[];
  error?: string;
}
// The kernel-namespace "bring your own tool" plugins (biopb-mcp#92), folded into
// /api/algorithms. A static listing (files + installed packages), NOT the live
// set of names a running kernel bound — the control reads it without executing
// user code (invariant I2).
interface PluginsRec {
  dir: string;
  files: { name: string; summary: string }[];
  entry_points: { name: string; dist: string }[];
}

// The control's /api/* is token-gated at this single origin. Attach the stored
// token ('biopb_token') as a Bearer header via the shared auth helper; in the
// common local deployment there is no token and the header is simply absent. A
// 401 means remote mode with a missing/stale token, so bounce to /unlock.
async function fetchAuth(
  url: string,
  opts: RequestInit = {},
): Promise<Response> {
  const r = await fetch(url, {
    ...opts,
    headers: authHeaders(opts.headers as Record<string, string> | undefined),
  });
  if (r.status === 401) {
    redirectToUnlock();
  }
  return r;
}

async function jpost(url: string): Promise<{ error?: string; data_plane?: DataPlane }> {
  try {
    const r = await fetchAuth(url, { method: "POST" });
    return await r.json().catch(() => ({}));
  } catch (e) {
    return { error: String(e) };
  }
}

export default function DashboardPage() {
  useDocumentTitle("BioPB control - dashboard");
  const [conn, setConn] = useState("…");
  // Connection health drives the status pill's color (green ok / red down),
  // mirroring the admin pages' status-pill. null = not yet probed (neutral).
  const [connOk, setConnOk] = useState<boolean | null>(null);
  const [dataPlane, setDataPlane] = useState<DataPlane>({ state: "unknown" });
  const [sessions, setSessions] = useState<SessionRec[] | null>(null);
  const [agents, setAgents] = useState<AgentRec[] | null>(null);
  const [algos, setAlgos] = useState<AlgoRec[] | null>(null);
  const [plugins, setPlugins] = useState<PluginsRec | null>(null);
  const [verbBusy, setVerbBusy] = useState(false);
  const [agentsBusy, setAgentsBusy] = useState(false);
  // Whether a token is held (remote mode). Lock only means something when there
  // is a token to drop; local mode has none, so the button is disabled.
  const [hasToken, setHasToken] = useState(() => !!getToken());

  const pollStatus = useCallback(async () => {
    try {
      const s = await (await fetchAuth("/api/status")).json();
      setConn("control: ok · " + (s.sessions || 0) + " session(s)");
      setConnOk(true);
      setDataPlane(s.data_plane || {});
    } catch {
      setConn("control unreachable");
      setConnOk(false);
    }
  }, []);

  const pollSessions = useCallback(async () => {
    try {
      const data = await (await fetchAuth("/api/sessions")).json();
      setSessions((data && data.sessions) || []);
    } catch {
      /* keep last */
    }
  }, []);

  const pollAgents = useCallback(async () => {
    try {
      const data = await (await fetchAuth("/api/agents")).json();
      setAgents((data && data.agents) || []);
    } catch {
      /* keep last */
    }
  }, []);

  const pollAlgos = useCallback(async () => {
    try {
      const data = await (await fetchAuth("/api/algorithms")).json();
      setAlgos((data && data.servers) || []);
      setPlugins((data && data.plugins) || null);
    } catch {
      /* keep last */
    }
  }, []);

  // Mode-driven unlock gate. Capture a ?token= handed over by the one-time
  // access URL, then — only in remote mode, where the control's public /health
  // advertises auth_required — bounce to /unlock if we still have no token. Local
  // mode advertises auth_required=false, so this never redirects. The /api/*
  // fetchAuth 401 path is the backstop for a stale/invalid token.
  useEffect(() => {
    captureUrlToken();
    setHasToken(!!getToken());
    if (!getToken()) {
      authRequired().then((req) => {
        if (req && !getToken()) redirectToUnlock();
      });
    }
  }, []);

  // Drop the stored token and return to the unlock page. Only reachable in remote
  // mode (the button is disabled with no token).
  const lock = useCallback(() => {
    clearToken();
    redirectToUnlock();
  }, []);

  // status + sessions poll on the interval; agents + algorithms are fetched on
  // load and via their ↻ buttons only (each touches third-party config / dials
  // every gRPC server, and nothing there changes on its own between actions).
  useEffect(() => {
    pollStatus();
    pollSessions();
    pollAgents();
    pollAlgos();
    const id = setInterval(() => {
      pollStatus();
      pollSessions();
    }, POLL_MS);
    return () => clearInterval(id);
  }, [pollStatus, pollSessions, pollAgents, pollAlgos]);

  const verb = useCallback(
    async (url: string, confirmMsg?: string) => {
      if (confirmMsg && !confirm(confirmMsg)) return;
      setVerbBusy(true);
      const res = await jpost(url);
      setVerbBusy(false);
      if (res && res.error) alert("Failed: " + res.error);
      if (res && res.data_plane) setDataPlane(res.data_plane);
      pollStatus();
      pollSessions();
    },
    [pollStatus, pollSessions],
  );

  const agentAction = useCallback(
    async (id: string, act: "register" | "unregister") => {
      if (act === "unregister" && !confirm("Unregister biopb from " + id + "?"))
        return;
      setAgentsBusy(true);
      const res = await jpost(
        "/api/agents/" + encodeURIComponent(id) + "/" + act,
      );
      if (res && res.error) alert("Failed: " + res.error);
      await pollAgents();
      setAgentsBusy(false);
    },
    [pollAgents],
  );

  const dpState = dataPlane.state || "unknown";
  const linksOff = dpState !== "serving";

  return (
    <div className="ctrl-dash">
      <header>
        <img
          className="topbar-logo"
          src={`${import.meta.env.BASE_URL}biopb-logo.png`}
          alt=""
          aria-hidden="true"
        />
        <h1>BioPB control - dashboard</h1>
        {/* Status pill on the LEFT (right after the title), matching the admin
            pages' status-pill placement; the spacer below pushes the actions to
            the right edge. Colored by connection health, like those pages. */}
        <span
          id="conn"
          className={connOk === null ? "" : connOk ? "ok" : "bad"}
        >
          {conn}
        </span>
        <div className="hdr-spacer" />
        {/* biopb-mcp's own global settings (transport/kernel/dask/algorithm
            servers), served by the control at /api/mcp_config. A top-level nav
            link — it is neither a data-plane nor a per-session concern. */}
        <a className="hdr-link" href="/mcp/admin" target="_blank" rel="noopener">
          <svg
            className="gear-icon"
            viewBox="0 0 16 16"
            width="13"
            height="13"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M8 0a8.2 8.2 0 0 1 .701.031C9.444.095 9.99.645 10.16 1.29l.288 1.107c.018.066.079.158.212.224.231.114.454.243.668.386.123.082.233.09.299.071l1.103-.303c.644-.176 1.392.021 1.82.63.27.385.506.792.704 1.218.315.675.111 1.422-.364 1.891l-.814.806c-.049.048-.098.147-.088.294.016.257.016.515 0 .772-.01.147.039.246.088.294l.814.806c.475.469.679 1.216.364 1.891a7.977 7.977 0 0 1-.704 1.217c-.428.61-1.176.807-1.82.63l-1.103-.303c-.066-.019-.176-.011-.299.071a4.909 4.909 0 0 1-.668.386c-.133.066-.194.158-.212.224l-.288 1.107c-.17.645-.716 1.195-1.459 1.259a8.147 8.147 0 0 1-1.402 0c-.743-.064-1.289-.614-1.459-1.259l-.288-1.107c-.018-.066-.079-.158-.212-.224a4.911 4.911 0 0 1-.668-.386c-.123-.082-.233-.09-.299-.071l-1.103.303c-.644.176-1.392-.021-1.82-.63a7.988 7.988 0 0 1-.704-1.217c-.315-.675-.111-1.422.364-1.891l.814-.806c.049-.048.098-.147.088-.294a6.214 6.214 0 0 1 0-.772c.01-.147-.039-.246-.088-.294l-.814-.806C.635 6.045.431 5.298.746 4.623a7.921 7.921 0 0 1 .704-1.218c.428-.609 1.176-.806 1.82-.63l1.103.303c.066.019.176.011.299-.071.214-.143.437-.272.668-.386.133-.066.194-.158.212-.224l.288-1.107C6.01.645 6.556.095 7.299.03 7.53.01 7.764 0 8 0Zm0 4.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7ZM8 6a2 2 0 1 1 0 4 2 2 0 0 1 0-4Z" />
          </svg>
          MCP Settings
        </a>
        <button
          className="lock-btn"
          disabled={!hasToken}
          onClick={lock}
          title={hasToken ? "Lock session" : "No token to lock (local mode)"}
        >
          Lock
        </button>
      </header>
      <main>
        <div className="card">
          <h2>Data plane</h2>
          <div>
            <span className={"badge " + dpState}>{dpState}</span>
          </div>
          <dl>
            <dt>gRPC</dt>
            <dd>{dataPlane.grpc_url}</dd>
            <dt>Web</dt>
            <dd>{dataPlane.web_url}</dd>
            <dt>PID</dt>
            <dd>{dataPlane.pid == null ? "—" : dataPlane.pid}</dd>
            <dt>Restarts</dt>
            <dd>{dataPlane.restarts == null ? 0 : dataPlane.restarts}</dd>
            {dataPlane.last_error ? (
              <>
                <dt>Error</dt>
                <dd className="err">{dataPlane.last_error}</dd>
              </>
            ) : null}
          </dl>
          <div className="controls">
            <button disabled={verbBusy} onClick={() => verb("/api/data_plane/ensure")}>
              Ensure up
            </button>
            <button
              disabled={verbBusy}
              onClick={() =>
                verb("/api/data_plane/restart", "Restart the data plane?")
              }
            >
              Restart
            </button>
            <button
              className="danger"
              disabled={verbBusy}
              onClick={() =>
                verb(
                  "/api/data_plane/stop",
                  "Stop the data plane? Clients lose it until an Ensure.",
                )
              }
            >
              Stop
            </button>
            <a
              className={"link" + (linksOff ? " off" : "")}
              href="/viewer"
              target="_blank"
              rel="noopener"
            >
              View Data →
            </a>
            <a
              className={"link" + (linksOff ? " off" : "")}
              href="/admin"
              target="_blank"
              rel="noopener"
            >
              Config →
            </a>
            {/* The data-plane log tail. Available regardless of plane state (a
                crashed plane's log is exactly what you want to read), so unlike
                the viewer/config links it is never disabled. */}
            <a className="link" href="/logs" target="_blank" rel="noopener">
              Logs →
            </a>
          </div>
        </div>

        <div className="card">
          <h2>
            Algorithm plane
            <button className="mini" onClick={pollAlgos}>
              ↻
            </button>
          </h2>
          <ul>
            {algos == null ? (
              <li className="empty">loading…</li>
            ) : algos.length === 0 ? (
              <li className="empty">no algorithm servers configured</li>
            ) : (
              algos.map((s, i) => <AlgoRow key={i} s={s} />)
            )}
          </ul>
          <p className="note">
            Read-only view of the biopb.image ProcessImage servers configured for
            agent kernels, with a live health + ops probe. Lifecycle control is
            not offered here.
          </p>

          <div className="subhead">Kernel plugins</div>
          <ul>
            {plugins == null ? (
              <li className="empty">loading…</li>
            ) : plugins.files.length === 0 && plugins.entry_points.length === 0 ? (
              <li className="empty">no kernel plugins</li>
            ) : (
              <>
                {plugins.files.map((f) => (
                  <li key={"f:" + f.name}>
                    <span className="dot serving"></span>
                    <span className="sid">{f.name}</span>
                    {f.summary ? (
                      <span className="ops" title={f.summary}>
                        {f.summary}
                      </span>
                    ) : null}
                  </li>
                ))}
                {plugins.entry_points.map((e) => (
                  <li key={"e:" + e.name}>
                    <span className="dot serving"></span>
                    <span className="sid">{e.name}</span>
                    <span className="tls">pkg</span>
                    {e.dist ? (
                      <span className="ops" title={e.dist}>
                        {e.dist}
                      </span>
                    ) : null}
                  </li>
                ))}
              </>
            )}
          </ul>
          <p className="note">
            User "bring your own tool" helpers loaded into agent kernels:{" "}
            <code>*.py</code> files in{" "}
            <code>{plugins?.dir || "~/.config/biopb/kernel/"}</code> and installed{" "}
            <code>biopb_mcp.namespace</code> packages. Static listing — the live set
            depends on each plugin.
          </p>
        </div>

        <div className="card">
          <h2>
            Agent clients
            <button className="mini" onClick={pollAgents}>
              ↻
            </button>
          </h2>
          <ul>
            {agents == null ? (
              <li className="empty">loading…</li>
            ) : agents.length === 0 ? (
              <li className="empty">none</li>
            ) : (
              agents.map((a) => (
                <AgentRowView
                  key={a.id}
                  a={a}
                  busy={agentsBusy}
                  onAction={agentAction}
                />
              ))
            )}
          </ul>
          <p className="note">
            Registers biopb-mcp with the client. Restart the client after
            (un)registering for the change to take effect.
          </p>
        </div>

        <div className="card">
          <h2>Agent sessions</h2>
          <ul>
            {sessions == null ? (
              <li className="empty">loading…</li>
            ) : sessions.length === 0 ? (
              <li className="empty">no agent sessions</li>
            ) : (
              sessions.map((s) => {
                const when = s.started_at
                  ? new Date(s.started_at * 1000).toLocaleTimeString()
                  : "";
                const kernel = s.kernel || "unknown";
                return (
                  <li key={s.session_id}>
                    <span className="sid">{s.session_id}</span>
                    <span className="when">
                      :{s.port}
                      {when ? " · " + when : ""}
                    </span>
                    <span className={"kbadge k-" + kernel}>
                      {kernel === "none" ? "no kernel" : "kernel: " + kernel}
                    </span>
                    <a
                      className="obs"
                      href={s.observe_url}
                      target="_blank"
                      rel="noopener"
                    >
                      observe →
                    </a>
                  </li>
                );
              })
            )}
          </ul>
        </div>
      </main>
      <style>{DASH_CSS}</style>
    </div>
  );
}

// One algorithm-plane row: a status dot, host:port (TLS tag for grpcs), the
// state + op count, and an ops preview (full list in the hover title). A
// non-serving server shows its error message in the preview slot instead.
function AlgoRow({ s }: { s: AlgoRec }) {
  const serving = s.state === "serving";
  const dotCls =
    "dot " + (serving ? "serving" : s.state === "unknown" ? "" : "down");
  let stateLabel = s.state;
  if (serving)
    stateLabel = s.single_op
      ? "serving · single-op"
      : "serving · " + (s.op_count || 0) + " op" + (s.op_count === 1 ? "" : "s");
  else if (s.state === "invalid") stateLabel = "invalid URL";
  else if (s.state === "unknown") stateLabel = "gRPC unavailable";
  const joined = s.ops ? s.ops.join(", ") : "";
  return (
    <li>
      <span className={dotCls}></span>
      <span className="sid">{s.target}</span>
      {s.scheme === "grpcs" ? <span className="tls">TLS</span> : null}
      <span className="state">{stateLabel}</span>
      {serving && s.ops && s.ops.length ? (
        <span className="ops" title={joined}>
          {joined}
        </span>
      ) : !serving && s.error ? (
        <span className="ops err" title={s.error}>
          {s.error}
        </span>
      ) : null}
    </li>
  );
}

// The agent-client buttons for one row. not_installed rows get none; a
// registered row offers Unregister, plus an amber "Re-register (update)" only
// when the stored command drifted from the current biopb-mcp location — a plain
// re-register on an up-to-date client is a no-op, so it's omitted; installed
// offers Register.
function AgentRowView({
  a,
  busy,
  onAction,
}: {
  a: AgentRec;
  busy: boolean;
  onAction: (id: string, act: "register" | "unregister") => void;
}) {
  const label =
    a.state === "registered"
      ? a.drifted
        ? "registered · update available"
        : "registered"
      : a.state.replace("_", " ");
  const dotCls = "dot " + a.state + (a.drifted ? " drift" : "");
  return (
    <li>
      <span className={dotCls}></span>
      <span className="sid">{a.name}</span>
      <span className="state">{label}</span>
      <span className="agent-btns">
        {a.state === "not_installed" ? null : a.state === "registered" ? (
          <>
            {a.drifted ? (
              <button
                className="warn"
                disabled={busy}
                onClick={() => onAction(a.id, "register")}
              >
                Re-register (update)
              </button>
            ) : null}
            <button
              className="danger"
              disabled={busy}
              onClick={() => onAction(a.id, "unregister")}
            >
              Unregister
            </button>
          </>
        ) : (
          <button disabled={busy} onClick={() => onAction(a.id, "register")}>
            Register
          </button>
        )}
      </span>
    </li>
  );
}

const DASH_CSS = `
  .ctrl-dash { min-height: 100vh; background: #111; color: #ddd;
               font: 14px/1.5 system-ui, sans-serif; }
  .ctrl-dash header { padding: 10px 16px; background: #1b1b1b; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 12px; position: sticky; top: 0; }
  .ctrl-dash h1 { font-size: 15px; margin: 0; font-weight: 600; }
  .ctrl-dash h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .5px; color: #6a8;
       margin: 0 0 10px; }
  /* Status pill on the left, colored by connection health like the admin
     status-pill. Neutral until the first probe resolves. */
  .ctrl-dash #conn { font-size: 12px; color: #9aa; background: #222;
           padding: 2px 9px; border-radius: 10px; white-space: nowrap; }
  .ctrl-dash #conn.ok { background: #243; color: #7e7; }
  .ctrl-dash #conn.bad { background: #422; color: #f99; }
  /* Pushes the header actions (MCP Settings, Lock) to the right edge, like the
     admin pages' topbar-spacer. */
  .ctrl-dash .hdr-spacer { flex: 1; }
  .ctrl-dash a.hdr-link { font-size: 12px; color: #8bf; text-decoration: none;
           display: inline-flex; align-items: center; gap: 5px; }
  .ctrl-dash a.hdr-link:hover { text-decoration: underline; }
  .ctrl-dash .gear-icon { flex: none; }
  .ctrl-dash main { padding: 16px; max-width: 760px; }
  .ctrl-dash .card { border: 1px solid #333; border-radius: 6px; padding: 14px 16px; margin-bottom: 16px;
          background: #161616; }
  .ctrl-dash .badge { font-size: 11px; padding: 1px 8px; border-radius: 10px; text-transform: uppercase;
           vertical-align: middle; }
  .ctrl-dash .serving { background: #243; color: #7e7; }
  .ctrl-dash .starting { background: #234; color: #8bf; }
  .ctrl-dash .down, .ctrl-dash .conflict { background: #422; color: #f99; }
  .ctrl-dash .stopped, .ctrl-dash .unknown { background: #333; color: #aaa; }
  .ctrl-dash dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 14px; margin: 12px 0 0; }
  .ctrl-dash dt { color: #888; }
  .ctrl-dash dd { margin: 0; font-family: ui-monospace, Menlo, monospace; font-size: 12px; word-break: break-all; }
  .ctrl-dash .err { color: #f99; }
  .ctrl-dash .controls { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .ctrl-dash button { font: inherit; padding: 4px 12px; border: 1px solid #444; border-radius: 4px;
           background: #222; color: #ddd; cursor: pointer; }
  .ctrl-dash button:hover:not(:disabled) { background: #2c2c2c; }
  .ctrl-dash button:disabled { opacity: .45; cursor: default; }
  .ctrl-dash button.danger { border-color: #844; }
  .ctrl-dash a.link { color: #8bf; text-decoration: none; margin-left: auto; }
  .ctrl-dash a.link + a.link { margin-left: 0; }
  .ctrl-dash a.link:hover { text-decoration: underline; }
  .ctrl-dash a.link.off { color: #667; pointer-events: none; }
  .ctrl-dash ul { list-style: none; margin: 0; padding: 0; }
  .ctrl-dash li { display: flex; align-items: center; gap: 10px; padding: 7px 0; border-top: 1px solid #262626; }
  .ctrl-dash li:first-child { border-top: 0; }
  .ctrl-dash .sid { font-family: ui-monospace, Menlo, monospace; font-weight: 600; }
  .ctrl-dash .when { color: #888; font-size: 12px; }
  .ctrl-dash .empty { color: #777; padding: 6px 0; }
  .ctrl-dash a.obs { color: #7e7; text-decoration: none; }
  .ctrl-dash a.obs:hover { text-decoration: underline; }
  /* Kernel (the heavy, on-demand component) state per session. Pushed right so
     it groups with the observe link. */
  .ctrl-dash .kbadge { margin-left: auto; font-size: 11px; padding: 1px 8px;
            border-radius: 10px; white-space: nowrap; }
  .ctrl-dash .k-ready { background: #243; color: #7e7; }
  .ctrl-dash .k-busy { background: #143; color: #9d7; }
  .ctrl-dash .k-starting { background: #234; color: #8bf; }
  .ctrl-dash .k-none { background: #2a2a2a; color: #999; }
  .ctrl-dash .k-error { background: #422; color: #f99; }
  .ctrl-dash .k-unknown { background: #2a2a2a; color: #777; }
  .ctrl-dash .mini { padding: 0 8px; font-size: 12px; margin-left: 8px; vertical-align: middle; }
  .ctrl-dash .state { color: #888; font-size: 12px; }
  .ctrl-dash .note { color: #667; font-size: 12px; margin: 12px 0 0; }
  .ctrl-dash .note code { color: #89a; font-family: ui-monospace, Menlo, monospace; }
  .ctrl-dash .subhead { color: #aab; font-size: 12px; font-weight: 600;
         text-transform: uppercase; letter-spacing: .04em; margin: 16px 0 6px;
         border-top: 1px solid #223; padding-top: 12px; }
  .ctrl-dash .dot { width: 9px; height: 9px; border-radius: 50%; background: #555;
         display: inline-block; flex: none; }
  .ctrl-dash .dot.registered { background: #7e7; }
  .ctrl-dash .dot.installed { background: #8bf; }
  .ctrl-dash .dot.drift { background: #fb4; }
  .ctrl-dash .agent-btns { margin-left: auto; display: flex; gap: 6px; }
  .ctrl-dash button.warn { border-color: #a83; color: #fc9; }
  .ctrl-dash .dot.serving { background: #7e7; }
  .ctrl-dash .dot.down { background: #f77; }
  .ctrl-dash .tls { font-size: 10px; color: #8bf; border: 1px solid #345; border-radius: 3px;
         padding: 0 4px; margin-left: 6px; vertical-align: middle; }
  .ctrl-dash .ops { margin-left: auto; color: #789; font-size: 12px; max-width: 55%;
         overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
         font-family: ui-monospace, Menlo, monospace; }
  .ctrl-dash .ops.err { color: #d88; }
`;
