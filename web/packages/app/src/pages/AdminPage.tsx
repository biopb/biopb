import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { Link } from "react-router-dom";
import {
  TensorApiError,
  normalizeConfigForSave,
  splitConfigErrors,
  validateConfig,
} from "@biopb/tensor-flight-client";
import type {
  AdminConfigError,
  AdminStatus,
  ConfigSchema,
} from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";
import { authHeaders, redirectToUnlock } from "../auth";
import { SourcesEditor, type SourceEntry } from "../components/SourcesEditor";
import { CredentialsEditor } from "../components/CredentialsEditor";
import { Modal } from "../components/Modal";
import { AdminNav } from "../components/admin/AdminNav";
import { SectionFields } from "../components/admin/SectionFields";
import { RawJsonPanel } from "../components/admin/RawJsonPanel";
import { FileBrowser } from "../components/admin/FileBrowser";
import {
  DEFAULT_ADMIN_NAV_ID,
  navItemById,
  navIdForErrorPath,
} from "../components/admin/adminSections";
import { useDocumentTitle } from "../hooks/useDocumentTitle";

type Config = Record<string, unknown>;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const RESTART_TIMEOUT_MS = 60_000;

/** Ask the control to bounce the data plane it owns. Root-relative, so it hits
 * the control (this page's own origin) — not the `/data_plane`-proxied sidecar.
 * The control's verb blocks until the plane is back (or errors); the caller's
 * poll loop then confirms serving state. See biopb/biopb#418. */
async function restartViaControl(): Promise<void> {
  const r = await fetch("/api/data_plane/restart", {
    method: "POST",
    headers: authHeaders(),
  });
  if (r.status === 401) {
    redirectToUnlock();
    throw new Error("Session locked — re-enter the access token.");
  }
  const body = await r.json().catch(() => ({}));
  if (!r.ok || body?.error) {
    throw new Error(body?.error || `Control restart failed (HTTP ${r.status}).`);
  }
}

function getSources(config: Config | null): SourceEntry[] {
  const s = config?.sources;
  return Array.isArray(s) ? (s as SourceEntry[]) : [];
}

/** "up 4m" / "up 1h 3m" / "up 12s" for the topbar read-out (doc UX). */
function formatUptime(seconds: number | null): string {
  if (seconds == null || seconds < 0) return "";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `up ${h}h ${m}m`;
  if (m > 0) return `up ${m}m`;
  return `up ${Math.floor(seconds)}s`;
}

export function AdminPage() {
  useDocumentTitle("BioPB tensor - admin");
  const client = useAppStore((s) => s.client);
  const connectionState = useAppStore((s) => s.connectionState);

  const [config, setConfig] = useState<Config | null>(null);
  const [schema, setSchema] = useState<ConfigSchema | null>(null);
  const [path, setPath] = useState<string>("");
  const [loadError, setLoadError] = useState<string | null>(null);
  // Selected settings section (left-nav) and the source row whose "Browse…"
  // opened the server-side file chooser (biopb/biopb#244), if any.
  const [activeSection, setActiveSection] = useState<string>(DEFAULT_ADMIN_NAV_ID);
  const [browseRow, setBrowseRow] = useState<number | null>(null);

  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  // Server-reported (422) field errors and non-field summary lines. Client-side
  // pre-flight errors are computed live below and merged for display.
  const [serverErrors, setServerErrors] = useState<AdminConfigError[]>([]);
  const [serverGeneral, setServerGeneral] = useState<string[]>([]);

  const [status, setStatus] = useState<AdminStatus | null>(null);
  const [confirmRestart, setConfirmRestart] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const [restartMsg, setRestartMsg] = useState<string | null>(null);
  const [restartScanning, setRestartScanning] = useState(false);
  const [restartError, setRestartError] = useState<string | null>(null);

  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // Mirror `restarting` into a ref so the poll loop's timeout branch can tell a
  // real timeout from a normal post-success teardown.
  const restartingRef = useRef(false);
  useEffect(() => {
    restartingRef.current = restarting;
  }, [restarting]);

  // The one canonical config object. The Sources editor, the structured
  // Advanced sections, and the raw-JSON modal all commit through here. Editing
  // invalidates any stale server-reported errors from a prior save attempt.
  const applyConfig = useCallback((next: Config, markDirty = true) => {
    setConfig(next);
    if (markDirty) {
      setDirty(true);
      setSaved(false);
      setServerErrors([]);
      setServerGeneral([]);
    }
  }, []);

  // Live client-side validation mirroring the server's PUT checks (enum / range
  // / required url), plus any not-yet-cleared server errors from the last save.
  const clientErrors = useMemo(
    () => validateConfig(config, schema),
    [config, schema],
  );
  const combinedErrors = useMemo(
    () => [...clientErrors, ...serverErrors],
    [clientErrors, serverErrors],
  );
  // Source errors render inline on their rows (by index). Advanced / credentials
  // client errors render inline in their own components; they are deliberately
  // NOT rolled into a top banner (that duplicated each inline error and leaked
  // dotted-path notation). The top summary is reserved for server-side 422 lines
  // that aren't attributable to a field (`serverGeneral`).
  const sourceErrors = useMemo(
    () => splitConfigErrors(combinedErrors).byIndex,
    [combinedErrors],
  );
  const hasErrors = combinedErrors.length > 0;

  // Which nav sections currently carry an error, so the nav can dot them (Save is
  // disabled while any error exists, and an errored field may sit on an inactive
  // panel — the dot is how the user finds it).
  const erroredNavIds = useMemo(() => {
    const ids = new Set<string>();
    for (const e of combinedErrors) {
      const id = navIdForErrorPath(e.path);
      if (id) ids.add(id);
    }
    return ids;
  }, [combinedErrors]);

  const refreshStatus = useCallback(async () => {
    if (!client) return;
    try {
      setStatus(await client.http.getAdminStatus());
    } catch {
      /* admin status is best-effort for the header read-out */
    }
  }, [client]);

  // Stable binding for the file chooser: FileBrowser's load effect depends on
  // this, so an inline `(p) => client.http.browse(p)` (a fresh identity every
  // render) would re-run the effect and reset the browser to its initial path on
  // any AdminPage re-render. `client!` is safe — the modal only mounts past the
  // `if (!client)` guard below. Bound to #244.
  const browse = useCallback(
    (p?: string) => client!.http.browse(p),
    [client],
  );

  // Initial load.
  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await client.http.getAdminConfig();
        if (cancelled) return;
        setPath(res.path);
        setSchema(res.schema as ConfigSchema);
        applyConfig(res.config, false);
        setLoadError(null);
      } catch (err) {
        if (!cancelled) {
          setLoadError(err instanceof Error ? err.message : String(err));
        }
      }
      refreshStatus();
    })();
    return () => {
      cancelled = true;
    };
  }, [client, applyConfig, refreshStatus]);

  function onSourcesChange(next: SourceEntry[]) {
    if (!config) return;
    applyConfig({ ...config, sources: next });
  }

  async function onSave() {
    if (!client || !config) return;
    // Block save while client-side errors exist; never round-trip a known bad
    // config just to have the server reject it.
    if (hasErrors) return;
    setSaving(true);
    setSaveError(null);
    setServerErrors([]);
    setServerGeneral([]);
    // Resolve deprecated aliases the runtime tolerates but the server's schema
    // validation rejects (source `path` -> `url`) before PUT, and reflect the
    // migration in the editor so the deprecated field clears on success.
    const payload = normalizeConfigForSave(config) as Config;
    try {
      await client.http.putAdminConfig(payload);
      if (!mounted.current) return;
      setConfig(payload);
      setSaved(true);
      setDirty(false);
    } catch (err) {
      if (!mounted.current) return;
      if (err instanceof TensorApiError && err.status === 422) {
        const body = err.detail as { errors?: AdminConfigError[] } | undefined;
        const errs = body?.errors ?? [];
        setServerErrors(errs);
        // Anything the splitter can't attribute to a source row is a general
        // line; keep at least one so a 422 is never silent.
        const { general } = splitConfigErrors(errs);
        setServerGeneral(general.length ? general : ["Config failed validation."]);
      } else {
        setSaveError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (mounted.current) setSaving(false);
    }
  }

  async function doRestart() {
    if (!client) return;
    setConfirmRestart(false);
    setRestarting(true);
    setRestartScanning(false);
    setRestartError(null);
    setRestartMsg("Restarting…");
    try {
      if (status?.supervised) {
        // The biopb control owns this data plane, so restarting it is a request
        // to the control — its own token-gated verb at this single origin —
        // rather than the sidecar, which the supervisor would race for ownership
        // (biopb/biopb#418). The control bounces + waits for the plane to come
        // back, then the poll loop below confirms it's serving.
        await restartViaControl();
      } else {
        // Not control-owned (a directly-launched `biopb-tensor-server launch`, or
        // a `--no-data-plane` adopted plane): the control does not own this
        // process and the sidecar has no self-restart, so restart is not available
        // from the browser. The operator restarts it where they launched it.
        setRestartError(
          "This data plane is self-managed — restart it where you launched it, " +
            "or run it under `biopb control` for managed restart.",
        );
        setRestarting(false);
        setRestartMsg(null);
        return;
      }
    } catch (err) {
      if (mounted.current) {
        setRestartError(err instanceof Error ? err.message : String(err));
        setRestarting(false);
      }
      return;
    }

    const deadline = Date.now() + RESTART_TIMEOUT_MS;
    // Phase 1: wait for the new daemon to bind (/livez answers).
    let alive = false;
    while (mounted.current && Date.now() < deadline) {
      await sleep(1000);
      try {
        await client.http.livez();
        alive = true;
        break;
      } catch {
        /* still down */
      }
    }
    // Phase 2: watch the discovery scan populate via /api/admin/status.
    while (alive && mounted.current && Date.now() < deadline) {
      try {
        const st = await client.http.getAdminStatus();
        if (!mounted.current) return;
        setStatus(st);
        const n = st.source_count ?? 0;
        if (st.full_scan_in_progress) {
          setRestartScanning(true);
          setRestartMsg(`Reconnected — scanning… ${n} sources`);
        } else if (st.health === "SERVING") {
          setRestartScanning(false);
          setRestartMsg(`Ready — ${n} sources`);
          await sleep(800);
          if (!mounted.current) return;
          setRestarting(false);
          setRestartMsg(null);
          setSaved(false);
          // Reload the freshly-applied config.
          try {
            const res = await client.http.getAdminConfig();
            if (mounted.current) {
              setPath(res.path);
              setSchema(res.schema as ConfigSchema);
              applyConfig(res.config, false);
            }
          } catch {
            /* leave the current view */
          }
          return;
        }
      } catch {
        /* admin routes briefly dead during the bounce; keep polling */
      }
      await sleep(1500);
    }
    if (mounted.current && restartingRef.current) {
      setRestarting(false);
      setRestartMsg(null);
      setRestartError(
        "Server did not come back within 60s. Check `biopb server status` or the log file.",
      );
    }
  }

  // ---- Render ----

  if (!client) {
    return (
      <div className="app-shell admin-shell">
        <div className="loading-overlay" style={{ position: "static" }}>
          {connectionState === "error"
            ? "Cannot reach the server."
            : "Connecting to server…"}
        </div>
      </div>
    );
  }

  // "Not running" means the backend health check couldn't reach the Flight
  // server at all (`health == null`) — a true down process. A reachable server
  // that is merely warming up (`health` is a non-SERVING string like STARTING /
  // NOT_SERVING) is NOT down: it keeps the normal read-out + Restart, not Start.
  const daemonDown =
    !restarting && !!status && status.running === false && status.health == null;
  const restartVerb = daemonDown ? "Start" : "Restart";

  const pill = restartScanning
    ? { cls: "scanning", text: restartMsg ?? "Scanning…" }
    : restarting
      ? { cls: "restarting", text: restartMsg ?? "Restarting…" }
      : daemonDown
        ? { cls: "error", text: "● Not running" }
        : loadError
          ? { cls: "error", text: "Error" }
          : status
            ? {
                cls: "connected",
                text: [
                  `● ${status.health ?? "–"}`,
                  `${status.source_count ?? 0} sources`,
                  formatUptime(status.uptime_seconds),
                ]
                  .filter(Boolean)
                  .join(" · "),
              }
            : { cls: "connecting", text: "Loading…" };

  const sources = getSources(config);
  const activeItem = navItemById(activeSection);
  // Local mode ⇒ the server's filesystem is this machine, so the "Browse…" file
  // chooser is safe to offer (biopb/biopb#244). Absent/false ⇒ typed path only.
  const localMode = status?.local === true;

  return (
    <div className="app-shell admin-shell">
      <header className="app-topbar">
        <img
          className="topbar-logo"
          src={`${import.meta.env.BASE_URL}biopb-logo.png`}
          alt=""
          aria-hidden="true"
        />
        <h1>BioPB tensor - admin</h1>
        <span className={`status-pill ${pill.cls}`}>{pill.text}</span>
        <div className="topbar-spacer" />
        {dirty && !saved && !hasErrors && (
          <span className="admin-hint">Unsaved changes</span>
        )}
        <button
          type="button"
          className="submit-btn topbar-save"
          disabled={!dirty || saving || restarting || hasErrors}
          onClick={onSave}
          title={hasErrors ? "Fix the highlighted settings to save" : undefined}
        >
          {saving ? "Saving…" : "Save"}
        </button>
        <button
          type="button"
          className="icon-btn"
          disabled={restarting || !config}
          onClick={() => setConfirmRestart(true)}
        >
          {restartVerb}
        </button>
        <Link className="icon-btn" to="/viewer">
          Viewer
        </Link>
      </header>

      <main className="app-main admin-main">
        <AdminNav
          active={activeSection}
          erroredIds={erroredNavIds}
          onSelect={setActiveSection}
        />

        <div className="admin-content">
          {loadError && (
            <div className="admin-banner error">Could not load config: {loadError}</div>
          )}

          {daemonDown && (
            <div className="admin-banner degraded">
              <strong>Server not running.</strong> The Flight server isn't serving
              (the config file is still editable — Save, then <em>{restartVerb}</em>).
              <button
                type="button"
                className="icon-btn"
                disabled={restarting}
                onClick={() => setConfirmRestart(true)}
              >
                {restartVerb} now
              </button>
            </div>
          )}

          {serverGeneral.length > 0 && (
            <div className="admin-banner error">
              <strong>Config not saved — fix these:</strong>
              <ul>
                {serverGeneral.map((m, i) => (
                  <li key={i}>{m}</li>
                ))}
              </ul>
            </div>
          )}

          {hasErrors && serverGeneral.length === 0 && (
            <div className="admin-banner error">
              Some settings need fixing — see the marked sections in the sidebar.
            </div>
          )}

          {saved && (
            <div className="admin-banner saved">
              Saved — {restartVerb.toLowerCase()} required to apply.
              <button
                type="button"
                className="icon-btn"
                disabled={restarting}
                onClick={() => setConfirmRestart(true)}
              >
                {restartVerb} now
              </button>
            </div>
          )}

          {config && (
            <section className="admin-panel">
              <div className="admin-panel-head">
                <h2>{activeItem.label}</h2>
                <p className="admin-panel-desc">{activeItem.description}</p>
              </div>

              {activeItem.kind === "sources" && (
                <SourcesEditor
                  sources={sources}
                  onChange={onSourcesChange}
                  schema={schema}
                  errorsByIndex={sourceErrors}
                  disabled={restarting}
                  onBrowse={localMode ? (i) => setBrowseRow(i) : undefined}
                />
              )}

              {activeItem.kind === "credentials" && (
                <CredentialsEditor
                  config={config}
                  schema={schema}
                  errors={combinedErrors}
                  disabled={restarting}
                  onChange={applyConfig}
                />
              )}

              {activeItem.kind === "fields" && activeItem.section && (
                <SectionFields
                  section={activeItem.section}
                  commonFields={activeItem.commonFields ?? []}
                  config={config}
                  schema={schema}
                  errors={combinedErrors}
                  disabled={restarting}
                  onChange={applyConfig}
                />
              )}

              {activeItem.kind === "raw" && (
                <RawJsonPanel
                  config={config}
                  disabled={restarting}
                  onApply={applyConfig}
                />
              )}
            </section>
          )}

          {path && (
            <div className="admin-config-path">
              Editing <code>{path}</code>
            </div>
          )}
        </div>
      </main>

      {browseRow !== null && config && (
        <FileBrowser
          browse={browse}
          initialPath={String(sources[browseRow]?.url ?? "")}
          onPick={(picked) => {
            onSourcesChange(
              sources.map((s, idx) =>
                idx === browseRow ? { ...s, url: picked } : s,
              ),
            );
            setBrowseRow(null);
          }}
          onClose={() => setBrowseRow(null)}
        />
      )}

      {confirmRestart && (
        <Modal
          title={daemonDown ? "Start the server?" : "Restart the server?"}
          onClose={() => setConfirmRestart(false)}
          labelId="admin-restart-title"
        >
          <p>
            {daemonDown
              ? "Start the tensor server daemon with the current config on disk."
              : "Restart interrupts the shared live session: connected clients (the napari/MCP kernel, browser viewers, in-flight analyses) drop while the daemon bounces."}
          </p>
          <div className="admin-modal-actions">
            <button type="button" className="icon-btn" onClick={() => setConfirmRestart(false)}>
              Cancel
            </button>
            <button type="button" className="submit-btn" onClick={doRestart}>
              {restartVerb}
            </button>
          </div>
        </Modal>
      )}

      {saveError && (
        <div className="error-toast">
          <strong>Save failed</strong>
          <br />
          {saveError}
        </div>
      )}
      {restartError && (
        <div className="error-toast">
          <strong>{restartVerb}</strong>
          <br />
          {restartError}
        </div>
      )}
    </div>
  );
}
