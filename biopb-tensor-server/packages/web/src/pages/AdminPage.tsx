import { useEffect, useRef, useState, useCallback } from "react";
import { Link } from "react-router-dom";
import { TensorApiError } from "@biopb/tensor-flight-client";
import type { AdminConfigError, AdminStatus } from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";
import { SourcesEditor, type SourceEntry } from "../components/SourcesEditor";

type Config = Record<string, unknown>;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const RESTART_TIMEOUT_MS = 60_000;

function getSources(config: Config | null): SourceEntry[] {
  const s = config?.sources;
  return Array.isArray(s) ? (s as SourceEntry[]) : [];
}

/** Split the flat 422 errors into per-source-row messages + a general summary. */
function splitErrors(errors: AdminConfigError[]): {
  byIndex: Record<number, string[]>;
  general: string[];
} {
  const byIndex: Record<number, string[]> = {};
  const general: string[] = [];
  for (const e of errors) {
    if (e.path[0] === "sources" && typeof e.path[1] === "number") {
      const i = e.path[1];
      (byIndex[i] ??= []).push(`${e.path.slice(2).join(".") || "source"}: ${e.message}`);
    } else {
      const where = e.path.length ? e.path.join(".") + ": " : "";
      general.push(where + e.message);
    }
  }
  return { byIndex, general };
}

export function AdminPage() {
  const client = useAppStore((s) => s.client);
  const connectionState = useAppStore((s) => s.connectionState);
  const clearSession = useAppStore((s) => s.clearSession);

  const [config, setConfig] = useState<Config | null>(null);
  const [path, setPath] = useState<string>("");
  const [rawText, setRawText] = useState<string>("");
  const [rawError, setRawError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [generalErrors, setGeneralErrors] = useState<string[]>([]);
  const [errorsByIndex, setErrorsByIndex] = useState<Record<number, string[]>>({});

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

  // Route every config mutation through here so the structured editor and the
  // raw-JSON textarea stay in sync over one canonical object.
  const applyConfig = useCallback((next: Config, markDirty = true) => {
    setConfig(next);
    setRawText(JSON.stringify(next, null, 2));
    setRawError(null);
    if (markDirty) {
      setDirty(true);
      setSaved(false);
    }
  }, []);

  const refreshStatus = useCallback(async () => {
    if (!client) return;
    try {
      setStatus(await client.http.getAdminStatus());
    } catch {
      /* admin status is best-effort for the header read-out */
    }
  }, [client]);

  // Initial load.
  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await client.http.getAdminConfig();
        if (cancelled) return;
        setPath(res.path);
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

  function onRawBlur() {
    try {
      const parsed = JSON.parse(rawText) as Config;
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        setRawError("Config must be a JSON object.");
        return;
      }
      applyConfig(parsed);
    } catch (e) {
      setRawError(e instanceof Error ? e.message : "Invalid JSON");
    }
  }

  async function onSave() {
    if (!client || !config) return;
    setSaving(true);
    setSaveError(null);
    setGeneralErrors([]);
    setErrorsByIndex({});
    try {
      await client.http.putAdminConfig(config);
      if (!mounted.current) return;
      setSaved(true);
      setDirty(false);
    } catch (err) {
      if (!mounted.current) return;
      if (err instanceof TensorApiError && err.status === 422) {
        const body = err.detail as { errors?: AdminConfigError[] } | undefined;
        const { byIndex, general } = splitErrors(body?.errors ?? []);
        setErrorsByIndex(byIndex);
        setGeneralErrors(general.length ? general : ["Config failed validation."]);
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
      await client.http.restartServer();
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

  const pill = restartScanning
    ? { cls: "scanning", text: restartMsg ?? "Scanning…" }
    : restarting
      ? { cls: "restarting", text: restartMsg ?? "Restarting…" }
      : loadError
        ? { cls: "error", text: "Error" }
        : status
          ? {
              cls: "connected",
              text: `${status.health ?? "–"} · ${status.source_count ?? 0} sources`,
            }
          : { cls: "connecting", text: "Loading…" };

  const sources = getSources(config);

  return (
    <div className="app-shell admin-shell">
      <header className="app-topbar">
        <h1>BioPB · Admin</h1>
        <span className={`status-pill ${pill.cls}`}>{pill.text}</span>
        <div className="topbar-spacer" />
        <button
          type="button"
          className="icon-btn"
          disabled={restarting || !config}
          onClick={() => setConfirmRestart(true)}
        >
          Restart
        </button>
        <Link className="icon-btn" to="/">
          ← Back
        </Link>
        <button className="icon-btn" onClick={clearSession} title="Lock session">
          Lock
        </button>
      </header>

      <main className="app-main admin-main">
        {loadError && (
          <div className="admin-banner error">Could not load config: {loadError}</div>
        )}

        {path && (
          <div className="admin-config-path">
            Editing <code>{path}</code>
          </div>
        )}

        {generalErrors.length > 0 && (
          <div className="admin-banner error">
            <strong>Config not saved — fix these:</strong>
            <ul>
              {generalErrors.map((m, i) => (
                <li key={i}>{m}</li>
              ))}
            </ul>
          </div>
        )}

        {saved && (
          <div className="admin-banner saved">
            Saved — restart required to apply.
            <button
              type="button"
              className="icon-btn"
              disabled={restarting}
              onClick={() => setConfirmRestart(true)}
            >
              Restart now
            </button>
          </div>
        )}

        {config && (
          <>
            <SourcesEditor
              sources={sources}
              onChange={onSourcesChange}
              errorsByIndex={errorsByIndex}
              disabled={restarting}
            />

            <details className="admin-advanced">
              <summary>Advanced — full config (raw JSON)</summary>
              <textarea
                className="admin-raw"
                spellCheck={false}
                value={rawText}
                disabled={restarting}
                onChange={(e) => setRawText(e.target.value)}
                onBlur={onRawBlur}
              />
              {rawError && <p className="error-msg">Invalid JSON: {rawError}</p>}
            </details>

            <div className="admin-actions">
              <button
                type="button"
                className="submit-btn"
                disabled={!dirty || saving || restarting || !!rawError}
                onClick={onSave}
              >
                {saving ? "Saving…" : "Save"}
              </button>
              {dirty && !saved && (
                <span className="admin-hint">
                  Unsaved changes — restart required to apply.
                </span>
              )}
            </div>
          </>
        )}
      </main>

      {confirmRestart && (
        <div className="admin-modal-backdrop" onClick={() => setConfirmRestart(false)}>
          <div className="admin-modal" onClick={(e) => e.stopPropagation()}>
            <h2>Restart the server?</h2>
            <p>
              Restart interrupts the shared live session: connected clients (the
              napari/MCP kernel, browser viewers, in-flight analyses) drop while the
              daemon bounces.
            </p>
            <div className="admin-modal-actions">
              <button className="icon-btn" onClick={() => setConfirmRestart(false)}>
                Cancel
              </button>
              <button className="submit-btn" onClick={doRestart}>
                Restart
              </button>
            </div>
          </div>
        </div>
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
          <strong>Restart</strong>
          <br />
          {restartError}
        </div>
      )}
    </div>
  );
}
