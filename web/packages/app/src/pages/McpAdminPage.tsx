import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  sectionProperties,
  validateConfig,
  type ConfigError,
  type ConfigSchema,
} from "@biopb/tensor-flight-client";
import { authHeaders, redirectToUnlock } from "../auth";
import { SectionFields } from "../components/admin/SectionFields";
import { RawJsonPanel } from "../components/admin/RawJsonPanel";
import {
  MCP_DEFAULT_NAV_ID,
  MCP_NAV,
  mcpNavIdForErrorPath,
  mcpNavItemById,
} from "../components/admin/mcpSections";
import { useDocumentTitle } from "../hooks/useDocumentTitle";

/**
 * The biopb-mcp settings page — sibling of the tensor AdminPage, but for the
 * agent client's own config (`~/.config/biopb/mcp-config.json`). The **control**
 * owns and serves it at `GET/PUT /api/mcp_config` (the config is global while mcp
 * sessions are ephemeral), so this page talks to the control's own origin
 * directly, not the `/data_plane`-proxied tensor sidecar. Nothing merges with the
 * tensor config; only the schema-driven machinery (`SectionFields` / `SchemaField`
 * / `validateConfig`) is shared.
 *
 * Unlike the tensor admin there is no "restart": each mcp session reads config
 * fresh at bootstrap, so a save applies to the *next* session.
 */

type Config = Record<string, unknown>;

interface McpConfigResponse {
  path: string;
  config: Config;
  schema: ConfigSchema;
}

async function loadMcpConfig(): Promise<McpConfigResponse> {
  // `no-store`: an admin config editor must never render a cached GET — a stale
  // response (e.g. an empty {} cached before the file was populated) would show
  // the wrong config and clobber it on save. The server also sends
  // Cache-Control: no-store, but this makes the client independent of that.
  const r = await fetch("/api/mcp_config", {
    headers: authHeaders(),
    cache: "no-store",
  });
  if (r.status === 401) {
    redirectToUnlock();
    throw new Error("Session locked — re-enter the access token.");
  }
  if (!r.ok) {
    const errBody = await r.json().catch(() => ({}));
    throw new Error(errBody?.error || `Could not load config (HTTP ${r.status}).`);
  }
  // A 200 with an empty/opaque or schema-less body is the fingerprint of an
  // intercepting layer (a stale service worker or proxy) silently serving
  // nothing. Surface that as a clear error rather than a confusing empty form
  // (header renders, no fields, Raw JSON shows {}). See the no-store note above.
  let body: unknown;
  try {
    body = await r.json();
  } catch {
    throw new Error(
      "Config response was not valid JSON — a stale service worker or proxy may " +
        "be intercepting /api/mcp_config. Clear the site's data (or unregister the " +
        "service worker) and reload.",
    );
  }
  const parsed = body as Partial<McpConfigResponse> | null;
  if (!parsed || typeof parsed !== "object" || !parsed.schema) {
    throw new Error(
      "Config response was missing its schema — a stale service worker or cache " +
        "may be intercepting /api/mcp_config. Clear the site's data (or unregister " +
        "the service worker) and reload.",
    );
  }
  return parsed as McpConfigResponse;
}

export default function McpAdminPage() {
  useDocumentTitle("BioPB mcp - settings");

  const [config, setConfig] = useState<Config | null>(null);
  const [schema, setSchema] = useState<ConfigSchema | null>(null);
  const [path, setPath] = useState<string>("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [active, setActive] = useState<string>(MCP_DEFAULT_NAV_ID);

  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [serverErrors, setServerErrors] = useState<ConfigError[]>([]);

  const applyConfig = useCallback((next: Config, markDirty = true) => {
    setConfig(next);
    if (markDirty) {
      setDirty(true);
      setSaved(false);
      setServerErrors([]);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await loadMcpConfig();
        if (cancelled) return;
        setPath(res.path);
        setSchema(res.schema);
        applyConfig(res.config ?? {}, false);
        setLoadError(null);
      } catch (err) {
        if (!cancelled) {
          setLoadError(err instanceof Error ? err.message : String(err));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [applyConfig]);

  // Client-side pre-flight mirroring the control's PUT checks (enum / range), so
  // Save is blocked before a known-bad round trip. Server 422 errors merge in.
  const clientErrors = useMemo(
    () => validateConfig(config, schema),
    [config, schema],
  );
  const combinedErrors = useMemo(
    () => [...clientErrors, ...serverErrors],
    [clientErrors, serverErrors],
  );
  const hasErrors = combinedErrors.length > 0;

  const erroredNavIds = useMemo(() => {
    const ids = new Set<string>();
    for (const e of combinedErrors) {
      const id = mcpNavIdForErrorPath(e.path);
      if (id) ids.add(id);
    }
    return ids;
  }, [combinedErrors]);

  async function onSave() {
    if (!config || hasErrors) return;
    setSaving(true);
    setSaveError(null);
    setServerErrors([]);
    try {
      const r = await fetch("/api/mcp_config", {
        method: "PUT",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(config),
      });
      if (r.status === 401) {
        redirectToUnlock();
        return;
      }
      const body = await r.json().catch(() => ({}));
      if (r.status === 422) {
        setServerErrors((body?.errors as ConfigError[]) ?? []);
        return;
      }
      if (!r.ok) {
        throw new Error(body?.error || `Save failed (HTTP ${r.status}).`);
      }
      setSaved(true);
      setDirty(false);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  const activeItem = mcpNavItemById(active);
  // Show every field of the active section directly (no common/advanced split for
  // the mcp page): commonFields = all of the section's schema keys.
  const commonFields = activeItem.section
    ? Object.keys(sectionProperties(schema, activeItem.section))
    : [];

  return (
    <div className="app-shell admin-shell">
      <header className="app-topbar">
        <img
          className="topbar-logo"
          src={`${import.meta.env.BASE_URL}biopb-logo.png`}
          alt=""
          aria-hidden="true"
        />
        <h1>BioPB mcp - settings</h1>
        <div className="topbar-spacer" />
        {dirty && !saved && !hasErrors && (
          <span className="admin-hint">Unsaved changes</span>
        )}
        <button
          type="button"
          className="submit-btn topbar-save"
          disabled={!dirty || saving || hasErrors || !config}
          onClick={onSave}
          title={hasErrors ? "Fix the highlighted settings to save" : undefined}
        >
          {saving ? "Saving…" : "Save"}
        </button>
        <Link className="icon-btn" to="/">
          Dashboard
        </Link>
      </header>

      <main className="app-main admin-main">
        <nav className="admin-nav" aria-label="Settings sections">
          {MCP_NAV.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`admin-nav-item${item.id === active ? " active" : ""}`}
              aria-current={item.id === active ? "page" : undefined}
              onClick={() => setActive(item.id)}
            >
              <span className="admin-nav-label">{item.label}</span>
              {erroredNavIds.has(item.id) && (
                <span className="admin-nav-errdot" title="Has errors" />
              )}
            </button>
          ))}
        </nav>

        <div className="admin-content">
          {loadError && (
            <div className="admin-banner error">Could not load config: {loadError}</div>
          )}

          {hasErrors && (
            <div className="admin-banner error">
              Some settings need fixing — see the marked sections in the sidebar.
            </div>
          )}

          {saved && (
            <div className="admin-banner saved">
              Saved — applies to newly-started sessions.
            </div>
          )}

          {config && !loadError && (
            <section className="admin-panel">
              <div className="admin-panel-head">
                <h2>{activeItem.label}</h2>
                <p className="admin-panel-desc">{activeItem.description}</p>
              </div>

              {activeItem.kind === "fields" && activeItem.section && (
                <SectionFields
                  section={activeItem.section}
                  commonFields={commonFields}
                  config={config}
                  schema={schema}
                  errors={combinedErrors}
                  onChange={applyConfig}
                />
              )}

              {activeItem.kind === "raw" && (
                <RawJsonPanel config={config} onApply={applyConfig} />
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

      {saveError && (
        <div className="error-toast">
          <strong>Save failed</strong>
          <br />
          {saveError}
        </div>
      )}
    </div>
  );
}
