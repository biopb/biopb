# Tensor-Server Admin Endpoint — Config / Status / Restart over HTTP

**Status:** implemented (`biopb/biopb#245`). HTTP routes on the FastAPI sidecar (`http_server.py`) plus a schema-driven admin page in the `web/` SPA; the `biopb` umbrella CLI provides the `server restart` lever and `save_config` lives in the tensor-server config model. **Related:** `progressive-discovery.md` (the scan-freshness `health` fields this page consumes), `remote-tensor-cache.md` (the config-GUI design this realizes), `config-format-migration.md` (JSON-canonical config).

## Why

A user needs to **reconfigure, inspect, and restart the local tensor server** without hand-editing `~/.config/biopb/biopb.json` and without a terminal: rewrite `sources` (local dirs/files + proxied remotes) and cache/pyramid/server knobs, check running/health/scan-progress, and restart to apply (the server reads config **once at startup** — no hot-reload).

This ships as a **web-app admin surface, not a napari Qt form.** The sidecar already has token auth, health probes, a `TensorFlightClient` to the Flight server, and a shipped React app on the same origin — so the whole integration is one backend route set plus a `webbrowser.open(f"{base}/admin?token=…")` menu action, and the schema-driven form stays in **one** place (React) instead of being reimplemented per-field in Qt. The editing backend (load / validate / `save_config`) lives in the tensor-server config model so a Qt form could reuse it later.

## Routes

All mount on `http_server.create_app`. `GET /api/config` and the `/api/admin/*` reads are token-gated; the two mutating routes additionally pass `_require_same_origin`.

| Route | Method | Does |
|---|---|---|
| `/api/config` | `GET` | Return the raw `biopb.json` dict, its on-disk path, and the JSON Schema (`build_config_schema()`). Secrets are redacted (see below). |
| `/api/config` | `PUT` | Validate the body against the schema → restore redacted secrets → `save_config()` atomically → `{saved, restart_required: true}`. Does **not** restart. |
| `/api/admin/status` | `GET` | `running / health / pid / version / config path / uptime` + the progressive-discovery freshness fields (`source_count`, `full_scan_in_progress`, `last_full_scan_finished_at`) + the `supervised` flag. |
| `/api/admin/browse` | `GET` | Filesystem directory listing for the Sources file chooser. **Local-mode only** — 404s when a token is enforced (`biopb/biopb#244`), since a browsable FS listing is an info-disclosure surface. Returns `{path, parent, entries:[{name,is_dir}], truncated}`. |
| `/api/admin/restart` | `POST` | Restart the daemon; see **Supervised-restart**. |

The unauthenticated `/livez` / `/readyz` are what the frontend polls **through the restart gap**, when the `/api/admin/*` routes themselves are dead.

**`GET /api/config`** returns the **raw dict** (not `dataclasses.asdict(load_config(...))`) so the `PUT` round-trip preserves keys the form does not surface. **`PUT`** validates with `Draft202012Validator(build_config_schema())`, returning `422` with structured `jsonschema` errors on failure — the same `_CONSTRAINTS` table drives both this pre-write check and the server's `__post_init__` validation, so they cannot disagree (`biopb/biopb#34`). The published schema keeps `additionalProperties: true`, so unknown/advanced keys pass while dangerous values (`downscale_factor` 0/1, out-of-range `port`, bad `backend`) are rejected before disk. The write never auto-restarts.

## `save_config` and credential redaction

`save_config(data, path)` writes canonical `biopb.json` **atomically** (temp file + `os.replace`), **round-trips on the raw dict** (edit by key, re-serialize — `dataclasses.asdict()` would clobber advanced/future keys), and migrates a legacy `biopb.toml` forward to JSON.

Credentials round-trip so secrets never reach the browser: `GET /api/config` replaces each `credentials.profiles[*]` `key`/`secret`/`token` with `REDACTED_SENTINEL` (`"***REDACTED***"`, in `core/config.py`); the form seeds those password fields with the sentinel; on `PUT`, `restore_redacted_secrets(body, existing)` puts the real on-disk value back wherever the field still equals the sentinel, so an untouched secret is preserved.

## Supervised-restart (the hazard — `biopb/biopb#418`)

The admin endpoint runs **inside the daemon it must restart.** A standalone `biopb server start` daemon has no supervisor, so `POST /api/admin/restart` spawns a **detached** `biopb server restart --config … --web-port … --web-host … [--static-dir …]` child that outlives the dying parent and runs the existing `restart()` flow verbatim (graceful stop — `SIGTERM` on POSIX, the Windows shutdown-sentinel file — with PID-identity checks, wait for the port to free, `start()` re-binds). The token is passed through the inherited environment (`BIOPB_TENSOR_TOKEN`), never on the visible command line; the daemon echoes its own launch args (config path, web host/port, static dir), threaded `launch → create_app → the restart-route closure`, so a restart comes back **identically** rather than falling back to defaults. Returns `202`; the browser reconnects via `/livez` → `/api/admin/status`.

**But that self-restart is a hazard under the control plane.** When `biopb-control` supervises the data plane, its supervisor *tracks* the child; a detached self-restart would `SIGTERM` the tracked child and **race the supervisor for the gRPC port**. So:

- The control marks its child with `BIOPB_DATA_PLANE_SUPERVISED=1`; `create_app` reads it into `self.supervised` (overridable via the `supervised` arg).
- The sidecar surfaces `supervised: true` in `/api/admin/status` and **refuses self-restart** — `POST /api/admin/restart` returns **`409`** when supervised (also `409` if a restart is already in progress).
- The admin UI (`AdminPage.tsx`) routes restart on the flag: `supervised` → `restartViaControl()` (`POST /api/data_plane/restart`, root-relative so it hits the **control's** own token-gated supervisor verb at this single origin, which bounces and waits); not supervised → the sidecar self-restart (`client.http.restartServer()`). A `running: false` status relabels **Restart → Start**.

(This hazard also appears in the merged control-plane doc; this is the tensor-side view.)

## Same-origin guard

`PUT /api/config` and `POST /api/admin/restart` are the sidecar's first **mutating** surface. Under local mode (no token) a page the user merely visits could fire a cross-origin `POST` at the loopback sidecar — it can't read the response (CORS) but a state change doesn't need to. `_require_same_origin` delegates to the shared `biopb._web_auth.is_forgeable_cross_site(headers.get)` policy: a request carrying a token header is not forgeable; a browser that stamped `Sec-Fetch-Site` cross-site is the vector and is refused (`403`); a non-browser client (curl) sends neither and is allowed (a token-gated server still enforces `check_token` independently). This blocks drive-by browser CSRF even with no token, at zero storage cost.

## The admin page

A React route `/admin` in the `web/` SPA, reusing the app shell, `status-pill`, and the `/unlock` token gate so it reads as the same product. `/` stays read-only browse; `/admin` is the only **editing** surface. The napari action opens `/admin?token=…` directly (same `?token` capture + URL-strip the bootstrap does for `/`).

- **Structured advanced sections** — `SectionFields.tsx` + `adminSections.ts` render one collapsible pane per schema section (`server`/`cache`/`pyramid`/`precache`/`metadata_db`), each field a schema-driven `SchemaField.tsx` control (checkbox / enum select / bounded number / text) with its `description` as helper text; `RawJsonPanel.tsx` is the escape-hatch modal.
- **Inline validation** — `validateConfig()` in `@biopb/tensor-flight-client` (unit-tested) mirrors the server's `PUT` checks (hard enum, numeric bounds, required `url`), leaving case-insensitive enums lenient (helper-text only). It disables **Save** while any field errors exist and force-opens any section holding an error.
- **Sources / credentials editors** — `SourcesEditor.tsx` lists source rows (type · url · `source_id`/alias · monitor) with inline edit/remove and an `+ Add ▾` for local folders/files and remote tensor servers; `FileBrowser.tsx` backs the "Browse…" chooser (shown only when status reports local). `CredentialsEditor.tsx` handles the nested `credentials.profiles` array — a profiles list + `default_profile` picker, `name` required inline, and `key`/`secret`/`token` as password fields seeded with `REDACTED_SENTINEL`.
- **Deprecated keys** (`watcher_type`, `poll_interval`, source `path`, `metadata_db.enabled`, pyramid knobs under `[precache]`) render with a "deprecated" tag + canonical-key hint **only when already present**, and are never offered fresh.

**Save → Restart is two distinct, sequential actions** (config written first, applied second). Save issues `PUT /api/config`: `422` renders inline field errors + summary, nothing written; `200` shows a persistent **"Saved — restart required to apply"** banner. Restart (topbar button, always present, and the banner's **Restart now**) opens a **confirm modal** — restart **interrupts the shared live session** (napari/MCP kernel, browser viewers, in-flight analyses drop) — then routes per `supervised` above and enters a disabled "Restarting…" state. Restart is available independently of a config change; Save never auto-restarts.

## Post-restart UX — show the scan, don't blind-wait

On a large/cloud root the startup discovery scan can run for **minutes** (`biopb/biopb#212` measured 142 s of silence). The daemon reaches `SERVING` immediately (backgrounded scan, `progressive-discovery.md`), so the page polls `/api/admin/status` and narrates the freshness fields rather than blind-waiting: `/livez` dead → "Restarting…"; answering with `full_scan_in_progress` true → "Reconnected — scanning… N sources" (climbing `source_count`); `last_full_scan_finished_at` set and `health == "SERVING"` → "Ready — N sources". A timeout (~60 s) surfaces a "server did not come back" toast pointing at `biopb server status` / the log — never an infinite spinner.

## Gotchas

- **The mutating routes are the only ones that can rewrite/restart.** Read routes being an unauthenticated localhost gate never mattered; `PUT /api/config` and `POST /api/admin/restart` under local mode (no token) would rewrite config / restart with no credential — which is exactly why they carry `_require_same_origin` on top of `check_token`.
- **Supervised self-restart races the supervisor for the gRPC port** (`biopb/biopb#418`). Never let a supervised sidecar self-restart; it must `409` and the UI must route to the control's `/api/data_plane/restart`. The three non-supervised cases (standalone `biopb server start`, a control that merely *adopts* a plane via `--no-data-plane`, an older sidecar with no `supervised` field) all correctly self-restart.
- **`/api/admin/browse` is local-mode only.** It 404s whenever a token is enforced; the UI hides the "Browse…" button unless status reports local. Don't expose the server filesystem to a remote browser.
- **Redaction is round-trip-critical.** `restore_redacted_secrets` on `PUT` is what keeps an untouched secret from being written as the literal `"***REDACTED***"`; the raw-dict round-trip (not `asdict`) is what keeps unsurfaced/advanced keys alive across a save.
- **The token is an ephemeral runtime capability, never stored at rest** — no browser-side encryption is added (the JS holding the key beside the ciphertext would be theater). A `sudo`-style fresh-token re-prompt for mutating admin actions is the meaningful store-free hardening, deferred as a follow-up.

## Follow-ups

`sudo`-style fresh-token re-prompt for mutating actions; a live `do_action("reconfigure")` for incremental source add/remove without a full restart+rescan; auto-restart-on-save; write/upload passthrough for proxied sources.
