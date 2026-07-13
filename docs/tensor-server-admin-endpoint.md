# Tensor-Server Admin Endpoint — Config / Status / Restart over HTTP

**Status:** Design / not yet implemented
**Component:** `biopb-tensor-server` (`http_server.py`, config model), `biopb`
umbrella CLI (`server restart` lever), `biopb-mcp` (napari "open admin" action)
**Related:** `biopb/biopb#34` (config validation + JSON Schema emitter),
`biopb/biopb#212` + `docs/progressive-discovery.md` (startup-scan progress /
freshness `health` fields), `docs/remote-tensor-cache.md` §8 (the config-GUI
design this realizes), `docs/config-format-migration.md` (JSON-canonical config).

---

## Goal

Give a user a way to **reconfigure, inspect, and restart the local tensor
server** without hand-editing `~/.config/biopb/biopb.json` and without a
terminal. Concretely, three capabilities:

1. **Rewrite the server config** — edit `sources` (local dirs/files and, later,
   proxied remotes) plus cache/pyramid/server knobs.
2. **Check status** — running / health / live discovery-scan progress.
3. **Restart the server** — apply a config change (the server reads config
   **once at startup**; there is no hot-reload).

This is the **`/api/config` web-app frontend** from
`docs/remote-tensor-cache.md` §8.6, chosen over a napari Qt form. The backend is
a small set of HTTP routes on the **existing FastAPI sidecar** (`:8814`); napari
integration is a single menu action that opens a browser window at the admin
page. No second form to keep in sync with `_CONSTRAINTS`.

### Why the web app, not a napari dock widget

- The sidecar already has token auth (`_check_token`), health endpoints
  (`/livez`/`/readyz`/`/healthz`), a `TensorFlightClient` to the Flight server,
  and a shipped React app served from the same origin.
- "Open a browser window to the endpoint" is the cheapest possible napari
  integration — `webbrowser.open(f"{base}/admin?token=…")` — and works for a
  headless server admin too.
- It keeps the schema-driven form in **one** place (the React app) instead of
  reimplementing per-field validation in Qt.

The editing **backend** (load / validate / `save_config`) lives in `biopb` root
(per `remote-tensor-cache.md` §8.2), so a Qt form could reuse it later; we do not
build one now.

---

## The four routes

All mount on the existing app in `http_server.py:create_app`.

| Route | Method | Does |
|---|---|---|
| `/api/config` | `GET` | Return the raw `biopb.json` dict, its on-disk path, and the JSON Schema (`build_config_schema()`) so the frontend can render a schema-driven form. |
| `/api/config` | `PUT` | Validate the request body against the schema → `save_config()` atomically → return `{saved: true, restart_required: true}`. Does **not** restart. |
| `/api/admin/status` | `GET` | `running / health / scan progress / pid / version / config path`. Proxies the Flight `health` action and adds process facts. |
| `/api/admin/restart` | `POST` | Spawn a **detached** `biopb server restart`, return `202 {restarting: true}`. The client then polls `/livez` → `/api/admin/status` until the new daemon answers. |

The existing unauthenticated `/livez` / `/readyz` are what the frontend polls
**through the restart gap** — the `/api/admin/*` routes themselves go dead while
the process is down.

### `GET /api/config`

Returns the config as a **raw dict** (not the parsed dataclass), plus the schema
and path:

```json
{
  "path": "/home/user/.config/biopb/biopb.json",
  "config": { "server": {...}, "cache": {...}, "sources": [...] },
  "schema": { "$schema": "https://json-schema.org/draft/2020-12/schema", ... }
}
```

Reading the raw dict (not `dataclasses.asdict(load_config(...))`) is what lets
the round-trip in `PUT` preserve keys the form does not surface (see
`save_config` below).

### `PUT /api/config`

1. Validate the body with `Draft202012Validator(build_config_schema())`. On
   failure, return `422` with the structured `jsonschema` errors (path + message
   per field) so the form can show inline errors. This is the **JSON Schema
   emitter's first runtime consumer** — the same `_CONSTRAINTS` table drives both
   server-side `__post_init__` validation and this pre-write check, so they
   cannot disagree (`#34`).
2. `save_config(body, path)` (see below). Return
   `{saved: true, restart_required: true}`.

The write does **not** trigger a restart — the user reviews status and presses
**Restart** explicitly. (Auto-restart-on-save is a later convenience, not v1.)

Because the published schema keeps `additionalProperties: true`, unknown/advanced
keys pass validation (matching the migration-window warn-not-reject posture),
while the dangerous values `#34` exists to catch (`downscale_factor` 0/1,
out-of-range `port`, bad `backend`, …) are rejected before they ever reach disk.

### `GET /api/admin/status`

Merges the Flight `health` action with process facts:

```json
{
  "running": true,
  "pid": 12345,
  "version": "…",
  "config_path": "/home/user/.config/biopb/biopb.json",
  "health": "SERVING",
  "source_count": 1287,
  "writable": true,
  "uptime_seconds": 42,
  "full_scan_in_progress": true,
  "last_full_scan_finished_at": null
}
```

`source_count`, `full_scan_in_progress`, and `last_full_scan_finished_at` are the
progressive-discovery freshness fields (`docs/progressive-discovery.md`) — they
drive the post-restart "scanning…" UX below.

### `POST /api/admin/restart`

See **Self-restart** — the only architecturally non-obvious route.

---

## The hard part: self-restart

The admin endpoint runs **inside the very daemon it must restart**. The daemon is
launched detached (`biopb.cli` `start()` uses `_detach_kwargs()` Popen) with **no
supervisor** in the personal/small-lab target (no systemd, no k8s). So a request
handler **cannot** kill-and-respawn its own process from within: it would die
mid-request with nothing to bring it back.

### Solution: shell out to the existing managed lever, detached

`POST /api/admin/restart` spawns a **detached child** running the existing
umbrella command:

```
biopb server restart --config <path> --web-port <p> --web-host <h> [--static-dir <d>]
```

using the same detach kwargs `start()` uses. It returns `202` immediately and
ends the request. That child **outlives the dying parent** and runs the existing
`restart()` flow verbatim:

- graceful stop of the current daemon — `SIGTERM` on POSIX, the
  shutdown-**sentinel file** on Windows (`http_server.shutdown_sentinel_path()` /
  `_install_windows_shutdown_listener`), with PID-identity checks so a reused PID
  is never killed;
- wait for the port to free;
- `start()` re-binds and re-launches a fresh daemon.

This reuses code that already handles PID identity, port-collision refusal, log
rotation, and the Windows sentinel path. **Zero new teardown logic.** It is
exactly `remote-tensor-cache.md` §8.5's "restart via the existing managed lever."

The browser reconnects by polling `/livez` (unauthenticated, answers as soon as
the new daemon binds) and then `/api/admin/status`.

### The one required plumb: restart *identically*

A bare `biopb server restart` falls back to **default** options (`DEFAULT_CONFIG`,
web port 8814, default static dir, no token). If the running daemon was launched
with a non-default config path, port, host, static dir, or an access token, a
bare restart would come back **mismatched** (wrong config, wrong port, or
suddenly token-gated/un-gated).

So the daemon must know its own launch arguments and echo them into the restart
command. Thread them `launch → create_app → the admin-route closure`:

- `create_app` already receives `flight_location`, `token`, `static_dir`; extend
  it (and `run()` / `launch`) to also carry the **config path** and the
  **web host/port** it was started with, and stash them on the app so the restart
  route can reconstruct the exact command.
- The **access token** is passed through the environment the child inherits
  (`BIOPB_TENSOR_TOKEN`; local mode with a loopback bind needs no token),
  matching how `start()` already hands it to the daemon — so it is **not** placed
  on the visible command line.

This is the only genuinely new wiring restart needs.

---

## Config writer prerequisite — `save_config`

The server-config path is **read-only** today: `load_config` / `parse_config`
parse JSON/TOML → dataclasses, and nothing serializes back
(`remote-tensor-cache.md` §8.2). Add `save_config(data: dict, path)` in the
config model that:

- writes canonical `biopb.json` **atomically** (temp file + `os.replace`,
  mirroring biopb-mcp's `_atomic_write_json`);
- **round-trips on the raw dict, not the dataclass** — load the existing file,
  edit by key, re-serialize — so advanced or future keys the form does not
  surface **survive** the write (`dataclasses.asdict()` would clobber them);
- migrates a legacy `biopb.toml` forward to JSON and backs the old file up, so
  the server's both-files shadow warning never fires (`#34`).

Per §8.2 the **pure-data** config layer (the dataclasses, `_CONSTRAINTS`,
`_validate_config`, `parse_config` / `load_config` / the new `save_config`) is
the part that moves down into `biopb` root; the adapter-dependent expansion
(`discover_sources`, `detect_source_type`, `resolve_all_sources`) stays in
`biopb-tensor-server`. That relocation is the GUI's prerequisite, not part of
this endpoint per se, but `save_config` lands with it so the same backend serves
a future web `/api/config` page and any later Qt form.

---

## UX specification — the admin page

The four routes are the contract; this is the page the user actually sees. It is
a **new React route `/admin`** in the shipped web app, reusing the existing
shell and idioms (`app-shell` topbar/sidebar/main, `status-pill`, the `/unlock`
token gate + reveal-token field, `error-toast`) so it reads as the same product,
not a bolted-on form. The browse UI stays read-only at `/`; `/admin` is the only
**editing** surface.

### Entry

- **From the web app:** an **Admin / ⚙ gear** button in the topbar of `/`
  (next to the lock button), routing to `/admin`. A "← Back to browser" link
  returns to `/`.
- **From napari:** the "open admin" action lands directly on
  `/admin?token=…` (same `?token` capture + URL-strip the bootstrap already does
  for `/`), so the scientist never sees the token.
- Same unlock gate as `/`: no/invalid token under a token-gated server → redirect
  to `/unlock`, then back to `/admin`.

### Layout

```
┌─ topbar ───────────────────────────────────────────────────────────┐
│ biopb · [● SERVING · 1,287 sources · up 4m]        [Restart] [⚙][🔒] │
├─────────────────────────────────────────────────────────────────────┤
│  Sources                                              [+ Add ▾]      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 📁 /data/microscopy        local · monitor ✓        [edit][✕]  │  │
│  │ 🗄 grpc://lab-nas:8815      remote · alias "nas"     [edit][✕]  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ▸ Server         (host, port, log level, …)            advanced    │
│  ▸ Cache          (backend, segment/total size, …)      advanced    │
│  ▸ Pyramid / precache                                   advanced    │
│                                                                     │
│  [ Save ]   ·   unsaved changes — restart required to apply         │
└─────────────────────────────────────────────────────────────────────┘
```

The headline pane is the **Sources editor**; server/cache/pyramid knobs are
**collapsed "advanced" sections** below it (most users only touch sources). The
persistent status read-out lives in the topbar `status-pill`, extending its
existing `idle|connecting|connected|error` vocabulary with `scanning` and
`restarting`.

> **Implemented (biopb/biopb#245).** The structured advanced sections
> (`AdvancedSections.tsx`) render one collapsible pane per schema section
> (`server`/`cache`/`pyramid`/`precache`/`metadata_db`), each field a
> schema-driven control (checkbox / enum `select` / bounded number / text) with
> its `description` as helper text; the raw-JSON modal stays as an escape hatch.
> Inline validation (`validateConfig()` in `@biopb/tensor-flight-client`,
> unit-tested) mirrors the server's `PUT` checks — hard enum, numeric bounds,
> required `url` — leaving the case-insensitive enums lenient (helper-text only),
> disables **Save** while any field errors exist, and force-opens any section
> holding an error. Deprecated keys (`watcher_type`, `poll_interval`, source
> `path`, `metadata_db.enabled`, the pyramid knobs under `[precache]`) render
> with a "deprecated" tag + canonical-key hint only when already present, and are
> never offered fresh. A **not-running** `/api/admin/status` (`running: false`)
> shows a degraded banner and relabels **Restart → Start**. A separate
> **`CredentialsEditor.tsx`** handles the nested `credentials.profiles` array
> (the home for `credentials_profile` targets): a profiles list + `default_profile`
> picker, `name` required inline, and `key`/`secret`/`token` as password fields
> seeded with the `REDACTED_SENTINEL` so an untouched secret round-trips and is
> preserved by `restore_redacted_secrets` on `PUT`.

### Sources editor (the headline)

A list of source rows, each showing **type · url · `source_id`/`alias` · monitor**
with inline **edit** and **remove (✕)**. **`+ Add ▾`** offers:

- **Add local folder / file** — a path field (+ optional `source_id`,
  `dim_labels`, `monitor`).
- **Add remote tensor server…** — url (`grpc://host:port`), `alias`,
  `credentials_profile`/token — the concrete home of
  `remote-tensor-cache.md` §7.3's lost direct-connect workflow.

Each field validates **inline against the schema** (`build_config_schema()`,
served by `GET /api/config`): bad enum / out-of-range / missing required `url`
shows a red field message *before* save, and the case-insensitive enums
(`log_level`, `reduction_method`) surface their accepted set from the property
`description` as helper text. Deprecated keys (`watcher_type`, source `path`, …)
are accepted on load but rendered with a "deprecated" tag and a hint to the
canonical key, never offered in the add forms.

### The Save → Restart flow (the core interaction)

Two **distinct, sequential** actions — config is written first, applied second —
because the server reads config only at startup:

1. **Edit → Save.** Any change marks the form dirty and enables **Save**. Save
   issues `PUT /api/config`:
   - `422` → render the structured `jsonschema` errors **inline on the offending
     fields** plus a top-of-form summary; nothing is written.
   - `200` → show a **persistent "Saved — restart required to apply"** banner
     (so a saved-but-unapplied config is never invisible) with a **Restart now**
     button. The form is no longer dirty.
2. **Restart.** The topbar **Restart** button (always present) and the banner's
   **Restart now** both open a **confirm modal** — restart **interrupts the
   shared live session**: connected clients (the napari/MCP kernel, browser
   viewers, in-flight analyses) drop while the daemon bounces. On confirm:
   - `POST /api/admin/restart` → `202`; the whole page enters a **disabled
     "Restarting…"** state (form locked, pill → `restarting`).
   - The page polls `/livez` then `/api/admin/status` and narrates progress (see
     **Post-restart UX** below): `Restarting… → Reconnected — scanning N… →
     Ready — N sources`.
   - If the new daemon does not answer within a timeout (e.g. 60 s), surface a
     **"server did not come back"** error toast pointing at `biopb server status`
     / the log file — never an infinite spinner.

Restart is available **independently of a config change** (a plain "bounce the
server" affordance), but Save never auto-restarts.

### Error / edge states

- **Validation** — inline per-field + summary; Save stays disabled while errors
  exist.
- **Save failure** (write/permission error from `save_config`) — error toast with
  the path; config on disk is untouched (atomic write).
- **Unreachable / not running** — if `/api/admin/status` reports the daemon down,
  the page shows a degraded banner; the config form may still load and Save (the
  file is editable), but Restart becomes **Start**.
- **Empty / first run** — no sources yet → the editor leads with a prominent
  **"Add a data folder"** CTA rather than an empty list.

### Safety affordances (recap)

- **Confirm modal on restart** (interrupts the shared session).
- **Persistent "restart required" banner** until applied.
- **Same-origin guard** on the mutating routes (transparent to the user; see
  Security).
- **Optional `sudo`-style token re-prompt** for Save/Restart (deferred hardening,
  below) — would reuse the `/unlock` reveal-token field as a one-shot modal, not
  a stored value.

## Post-restart UX — show the scan, don't blind-wait

After `202`, the frontend polls `/api/admin/status`. On a large or cloud root the
startup discovery scan can run for **minutes** with no output — `#212` measured
**142 s of silence** that "looks hung." The admin page must not reproduce that
blind wait. Using the freshness fields:

- while `/livez` is dead → "**Restarting…**"
- once it answers but `full_scan_in_progress` is true → "**Reconnected —
  scanning… N sources**" with the climbing `source_count`
- when `last_full_scan_finished_at` is set and `health == "SERVING"` →
  "**Ready — N sources**"

The daemon reaches `SERVING` immediately (backgrounded scan,
`docs/progressive-discovery.md`), so the page watches the catalog **populate
live** rather than waiting on a complete scan. `#212` (server-side progress
logging) and the progressive-discovery freshness `health` fields are the
**server-side signal**; this page is their **consumer**.

---

## Security / auth

The token is an **ephemeral runtime capability, never stored at rest**: `start()`
takes `--token` / `BIOPB_TENSOR_TOKEN` or auto-generates
`secrets.token_urlsafe(32)`, prints it once, and holds it only in the daemon's
process env. It is **not** in `biopb.json` (the `tensor-server.pid` file holds an
unrelated process-identity token), and biopb-mcp's `persist_url()` writes the URL
back to its config but **deliberately not the token**. The web app keeps it in
**`sessionStorage` (`biopb_token`), plaintext**, per-tab, cleared on tab close —
treated as a same-origin **UI gate**, not a hardened secret (the UnlockPage says
as much: *"This token gates the web UI only — it does not affect server data"*).

Implications that shape the admin auth design:

- **No token store to encrypt — and encrypting the browser copy would be
  theater.** The same JS that uses the token would hold the decryption key beside
  the ciphertext. We add **no** browser-side encryption and **no** new at-rest
  token anywhere.
- **Admin routes share the existing `_check_token`** (the decision for this
  endpoint): a configured token gates them exactly as it gates the read routes,
  and the localhost no-token **dev-bypass** applies the same way.
- **But the admin routes are the first *mutating* surface.** Read routes being an
  unauthenticated localhost gate has never mattered; `PUT /api/config` and
  `POST /api/admin/restart` under dev-bypass would rewrite config / restart with
  **no credential**. A page the user merely visits can fire a cross-origin
  `POST http://127.0.0.1:8814/api/admin/restart` — it can't read the response
  (CORS), but a state change doesn't need to. So the mutating routes get a
  **store-free same-origin guard**: require a non-simple request header
  (the existing `X-Biopb-Token`, or check `Sec-Fetch-Site: same-origin`). A
  cross-origin browser form cannot set a custom header without a CORS preflight,
  and the sidecar's CORS allowlist is localhost-only, so the preflight fails.
  This blocks drive-by browser CSRF even with no token, at zero storage cost, and
  pairs naturally with the same-origin napari "open admin" flow.

### Optional hardening (not v1-required)

Because admin is privileged and shares the same weak, XSS-readable,
sessionStorage token as the read UI, a **`sudo`-style re-prompt** is the
meaningful (still store-free) way to make admin assurance exceed read-UI
assurance: for `PUT /api/config` / `POST /api/admin/restart`, **do not reuse the
stored token** — prompt for it fresh and send that one-shot value. A token leaked
from storage (or replayed by an XSS) then cannot silently rewrite config or
restart, because the privileged action requires a value that was never persisted.
This is the defense encryption-at-rest cannot give; flagged as a follow-up, not a
v1 gate.

---

## napari integration

A single menu action / one-button dock widget in `biopb-mcp` that resolves the
sidecar base URL + token the way the browser bootstrap already does and calls:

```python
webbrowser.open(f"{base}/admin?token={token}")
```

No form logic in Qt — the browser is the form, and the open is same-origin by
construction. This is the cheap payback for `remote-tensor-cache.md` §7's lost
direct-connect UI.

---

## Relationship to deferred `$schema` embedding / auto-validation (`#34`)

`config-format-migration.md` ## Deferred lists *"Embed `$schema` in generated
configs"* — write a `"$schema"` pointer so editors validate the config out of the
box — **blocked on "the schema hosted at a stable URL"** (the `$id`
`https://biopb.org/schemas/tensor-server-config.json`). This endpoint and
`save_config` are where that item gets resolved, and resolve it **without the
hosting dependency**:

- **`save_config` is the natural writer for `$schema`.** When it writes
  `biopb.json`, it also drops a sibling `biopb.schema.json` (the output of
  `build_config_schema()`, same as `biopb-tensor-server config-schema -o`) and
  embeds a **relative** pointer `"$schema": "./biopb.schema.json"`. A relative
  local `$schema` resolves from disk, so editors (VS Code, etc.) get
  **auto-validation offline, with no hosted URL** — the original blocker
  disappears. When the `$id` URL is eventually hosted, the embedded pointer can
  switch to it; until then the local file is the source.
- **Hosting the `$id` is a small, well-precedented add when wanted.** The
  landing site (`../biopb-site`) rsyncs its static root to
  `/var/www/biopb.org/` on push-to-main, so a file at
  `biopb-site/schemas/tensor-server-config.json` serves at exactly the `$id`
  `https://biopb.org/schemas/tensor-server-config.json`. To avoid hand-maintaining
  it against the emitter, generate it from `build_config_schema()` and publish it
  **out-of-band from the biopb release workflow** — the same pattern that already
  owns `/docs/api/<lang>/` (generated on a `v*` / `release-v*` tag and rsynced into
  the site, carved out of the site's own `--delete`). So the offline relative
  `$schema` is the default; the hosted `$id` is the published canonical once that
  one-step CI job is added.
- **One schema, three consumers, no drift.** The editor's `$schema`, the admin
  form's `PUT /api/config` validation, and the server's `_CONSTRAINTS`
  `__post_init__` checks all resolve to the same `build_config_schema()`. The
  admin `GET /api/config` already serves that schema live, so the form never
  ships a stale copy.
- **This is editor/pre-flight validation, distinct from the end-state flip.** It
  does not change the server's warn-level posture (`_STRICT_VALIDATION = False`);
  it makes the *authoring* surface catch errors early. The separate deferred
  "flip warn → raise" end state is unaffected and still gated on dropping the
  `.toml` read path.

**Implementation note — `$schema` must be a tolerated meta key.** Embedding
`"$schema"` (and any `"$id"`) as a top-level key means the config loader and the
`#234` unknown-key warning (`_warn_unknown_config_keys`) must **whitelist** these
`$`-prefixed meta keys so they are neither parsed as a config section nor warned
about. Add them to the known-top-level set (or skip `$`-prefixed keys) when
`config_schema.known_config_keys()` is derived.

## Scope

**v1:** the four routes; `save_config` (atomic, raw-dict round-trip, sibling
`biopb.schema.json` + relative `$schema` embed); schema validation on `PUT`;
detached `biopb server restart` with launch-arg pass-through; post-restart
scan-progress polling; the same-origin guard on mutating routes; the napari "open
admin" action.

**Follow-ups:**

- `sudo`-style fresh-token re-prompt for mutating admin actions (above).
- A live `do_action("reconfigure")` for incremental source add/remove without a
  full restart + rescan (`remote-tensor-cache.md` §8.5 / §10).
- Auto-restart-on-save convenience.
- Write/upload passthrough for proxied sources (`remote-tensor-cache.md` §10).

---

## Where to look first

- **Sidecar app + auth + lifecycle:** `biopb-tensor-server/biopb_tensor_server/http_server.py`
  (`create_app`, `_check_token`, `run`, `shutdown_sentinel_path`,
  `_install_windows_shutdown_listener`).
- **The restart lever:** `src/main/python/biopb/cli.py`
  (`server start` / `stop` / `restart` / `status`, `_detach_kwargs`,
  `_read_pid_record`, `_graceful_stop`).
- **Config model + schema:** `biopb-tensor-server/biopb_tensor_server/config.py`
  (`_CONSTRAINTS`, `load_config`, `parse_config`; new `save_config`) and
  `config_schema.py` (`build_config_schema`).
- **Scan-progress signal:** `docs/progressive-discovery.md` and the Flight
  `health` action's freshness fields.
- **The web app to extend (shell, routes, idioms):**
  `web/packages/app/src/` — `main.tsx` (routes `/viewer`, `/admin`, `/unlock`),
  `pages/HomePage.tsx` (`app-shell` topbar/sidebar/main, `status-pill`),
  `pages/AdminPage.tsx`, `pages/UnlockPage.tsx` (token gate + reveal field),
  `components/SourceTree.tsx` (read-only browse the editor mirrors), `store.ts`.
  (Now the top-level `web/` workspace, served by the control; see `web/README.md`.)
- **The broader GUI design this implements:** `docs/remote-tensor-cache.md` §§7–8.
