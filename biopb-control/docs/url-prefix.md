# Publishing the control under a URL prefix

The control plane is the single web origin, normally at `/`. `--url-prefix`
(or `BIOPB_URL_PREFIX`) publishes it under a path prefix instead — e.g.
`/node/mantis-051/29847/` — for serving behind a reverse proxy that mounts it
below the root, such as an Open OnDemand interactive app. The prefix is a
per-job value (hostname plus a port allocated at job start), so it can't be a
vite build-time `--base`; the SPA learns it at runtime.

## Configuring it

```
biopb control start --url-prefix /node/$host/$port     # or BIOPB_URL_PREFIX
biopb control run   --url-prefix /node/$host/$port
python -m biopb_control run --url-prefix /node/$host/$port ...
```

`normalize_url_prefix` normalizes: one leading slash, no trailing slash, empty
segments dropped; `None`/`""`/`"/"` all mean "serve at the root," and unset is
a no-op. It is never read off a request header (`X-Forwarded-Prefix` or
similar) — only a value set at the process is trusted, since a
request-controlled `<base href>` would let any caller repoint every relative
URL on the page.

It must be a path: `normalize_url_prefix` rejects a segment containing `\`,
tab, newline, space, `?`, `#`, `%`, `:`, or equal to `.`/`..`, and every entry
point (`biopb control start`/`run`, `python -m biopb_control run`) exits 2
naming the offending segment. This matters because a `<base href>` is HTML: a
stray backslash after the leading slash parses into the authority
(`<base href="/\evil.com/">` resolves to `http://evil.com/`), and
`%`/`:`/`.`/`..` would each let the served page and the path the middleware
strips disagree. The prefix is additionally HTML-escaped everywhere it's
rendered (the `<base href>`, rewritten `src=`/`href=`, the JSON script
literal).

## What the control does (`biopb-control/src/biopb_control/_control.py`)

**`_URLPrefixMiddleware` strips the prefix off the request path** —
rewriting `scope["path"]`/`raw_path` for `http` and `websocket` scopes — and
is the **outermost** middleware, ahead of `_ControlAuthMiddleware`. Order
matters: the auth middleware decides what to gate by reading `scope["path"]`
directly, so a still-prefixed `/node/h/p/api/data_plane/restart` would sail
past its `startswith("/api/")` check if the prefix weren't already stripped —
an auth bypass, not just a 404. Unprefixed requests (biopb-mcp's
`_control_client`, the installer polling `/health` over loopback) pass
through untouched.

It strips the path outright rather than using ASGI's `root_path` convention,
because `Mount` composes `root_path + matched_path` for its sub-apps while
`get_route_path` only subtracts `root_path` when the path still starts with
it — a stripped path plus a set `root_path` would silently no-op that
subtraction inside `/data_plane` or `/session/{id}`, breaking their own
routing.

**`_rewrite_shell_html` rewrites the served SPA shell once, at `build_app`
time**, and serves the cached result to every request:

- `<base href="<prefix>/">` first in `<head>`, so relative URLs and
  `new URL(x, document.baseURI)` resolve under the prefix;
- root-absolute `src=`/`href=` values (the entry chunk, stylesheet, icons)
  rewritten to `<prefix>/…` — `<base>` has no effect on those;
- `window.__BIOPB_BASE__ = "<prefix>"`, the runtime hook the SPA reads
  instead of a build-time `import.meta.env.BASE_URL`.

Nothing else in the bundle needs touching: lazy route chunks are relative
module specifiers, which resolve against the importing module's URL. The
shell is served identically to every request, including an unprefixed
`http://127.0.0.1:8813/` — the direct root `biopb ui` opens and the `ssh -L`
hint points at — so the app itself, not the server, has to decide whether the
injected prefix applies (see `BASE` below).

## The web side (`web/packages/app/src/base.ts`)

`window.__BIOPB_BASE__` is absent when no prefix is configured.

- **`BASE`** — the normalized prefix, or `""` at the root. It degrades to
  `""` whenever `location.pathname` is not under the declared prefix (equal
  to it or starting with `prefix + "/"`) — this is what keeps the shared,
  unconditionally-rewritten shell safe: honoring the prefix at an unprefixed
  location would set the router basename to `/node/h/p` against a location of
  `/`, and React Router renders a blank page rather than falling back.
- **`withBase(path)`** — every root-absolute URL the app builds (API
  fetches, full-page navigations, `<a href>`, asset `src`) goes through this;
  a bare `fetch("/api/x")` would silently escape the prefix.
- **`appPath(pathname)`** — the inverse, turning a real
  `window.location.pathname` back into a router path, so a captured pathname
  isn't resolved against the basename twice.

Router paths (`<Route path="/viewer">`, `navigate("/viewer")`) stay relative
to `<BrowserRouter basename>`, set from `BASE` in `main.tsx`. The data plane
needs nothing beyond `apiBase = withBase("/data_plane")` in
`ClientBootstrap.tsx`.

`vite.config.ts` sets `base: "/"` for a normal build — conditional on
`BIOPB_URL_PREFIX` only for the `pnpm dev` prefix-testing path below. There is
no `VITE_TENSOR_API` build-time variable — a CI-baked `/data_plane` would miss
the prefix, so the value is derived at runtime and one bundle serves every
deployment.

## Developing against a prefix

```sh
BIOPB_URL_PREFIX=/node/$host/$port biopb control start
BIOPB_URL_PREFIX=/node/$host/$port pnpm dev
```

`vite.config.ts` reproduces what the control does under this env var: sets
vite `base` for asset URLs, injects `window.__BIOPB_BASE__` via a dev-only
`transformIndexHtml` plugin, and builds the dev proxy keys with the prefix —
unrewritten, so the stripping middleware is genuinely exercised (a control
started without the flag 404s rather than silently diverging). The dev proxy
must cover every root the app calls (`/api`, `/data_plane`, `/health`,
`/session/<id>/api`, `/session/<id>/console`); a missing one falls through to
vite's SPA fallback and returns HTML where JSON was expected —
`consoleEnabled()`/`authRequired()` read `/health` and treat any failure as
`false`.

## Still true after this

The control speaks plain HTTP with no TLS, so a prefix published through a
portal puts the access token on the wire in the clear unless the portal
terminates TLS in front of it.

The user console is gated on the control's own bind
(`console_enabled = not host_is_public_bind`), independent of the prefix: a
loopback-bound control published through a portal still reads as local and
still carries the console — which for an OnDemand deployment is also the
intent, since the portal authenticates the job's owner.
