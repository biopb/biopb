# Publishing the control under a URL prefix

The control plane is the single web origin, and it normally lives at `/`.
`--url-prefix` publishes it under a path prefix instead — e.g.
`/node/mantis-051/29847/` — so it can be served through a reverse proxy that
mounts it below the root. The driver is an **Open OnDemand interactive app**
(biopb/biopb#728), whose `/node/<host>/<port>/` route passes the full, untouched
path to the backend and rewrites nothing in the response body.

The prefix is a per-job value: it carries the compute node's hostname and a port
allocated at job start. There is therefore no build-time answer — `vite build
--base=…` cannot bake it — and the SPA has to learn it at run time.

## Configuring it

```
biopb control start --url-prefix /node/$host/$port     # or BIOPB_URL_PREFIX
biopb control run   --url-prefix /node/$host/$port
python -m biopb_control run --url-prefix /node/$host/$port ...
```

Normalized once, in `build_app`: one leading slash, no trailing slash, empty
segments dropped; `None` / `""` / `"/"` all mean "serve at the root". Unset is a
no-op — the origin behaves exactly as it always has.

**Configuration only.** The prefix is never read off a request header such as
`X-Forwarded-Prefix`. A request-controlled `<base href>` would let any caller
repoint every relative URL in the served document at an origin of their choosing
— a considerably worse bug than the one being fixed. Nothing needs inferring: an
OnDemand `before.sh` knows `$host` and `$port` before the job starts.

**It must also be a *path*.** `normalize_url_prefix` rejects anything that is not
one, and the entry points (`biopb control start` / `run`, `python -m
biopb_control run`) exit 2 naming the offending segment rather than starting.
This is hardening, not a patched exploit — whoever sets the prefix can already
pass `--static-dir` or `PYTHONPATH` — but it is what makes "the prefix can only
name a path on this origin" a property of the code:

| rejected | because |
| --- | --- |
| `\`, tab, newline, space | WHATWG URL parsing resolves `<base href="/\evil.com/">` to `http://evil.com/` — a backslash after the leading slash enters the *authority*, so every relative URL and every `new URL(x, document.baseURI)` leaves the origin. Browsers strip tabs/newlines *before* parsing, so those reach the same position. |
| `?`, `#` | a query or fragment in a `<base href>` silently changes what every relative URL resolves to |
| `%` | `scope["path"]` reaches the middleware percent-*decoded* while the shell carries the prefix *encoded*; barring it keeps the two identical by construction |
| `:` | legal in a path segment, but excluded so the likely slip — `--url-prefix https://host/biopb` — fails loudly instead of quietly becoming `/https:/host/biopb` |
| `.`, `..` segments | would make the served `<base href>` and the path the middleware strips disagree |

The prefix is additionally HTML-escaped at each of the sites it lands in (the
`<base href>`, the rewritten `src=`/`href=` values, and the JSON-quoted script
literal), so neither the charset nor the escaping is load-bearing on its own.

## What the control does (`biopb-control/src/biopb_control/_control.py`)

**1. Strips the prefix off the request path.** `_URLPrefixMiddleware` rewrites
`scope["path"]` (and `raw_path`) for `http` and `websocket` scopes under the
prefix, so every route below sees byte-for-byte the request it would see at the
origin root. Two properties it must keep:

- It is the **outermost** middleware. `_ControlAuthMiddleware` decides what to
  gate by reading `scope["path"]` directly, so a still-prefixed
  `/node/h/p/api/data_plane/restart` would sail past its `startswith("/api/")` —
  an auth bypass, not merely a 404.
- Unprefixed requests pass through **untouched**. biopb-mcp's `_control_client`
  and the installer poll `http://127.0.0.1:8813/health` over loopback with no
  prefix and must keep working while a prefix is configured for the portal.

It deliberately does *not* use the ASGI `root_path` convention (leave the path
whole, name the prefix in `scope["root_path"]`), nor a hybrid of the two.
`Mount` composes `root_path + matched_path` for its sub-app, while
`get_route_path` subtracts `root_path` from `path` only when the path still
starts with it — so a stripped path *plus* a `root_path` makes that subtraction
silently no-op inside `/data_plane` and `/session/{id}`, and the sub-app sees its
own mount prefix again (`/ws/render` stops matching). The un-stripped variant
routes correctly but hands the auth gate a prefixed path, which is the bypass
above. Nothing in the control builds absolute URLs out of `root_path` — the
browser side is carried by the rewritten shell — so stripping outright is both
the simpler and the correct half.

**2. Rewrites the served SPA shell.** `_rewrite_shell_html` computes the
document once at `build_app` time (the bundle is static) with three edits:

- `<base href="<prefix>/">` first in `<head>`, so every *relative* URL in the
  document and every runtime `new URL(x, document.baseURI)` lands under the
  prefix;
- the root-absolute `src=` / `href=` values rewritten to `<prefix>/…` — the entry
  chunk, the stylesheet, the three icons. `<base>` has no effect on root-absolute
  URLs, so they must be rewritten regardless;
- `window.__BIOPB_BASE__ = "<prefix>"`, the runtime hook the SPA reads in place
  of the build-time `import.meta.env.BASE_URL`.

Nothing outside `index.html` is touched. The built bundle needs no more than
that: its lazy route chunks are relative module specifiers
(`import("./DashboardPage-*.js")`), which resolve against the importing module's
URL and so follow the prefix for free.

Serving this same rewritten document to an *unprefixed* request still boots the
app: the browser asks for `<prefix>/assets/…` on the same origin, and the
middleware strips the prefix straight back off. A direct
`http://127.0.0.1:8813/` therefore keeps working alongside the portal route.

## The web side

`window.__BIOPB_BASE__` is the runtime global the SPA consumes; it is absent when
no prefix is configured. `web/packages/app/src/base.ts` owns it:

- `BASE` — the normalized prefix, `""` at the origin root. Re-validated against
  the same shape the control enforces, degrading to `""` rather than trusting a
  value that could send the app off-origin.
- `withBase(path)` — a root-relative path placed under the prefix. **Every
  root-absolute URL the app builds goes through this**: API fetches, full-page
  navigations (`window.location`), `<a href>`, and asset `src`s. A bare
  `fetch("/api/x")` works at the root and silently leaves the app's namespace
  under a prefix, which is the whole bug.
- `appPath(pathname)` — the inverse, turning a real `window.location.pathname`
  back into a router path. Without it, a pathname captured under a prefix and
  handed to `navigate()` would be resolved against the basename *again* and land
  at `/node/h/p/node/h/p/admin`.

Router paths are the exception and stay as they are: `<Route path="/viewer">` and
`navigate("/viewer")` are already relative to `<BrowserRouter basename>`, which
`main.tsx` sets from `BASE`.

The data plane needs no special handling beyond `apiBase = withBase("/data_plane")`
in `ClientBootstrap.tsx`. That also carries the render WebSocket for free, since
`useRenderWebSocket.ts` resolves `${apiBase}/ws/render` against the page origin,
and `tensor-flight-client` is already parameterized on `apiBase`.

`vite.config.ts` keeps `base: "/"` — the runtime global, not the build-time base,
carries the prefix. There is deliberately **no** `VITE_TENSOR_API` any more: a
build-time `/data_plane` baked by CI would have been missing the prefix and would
have silently defeated it, so the value is derived at runtime instead and one
bundle serves every deployment.

## Developing against a prefix

`pnpm dev` can stand in for a prefixed control, so the prefixed app does not have
to be built to be worked on. Set `BIOPB_URL_PREFIX` to the same value on both
sides:

```sh
BIOPB_URL_PREFIX=/node/$host/$port biopb control start
BIOPB_URL_PREFIX=/node/$host/$port pnpm dev
```

`vite.config.ts` then reproduces all three things the control does: sets vite
`base` so the shell's asset URLs carry the prefix, injects
`window.__BIOPB_BASE__` through a dev-only `transformIndexHtml` plugin, and
builds its proxy keys with the prefix. It injects the *same global* the control
does rather than teaching `base.ts` to read `import.meta.env.BASE_URL`, so the
app keeps one source of truth and dev exercises the production code path instead
of a parallel one.

The prefixed path is forwarded to the control **unrewritten**, so the stripping
middleware is genuinely under test; start the control without the flag and
requests 404 rather than silently diverging. Unset, every part of this is a no-op
and the dev server is an ordinary root-origin one.

Note the dev proxy must carry every root the app calls — `/api`, `/data_plane`,
`/health`, `/session/<id>/api`, `/session/<id>/console`. A missing root does not
error: it falls through to vite's SPA fallback and the caller gets HTML where it
expected JSON. `consoleEnabled()` and `authRequired()` read `/health` and treat
any failure as `false`, so an unproxied `/health` silently reported "no token
needed, no console" on every dev server regardless of what the control said.

## Still true after this

The control speaks plain HTTP with no TLS (biopb/biopb#614), so publishing it
through a portal puts the access token on the wire in the clear unless the portal
terminates TLS in front of it. Prefix support makes an OnDemand app *possible*;
TLS is what would make it safe to enable by default.

The user console is gated on the control's own **bind** (`_session_proxy_roots`),
which a prefix does not change: a loopback-bound control published by a portal
still reads as local and still carries the console. That is the pre-existing
reverse-proxy caveat noted in `_control.py`, and it now has a named topology —
for OnDemand it is also the intent, since the portal authenticates the job's
owner and the kernel is theirs.
