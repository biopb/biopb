// The path prefix this bundle is served under, learned at **runtime**.
//
// The control is normally the origin root, but it can be published below it —
// `biopb control start --url-prefix /node/<host>/<port>`, an Open OnDemand
// interactive app (biopb/biopb#728, `docs/url-prefix.md`). That prefix carries a
// compute node's hostname and a port allocated at job start, so there is no
// build-time answer: `vite build --base=…` cannot bake it and
// `import.meta.env.BASE_URL` is always "/". The control instead rewrites the
// served index.html, injecting `window.__BIOPB_BASE__` (plus a `<base href>` for
// the document's own relative URLs), and this module is where the app reads it.
//
// **Every root-absolute URL the app builds must go through `withBase`** — API
// fetches, full-page navigations, and asset `src`s alike. A bare `fetch("/api/x")`
// or `href="/viewer"` works at the root and silently leaves the app's namespace
// under a prefix, which is the whole bug this exists to prevent. Router paths are
// the exception: they are already relative to `<BrowserRouter basename>`, so
// `navigate("/viewer")` and `<Route path="/viewer">` stay as they are, and
// `appPath` converts a real `window.location.pathname` back into one.

declare global {
  interface Window {
    __BIOPB_BASE__?: string;
  }
}

// The same shape the control enforces before it injects the value (letters,
// digits and a few path punctuation marks per segment). Re-checked here rather
// than trusted: a value that is not a plain same-origin path — `//evil.com`,
// `https://evil.com`, a backslash the URL parser reads as an authority — would
// turn every link and fetch in the app into an off-origin request. Anything that
// does not match degrades to "no prefix", which is the safe reading.
const SAFE_BASE = /^\/[A-Za-z0-9._~!$&'()*+,;=@-]+(?:\/[A-Za-z0-9._~!$&'()*+,;=@-]+)*$/;

function normalize(raw: unknown): string {
  if (typeof raw !== "string") return "";
  const trimmed = raw.trim().replace(/\/+$/, "");
  if (!trimmed || trimmed === "/") return "";
  const candidate = trimmed.startsWith("/") ? trimmed : "/" + trimmed;
  return SAFE_BASE.test(candidate) ? candidate : "";
}

/**
 * Whether this document actually *arrived* under `prefix`.
 *
 * The control rewrites the shell once at startup and serves that one document to
 * every request, prefixed or not — it never sees the request path. So a direct
 * `http://127.0.0.1:8813/` on a prefixed control is handed the prefixed shell
 * too, and taking the injected value at face value there would set the router
 * basename to `/node/h/p` while `location.pathname` is `/`. React Router does not
 * fall back: `stripBasename` returns null, `<Router>` renders null, and the whole
 * tree disappears — a blank page with only a console warning. That root is the URL
 * `biopb ui` opens and the one the `ssh -L` hint points at, so it has to work.
 *
 * Reading the location rather than the injected value also keeps the case where a
 * proxy *strips* the prefix before the control sees it: the browser is still at a
 * prefixed URL, so the app still needs the prefix even though the request did not
 * carry one by the time it landed.
 */
function servedUnder(prefix: string): boolean {
  if (!prefix) return false;
  const here =
    typeof window === "undefined" ? "" : (window.location?.pathname ?? "");
  return here === prefix || here.startsWith(prefix + "/");
}

/** The prefix this app is served under: "" at the origin root, else "/a/b". */
const declared = normalize(
  typeof window === "undefined" ? "" : window.__BIOPB_BASE__,
);
export const BASE = servedUnder(declared) ? declared : "";

/** A root-relative path (`/api/status`, `/biopb-logo.png`) placed under the
 *  prefix. Use for anything the browser resolves against the origin: `fetch`,
 *  `window.location`, `<a href>`, `<img src>`. */
export function withBase(path: string): string {
  return BASE + path;
}

/** The inverse: a real `window.location.pathname` as a router path, so it can be
 *  handed to `navigate()` or round-tripped through `?next=`. Without this, a
 *  pathname captured under a prefix (`/node/h/p/admin`) would be navigated to
 *  *again* under the router basename and land at `/node/h/p/node/h/p/admin`. */
export function appPath(pathname: string): string {
  if (BASE && (pathname === BASE || pathname.startsWith(BASE + "/"))) {
    return pathname.slice(BASE.length) || "/";
  }
  return pathname;
}
