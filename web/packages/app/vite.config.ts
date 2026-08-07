import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// The control front is biopb's single web origin: it serves this bundle at its
// root and proxies the data plane under /data_plane and each MCP session under
// /session/<id>. So the build is always root-based (base "/"): index.html
// requests /assets/*, which resolve at the control root no matter which prefix
// (/, /viewer, /session/<id>/observe) the SPA shell was served under.
//
// The *build* base stays "/" even when the control is published below the root
// (`--url-prefix`, an Open OnDemand app): that prefix names a compute node and a
// port allocated at job start, so it cannot be baked here. The control injects it
// into the served index.html instead and the app reads it at runtime — see
// src/base.ts, which feeds the router basename and every URL the app builds.
//
// `pnpm dev` can stand in for that. Set BIOPB_URL_PREFIX to the same value the
// control was started with and the dev server reproduces all three things the
// control does, so the prefixed app can be developed against without a build:
//
//   BIOPB_URL_PREFIX=/node/$host/$port biopb control start
//   BIOPB_URL_PREFIX=/node/$host/$port pnpm dev
//
// The prefixed path is forwarded to the control **unrewritten**, so the control's
// own stripping middleware is genuinely under test rather than mocked out. Start
// the control without the flag and the requests 404 — a loud failure, not a
// silent divergence. Unset (the common case) is a plain root-origin dev server
// and every line below is a no-op.
const urlPrefix = (() => {
  const segments = (process.env.BIOPB_URL_PREFIX ?? "").trim().split("/");
  const kept = segments.filter(Boolean);
  return kept.length ? "/" + kept.join("/") : "";
})();

/**
 * Inject `window.__BIOPB_BASE__`, the way the control's index.html rewrite does.
 *
 * Vite's own `base` handles the asset and module URLs, but nothing would tell the
 * *app* its prefix, so src/base.ts would read "" and build root-absolute URLs
 * while living under the prefix — the router basename would not match the
 * location and the page would come up blank.
 *
 * Deliberately the same global rather than teaching base.ts to fall back to
 * `import.meta.env.BASE_URL`: the app keeps one source of truth, and dev
 * exercises the same code path production does instead of a parallel one that
 * could rot unnoticed.
 */
function injectBaseGlobal(prefix: string): Plugin {
  return {
    name: "biopb:inject-base-global",
    apply: "serve",
    transformIndexHtml: () =>
      prefix
        ? [
            {
              tag: "script",
              injectTo: "head-prepend",
              children: `window.__BIOPB_BASE__=${JSON.stringify(prefix)};`,
            },
          ]
        : [],
  };
}

// The proxy keys have to carry the prefix too, since that is what the app now
// requests. A key beginning with ^ is matched as a RegExp by vite, so the prefix
// is escaped before being spliced in.
const rx = urlPrefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const control = { target: "http://localhost:8813", changeOrigin: true };

export default defineConfig({
  base: urlPrefix ? `${urlPrefix}/` : "/",
  plugins: [react(), injectBaseGlobal(urlPrefix)],
  resolve: {
    dedupe: ["react", "react-dom"],
  },
  server: {
    // `pnpm dev` runs against a live control on :8813. Forward the API namespaces
    // the SPA calls: the control's own /api/*, the proxied data plane /data_plane/*
    // (incl. the /data_plane/ws/render websocket), and each session's
    // /session/<id>/api/*. The viewer defaults to the proxied plane in dev
    // (ClientBootstrap), so no env var is needed.
    proxy: {
      [`^${rx}/api`]: control,
      [`^${rx}/data_plane`]: { ...control, ws: true },
      // /health is unauthenticated and outside /api, and auth.ts reads it twice
      // — authRequired() and consoleEnabled(). Unproxied it fell to the SPA
      // fallback, and since both callers treat any failure as false, dev
      // silently reported "no token needed, no console" whatever the control
      // said: the console editor could never appear on a dev server.
      [`^${rx}/health`]: control,
      // Only the session *API* proxies to control — NOT the observe page at
      // /session/<id>/observe, which must fall through to vite's SPA fallback so
      // the dev bundle + HMR serve it (a bare "/session" prefix would proxy the
      // page HTML to control's built dist and break the dev module graph).
      [`^${rx}/session/[^/]+/api`]: control,
      // The user console is a *separate* root from the session api on purpose
      // (the control proxies it only on a loopback bind), so it needs its own
      // rule — without it a submitted cell POSTs into the SPA fallback above,
      // gets index.html back, and fails on the JSON parse with the editor
      // looking perfectly functional.
      [`^${rx}/session/[^/]+/console`]: control,
    },
  },
});
