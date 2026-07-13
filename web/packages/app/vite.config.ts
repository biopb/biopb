import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The control front is biopb's single web origin: it serves this bundle at its
// root and proxies the data plane under /data_plane and each MCP session under
// /session/<id>. So the build is always root-based (base "/"): index.html
// requests /assets/*, which resolve at the control root no matter which prefix
// (/, /viewer, /session/<id>/observe) the SPA shell was served under. The router
// basename tracks import.meta.env.BASE_URL ("/") in main.tsx.
export default defineConfig({
  base: "/",
  plugins: [react()],
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
      "/api": { target: "http://localhost:8813", changeOrigin: true },
      "/data_plane": {
        target: "http://localhost:8813",
        changeOrigin: true,
        ws: true,
      },
      // Only the session *API* proxies to control — NOT the observe page at
      // /session/<id>/observe, which must fall through to vite's SPA fallback so
      // the dev bundle + HMR serve it (a bare "/session" prefix would proxy the
      // page HTML to control's built dist and break the dev module graph). A key
      // beginning with ^ is matched as a RegExp by vite.
      "^/session/[^/]+/api": {
        target: "http://localhost:8813",
        changeOrigin: true,
      },
    },
  },
});
