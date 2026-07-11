import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // Base public path. Empty/"/" for a dev server or a root-served bundle; the
  // control front serves the dataviewer under a namespace, so its release build
  // sets VITE_BASE_PATH="/data_plane/viewer/" (see the control's _control.py and
  // the migration doc §6.1). import.meta.env.BASE_URL then drives both the asset
  // URLs and the React Router basename (see main.tsx).
  base: process.env.VITE_BASE_PATH || "/",
  plugins: [react()],
  resolve: {
    dedupe: ["react", "react-dom"],
  },
  server: {
    proxy: {
      // Proxy WebSocket connections to the backend
      "/ws": {
        target: "http://localhost:8816",
        ws: true,
        changeOrigin: true,
      },
      // Proxy API requests to the backend
      "/api": {
        target: "http://localhost:8816",
        changeOrigin: true,
      },
      // Proxy health endpoints
      "/livez": {
        target: "http://localhost:8816",
        changeOrigin: true,
      },
      "/readyz": {
        target: "http://localhost:8816",
        changeOrigin: true,
      },
      "/healthz": {
        target: "http://localhost:8816",
        changeOrigin: true,
      },
    },
  },
});
