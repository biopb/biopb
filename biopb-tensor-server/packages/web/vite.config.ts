import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
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