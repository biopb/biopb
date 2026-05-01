/**
 * ClientBootstrap — client component that fetches /api/token on mount
 * and initialises the TensorFlightClient store.
 */
"use client";

import { useEffect, useRef } from "react";
import { useAppStore } from "./store";

export function ClientBootstrap() {
  const initClient = useAppStore((s) => s.initClient);
  const loadSources = useAppStore((s) => s.loadSources);
  const devMode = useAppStore((s) => s.devMode);
  const initialised = useRef(false);

  useEffect(() => {
    if (initialised.current) return;
    initialised.current = true;

    (async () => {
      try {
        const res = await fetch("/api/token");
        const { token, apiBase, devMode: dm } = await res.json();
        initClient(apiBase ?? "http://localhost:8816", token ?? null, dm ?? false);
        await loadSources();
      } catch {
        // initClient will be null; the UI will show an error state
      }
    })();
  }, [initClient, loadSources]);

  return devMode ? (
    <div className="dev-banner">
      DEV MODE: Website token bypass is active (localhost only).
    </div>
  ) : null;
}
