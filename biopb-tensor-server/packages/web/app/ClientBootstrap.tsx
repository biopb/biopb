/**
 * ClientBootstrap — client component that reads the token from sessionStorage
 * and initialises the TensorFlightClient store.
 */
"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "./store";

export function ClientBootstrap() {
  const initClient = useAppStore((s) => s.initClient);
  const loadSources = useAppStore((s) => s.loadSources);
  const devMode = useAppStore((s) => s.devMode);
  const initialised = useRef(false);
  const router = useRouter();

  useEffect(() => {
    if (initialised.current) return;
    initialised.current = true;

    const token = sessionStorage.getItem("biopb_token") ?? "";
    if (!token) {
      router.push("/unlock");
      return;
    }

    const apiBase =
      process.env.NEXT_PUBLIC_TENSOR_API ?? "http://localhost:8816";

    (async () => {
      try {
        initClient(apiBase, token, false);
        await loadSources();
      } catch (err) {
        useAppStore.setState({
          connectionState: "error",
          connectionError:
            err instanceof Error ? err.message : "Failed to connect to server",
        });
      }
    })();
  }, [initClient, loadSources, router]);

  return devMode ? (
    <div className="dev-banner">
      DEV MODE: Website token bypass is active (localhost only).
    </div>
  ) : null;
}
