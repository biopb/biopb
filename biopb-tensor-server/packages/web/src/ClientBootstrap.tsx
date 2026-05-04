import { useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useAppStore } from "./store";

export function ClientBootstrap() {
  const initClient = useAppStore((s) => s.initClient);
  const loadSources = useAppStore((s) => s.loadSources);
  const startCatalogPolling = useAppStore((s) => s.startCatalogPolling);
  const stopCatalogPolling = useAppStore((s) => s.stopCatalogPolling);
  const initialised = useRef(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (initialised.current) return;
    initialised.current = true;

    const apiBase =
      import.meta.env.VITE_TENSOR_API ?? "http://localhost:8816";

    (async () => {
      try {
        // Check server status first (unauthenticated endpoint)
        const readyzRes = await fetch(`${apiBase}/readyz`);
        const status = await readyzRes.json();

        if (status.dev_mode) {
          // Dev mode active - bypass token requirement
          initClient(apiBase, null, true);
          await loadSources();
          startCatalogPolling();
          return;
        }

        // Production mode - require token
        const token = sessionStorage.getItem("biopb_token") ?? "";
        if (!token) {
          navigate("/unlock");
          return;
        }

        initClient(apiBase, token, false);
        await loadSources();
        startCatalogPolling();
      } catch (err) {
        useAppStore.setState({
          connectionState: "error",
          connectionError:
            err instanceof Error ? err.message : "Failed to connect to server",
        });
      }
    })();
  }, [initClient, loadSources, startCatalogPolling, navigate]);

  // Stop polling on unmount
  useEffect(() => {
    return () => {
      stopCatalogPolling();
    };
  }, [stopCatalogPolling]);

  return null;
}
