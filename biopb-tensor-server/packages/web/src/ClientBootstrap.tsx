import { useEffect, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAppStore } from "./store";
import type { ReadyzSnapshot } from "@biopb/tensor-flight-client";

/**
 * Wait for server to be ready, with exponential backoff retry.
 * Handles cases where nginx returns HTML error pages (502/503) during startup.
 */
async function waitForServer(
  apiBase: string,
  maxRetries: number = 10,
): Promise<ReadyzSnapshot> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const res = await fetch(`${apiBase}/readyz`);

    if (res.ok) {
      try {
        return await res.json();
      } catch {
        // JSON parse error on OK response - unexpected, retry
        console.warn("Failed to parse readyz response, retrying...");
      }
    }

    // Check if response is HTML error page (nginx 502/503 during startup)
    const contentType = res.headers.get("content-type") ?? "";
    if (contentType.includes("text/html") || res.status === 502 || res.status === 503) {
      // Server is starting up, wait and retry
      const delay = 1000 * Math.pow(1.5, attempt); // exponential backoff: 1s, 1.5s, 2.25s, ...
      console.log(`Server starting (attempt ${attempt + 1}/${maxRetries}), waiting ${Math.round(delay)}ms...`);
      await new Promise((r) => setTimeout(r, delay));
      continue;
    }

    // Other HTTP error - throw with meaningful message
    throw new Error(`Server error: ${res.status} ${res.statusText}`);
  }

  throw new Error("Server failed to start within timeout. Please check container logs.");
}

export function ClientBootstrap() {
  const initClient = useAppStore((s) => s.initClient);
  const loadSources = useAppStore((s) => s.loadSources);
  const startCatalogPolling = useAppStore((s) => s.startCatalogPolling);
  const stopCatalogPolling = useAppStore((s) => s.stopCatalogPolling);
  const initialised = useRef(false);
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  useEffect(() => {
    if (initialised.current) return;
    initialised.current = true;

    // Check for token in URL first (bypass unlock page)
    const urlToken = searchParams.get("token");
    if (urlToken) {
      sessionStorage.setItem("biopb_token", urlToken.trim());
      // Clean URL to remove token parameter (prevent leaking in history)
      searchParams.delete("token");
      setSearchParams(searchParams, { replace: true });
    }

    const apiBase =
      import.meta.env.VITE_TENSOR_API ?? "http://localhost:8814";

    (async () => {
      try {
        // Wait for server to be ready (handles startup gracefully)
        const status = await waitForServer(apiBase);

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
  }, [initClient, loadSources, startCatalogPolling, navigate, searchParams, setSearchParams]);

  // Stop polling on unmount
  useEffect(() => {
    return () => {
      stopCatalogPolling();
    };
  }, [stopCatalogPolling]);

  return null;
}
