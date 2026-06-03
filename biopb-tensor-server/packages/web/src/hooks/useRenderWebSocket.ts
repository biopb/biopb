/**
 * WebSocket hook for image rendering.
 *
 * Manages WebSocket connection to /ws/render endpoint.
 * Sends render requests and receives binary image data.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export interface RenderParams {
  source_id: string;
  tensor_id: string;
  slice_start: number[];
  slice_stop: number[];
  scale_hint: number[];
  reduction_method?: string;
  percentile_lo: number;
  percentile_hi: number;
  color: string;
  channel_name?: string;
  use_min_max: boolean;
  output_format: string;
  pixel_budget?: number;
}

export interface RenderMetadata {
  width: number;
  height: number;
  format: string;
  percentile_lo_value: number;
  percentile_hi_value: number;
}

export interface LoadedRegionInfo {
  x: number;      // world X start
  y: number;      // world Y start
  width: number;  // world width
  height: number; // world height
  scaleFactors: number[];
}

export interface WebSocketState {
  imageUrl: string | null;
  metadata: RenderMetadata | null;
  loadedRegion: LoadedRegionInfo | null;  // Region info for current image
  loading: boolean;
  error: string | null;
  connected: boolean;
}

export interface UseRenderWebSocketOptions {
  apiBase: string;
  token: string | null;
  enabled?: boolean;
}

export interface UseRenderWebSocketResult extends WebSocketState {
  requestRender: (params: RenderParams) => void;
  reconnect: () => void;
  reset: () => void;
}

/**
 * Hook for WebSocket-based image rendering.
 */
export function useRenderWebSocket(options: UseRenderWebSocketOptions): UseRenderWebSocketResult {
  const { apiBase, token, enabled = true } = options;

  const [state, setState] = useState<WebSocketState>({
    imageUrl: null,
    metadata: null,
    loadedRegion: null,
    loading: false,
    error: null,
    connected: false,
  });

  // Use refs to avoid re-creating callbacks and causing reconnection loops
  const wsRef = useRef<WebSocket | null>(null);
  const imageUrlRef = useRef<string | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingRequestRef = useRef<RenderParams | null>(null);
  const lastSentParamsRef = useRef<RenderParams | null>(null);  // Last request sent (for reconnect retry)
  const pendingMetadataRef = useRef<RenderMetadata | null>(null);  // Store metadata from render_start
  const pendingLoadedRegionRef = useRef<LoadedRegionInfo | null>(null);  // Store region from render_start
  const renderGenerationRef = useRef(0);  // Incremented on each render request, used to discard stale responses
  const pendingRenderGenerationRef = useRef<number | null>(null);  // Generation at time of render_start
  const currentSourceTensorRef = useRef<{ source_id: string; tensor_id: string } | null>(null);  // Current source/tensor being displayed
  const enabledRef = useRef(enabled);
  const tokenRef = useRef(token);
  const apiBaseRef = useRef(apiBase);

  // Keep refs updated
  enabledRef.current = enabled;
  tokenRef.current = token;
  apiBaseRef.current = apiBase;

  // Build WebSocket URL (stable - uses refs)
  const buildWsUrl = useCallback(() => {
    // The socket needs an absolute ws/wss URL. Older Safari does NOT coerce a
    // relative or http(s) URL to ws in the WebSocket constructor — it throws
    // "The string did not match the expected pattern" — so build the scheme
    // ourselves instead of passing a relative path. Base origin:
    //  - dev: the page origin, so the vite proxy forwards /ws to the backend.
    //  - prod: the configured apiBase, or the page origin when the webapp is
    //    served same-origin by FastAPI (built with VITE_TENSOR_API="", which
    //    left apiBase empty and produced a relative URL — the bug this fixes).
    const base = import.meta.env.DEV
      ? window.location.origin
      : apiBaseRef.current || window.location.origin;
    const wsBase = base.replace(/^http/, "ws"); // http->ws, https->wss
    const url = `${wsBase}/ws/render`;
    return tokenRef.current ? `${url}?token=${tokenRef.current}` : url;
  }, []); // No dependencies - uses refs

  // Clean up blob URL
  const cleanupUrl = useCallback(() => {
    if (imageUrlRef.current) {
      URL.revokeObjectURL(imageUrlRef.current);
      imageUrlRef.current = null;
    }
  }, []);

  // Connect WebSocket (stable - uses refs)
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return; // Already connected or connecting
    }

    cleanupUrl();
    setState((s) => ({ ...s, error: null, connected: false, loading: false, imageUrl: null }));

    const url = buildWsUrl();

    const ws = new WebSocket(url);

    ws.onopen = () => {
      wsRef.current = ws;
      setState((s) => ({ ...s, connected: true, error: null }));

      // Send pending request if any, or retry last request if no image loaded yet
      const toSend = pendingRequestRef.current ??
        (imageUrlRef.current === null ? lastSentParamsRef.current : null);
      if (toSend) {
        setTimeout(() => {
          if (wsRef.current === ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
              action: "render",
              params: toSend,
            }));
            pendingRequestRef.current = null;
          }
        }, 50);
      }
    };

    ws.onclose = (event) => {
      wsRef.current = null;
      setState((s) => ({ ...s, connected: false, loading: false }));

      // Auth failure (4001)
      if (event.code === 4001) {
        setState((s) => ({ ...s, error: "Authentication failed" }));
      }

      // Auto reconnect on unexpected close (not manual close or auth failure)
      if (event.code !== 1000 && event.code !== 4001 && enabledRef.current) {
        reconnectTimerRef.current = setTimeout(() => {
          if (enabledRef.current) {
            connect();
          }
        }, 2000);
      }
    };

    ws.onerror = () => {
      setState((s) => ({ ...s, error: "WebSocket connection failed", connected: false }));
    };

    ws.onmessage = (event) => {
      // Handle JSON metadata message
      if (typeof event.data === "string") {
        try {
          const msg = JSON.parse(event.data);

          if (msg.action === "render_start") {
            // Store the current render generation - will be checked when binary blob arrives
            pendingRenderGenerationRef.current = renderGenerationRef.current;
            // Store metadata and loaded_region in refs - will be set with binary blob
            pendingMetadataRef.current = {
              width: msg.width,
              height: msg.height,
              format: msg.format,
              percentile_lo_value: msg.percentile_lo_value,
              percentile_hi_value: msg.percentile_hi_value,
            };
            if (msg.loaded_region) {
              pendingLoadedRegionRef.current = {
                x: msg.loaded_region.x,
                y: msg.loaded_region.y,
                width: msg.loaded_region.width,
                height: msg.loaded_region.height,
                scaleFactors: msg.loaded_region.scale_factors,
              };
            }
            // No setState here - wait for binary blob to update all state together
          } else if (msg.action === "error") {
            setState((s) => ({ ...s, error: msg.message, loading: false }));
          }
        } catch (e) {
          setState((s) => ({ ...s, error: `Invalid JSON: ${e}`, loading: false }));
        }
      } else if (event.data instanceof Blob) {
        // Handle binary image data
        // Check if this blob belongs to the current render generation
        const blobGeneration = pendingRenderGenerationRef.current;
        const currentGeneration = renderGenerationRef.current;

        // Discard stale responses (blob from an older render request)
        if (blobGeneration !== null && blobGeneration !== currentGeneration) {
          // Stale response - discard and don't update state
          pendingMetadataRef.current = null;
          pendingLoadedRegionRef.current = null;
          pendingRenderGenerationRef.current = null;
          return;
        }

        // Keep old URL - ImageViewer will revoke it after new image loads
        const imageUrl = URL.createObjectURL(event.data);
        imageUrlRef.current = imageUrl;

        // Get metadata and loaded_region from render_start message
        const metadata = pendingMetadataRef.current;
        const loadedRegion = pendingLoadedRegionRef.current;
        pendingMetadataRef.current = null;
        pendingLoadedRegionRef.current = null;
        pendingRenderGenerationRef.current = null;

        // Single setState with all info from this render cycle
        setState((s) => ({
          ...s,
          imageUrl,
          metadata,
          loadedRegion,
          loading: false,
          error: null,
        }));
      }
    };

    wsRef.current = ws;
  }, [buildWsUrl, cleanupUrl]);

  // Request render action (stable)
  const requestRender = useCallback((params: RenderParams) => {
    lastSentParamsRef.current = params;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        action: "render",
        params,
      }));
    } else if (ws && ws.readyState === WebSocket.CONNECTING) {
      pendingRequestRef.current = params;
    } else {
      pendingRequestRef.current = params;
      connect();
    }
  }, [connect]);

  // Reconnect action
  const reconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close(1000, "Manual reconnect");
      wsRef.current = null;
    }
    connect();
  }, [connect]);

  // Reset state (clear image and loadedRegion) without disconnecting
  const reset = useCallback(() => {
    cleanupUrl();
    pendingMetadataRef.current = null;
    pendingLoadedRegionRef.current = null;
    pendingRenderGenerationRef.current = null;
    lastSentParamsRef.current = null;
    setState((s) => ({ ...s, imageUrl: null, loadedRegion: null, metadata: null }));
  }, [cleanupUrl]);

  // Connect on mount when enabled, disconnect on unmount
  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      cleanupUrl();
      if (wsRef.current) {
        wsRef.current.close(1000, "Component unmount");
        wsRef.current = null;
      }
    };
  }, [enabled]); // Only depend on enabled - connect is stable

  return {
    ...state,
    requestRender,
    reconnect,
    reset,
  };
}