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
  const pendingMetadataRef = useRef<RenderMetadata | null>(null);  // Store metadata from render_start
  const pendingLoadedRegionRef = useRef<LoadedRegionInfo | null>(null);  // Store region from render_start
  const enabledRef = useRef(enabled);
  const tokenRef = useRef(token);
  const apiBaseRef = useRef(apiBase);

  // Keep refs updated
  enabledRef.current = enabled;
  tokenRef.current = token;
  apiBaseRef.current = apiBase;

  // Build WebSocket URL (stable - uses refs)
  const buildWsUrl = useCallback(() => {
    // In dev mode with vite proxy, use relative URL
    // In production, use full URL from apiBase
    const isDev = import.meta.env.DEV;
    if (isDev) {
      // Use relative URL - vite proxy will forward to backend
      const url = "/ws/render";
      if (tokenRef.current) {
        return `${url}?token=${tokenRef.current}`;
      }
      return url;
    } else {
      // Production: convert HTTP base to WS
      const httpBase = apiBaseRef.current.replace(/^http/, "ws");
      const url = `${httpBase}/ws/render`;
      if (tokenRef.current) {
        return `${url}?token=${tokenRef.current}`;
      }
      return url;
    }
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

      // Send pending request if any (small delay to ensure connection is stable)
      if (pendingRequestRef.current) {
        setTimeout(() => {
          if (wsRef.current === ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
              action: "render",
              params: pendingRequestRef.current,
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

        // Keep old URL - ImageViewer will revoke it after new image loads
        const imageUrl = URL.createObjectURL(event.data);
        imageUrlRef.current = imageUrl;

        // Get metadata and loaded_region from render_start message
        const metadata = pendingMetadataRef.current;
        const loadedRegion = pendingLoadedRegionRef.current;
        pendingMetadataRef.current = null;
        pendingLoadedRegionRef.current = null;

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
  };
}