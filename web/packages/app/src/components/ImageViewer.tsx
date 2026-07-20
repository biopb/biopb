"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAppStore } from "../store";
import { useRenderWebSocket, type RenderParams } from "../hooks/useRenderWebSocket";
import {
  buildAxisMap,
  computeImageCSS,
  computeInitialViewportState,
  clampViewportToBounds,
  computeVisibleBounds,
  computeScaleFactors,
  computeSliceRange,
  shouldReload,
  type AxisMap,
  type LoadedRegion,
  type ViewportState,
} from "../utils/regionUtils";
import { resolveAutoColor, type ColorValue } from "../utils/colorUtils";

interface ImageViewerProps {
  sourceId: string;
  tensorId: string;
}

// Zoom limits
const MIN_SCALE = 0.1;
const MAX_SCALE = 10.0;
const ZOOM_SENSITIVITY = 0.001;
const RELOAD_DEBOUNCE_MS = 150;

export function ImageViewer({ sourceId, tensorId }: ImageViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);

  const sources = useAppStore((s) => s.sources);
  const slice = useAppStore((s) => s.slice);
  const channelNames = useAppStore((s) => s.channelNames);
  const channelColors = useAppStore((s) => s.channelColors);
  const apiBase = useAppStore((s) => s.apiBase);
  const devMode = useAppStore((s) => s.devMode);

  // Compute percentile cutoffs from state
  const percentileLo = slice.useMinMax ? 0 : slice.percentileScale / 100;
  const percentileHi = slice.useMinMax ? 1 : 1 - slice.percentileScale / 100;

  // Get descriptor
  const descriptor = useMemo(() => {
    const src = sources.find((s) => s.source_id === sourceId);
    return src?.tensors.find((t) => t.array_id === tensorId) ?? null;
  }, [sourceId, sources, tensorId]);

  // Get current channel name for color resolution
  const currentChannelName = useMemo(() => {
    const names = channelNames[sourceId];
    return names?.[slice.c] ?? undefined;
  }, [channelNames, sourceId, slice.c]);

  // Get current color - subscribe to channelColors directly so color changes trigger re-renders
  const color: ColorValue = channelColors[sourceId]?.[slice.c] ?? "auto";

  // Get token for WebSocket auth
  const token = useMemo(() => {
    if (devMode) return null;
    return sessionStorage.getItem("biopb_token") || null;
  }, [devMode]);

  // Axis map
  const axisMap: AxisMap = useMemo(() => {
    if (!descriptor) return { t: null, z: null, c: null, y: null, x: null };
    return buildAxisMap(descriptor.dim_labels);
  }, [descriptor]);

  // Tensor dimensions
  const yIdx = axisMap.y ?? Math.max(0, (descriptor?.shape.length ?? 2) - 2);
  const xIdx = axisMap.x ?? Math.max(0, (descriptor?.shape.length ?? 2) - 1);
  const fullHeight = descriptor?.shape[yIdx] ?? 1;
  const fullWidth = descriptor?.shape[xIdx] ?? 1;

  // WebSocket hook
  const ws = useRenderWebSocket({
    apiBase,
    token,
    enabled: !!descriptor,
  });

  // Use refs for ws.requestRender to avoid dependency cycles
  const wsRef = useRef(ws);
  wsRef.current = ws;

  // Viewport state (zoom/pan) - managed locally with refs
  const [viewportState, setViewportState] = useState<ViewportState>(() => ({
    centerX: fullWidth / 2,
    centerY: fullHeight / 2,
    scale: 1,
  }));
  const viewportStateRef = useRef(viewportState);
  viewportStateRef.current = viewportState;

  // Use loadedRegion from WebSocket hook (set by backend)
  const loadedRegion = ws.loadedRegion;
  const loadedRegionRef = useRef(loadedRegion);
  loadedRegionRef.current = loadedRegion;

  // Track previous slice values to detect changes
  const prevSliceRef = useRef({
    t: slice.t,
    z: slice.z,
    c: slice.c,
    reductionMethod: slice.reductionMethod,
    useMinMax: slice.useMinMax,
    percentileScale: slice.percentileScale,
    color,
    currentChannelName,
    percentileLo,
    percentileHi,
  });

  // Debounce timer
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Drag state refs
  const isDraggingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0, centerX: 0, centerY: 0 });

  // Pressed keys for slice navigation
  const pressedKeysRef = useRef<Set<string>>(new Set());
  const sliceWheelTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Descriptor refs
  const descriptorRef = useRef(descriptor);
  descriptorRef.current = descriptor;
  const axisMapRef = useRef(axisMap);
  axisMapRef.current = axisMap;

  // Build render params (uses refs - stable)
  const buildRenderParams = useCallback(
    (
      sliceRange: { y: [number, number]; x: [number, number] },
      scaleFactors: number[],
    ): RenderParams | null => {
      const desc = descriptorRef.current;
      if (!desc) return null;

      const am = axisMapRef.current;
      const ndim = desc.shape.length;
      const currentSlice = useAppStore.getState().slice;

      const sliceStart: number[] = [];
      const sliceStop: number[] = [];

      for (let i = 0; i < ndim; i++) {
        if (i === am.y) {
          sliceStart.push(sliceRange.y[0]);
          sliceStop.push(sliceRange.y[1]);
        } else if (i === am.x) {
          sliceStart.push(sliceRange.x[0]);
          sliceStop.push(sliceRange.x[1]);
        } else if (i === am.t) {
          sliceStart.push(currentSlice.t);
          sliceStop.push(currentSlice.t + 1);
        } else if (i === am.z) {
          sliceStart.push(currentSlice.z);
          sliceStop.push(currentSlice.z + 1);
        } else if (i === am.c) {
          sliceStart.push(currentSlice.c);
          sliceStop.push(currentSlice.c + 1);
        } else {
          sliceStart.push(0);
          sliceStop.push(desc.shape[i] ?? 1);
        }
      }

      // Resolve color: if "auto", use channel name or fallback based on channel index
      const rawColor = useAppStore.getState().getChannelColor(sourceId, currentSlice.c);
      const names = useAppStore.getState().channelNames[sourceId];
      const channelName = names?.[currentSlice.c] ?? undefined;

      // Resolve "auto" to actual pseudo-color
      let resolvedColor: ColorValue;
      if (rawColor === "auto") {
        if (channelName) {
          resolvedColor = resolveAutoColor("auto", channelName);
        } else {
          // No channel name loaded yet - use default based on channel index
          // Cycle through: green, red, blue, magenta, cyan
          const defaultColors: ColorValue[] = ["green", "red", "blue", "magenta", "cyan"];
          resolvedColor = defaultColors[currentSlice.c % defaultColors.length] ?? "green";
        }
      } else {
        resolvedColor = rawColor;
      }

      const pLo = currentSlice.useMinMax ? 0 : currentSlice.percentileScale / 100;
      const pHi = currentSlice.useMinMax ? 1 : 1 - currentSlice.percentileScale / 100;

      return {
        source_id: sourceId,
        tensor_id: tensorId,
        slice_start: sliceStart,
        slice_stop: sliceStop,
        scale_hint: scaleFactors,
        reduction_method: currentSlice.reductionMethod,
        percentile_lo: pLo * 100,
        percentile_hi: pHi * 100,
        color: resolvedColor,
        channel_name: channelName,
        use_min_max: currentSlice.useMinMax,
        output_format: "jpeg",
        pixel_budget: 1_000_000,
      };
    },
    [sourceId, tensorId],
  );

  // Request render helper (uses refs - stable)
  const requestRender = useCallback(
    (
      sliceRange: { y: [number, number]; x: [number, number] },
      scaleFactors: number[],
    ) => {
      const params = buildRenderParams(sliceRange, scaleFactors);
      if (params) {
        wsRef.current.requestRender(params);
      }
    },
    [buildRenderParams],
  );

  // Trigger render if needed (with debounce) - uses refs
  const triggerRenderIfNeeded = useCallback(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      debounceTimerRef.current = null;

      const wrapper = hostRef.current;
      const desc = descriptorRef.current;
      if (!wrapper || !desc) return;

      const bounds = computeVisibleBounds(
        viewportStateRef.current,
        wrapper.clientWidth,
        wrapper.clientHeight,
        fullWidth,
        fullHeight,
      );

      const viewportW = Math.max(1, Math.floor(wrapper.clientWidth * window.devicePixelRatio));
      const viewportH = Math.max(1, Math.floor(wrapper.clientHeight * window.devicePixelRatio));
      const scaleResult = computeScaleFactors(
        desc.shape,
        axisMapRef.current,
        viewportW,
        viewportH,
        1_000_000,
        viewportStateRef.current.scale,
      );

      if (shouldReload(scaleResult.factors, bounds, loadedRegionRef.current)) {
        const sliceRange = computeSliceRange(
          bounds,
          axisMapRef.current,
          desc.shape,
          scaleResult.factors,
        );
        requestRender(sliceRange, scaleResult.factors);
      }
    }, RELOAD_DEBOUNCE_MS);
  }, [fullWidth, fullHeight, requestRender]);

  // Reset WebSocket state and transition refs when source changes
  useEffect(() => {
    wsRef.current.reset();
    prevImageUrlRef.current = null;
    prevLoadedRegionRef.current = null;
    currentImageUrlRef.current = null;
    currentLoadedRegionRef.current = null;
    // Sync prevSliceRef so the slice-change effect doesn't misfire on loadedRegion change
    const s = useAppStore.getState();
    prevSliceRef.current = {
      t: s.slice.t, z: s.slice.z, c: s.slice.c,
      reductionMethod: s.slice.reductionMethod,
      useMinMax: s.slice.useMinMax,
      percentileScale: s.slice.percentileScale,
      color: s.getChannelColor(sourceId, s.slice.c),
      currentChannelName: s.channelNames[sourceId]?.[s.slice.c] ?? undefined,
      percentileLo: s.slice.useMinMax ? 0 : s.slice.percentileScale / 100,
      percentileHi: s.slice.useMinMax ? 1 : 1 - s.slice.percentileScale / 100,
    };
  }, [sourceId, tensorId]);

  // Initialize viewport and request initial render when descriptor changes
  useEffect(() => {
    const desc = descriptorRef.current;
    if (!desc) return;

    const wrapper = hostRef.current;
    if (!wrapper) return;

    // Reset viewport to fit image
    const initial = computeInitialViewportState(
      fullWidth,
      fullHeight,
      wrapper.clientWidth,
      wrapper.clientHeight,
    );
    setViewportState(initial);
    viewportStateRef.current = initial;

    const viewportW = Math.max(1, Math.floor(wrapper.clientWidth * window.devicePixelRatio));
    const viewportH = Math.max(1, Math.floor(wrapper.clientHeight * window.devicePixelRatio));
    const scaleResult = computeScaleFactors(
      desc.shape, axisMapRef.current, viewportW, viewportH, 1_000_000, initial.scale,
    );

    requestRender({ y: [0, fullHeight], x: [0, fullWidth] }, scaleResult.factors);
  }, [descriptor, fullWidth, fullHeight, requestRender]);

  // Keep prevSliceRef.currentChannelName in sync so the slice-change effect below doesn't
  // misfire when channel names load asynchronously after the initial render. This effect
  // runs first (effects run in declaration order), so by the time the slice-change effect
  // evaluates, prev.currentChannelName already matches.
  useEffect(() => {
    prevSliceRef.current.currentChannelName = currentChannelName;
  }, [currentChannelName]);

  // Detect slice changes and request render - compare with previous values
  useEffect(() => {
    const prev = prevSliceRef.current;
    const changed =
      prev.t !== slice.t ||
      prev.z !== slice.z ||
      prev.c !== slice.c ||
      prev.reductionMethod !== slice.reductionMethod ||
      prev.useMinMax !== slice.useMinMax ||
      prev.percentileScale !== slice.percentileScale ||
      prev.color !== color ||
      prev.currentChannelName !== currentChannelName ||
      prev.percentileLo !== percentileLo ||
      prev.percentileHi !== percentileHi;

    // Only request render if values changed and conditions are met
    if (!changed || !descriptorRef.current || !wsRef.current.connected || !loadedRegionRef.current) {
      return;
    }

    const wrapper = hostRef.current;
    if (!wrapper) return;

    const bounds = computeVisibleBounds(
      viewportStateRef.current,
      wrapper.clientWidth,
      wrapper.clientHeight,
      fullWidth,
      fullHeight,
    );

    const viewportW = Math.max(1, Math.floor(wrapper.clientWidth * window.devicePixelRatio));
    const viewportH = Math.max(1, Math.floor(wrapper.clientHeight * window.devicePixelRatio));
    const scaleResult = computeScaleFactors(
      descriptorRef.current.shape,
      axisMapRef.current,
      viewportW,
      viewportH,
      1_000_000,
      viewportStateRef.current.scale,
    );

    const sliceRange = computeSliceRange(bounds, axisMapRef.current, descriptorRef.current.shape, scaleResult.factors);

    // Update previous values AFTER requesting render (so we don't lose changes if conditions weren't met)
    prevSliceRef.current = {
      t: slice.t,
      z: slice.z,
      c: slice.c,
      reductionMethod: slice.reductionMethod,
      useMinMax: slice.useMinMax,
      percentileScale: slice.percentileScale,
      color,
      currentChannelName,
      percentileLo,
      percentileHi,
    };

    requestRender(sliceRange, scaleResult.factors);
  }, [
    slice.t,
    slice.z,
    slice.c,
    slice.reductionMethod,
    slice.useMinMax,
    slice.percentileScale,
    color,
    currentChannelName,
    percentileLo,
    percentileHi,
    fullWidth,
    fullHeight,
    requestRender,
    loadedRegion,  // Re-check when loadedRegion changes (initial render complete)
  ]);

  // Mouse, key, and wheel handlers for zoom, pan, and slice navigation
  useEffect(() => {
    const desc = descriptorRef.current;
    if (!desc) return;

    const wrapper = hostRef.current;
    if (!wrapper) return;

    // Wheel handler (zoom or slice navigation)
    const handleWheel = (e: WheelEvent) => {
      const keys = pressedKeysRef.current;

      // Slice navigation mode (c/t/z + scroll)
      if (keys.has("c") || keys.has("t") || keys.has("z")) {
        e.preventDefault();
        e.stopPropagation();

        if (sliceWheelTimerRef.current) {
          clearTimeout(sliceWheelTimerRef.current);
        }

        sliceWheelTimerRef.current = setTimeout(() => {
          const delta = Math.sign(e.deltaY) * -1;
          const shape = desc.shape;
          const currentSlice = useAppStore.getState().slice;
          const am = axisMapRef.current;

          if (keys.has("t") && am.t !== null) {
            const tMax = Math.max(0, (shape[am.t] ?? 1) - 1);
            useAppStore.getState().setSlice({ t: Math.max(0, Math.min(tMax, currentSlice.t + delta)) });
          } else if (keys.has("z") && am.z !== null) {
            const zMax = Math.max(0, (shape[am.z] ?? 1) - 1);
            useAppStore.getState().setSlice({ z: Math.max(0, Math.min(zMax, currentSlice.z + delta)) });
          } else if (keys.has("c") && am.c !== null) {
            const cMax = Math.max(0, (shape[am.c] ?? 1) - 1);
            useAppStore.getState().setSlice({ c: Math.max(0, Math.min(cMax, currentSlice.c + delta)) });
          }
        }, 150);
        return;
      }

      // Zoom mode
      e.preventDefault();

      const current = viewportStateRef.current;
      const deltaScale = 1 - e.deltaY * ZOOM_SENSITIVITY;
      const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, current.scale * deltaScale));

      // Compute new center to keep zoom centered on mouse position
      const rect = wrapper.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const mouseWorldX = current.centerX + (mouseX - wrapper.clientWidth / 2) / current.scale;
      const mouseWorldY = current.centerY + (mouseY - wrapper.clientHeight / 2) / current.scale;

      const newCenterX = mouseWorldX - (mouseX - wrapper.clientWidth / 2) / newScale;
      const newCenterY = mouseWorldY - (mouseY - wrapper.clientHeight / 2) / newScale;

      const newState = clampViewportToBounds(
        { centerX: newCenterX, centerY: newCenterY, scale: newScale },
        wrapper.clientWidth,
        wrapper.clientHeight,
        fullWidth,
        fullHeight,
      );

      setViewportState(newState);
      viewportStateRef.current = newState;

      triggerRenderIfNeeded();
    };

    // Mouse handlers for drag
    const handleMouseDown = (e: MouseEvent) => {
      if (e.button !== 0) return;
      isDraggingRef.current = true;
      dragStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        centerX: viewportStateRef.current.centerX,
        centerY: viewportStateRef.current.centerY,
      };
      e.preventDefault();
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;

      const current = viewportStateRef.current;
      const start = dragStartRef.current;

      const deltaX = e.clientX - start.x;
      const deltaY = e.clientY - start.y;

      const worldDeltaX = -deltaX / current.scale;
      const worldDeltaY = -deltaY / current.scale;

      const newCenterX = start.centerX + worldDeltaX;
      const newCenterY = start.centerY + worldDeltaY;

      const newState = clampViewportToBounds(
        { centerX: newCenterX, centerY: newCenterY, scale: current.scale },
        wrapper.clientWidth,
        wrapper.clientHeight,
        fullWidth,
        fullHeight,
      );

      setViewportState(newState);
      viewportStateRef.current = newState;
    };

    const handleMouseUp = () => {
      if (!isDraggingRef.current) return;
      isDraggingRef.current = false;
      triggerRenderIfNeeded();
    };

    // Key handlers for slice navigation mode
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (key === "c" || key === "t" || key === "z") {
        pressedKeysRef.current.add(key);
        e.preventDefault();
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      pressedKeysRef.current.delete(key);
    };

    wrapper.addEventListener("wheel", handleWheel, { passive: false });
    wrapper.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    wrapper.addEventListener("keydown", handleKeyDown);
    wrapper.addEventListener("keyup", handleKeyUp);

    wrapper.setAttribute("tabindex", "0");
    (wrapper as HTMLElement).style.outline = "none";
    (wrapper as HTMLElement).focus();

    return () => {
      wrapper.removeEventListener("wheel", handleWheel);
      wrapper.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      wrapper.removeEventListener("keydown", handleKeyDown);
      wrapper.removeEventListener("keyup", handleKeyUp);

      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
      if (sliceWheelTimerRef.current) {
        clearTimeout(sliceWheelTimerRef.current);
      }
    };
  }, [descriptor, fullWidth, fullHeight, triggerRenderIfNeeded]);

  // Track previous image URL for smooth transition - revoke after new image loads
  const prevImageUrlRef = useRef<string | null>(null);
  const prevLoadedRegionRef = useRef<LoadedRegion | null>(null);
  const currentImageUrlRef = useRef<string | null>(null);
  const currentLoadedRegionRef = useRef<LoadedRegion | null>(null);

  // Capture current values as "previous" before new values render
  if (ws.imageUrl !== currentImageUrlRef.current || loadedRegion !== currentLoadedRegionRef.current) {
    prevImageUrlRef.current = currentImageUrlRef.current;
    prevLoadedRegionRef.current = currentLoadedRegionRef.current;
    currentImageUrlRef.current = ws.imageUrl;
    currentLoadedRegionRef.current = loadedRegion;
  }

  // Handle image load - revoke old URL only after new one is ready
  const handleImageLoad = useCallback(() => {
    if (prevImageUrlRef.current) {
      URL.revokeObjectURL(prevImageUrlRef.current);
      prevImageUrlRef.current = null;
      prevLoadedRegionRef.current = null;
    }
  }, []);

  // Compute image CSS for current image (NEW)
  const imageCSS = useMemo(() => {
    if (!loadedRegion || !hostRef.current) return null;
    return computeImageCSS(
      loadedRegion,
      viewportState,
      hostRef.current.clientWidth,
      hostRef.current.clientHeight,
    );
  }, [loadedRegion, viewportState]);

  // Compute CSS for previous image (OLD) - computed inline to react to ref changes
  const prevImageCSS = prevLoadedRegionRef.current && hostRef.current
    ? computeImageCSS(
        prevLoadedRegionRef.current,
        viewportState,
        hostRef.current.clientWidth,
        hostRef.current.clientHeight,
      )
    : null;

  // Compute "missing content" pattern - areas outside loaded region that need rendering
  const missingContentCSS = useMemo(() => {
    if (!hostRef.current || !loadedRegion) return null;

    const wrapper = hostRef.current;
    const wrapperW = wrapper.clientWidth;
    const wrapperH = wrapper.clientHeight;

    // Pattern tile size scales with viewport
    const baseTileWorld = 80;
    const tileScreen = baseTileWorld * viewportState.scale;

    // The pattern should cover the full image extent, but we need to position it correctly
    // Position at world origin (0, 0) - same as image would be positioned
    const patternLeft = (0 - viewportState.centerX) * viewportState.scale + wrapperW / 2;
    const patternTop = (0 - viewportState.centerY) * viewportState.scale + wrapperH / 2;
    const patternWidth = fullWidth * viewportState.scale;
    const patternHeight = fullHeight * viewportState.scale;

    return {
      left: patternLeft,
      top: patternTop,
      width: patternWidth,
      height: patternHeight,
      tileScreen,
    };
  }, [loadedRegion, viewportState, fullWidth, fullHeight]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div
        ref={hostRef}
        style={{
          width: "100%",
          height: "100%",
          overflow: "hidden",
          position: "relative",
          backgroundColor: "#1a1a2e",
          cursor: "grab",
        }}
      >
        {/* Background pattern - covers full image extent, visible in areas outside loaded region */}
        {missingContentCSS && (
          <div
            style={{
              position: "absolute",
              left: missingContentCSS.left,
              top: missingContentCSS.top,
              width: missingContentCSS.width,
              height: missingContentCSS.height,
              // Checkerboard pattern - tiles seamlessly
              backgroundImage: `linear-gradient(45deg, rgba(60, 60, 80, 0.15) 25%, transparent 25%, transparent 75%, rgba(60, 60, 80, 0.15) 75%, rgba(60, 60, 80, 0.15)),
                linear-gradient(45deg, rgba(60, 60, 80, 0.15) 25%, transparent 25%, transparent 75%, rgba(60, 60, 80, 0.15) 75%, rgba(60, 60, 80, 0.15))`,
              backgroundSize: `${missingContentCSS.tileScreen}px ${missingContentCSS.tileScreen}px`,
              backgroundPosition: `0 0, ${missingContentCSS.tileScreen * 0.5}px ${missingContentCSS.tileScreen * 0.5}px`,
              userSelect: "none",
              pointerEvents: "none",
              zIndex: 0,
            }}
          />
        )}
        {/* NEW image - behind OLD */}
        {ws.imageUrl && imageCSS && (
          <img
            src={ws.imageUrl}
            alt="Tensor slice"
            onLoad={handleImageLoad}
            style={{
              position: "absolute",
              left: imageCSS.left,
              top: imageCSS.top,
              width: imageCSS.width,
              height: imageCSS.height,
              userSelect: "none",
              pointerEvents: "none",
              zIndex: 1,
            }}
          />
        )}
        {/* OLD image - on top, only when loadedRegion changed (different position) */}
        {prevImageUrlRef.current && prevImageCSS && prevLoadedRegionRef.current &&
         (prevLoadedRegionRef.current.x !== loadedRegion?.x ||
          prevLoadedRegionRef.current.y !== loadedRegion?.y ||
          prevLoadedRegionRef.current.width !== loadedRegion?.width ||
          prevLoadedRegionRef.current.height !== loadedRegion?.height) && (
          <img
            src={prevImageUrlRef.current}
            alt="Previous slice"
            style={{
              position: "absolute",
              left: prevImageCSS.left,
              top: prevImageCSS.top,
              width: prevImageCSS.width,
              height: prevImageCSS.height,
              userSelect: "none",
              pointerEvents: "none",
              zIndex: 2,  // On top until new image onLoad fires
            }}
          />
        )}
        {ws.error && (
          <div
            style={{
              position: "absolute",
              bottom: "10px",
              left: "10px",
              color: "#ff6b6b",
              fontSize: "12px",
              background: "rgba(0,0,0,0.7)",
              padding: "8px",
              borderRadius: "4px",
              zIndex: 3,
            }}
          >
            {ws.error}
          </div>
        )}
      </div>
    </div>
  );
}
