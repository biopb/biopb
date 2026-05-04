"use client";

import {
  Application,
  Graphics,
  Sprite,
  Texture,
} from "pixi.js";
import { Viewport } from "pixi-viewport";
import {
  buildAxisMap,
  computeScaleHint,
  type AxisMap,
} from "@biopb/tensor-flight-client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useAppStore } from "../store";

interface ImageViewerProps {
  sourceId: string;
  tensorId: string;
}

type NumericArray = Uint8Array | Uint16Array | Uint32Array | Float32Array | Float64Array | Int16Array | Int32Array;

interface LoadedRegion {
  x: number;      // world X start
  y: number;      // world Y start
  width: number;  // world width
  height: number; // world height
  scaleFactors: number[];
}

interface VisibleBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

function normalizeDtype(dtype: string): string {
  const dt = dtype.toLowerCase().replace(/\s+/g, "");
  const core = dt.replace(/^[<>=|]/, "");

  if (core === "u1" || core === "uint8") return "uint8";
  if (core === "u2" || core === "uint16") return "uint16";
  if (core === "u4" || core === "uint32") return "uint32";
  if (core === "i2" || core === "int16") return "int16";
  if (core === "i4" || core === "int32") return "int32";
  if (core === "f4" || core === "float32" || core === "float") return "float32";
  if (core === "f8" || core === "float64") return "float64";
  return core;
}

function toNumericArray(dtype: string, buffer: ArrayBuffer): NumericArray {
  const dt = normalizeDtype(dtype);
  if (dt === "uint16") return new Uint16Array(buffer);
  if (dt === "uint32") return new Uint32Array(buffer);
  if (dt === "int16") return new Int16Array(buffer);
  if (dt === "int32") return new Int32Array(buffer);
  if (dt === "float64") return new Float64Array(buffer);
  if (dt === "float32") return new Float32Array(buffer);
  return new Uint8Array(buffer);
}

function computeStrides(shape: number[]): number[] {
  const strides = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) {
    const nextStride = strides[i + 1] as number;
    const nextShape = shape[i + 1] as number;
    strides[i] = nextStride * nextShape;
  }
  return strides;
}

function computePercentileCutoffs(
  data: ArrayLike<number>,
  lo = 0.01,
  hi = 0.99,
): [number, number] {
  const n = data.length;
  if (n === 0) return [0, 1];

  const sampleSize = Math.min(n, 65536);
  const step = n / sampleSize;
  const sample = new Float32Array(sampleSize);
  for (let i = 0; i < sampleSize; i++) {
    sample[i] = Number(data[Math.floor(i * step)]);
  }
  sample.sort();

  const loVal = sample[Math.floor(sampleSize * lo)]!;
  const hiVal = sample[Math.min(sampleSize - 1, Math.ceil(sampleSize * hi))]!;
  return loVal < hiVal ? [loVal, hiVal] : [0, 1];
}

function toGrayscaleRgba(
  shape: number[],
  dimLabels: string[],
  dtype: string,
  buffer: ArrayBuffer,
): { rgba: Uint8ClampedArray<ArrayBuffer>; width: number; height: number } {
  const dt = normalizeDtype(dtype);
  const labels = dimLabels.length ? dimLabels : shape.map((_, i) => `d${i}`);
  const axisMap = buildAxisMap(labels);
  const yIdx = axisMap.y ?? Math.max(0, shape.length - 2);
  const xIdx = axisMap.x ?? Math.max(0, shape.length - 1);

  const height = shape[yIdx] ?? 1;
  const width = shape[xIdx] ?? 1;

  const data = toNumericArray(dtype, buffer);
  const strides = computeStrides(shape);

  const needsNormalization = dt !== "uint8";
  let loVal = 0;
  let hiVal = 1;

  if (needsNormalization) {
    [loVal, hiVal] = computePercentileCutoffs(data);
  }

  const rgba = new Uint8ClampedArray(width * height * 4);
  const coords = new Array<number>(shape.length).fill(0);

  let out = 0;
  for (let y = 0; y < height; y++) {
    coords[yIdx] = y;
    for (let x = 0; x < width; x++) {
      coords[xIdx] = x;
      let flat = 0;
      for (let d = 0; d < coords.length; d++) {
        flat += (coords[d] as number) * (strides[d] as number);
      }
      const raw = Number(data[flat] ?? 0);

      let gray = 0;
      if (dt === "uint8") {
        gray = Math.max(0, Math.min(255, Math.round(raw)));
      } else if (needsNormalization) {
        const n = (raw - loVal) / Math.max(1e-8, hiVal - loVal);
        gray = Math.max(0, Math.min(255, Math.round(n * 255)));
      } else {
        gray = Math.max(0, Math.min(255, Math.round(raw)));
      }

      rgba[out++] = gray;
      rgba[out++] = gray;
      rgba[out++] = gray;
      rgba[out++] = 255;
    }
  }

  return { rgba, width, height };
}

const HYSTERESIS = 0.2;

/**
 * Get visible world bounds from viewport, clipped to tensor bounds.
 */
function getVisibleWorldBounds(
  viewport: Viewport,
  fullWidth: number,
  fullHeight: number,
): VisibleBounds {
  const bounds = viewport.getVisibleBounds();
  // Clip visible region to actual tensor bounds (no margins)
  const x = Math.max(0, bounds.x);
  const y = Math.max(0, bounds.y);
  const endX = Math.min(fullWidth, bounds.x + bounds.width);
  const endY = Math.min(fullHeight, bounds.y + bounds.height);
  return {
    x,
    y,
    width: Math.max(0, endX - x),
    height: Math.max(0, endY - y),
  };
}

/**
 * Compute tensor slice range from visible bounds with 50% buffer.
 * World coordinates = original tensor indices (world is sized to full tensor).
 */
function computeSliceRange(
  visibleBounds: VisibleBounds,
  axisMap: AxisMap,
  tensorShape: number[],
): { y: [number, number]; x: [number, number] } {
  const yIdx = axisMap.y ?? 0;
  const xIdx = axisMap.x ?? 1;
  const fullHeight = tensorShape[yIdx] ?? 1;
  const fullWidth = tensorShape[xIdx] ?? 1;

  // 50% buffer margin for smooth panning (in world units)
  const marginY = visibleBounds.height * 0.5;
  const marginX = visibleBounds.width * 0.5;

  // World coords = original tensor indices, no division by scaleFactor needed
  const yStart = Math.max(0, Math.floor(visibleBounds.y - marginY));
  const yEnd = Math.min(fullHeight, Math.ceil(visibleBounds.y + visibleBounds.height + marginY));
  const xStart = Math.max(0, Math.floor(visibleBounds.x - marginX));
  const xEnd = Math.min(fullWidth, Math.ceil(visibleBounds.x + visibleBounds.width + marginX));

  return {
    y: [yStart, yEnd],
    x: [xStart, xEnd],
  };
}

/**
 * Check if reload needed: scale change OR viewport extends outside loaded region.
 */
function shouldReload(
  newScaleFactors: number[],
  visibleBounds: VisibleBounds,
  loadedRegion: LoadedRegion | null,
): boolean {
  if (!loadedRegion) return true;

  // Check scale change (hysteresis)
  for (let i = 0; i < newScaleFactors.length; i++) {
    const newVal = newScaleFactors[i] ?? 1;
    const loadedVal = loadedRegion.scaleFactors[i] ?? 1;
    if (newVal < loadedVal * (1 - HYSTERESIS) || newVal > loadedVal * (1 + HYSTERESIS)) {
      return true;
    }
  }

  // Check region coverage: reload when visible region extends outside loaded region
  const visibleEndX = visibleBounds.x + visibleBounds.width;
  const visibleEndY = visibleBounds.y + visibleBounds.height;
  const loadedEndX = loadedRegion.x + loadedRegion.width;
  const loadedEndY = loadedRegion.y + loadedRegion.height;

  // Small tolerance to avoid reload on rounding errors at clamped edges
  // 1 world unit = 1 pixel, missing 1 pixel is negligible
  const TOLERANCE = 1.0;

  // Reload if any edge of visible region is outside loaded region (beyond tolerance)
  if (visibleBounds.x < loadedRegion.x - TOLERANCE) return true;
  if (visibleBounds.y < loadedRegion.y - TOLERANCE) return true;
  if (visibleEndX > loadedEndX + TOLERANCE) return true;
  if (visibleEndY > loadedEndY + TOLERANCE) return true;

  return false;
}

export function ImageViewer({ sourceId, tensorId }: ImageViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const appRef = useRef<Application | null>(null);
  const viewportRef = useRef<Viewport | null>(null);
  const spriteRef = useRef<Sprite | null>(null);
  const textureRef = useRef<Texture | null>(null);
  const backgroundRef = useRef<Graphics | null>(null);

  const client = useAppStore((s) => s.client);
  const sources = useAppStore((s) => s.sources);
  const slice = useAppStore((s) => s.slice);

  const [appReady, setAppReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadedRegionRef = useRef<LoadedRegion | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sliceRef = useRef(slice);  // Keep current slice for event handlers

  // Update sliceRef whenever slice changes
  sliceRef.current = slice;

  // Track current source/tensor to reset when switching
  const prevSourceIdRef = useRef<string>(sourceId);
  const prevTensorIdRef = useRef<string>(tensorId);

  // Reset state when switching source/tensor
  if (prevSourceIdRef.current !== sourceId || prevTensorIdRef.current !== tensorId) {
    prevSourceIdRef.current = sourceId;
    prevTensorIdRef.current = tensorId;
    loadedRegionRef.current = null;
  }

  const descriptor = useMemo(() => {
    const src = sources.find((s) => s.source_id === sourceId);
    return src?.tensors.find((t) => t.array_id === tensorId) ?? null;
  }, [sourceId, sources, tensorId]);

  // Setup PixiJS Application and Viewport - depends on descriptor for correct world size
  useEffect(() => {
    let cancelled = false;
    const host = hostRef.current;
    if (!host || !descriptor) return;

    const axisMap = buildAxisMap(descriptor.dim_labels);
    const yIdx = axisMap.y ?? Math.max(0, descriptor.shape.length - 2);
    const xIdx = axisMap.x ?? Math.max(0, descriptor.shape.length - 1);
    const fullHeight = descriptor.shape[yIdx] ?? 1;
    const fullWidth = descriptor.shape[xIdx] ?? 1;

    (async () => {
      const app = new Application();
      await app.init({
        backgroundAlpha: 0,
        antialias: true,
        resizeTo: host,
      });
      if (cancelled) {
        app.destroy(true, { children: true, texture: true });
        return;
      }

      host.innerHTML = "";
      host.appendChild(app.canvas);

      // Create viewport with correct world dimensions from tensor descriptor
      const viewport = new Viewport({
        events: app.renderer.events,
        screenWidth: host.clientWidth,
        screenHeight: host.clientHeight,
        worldWidth: fullWidth,
        worldHeight: fullHeight,
      });
      app.stage.addChild(viewport);

      // Add background to show pending-data regions when panning
      const background = new Graphics();
      background.rect(0, 0, fullWidth, fullHeight);
      background.fill({ color: 0x1a1a2e });  // Dark blue-gray for "no data" cue
      viewport.addChild(background);
      backgroundRef.current = background;

      // Set initial fit-to-window scale
      const hostW = Math.max(1, host.clientWidth);
      const hostH = Math.max(1, host.clientHeight);
      const fitScale = Math.min(1, hostW / Math.max(1, fullWidth), hostH / Math.max(1, fullHeight));
      viewport.scale.set(fitScale);
      const offsetX = Math.round((hostW - fullWidth * fitScale) / 2);
      const offsetY = Math.round((hostH - fullHeight * fitScale) / 2);
      viewport.position.set(offsetX, offsetY);

      viewport
        .drag()
        .clamp({ direction: "all" })
        .pinch()
        .wheel();

      appRef.current = app;
      viewportRef.current = viewport;
      setAppReady(true);

      console.log("Viewport created: fitScale=%.3f, worldSize=%dx%d", fitScale, fullWidth, fullHeight);

      // Helper to queue reload with debounce
      const queueReload = (scaleFactors: number[]) => {
        if (debounceTimerRef.current) {
          clearTimeout(debounceTimerRef.current);
        }
        debounceTimerRef.current = setTimeout(() => {
          triggerTensorFetch(scaleFactors);
        }, 300);
      };

      // Listen for zoom changes
      viewport.on("zoomed-end", () => {
        if (!descriptor || !client) return;
        const currentZoom = viewport.scale.x;
        const hostW = Math.max(1, host.clientWidth);
        const hostH = Math.max(1, host.clientHeight);
        const viewportW = Math.max(1, Math.floor(hostW * window.devicePixelRatio));
        const viewportH = Math.max(1, Math.floor(hostH * window.devicePixelRatio));

        const axisMap = buildAxisMap(descriptor.dim_labels);
        const scale = computeScaleHint(
          descriptor.shape,
          axisMap,
          viewportW,
          viewportH,
          1_000_000,
          currentZoom,
        );

        const yIdx = axisMap.y ?? Math.max(0, descriptor.shape.length - 2);
        const xIdx = axisMap.x ?? Math.max(0, descriptor.shape.length - 1);
        const fullHeight = descriptor.shape[yIdx] ?? 1;
        const fullWidth = descriptor.shape[xIdx] ?? 1;

        const visibleBounds = getVisibleWorldBounds(viewport, fullWidth, fullHeight);
        if (shouldReload(scale.factors, visibleBounds, loadedRegionRef.current)) {
          queueReload(scale.factors);
        }
      });

      // Listen for pan changes
      viewport.on("moved-end", () => {
        if (!descriptor || !client) return;
        const currentZoom = viewport.scale.x;
        const hostW = Math.max(1, host.clientWidth);
        const hostH = Math.max(1, host.clientHeight);
        const viewportW = Math.max(1, Math.floor(hostW * window.devicePixelRatio));
        const viewportH = Math.max(1, Math.floor(hostH * window.devicePixelRatio));

        const axisMap = buildAxisMap(descriptor.dim_labels);
        const scale = computeScaleHint(
          descriptor.shape,
          axisMap,
          viewportW,
          viewportH,
          1_000_000,
          currentZoom,
        );

        const yIdx = axisMap.y ?? Math.max(0, descriptor.shape.length - 2);
        const xIdx = axisMap.x ?? Math.max(0, descriptor.shape.length - 1);
        const fullHeight = descriptor.shape[yIdx] ?? 1;
        const fullWidth = descriptor.shape[xIdx] ?? 1;

        const visibleBounds = getVisibleWorldBounds(viewport, fullWidth, fullHeight);
        if (shouldReload(scale.factors, visibleBounds, loadedRegionRef.current)) {
          queueReload(scale.factors);
        }
      });
    })();

    return () => {
      cancelled = true;
      setAppReady(false);
      loadedRegionRef.current = null;  // Reset for next viewport
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
        debounceTimerRef.current = null;
      }
      abortControllerRef.current?.abort();
      spriteRef.current?.destroy();
      textureRef.current?.destroy(true);
      backgroundRef.current?.destroy();
      appRef.current?.destroy(true, { children: true, texture: true });
      appRef.current = null;
      viewportRef.current = null;
      spriteRef.current = null;
      textureRef.current = null;
      backgroundRef.current = null;
    };
  }, [descriptor]);

  // Function to fetch tensor with cancellation support
  const triggerTensorFetch = (scaleFactors: number[], forceFull = false) => {
    const viewport = viewportRef.current;
    const host = hostRef.current;
    if (!viewport || !host || !client || !descriptor) return;

    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    setLoading(true);
    setError(null);

    const axisMap = buildAxisMap(descriptor.dim_labels);
    const yIdx = axisMap.y ?? Math.max(0, descriptor.shape.length - 2);
    const xIdx = axisMap.x ?? Math.max(0, descriptor.shape.length - 1);
    const fullHeight = descriptor.shape[yIdx] ?? 1;
    const fullWidth = descriptor.shape[xIdx] ?? 1;

    // On initial load (forceFull), request full tensor extent
    // On subsequent loads, request visible region + buffer
    let sliceRange: { y: [number, number]; x: [number, number] };
    if (forceFull) {
      sliceRange = {
        y: [0, fullHeight],
        x: [0, fullWidth],
      };
    } else {
      const visibleBounds = getVisibleWorldBounds(viewport, fullWidth, fullHeight);
      sliceRange = computeSliceRange(visibleBounds, axisMap, descriptor.shape);
    }

    const tensor = client.getTensor(sourceId, tensorId);

    tensor
      .compute({
        t: sliceRef.current.t,
        z: sliceRef.current.z,
        c: sliceRef.current.c,
        y: sliceRange.y,
        x: sliceRange.x,
        scaleHint: scaleFactors,
        reductionMethod: sliceRef.current.reductionMethod,
        pixelBudget: 1_000_000,
      })
      .then((arr) => {
        if (signal.aborted) return;
        const shape = arr.shape.length ? arr.shape : descriptor.shape;
        const labels = arr.dimLabels.length ? arr.dimLabels : descriptor.dim_labels;
        const dtype = arr.dtype || descriptor.dtype || "uint8";
        const { rgba, width, height } = toGrayscaleRgba(shape, labels, dtype, arr.buffer);

        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Failed to create 2D canvas context");
        ctx.putImageData(new ImageData(rgba, width, height), 0, 0);

        const tex = Texture.from(canvas);
        const sprite = new Sprite(tex);

        // Compute scale factors for sprite size (convert returned pixels to world units)
        const scaleFactorY = scaleFactors[yIdx] ?? 1;
        const scaleFactorX = scaleFactors[xIdx] ?? 1;

        // Debug: log dimensions
        const sliceHeight = sliceRange.y[1] - sliceRange.y[0];
        const sliceWidth = sliceRange.x[1] - sliceRange.x[0];
        console.log(
          "Loaded tensor: shape=%s, width=%d, height=%d, bufferLen=%d, forceFull=%s",
          shape.join(","),
          width,
          height,
          arr.buffer.byteLength,
          forceFull,
        );
        console.log(
          "Slice range: y=%s, x=%s (world), scaleFactors=%s",
          sliceRange.y.join(","),
          sliceRange.x.join(","),
          scaleFactors.join(","),
        );
        console.log(
          "Sprite world: pos=(%d, %d), size=(%d x %d), requested slice=(%d x %d)",
          sliceRange.y[0],
          sliceRange.x[0],
          height * scaleFactorY,
          width * scaleFactorX,
          sliceHeight,
          sliceWidth,
        );
        console.log("Full tensor: %d x %d", fullHeight, fullWidth);

        textureRef.current?.destroy(true);
        textureRef.current = tex;

        // Position sprite at correct world coordinates (sliceRange is in world units)
        const worldY = sliceRange.y[0];
        const worldX = sliceRange.x[0];
        sprite.position.set(worldX, worldY);
        // Sprite size: convert returned array dimensions to world units
        sprite.width = width * scaleFactorX;
        sprite.height = height * scaleFactorY;

        // Remove old sprite, keep background (background is always first child)
        if (spriteRef.current) {
          viewport.removeChild(spriteRef.current);
          spriteRef.current.destroy();
        }
        viewport.addChild(sprite);  // Adds on top of background
        spriteRef.current = sprite;

        // World size fixed to full tensor dimensions
        viewport.worldWidth = fullWidth;
        viewport.worldHeight = fullHeight;
        viewport.clamp({ direction: "all" });

        // Track loaded region
        loadedRegionRef.current = {
          x: worldX,
          y: worldY,
          width: sprite.width,
          height: sprite.height,
          scaleFactors: scaleFactors,
        };
      })
      .catch((e) => {
        if (!signal.aborted) {
          setError(e instanceof Error ? e.message : String(e));
        }
      })
      .finally(() => {
        if (!signal.aborted) {
          setLoading(false);
        }
      });
  };

  // Fetch tensor when slice/source/tensor changes (viewport already initialized)
  useEffect(() => {
    if (!appReady || !client || !descriptor) return;
    const host = hostRef.current;
    const viewport = viewportRef.current;
    if (!host || !viewport) return;

    const axisMap = buildAxisMap(descriptor.dim_labels);
    const hostW = Math.max(1, host.clientWidth);
    const hostH = Math.max(1, host.clientHeight);
    const viewportW = Math.max(1, Math.floor(hostW * window.devicePixelRatio));
    const viewportH = Math.max(1, Math.floor(hostH * window.devicePixelRatio));

    // viewport.scale.x is correct (world size was set during viewport creation)
    const currentZoom = viewport.scale.x;
    const scale = computeScaleHint(
      descriptor.shape,
      axisMap,
      viewportW,
      viewportH,
      1_000_000,
      currentZoom,
    );

    console.log("computeScaleHint: currentZoom=%.3f, scaleFactors=%s", currentZoom, scale.factors.join(","));

    // First fetch for this viewport should load full tensor
    const isInitialFetch = loadedRegionRef.current === null;
    triggerTensorFetch(scale.factors, isInitialFetch);
  }, [
    appReady,
    client,
    descriptor,
    slice.c,
    slice.reductionMethod,
    slice.t,
    slice.z,
    sourceId,
    tensorId,
  ]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div ref={hostRef} style={{ width: "100%", height: "100%" }} />

      {loading && <div className="loading-overlay">Loading slice...</div>}
      {error && !loading && <div className="error-toast">{error}</div>}
    </div>
  );
}