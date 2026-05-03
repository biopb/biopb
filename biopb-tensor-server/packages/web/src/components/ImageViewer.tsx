"use client";

import {
  Application,
  Sprite,
  Texture,
} from "pixi.js";
import { Viewport } from "pixi-viewport";
import {
  buildAxisMap,
  computeScaleHint,
} from "@biopb/tensor-flight-client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useAppStore } from "../store";

interface ImageViewerProps {
  sourceId: string;
  tensorId: string;
}

type NumericArray = Uint8Array | Uint16Array | Uint32Array | Float32Array | Float64Array | Int16Array | Int32Array;

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

/**
 * Compute lo/hi contrast cutoffs using 1%–99% percentile normalization.
 * Systematic sampling keeps cost O(1) for large tiles (≤65536 samples).
 */
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
  sample.sort(); // numeric sort on TypedArray — no comparator needed

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

const HYSTERESIS = 0.2; // 20% threshold

/**
 * Check if scale factors differ enough to warrant a reload.
 * Hysteresis prevents rapid switching when zooming small amounts.
 */
function shouldReload(
  newFactors: number[],
  loadedFactors: number[],
): boolean {
  if (loadedFactors.length === 0) return true; // First load
  for (let i = 0; i < newFactors.length; i++) {
    const newVal = newFactors[i] ?? 1;
    const loadedVal = loadedFactors[i] ?? 1;
    // Only reload if scale changed by more than 20%
    if (newVal < loadedVal * (1 - HYSTERESIS) || newVal > loadedVal * (1 + HYSTERESIS)) {
      return true;
    }
  }
  return false;
}

export function ImageViewer({ sourceId, tensorId }: ImageViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const appRef = useRef<Application | null>(null);
  const viewportRef = useRef<Viewport | null>(null);
  const spriteRef = useRef<Sprite | null>(null);
  const textureRef = useRef<Texture | null>(null);

  const client = useAppStore((s) => s.client);
  const sources = useAppStore((s) => s.sources);
  const slice = useAppStore((s) => s.slice);

  const [appReady, setAppReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadedScaleFactorsRef = useRef<number[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);
  const isFirstLoadRef = useRef(true);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const queuedScaleFactorsRef = useRef<number[] | null>(null);

  // Track current source/tensor to reset pan-zoom when switching
  const prevSourceIdRef = useRef<string>(sourceId);
  const prevTensorIdRef = useRef<string>(tensorId);

  // Reset first load flag when switching to a different source or tensor
  if (prevSourceIdRef.current !== sourceId || prevTensorIdRef.current !== tensorId) {
    prevSourceIdRef.current = sourceId;
    prevTensorIdRef.current = tensorId;
    isFirstLoadRef.current = true;
    loadedScaleFactorsRef.current = [];
  }

  const descriptor = useMemo(() => {
    const src = sources.find((s) => s.source_id === sourceId);
    return src?.tensors.find((t) => t.array_id === tensorId) ?? null;
  }, [sourceId, sources, tensorId]);

  // Setup PixiJS Application and Viewport
  useEffect(() => {
    let cancelled = false;
    const host = hostRef.current;
    if (!host) return;

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

      // Initial viewport - world size will be updated when tensor is loaded
      const viewport = new Viewport({
        events: app.renderer.events,
        screenWidth: host.clientWidth,
        screenHeight: host.clientHeight,
        worldWidth: 1000,
        worldHeight: 1000,
      });
      app.stage.addChild(viewport);

      viewport
        .drag()
        .decelerate()
        .clamp({ direction: "all" })
        .pinch()
        .wheel();

      appRef.current = app;
      viewportRef.current = viewport;
      setAppReady(true);

      // Listen for zoom changes to trigger reload
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

        if (shouldReload(scale.factors, loadedScaleFactorsRef.current)) {
          // Queue the request - store scale factors
          queuedScaleFactorsRef.current = scale.factors;

          // Cancel previous debounce timer
          if (debounceTimerRef.current) {
            clearTimeout(debounceTimerRef.current);
          }

          // Start new debounce timer (300ms)
          debounceTimerRef.current = setTimeout(() => {
            if (queuedScaleFactorsRef.current) {
              triggerTensorFetch(queuedScaleFactorsRef.current);
              queuedScaleFactorsRef.current = null;
            }
          }, 300);
        }
      });
    })();

    return () => {
      cancelled = true;
      setAppReady(false);
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
        debounceTimerRef.current = null;
      }
      abortControllerRef.current?.abort();
      spriteRef.current?.destroy();
      textureRef.current?.destroy(true);
      appRef.current?.destroy(true, { children: true, texture: true });
      appRef.current = null;
      viewportRef.current = null;
      spriteRef.current = null;
      textureRef.current = null;
    };
  }, []);

  // Function to fetch tensor with cancellation support
  const triggerTensorFetch = (scaleFactors: number[]) => {
    const viewport = viewportRef.current;
    const host = hostRef.current;
    if (!viewport || !host || !client || !descriptor) return;

    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    setLoading(true);
    setError(null);

    // Get full tensor dimensions from descriptor
    const axisMap = buildAxisMap(descriptor.dim_labels);
    const yIdx = axisMap.y ?? Math.max(0, descriptor.shape.length - 2);
    const xIdx = axisMap.x ?? Math.max(0, descriptor.shape.length - 1);
    const fullHeight = descriptor.shape[yIdx] ?? 1;
    const fullWidth = descriptor.shape[xIdx] ?? 1;

    const tensor = client.getTensor(sourceId, tensorId);

    tensor
      .compute({
        t: slice.t,
        z: slice.z,
        c: slice.c,
        scaleHint: scaleFactors,
        reductionMethod: slice.reductionMethod,
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

        textureRef.current?.destroy(true);
        textureRef.current = tex;

        // Scale sprite to occupy full tensor extent in world coordinates
        // (returned data is at scaleFactor, so sprite dimensions = returned_size * scaleFactor)
        const scaleFactorY = scaleFactors[yIdx] ?? 1;
        const scaleFactorX = scaleFactors[xIdx] ?? 1;
        sprite.width = width * scaleFactorX;
        sprite.height = height * scaleFactorY;

        viewport.removeChildren();
        viewport.addChild(sprite);
        spriteRef.current = sprite;

        // World size is FIXED to full tensor dimensions (independent of scale factors)
        viewport.worldWidth = fullWidth;
        viewport.worldHeight = fullHeight;
        viewport.clamp({ direction: "all" });

        loadedScaleFactorsRef.current = scaleFactors;

        const hostW = Math.max(1, host.clientWidth);
        const hostH = Math.max(1, host.clientHeight);

        if (isFirstLoadRef.current) {
          // Fit full tensor to screen
          const fitScale = Math.min(1, hostW / Math.max(1, fullWidth), hostH / Math.max(1, fullHeight));
          viewport.scale.set(fitScale);
          const offsetX = Math.round((hostW - fullWidth * fitScale) / 2);
          const offsetY = Math.round((hostH - fullHeight * fitScale) / 2);
          viewport.position.set(offsetX, offsetY);
          isFirstLoadRef.current = false;
        }
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

  // Fetch tensor when slice/source/tensor changes
  useEffect(() => {
    if (!appReady || !client || !descriptor) return;

    const host = hostRef.current;
    const viewport = viewportRef.current;
    if (!host || !viewport) return;

    const viewportW = Math.max(1, Math.floor(host.clientWidth * window.devicePixelRatio));
    const viewportH = Math.max(1, Math.floor(host.clientHeight * window.devicePixelRatio));

    const axisMap = buildAxisMap(descriptor.dim_labels);
    const currentZoom = viewport.scale.x;
    const scale = computeScaleHint(
      descriptor.shape,
      axisMap,
      viewportW,
      viewportH,
      1_000_000,
      currentZoom,
    );

    triggerTensorFetch(scale.factors);
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