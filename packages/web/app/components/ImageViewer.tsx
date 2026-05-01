"use client";

import {
  Application,
  Container,
  Sprite,
  Texture,
} from "pixi.js";
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

function toGrayscaleRgba(
  shape: number[],
  dimLabels: string[],
  dtype: string,
  buffer: ArrayBuffer,
): { rgba: Uint8ClampedArray; width: number; height: number } {
  const dt = normalizeDtype(dtype);
  const labels = dimLabels.length ? dimLabels : shape.map((_, i) => `d${i}`);
  const axisMap = buildAxisMap(labels);
  const yIdx = axisMap.y ?? Math.max(0, shape.length - 2);
  const xIdx = axisMap.x ?? Math.max(0, shape.length - 1);

  const height = shape[yIdx] ?? 1;
  const width = shape[xIdx] ?? 1;

  const data = toNumericArray(dtype, buffer);
  const strides = computeStrides(shape);

  let minVal = Number.POSITIVE_INFINITY;
  let maxVal = Number.NEGATIVE_INFINITY;
  const needsNormalization = dt !== "uint8";

  if (needsNormalization) {
    for (let i = 0; i < data.length; i++) {
      const v = Number(data[i]);
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal) || minVal === maxVal) {
      minVal = 0;
      maxVal = 1;
    }
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
        const n = (raw - minVal) / Math.max(1e-8, maxVal - minVal);
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

export function ImageViewer({ sourceId, tensorId }: ImageViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const appRef = useRef<Application | null>(null);
  const viewportRef = useRef<Container | null>(null);
  const spriteRef = useRef<Sprite | null>(null);
  const textureRef = useRef<Texture | null>(null);
  const interactionCleanupRef = useRef<(() => void) | null>(null);

  const client = useAppStore((s) => s.client);
  const sources = useAppStore((s) => s.sources);
  const slice = useAppStore((s) => s.slice);
  const setSlice = useAppStore((s) => s.setSlice);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const descriptor = useMemo(() => {
    const src = sources.find((s) => s.source_id === sourceId);
    return src?.tensors.find((t) => t.array_id === tensorId) ?? null;
  }, [sourceId, sources, tensorId]);

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

      const viewport = new Container();
      app.stage.addChild(viewport);

      appRef.current = app;
      viewportRef.current = viewport;

      let dragging = false;
      let lastX = 0;
      let lastY = 0;

      const onPointerDown = (ev: PointerEvent) => {
        dragging = true;
        lastX = ev.clientX;
        lastY = ev.clientY;
      };
      const onPointerMove = (ev: PointerEvent) => {
        if (!dragging) return;
        const dx = ev.clientX - lastX;
        const dy = ev.clientY - lastY;
        lastX = ev.clientX;
        lastY = ev.clientY;
        viewport.position.x += dx;
        viewport.position.y += dy;
      };
      const onPointerUp = () => {
        dragging = false;
      };
      const onWheel = (ev: WheelEvent) => {
        ev.preventDefault();
        const current = viewport.scale.x || 1;
        const next = ev.deltaY < 0 ? current * 1.1 : current * 0.9;
        const clamped = Math.max(0.05, Math.min(40, next));
        viewport.scale.set(clamped);
      };

      app.canvas.addEventListener("pointerdown", onPointerDown);
      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerUp);
      app.canvas.addEventListener("wheel", onWheel, { passive: false });

      interactionCleanupRef.current = () => {
        app.canvas.removeEventListener("pointerdown", onPointerDown);
        window.removeEventListener("pointermove", onPointerMove);
        window.removeEventListener("pointerup", onPointerUp);
        app.canvas.removeEventListener("wheel", onWheel);
      };
    })();

    return () => {
      cancelled = true;
      interactionCleanupRef.current?.();
      interactionCleanupRef.current = null;
      spriteRef.current?.destroy();
      textureRef.current?.destroy(true);
      appRef.current?.destroy(true, { children: true, texture: true });
      appRef.current = null;
      viewportRef.current = null;
      spriteRef.current = null;
      textureRef.current = null;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    if (!client || !descriptor) return;

    const host = hostRef.current;
    if (!host) return;

    const viewportW = Math.max(1, Math.floor(host.clientWidth * window.devicePixelRatio));
    const viewportH = Math.max(1, Math.floor(host.clientHeight * window.devicePixelRatio));

    const axisMap = buildAxisMap(descriptor.dim_labels);
    const scale = computeScaleHint(
      descriptor.shape,
      axisMap,
      viewportW,
      viewportH,
      1_000_000,
      slice.scaleFactors.length ? slice.scaleFactors : undefined,
    );

    const prev = slice.scaleFactors;
    const sameScale = prev.length === scale.factors.length && prev.every((v, i) => v === scale.factors[i]);
    if (!sameScale) setSlice({ scaleFactors: scale.factors });

    const tensor = client.getTensor(sourceId, tensorId);
    setLoading(true);
    setError(null);

    tensor
      .compute({
        t: slice.t,
        z: slice.z,
        c: slice.c,
        scaleHint: scale.factors,
        reductionMethod: slice.reductionMethod,
        pixelBudget: 1_000_000,
      })
      .then((arr) => {
        if (cancelled) return;
        const shape = arr.shape.length ? arr.shape : descriptor.shape;
        const labels = arr.dimLabels.length ? arr.dimLabels : descriptor.dim_labels;
        const dtype = arr.dtype || descriptor.dtype || "uint8";
        const { rgba, width, height } = toGrayscaleRgba(shape, labels, dtype, arr.buffer);

        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Failed to create 2D canvas context");
        const rgbaArray: Uint8ClampedArray<ArrayBuffer> = new Uint8ClampedArray(
          new ArrayBuffer(rgba.length),
        );
        rgbaArray.set(rgba);
        ctx.putImageData(new ImageData(rgbaArray, width, height), 0, 0);

        const tex = Texture.from(canvas);
        const sprite = new Sprite(tex);

        textureRef.current?.destroy(true);
        textureRef.current = tex;

        const viewport = viewportRef.current;
        if (!viewport) return;

        viewport.removeChildren();
        viewport.addChild(sprite);
        spriteRef.current = sprite;

        const hostW = Math.max(1, host.clientWidth);
        const hostH = Math.max(1, host.clientHeight);
        const fitScale = Math.min(1, hostW / Math.max(1, width), hostH / Math.max(1, height));

        viewport.scale.set(fitScale);
        const offsetX = Math.round((hostW - width * fitScale) / 2);
        const offsetY = Math.round((hostH - height * fitScale) / 2);
        viewport.position.set(
          offsetX,
          offsetY,
        );
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [
    client,
    descriptor,
    setSlice,
    slice.c,
    slice.reductionMethod,
    slice.scaleFactors,
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
