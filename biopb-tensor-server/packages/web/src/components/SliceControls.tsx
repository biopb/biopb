"use client";

import { buildAxisMap } from "@biopb/tensor-flight-client";
import { useMemo } from "react";
import { useAppStore } from "../store";

interface SliceControlsProps {
  sourceId: string;
  tensorId: string;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function SliceControls({ sourceId, tensorId }: SliceControlsProps) {
  const sources = useAppStore((s) => s.sources);
  const slice = useAppStore((s) => s.slice);
  const setSlice = useAppStore((s) => s.setSlice);

  const descriptor = useMemo(() => {
    const src = sources.find((s) => s.source_id === sourceId);
    return src?.tensors.find((t) => t.array_id === tensorId) ?? null;
  }, [sourceId, sources, tensorId]);

  const axisMap = useMemo(() => {
    if (!descriptor) return { t: null, z: null, c: null, y: null, x: null };
    return buildAxisMap(descriptor.dim_labels);
  }, [descriptor]);

  if (!descriptor) {
    return <section className="slice-controls">Tensor metadata unavailable</section>;
  }

  const shape = descriptor.shape;
  const tSize = axisMap.t !== null ? shape[axisMap.t] ?? 1 : 1;
  const zSize = axisMap.z !== null ? shape[axisMap.z] ?? 1 : 1;
  const cSize = axisMap.c !== null ? shape[axisMap.c] ?? 1 : 1;
  const tMax = axisMap.t !== null ? Math.max(0, tSize - 1) : 0;
  const zMax = axisMap.z !== null ? Math.max(0, zSize - 1) : 0;
  const cMax = axisMap.c !== null ? Math.max(0, cSize - 1) : 0;

  return (
    <section className="slice-controls">
      <div className="slice-grid" style={{ display: "grid", gap: 8 }}>
        {axisMap.t !== null && tSize > 1 && (
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 20 }}>T</span>
            <input
              type="range"
              min={0}
              max={tMax}
              value={clamp(slice.t, 0, tMax)}
              onChange={(e) => setSlice({ t: Number(e.target.value) })}
              style={{ flex: 1 }}
            />
            <span style={{ width: 40, textAlign: "right", fontSize: 11 }}>{slice.t}/{tMax}</span>
          </label>
        )}

        {axisMap.z !== null && zSize > 1 && (
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 20 }}>Z</span>
            <input
              type="range"
              min={0}
              max={zMax}
              value={clamp(slice.z, 0, zMax)}
              onChange={(e) => setSlice({ z: Number(e.target.value) })}
              style={{ flex: 1 }}
            />
            <span style={{ width: 40, textAlign: "right", fontSize: 11 }}>{slice.z}/{zMax}</span>
          </label>
        )}

        {axisMap.c !== null && cSize > 1 && (
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 20 }}>C</span>
            <input
              type="range"
              min={0}
              max={cMax}
              value={clamp(slice.c, 0, cMax)}
              onChange={(e) => setSlice({ c: Number(e.target.value) })}
              style={{ flex: 1 }}
            />
            <span style={{ width: 40, textAlign: "right", fontSize: 11 }}>{slice.c}/{cMax}</span>
          </label>
        )}

        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ minWidth: 60 }}>Reduction</span>
          <select
            value={slice.reductionMethod}
            onChange={(e) => setSlice({ reductionMethod: e.target.value })}
            style={{ flex: 1 }}
          >
            <option value="nearest">nearest</option>
            <option value="linear">linear</option>
            <option value="area">area</option>
            <option value="mean">mean</option>
          </select>
        </label>
      </div>
    </section>
  );
}
