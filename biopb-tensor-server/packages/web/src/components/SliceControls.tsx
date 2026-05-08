"use client";

import { buildAxisMap } from "@biopb/tensor-flight-client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useAppStore } from "../store";
import {
  PRESET_COLORS,
  type ColorValue,
  colorToHex,
  isHexColor,
  resolveAutoColor,
} from "../utils/colorUtils";

// Debounce delay for slider updates (matches ImageViewer's keyboard+wheel debounce)
const SLIDER_DEBOUNCE_MS = 150;

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
  const channelNames = useAppStore((s) => s.channelNames);
  const channelColors = useAppStore((s) => s.channelColors);
  const getChannelColor = useAppStore((s) => s.getChannelColor);
  const setChannelColor = useAppStore((s) => s.setChannelColor);
  const loadChannelNames = useAppStore((s) => s.loadChannelNames);
  const showAdvancedOptions = useAppStore((s) => s.showAdvancedOptions);

  // Track custom color picker state (separate from preset dropdown)
  const [useCustomColor, setUseCustomColor] = useState(false);

  // Debounce timer ref for slider updates
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Local state for slider values (for immediate visual feedback)
  const [localT, setLocalT] = useState(slice.t);
  const [localZ, setLocalZ] = useState(slice.z);
  const [localC, setLocalC] = useState(slice.c);
  const [localPercentile, setLocalPercentile] = useState(slice.percentileScale);

  // Sync local state when store slice changes (e.g., from keyboard navigation in ImageViewer)
  useEffect(() => {
    setLocalT(slice.t);
    setLocalZ(slice.z);
    setLocalC(slice.c);
    setLocalPercentile(slice.percentileScale);
  }, [slice.t, slice.z, slice.c, slice.percentileScale]);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  // Load channel names when source changes
  useEffect(() => {
    loadChannelNames(sourceId);
  }, [sourceId, loadChannelNames]);

  const descriptor = useMemo(() => {
    const src = sources.find((s) => s.source_id === sourceId);
    return src?.tensors.find((t) => t.array_id === tensorId) ?? null;
  }, [sourceId, sources, tensorId]);

  const axisMap = useMemo(() => {
    if (!descriptor) return { t: null, z: null, c: null, y: null, x: null };
    return buildAxisMap(descriptor.dim_labels);
  }, [descriptor]);

  // Get channel name for current channel index
  const currentChannelName = useMemo(() => {
    const names = channelNames[sourceId];
    if (names && names[slice.c]) {
      return names[slice.c];
    }
    return null;
  }, [channelNames, sourceId, slice.c]);

  // Get current color for the channel
  const currentColor = useMemo(() => {
    return getChannelColor(sourceId, slice.c);
  }, [getChannelColor, sourceId, slice.c, channelColors]);

  // Determine if current color is a custom hex color
  const isCustomColor = useMemo(() => {
    return isHexColor(currentColor);
  }, [currentColor]);

  // Check if current color is "auto" mode
  const isAutoColor = useMemo(() => {
    return currentColor === "auto";
  }, [currentColor]);

  // Update useCustomColor state when color changes
  useEffect(() => {
    setUseCustomColor(isCustomColor);
  }, [isCustomColor]);

  // Get hex representation for the color picker input (resolves auto)
  const currentHexColor = useMemo(() => {
    return colorToHex(currentColor, currentChannelName ?? undefined);
  }, [currentColor, currentChannelName]);

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

  // Always show color picker - pseudo-color rendering is useful for any image
  const showColorPicker = true;

  // Handle preset color selection
  const handlePresetChange = (value: string) => {
    if (value === "custom") {
      setUseCustomColor(true);
    } else {
      setUseCustomColor(false);
      setChannelColor(sourceId, slice.c, value as ColorValue);
    }
  };

  // Handle custom color picker change
  const handleCustomColorChange = (hex: string) => {
    setChannelColor(sourceId, slice.c, hex);
  };

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
              value={clamp(localT, 0, tMax)}
              onChange={(e) => {
                const val = Number(e.target.value);
                setLocalT(val);
                if (debounceRef.current) clearTimeout(debounceRef.current);
                debounceRef.current = setTimeout(() => {
                  setSlice({ t: val });
                }, SLIDER_DEBOUNCE_MS);
              }}
              style={{ flex: 1 }}
            />
            <span style={{ width: 40, textAlign: "right", fontSize: 11 }}>{localT}/{tMax}</span>
          </label>
        )}

        {axisMap.z !== null && zSize > 1 && (
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 20 }}>Z</span>
            <input
              type="range"
              min={0}
              max={zMax}
              value={clamp(localZ, 0, zMax)}
              onChange={(e) => {
                const val = Number(e.target.value);
                setLocalZ(val);
                if (debounceRef.current) clearTimeout(debounceRef.current);
                debounceRef.current = setTimeout(() => {
                  setSlice({ z: val });
                }, SLIDER_DEBOUNCE_MS);
              }}
              style={{ flex: 1 }}
            />
            <span style={{ width: 40, textAlign: "right", fontSize: 11 }}>{localZ}/{zMax}</span>
          </label>
        )}

        {axisMap.c !== null && cSize > 1 && (
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 20 }}>C</span>
            <input
              type="range"
              min={0}
              max={cMax}
              value={clamp(localC, 0, cMax)}
              onChange={(e) => {
                const val = Number(e.target.value);
                setLocalC(val);
                if (debounceRef.current) clearTimeout(debounceRef.current);
                debounceRef.current = setTimeout(() => {
                  setSlice({ c: val });
                }, SLIDER_DEBOUNCE_MS);
              }}
              style={{ flex: 1 }}
            />
            <span style={{ width: 40, textAlign: "right", fontSize: 11 }}>{localC}/{cMax}</span>
          </label>
        )}

        {/* Intensity scaling controls */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 20, fontSize: 11, color: "#64748b" }}>Int</span>
          <button
            onClick={() => {
              setLocalPercentile(0);
              setSlice({ useMinMax: true, percentileScale: 0 });
            }}
            disabled={slice.useMinMax}
            style={{
              padding: "2px 6px",
              fontSize: 10,
              cursor: slice.useMinMax ? "default" : "pointer",
              background: slice.useMinMax ? "#4a5568" : "#2d3748",
              border: "1px solid #4a5568",
              borderRadius: 4,
              color: "#e2e8f0",
            }}
          >
            Min/Max
          </button>
          <input
            type="range"
            min={0}
            max={4}
            step={0.1}
            value={localPercentile}
            onChange={(e) => {
              const val = Number(e.target.value);
              setLocalPercentile(val);
              if (debounceRef.current) clearTimeout(debounceRef.current);
              debounceRef.current = setTimeout(() => {
                setSlice({ percentileScale: val, useMinMax: false });
              }, SLIDER_DEBOUNCE_MS);
            }}
            style={{ flex: 1 }}
          />
          <span style={{ width: 40, textAlign: "right", fontSize: 11 }}>
            {slice.useMinMax ? "0-100" : `${localPercentile.toFixed(1)}-${(100 - localPercentile).toFixed(1)}`}
          </span>
        </div>

        {showColorPicker && (
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ minWidth: 60, fontSize: 11, color: "#64748b" }}>Color</span>
            <select
              value={useCustomColor ? "custom" : currentColor}
              onChange={(e) => handlePresetChange(e.target.value)}
              style={{ flex: 1 }}
            >
              {PRESET_COLORS.map((c) => (
                <option key={c.value} value={c.value}>{c.name}</option>
              ))}
              <option value="custom">Custom...</option>
            </select>
            {/* Show color swatch for custom color */}
            {useCustomColor && (
              <input
                type="color"
                value={currentHexColor}
                onChange={(e) => handleCustomColorChange(e.target.value)}
                style={{
                  width: 32,
                  height: 24,
                  border: "1px solid #2d3748",
                  borderRadius: 4,
                  cursor: "pointer",
                  background: "transparent",
                }}
              />
            )}
            {/* Show resolved color swatch for auto mode */}
            {isAutoColor && !useCustomColor && currentChannelName && (
              <div
                style={{
                  width: 16,
                  height: 16,
                  borderRadius: 3,
                  background: currentHexColor,
                  border: "1px solid #2d3748",
                }}
                title={`Auto → ${resolveAutoColor(currentColor, currentChannelName ?? undefined)}`}
              />
            )}
            {currentChannelName && (
              <span
                style={{
                  fontSize: 10,
                  color: "#64748b",
                  maxWidth: 100,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
                title={currentChannelName}
              >
                {currentChannelName}
              </span>
            )}
          </div>
        )}

        {showAdvancedOptions && (
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
        )}
      </div>
    </section>
  );
}