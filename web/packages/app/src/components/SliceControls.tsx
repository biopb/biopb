"use client";

import { sliderAxes, type SliderAxis } from "@biopb/tensor-flight-client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useAppStore } from "../store";
import {
  PRESET_COLORS,
  type ColorValue,
  colorToHex,
  isHexColor,
  resolveAutoColor,
} from "../utils/colorUtils";
import { GAMMA_OCTAVES, gammaFromOctaves, octavesFromGamma, sliderGrid } from "../utils/vivUtils";
import { VOLUME_RENDER_MODES } from "../utils/volumeUtils";

// Debounce delay for slider updates (matches the viewer's keyboard+wheel debounce)
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
  const tileInfo = useAppStore((s) => s.tileInfo);
  const slice = useAppStore((s) => s.slice);
  const setSlice = useAppStore((s) => s.setSlice);
  const channelNames = useAppStore((s) => s.channelNames);
  const channelColors = useAppStore((s) => s.channelColors);
  const getChannelColor = useAppStore((s) => s.getChannelColor);
  const setChannelColor = useAppStore((s) => s.setChannelColor);
  const loadChannelNames = useAppStore((s) => s.loadChannelNames);
  const render3d = useAppStore((s) => s.render3d);
  const setRender3d = useAppStore((s) => s.setRender3d);
  const volumeRenderMode = useAppStore((s) => s.volumeRenderMode);
  const setVolumeRenderMode = useAppStore((s) => s.setVolumeRenderMode);

  // Track custom color picker state (separate from preset dropdown)
  const [useCustomColor, setUseCustomColor] = useState(false);

  // Debounce timer ref for slider updates
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Local state for slider values (for immediate visual feedback), keyed by
  // SliderAxis.key so an axis with no name is held the same way as T/Z/C.
  const [localAxes, setLocalAxes] = useState<Record<string, number>>({});
  const [localPercentile, setLocalPercentile] = useState(slice.percentileScale);
  // Held in octaves, the units of the slider, so a drag does not round-trip
  // through log2/exp and drift off the position the user put it at.
  const [localOctaves, setLocalOctaves] = useState(() => octavesFromGamma(slice.gamma));

  // Sync local state when store slice changes (e.g., from wheel navigation in the viewer)
  useEffect(() => {
    setLocalAxes({ t: slice.t, z: slice.z, c: slice.c, ...slice.axes });
    setLocalPercentile(slice.percentileScale);
    setLocalOctaves(octavesFromGamma(slice.gamma));
  }, [slice.t, slice.z, slice.c, slice.axes, slice.percentileScale, slice.gamma]);

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

  // See `sliderGrid`: the live grid bounds the sliders, the catalog is only a
  // fallback for before a viewer has loaded.
  const descriptor = useMemo(
    () => sliderGrid(tileInfo, sources, sourceId, tensorId),
    [tileInfo, sources, sourceId, tensorId],
  );

  // Not buildAxisMap: its positional fallback would title a TIFF sequence's
  // `i` axis "Z", asserting depth about 155 stacked files on the strength of
  // nothing. sliderAxes navigates the same axes and names only the named ones.
  const axes: SliderAxis[] = useMemo(() => {
    if (!descriptor) return [];
    return sliderAxes(descriptor.dim_labels, descriptor.shape).filter(
      (axis) => axis.extent > 1,
    );
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
    // getChannelColor is a stable store method that reads channelColors via get();
    // list channelColors so the memo recomputes when a color is edited.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  // Always show color picker - pseudo-color rendering is useful for any image
  const showColorPicker = true;

  /** Write a slider's new index back, under its name or into `axes`. */
  const commitAxis = (axis: SliderAxis, value: number) => {
    if (axis.named) {
      setSlice({ [axis.named]: value });
      return;
    }
    // Read `axes` at commit time rather than closing over the render's copy:
    // these writes are debounced, so a stale map here would silently drop a
    // sibling axis's index every time two of them are moved in quick
    // succession. The named axes have their own store fields and cannot
    // collide this way.
    setSlice({
      axes: { ...useAppStore.getState().slice.axes, [axis.key]: value },
    });
  };

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
        {/* Which viewer is mounted, not which pixels it asks for. Offered
            unconditionally: whether this tensor has a volume is the server's
            answer (`tile_info.volume`), and the 3-D pane is the thing that
            asks — disabling the toggle here would mean fetching that a second
            time just to grey out a button. */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 20, fontSize: 11, color: "#64748b" }}>View</span>
          {([false, true] as const).map((mode) => (
            <button
              key={String(mode)}
              onClick={() => setRender3d(mode)}
              disabled={render3d === mode}
              style={{
                padding: "2px 8px",
                fontSize: 10,
                cursor: render3d === mode ? "default" : "pointer",
                background: render3d === mode ? "#4a5568" : "#2d3748",
                border: "1px solid #4a5568",
                borderRadius: 4,
                color: "#e2e8f0",
              }}
            >
              {mode ? "3D" : "2D"}
            </button>
          ))}
        </div>

        {/* How the ray-cast combines voxels. Only in 3-D: it names nothing in a
            plane view, where there is one voxel along the ray. */}
        {render3d && (
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 20, fontSize: 11, color: "#64748b" }}>Proj</span>
            {VOLUME_RENDER_MODES.map((mode) => (
              <button
                key={mode.key}
                onClick={() => setVolumeRenderMode(mode.key)}
                disabled={volumeRenderMode === mode.key}
                title={mode.title}
                style={{
                  padding: "2px 8px",
                  fontSize: 10,
                  cursor: volumeRenderMode === mode.key ? "default" : "pointer",
                  background: volumeRenderMode === mode.key ? "#4a5568" : "#2d3748",
                  border: "1px solid #4a5568",
                  borderRadius: 4,
                  color: "#e2e8f0",
                }}
              >
                {mode.label}
              </button>
            ))}
          </div>
        )}

        {axes
          // Z is the volume's depth in 3-D, read whole — there is no plane to
          // step through, so a slider for it would move nothing.
          .filter((axis) => !(render3d && axis.named === "z"))
          .map((axis) => {
          const max = Math.max(0, axis.extent - 1);
          const value = localAxes[axis.key] ?? 0;
          return (
            <label
              key={axis.key}
              style={{ display: "flex", alignItems: "center", gap: 8 }}
              // The wire index is what the tile route is actually asked for, so
              // it is what to check against when a plane looks wrong.
              title={`${axis.title} — wire axis ${axis.axis}, ${axis.extent} positions`}
            >
              <span
                style={{
                  width: 20,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  // An unnamed axis carries the source's own label, which is
                  // not a single letter and should not pretend to be one.
                  fontSize: axis.named ? undefined : 10,
                  color: axis.named ? undefined : "#94a3b8",
                }}
              >
                {axis.title}
              </span>
              <input
                type="range"
                min={0}
                max={max}
                value={clamp(value, 0, max)}
                onChange={(e) => {
                  const val = Number(e.target.value);
                  setLocalAxes((prev) => ({ ...prev, [axis.key]: val }));
                  if (debounceRef.current) clearTimeout(debounceRef.current);
                  debounceRef.current = setTimeout(() => {
                    commitAxis(axis, val);
                  }, SLIDER_DEBOUNCE_MS);
                }}
                style={{ flex: 1 }}
              />
              <span style={{ width: 40, textAlign: "right", fontSize: 11 }}>
                {value}/{max}
              </span>
            </label>
          );
        })}

        {/* Navigation above, display below. Suppressed when nothing is above
            it: a rule against the top of the panel divides nothing. */}
        {axes.length > 0 && (
          <hr
            style={{
              width: "100%",
              margin: 0,
              border: 0,
              borderTop: "1px solid #2d3748",
            }}
          />
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

        {/* Gamma: the shape of the ramp between those two limits. */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{ width: 20, fontSize: 11, color: "#64748b" }}
            title="Gamma: exponent applied to the normalized intensity"
          >
            Gam
          </span>
          <button
            onClick={() => {
              setLocalOctaves(0);
              setSlice({ gamma: 1 });
            }}
            disabled={slice.gamma === 1}
            style={{
              padding: "2px 6px",
              fontSize: 10,
              cursor: slice.gamma === 1 ? "default" : "pointer",
              background: slice.gamma === 1 ? "#4a5568" : "#2d3748",
              border: "1px solid #4a5568",
              borderRadius: 4,
              color: "#e2e8f0",
            }}
          >
            Linear
          </button>
          <input
            type="range"
            // In octaves, not in gamma: halving and doubling are equal and
            // opposite corrections, so they belong the same distance from the
            // centre. A linear 0.25-4 track would spend four fifths of its
            // travel on darkening.
            min={-GAMMA_OCTAVES}
            max={GAMMA_OCTAVES}
            step={0.05}
            value={localOctaves}
            onChange={(e) => {
              const val = Number(e.target.value);
              setLocalOctaves(val);
              if (debounceRef.current) clearTimeout(debounceRef.current);
              debounceRef.current = setTimeout(() => {
                setSlice({ gamma: gammaFromOctaves(val) });
              }, SLIDER_DEBOUNCE_MS);
            }}
            style={{ flex: 1 }}
          />
          <span style={{ width: 40, textAlign: "right", fontSize: 11 }}>
            {gammaFromOctaves(localOctaves).toFixed(2)}
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
      </div>
    </section>
  );
}
