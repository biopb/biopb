"use client";

import { sliderAxes, vivDtype, type SliderAxis } from "@biopb/tensor-flight-client";
import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { selectTileInfo, useAppStore } from "../store";
import {
  PRESET_COLORS,
  type ColorValue,
  colorToHex,
  isHexColor,
  resolveAutoColor,
} from "../utils/colorUtils";
import {
  PLAY_FPS,
  PLAY_FRAME_MS,
  PLAY_READY_POLL_MS,
  PLAY_STALL_MS,
  nextPlayIndex,
  orderSliderAxes,
  sliderThumbPx,
} from "../utils/sliceUi";
import {
  GAMMA_OCTAVES,
  clampContrastLimits,
  contrastLabel,
  contrastStep,
  dtypeContrastLimits,
  gammaFromOctaves,
  octavesFromGamma,
  percentileLabel,
  sliderGrid,
  withContrastLimit,
} from "../utils/vivUtils";
import { VOLUME_RENDER_MODES, volumeRefusal } from "../utils/volumeUtils";

// Debounce delay for slider updates (matches the viewer's keyboard+wheel debounce)
const SLIDER_DEBOUNCE_MS = 150;

interface SliceControlsProps {
  sourceId: string;
  tensorId: string;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** This axis's index in the live store, whatever it is keyed under. */
function axisIndex(axis: SliderAxis): number {
  const { slice } = useAppStore.getState();
  return axis.named ? slice[axis.named] : (slice.axes[axis.key] ?? 0);
}

/**
 * Write a slider's new index back, under its name or into `axes`.
 *
 * Reads the store at commit time rather than closing over a render's copy:
 * these writes are debounced and, under play, fired from a timer — a stale
 * `axes` map here would silently drop a sibling axis's index. The named axes
 * have their own store fields and cannot collide this way.
 */
function commitAxis(axis: SliderAxis, value: number) {
  const { setSlice, slice } = useAppStore.getState();
  if (axis.named) {
    setSlice({ [axis.named]: value });
    return;
  }
  setSlice({ axes: { ...slice.axes, [axis.key]: value } });
}

const CONTRAST_MODES = [
  {
    key: "auto" as const,
    label: "Auto",
    title: "Window taken from each plane's own histogram",
  },
  {
    key: "fixed" as const,
    label: "Fixed",
    title: "Window fixed at two grey levels, so planes stay comparable",
  },
];

/** Where a grey level sits on a track spanning `range`, as a percentage. */
function trackFraction(value: number, range: [number, number]): number {
  const span = range[1] - range[0];
  if (span <= 0) return 0;
  return ((value - range[0]) / span) * 100;
}

/** One readout, wide enough for its longest string and never wrapped. */
const VALUE_READOUT: CSSProperties = {
  width: 76,
  flexShrink: 0,
  textAlign: "right",
  fontSize: 11,
  // A readout that rewraps as the number grows moves the whole row. Fixed
  // width plus tabular figures also keeps the digits from jittering
  // left and right while a slider is dragged.
  whiteSpace: "nowrap",
  fontVariantNumeric: "tabular-nums",
};

export function SliceControls({ sourceId, tensorId }: SliceControlsProps) {
  const sources = useAppStore((s) => s.sources);
  const tileInfo = useAppStore(selectTileInfo);
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
  const appliedLimits = useAppStore((s) => s.appliedLimits);
  const planeLimits = useAppStore((s) => s.planeLimits);
  const playAxis = useAppStore((s) => s.playAxis);
  const setPlayAxis = useAppStore((s) => s.setPlayAxis);

  // Track custom color picker state (separate from preset dropdown)
  const [useCustomColor, setUseCustomColor] = useState(false);

  // Debounce timer ref for slider updates
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Local state for slider values (for immediate visual feedback), keyed by
  // SliderAxis.key so an axis with no name is held the same way as T/Z/C.
  const [localAxes, setLocalAxes] = useState<Record<string, number>>({});
  const [localPercentile, setLocalPercentile] = useState(slice.percentileScale);
  // The fixed window mid-drag, in grey levels; null when nothing is being
  // dragged and the store's own value is what to show.
  const [localFixed, setLocalFixed] = useState<[number, number] | null>(null);
  // Held in octaves, the units of the slider, so a drag does not round-trip
  // through log2/exp and drift off the position the user put it at.
  const [localOctaves, setLocalOctaves] = useState(() => octavesFromGamma(slice.gamma));

  // Sync local state when store slice changes (e.g., from wheel navigation in the viewer)
  useEffect(() => {
    setLocalAxes({ t: slice.t, z: slice.z, c: slice.c, ...slice.axes });
    setLocalPercentile(slice.percentileScale);
    setLocalFixed(null);
    setLocalOctaves(octavesFromGamma(slice.gamma));
  }, [
    slice.t,
    slice.z,
    slice.c,
    slice.axes,
    slice.percentileScale,
    slice.fixedLimits,
    slice.gamma,
  ]);

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

  // What the panel actually shows, in display order. Z is the volume's depth in
  // 3-D, read whole -- there is no plane to step through, so a slider for it
  // would move nothing.
  const visibleAxes = useMemo(
    () => orderSliderAxes(axes.filter((axis) => !(render3d && axis.named === "z"))),
    [axes, render3d],
  );

  // The axis sliders' track width, measured from the first of them -- every
  // axis row has the same layout, so one measurement sizes all their thumbs.
  // `sliderThumbPx` needs it because a range input's thumb cannot be sized as a
  // percentage of its track.
  const trackRef = useRef<HTMLInputElement | null>(null);
  const [trackPx, setTrackPx] = useState(0);
  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const measure = () => setTrackPx((w) => (w === el.clientWidth ? w : el.clientWidth));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [visibleAxes.length]);

  // A play cannot outlive the slider driving it: switching to 3-D takes Z off
  // the panel, and a tensor swap replaces the axes entirely.
  useEffect(() => {
    if (playAxis && !visibleAxes.some((axis) => axis.key === playAxis)) {
      setPlayAxis(null);
    }
  }, [playAxis, visibleAxes, setPlayAxis]);

  // Automatic scrubbing. Paced to the data plane rather than to the timer
  // alone: the next frame is asked for once the last one is actually on the
  // canvas, so a source that cannot deliver 10/s plays slower instead of
  // queueing reads it will never catch up on. PLAY_STALL_MS is the escape --
  // a plane that never loads must not stop the sequence.
  useEffect(() => {
    if (!playAxis) return;
    const axis = visibleAxes.find((a) => a.key === playAxis);
    if (!axis) return;
    let timer: ReturnType<typeof setTimeout>;
    let asked = Date.now();

    const step = () => {
      const store = useAppStore.getState();
      if (!store.planeReady && Date.now() - asked < PLAY_STALL_MS) {
        timer = setTimeout(step, PLAY_READY_POLL_MS);
        return;
      }
      commitAxis(axis, nextPlayIndex(axisIndex(axis), axis.extent));
      // Said here rather than waited for: the viewer publishes the same fact
      // from an effect, and this timer is armed before that effect runs.
      store.setPlaneReady(false);
      asked = Date.now();
      timer = setTimeout(step, PLAY_FRAME_MS);
    };

    timer = setTimeout(step, PLAY_FRAME_MS);
    return () => clearTimeout(timer);
  }, [playAxis, visibleAxes]);

  // Whether there is a second viewer to switch to. `volumeRefusal` answers for
  // the grid a viewer actually fetched, so this is null-when-unknown rather
  // than false-when-unknown: a tensor whose grid has not landed keeps the
  // toggle, and the 3-D pane reports its own refusal if one comes.
  const volumeOffered = useMemo(
    () => (tileInfo ? volumeRefusal(tileInfo) === null : true),
    [tileInfo],
  );

  // The track a fixed window is chosen on: the dtype's whole range, so the
  // window's position on the bar says what part of the possible signal is in
  // view. `sliderGrid` supplies the dtype from the live grid or the catalog.
  const dtypeRange = useMemo<[number, number]>(
    () => (descriptor ? dtypeContrastLimits(vivDtype(descriptor.dtype)) : [0, 1]),
    [descriptor],
  );
  const fixedStep = useMemo(() => contrastStep(dtypeRange), [dtypeRange]);
  // Local first (a drag in progress), then the committed window, then whatever
  // the viewer is applying -- which is the dtype range until one is chosen.
  const fixedWindow = useMemo<[number, number]>(
    () =>
      clampContrastLimits(localFixed ?? slice.fixedLimits ?? appliedLimits ?? dtypeRange, dtypeRange),
    [localFixed, slice.fixedLimits, appliedLimits, dtypeRange],
  );

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

  // "Already there" differs by mode: an untrimmed percentile, or a fixed window
  // already sitting on the plane's extremes. With nothing sampled yet there is
  // no image min/max to reset onto, so there is nothing for the button to do.
  const minMaxActive =
    slice.contrastMode === "auto"
      ? slice.percentileScale === 0
      : planeLimits === null ||
        (fixedWindow[0] === planeLimits[0] && fixedWindow[1] === planeLimits[1]);

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
        {/* Which viewer is mounted, not which pixels it asks for. The 3-D
            choice drops out when the live grid says this tensor has no volume
            to render -- the answer is the server's (`tile_info.volume`), and it
            is already here because the mounted viewer published the grid it
            fetched. The row itself stays, so the pane still says what it is
            showing. Offered while that grid is missing: unknown is not a
            refusal, and the 3-D pane says so itself for a tensor it cannot
            open. */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 20, fontSize: 11, color: "#64748b" }}>View</span>
          {(volumeOffered ? ([false, true] as const) : ([false] as const)).map((mode) => (
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

        {visibleAxes.map((axis, i) => {
          const max = Math.max(0, axis.extent - 1);
          const value = localAxes[axis.key] ?? 0;
          const playing = playAxis === axis.key;
          return (
            <div key={axis.key} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <label
                style={{ display: "flex", alignItems: "center", gap: 8, flex: 1 }}
                // The wire index is what the tile route is actually asked for,
                // so it is what to check against when a plane looks wrong.
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
                  ref={i === 0 ? trackRef : undefined}
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
                  // The grab handle is the axis's share of the track, as a
                  // scrollbar's is: two channels get half the bar, not the same
                  // sliver a 4000-frame timelapse is dragged by.
                  style={
                    {
                      flex: 1,
                      "--thumb-w": `${sliderThumbPx(axis.extent, trackPx)}px`,
                    } as CSSProperties
                  }
                />
                <span style={VALUE_READOUT}>
                  {value}/{max}
                </span>
              </label>
              <button
                onClick={() => setPlayAxis(playing ? null : axis.key)}
                title={
                  playing
                    ? `Stop scrubbing ${axis.title}`
                    : `Play ${axis.title} at up to ${PLAY_FPS}/s`
                }
                aria-label={playing ? `Stop ${axis.title}` : `Play ${axis.title}`}
                style={{
                  width: 24,
                  padding: "2px 0",
                  fontSize: 10,
                  lineHeight: "12px",
                  cursor: "pointer",
                  background: playing ? "#4f8ef7" : "#2d3748",
                  border: "1px solid #4a5568",
                  borderRadius: 4,
                  color: "#e2e8f0",
                }}
              >
                {playing ? "■" : "▶"}
              </button>
            </div>
          );
        })}

        {/* Navigation above, display below. Suppressed when nothing is above
            it: a rule against the top of the panel divides nothing. */}
        {visibleAxes.length > 0 && (
          <hr
            style={{
              width: "100%",
              margin: 0,
              border: 0,
              borderTop: "1px solid #2d3748",
            }}
          />
        )}

        {/* Intensity: how the contrast window is chosen, then the window. */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 20, fontSize: 11, color: "#64748b" }}>Int</span>
          {CONTRAST_MODES.map((mode) => (
            <button
              key={mode.key}
              onClick={() => {
                if (mode.key === "fixed") {
                  // Seeded from what is on screen, so turning the mode on does
                  // not change the image -- it stops it from changing.
                  setSlice({
                    contrastMode: "fixed",
                    fixedLimits: clampContrastLimits(appliedLimits ?? dtypeRange, dtypeRange),
                  });
                  return;
                }
                // `fixedLimits` is kept: toggling back must return to the
                // window the user chose, not to the dtype's whole range.
                setSlice({ contrastMode: "auto" });
              }}
              disabled={slice.contrastMode === mode.key}
              title={mode.title}
              style={{
                padding: "2px 8px",
                fontSize: 10,
                cursor: slice.contrastMode === mode.key ? "default" : "pointer",
                background: slice.contrastMode === mode.key ? "#4a5568" : "#2d3748",
                border: "1px solid #4a5568",
                borderRadius: 4,
                color: "#e2e8f0",
              }}
            >
              {mode.label}
            </button>
          ))}
          {/* The untrimmed window, in whichever sense the mode gives it: the
              automatic window with neither tail trimmed, or a fixed one moved
              onto the extremes of the plane in view. */}
          <button
            onClick={() => {
              if (slice.contrastMode === "fixed") {
                if (!planeLimits) return;
                setLocalFixed(null);
                setSlice({ fixedLimits: clampContrastLimits(planeLimits, dtypeRange) });
                return;
              }
              setLocalPercentile(0);
              setSlice({ percentileScale: 0 });
            }}
            disabled={minMaxActive}
            title={
              slice.contrastMode === "fixed"
                ? "Set the window to this image's own min and max grey level"
                : "Automatic window with neither tail trimmed"
            }
            style={{
              padding: "2px 6px",
              fontSize: 10,
              cursor: minMaxActive ? "default" : "pointer",
              background: minMaxActive ? "#4a5568" : "#2d3748",
              border: "1px solid #4a5568",
              borderRadius: 4,
              color: "#e2e8f0",
            }}
          >
            Min/Max
          </button>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 20 }} />
          {slice.contrastMode === "auto" ? (
            <input
              type="range"
              min={0}
              max={4}
              step={0.1}
              value={localPercentile}
              aria-label="Percentile window width"
              title="How much of each tail the automatic window trims"
              onChange={(e) => {
                const val = Number(e.target.value);
                setLocalPercentile(val);
                if (debounceRef.current) clearTimeout(debounceRef.current);
                debounceRef.current = setTimeout(() => {
                  setSlice({ percentileScale: val });
                }, SLIDER_DEBOUNCE_MS);
              }}
              style={{ flex: 1 }}
            />
          ) : (
            // Two grabs on one bar, and the bar is the dtype's whole range --
            // so where the window sits says what part of the possible signal is
            // being shown, which a self-scaled bar could not.
            <div className="dual-range" style={{ flex: 1 }}>
              <div className="dual-range-track" />
              <div
                className="dual-range-fill"
                style={{
                  left: `${trackFraction(fixedWindow[0], dtypeRange)}%`,
                  right: `${100 - trackFraction(fixedWindow[1], dtypeRange)}%`,
                }}
              />
              {(["lo", "hi"] as const).map((end) => (
                <input
                  key={end}
                  type="range"
                  min={dtypeRange[0]}
                  max={dtypeRange[1]}
                  step={fixedStep}
                  value={end === "lo" ? fixedWindow[0] : fixedWindow[1]}
                  aria-label={end === "lo" ? "Black level" : "White level"}
                  title={end === "lo" ? "Grey level rendered black" : "Grey level rendered white"}
                  // The ends may sit one step apart, which puts the two thumbs
                  // on the same pixel. Whichever one has travelled past the
                  // middle goes on top, so the grab that can still move the
                  // window is always the one the pointer finds.
                  style={{
                    zIndex:
                      end === "lo" && trackFraction(fixedWindow[0], dtypeRange) > 50 ? 2 : 1,
                  }}
                  onChange={(e) => {
                    const next = withContrastLimit(
                      fixedWindow,
                      end,
                      Number(e.target.value),
                      dtypeRange,
                      fixedStep,
                    );
                    setLocalFixed(next);
                    if (debounceRef.current) clearTimeout(debounceRef.current);
                    debounceRef.current = setTimeout(() => {
                      setSlice({ contrastMode: "fixed", fixedLimits: next });
                    }, SLIDER_DEBOUNCE_MS);
                  }}
                />
              ))}
            </div>
          )}
          <span style={VALUE_READOUT}>
            {slice.contrastMode === "auto"
              ? percentileLabel(localPercentile)
              : contrastLabel(fixedWindow, fixedStep)}
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
          <span style={VALUE_READOUT}>{gammaFromOctaves(localOctaves).toFixed(2)}</span>
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
