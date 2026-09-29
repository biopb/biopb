"use client";

/**
 * Client-side rendered viewer: Viv over the tile API.
 *
 * Pixels arrive as raw tiles and contrast is applied in the shader, so panning
 * refetches only the tiles that came into view and a contrast drag costs no
 * round trip at all. That is what retired the server-rendered viewer this
 * replaced, which asked the server to re-render the whole region on every
 * interaction — fine on loopback, unusable across a WAN.
 *
 * This component is composition: each concern is a hook with a narrow contract
 * (`usePixelSources`, `usePlaneGate`, `useContrastSamples`, `useRoiOverlay`,
 * `useRoiAuthoring`, `useLabelOverlayLayers`, `useHoverReadout`) and the render
 * code is what is left. The deck.gl half is {@link VivStage}.
 *
 * Default-exported so the route can `lazy()` it: deck.gl and luma.gl are by far
 * the largest thing the app depends on and no other page needs them.
 */

import { useEffect, useMemo, useRef } from "react";
import { vivDtype, type TileInfo } from "@biopb/tensor-flight-client";
import { useShallow } from "zustand/react/shallow";
import { selectContrastWindow, useAppStore } from "../store";
import type { ViewerErrorKind } from "../store";
import { useContrastSamples } from "../hooks/useContrastSamples";
import { useElementSize } from "../hooks/useElementSize";
import { useHoverReadout } from "../hooks/useHoverReadout";
import { useLabelOverlayLayers } from "../hooks/useLabelOverlayLayers";
import { useMountEpoch, usePublishPlaneReady } from "../hooks/useMountEpoch";
import { usePixelSources } from "../hooks/usePixelSources";
import { usePlaneGate } from "../hooks/usePlaneGate";
import { useRoiAuthoring } from "../hooks/useRoiAuthoring";
import { useRoiOverlay } from "../hooks/useRoiOverlay";
import { useSliceWheelNavigation } from "../hooks/useSliceWheelNavigation";
import { clampGamma, samplesPerPixel, tileCacheSize, vivColor } from "../utils/vivUtils";
import { HoverReadout } from "./HoverReadout";
import { RoiToolStrip } from "./RoiToolStrip";
import { VivStage } from "./VivStage";
import { BADGE, OVERLAY_TEXT, greyLevel } from "./viewerStyles";

interface TileViewerProps {
  sourceId: string;
  /** The resolved `tile_info`, fetched once by `openTensor`. */
  info: TileInfo;
  /**
   * The tiled viewer gave up. `kind` separates a fact about the tensor
   * ("capability": no tile route, an unsupported dtype) from a bad moment
   * ("transport": the server timed out). Only the second is worth retrying.
   */
  onUnsupported: (reason: string, kind: ViewerErrorKind) => void;
}

/** The window before there is a grid to derive one from. */
const FALLBACK_WINDOW: [number, number] = [0, 1];

export default function TileViewer({ sourceId, info, onUnsupported }: TileViewerProps) {
  const epoch = useMountEpoch();
  const hostRef = useRef<HTMLDivElement | null>(null);
  const size = useElementSize(hostRef);
  useSliceWheelNavigation(hostRef, info);

  const { sources, tileError, clearTileError } = usePixelSources(info, onUnsupported);
  const { selection, loadedSelection, dataValid, onViewportLoad } = usePlaneGate(info);
  // Scoped to the plane that produced it, so a failed read cannot go on
  // labelling later planes that loaded perfectly well.
  useEffect(clearTileError, [selection, clearTileError]);

  const uniformValue = useContrastSamples(sources, info, selection);
  // The same selector the panel's bar reads (biopb/biopb#955).
  const contrastLimits = useAppStore(useShallow(selectContrastWindow)) ?? FALLBACK_WINDOW;
  // Never trusted straight from the store: a persisted or hand-edited value of 0
  // or below is a uniform white plane, not a dim one.
  const gamma = useAppStore((s) => clampGamma(s.display.gamma));

  // --- colour -------------------------------------------------------------
  const channel = useAppStore((s) => s.position.c);
  const channelNames = useAppStore((s) => s.channelNames);
  const channelColors = useAppStore((s) => s.channelColors);
  const color = useMemo(() => {
    const stored = channelColors[sourceId]?.[channel] ?? "auto";
    // `channelNames` is filled asynchronously, so this runs at least once with
    // the name still unknown. `resolveAutoColor` answers grey then and grey
    // again once an unrecognised name lands, which is what keeps the first
    // frames from being a different colour than the settled one.
    return vivColor(stored, channelNames[sourceId]?.[channel]);
  }, [channelColors, channelNames, sourceId, channel]);

  const maxCacheSize = useMemo(
    () => tileCacheSize(info.tile_size, vivDtype(info.dtype), samplesPerPixel(info), 1),
    [info],
  );

  // --- annotations and overlays ---------------------------------------------
  const { shownPlane, shown, roiLayers, selectionLayers } = useRoiOverlay(info, loadedSelection);
  const authoring = useRoiAuthoring(info, shownPlane, shown);
  const { hover, bindHover } = useHoverReadout(authoring.draftCursorSinkRef);
  const label = useLabelOverlayLayers(info, selection, loadedSelection);

  // Paced on the image *and* the overlay: see `useLabelOverlayLayers`.
  usePublishPlaneReady(dataValid && label.ready, epoch);

  // Under play the cover is dropped: at 10 frames a second it would be on
  // screen for most of every frame, which is a flicker rather than a warning,
  // and the thing it guards against -- mistaking a stale plane for the one
  // asked for -- cannot happen while the planes are deliberately marching past.
  const playing = useAppStore((s) => s.playAxis !== null);
  const showRois = useAppStore((s) => s.showRois);

  // The label fill goes under the annotations, which are line work a few pixels
  // wide: drawn over them it would cover them outright, drawn under it is the
  // background they are read against.
  const overlayLayers = useMemo(
    () => [...label.layers, ...roiLayers, ...selectionLayers, ...authoring.draftLayers],
    [label.layers, roiLayers, selectionLayers, authoring.draftLayers],
  );

  return (
    <div
      ref={hostRef}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden",
        background: "#1a1a2e",
      }}
    >
      {sources && size ? (
        <VivStage
          sources={sources}
          selection={selection}
          contrastLimits={contrastLimits}
          gamma={gamma}
          color={color}
          maxCacheSize={maxCacheSize}
          onViewportLoad={onViewportLoad}
          onHover={hover.onHover}
          hoverHooks={hover.hooks}
          overlayLayers={overlayLayers}
          onDeckClick={authoring.onDeckClick}
          width={size.width}
          height={size.height}
        />
      ) : (
        <div style={OVERLAY_TEXT}>Loading tiles…</div>
      )}
      {sources && size && !dataValid && !playing && (
        // Opaque, not a scrim: the point is that the stale plane stops being
        // visible, which a translucent overlay would not achieve.
        <div style={{ ...OVERLAY_TEXT, background: "#1a1a2e", zIndex: 1 }}>
          {tileError ? "Plane unavailable" : "Reading plane…"}
        </div>
      )}
      {sources && size && (dataValid || playing) && (
        <div style={{ position: "absolute", bottom: 10, left: 10, display: "grid", gap: 4, zIndex: 2 }}>
          {dataValid && uniformValue !== null && (
            <div
              style={{ ...BADGE, position: "static" }}
              title="Measured from the coarsest pyramid level, subsampled for the contrast histogram."
            >
              {uniformValue === 0
                ? "empty plane (all zeros)"
                : `uniform plane (value ${greyLevel(uniformValue)})`}
            </div>
          )}
          <HoverReadout bind={bindHover} />
        </div>
      )}
      {sources && size && showRois && (
        // Over the canvas, and above the "Reading plane" cover: switching tool
        // is a per-gesture action, and it should not disappear while a read is
        // outstanding.
        <div style={{ position: "absolute", top: 10, left: 10, zIndex: 2 }}>
          <RoiToolStrip draft={authoring.draft} onFinish={authoring.finishDraft} />
        </div>
      )}
      {tileError && (
        <div style={{ ...BADGE, bottom: 10, right: 10, color: "#ff6b6b" }}>{tileError}</div>
      )}
      {label.error && (
        // Its own badge, above the image's: an overlay that failed says nothing
        // about the pixels on screen, and the two must not be read as one fault.
        <div style={{ ...BADGE, bottom: tileError ? 38 : 10, right: 10, color: "#fbbf24" }}>
          Label overlay: {label.error}
        </div>
      )}
    </div>
  );
}
