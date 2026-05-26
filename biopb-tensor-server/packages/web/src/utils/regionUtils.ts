/**
 * Region utilities for image viewer.
 *
 * Handles computing slice ranges, scale hints, and missing content detection.
 */

import { buildAxisMap, computeScaleHint, type AxisMap } from "@biopb/tensor-flight-client";

// Re-export AxisMap for convenience
export type { AxisMap } from "@biopb/tensor-flight-client";

export interface LoadedRegion {
  x: number;      // world X start
  y: number;      // world Y start
  width: number;  // world width
  height: number; // world height
  scaleFactors: number[];
}

export interface VisibleBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ViewportState {
  centerX: number;  // world coordinates - center of viewport
  centerY: number;  // world coordinates - center of viewport
  scale: number;    // zoom level (1.0 = fit-to-view, higher = zoomed in)
}

export interface ScaleResult {
  factors: number[];
}

/** Hysteresis threshold for scale changes (20%) */
export const HYSTERESIS = 0.2;

/**
 * Compute visible world bounds from viewport state and wrapper dimensions.
 *
 * @param viewportState Current viewport center and scale
 * @param wrapperWidth Wrapper element width in pixels
 * @param wrapperHeight Wrapper element height in pixels
 * @param fullWidth Full tensor width in world units
 * @param fullHeight Full tensor height in world units
 * @returns Visible bounds in world coordinates, clipped to tensor bounds
 */
export function computeVisibleBounds(
  viewportState: ViewportState,
  wrapperWidth: number,
  wrapperHeight: number,
  fullWidth: number,
  fullHeight: number,
): VisibleBounds {
  // Viewport dimensions in world units
  const viewWidthWorld = wrapperWidth / viewportState.scale;
  const viewHeightWorld = wrapperHeight / viewportState.scale;

  // Visible region center in world coords
  const centerX = viewportState.centerX;
  const centerY = viewportState.centerY;

  // Compute visible bounds (centered on viewport center)
  const x = centerX - viewWidthWorld / 2;
  const y = centerY - viewHeightWorld / 2;

  // Clip to tensor bounds
  const clippedX = Math.max(0, x);
  const clippedY = Math.max(0, y);
  const endX = Math.min(fullWidth, x + viewWidthWorld);
  const endY = Math.min(fullHeight, y + viewHeightWorld);

  return {
    x: clippedX,
    y: clippedY,
    width: Math.max(0, endX - clippedX),
    height: Math.max(0, endY - clippedY),
  };
}

/**
 * Compute tensor slice range from visible bounds with buffer margin.
 *
 * World coordinates = original tensor indices.
 * IMPORTANT: slice coordinates are snapped to be divisible by scaleFactors
 * to avoid 1-pixel shift issues when server processes slice_hint with scale_hint.
 *
 * @param visibleBounds Current visible region in world coordinates
 * @param axisMap Axis mapping for Y/X dimensions
 * @param tensorShape Full tensor shape
 * @param scaleFactors Scale factors for each dimension
 * @param bufferFraction Buffer margin fraction (default 0.5 = 50%)
 * @returns Slice ranges for Y and X dimensions
 */
export function computeSliceRange(
  visibleBounds: VisibleBounds,
  axisMap: AxisMap,
  tensorShape: number[],
  scaleFactors: number[],
  bufferFraction: number = 0.5,
): { y: [number, number]; x: [number, number] } {
  const yIdx = axisMap.y ?? 0;
  const xIdx = axisMap.x ?? 1;
  const fullHeight = tensorShape[yIdx] ?? 1;
  const fullWidth = tensorShape[xIdx] ?? 1;
  const scaleY = scaleFactors[yIdx] ?? 1;
  const scaleX = scaleFactors[xIdx] ?? 1;

  // Buffer margin for smooth panning (in world units)
  const marginY = visibleBounds.height * bufferFraction;
  const marginX = visibleBounds.width * bufferFraction;

  // World coords = original tensor indices
  // Snap to scale factor divisibility to avoid rounding issues
  const yStartRaw = Math.max(0, Math.floor(visibleBounds.y - marginY));
  const yEndRaw = Math.min(fullHeight, Math.ceil(visibleBounds.y + visibleBounds.height + marginY));
  const xStartRaw = Math.max(0, Math.floor(visibleBounds.x - marginX));
  const xEndRaw = Math.min(fullWidth, Math.ceil(visibleBounds.x + visibleBounds.width + marginX));

  // Snap to multiples of scale factor (floor for start, ceil for end)
  const yStart = scaleY > 1 ? Math.floor(yStartRaw / scaleY) * scaleY : yStartRaw;
  const yEnd = scaleY > 1 ? Math.ceil(yEndRaw / scaleY) * scaleY : yEndRaw;
  const xStart = scaleX > 1 ? Math.floor(xStartRaw / scaleX) * scaleX : xStartRaw;
  const xEnd = scaleX > 1 ? Math.ceil(xEndRaw / scaleX) * scaleX : xEndRaw;

  // Clamp again after snapping
  const yStartClamped = Math.max(0, yStart);
  const yEndClamped = Math.min(fullHeight, yEnd);
  const xStartClamped = Math.max(0, xStart);
  const xEndClamped = Math.min(fullWidth, xEnd);

  return {
    y: [yStartClamped, yEndClamped],
    x: [xStartClamped, xEndClamped],
  };
}

/**
 * Check if reload needed based on visible region coverage (no scale check).
 * Used at event time to decide whether to queue a reload.
 *
 * @param visibleBounds Current visible region
 * @param loadedRegion Currently loaded region
 * @returns true if reload needed (visible extends outside loaded)
 */
export function shouldReloadCheck(
  visibleBounds: VisibleBounds,
  loadedRegion: LoadedRegion | null,
): boolean {
  if (!loadedRegion) return true;

  // Check region coverage: reload when visible region extends outside loaded region
  const visibleEndX = visibleBounds.x + visibleBounds.width;
  const visibleEndY = visibleBounds.y + visibleBounds.height;
  const loadedEndX = loadedRegion.x + loadedRegion.width;
  const loadedEndY = loadedRegion.y + loadedRegion.height;

  // Small tolerance to avoid reload on rounding errors at clamped edges
  const TOLERANCE = 1.0;

  // Reload if any edge of visible region is outside loaded region (beyond tolerance)
  if (visibleBounds.x < loadedRegion.x - TOLERANCE) return true;
  if (visibleBounds.y < loadedRegion.y - TOLERANCE) return true;
  if (visibleEndX > loadedEndX + TOLERANCE) return true;
  if (visibleEndY > loadedEndY + TOLERANCE) return true;

  return false;
}

/**
 * Check if reload needed: scale change OR viewport extends outside loaded region.
 * Used when both current scaleFactors and visibleBounds are known.
 *
 * @param newScaleFactors New scale factors
 * @param visibleBounds Current visible region
 * @param loadedRegion Currently loaded region
 * @returns true if reload needed
 */
export function shouldReload(
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

  return shouldReloadCheck(visibleBounds, loadedRegion);
}

/**
 * Compute scale factors from viewport zoom level.
 *
 * @param tensorShape Full tensor shape
 * @param axisMap Axis mapping
 * @param viewportW Viewport width in pixels
 * @param viewportH Viewport height in pixels
 * @param pixelBudget Pixel budget for rendering
 * @param zoom Current zoom level
 * @returns Scale result with factors array
 */
export function computeScaleFactors(
  tensorShape: number[],
  axisMap: AxisMap,
  viewportW: number,
  viewportH: number,
  pixelBudget: number,
  zoom: number,
): ScaleResult {
  return computeScaleHint(
    tensorShape,
    axisMap,
    viewportW,
    viewportH,
    pixelBudget,
    zoom,
  );
}

/**
 * Compute initial viewport state to fit image in wrapper.
 *
 * @param fullWidth Full image width in world units
 * @param fullHeight Full image height in world units
 * @param wrapperWidth Wrapper width in pixels
 * @param wrapperHeight Wrapper height in pixels
 * @returns Initial viewport state (center at image center, fit scale)
 */
export function computeInitialViewportState(
  fullWidth: number,
  fullHeight: number,
  wrapperWidth: number,
  wrapperHeight: number,
): ViewportState {
  const fitScale = Math.min(
    1,
    wrapperWidth / Math.max(1, fullWidth),
    wrapperHeight / Math.max(1, fullHeight),
  );

  return {
    centerX: fullWidth / 2,
    centerY: fullHeight / 2,
    scale: fitScale,
  };
}

/**
 * Clamp viewport center to stay within image bounds after zoom/pan.
 *
 * @param viewportState Current viewport state
 * @param wrapperWidth Wrapper width in pixels
 * @param wrapperHeight Wrapper height in pixels
 * @param fullWidth Full image width in world units
 * @param fullHeight Full image height in world units
 * @returns Clamped viewport state
 */
export function clampViewportToBounds(
  viewportState: ViewportState,
  wrapperWidth: number,
  wrapperHeight: number,
  fullWidth: number,
  fullHeight: number,
): ViewportState {
  // Viewport dimensions in world units
  const viewWidthWorld = wrapperWidth / viewportState.scale;
  const viewHeightWorld = wrapperHeight / viewportState.scale;

  // Clamp center so viewport stays overlapping with image
  // When zoomed in (viewport smaller than image): allow pan to image edges
  // When zoomed out (viewport larger than image): center the image

  let centerX: number;
  let centerY: number;

  if (viewWidthWorld <= fullWidth) {
    // Zoomed in: allow panning until viewport edge hits image edge
    const minX = viewWidthWorld / 2;
    const maxX = fullWidth - viewWidthWorld / 2;
    centerX = Math.max(minX, Math.min(maxX, viewportState.centerX));
  } else {
    // Zoomed out: keep image centered in viewport
    centerX = fullWidth / 2;
  }

  if (viewHeightWorld <= fullHeight) {
    // Zoomed in: allow panning until viewport edge hits image edge
    const minY = viewHeightWorld / 2;
    const maxY = fullHeight - viewHeightWorld / 2;
    centerY = Math.max(minY, Math.min(maxY, viewportState.centerY));
  } else {
    // Zoomed out: keep image centered in viewport
    centerY = fullHeight / 2;
  }

  return { ...viewportState, centerX, centerY };
}

/**
 * Compute CSS transform for positioning the image element.
 *
 * The image shows loadedRegion, positioned relative to viewport.
 *
 * @param loadedRegion Currently loaded region (what the image shows)
 * @param viewportState Current viewport state (where user is looking)
 * @param wrapperWidth Wrapper width in pixels
 * @param wrapperHeight Wrapper height in pixels
 * @returns CSS positioning { left, top, width, height } in pixels
 */
export function computeImageCSS(
  loadedRegion: LoadedRegion,
  viewportState: ViewportState,
  wrapperWidth: number,
  wrapperHeight: number,
): { left: number; top: number; width: number; height: number } {
  // Image position relative to viewport center
  const left = (loadedRegion.x - viewportState.centerX) * viewportState.scale + wrapperWidth / 2;
  const top = (loadedRegion.y - viewportState.centerY) * viewportState.scale + wrapperHeight / 2;

  // Image dimensions scaled by viewport scale
  const width = loadedRegion.width * viewportState.scale;
  const height = loadedRegion.height * viewportState.scale;

  return { left, top, width, height };
}

// Re-export from tensor-flight-client for convenience
export { buildAxisMap } from "@biopb/tensor-flight-client";