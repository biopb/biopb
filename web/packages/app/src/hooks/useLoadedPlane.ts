import { useCallback, useRef, useState } from "react";

/**
 * Whether a `TileLayer`/`ImageLayer` viewport-load report means the plane
 * actually landed.
 *
 * Two different things report. A pyramid gets Viv's MultiscaleImageLayer and
 * deck.gl's TileLayer under it, which reports the array of tiles; an image small
 * enough to need only one level gets Viv's plain ImageLayer, which reports the
 * single raster it just read. Assuming the array shape leaves every single-level
 * image permanently covered.
 *
 * A *failed* tile still counts as loaded to deck.gl -- `_isLoaded = true` with
 * `content = null` -- so a viewport whose reads all errored reports itself
 * complete. Taking that at face value would show a plane that is not there. An
 * aborted tile is not affected: deck.gl leaves that one unloaded, so it never
 * reaches here. The ImageLayer branch needs no check: a raster that failed
 * rejects, and it only reports on the resolved path.
 */
export function viewportHoldsPlane(loaded?: unknown): boolean {
  if (!Array.isArray(loaded)) return true;
  return !loaded.some((tile: { content?: unknown } | null) => tile?.content == null);
}

/**
 * Track which request a layer's viewport last finished loading.
 *
 * Viv layers keep their previous raster until a new read resolves, so "the
 * request I made" and "what is on screen" differ for as long as a read takes.
 * `requested` is the current request, compared by `===`: pass an object when the
 * layer refetches on a new reference, a string when a key is enough. `loaded`
 * is the last request the viewport completed for, or null before the first.
 *
 * `onViewportLoad` reads the request through a ref and so keeps one identity for
 * the life of the hook: a fresh one every render would look like a prop change to
 * deck.gl. It records the request current when the load *reports*, which is the
 * one the layer was last asked for.
 */
export function useLoadedPlane<T>(requested: T): {
  loaded: T | null;
  onViewportLoad: (loaded?: unknown) => void;
} {
  const [loaded, setLoaded] = useState<T | null>(null);
  const requestedRef = useRef(requested);
  requestedRef.current = requested;
  const onViewportLoad = useCallback((report?: unknown) => {
    if (viewportHoldsPlane(report)) setLoaded(requestedRef.current);
  }, []);
  return { loaded, onViewportLoad };
}
