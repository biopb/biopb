import { useCallback, useMemo, useRef, useState } from "react";
import type { TileInfo } from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";
import { vivSelection } from "../utils/vivUtils";

/**
 * Is what is on screen the plane that was asked for?
 *
 * Both Viv layers keep their previous raster until a new read resolves, so a
 * t/c/z change leaves the old plane painted for exactly as long as the read
 * takes -- with nothing on screen to say so. That is worse than a blank frame: a
 * stale plane is indistinguishable from the right one, and a plane that never
 * changed reads as a hung viewer rather than a slow one.
 *
 * `selection` is the plane asked for and `loadedSelection` the last one whose
 * viewport finished loading. Both are compared by identity: `selection` is
 * derived from `position` alone, which only a move within the grid replaces
 * (`setPosition` keeps the object when nothing changed), so a contrast drag or a
 * colour change leaves it the same object and Viv's ImageLayer, which refetches
 * on a new *reference*, does not refetch.
 */
export function usePlaneGate(info: TileInfo) {
  const position = useAppStore((s) => s.position);
  const selection = useMemo<Record<string, number>>(
    () => vivSelection(info, position),
    [info, position],
  );

  const [loadedSelection, setLoadedSelection] = useState<Record<string, number> | null>(null);
  // Read through a ref: the callback's identity has to stay stable or every
  // layerProps rebuild would look like a prop change to deck.gl.
  const selectionRef = useRef(selection);
  selectionRef.current = selection;

  // deck.gl's TileLayer reports when the viewport's tiles have all landed.
  // Reached through Viv, which forwards unknown props down to it and pins the
  // background ImageLayer's own callback to null, so this fires once per
  // completed viewport and not twice.
  const onViewportLoad = useCallback((loaded?: unknown) => {
    // Two different things call this. A pyramid gets Viv's MultiscaleImageLayer
    // and deck.gl's TileLayer under it, which reports the array of tiles; an
    // image small enough to need only one level gets Viv's plain ImageLayer,
    // which reports the single raster it just read. Assuming the array shape
    // leaves every single-level image permanently covered.
    if (Array.isArray(loaded)) {
      // A *failed* tile still counts as loaded to deck.gl -- `_isLoaded = true`
      // with `content = null` -- so a viewport whose reads all errored reports
      // itself complete. Taking that at face value would clear the cover over a
      // canvas that never got the plane, which is the ambiguity this gate exists
      // to remove. An aborted tile is not affected: deck.gl leaves that one
      // unloaded, so it never reaches here.
      if (loaded.some((tile: { content?: unknown } | null) => tile?.content == null)) return;
    }
    // The ImageLayer branch needs no such check: a raster that failed rejects,
    // and it only calls this on the resolved path.
    setLoadedSelection(selectionRef.current);
  }, []);

  // Zoom and pan never invalidate: they change which tiles are wanted, not
  // which plane, so their partial state is legitimate progressive refinement.
  const dataValid = loadedSelection !== null && loadedSelection === selection;

  return { selection, loadedSelection, dataValid, onViewportLoad };
}
