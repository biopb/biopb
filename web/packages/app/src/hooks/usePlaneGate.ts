import { useMemo } from "react";
import type { TileInfo } from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";
import { vivSelection } from "../utils/vivUtils";
import { useLoadedPlane } from "./useLoadedPlane";

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

  // deck.gl's TileLayer reports when the viewport's tiles have all landed.
  // Reached through Viv, which forwards unknown props down to it and pins the
  // background ImageLayer's own callback to null, so this fires once per
  // completed viewport and not twice.
  const { loaded: loadedSelection, onViewportLoad } = useLoadedPlane(selection);

  // Zoom and pan never invalidate: they change which tiles are wanted, not
  // which plane, so their partial state is legitimate progressive refinement.
  const dataValid = loadedSelection !== null && loadedSelection === selection;

  return { selection, loadedSelection, dataValid, onViewportLoad };
}
