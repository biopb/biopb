import { useMemo } from "react";
import type { TileInfo } from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";
import { compositeChannels, compositeSelections, withChannel } from "../utils/composite";
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
  const composite = useAppStore((s) => s.composite);

  // The channels drawn together, or null for the one the C slider names.
  const channels = useMemo(() => compositeChannels(info, composite), [info, composite]);

  // What is asked of Viv: one selection per drawn channel. A composite draws
  // them all whatever `position.c` is, so it must not be a dependency there --
  // a C move would otherwise hand Viv a new array of the same planes, which it
  // refetches, and the gate below would cover a picture that never changed.
  const planeKey = `${position.t}|${position.z}|${JSON.stringify(position.axes)}`;
  const slider = channels ? -1 : position.c;
  const selections = useMemo<Record<string, number>[]>(
    () =>
      channels
        ? compositeSelections(info, position, channels)
        : [vivSelection(info, position)],
    // `position` itself is replaced by any move; the key and the channel are
    // the parts of it a selection reads.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [info, channels, planeKey, slider],
  );
  // The plane the gate tracks: the first drawn channel's, which is stable across
  // C moves in a composite and is the current channel's otherwise.
  const selection = selections[0] as Record<string, number>;

  // deck.gl's TileLayer reports when the viewport's tiles have all landed.
  // Reached through Viv, which forwards unknown props down to it and pins the
  // background ImageLayer's own callback to null, so this fires once per
  // completed viewport and not twice.
  const { loaded: loadedSelection, onViewportLoad } = useLoadedPlane(selection);

  // Zoom and pan never invalidate: they change which tiles are wanted, not
  // which plane, so their partial state is legitimate progressive refinement.
  const dataValid = loadedSelection !== null && loadedSelection === selection;

  // What annotations are drawn for and placed on. In a composite no single
  // channel is *the* plane, so they follow the C slider: the plane it names is
  // the one a user means, and it is the one they will see again with the
  // composite off.
  const annotationSelection = useMemo(
    () => (channels ? withChannel(info, selection, position.c) : selection),
    [info, channels, selection, position.c],
  );
  const annotationLoaded = useMemo(
    () => (channels && loadedSelection ? withChannel(info, loadedSelection, position.c) : loadedSelection),
    [info, channels, loadedSelection, position.c],
  );

  return {
    channels,
    selections,
    selection,
    loadedSelection,
    annotationSelection,
    annotationLoaded,
    dataValid,
    onViewportLoad,
  };
}
