import { useCallback, useMemo } from "react";
import { labelSelection, type TileInfo } from "@biopb/tensor-flight-client";
import { selectLabelOverlay, useAppStore } from "../store";
import { buildLabelLayers } from "../utils/labelLayers";
import { useLabelOverlay } from "./useLabelOverlay";
import { useLoadedPlane } from "./useLoadedPlane";

/**
 * The label set drawn over the image, and whether it holds the plane on screen.
 *
 * The overlay reads the plane the viewer has ASKED for, so its tiles load
 * alongside the image's rather than behind them -- and it is *drawn* only once
 * it holds the plane actually ON SCREEN. Two reads of two tensors land when they
 * land, so for a moment after every plane change one of them has arrived and the
 * other has not; during play the cover is deliberately dropped, and a mask of
 * plane N+1 over plane N's pixels for the whole of a frame is a wrong picture
 * that looks like a right one. This is the same rule the annotations follow
 * through `shownPlane` -- draw the plane on screen -- except that a set has to
 * fetch its plane, so "cannot" means hidden rather than merely different.
 *
 * `ready` is for the play driver, which paces on the image *and* the overlay:
 * paced on the image alone, play advances the moment the image's tiles land, so
 * a set whose read is slower is asked for the next plane before it has finished
 * the last and is out of step for the whole of playback. It fails open, and
 * deliberately: a set that errored, or none at all, must not hold the sequence.
 *
 * The keys are JSON strings because deck.gl refetches on a changed *reference*
 * and `labelSelection` builds a new object per call, so a memo keyed on that
 * object would refetch every frame.
 */
export function useLabelOverlayLayers(
  info: TileInfo,
  selection: Record<string, number>,
  loadedSelection: Record<string, number> | null,
) {
  const client = useAppStore((s) => s.client);
  // Scoped, like the annotation state: a set chosen on the previous image is not
  // this one's overlay, and drawing it here would be worse than drawing nothing.
  const overlayId = useAppStore(selectLabelOverlay);
  const labelOpacity = useAppStore((s) => s.labelOpacity);
  const { overlay, error } = useLabelOverlay(client, overlayId);

  // `useCallback`, not a plain function: the two memos below take it as a
  // dependency, and a fresh identity every render would rebuild them every
  // render -- which is the refetch they exist to avoid.
  const deriveKey = useCallback(
    (sel: Record<string, number> | null) => {
      if (!overlay || !sel) return "";
      return JSON.stringify(labelSelection(info, overlay.info, sel));
    },
    [info, overlay],
  );
  const selectionKey = useMemo(() => deriveKey(selection), [deriveKey, selection]);
  const shownKey = useMemo(() => deriveKey(loadedSelection), [deriveKey, loadedSelection]);

  // "Which plane of which set". The set has to be in the key: two sets of one
  // image produce identical selections, so the one switched off a moment ago
  // would otherwise have its landing counted as this one's.
  const planeKey = (key: string) => (overlay && key ? `${overlay.arrayId}|${key}` : "");
  const { loaded: loadedKey, onViewportLoad } = useLoadedPlane(planeKey(selectionKey));
  // A key from a set that is no longer the overlay can never match, so switching
  // sets hides the old one without a reset to remember.
  const showing = loadedKey !== null && loadedKey === planeKey(shownKey);

  const layers = useMemo(() => {
    // Keyed on the id it was loaded for: `useLabelOverlay` clears its state on a
    // change, so this can only disagree in the harmless direction, and checking
    // it is what makes that a property of the code rather than of the order two
    // effects happen to run in.
    if (!overlay || overlay.arrayId !== overlayId || !selectionKey) return [];
    return buildLabelLayers({
      name: overlay.name,
      sources: overlay.sources,
      selection: JSON.parse(selectionKey) as Record<string, number>,
      opacity: labelOpacity,
      showing,
      onViewportLoad,
    });
  }, [overlay, overlayId, selectionKey, labelOpacity, showing, onViewportLoad]);

  const ready = overlayId === null || error !== null || showing;
  return { layers, ready, error };
}
