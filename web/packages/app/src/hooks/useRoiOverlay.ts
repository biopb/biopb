import { useEffect, useMemo } from "react";
import type { TileInfo } from "@biopb/tensor-flight-client";
import {
  selectRoiScopes,
  selectRois,
  selectSelectedRoiId,
  selectVisibleSets,
  useAppStore,
} from "../store";
import {
  buildRoiLayers,
  buildSelectionLayers,
  planeFromSelection,
  visibleRois,
} from "../utils/roiLayers";

/**
 * The stored annotations over the plane on screen: which are drawn, and their
 * layers.
 *
 * Fetched from here rather than from the store's tensor-change path, so the 3-D
 * viewer never pays for a set it cannot draw: the 2-D viewer does not exist in
 * volume mode. `loadRois` is idempotent on what has landed, which is what keeps
 * a 2-D -> 3-D -> 2-D round trip (a full remount, see ViewerPane's key) from
 * refetching. Re-run on the visible sets, because a server-owned set is fetched
 * only once it is switched on; and on the landed scopes, because a landing is
 * what tells `loadRois` which named sets exist -- and a landing can also
 * un-land a scope whose rows went stale, which is fetched again from here.
 *
 * The plane the overlay is drawn for is the one ON SCREEN (`loadedSelection`),
 * not the one asked for. They are the same except while a read is outstanding --
 * and during play the cover is deliberately dropped, so the stale plane stays
 * visible while `position` has already moved on. Driving the overlay from
 * `position` there would put plane N+1's annotations over plane N's pixels for
 * the whole of playback: a systematic off-by-one, not a flicker. Gating on
 * `dataValid` instead would strobe: the play driver paces on exactly that flag,
 * so it toggles ~10 times a second while playing.
 */
export function useRoiOverlay(info: TileInfo, loadedSelection: Record<string, number> | null) {
  const client = useAppStore((s) => s.client);
  const loadRois = useAppStore((s) => s.loadRois);
  const key = useAppStore((s) => s.target.key);
  // Scoped selectors, not raw fields: a set held for another tensor is not this
  // one's, and drawing it over a new image would be worse than drawing nothing.
  const rois = useAppStore(selectRois);
  const showRois = useAppStore((s) => s.showRois);
  const visibleSets = useAppStore(selectVisibleSets);
  const roiScopes = useAppStore(selectRoiScopes);
  const selectedRoiId = useAppStore(selectSelectedRoiId);

  useEffect(() => {
    if (!client || !key) return;
    void loadRois();
  }, [client, key, loadRois, visibleSets, roiScopes]);

  const shownPlane = useMemo(
    () => (loadedSelection === null ? null : planeFromSelection(info, loadedSelection)),
    [info, loadedSelection],
  );

  // Selection is deliberately not a dependency: it would change this memo's
  // output identity, and deck.gl answers a changed `data` by regenerating every
  // attribute of every layer -- re-tessellating the whole set for a click. The
  // emphasis is its own layer, over one annotation, below.
  const roiLayers = useMemo(
    () =>
      buildRoiLayers({
        rois,
        currentPlane: shownPlane ?? {},
        visibleSets,
        // Nothing has landed yet, so no plane is on screen to annotate. An
        // empty pin would match every unpinned annotation and draw them over a
        // frame that is not there.
        visible: showRois && shownPlane !== null,
      }),
    [rois, shownPlane, visibleSets, showRois],
  );

  // What a click can hit: exactly what is drawn, so selection cannot pick an
  // annotation the user cannot see.
  const shown = useMemo(
    () => (showRois ? visibleRois(rois, shownPlane ?? {}, visibleSets) : []),
    [rois, shownPlane, visibleSets, showRois],
  );

  // Only what is drawn can be emphasised, which is the same condition the panel
  // and the Delete key use -- so a selection the plane has moved off is not
  // marked on a shape that is not there.
  const selectionLayers = useMemo(
    () => buildSelectionLayers(shown.find((roi) => roi.roiId === selectedRoiId) ?? null),
    [shown, selectedRoiId],
  );

  return { shownPlane, shown, roiLayers, selectionLayers };
}
