import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TileInfo } from "@biopb/tensor-flight-client";
import type { RoiAnnotation } from "@biopb/tensor-flight-client";
import { selectBroadcastAxes, selectDraft, selectSelectedRoiId, useAppStore } from "../store";
import {
  closeDraft,
  closesOnFirstVertex,
  isCompletable,
  isRealClick,
  isTextEntryTarget,
  placePoint,
  undoPoint,
} from "../utils/roiDraft";
import { roiAt } from "../utils/roiHitTest";
import { buildDraftLayers, defaultBroadcastAxes, pinForNewRoi, type XY } from "../utils/roiLayers";

/**
 * Drawing, selecting and deleting annotations on the plane on screen.
 *
 * `shownPlane` pins a new annotation and `shown` is what a click can hit; both
 * come from {@link useRoiOverlay}. Returns the click handler for the deck, the
 * draft's layers, and `draftCursorSinkRef`, the sink the hover handler feeds
 * with the pointer while a shape is open (null otherwise, so an idle viewer
 * pays nothing for it).
 */
export function useRoiAuthoring(
  info: TileInfo,
  shownPlane: Record<number, number> | null,
  shown: RoiAnnotation[],
) {
  const showRois = useAppStore((s) => s.showRois);
  const tool = useAppStore((s) => s.tool);
  const draft = useAppStore(selectDraft);
  const selectedRoiId = useAppStore(selectSelectedRoiId);
  const setDraft = useAppStore((s) => s.setDraft);
  const setSelectedRoi = useAppStore((s) => s.setSelectedRoi);
  const createRoi = useAppStore((s) => s.createRoi);
  const deleteRoi = useAppStore((s) => s.deleteRoi);
  const polylineWidth = useAppStore((s) => s.newPolylineWidth);

  // Where the pointer is, for the segment trailing an in-progress shape. State,
  // not a ref, because a deck.gl layer reads it -- but written only while a
  // draft is open.
  const [draftCursor, setDraftCursor] = useState<XY | null>(null);
  const draftCursorSinkRef = useRef<((at: XY | null) => void) | null>(null);
  const draftRef = useRef(draft);
  draftRef.current = draft;
  useEffect(() => {
    if (!draft) {
      setDraftCursor(null);
      draftCursorSinkRef.current = null;
      return;
    }
    draftCursorSinkRef.current = setDraftCursor;
    return () => {
      draftCursorSinkRef.current = null;
    };
  }, [draft]);

  // Memoised because the selector hands `defaults` straight back when the store
  // holds no per-tensor choice: computing it inside the selector would return a
  // new array on every snapshot read, which zustand v5 reads as a changed slice
  // and React turns into an unbounded re-render.
  const broadcastDefaults = useMemo(() => defaultBroadcastAxes(info), [info]);
  const broadcastAxes = useAppStore((s) => selectBroadcastAxes(s, broadcastDefaults));

  const finishDraft = useCallback(() => {
    const geometry = closeDraft(draftRef.current, polylineWidth);
    if (!geometry) return;
    setDraft(null);
    void createRoi(geometry, pinForNewRoi(info, shownPlane ?? {}, broadcastAxes));
  }, [setDraft, createRoi, info, shownPlane, broadcastAxes, polylineWidth]);

  const onDeckClick = useCallback(
    (
      info_: { coordinate?: number[]; viewport?: { zoom?: number | number[] } },
      event?: { type?: string },
    ) => {
      // deck reports the second tap of a double click twice -- once as a click,
      // once as a dblclick -- and routes both here.
      if (!isRealClick(event)) return;
      // Nothing is drawn, so nothing is placeable or selectable: the toggle
      // turns the surface off rather than only hiding what is stored.
      if (!showRois) return;
      const c = info_?.coordinate;
      if (!c || c[0] === undefined || c[1] === undefined) return;
      const at: XY = [c[0], c[1]];
      // World units per screen pixel, so a tolerance stays constant on screen.
      // Read off the viewport the click came through rather than the store's
      // mirrored camera, which trails a gesture by CAMERA_MIRROR_MS.
      const z = info_.viewport?.zoom;
      const zoom = Array.isArray(z) ? (z[0] ?? 0) : (z ?? 0);
      const scale = 2 ** -zoom;

      if (tool === "select") {
        setSelectedRoi(roiAt(shown, at, scale)?.roiId ?? null);
        return;
      }
      // Clicking the first vertex closes the shape -- the one affordance saying
      // an open-ended draft can end without reaching for the keyboard.
      if (closesOnFirstVertex(draftRef.current, at, scale)) {
        finishDraft();
        return;
      }
      const { draft: next, completed } = placePoint(draftRef.current, tool, at);
      setDraft(next);
      if (completed) {
        void createRoi(completed, pinForNewRoi(info, shownPlane ?? {}, broadcastAxes));
      }
    },
    [
      tool,
      shown,
      showRois,
      setSelectedRoi,
      finishDraft,
      setDraft,
      createRoi,
      info,
      shownPlane,
      broadcastAxes,
    ],
  );

  // Enter finishes, Escape abandons, Backspace takes back the last vertex.
  // Bound only while a draft is open, so the viewer never swallows a key it has
  // no use for.
  useEffect(() => {
    if (!draft) return;
    const onKey = (event: KeyboardEvent) => {
      // Mid-composition an IME owns Enter and Backspace -- committing a
      // candidate is not finishing a polygon. `keyCode === 229` is the same
      // state in browsers that do not set `isComposing` on keydown.
      if (event.isComposing || event.keyCode === 229) return;
      // Nor are they ours while someone is typing. These are window-level, so
      // they reach the source search and every other field on the route.
      if (isTextEntryTarget(event.target)) return;
      if (event.key === "Enter") {
        event.preventDefault();
        finishDraft();
      } else if (event.key === "Escape") {
        event.preventDefault();
        setDraft(null);
      } else if (event.key === "Backspace") {
        event.preventDefault();
        setDraft(undoPoint(draftRef.current));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [draft, finishDraft, setDraft]);

  // Delete removes the selection -- the only way to, now that the panel says
  // what is selected on one line and has no button.
  //
  // Bound only while the selection is one of the shapes actually drawn: the
  // panel refuses to act on a selection the plane has moved off, and a key that
  // ignored that would be the way around it.
  //
  // Delete alone, not Backspace: finishing a shape selects it, so Backspace
  // would mean "take back the last vertex" and "delete the whole annotation"
  // one keystroke apart.
  useEffect(() => {
    if (draft || !selectedRoiId) return;
    if (!shown.some((roi) => roi.roiId === selectedRoiId)) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.isComposing || event.keyCode === 229) return;
      if (isTextEntryTarget(event.target)) return;
      if (event.key !== "Delete") return;
      event.preventDefault();
      void deleteRoi(selectedRoiId);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [draft, selectedRoiId, shown, deleteRoi]);

  const draftLayers = useMemo(
    () =>
      buildDraftLayers({
        draft,
        cursor: draftCursor,
        closeable: isCompletable(draft),
        polylineWidth,
      }),
    [draft, draftCursor, polylineWidth],
  );

  return { draft, draftLayers, onDeckClick, finishDraft, draftCursorSinkRef };
}
