"use client";

/**
 * The annotations panel: what this tensor holds, and what is on screen.
 *
 * Read-only for now -- it names the sets, counts them, and switches them on and
 * off. Drawing tools land in the next phase (docs/roi-annotations-ui.md).
 *
 * Rendered only alongside the 2-D viewer. In volume mode there is no overlay to
 * govern, so a list of annotations drawn nowhere would be a control over
 * nothing -- and keeping the panel out of that tree is also what keeps the "3-D
 * fetches nothing" property true, since this and the overlay are the only two
 * callers of `loadRois`.
 *
 * Split into a presentational view and a store container because the workspace
 * renders components through `renderToStaticMarkup` to test them, and zustand's
 * server snapshot is `getInitialState()` -- a store-reading component always
 * renders its defaults there, whatever the test set. The view takes props and
 * is therefore actually testable.
 */

import { useMemo } from "react";
import {
  selectBroadcastAxes,
  selectHiddenSets,
  selectRois,
  selectRoisError,
  selectRoisLoading,
  selectRoisSkipped,
  selectRoisTruncated,
  selectSelectedRoi,
  selectTileInfo,
  useAppStore,
} from "../store";
import {
  currentPlaneFor,
  defaultBroadcastAxes,
  roiSetCounts,
  setColor,
  visibleRois,
} from "../utils/roiLayers";
import { pinnableAxes, roiVisibleOnPlane } from "@biopb/tensor-flight-client";
import type { RoiAnnotation, SliderAxis } from "@biopb/tensor-flight-client";

function swatch(setName: string) {
  const [r, g, b] = setColor(setName);
  return {
    background: `rgb(${r} ${g} ${b} / 0.35)`,
    border: `1px solid rgb(${r} ${g} ${b})`,
    borderRadius: 2,
    display: "inline-block",
    height: 10,
    marginRight: 6,
    width: 10,
  };
}

export interface RoiPanelViewProps {
  /** This tensor's annotations. The caller has already checked they are its own. */
  rois: RoiAnnotation[];
  /** `axis -> index` for the plane on screen. */
  currentPlane: Record<number, number>;
  hiddenSets: string[];
  showRois: boolean;
  loading: boolean;
  error: string | null;
  /** The per-tensor cap clipped the set: what is shown is not all there is. */
  truncated: boolean;
  /** Rows whose geometry this client could not read. */
  skipped: number;
  /** The server does not offer annotations at all -- render nothing. */
  unavailable: boolean;
  onToggleOverlay: (value: boolean) => void;
  onToggleSet: (setName: string) => void;
}

export function RoiPanelView({
  rois,
  currentPlane,
  hiddenSets,
  showRois,
  loading,
  error,
  truncated,
  skipped,
  unavailable,
  onToggleOverlay,
  onToggleSet,
}: RoiPanelViewProps) {
  // A server without annotations is not a fault to report: say nothing at all
  // rather than show a permanently empty panel.
  if (unavailable) return null;

  const sets = roiSetCounts(rois);
  const onPlane = visibleRois(rois, currentPlane, hiddenSets).length;

  return (
    <section className="roi-panel">
      <header className="roi-panel-head">
        <label>
          <input
            type="checkbox"
            checked={showRois}
            onChange={(e) => onToggleOverlay(e.target.checked)}
          />{" "}
          Annotations
        </label>
        <span className="roi-count">
          {loading && rois.length === 0
            ? "loading…"
            : rois.length === 0
              ? "none"
              : `${onPlane} of ${rois.length} here`}
        </span>
      </header>

      {sets.length > 0 && (
        <ul className="roi-sets">
          {sets.map(({ setName, count }) => {
            const hidden = hiddenSets.includes(setName);
            return (
              <li key={setName}>
                <label style={{ opacity: hidden ? 0.45 : 1 }}>
                  <input type="checkbox" checked={!hidden} onChange={() => onToggleSet(setName)} />{" "}
                  <span style={swatch(setName)} />
                  {setName}
                </label>
                <span className="roi-count">{count}</span>
              </li>
            );
          })}
        </ul>
      )}

      {truncated && (
        <p className="roi-note">
          The server returned its per-tensor maximum — this tensor holds more
          annotations than are shown.
        </p>
      )}
      {skipped > 0 && (
        <p className="roi-note">
          {skipped} annotation{skipped === 1 ? "" : "s"} could not be read by this
          viewer and {skipped === 1 ? "is" : "are"} not drawn.
        </p>
      )}
      {error && <p className="roi-note roi-error">Could not load annotations — {error}</p>}
    </section>
  );
}

/**
 * Everything here comes through a scoping selector, so state belonging to a
 * tensor that is no longer in view cannot reach the panel -- including the
 * warnings, which are the ones that would be believed: a stale "per-tensor
 * maximum" reads as a fact about the image on screen.
 */
export function RoiPanel() {
  const rois = useAppStore(selectRois);
  const loading = useAppStore(selectRoisLoading);
  const error = useAppStore(selectRoisError);
  const truncated = useAppStore(selectRoisTruncated);
  const skipped = useAppStore(selectRoisSkipped);
  const hiddenSets = useAppStore(selectHiddenSets);
  const unavailable = useAppStore((s) => s.roisUnavailable);
  const showRois = useAppStore((s) => s.showRois);
  const onToggleOverlay = useAppStore((s) => s.setShowRois);
  const onToggleSet = useAppStore((s) => s.toggleSetHidden);
  const tileInfo = useAppStore(selectTileInfo);
  const slice = useAppStore((s) => s.slice);
  const currentPlane = useMemo(() => currentPlaneFor(tileInfo, slice), [tileInfo, slice]);

  return (
    <RoiPanelView
      rois={rois}
      currentPlane={currentPlane}
      hiddenSets={hiddenSets}
      showRois={showRois}
      loading={loading}
      error={error}
      truncated={truncated}
      skipped={skipped}
      unavailable={unavailable}
      onToggleOverlay={onToggleOverlay}
      onToggleSet={onToggleSet}
    />
  );
}

// ---------------------------------------------------------------------------
// Authoring
// ---------------------------------------------------------------------------

export interface RoiAuthorViewProps {
  newLabel: string;
  newSetName: string;
  /** Every axis a new annotation could pin, with what to call it. */
  axes: SliderAxis[];
  /** Axes it will NOT pin, i.e. broadcast across. */
  broadcastAxes: number[];
  /** `axis -> index` for the plane on screen, so each row can show its value. */
  currentPlane: Record<number, number>;
  selected: RoiAnnotation | null;
  writeError: string | null;
  onSetNewLabel: (label: string) => void;
  onSetNewSetName: (setName: string) => void;
  onToggleBroadcast: (axis: number) => void;
  onDeleteSelected: () => void;
}

/**
 * What a new annotation will be, and what the selected one is.
 *
 * The pin rows are the load-bearing part. Broadcast is not something the user
 * sets, it is an axis the pin omits -- and a broadcast annotation is
 * pixel-identical to a pinned one on the plane where they coincide, so without
 * this the only way to tell them apart is to scrub and see what follows.
 */
export function RoiAuthorView({
  newLabel,
  newSetName,
  axes,
  broadcastAxes,
  currentPlane,
  selected,
  writeError,
  onSetNewLabel,
  onSetNewSetName,
  onToggleBroadcast,
  onDeleteSelected,
}: RoiAuthorViewProps) {
  const broadcast = new Set(broadcastAxes);
  // Only a selection that is actually drawn. The Delete button acts on it, and
  // a plane change must not leave the user able to delete a shape they cannot
  // see -- the selection itself survives, so scrubbing back brings it into
  // reach again.
  const shown = selected && roiVisibleOnPlane(selected.plane, currentPlane) ? selected : null;
  return (
    <div className="roi-author">
      <label className="roi-field">
        <span>Label</span>
        <input value={newLabel} placeholder="cell" onChange={(e) => onSetNewLabel(e.target.value)} />
      </label>
      <label className="roi-field">
        <span>Set</span>
        <input
          value={newSetName}
          placeholder="default"
          onChange={(e) => onSetNewSetName(e.target.value)}
        />
      </label>

      {axes.length > 0 && (
        <>
          <div className="roi-field-head">New annotations apply to</div>
          <ul className="roi-sets">
            {axes.map((axis) => {
              const all = broadcast.has(axis.axis);
              return (
                <li key={axis.axis}>
                  <span>{axis.title}</span>
                  <button
                    type="button"
                    className="roi-pin-toggle"
                    aria-pressed={all}
                    title={
                      all
                        ? `Every index of ${axis.title}`
                        : `Only ${axis.title} ${currentPlane[axis.axis] ?? 0}`
                    }
                    onClick={() => onToggleBroadcast(axis.axis)}
                  >
                    {all ? "all" : `${currentPlane[axis.axis] ?? 0}`}
                  </button>
                </li>
              );
            })}
          </ul>
        </>
      )}

      {shown && (
        <div className="roi-selected">
          <div className="roi-field-head">Selected</div>
          <div>
            {shown.label || <em>no label</em>} — {shown.geometry.kind} in {shown.setName}
          </div>
          <div className="roi-count">
            {Object.keys(shown.plane).length === 0
              ? "on every plane"
              : `pinned: ${Object.entries(shown.plane)
                  .map(([axis, index]) => `${axisTitle(axes, Number(axis))} ${index}`)
                  .join(", ")}`}
          </div>
          <button type="button" className="roi-tool-text" onClick={onDeleteSelected}>
            Delete
          </button>
        </div>
      )}

      {writeError && <p className="roi-note roi-error">{writeError}</p>}
    </div>
  );
}

/** An axis's display name, falling back to its position when the grid is gone. */
function axisTitle(axes: SliderAxis[], axis: number): string {
  return axes.find((a) => a.axis === axis)?.title ?? `axis ${axis}`;
}

export function RoiAuthor() {
  const tileInfo = useAppStore(selectTileInfo);
  const slice = useAppStore((s) => s.slice);
  const newLabel = useAppStore((s) => s.newLabel);
  const newSetName = useAppStore((s) => s.newSetName);
  const selected = useAppStore(selectSelectedRoi);
  const writeError = useAppStore((s) => s.roiWriteError);
  const onSetNewLabel = useAppStore((s) => s.setNewLabel);
  const onSetNewSetName = useAppStore((s) => s.setNewSetName);
  const toggleBroadcastAxis = useAppStore((s) => s.toggleBroadcastAxis);
  const deleteRoi = useAppStore((s) => s.deleteRoi);
  const unavailable = useAppStore((s) => s.roisUnavailable);
  const showRois = useAppStore((s) => s.showRois);

  const defaults = useMemo(() => defaultBroadcastAxes(tileInfo), [tileInfo]);
  const broadcastAxes = useAppStore((s) => selectBroadcastAxes(s, defaults));
  const axes = useMemo(() => (tileInfo ? pinnableAxes(tileInfo) : []), [tileInfo]);
  const currentPlane = useMemo(() => currentPlaneFor(tileInfo, slice), [tileInfo, slice]);

  // With the overlay off there is nothing to author against and nothing drawn
  // to act on -- including a Delete button for a selection the user cannot see.
  if (unavailable || !showRois) return null;

  return (
    <RoiAuthorView
      newLabel={newLabel}
      newSetName={newSetName}
      axes={axes}
      broadcastAxes={broadcastAxes}
      currentPlane={currentPlane}
      selected={selected}
      writeError={writeError}
      onSetNewLabel={onSetNewLabel}
      onSetNewSetName={onSetNewSetName}
      onToggleBroadcast={(axis) => toggleBroadcastAxis(axis, defaults)}
      onDeleteSelected={() => selected && void deleteRoi(selected.roiId)}
    />
  );
}
