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
import { selectTileInfo, useAppStore } from "../store";
import { currentPlaneFor, roiSetCounts, setColor, visibleRois } from "../utils/roiLayers";
import type { RoiAnnotation } from "@biopb/tensor-flight-client";

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
  /** `dim_label -> index` for the plane on screen. */
  currentPlane: Record<string, number>;
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

interface RoiPanelProps {
  /** The tensor in view -- the same address the viewer renders. */
  arrayId: string;
}

export function RoiPanel({ arrayId }: RoiPanelProps) {
  const rois = useAppStore((s) => s.rois);
  const roisFor = useAppStore((s) => s.roisFor);
  const loading = useAppStore((s) => s.roisLoading);
  const error = useAppStore((s) => s.roisError);
  const truncated = useAppStore((s) => s.roisTruncated);
  const skipped = useAppStore((s) => s.roisSkipped);
  const unavailable = useAppStore((s) => s.roisUnavailable);
  const showRois = useAppStore((s) => s.showRois);
  const hiddenSets = useAppStore((s) => s.hiddenSets);
  const onToggleOverlay = useAppStore((s) => s.setShowRois);
  const onToggleSet = useAppStore((s) => s.toggleSetHidden);
  const tileInfo = useAppStore(selectTileInfo);
  const slice = useAppStore((s) => s.slice);
  const currentPlane = useMemo(() => currentPlaneFor(tileInfo, slice), [tileInfo, slice]);

  return (
    <RoiPanelView
      // `rois` outlives a tensor switch until the next fetch lands, so counts
      // are shown only once they describe the tensor actually in view.
      rois={roisFor === arrayId ? rois : []}
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
