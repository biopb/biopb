"use client";

/**
 * The label overlay's controls: which set is drawn, and how strongly.
 *
 * *Which* set is chosen in the source tree, where the sets are listed under
 * their image -- so this panel does not repeat the list. What it owns is the one
 * thing the tree cannot say: how much of the image below the overlay is left
 * visible, and the way back to none.
 *
 * Rendered only when a set is actually drawn. A permanently present panel over
 * an image that has no label sets would be a control over nothing, which is the
 * same rule `RoiPanel` follows for a server without annotations.
 *
 * Split into a presentational view and a store container for the reason
 * `RoiPanel` is: zustand's server snapshot is `getInitialState()`, so a
 * store-reading component renders its defaults under `renderToStaticMarkup` and
 * is not testable there.
 */

import { selectLabelOverlay, useAppStore } from "../store";
import { splitLabelArrayId } from "@biopb/tensor-flight-client";

export interface LabelPanelViewProps {
  /** The set drawn over the tensor in view, as its whole `array_id`, or null. */
  arrayId: string | null;
  /** 0-1. */
  opacity: number;
  onOpacity: (value: number) => void;
  onHide: () => void;
}

export function LabelPanelView({ arrayId, opacity, onOpacity, onHide }: LabelPanelViewProps) {
  if (!arrayId) return null;
  // The name, not the whole address: the panel sits under the image the set
  // belongs to, so repeating the image's id says nothing. The title carries it.
  const name = splitLabelArrayId(arrayId)?.name ?? arrayId;
  const percent = Math.round(opacity * 100);

  return (
    <section className="label-panel">
      <header className="label-panel-head">
        <span className="label-panel-name" title={arrayId}>
          Labels: {name}
        </span>
        <button type="button" className="label-hide" onClick={onHide}>
          Hide
        </button>
      </header>
      <label className="label-opacity">
        <span>Opacity</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={opacity}
          aria-label="Label overlay opacity"
          onChange={(e) => onOpacity(Number(e.target.value))}
        />
        <span className="label-opacity-value">{percent}%</span>
      </label>
    </section>
  );
}

export function LabelPanel() {
  // Scoped: a set chosen on the previous image is not this image's overlay, and
  // the viewer is not drawing it either.
  const arrayId = useAppStore(selectLabelOverlay);
  const opacity = useAppStore((s) => s.labelOpacity);
  const onOpacity = useAppStore((s) => s.setLabelOpacity);
  const setLabelOverlay = useAppStore((s) => s.setLabelOverlay);

  return (
    <LabelPanelView
      arrayId={arrayId}
      opacity={opacity}
      onOpacity={onOpacity}
      onHide={() => setLabelOverlay(null)}
    />
  );
}
