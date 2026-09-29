import { useEffect, useState } from "react";
import { BADGE, greyLevel } from "./viewerStyles";

/** What the pointer is over: image coordinates and, once read, the value. */
export interface HoverSample {
  x: number;
  y: number;
  /** Null until Viv reads it out of a loaded tile. */
  value: number | null;
  /** Reduction of the level the value came from; 1 is full resolution. */
  scale: number;
}

/**
 * The value under the pointer, in its own component.
 *
 * Its own so that a pointer move repaints one badge rather than re-rendering
 * the viewer around the deck.gl stage. The parent hands it a sink through
 * `bind` and never holds the sample itself.
 */
export function HoverReadout({
  bind,
}: {
  bind: (sink: ((sample: HoverSample | null) => void) | null) => void;
}) {
  const [sample, setSample] = useState<HoverSample | null>(null);
  useEffect(() => {
    bind(setSample);
    return () => bind(null);
  }, [bind]);
  if (!sample) return null;
  return (
    <div
      style={{ ...BADGE, position: "static" }}
      title={
        sample.scale > 1
          ? "Read from the pyramid level on screen, not from the full-resolution pixel. Zoom in for the pixel's own value."
          : "The pixel under the pointer, read from the tile already on screen."
      }
    >
      {/* The coordinate is always there; the value is not, and saying which is
          missing is the difference between "no tile here yet" and "zero". */}
      ({sample.x}, {sample.y}){" "}
      {sample.value === null ? "—" : greyLevel(sample.value)}
      {sample.scale > 1 && ` · at 1/${sample.scale}`}
    </div>
  );
}
