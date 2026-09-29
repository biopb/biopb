/**
 * A label set drawn over its image, as a deck.gl layer.
 *
 * The overlay is an ordinary Viv image layer over a second pixel source -- the
 * label tensor reads through the same tile route as the image, and the server
 * forces `nearest` on every computed level of a set (`adapters/labels.py`), so
 * ids are never averaged into ids that were never stored. What differs from the
 * image is the colour step, which {@link LabelPaletteExtension} replaces
 * wholesale.
 *
 * How a set's selection lines up with its image's is deliberately not here:
 * that is a property of the pair, not of the drawing, and it lives beside the
 * other identity rules as `labelSelection` in `@biopb/tensor-flight-client`.
 *
 * **The layer id must contain Viv's view id** -- see {@link vivLayerId}.
 */

import { DETAIL_VIEW_ID, ImageLayer, MultiscaleImageLayer } from "@hms-dbmi/viv";
import { LABEL_CONTRAST_LIMITS, LabelPaletteExtension } from "./labelPalette";
import { vivLayerId } from "./vivUtils";

/** A layer id Viv's `layerFilter` will accept; see {@link vivLayerId}. */
export function labelLayerId(name: string): string {
  return vivLayerId("labels", name);
}

/**
 * Module-level for the reason `VIV_EXTENSIONS` is: deck.gl reads a new
 * `extensions` array as a change of extensions and recompiles every shader, so
 * a fresh instance per render would rebuild the overlay on every slider move.
 */
const LABEL_EXTENSIONS = [new LabelPaletteExtension()];

export interface LabelLayerOptions {
  /** The set's name, for the layer id -- one layer per set drawn. */
  name: string;
  /** The label tensor's pixel sources, finest first. */
  sources: unknown[];
  /** {@link labelSelection}'s answer for the plane the viewer has asked for. */
  selection: Record<string, number>;
  /** 0-1. The image under the overlay is the point, so this is never 1 by default. */
  opacity: number;
  /**
   * Whether the tiles this layer is holding are for the plane **on screen**.
   *
   * False while it is catching up: its read and the image's are independent, so
   * for a moment after every plane change one of the two has landed and the
   * other has not. A mask drawn then is a mask of a different plane, which
   * during play is a wrong picture that looks like a right one -- so it is held
   * back until it agrees with the pixels underneath.
   */
  showing: boolean;
  /**
   * Called when this layer's viewport is fully loaded, exactly as the image's
   * own is: `MultiscaleImageLayer` pins its background layer's copy to null, so
   * this fires once per completed viewport. What {@link LabelLayerOptions.showing}
   * is decided from.
   */
  onViewportLoad: (loaded?: unknown) => void;
}

/**
 * The overlay's layer, or `[]` when there is nothing to draw.
 *
 * Returned as an array so the caller can splice it into the overlay list
 * without a null check, the way the ROI builders are used.
 */
export function buildLabelLayers(options: LabelLayerOptions | null): unknown[] {
  if (!options || options.sources.length === 0) return [];
  const { name, sources, selection, opacity, showing, onViewportLoad } = options;
  // Viv's own `getImageLayer` chooses between the two the same way: a single
  // level has no tile grid to walk, and `MultiscaleImageLayer` takes the whole
  // pyramid where `ImageLayer` takes one source.
  const multiscale = sources.length > 1;
  const Layer = (multiscale ? MultiscaleImageLayer : ImageLayer) as new (
    props: Record<string, unknown>,
  ) => unknown;
  return [
    new Layer({
      id: labelLayerId(name),
      viewportId: DETAIL_VIEW_ID,
      loader: multiscale ? sources : sources[0],
      selections: [selection],
      // The identity ramp, not a window -- see `LABEL_CONTRAST_LIMITS`.
      contrastLimits: [LABEL_CONTRAST_LIMITS],
      channelsVisible: [true],
      extensions: LABEL_EXTENSIONS,
      // Held back by alpha rather than by `visible` or by dropping the layer:
      // an out-of-step overlay is one that is *loading*, and it has to keep
      // loading to catch up. Unmounting it would restart the read, and deck.gl's
      // `visible: false` skips the draw without promising the tileset still
      // updates -- which would be a layer that can never become visible again.
      opacity: showing ? opacity : 0,
      onViewportLoad,
      // Already the default, said out loud because it is the property that
      // makes the overlay correct rather than merely pretty: a linear filter
      // would blend two ids into a third that names no object.
      interpolation: "nearest",
      // Nothing on the overlay is clickable, and leaving it pickable would put
      // it between the pointer and the ROI surface underneath -- deck reports
      // the topmost picked layer, and a click that places a vertex needs to
      // arrive with none.
      pickable: false,
    }),
  ];
}
