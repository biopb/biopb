/**
 * A label set drawn over its image, as a deck.gl layer.
 *
 * The overlay is an ordinary Viv image layer over a second pixel source -- the
 * label tensor reads through the same tile route as the image, and the server
 * forces `nearest` on every computed level of a set (`adapters/labels.py`), so
 * ids are never averaged into ids that were never stored. What differs from the
 * image is the colour step, which {@link LabelPaletteExtension} replaces
 * wholesale, and the alignment of the selection, which {@link labelSelection}
 * answers.
 *
 * **The layer id must contain Viv's view id**, for the reason `roiLayers.ts`
 * states: `VivViewer` filters every layer through
 * `layer.id.includes(getVivId(viewport.id))`, and a layer that fails it is
 * silently never drawn.
 */

import { DETAIL_VIEW_ID, ImageLayer, MultiscaleImageLayer } from "@hms-dbmi/viv";
import { sliderAxes, type TileInfo } from "@biopb/tensor-flight-client";
import { LABEL_CONTRAST_LIMITS, LabelPaletteExtension } from "./labelPalette";
import { vivSelection, type SliceIndices } from "./vivUtils";

/** A layer id Viv's `layerFilter` will accept; see `roiLayers.ts::roiLayerId`. */
export function labelLayerId(name: string): string {
  return `labels-${name}-#${DETAIL_VIEW_ID}#`;
}

/**
 * Module-level for the reason `VIV_EXTENSIONS` is: deck.gl reads a new
 * `extensions` array as a change of extensions and recompiles every shader, so
 * a fresh instance per render would rebuild the overlay on every slider move.
 */
const LABEL_EXTENSIONS = [new LabelPaletteExtension()];

/**
 * The label tensor's selection for the plane the image is showing.
 *
 * The two tensors do not share an axis numbering: a set spans the image's
 * **non-channel** extent (biopb/biopb#1059), so every axis after the image's
 * `c` sits one place to the left in the set. Matching by Viv's selection *key*
 * would therefore be right for named axes and wrong for unnamed ones -- an
 * image's `a3` is the set's `a2`, and reading frame 0 of a timelapse instead of
 * frame 40 is a silently wrong picture rather than a visible failure.
 *
 * So the axes are matched positionally, through the one rule that relates them:
 * the set's axis `j` is the image's `j`-th non-channel axis. A set that does not
 * satisfy the rule (a server that listed something else under `labels/`) falls
 * back to matching by name, which is the best answer available and is exact for
 * an ordinary TZYX set.
 */
export function labelSelection(
  imageInfo: TileInfo,
  labelInfo: TileInfo,
  slice: SliceIndices,
): Record<string, number> {
  const nonChannel = imageInfo.shape
    .map((_, i) => i)
    .filter((i) => i !== imageInfo.selectable.c);
  if (nonChannel.length !== labelInfo.shape.length) {
    return vivSelection(labelInfo, slice);
  }

  const chosen = vivSelection(imageInfo, slice);
  const byImageAxis: Record<number, number> = {};
  for (const axis of sliderAxes(imageInfo.dim_labels, imageInfo.shape)) {
    byImageAxis[axis.axis] = chosen[axis.key] ?? 0;
  }

  const out: Record<string, number> = {};
  for (const axis of sliderAxes(labelInfo.dim_labels, labelInfo.shape)) {
    const imageAxis = nonChannel[axis.axis];
    const want = imageAxis === undefined ? 0 : (byImageAxis[imageAxis] ?? 0);
    // Clamped against the set's own extent, exactly as `vivSelection` clamps
    // against the image's: the two are equal by the extent rule, and a server
    // that broke it should show the last plane rather than fetch past the end.
    out[axis.key] = Math.min(Math.max(0, want), Math.max(0, axis.extent - 1));
  }
  return out;
}

export interface LabelLayerOptions {
  /** The set's name, for the layer id -- one layer per set drawn. */
  name: string;
  /** The label tensor's pixel sources, finest first. */
  sources: unknown[];
  /** {@link labelSelection}'s answer for the plane on screen. */
  selection: Record<string, number>;
  /** 0-1. The image under the overlay is the point, so this is never 1 by default. */
  opacity: number;
}

/**
 * The overlay's layer, or `[]` when there is nothing to draw.
 *
 * Returned as an array so the caller can splice it into the overlay list
 * without a null check, the way the ROI builders are used.
 */
export function buildLabelLayers(options: LabelLayerOptions | null): unknown[] {
  if (!options || options.sources.length === 0) return [];
  const { name, sources, selection, opacity } = options;
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
      opacity,
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
