/**
 * Where a label set's `array_id` says it belongs.
 *
 * A label set is an ordinary tensor of its image, addressed under a marked
 * segment: `<image array_id>/@labels/<name>` (biopb/biopb#1059). Nothing else in
 * the wire format marks one: the catalog has no `role` column, and a listing's
 * per-tensor entry carries no `metadata_json` to read the `image-label` block
 * out of. The path *is* the statement, so this module is the one place that
 * reads it.
 *
 * Mirror of the server's `core/labels.py::split_label_field`, and deliberately
 * the same rule rather than a looser one: the **last** `@labels` segment with a
 * name after it is the one, whatever the image's own field contains, and a name
 * is slash-free, so anything past it is a native pyramid level.
 *
 * The marker is on the id, not on disk: an OME-Zarr store's own group stays
 * `labels/`. It is what stops an uploaded tensor's id colliding with a native
 * one, since a scene may plausibly be called `labels` and not `@labels`.
 */

import { sliderAxes } from "./tensor-array.js";
import type { TileInfo } from "./types.js";

/** The segment that marks a label set under its image, in a wire id. */
export const LABELS_SEGMENT = "@labels";

/**
 * A name under this prefix is the server's own -- the set rasterized from a
 * file's masks, or a native NGFF set the source came with. Clients read one;
 * they never upload or delete one.
 */
export const RESERVED_LABEL_PREFIX = "@";

/** A label set's `array_id`, taken apart. */
export interface LabelAddress {
  /** The image this set annotates, as an `array_id`. */
  imageArrayId: string;
  /** The set's name, slash-free. */
  name: string;
  /** A native pyramid level addressed under the set, if the id named one. */
  level: string | null;
}

/**
 * Take a label set's `array_id` apart, or null when it names no set.
 *
 * `"src0/@labels/nuclei"` -> image `"src0"`, name `"nuclei"`. Pass a *stable*
 * id: a content-pinned `id@token` is not one, so split it with
 * {@link splitArrayVersion} first.
 */
export function splitLabelArrayId(arrayId: string): LabelAddress | null {
  const parts = arrayId.split("/");
  // From 1: `parts[0]` is the source_id, which is slash-free and cannot be the
  // `@labels` segment of a field. A bare source called "@labels" is a source.
  for (let i = parts.length - 2; i >= 1; i--) {
    if (parts[i] !== LABELS_SEGMENT || !parts[i + 1]) continue;
    return {
      imageArrayId: parts.slice(0, i).join("/"),
      name: parts[i + 1] as string,
      level: parts.length > i + 2 ? parts.slice(i + 2).join("/") : null,
    };
  }
  return null;
}

/** Whether `name` is a set the server owns -- read-only to every client. */
export function isReservedLabelName(name: string): boolean {
  return name.startsWith(RESERVED_LABEL_PREFIX);
}

/**
 * A label set's selection for a plane of its image.
 *
 * Takes the **image's** selection rather than a slice position, because the
 * plane an overlay belongs to is the one on screen rather than the one asked
 * for -- a caller that conflates the two draws a mask of the wrong frame while
 * a read is outstanding.
 *
 * The two tensors do not number their axes alike: a set spans the image's
 * **non-channel** extent (biopb/biopb#1059), so every axis after the image's
 * `c` sits one place to the left in the set. Matching by Viv's selection *key*
 * would therefore be right for named axes and wrong for unnamed ones -- an
 * image's `a3` is the set's `a2`, and reading frame 0 of a timelapse instead
 * of frame 40 is a silently wrong picture rather than a visible failure.
 *
 * So the mapping is **read, not derived**: `TileInfo.image_axes` is the
 * server's own statement of which image axis each of the set's indexes. A
 * server that predates the field leaves it undefined, and the extent rule is
 * re-derived here as the fallback -- the same answer, from the one place that
 * still has to know the rule.
 */
export function labelSelection(
  imageInfo: TileInfo,
  labelInfo: TileInfo,
  imageSelection: Record<string, number>,
): Record<string, number> {
  const stated = labelInfo.image_axes;
  const mapping =
    stated !== undefined && stated.length === labelInfo.shape.length
      ? stated
      : derivedImageAxes(imageInfo, labelInfo);

  const byImageAxis: Record<number, number> = {};
  for (const axis of sliderAxes(imageInfo.dim_labels, imageInfo.shape)) {
    byImageAxis[axis.axis] = imageSelection[axis.key] ?? 0;
  }

  const out: Record<string, number> = {};
  for (const axis of sliderAxes(labelInfo.dim_labels, labelInfo.shape)) {
    const imageAxis = mapping?.[axis.axis];
    const want =
      imageAxis === undefined
        ? (imageSelection[axis.key] ?? 0)
        : (byImageAxis[imageAxis] ?? 0);
    // Clamped against the set's own extent, exactly as `vivSelection` clamps
    // against the image's: the two are equal by the extent rule, and a server
    // that broke it should show the last plane rather than read past the end.
    out[axis.key] = Math.min(Math.max(0, want), Math.max(0, axis.extent - 1));
  }
  return out;
}

/**
 * The extent rule, re-derived: the set's axis *j* is the image's *j*-th
 * non-channel axis. Only for a server that does not state `image_axes`; null
 * when the ranks say the set does not span the image at all, which sends
 * {@link labelSelection} to matching by key -- the best answer left, and exact
 * for an ordinary TZYX set.
 */
function derivedImageAxes(imageInfo: TileInfo, labelInfo: TileInfo): number[] | null {
  const nonChannel = imageInfo.shape
    .map((_, i) => i)
    .filter((i) => i !== imageInfo.selectable.c);
  return nonChannel.length === labelInfo.shape.length ? nonChannel : null;
}
