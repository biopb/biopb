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
 * A set has the image's axes at the image's lengths, with the channel axis a
 * singleton and an RGB samples axis left out (biopb/biopb#1059), so its axis *j*
 * is the image's axis *j* among those that remain, and the two align by
 * position. Matching by Viv's selection *key* would be wrong for an unnamed
 * axis -- an image's `a3` is the set's `a2` once the samples axis is gone --
 * and reading frame 0 of a timelapse instead of frame 40 is a silently wrong
 * picture rather than a visible failure.
 */
export function labelSelection(
  imageInfo: TileInfo,
  labelInfo: TileInfo,
  imageSelection: Record<string, number>,
): Record<string, number> {
  // The image's axes a set has: all but the samples axis, in order.
  const imageAxes: number[] = [];
  for (let i = 0; i < imageInfo.shape.length; i++) {
    if (i !== imageInfo.plane.s) imageAxes.push(i);
  }

  const byImageAxis: Record<number, number> = {};
  for (const axis of sliderAxes(imageInfo.dim_labels, imageInfo.shape)) {
    byImageAxis[axis.axis] = imageSelection[axis.key] ?? 0;
  }

  const out: Record<string, number> = {};
  for (const axis of sliderAxes(labelInfo.dim_labels, labelInfo.shape)) {
    const want = byImageAxis[imageAxes[axis.axis] ?? -1] ?? 0;
    // Clamped against the set's own extent, exactly as `vivSelection` clamps
    // against the image's: the two are equal by the extent rule, and a server
    // that broke it should show the last plane rather than read past the end.
    out[axis.key] = Math.min(Math.max(0, want), Math.max(0, axis.extent - 1));
  }
  return out;
}
