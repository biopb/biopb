/**
 * Where a label set's `array_id` says it belongs.
 *
 * A label set is an ordinary tensor of its image, addressed by the NGFF layout
 * `<image array_id>/labels/<name>` (biopb/biopb#1059). Nothing else in the wire
 * format marks one: the catalog has no `role` column, and a listing's per-tensor
 * entry carries no `metadata_json` to read the `image-label` block out of. The
 * path *is* the statement, so this module is the one place that reads it.
 *
 * Mirror of the server's `core/labels.py::split_label_field`, and deliberately
 * the same rule rather than a looser one: the **last** `labels` segment with a
 * name after it is the one, whatever the image's own field contains, and a name
 * is slash-free, so anything past it is a native pyramid level.
 */

/** The path segment that marks a label set under its image. */
export const LABELS_SEGMENT = "labels";

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
 * `"src0/labels/nuclei"` -> image `"src0"`, name `"nuclei"`. Pass a *stable*
 * id: a content-pinned `id@token` is not one, so split it with
 * {@link splitArrayVersion} first.
 */
export function splitLabelArrayId(arrayId: string): LabelAddress | null {
  const parts = arrayId.split("/");
  // From 1: `parts[0]` is the source_id, which is slash-free and cannot be the
  // `labels` segment of a field. A bare source called "labels" is a source.
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
