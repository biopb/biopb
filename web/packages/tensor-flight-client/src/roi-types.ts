/**
 * ROI annotations, in the shape the app wants them.
 *
 * Deliberately not the proto shape and not the wire shape: the wire is proto3
 * canonical JSON (int64 as strings, bytes as base64, oneofs as bare keys,
 * every default-valued field simply absent), which is a poor thing to hand a
 * React component. `roi-json.ts` converts at this seam.
 *
 * Backend contract: biopb-tensor-server/docs/roi-annotations.md.
 * SPA design: docs/roi-annotations-ui.md.
 */

/** A vertex in LEVEL-0 pixel coordinates, in the tensor's own Y/X axes. */
export interface RoiPoint {
  x: number;
  y: number;
  /** Only meaningful for sources with a z axis; the 2-D viewer never sets it. */
  z?: number;
}

/**
 * The five 2-D vector arms the server accepts. `mask` and `mesh` exist in
 * `biopb.image.ROI` but are rejected by the store -- instance segmentation is a
 * label tensor, not an annotation set.
 */
export type RoiShapeKind = "point" | "rectangle" | "ellipse" | "polygon" | "polyline";

export type RoiGeometry =
  | { kind: "point"; at: RoiPoint }
  | { kind: "rectangle"; topLeft: RoiPoint; bottomRight: RoiPoint }
  | { kind: "ellipse"; center: RoiPoint; radius: RoiPoint }
  | { kind: "polygon"; points: RoiPoint[] }
  /** `width` is geometry, not styling: the stroke has real extent in pixels. */
  | { kind: "polyline"; points: RoiPoint[]; width: number };

export interface RoiAnnotation {
  roiId: string;
  arrayId: string;
  setName: string;
  label: string;
  geometry: RoiGeometry;
  /**
   * Sparse plane pin, `dim_label -> index`. A dimension ABSENT from this map
   * applies at every index of that dimension -- that is how one ROI follows a
   * z-stack. See {@link roiVisibleOnPlane}.
   */
  plane: Record<string, number>;
  /** Parsed `props_json`. `{}` when the server held nothing or unparsable text. */
  props: Record<string, unknown>;
  /** Server-assigned, bumped on every write. Echo it back for a conditional put. */
  rev: number;
  createdAtMs: number;
  updatedAtMs: number;
  /**
   * The tensor's `content_version` when this was written, base64, **opaque**.
   *
   * Carried verbatim rather than decoded because no HTTP client can produce one
   * (the sidecar publishes only the 8-hex token, in `array_id`) -- but it is a
   * client column the store rewrites on every update, so dropping it here would
   * erase it on the next read-modify-write.
   */
  drawnAgainstVersion?: string;
}

/** What a caller supplies to {@link TensorHttpClient.putRois}. */
export interface RoiAnnotationInput {
  /** Empty or absent on create -- the server mints a uuid4 hex. */
  roiId?: string;
  /** Defaults to the tensor the put is addressed to; the server checks it matches. */
  arrayId?: string;
  /** Empty -> "default". */
  setName?: string;
  label?: string;
  geometry: RoiGeometry;
  plane?: Record<string, number>;
  props?: Record<string, unknown>;
  /** Required only under `checkRev`, where it is the rev the edit was based on. */
  rev?: number;
  drawnAgainstVersion?: string;
}

export interface RoiListResult {
  rois: RoiAnnotation[];
  /** True when the server's per-tensor cap clipped the set. */
  truncated: boolean;
  /**
   * Rows dropped because their geometry did not decode -- a newer server's shape
   * arm, in practice. Non-zero means the overlay is showing less than the store
   * holds, which is worth saying out loud rather than rendering silently short.
   */
  skipped: number;
}

/** One annotation that lost a conditional write. */
export interface RoiConflict {
  roiId: string;
  /** The rev now in the store, so the caller can re-read and merge. */
  storedRev: number;
}

export interface RoiPutResult {
  stored: RoiAnnotation[];
  conflicts: RoiConflict[];
  skipped: number;
}
