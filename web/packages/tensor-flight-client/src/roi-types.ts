/**
 * ROI annotations, in the shape the app wants them.
 *
 * Deliberately not the proto shape and not the wire shape: the wire is proto3
 * canonical JSON (int64 as strings, bytes as base64, oneofs as bare keys,
 * every default-valued field simply absent), which is a poor thing to hand a
 * React component. `roi-json.ts` converts at this seam.
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
  /** `rotation` is in-plane, radians about `center`, +x toward +y; 0 is axis-aligned. */
  | { kind: "ellipse"; center: RoiPoint; radius: RoiPoint; rotation: number }
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
   * Sparse plane pin, **wire axis index** -> index on that axis. A dimension
   * ABSENT from this map applies at every index of it -- that is how one ROI
   * follows a z-stack. See {@link roiVisibleOnPlane}.
   *
   * Positional, not label-keyed: a label cannot address an unlabelled axis or
   * one of two sharing a label, and such an axis would then broadcast silently.
   * Turn an index into a name with the tensor's `dim_labels` when displaying it.
   */
  plane: Record<number, number>;
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
  plane?: Record<number, number>;
  props?: Record<string, unknown>;
  /** Required only under `checkRev`, where it is the rev the edit was based on. */
  rev?: number;
  drawnAgainstVersion?: string;
}

/**
 * Whether a set name is server-owned, by the server's own naming rule.
 *
 * `RoiSetInfo.reserved` is the authoritative answer, but it exists only once a
 * listing has landed; a name met before that -- one a link carries -- is judged
 * by the prefix the server reserves.
 */
export function isReservedSetName(setName: string): boolean {
  return setName.startsWith("@");
}

/** One annotation layer on a tensor, as the server enumerates it. */
export interface RoiSetInfo {
  setName: string;
  /** Rows stored in this set, whatever `rois` covers. */
  count: number;
  /**
   * The server owns this set: it is rebuilt from the source file, is read-only,
   * and `rois` carries it only when the request named it.
   */
  reserved: boolean;
}

export interface RoiListResult {
  rois: RoiAnnotation[];
  /**
   * Every set on the tensor, including those `rois` does not cover -- an
   * unqualified list returns the client-owned sets alone.
   */
  sets: RoiSetInfo[];
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
