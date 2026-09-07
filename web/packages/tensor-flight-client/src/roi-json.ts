/**
 * The proto3-canonical-JSON codec for ROI annotations.
 *
 * Hand-written rather than generated (see docs/roi-annotations-ui.md). Four
 * properties of the wire format drive every line here, and all four are silent
 * failures if missed:
 *
 * 1. **Every default-valued field is absent.** `Point(x=0, y=0)` serializes to
 *    `{}` -- a vertex at the origin arrives as an empty object, and a polyline
 *    of width 0 has no `width` key. So each field is read as "absent means the
 *    proto default", never as "absent means malformed".
 * 2. **int64 is a JSON string.** `rev`, the timestamps, and every value of the
 *    `plane` map arrive quoted.
 * 3. **bytes is base64**, standard alphabet (`+/`), not URL-safe.
 * 4. **A oneof is a bare key**, so the shape arm is found by probing.
 *
 * The server's parser (`json_format.ParseDict`) accepts both camelCase and
 * snake_case and takes int64 as either a string or a number; this emits the
 * canonical form regardless.
 */

import type {
  RoiAnnotation,
  RoiAnnotationInput,
  RoiConflict,
  RoiGeometry,
  RoiListResult,
  RoiPoint,
  RoiPutResult,
} from "./roi-types.js";

type Json = Record<string, unknown>;

function obj(value: unknown): Json {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Json)
    : {};
}

function arr(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/** A proto3 number field: absent -> 0, and int64 arrives as a string. */
function num(value: unknown): number {
  if (typeof value === "number") return Number.isFinite(value) ? value : 0;
  if (typeof value === "string") {
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

function decodePoint(value: unknown): RoiPoint {
  const p = obj(value);
  const point: RoiPoint = { x: num(p.x), y: num(p.y) };
  if (p.z !== undefined) point.z = num(p.z);
  return point;
}

function decodePoints(value: unknown): RoiPoint[] {
  return arr(value).map(decodePoint);
}

function encodePoint(p: RoiPoint): Json {
  // x/y always emitted, zero included: the server reads an absent field as 0
  // either way, and a literal `{}` for the origin is needlessly cryptic on the
  // wire.
  const out: Json = { x: p.x, y: p.y };
  if (p.z !== undefined) out.z = p.z;
  return out;
}

/**
 * The geometry, or null when no arm this client understands is present.
 *
 * Null rather than a throw: an unreadable row must cost that row, not the whole
 * overlay. `mask` and `mesh` are the arms the store rejects, so in practice this
 * is only reached against a server that grew a new one.
 */
export function decodeRoiGeometry(value: unknown): RoiGeometry | null {
  const roi = obj(value);
  if (roi.point !== undefined) {
    return { kind: "point", at: decodePoint(roi.point) };
  }
  if (roi.rectangle !== undefined) {
    const r = obj(roi.rectangle);
    return {
      kind: "rectangle",
      topLeft: decodePoint(r.topLeft ?? r.top_left),
      bottomRight: decodePoint(r.bottomRight ?? r.bottom_right),
    };
  }
  if (roi.ellipse !== undefined) {
    const e = obj(roi.ellipse);
    return {
      kind: "ellipse",
      center: decodePoint(e.center),
      radius: decodePoint(e.radius),
      rotation: num(e.rotation),
    };
  }
  if (roi.polygon !== undefined) {
    return { kind: "polygon", points: decodePoints(obj(roi.polygon).points) };
  }
  if (roi.polyline !== undefined) {
    const p = obj(roi.polyline);
    return { kind: "polyline", points: decodePoints(p.points), width: num(p.width) };
  }
  return null;
}

export function encodeRoiGeometry(geometry: RoiGeometry): Json {
  switch (geometry.kind) {
    case "point":
      return { point: encodePoint(geometry.at) };
    case "rectangle":
      return {
        rectangle: {
          topLeft: encodePoint(geometry.topLeft),
          bottomRight: encodePoint(geometry.bottomRight),
        },
      };
    case "ellipse":
      return {
        ellipse: {
          center: encodePoint(geometry.center),
          radius: encodePoint(geometry.radius),
          rotation: geometry.rotation,
        },
      };
    case "polygon":
      return { polygon: { points: geometry.points.map(encodePoint) } };
    case "polyline":
      return {
        polyline: { points: geometry.points.map(encodePoint), width: geometry.width },
      };
  }
}

// ---------------------------------------------------------------------------
// Annotation
// ---------------------------------------------------------------------------

/**
 * The plane pin, `{"2": 12}` on the wire: axis 2 at index 12.
 *
 * Both halves are uint32, so unlike the int64 fields the VALUE is a plain JSON
 * number -- proto3 stringifies only the 64-bit integer types. The key is a
 * string regardless, because a JSON object key always is.
 *
 * Neither half can be negative from a conforming server, so the guards below
 * are belt-and-braces against one that is not.
 */
function decodePlane(value: unknown): Record<number, number> {
  const out: Record<number, number> = {};
  for (const [axis, index] of Object.entries(obj(value))) {
    const key = Number(axis);
    const at = num(index);
    if (Number.isInteger(key) && key >= 0 && at >= 0) out[key] = at;
  }
  return out;
}

/** `props_json` is opaque to the server, so anything but a JSON object is `{}`. */
function decodeProps(value: unknown): Record<string, unknown> {
  const raw = str(value);
  if (!raw) return {};
  try {
    const parsed: unknown = JSON.parse(raw);
    return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

/** One annotation, or null when its geometry did not decode. */
export function decodeRoiAnnotation(value: unknown): RoiAnnotation | null {
  const a = obj(value);
  const geometry = decodeRoiGeometry(a.roi);
  if (!geometry) return null;
  const annotation: RoiAnnotation = {
    roiId: str(a.roiId),
    arrayId: str(a.arrayId),
    // The server substitutes "default" for an empty set_name before storing, so
    // this only fills in for a record that never reached the store.
    setName: str(a.setName) || "default",
    label: str(a.label),
    geometry,
    plane: decodePlane(a.plane),
    props: decodeProps(a.propsJson),
    rev: num(a.rev),
    createdAtMs: num(a.createdAtUnixMs),
    updatedAtMs: num(a.updatedAtUnixMs),
  };
  const version = str(a.drawnAgainstVersion);
  if (version) annotation.drawnAgainstVersion = version;
  return annotation;
}

/** Decode a list of annotations, counting rather than throwing on bad rows. */
function decodeAnnotations(value: unknown): { rois: RoiAnnotation[]; skipped: number } {
  const rois: RoiAnnotation[] = [];
  let skipped = 0;
  for (const entry of arr(value)) {
    const decoded = decodeRoiAnnotation(entry);
    if (decoded) rois.push(decoded);
    else skipped += 1;
  }
  return { rois, skipped };
}

export function encodeRoiAnnotation(input: RoiAnnotationInput): Json {
  const out: Json = { roi: encodeRoiGeometry(input.geometry) };
  if (input.roiId) out.roiId = input.roiId;
  if (input.arrayId) out.arrayId = input.arrayId;
  if (input.setName) out.setName = input.setName;
  if (input.label) out.label = input.label;
  if (input.plane && Object.keys(input.plane).length > 0) {
    // Numbers, not strings: uint32 is small enough that proto3 JSON spells it
    // as a number, and this is what the server echoes back.
    out.plane = Object.fromEntries(
      Object.entries(input.plane).map(([axis, index]) => [axis, Math.trunc(index)]),
    );
  }
  if (input.props && Object.keys(input.props).length > 0) {
    out.propsJson = JSON.stringify(input.props);
  }
  if (input.rev) out.rev = String(Math.trunc(input.rev));
  if (input.drawnAgainstVersion) out.drawnAgainstVersion = input.drawnAgainstVersion;
  return out;
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

export function decodeRoiListResult(value: unknown): RoiListResult {
  const body = obj(value);
  const { rois, skipped } = decodeAnnotations(body.rois);
  return { rois, truncated: body.truncated === true, skipped };
}

function decodeConflict(value: unknown): RoiConflict {
  const c = obj(value);
  return { roiId: str(c.roiId), storedRev: num(c.storedRev) };
}

export function decodeRoiPutResult(value: unknown): RoiPutResult {
  const body = obj(value);
  const { rois, skipped } = decodeAnnotations(body.stored);
  return { stored: rois, conflicts: arr(body.conflicts).map(decodeConflict), skipped };
}

/** The ids actually removed; an id that was not there is simply absent. */
export function decodeRoiDeleteResult(value: unknown): string[] {
  return arr(obj(value).deleted).map(str);
}
