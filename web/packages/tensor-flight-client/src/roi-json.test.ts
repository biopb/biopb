/**
 * The ROI wire codec.
 *
 * The expected payloads here are not invented: they are what
 * `json_format.MessageToDict` actually emits for these messages, which is what
 * makes the absent-field cases (an origin vertex serializing to `{}`) worth
 * pinning rather than assuming.
 */

import { describe, it, expect } from "vitest";
import {
  decodeRoiAnnotation,
  decodeRoiDeleteResult,
  decodeRoiGeometry,
  decodeRoiListResult,
  decodeRoiPutResult,
  encodeRoiAnnotation,
  encodeRoiGeometry,
} from "./roi-json.js";
import type { RoiGeometry } from "./roi-types.js";

describe("decodeRoiGeometry", () => {
  it("reads an absent coordinate as the proto default, not as missing data", () => {
    // Point(x=0, y=0) serializes to {} -- a vertex at the origin.
    expect(decodeRoiGeometry({ point: {} })).toEqual({ kind: "point", at: { x: 0, y: 0 } });
  });

  it("keeps a polygon's origin vertex in place", () => {
    const g = decodeRoiGeometry({ polygon: { points: [{}, { x: 5.5, y: 2 }] } });
    expect(g).toEqual({
      kind: "polygon",
      points: [{ x: 0, y: 0 }, { x: 5.5, y: 2 }],
    });
  });

  it("defaults a polyline of width 0, whose width key is absent", () => {
    const g = decodeRoiGeometry({ polyline: { points: [{ x: 1, y: 1 }] } });
    expect(g).toEqual({ kind: "polyline", points: [{ x: 1, y: 1 }], width: 0 });
  });

  it("reads a rectangle's camelCase corners", () => {
    expect(decodeRoiGeometry({ rectangle: { topLeft: { x: 3 }, bottomRight: { y: 9 } } })).toEqual({
      kind: "rectangle",
      topLeft: { x: 3, y: 0 },
      bottomRight: { x: 0, y: 9 },
    });
  });

  it("reads an ellipse", () => {
    expect(
      decodeRoiGeometry({
        ellipse: { center: { x: 4, y: 4 }, radius: { x: 2, y: 1 }, rotation: 0.75 },
      }),
    ).toEqual({ kind: "ellipse", center: { x: 4, y: 4 }, radius: { x: 2, y: 1 }, rotation: 0.75 });
  });

  it("reads an ellipse written before rotation existed as axis-aligned", () => {
    // proto3 JSON omits a zero float, so this is also what a rotation of 0
    // looks like on the wire -- the two are the same reading by construction.
    expect(decodeRoiGeometry({ ellipse: { center: { x: 4, y: 4 }, radius: { x: 2, y: 1 } } })).toEqual(
      { kind: "ellipse", center: { x: 4, y: 4 }, radius: { x: 2, y: 1 }, rotation: 0 },
    );
  });

  it("keeps z when the source carried one", () => {
    expect(decodeRoiGeometry({ point: { x: 1, y: 2, z: 7 } })).toEqual({
      kind: "point",
      at: { x: 1, y: 2, z: 7 },
    });
  });

  it("returns null for an arm this client does not render", () => {
    // mask/mesh are rejected by the store, so this is really the newer-server case.
    expect(decodeRoiGeometry({ mask: {} })).toBeNull();
    expect(decodeRoiGeometry({})).toBeNull();
    expect(decodeRoiGeometry(undefined)).toBeNull();
  });
});

describe("decodeRoiAnnotation", () => {
  const WIRE = {
    roiId: "r1",
    arrayId: "src/Image:0",
    setName: "default",
    roi: { polygon: { points: [{}, { x: 5.5, y: 2 }] } },
    plane: { "2": 12 },
    drawnAgainstVersion: "AQL/",
    rev: "3",
    createdAtUnixMs: "1757000000123",
  };

  it("reads int64 fields, which arrive as strings", () => {
    const a = decodeRoiAnnotation(WIRE)!;
    expect(a.rev).toBe(3);
    expect(a.createdAtMs).toBe(1757000000123);
    // The key is a string because JSON keys are; the uint32 value is a number.
    expect(a.plane).toEqual({ 2: 12 });
  });

  it("defaults every absent field rather than failing", () => {
    const a = decodeRoiAnnotation(WIRE)!;
    expect(a.label).toBe("");
    expect(a.props).toEqual({});
    expect(a.updatedAtMs).toBe(0);
  });

  it("carries drawn_against_version verbatim", () => {
    // Opaque base64. Dropping it would erase the stored column on the next
    // read-modify-write, since the store rewrites it on every update.
    expect(decodeRoiAnnotation(WIRE)!.drawnAgainstVersion).toBe("AQL/");
  });

  it("omits drawn_against_version when the server sent none", () => {
    expect(decodeRoiAnnotation({ roi: { point: {} } })!.drawnAgainstVersion).toBeUndefined();
  });

  it("parses props_json, and shrugs off anything that is not an object", () => {
    const props = decodeRoiAnnotation({ ...WIRE, propsJson: '{"color":"#f00"}' })!.props;
    expect(props).toEqual({ color: "#f00" });
    expect(decodeRoiAnnotation({ ...WIRE, propsJson: "not json" })!.props).toEqual({});
    expect(decodeRoiAnnotation({ ...WIRE, propsJson: "[1,2]" })!.props).toEqual({});
  });

  it("falls back to the default set name", () => {
    expect(decodeRoiAnnotation({ roi: { point: {} } })!.setName).toBe("default");
  });

  it("is null when the geometry does not decode", () => {
    expect(decodeRoiAnnotation({ roiId: "r1", roi: { mesh: {} } })).toBeNull();
  });
});

describe("encodeRoiAnnotation", () => {
  const GEOM: RoiGeometry = { kind: "polygon", points: [{ x: 0, y: 0 }, { x: 4, y: 1 }] };

  it("emits int64 as a string but a uint32 plane index as a number", () => {
    // proto3 JSON stringifies only the 64-bit integer types; the plane is
    // uint32 on both halves, so only its keys are strings.
    const out = encodeRoiAnnotation({ geometry: GEOM, plane: { 2: 12, 0: 0 }, rev: 3 });
    expect(out.plane).toEqual({ "2": 12, "0": 0 });
    expect(out.rev).toBe("3");
  });

  it("omits an empty roi_id so the server mints one", () => {
    expect(encodeRoiAnnotation({ geometry: GEOM })).not.toHaveProperty("roiId");
  });

  it("writes an origin vertex explicitly rather than as {}", () => {
    expect(encodeRoiGeometry(GEOM)).toEqual({
      polygon: { points: [{ x: 0, y: 0 }, { x: 4, y: 1 }] },
    });
  });

  it("serializes props and keeps an empty bag off the wire", () => {
    expect(encodeRoiAnnotation({ geometry: GEOM, props: { a: 1 } }).propsJson).toBe('{"a":1}');
    expect(encodeRoiAnnotation({ geometry: GEOM, props: {} })).not.toHaveProperty("propsJson");
  });

  it("round-trips every geometry arm through a decode", () => {
    const arms: RoiGeometry[] = [
      { kind: "point", at: { x: 1, y: 2 } },
      { kind: "rectangle", topLeft: { x: 0, y: 0 }, bottomRight: { x: 4, y: 4 } },
      { kind: "ellipse", center: { x: 2, y: 2 }, radius: { x: 1, y: 3 }, rotation: 0 },
      { kind: "ellipse", center: { x: 2, y: 2 }, radius: { x: 1, y: 3 }, rotation: -1.25 },
      GEOM,
      { kind: "polyline", points: [{ x: 0, y: 0 }, { x: 9, y: 9 }], width: 2.5 },
    ];
    for (const arm of arms) {
      expect(decodeRoiGeometry(encodeRoiGeometry(arm))).toEqual(arm);
    }
  });

  it("round-trips a version token so an edit does not erase it", () => {
    const out = encodeRoiAnnotation({ geometry: GEOM, drawnAgainstVersion: "AQL/" });
    expect(out.drawnAgainstVersion).toBe("AQL/");
  });
});

describe("result decoders", () => {
  it("reads an empty list result, which is `{}` on the wire", () => {
    expect(decodeRoiListResult({})).toEqual({ rois: [], truncated: false, skipped: 0 });
  });

  it("reports truncation", () => {
    expect(decodeRoiListResult({ truncated: true }).truncated).toBe(true);
  });

  it("counts undecodable rows instead of losing the whole set", () => {
    const result = decodeRoiListResult({
      rois: [{ roi: { point: {} } }, { roi: { mask: {} } }, { roi: { polygon: {} } }],
    });
    expect(result.rois).toHaveLength(2);
    expect(result.skipped).toBe(1);
  });

  it("reads conflicts, whose stored_rev is an int64 string", () => {
    const result = decodeRoiPutResult({ conflicts: [{ roiId: "r1", storedRev: "7" }] });
    expect(result.conflicts).toEqual([{ roiId: "r1", storedRev: 7 }]);
    expect(result.stored).toEqual([]);
  });

  it("reads a delete result, absent when nothing matched", () => {
    expect(decodeRoiDeleteResult({ deleted: ["a", "b"] })).toEqual(["a", "b"]);
    expect(decodeRoiDeleteResult({})).toEqual([]);
  });
});
