import { describe, expect, it } from "vitest";
import type { TileInfo, VolumeAvailable } from "@biopb/tensor-flight-client";
import {
  DEFAULT_VOLUME_RENDER_MODE,
  VOLUME_MAX_BYTES,
  VOLUME_MAX_VOXELS,
  VOLUME_RENDER_MODES,
  volumeCentre,
  volumeKey,
  volumeRefusal,
  volumeRequest,
  volumeScaleRatio,
  volumeZoom,
} from "./volumeUtils";

function makeVolume(over: Partial<VolumeAvailable> = {}): VolumeAvailable {
  return {
    available: true,
    reason: null,
    axes: { z: 2, y: 3, x: 4 },
    scale_hint: [1, 1, 1, 4, 4],
    depth: 200,
    height: 512,
    width: 512,
    bytes: 200 * 512 * 512 * 2,
    spacing: null,
    unit: null,
    ...over,
  };
}

/** TCZYX, the shape the volume above is the plan for. */
function makeInfo(over: Partial<TileInfo> = {}): TileInfo {
  return {
    array_id: "src0/Image:0",
    dim_labels: ["t", "c", "z", "y", "x"],
    shape: [5, 3, 200, 2048, 2048],
    chunk_shape: [1, 1, 1, 512, 512],
    dtype: "uint16",
    tile_size: 512,
    plane: { y: 3, x: 4, s: null },
    selectable: { t: 0, z: 2, c: 1 },
    sel_axes: [],
    levels: [],
    volume: makeVolume(),
    ...over,
  };
}

const AT_ORIGIN = { t: 0, z: 0, c: 0, axes: {} };

/** Extents whose product is `voxels`, shaped like a deep stack. */
function volumeOf(voxels: number) {
  const plane = 512;
  return { depth: Math.ceil(voxels / (plane * plane)), height: plane, width: plane };
}

describe("volumeRefusal", () => {
  it("passes a tensor the server says has a volume", () => {
    expect(volumeRefusal(makeInfo())).toBeNull();
  });

  it("relays the server's own reason verbatim", () => {
    const info = makeInfo({
      volume: { available: false, reason: "z axis (axis 2) has extent 1, not a volume" },
    });
    expect(volumeRefusal(info)).toBe("z axis (axis 2) has extent 1, not a volume");
  });

  it("treats a server with no volume field as unable to serve one", () => {
    const info = makeInfo();
    delete info.volume;
    expect(volumeRefusal(info)).toMatch(/does not serve volumes/);
  });

  it("refuses a volume too large to hold in a browser tab", () => {
    const info = makeInfo({ volume: makeVolume({ bytes: VOLUME_MAX_BYTES + 1 }) });
    expect(volumeRefusal(info)).toMatch(/over this viewer's/);
  });

  it("accepts one exactly at the limit", () => {
    const info = makeInfo({ volume: makeVolume({ bytes: VOLUME_MAX_BYTES }) });
    expect(volumeRefusal(info)).toBeNull();
  });
});

describe("volumeRequest", () => {
  it("spans z/y/x whole and pins the rest", () => {
    const info = makeInfo();
    const req = volumeRequest(info, makeVolume(), { t: 2, z: 7, c: 1, axes: {} });
    // z is NOT pinned by slice.z: the volume is the whole axis.
    expect(req.slice_start).toEqual([2, 1, 0, 0, 0]);
    expect(req.slice_stop).toEqual([3, 2, 200, 2048, 2048]);
  });

  it("delegates the scale rather than naming one", () => {
    const req = volumeRequest(makeInfo(), makeVolume(), AT_ORIGIN);
    expect(req.scale_policy).toBe("volume");
    expect(req.scale_hint).toBeUndefined();
  });

  it("sends full-resolution bounds, not the volume's own extents", () => {
    // The server applies its scale AFTER slice_hint, so a 2048 plane stays 2048
    // here even though the volume it returns is 512 wide.
    const req = volumeRequest(makeInfo(), makeVolume(), AT_ORIGIN);
    expect(req.slice_stop?.[4]).toBe(2048);
  });

  it("clamps an index carried over from a larger tensor", () => {
    const req = volumeRequest(makeInfo(), makeVolume(), { t: 99, z: 0, c: 0, axes: {} });
    expect(req.slice_start?.[0]).toBe(4);
    expect(req.slice_stop?.[0]).toBe(5);
  });

  it("pins an axis that t/z/c cannot name, by its positional key", () => {
    const info = makeInfo({
      dim_labels: ["i", "z", "y", "x"],
      shape: [155, 40, 1024, 1024],
      plane: { y: 2, x: 3, s: null },
      selectable: { t: null, z: 1, c: null },
    });
    const volume = makeVolume({ axes: { z: 1, y: 2, x: 3 }, depth: 40 });
    const req = volumeRequest(info, volume, { t: 0, z: 0, c: 0, axes: { a0: 12 } });
    expect(req.slice_start).toEqual([12, 0, 0, 0]);
    expect(req.slice_stop).toEqual([13, 40, 1024, 1024]);
  });
});

describe("volumeKey", () => {
  it("is the same for two store states that read the same pixels", () => {
    const info = makeInfo();
    const a = volumeRequest(info, makeVolume(), { t: 1, z: 0, c: 0, axes: {} });
    // Only z differs, and z is read whole — so this must not refetch.
    const b = volumeRequest(info, makeVolume(), { t: 1, z: 42, c: 0, axes: {} });
    expect(volumeKey(a)).toBe(volumeKey(b));
  });

  it("differs when the selection does", () => {
    const info = makeInfo();
    const a = volumeRequest(info, makeVolume(), { t: 1, z: 0, c: 0, axes: {} });
    const b = volumeRequest(info, makeVolume(), { t: 2, z: 0, c: 0, axes: {} });
    expect(volumeKey(a)).not.toBe(volumeKey(b));
  });
});

describe("volumeScaleRatio", () => {
  it("is isotropic when the source declares no physical size", () => {
    expect(volumeScaleRatio(makeVolume({ spacing: null }))).toEqual([1, 1, 1]);
  });

  it("normalises to the finest axis, in deck.gl's x/y/z order", () => {
    const volume = makeVolume({ spacing: { z: 2, y: 0.5, x: 0.5 }, unit: "micrometer" });
    expect(volumeScaleRatio(volume)).toEqual([1, 1, 4]);
  });

  it("ignores a partial physical scale rather than stretching one axis", () => {
    const volume = makeVolume({ spacing: { z: 2, y: 0, x: 0.5 } });
    expect(volumeScaleRatio(volume)).toEqual([1, 1, 1]);
  });
});

describe("volumeCentre", () => {
  it("is half the volume in each axis when isotropic", () => {
    expect(volumeCentre(makeVolume())).toEqual([256, 256, 100]);
  });

  it("follows the anisotropy, so an orbit stays on-axis", () => {
    const volume = makeVolume({ spacing: { z: 2, y: 0.5, x: 0.5 } });
    // z spans 200 * 4 world units, so its centre is 400, not 100.
    expect(volumeCentre(volume)).toEqual([256, 256, 400]);
  });
});

describe("volumeZoom", () => {
  it("fits the longest scaled extent into the smaller pane dimension", () => {
    // 512 wide in a 512px pane is 2**0 px per world unit, less the backoff.
    expect(volumeZoom(makeVolume(), { width: 512, height: 1024 }, 0)).toBe(0);
    expect(volumeZoom(makeVolume(), { width: 256, height: 1024 }, 0)).toBe(-1);
  });

  it("measures depth too, so a deep stack is not framed by its plane alone", () => {
    const volume = makeVolume({ spacing: { z: 4, y: 0.5, x: 0.5 } });
    // z now spans 200 * 8 = 1600 world units, longer than the 512 plane.
    expect(volumeZoom(volume, { width: 1600, height: 1600 }, 0)).toBe(0);
  });

  it("answers 0 rather than -Infinity before the pane is measured", () => {
    expect(volumeZoom(makeVolume(), { width: 0, height: 0 })).toBe(0);
  });
});

describe("VOLUME_RENDER_MODES", () => {
  it("defaults to maximum intensity projection", () => {
    // Fluorescence is sparse signal on a dark ground: the brightest voxel along
    // a ray is the structure. Additive sums the whole ray and hazes it out.
    expect(DEFAULT_VOLUME_RENDER_MODE).toBe("mip");
  });

  it("offers the default first, so the leading button is the one in effect", () => {
    expect(VOLUME_RENDER_MODES[0].key).toBe(DEFAULT_VOLUME_RENDER_MODE);
  });

  it("covers Viv's three rendering modes, with unique keys", () => {
    const keys = VOLUME_RENDER_MODES.map((m) => m.key);
    expect(new Set(keys).size).toBe(keys.length);
    expect(keys).toEqual(["mip", "additive", "minip"]);
  });
});

describe("volumeRefusal — the voxel ceiling", () => {
  // A native pyramid is advertised instead of the computed plan, so one that
  // downsamples only Y/X leaves a full-depth volume with no 3-D budget applied
  // (biopb/biopb#891). Bytes do not catch it: at uint8 a volume can be six
  // times the server's voxel budget and still land under VOLUME_MAX_BYTES.
  it("refuses a volume that would not fit in VRAM even when the bytes fit", () => {
    const voxels = VOLUME_MAX_VOXELS * 2;
    const bytes = voxels; // uint8: one byte per voxel
    expect(bytes).toBeLessThan(VOLUME_MAX_BYTES);
    const info = makeInfo({ volume: makeVolume({ ...volumeOf(voxels), bytes }) });
    expect(volumeRefusal(info)).toMatch(/GPU memory/);
  });

  it("admits the server's own budget with headroom", () => {
    // 448**3 is 343 MB at 4 bytes per voxel -- comfortably inside the ceiling,
    // so the guard never refuses a plan the server actually bounded.
    const voxels = 448 ** 3;
    expect(voxels).toBeLessThan(VOLUME_MAX_VOXELS);
    const info = makeInfo({
      volume: makeVolume({ ...volumeOf(voxels), bytes: voxels * 2 }),
    });
    expect(volumeRefusal(info)).toBeNull();
  });
});
