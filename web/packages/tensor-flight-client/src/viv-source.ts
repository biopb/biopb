/**
 * Viv `PixelSource` adapter over the tile API.
 *
 * Viv addresses an image as `PixelSource[]`, one entry per resolution with
 * index 0 the finest, and asks each for tiles by grid position. `/api/tile_info`
 * already describes exactly that, so this is a mapping, not a data layer -- no
 * zarr, no OME-TIFF, no second wire format.
 *
 * Types come from `@vivjs/types` rather than being restated here, so a Viv
 * upgrade that changes the contract fails the build instead of at runtime. The
 * import is type-only: nothing from Viv ends up in the built JS, and deck.gl
 * belongs to whichever app renders these, not to this client.
 *
 *   const { data, info } = await createTensorPixelSources(client, "src0");
 *   <PictureInPictureViewer loader={data} ... />
 */

import type {
  PixelData,
  PixelSource,
  PixelSourceSelection,
  RasterSelection,
  SupportedDtype,
  SupportedTypedArray,
  TileSelection,
} from "@vivjs/types";

import { TensorAbortError, type TensorHttpClient } from "./client.js";
import type { TileInfo, TileLevel } from "./types.js";

// ---------------------------------------------------------------------------
// dtype
// ---------------------------------------------------------------------------

/**
 * NumPy dtype -> Viv dtype.
 *
 * Both spellings the server uses are accepted: `/api/tile_info` reports the
 * descriptor's own `"<u2"` form, while the tile response header says `"uint16"`.
 */
const DTYPE_BY_NUMPY: Record<string, SupportedDtype> = {
  "|u1": "Uint8", "<u1": "Uint8", ">u1": "Uint8", uint8: "Uint8",
  "<u2": "Uint16", ">u2": "Uint16", uint16: "Uint16",
  "<u4": "Uint32", ">u4": "Uint32", uint32: "Uint32",
  "|i1": "Int8", "<i1": "Int8", ">i1": "Int8", int8: "Int8",
  "<i2": "Int16", ">i2": "Int16", int16: "Int16",
  "<i4": "Int32", ">i4": "Int32", int32: "Int32",
  "<f4": "Float32", ">f4": "Float32", float32: "Float32",
  "<f8": "Float64", ">f8": "Float64", float64: "Float64",
};

const TYPED_ARRAY_BY_DTYPE = {
  Uint8: Uint8Array, Uint16: Uint16Array, Uint32: Uint32Array,
  Int8: Int8Array, Int16: Int16Array, Int32: Int32Array,
  Float32: Float32Array, Float64: Float64Array,
} as const;

export function vivDtype(numpyDtype: string): SupportedDtype {
  const dtype = DTYPE_BY_NUMPY[numpyDtype.trim()];
  if (!dtype) {
    throw new Error(
      `Tensor dtype "${numpyDtype}" has no Viv equivalent ` +
        `(supported: ${Object.keys(TYPED_ARRAY_BY_DTYPE).join(", ")})`,
    );
  }
  return dtype;
}

/**
 * View the response bytes as the right typed array.
 *
 * No byte-swapping: the server normalises to native order before writing the
 * body, and every platform a browser runs on is little-endian.
 */
function asTypedArray(buffer: ArrayBuffer, dtype: SupportedDtype): SupportedTypedArray {
  return new TYPED_ARRAY_BY_DTYPE[dtype](buffer) as SupportedTypedArray;
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

/**
 * Wire dim_labels -> Viv labels.
 *
 * Viv requires the plane last: `[...rest, "y", "x"]`, with `"_c"` after it for
 * an interleaved RGB(A) samples axis. The data plane already guarantees the
 * canonical `[..., Z, Y, X, S]` order, so this lowercases and renames the
 * samples axis; a tensor that does not satisfy it is rejected here rather than
 * silently rendering a transposed plane.
 */
export function vivLabels(info: TileInfo): string[] {
  const { plane, dim_labels, shape } = info;
  const ndim = shape.length;
  const expectedX = plane.s === null ? ndim - 1 : ndim - 2;
  if (plane.x !== expectedX || plane.y !== expectedX - 1) {
    throw new Error(
      `Tensor ${info.array_id} is not in canonical [..., Y, X, S] order ` +
        `(plane y=${plane.y} x=${plane.x} s=${plane.s} of ${ndim} axes)`,
    );
  }
  return dim_labels.map((label, i) => {
    if (i === plane.y) return "y";
    if (i === plane.x) return "x";
    if (i === plane.s) return "_c";
    return label.toLowerCase();
  });
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface TensorPixelSourceOptions {
  /** Address a specific tensor of a multi-tensor source. */
  tensorId?: string;
  /**
   * Refuse a `getRaster` bigger than this many pixels.
   *
   * `getRaster` asks for a whole level at once, which Viv does for the overview
   * -- harmless on the coarsest source and a request for the entire image on
   * the finest. A clear refusal beats a multi-gigabyte read nobody meant to
   * start.
   */
  maxRasterPixels?: number;
  /** Reported when a tile fails. Aborted tiles never reach it. */
  onTileError?: (err: Error) => void;
  /** Applies to the `tile_info` fetch only; per-tile signals come from Viv. */
  signal?: AbortSignal;
}

const DEFAULT_MAX_RASTER_PIXELS = 16_777_216; // 4096 x 4096

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

export interface TensorPixelSources {
  /** One source per level, index 0 = full resolution, as Viv expects. */
  data: PixelSource<string[]>[];
  /** The grid these were built from; keep it for slider ranges and labels. */
  info: TileInfo;
}

/**
 * Build the `PixelSource[]` for a tensor (one `tile_info` round trip).
 */
export async function createTensorPixelSources(
  client: TensorHttpClient,
  sourceId: string,
  options: TensorPixelSourceOptions = {},
): Promise<TensorPixelSources> {
  const info = await client.tileInfo(sourceId, options.tensorId, { signal: options.signal });
  return { data: pixelSourcesFromInfo(client, sourceId, info, options), info };
}

/** The pure half of {@link createTensorPixelSources}, for an already-fetched grid. */
export function pixelSourcesFromInfo(
  client: TensorHttpClient,
  sourceId: string,
  info: TileInfo,
  options: TensorPixelSourceOptions = {},
): PixelSource<string[]>[] {
  const labels = vivLabels(info);
  const dtype = vivDtype(info.dtype);
  return info.levels.map((level) =>
    makeSource(client, sourceId, info, level, labels, dtype, options),
  );
}

function makeSource(
  client: TensorHttpClient,
  sourceId: string,
  info: TileInfo,
  level: TileLevel,
  labels: string[],
  dtype: SupportedDtype,
  options: TensorPixelSourceOptions,
): PixelSource<string[]> {
  const { plane } = info;
  // Per-level shape: only the plane shrinks, the slider axes are unchanged.
  const shape = info.shape.map((extent, i) =>
    i === plane.y ? level.height : i === plane.x ? level.width : extent,
  );
  const maxRasterPixels = options.maxRasterPixels ?? DEFAULT_MAX_RASTER_PIXELS;

  const planeOf = (bufferShape: number[]): { width: number; height: number } => ({
    height: bufferShape[plane.y] ?? level.height,
    width: bufferShape[plane.x] ?? level.width,
  });

  return {
    labels: labels as PixelSource<string[]>["labels"],
    shape,
    dtype,
    tileSize: info.tile_size,

    async getTile({ x, y, selection, signal }: TileSelection<string[]>): Promise<PixelData> {
      // deck.gl can ask for a tile past the edge while the viewport settles.
      // Answering locally with zeros costs no round trip and keeps the layer
      // from treating a routine 404 as a load failure.
      if (x < 0 || y < 0 || x >= level.cols || y >= level.rows) {
        return emptyTile(info.tile_size, dtype);
      }
      const result = await client.tile(
        {
          source_id: sourceId,
          // The canonical array_id, not the caller's shorthand: `info` was
          // fetched for this tensor, so this is unambiguous for both routes.
          tensor_id: info.array_id,
          level: level.level,
          col: x,
          row: y,
          ...tileSelection(info, selection),
        },
        { signal },
      );
      return {
        data: asTypedArray(result.buffer, dtype),
        ...planeOf(result.shape),
      };
    },

    async getRaster({ selection, signal }: RasterSelection<string[]>): Promise<PixelData> {
      const pixels = level.width * level.height;
      if (pixels > maxRasterPixels) {
        throw new Error(
          `getRaster on level ${level.level} would read ${pixels} pixels ` +
            `(limit ${maxRasterPixels}). Use a coarser level, or raise ` +
            `maxRasterPixels deliberately.`,
        );
      }
      const scale = level.scale;
      const sel = tileSelection(info, selection);
      const start: number[] = [];
      const stop: number[] = [];
      const scaleHint: number[] = [];
      info.shape.forEach((extent, i) => {
        const isPlane = i === plane.y || i === plane.x || i === plane.s;
        const index = isPlane ? 0 : sliderIndexAt(info, sel, i);
        start.push(isPlane ? 0 : index);
        stop.push(isPlane ? extent : index + 1);
        scaleHint.push(i === plane.y || i === plane.x ? scale : 1);
      });
      const arr = await client.slice(
        {
          source_id: sourceId,
          tensor_id: info.array_id,
          slice_start: start,
          slice_stop: stop,
          scale_hint: scaleHint,
        },
        { signal },
      );
      return { data: asTypedArray(arr.buffer, dtype), ...planeOf(arr.shape) };
    },

    onTileError(err: Error) {
      // A tile the viewport moved past is a normal outcome, not a failure.
      if (err instanceof TensorAbortError || err.name === "AbortError") return;
      if (options.onTileError) options.onTileError(err);
      else console.error(`tile error (${info.array_id})`, err);
    },
  };
}

/** A zero tile for an out-of-grid request, sized as Viv expects. */
function emptyTile(tileSize: number, dtype: SupportedDtype): PixelData {
  return {
    data: new TYPED_ARRAY_BY_DTYPE[dtype](tileSize * tileSize) as SupportedTypedArray,
    width: tileSize,
    height: tileSize,
  };
}

/**
 * Viv's label-keyed selection -> the tile API's `t`/`z`/`c` parameters.
 *
 * Those three are all the endpoint can address. Any other slider axis is served
 * at index 0, which is right for the default and wrong for anything else, so a
 * non-zero request on one is refused rather than quietly returning the wrong
 * plane. Extending the endpoint is the fix if a dataset needs it.
 */
function tileSelection(
  info: TileInfo,
  selection: PixelSourceSelection<string[]>,
): { t?: number; z?: number; c?: number } {
  const out: { t?: number; z?: number; c?: number } = {};
  for (const [label, rawIndex] of Object.entries(selection ?? {})) {
    const index = Number(rawIndex ?? 0);
    if (label === "t" || label === "z" || label === "c") {
      if (info.selectable[label] === null) {
        if (index !== 0) {
          throw new Error(`Tensor ${info.array_id} has no "${label}" axis to select`);
        }
        continue;
      }
      out[label] = index;
      continue;
    }
    if (label === "y" || label === "x" || label === "_c") continue;
    if (index !== 0) {
      throw new Error(
        `Axis "${label}" of ${info.array_id} cannot be selected through the tile ` +
          `API (only t/z/c); index ${index} would silently read index 0`,
      );
    }
  }
  return out;
}

/** The index the caller chose for wire axis `i`, or 0. */
function sliderIndexAt(
  info: TileInfo,
  sel: { t?: number; z?: number; c?: number },
  i: number,
): number {
  for (const axis of ["t", "z", "c"] as const) {
    if (info.selectable[axis] === i) return sel[axis] ?? 0;
  }
  return 0;
}
