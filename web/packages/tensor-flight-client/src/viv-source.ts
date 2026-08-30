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
import { sliderAxes } from "./tensor-array.js";
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
export function asTypedArray(
  buffer: ArrayBuffer,
  dtype: SupportedDtype,
): SupportedTypedArray {
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
 * canonical `[..., Z, Y, X, S]` order, so this renames the plane axes and takes
 * the rest from {@link sliderAxes}; a tensor that does not satisfy the order is
 * rejected here rather than silently rendering a transposed plane.
 *
 * The non-plane labels come from `sliderAxes` rather than from `dim_labels`
 * directly because Viv's selection is a *record keyed by label*, so the labels
 * have to be unique — and a source's own are not. Two axes may share one (the
 * duplicate-`c` case) and one may be empty, either of which silently collapses
 * two axes into one selection entry. `sliderAxes` keys the unnamed ones by wire
 * index instead, which is unique by construction; {@link tileSelection} reads
 * the mapping back out of the same function, so the two cannot drift.
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
  const keys = new Map(
    sliderAxes(dim_labels, shape).map((axis) => [axis.axis, axis.key]),
  );
  return dim_labels.map((label, i) => {
    if (i === plane.y) return "y";
    if (i === plane.x) return "x";
    if (i === plane.s) return "_c";
    return keys.get(i) ?? label.toLowerCase();
  });
}

// ---------------------------------------------------------------------------
// Server capability
// ---------------------------------------------------------------------------

/**
 * Whether this server's tile route can address an axis by wire index (`sel`).
 *
 * `sel_axes` is the marker because it arrived with `sel` in the same change: a
 * server that publishes the list is a server that accepts the parameter, and one
 * that predates both sends `pinned` instead and leaves this `undefined`.
 *
 * The probe is needed because the failure is otherwise **silent**. Starlette
 * drops undeclared query parameters, so an old server answers `?sel=0:154` with
 * index 0's pixels, HTTP 200, and an ETag identical to every other frame's — the
 * viewer shows one plane of 155 and nothing anywhere says so. That is the exact
 * failure `_resolve_tile_selection` already refuses one level down ("a client
 * told it got a plane it did not get"), and version skew is not a reason to
 * accept it: a browser holding a cached bundle, or a sidecar upgraded on its own
 * schedule, reaches this without anyone doing anything unusual.
 *
 * Deliberately a *capability* question and not a *tensor* one. A TCZYX tensor
 * needs no `sel` and still tiles perfectly against an old server; only a tensor
 * with an axis nothing names is affected.
 */
export function supportsSelParameter(info: TileInfo): boolean {
  return Array.isArray(info.sel_axes);
}

/** The unnamed axes this server cannot reach, or `[]` when it can reach them. */
function unreachableAxes(info: TileInfo) {
  if (supportsSelParameter(info)) return [];
  return sliderAxes(info.dim_labels, info.shape).filter(
    (axis) => !axis.named && axis.extent > 1,
  );
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface TensorPixelSourceOptions {
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
// getRaster coalescing
// ---------------------------------------------------------------------------

/**
 * Viv's abort convention, restated because `@vivjs/loaders` does not export it
 * as a type and importing a value from it would pull the loader stack into a
 * client that has no other use for it.
 *
 * It has to be this exact value. `ImageLayer.updateState` ends its promise
 * chain with `catch(e => { if (e !== SIGNAL_ABORTED) throw e })`, so any other
 * rejection -- an `AbortError`, our own `TensorAbortError` -- is rethrown from
 * inside a `.catch` nobody follows, and surfaces as an unhandled rejection.
 * Viv's own `TiffPixelSource` throws the same value for the same reason.
 */
const VIV_SIGNAL_ABORTED = "__vivSignalAborted";

interface RasterRequest {
  /** The selection this was started for; the generation marker. */
  key: string;
  promise: Promise<PixelData>;
  controller: AbortController;
  /** Callers still waiting. At zero the answer is wanted by nobody. */
  waiters: number;
}

/**
 * One in-flight `getRaster` per (level, selection), shared by every caller, and
 * only ever for the newest selection.
 *
 * Two things go wrong without this, both from Viv's `ImageLayer.updateState`
 * making a *new* `AbortController` per selection and dropping the old one
 * un-aborted (only `finalizeState` aborts, and only the current one):
 *
 *  - superseded reads run to completion, or to `chunkTimeoutMs`, and survive
 *    unmount. A scrub accumulates them without bound, and past the browser's
 *    six-connections-per-origin cap the newest and most-wanted read queues
 *    behind stale ones nobody wants.
 *  - the coarsest level is read twice per selection change, by Viv's background
 *    `ImageLayer` and by the viewer's contrast sampling, neither aware of the
 *    other.
 *
 * A raster is only ever wanted for the current view, so superseding is the
 * correct semantics and not merely a convenient one. Each caller keeps its own
 * signal: one caller aborting detaches only itself, and the shared request is
 * aborted when its last waiter leaves.
 */
class RasterRequests {
  private readonly inFlight = new Map<string, RasterRequest>();
  private generation: string | null = null;

  run(
    level: number,
    key: string,
    start: (signal: AbortSignal) => Promise<PixelData>,
    caller?: AbortSignal,
  ): Promise<PixelData> {
    // Before anything else, and before touching any shared state: a caller that
    // has already given up must not supersede the generation on the strength of
    // a selection it no longer wants, nor start a read only to abort it.
    if (caller?.aborted) return Promise.reject(VIV_SIGNAL_ABORTED);

    if (key !== this.generation) {
      this.generation = key;
      for (const [id, req] of this.inFlight) {
        if (req.key !== key) {
          this.inFlight.delete(id);
          req.controller.abort(VIV_SIGNAL_ABORTED);
        }
      }
    }

    const id = `${level}@${key}`;
    let entry = this.inFlight.get(id);
    if (!entry) {
      const controller = new AbortController();
      const started: RasterRequest = {
        key,
        controller,
        waiters: 0,
        promise: start(controller.signal),
      };
      this.inFlight.set(id, started);
      // Both arms: clears the slot on failure too, and keeps the shared promise
      // from counting as unhandled when every waiter has already detached.
      const release = () => {
        if (this.inFlight.get(id) === started) this.inFlight.delete(id);
      };
      started.promise.then(release, release);
      entry = started;
    }
    return this.join(id, entry, caller);
  }

  private join(id: string, entry: RasterRequest, caller?: AbortSignal): Promise<PixelData> {
    entry.waiters += 1;
    const leave = () => {
      entry.waiters -= 1;
      if (entry.waiters <= 0 && this.inFlight.get(id) === entry) {
        this.inFlight.delete(id);
        entry.controller.abort(VIV_SIGNAL_ABORTED);
      }
    };

    // `run` has already rejected an aborted caller, and nothing awaits between
    // there and here, so the signal can only fire from now on.
    return new Promise<PixelData>((resolve, reject) => {
      let settled = false;
      const onAbort = () => {
        if (settled) return;
        settled = true;
        detach();
        leave();
        reject(VIV_SIGNAL_ABORTED);
      };
      const detach = () => caller?.removeEventListener("abort", onAbort);
      caller?.addEventListener("abort", onAbort, { once: true });

      entry.promise.then(
        (data) => {
          if (settled) return;
          settled = true;
          detach();
          entry.waiters -= 1;
          resolve(data);
        },
        (err) => {
          if (settled) return;
          settled = true;
          detach();
          entry.waiters -= 1;
          reject(isAbort(err) ? VIV_SIGNAL_ABORTED : err);
        },
      );
    });
  }
}

function isAbort(err: unknown): boolean {
  return (
    err === VIV_SIGNAL_ABORTED ||
    err instanceof TensorAbortError ||
    (err instanceof Error && err.name === "AbortError")
  );
}

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
 *
 * `arrayId` is the whole address: `source_id` for a single-tensor source,
 * `source_id/field` otherwise.
 */
export async function createTensorPixelSources(
  client: TensorHttpClient,
  arrayId: string,
  options: TensorPixelSourceOptions = {},
): Promise<TensorPixelSources> {
  const info = await client.tileInfo(arrayId, { signal: options.signal });
  return { data: pixelSourcesFromInfo(client, info, options), info };
}

/** The pure half of {@link createTensorPixelSources}, for an already-fetched grid. */
export function pixelSourcesFromInfo(
  client: TensorHttpClient,
  info: TileInfo,
  options: TensorPixelSourceOptions = {},
): PixelSource<string[]>[] {
  const unreachable = unreachableAxes(info);
  if (unreachable.length) {
    // Refusing the tensor, not the axis. There is no second viewer to fall
    // back to any more, so this costs the tensor its display against an old
    // server -- accepted deliberately: the alternative is a slider that scrolls
    // through 155 copies of frame 0, and silently wrong pixels are worse than a
    // stated refusal. The window is a new SPA against an older sidecar; both
    // ship from the same release.
    const named = unreachable
      .map((axis) => `${axis.title} (${axis.extent} positions)`)
      .join(", ");
    throw new Error(
      `Tensor ${info.array_id} has an axis this server's tile route cannot ` +
        `select: ${named}. Addressing it needs the \`sel\` parameter, which this ` +
        `server predates — it would answer every position with index 0.`,
    );
  }
  const labels = vivLabels(info);
  const dtype = vivDtype(info.dtype);
  // Shared across the levels of this image, so a selection change supersedes
  // reads on every level and not just the one being asked again.
  const requests = new RasterRequests();
  return info.levels.map((level) =>
    makeSource(client, info, level, labels, dtype, options, requests),
  );
}

function makeSource(
  client: TensorHttpClient,
  info: TileInfo,
  level: TileLevel,
  labels: string[],
  dtype: SupportedDtype,
  options: TensorPixelSourceOptions,
  requests: RasterRequests,
): PixelSource<string[]> {
  const { plane } = info;
  // Values per pixel: >1 only for an interleaved RGB(A) samples axis.
  const samples = plane.s === null ? 1 : Math.max(1, info.shape[plane.s] ?? 1);
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
    // `meta` is optional in Viv's own type, but `ImageLayer.renderLayers` reads
    // `loader.meta.photometricInterpretation` unguarded on the interleaved
    // branch -- the destructuring default only covers a missing property, not a
    // missing object. Without this an RGB(A) tensor throws inside deck.gl and
    // renders nothing at all, silently: the tiles are fetched, the canvas stays
    // empty, and the failure only shows up in the console.
    //
    // 2 is TIFF PhotometricInterpretation RGB, which is what the server sends
    // for an interleaved samples axis and what Viv itself defaults to.
    meta: { photometricInterpretation: 2 },

    async getTile({ x, y, selection, signal }: TileSelection<string[]>): Promise<PixelData> {
      // deck.gl can ask for a tile past the edge while the viewport settles.
      // Answering locally with zeros costs no round trip and keeps the layer
      // from treating a routine 404 as a load failure.
      if (x < 0 || y < 0 || x >= level.cols || y >= level.rows) {
        return emptyTile(info.tile_size, dtype, samples);
      }
      const result = await client.tile(
        {
          // The array_id the grid was measured from, so geometry and pixels
          // cannot address different tensors.
          array_id: info.array_id,
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
      // The key is the resolved selection, not the caller's selection object:
      // the background layer and the contrast sampler build their own objects
      // for the same plane, and only the resolved form makes those the same
      // read. It must cover `sel` too -- keyed on t/z/c alone, every frame of a
      // TIFF sequence shares one key, so the second plane read would be served
      // the first plane's in-flight promise.
      const key =
        `t${sel.t ?? 0}/z${sel.z ?? 0}/c${sel.c ?? 0}/` +
        (sel.sel ?? []).map(([axis, index]) => `${axis}:${index}`).join(",");

      // A level that fits one tile is one tile. `_tile_levels` stops halving as
      // soon as the plane fits the edge, so the coarsest level always has
      // cols === rows === 1 -- and that is the only level Viv reads a raster
      // from (the background ImageLayer and the contrast sampler both take
      // `loader[loader.length - 1]`). The tile route answers with identical
      // bytes and, unlike POST /api/slice, is cacheable: ETag plus
      // `Cache-Control: private, max-age=3600`. So a revisited plane costs no
      // backend read at all instead of one uncacheable read of the whole
      // coarsest level.
      if (level.cols === 1 && level.rows === 1) {
        return requests.run(
          level.level,
          key,
          async (shared) => {
            const result = await client.tile(
              {
                array_id: info.array_id,
                level: level.level,
                col: 0,
                row: 0,
                ...sel,
              },
              { signal: shared },
            );
            return { data: asTypedArray(result.buffer, dtype), ...planeOf(result.shape) };
          },
          signal,
        );
      }

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
      return requests.run(
        level.level,
        key,
        async (shared) => {
          const arr = await client.slice(
            {
              array_id: info.array_id,
              slice_start: start,
              slice_stop: stop,
              scale_hint: scaleHint,
            },
            { signal: shared },
          );
          return { data: asTypedArray(arr.buffer, dtype), ...planeOf(arr.shape) };
        },
        signal,
      );
    },

    onTileError(err: Error) {
      // A tile the viewport moved past is a normal outcome, not a failure.
      if (err instanceof TensorAbortError || err.name === "AbortError") return;
      if (options.onTileError) options.onTileError(err);
      else console.error(`tile error (${info.array_id})`, err);
    },
  };
}

/**
 * A zero tile for an out-of-grid request, shaped like a real one.
 *
 * `samples` is load-bearing: an interleaved RGB(A) tensor carries that many
 * values per pixel, so a plain `tileSize * tileSize` buffer is a third (or a
 * quarter) of what the layer will upload. WebGL then rejects the texture or
 * reads past the end -- and only when a viewport pans off the edge of an RGB
 * image, which is ordinary interaction rather than an edge case.
 */
function emptyTile(
  tileSize: number,
  dtype: SupportedDtype,
  samples: number,
): PixelData {
  const data = new TYPED_ARRAY_BY_DTYPE[dtype](tileSize * tileSize * samples);
  return { data: data as SupportedTypedArray, width: tileSize, height: tileSize };
}

/** A selection resolved into the two forms the tile route accepts. */
export interface TileParams {
  t?: number;
  z?: number;
  c?: number;
  /** Axes with no name, addressed by wire index. */
  sel?: Array<[number, number]>;
}

/**
 * Viv's label-keyed selection -> the tile route's parameters.
 *
 * Two spellings, one rule for choosing between them: an axis goes under its
 * *name* when the server says that name resolves to this very axis
 * (`info.selectable`), and under `sel` — by wire index — when it does not. The
 * server refuses an axis sent both ways, so the choice has to be exclusive
 * rather than belt-and-braces.
 *
 * Asking `selectable` rather than trusting this side's own naming is what keeps
 * a disagreement harmless. The two resolvers share a vocabulary and should
 * always agree, but if they ever did not, the axis simply falls to `sel` — which
 * addresses it correctly regardless of what either side would have called it. A
 * drift becomes a wrong slider *title*, never a wrong plane.
 */
function tileSelection(
  info: TileInfo,
  selection: PixelSourceSelection<string[]>,
): TileParams {
  const out: TileParams = {};
  const byKey = new Map(
    sliderAxes(info.dim_labels, info.shape).map((axis) => [axis.key, axis]),
  );
  const sel: Array<[number, number]> = [];

  for (const [label, rawIndex] of Object.entries(selection ?? {})) {
    if (label === "y" || label === "x" || label === "_c") continue;
    const index = Number(rawIndex ?? 0);
    const axis = byKey.get(label);
    if (!axis) {
      // A label naming no axis of this tensor. Index 0 is the default every
      // client sends, so only a non-zero one is a mistake worth reporting --
      // the same exemption the server applies to `t`/`z`/`c`.
      if (index !== 0) {
        throw new Error(
          `Tensor ${info.array_id} has no "${label}" axis to select ` +
            `(it has ${[...byKey.keys()].join(", ") || "none"})`,
        );
      }
      continue;
    }
    if (axis.named && info.selectable[axis.named] === axis.axis) {
      out[axis.named] = index;
    } else {
      sel.push([axis.axis, index]);
    }
  }
  if (sel.length) {
    // Unreachable through `pixelSourcesFromInfo`, which refuses such a tensor
    // up front. Kept because the cost of being wrong here is silent wrong
    // pixels: this also covers the drift path above, where a *named* axis falls
    // through to `sel` because the two sides disagree about its name.
    if (!supportsSelParameter(info)) {
      throw new Error(
        `Tensor ${info.array_id} needs the \`sel\` parameter to select ` +
          `axis ${sel.map(([axis]) => axis).join(", ")}, which this server predates`,
      );
    }
    out.sel = sel;
  }
  return out;
}

/** The index the caller chose for wire axis `i`, or 0. */
function sliderIndexAt(info: TileInfo, params: TileParams, i: number): number {
  for (const axis of ["t", "z", "c"] as const) {
    if (info.selectable[axis] === i && params[axis] !== undefined) return params[axis];
  }
  for (const [axis, index] of params.sel ?? []) {
    if (axis === i) return index;
  }
  return 0;
}
