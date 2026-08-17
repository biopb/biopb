/**
 * HTTP client for the BioPB Tensor FastAPI sidecar.
 *
 * Communicates with the Python FastAPI sidecar (default :8816) over plain
 * HTTP/JSON + binary.  All protected endpoints require a token passed as
 * ``Authorization: Bearer <token>`` or ``X-Biopb-Token``.
 *
 * Usage:
 *   const client = new TensorHttpClient("http://localhost:8816", token);
 *   const sources = await client.listSources();
 *   const arr = await client.slice({ source_id: "...", tensor_id: "...", ... });
 */

import type {
  AdminConfigResponse,
  AdminConfigSaveResult,
  AdminStatus,
  BrowseResponse,
  DataSourceDescriptor,
  DiagnosticsSnapshot,
  QuerySourcesResult,
  ReadyzSnapshot,
  RenderRequest,
  RenderResult,
  SliceRequest,
  TileImageRequest,
  TileInfo,
  TileRequest,
  TileResult,
  TypedNdArray,
} from "./types.js";

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

export class TensorApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
    public readonly detail?: unknown,
  ) {
    // Include detail in message if it's a string or has a 'detail' property
    const detailStr = typeof detail === 'string'
      ? detail
      : (detail as Record<string, unknown>)?.detail as string | undefined;
    const fullMessage = detailStr ? `${message}: ${detailStr}` : message;
    super(`TensorApi ${status}: ${fullMessage}`);
    this.name = "TensorApiError";
  }
}

/**
 * The caller aborted this request through its own signal.
 *
 * Deliberately NOT a {@link TensorApiError}: a tile the user panned away from is
 * a normal outcome, not a failure, and reporting it as the 408 timeout the
 * helpers used to synthesise put self-inflicted cancellations into the error UI
 * and the retry path. Callers discard these silently.
 *
 * `name` stays `"AbortError"` so the standard `err.name === "AbortError"` idiom
 * -- what deck.gl/Viv tile layers use to drop a cancelled tile -- keeps working.
 */
export class TensorAbortError extends Error {
  constructor(
    public readonly path: string,
    public readonly reason?: unknown,
  ) {
    super(`Request aborted by caller (${path})`);
    this.name = "AbortError";
  }
}

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------

/** Per-call options accepted by every read method. */
export interface RequestOptions {
  /**
   * Caller's cancellation signal, composed with the method's own timeout.
   *
   * Aborting drops the connection, which is what lets the server skip a read it
   * has not started yet (it answers 499 and counts `cancelled_reads`); without
   * it a viewer that pans away still pays for every tile it asked for.
   */
  signal?: AbortSignal;
}

interface ComposedSignal {
  signal: AbortSignal;
  cleanup: () => void;
}

/**
 * One signal that fires on either the timeout or the caller's abort.
 *
 * Hand-wired rather than `AbortSignal.any()`, which is too recent to rely on
 * here (this package still works around Safari versions predating it).
 * Listeners are removed on the way out so a long-lived caller signal -- one
 * AbortController per viewport, reused across many tiles -- cannot accumulate
 * them.
 */
function composeSignal(timeoutMs: number | undefined, caller?: AbortSignal): ComposedSignal {
  const controller = new AbortController();
  const cleanups: Array<() => void> = [];

  if (timeoutMs != null) {
    const id = setTimeout(() => controller.abort(), timeoutMs);
    cleanups.push(() => clearTimeout(id));
  }

  if (caller) {
    if (caller.aborted) {
      controller.abort(caller.reason);
    } else {
      const onAbort = () => controller.abort(caller.reason);
      caller.addEventListener("abort", onAbort);
      cleanups.push(() => caller.removeEventListener("abort", onAbort));
    }
  }

  return {
    signal: controller.signal,
    cleanup: () => { for (const fn of cleanups) fn(); },
  };
}

/**
 * Turn a fetch rejection into the right error.
 *
 * Order matters: an abort raised by the caller and one raised by the timeout are
 * the same `AbortError` on the wire, and only the caller's signal distinguishes
 * them.
 */
function abortAwareError(e: unknown, path: string, timeoutMs?: number, caller?: AbortSignal): unknown {
  if (e instanceof Error && e.name === "AbortError") {
    if (caller?.aborted) return new TensorAbortError(path, caller.reason);
    return new TensorApiError(408, `Timeout after ${timeoutMs}ms (${path})`);
  }
  return e;
}

/** Reject on a non-2xx response, unwrapping the server's JSON detail. */
async function assertOk(res: Response): Promise<void> {
  if (res.ok) return;
  // An HTML body here is the reverse proxy's error page, not the sidecar's:
  // the data plane is still starting and has nothing to say yet.
  const contentType = res.headers.get("content-type") ?? "";
  if (contentType.includes("text/html")) {
    throw new TensorApiError(
      res.status,
      "Server unavailable - may be starting up. Please wait and retry.",
    );
  }
  let detail: unknown;
  try { detail = await res.json(); } catch { /* ignore */ }
  throw new TensorApiError(res.status, res.statusText, detail);
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export class TensorHttpClient {
  private readonly base: string;
  private readonly token: string | null;

  /** In-flight {@link getSourceMetadata} calls, so simultaneous callers share one. */
  private readonly metadataInFlight = new Map<string, Promise<Record<string, unknown>>>();

  /** Timeout for metadata / listing requests (ms). */
  metadataTimeoutMs = 3_000;
  /** Timeout for binary chunk/slice requests (ms). */
  chunkTimeoutMs = 8_000;

  /**
   * @param apiBase   Base URL of the FastAPI sidecar, e.g. "http://localhost:8816".
   * @param token     Website token.  Pass null or "" to skip auth header
   *                  (dev-mode bypass on the server side).
   */
  constructor(apiBase: string, token: string | null) {
    this.base = apiBase.replace(/\/$/, "");
    this.token = token || null;
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  private headers(extra?: Record<string, string>): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.token) {
      h["Authorization"] = `Bearer ${this.token}`;
    }
    return { ...h, ...extra };
  }

  private async fetchJson<T>(
    path: string,
    options?: RequestInit,
    timeoutMs?: number,
    opts?: RequestOptions,
  ): Promise<T> {
    return this.send(path, {
      ...options,
      headers: { ...this.headers(), ...(options?.headers as Record<string, string> ?? {}) },
    }, timeoutMs, opts, (res) => res.json() as Promise<T>);
  }

  private async fetchBinary<T>(
    path: string,
    body: unknown,
    timeoutMs: number | undefined,
    opts: RequestOptions | undefined,
    consume: (res: Response) => Promise<T>,
  ): Promise<T> {
    return this.send(path, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    }, timeoutMs, opts, consume);
  }

  private async fetchJsonWithHeaders<T>(
    path: string,
    body: unknown,
    timeoutMs?: number,
    opts?: RequestOptions,
  ): Promise<{ data: T; headers: Headers }> {
    return this.send(path, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    }, timeoutMs, opts, async (res) => ({ data: await res.json() as T, headers: res.headers }));
  }

  /**
   * The one place a request is actually issued: composes the caller's signal
   * with the timeout, checks the status, reads the body, and maps an abort to
   * the error that says which of the two fired.
   *
   * `consume` reads the body HERE, inside the guard, rather than the caller
   * doing it afterwards. That is the whole point of the callback: `fetch`
   * resolves on the response *headers* and the body streams after, so cleaning
   * up once the Response object exists would clear the timeout and detach the
   * caller's abort for the entire body phase -- the expensive part. A 512 KB
   * tile whose headers arrive in a millisecond was, measurably, uncancellable
   * and un-timeout-able while its bytes were in flight.
   */
  private async send<T>(
    path: string,
    init: RequestInit,
    timeoutMs: number | undefined,
    opts: RequestOptions | undefined,
    consume: (res: Response) => Promise<T>,
  ): Promise<T> {
    const composed = composeSignal(timeoutMs, opts?.signal);
    try {
      const res = await fetch(`${this.base}${path}`, { ...init, signal: composed.signal });
      await assertOk(res);
      return await consume(res);
    } catch (e) {
      throw abortAwareError(e, path, timeoutMs, opts?.signal);
    } finally {
      composed.cleanup();
    }
  }


  // -------------------------------------------------------------------------
  // Health (no auth required)
  // -------------------------------------------------------------------------

  async livez(opts?: RequestOptions): Promise<{ status: string; timestamp: string }> {
    return this.fetchJson("/livez", undefined, this.metadataTimeoutMs, opts);
  }

  async readyz(opts?: RequestOptions): Promise<ReadyzSnapshot> {
    return this.fetchJson("/readyz", undefined, this.metadataTimeoutMs, opts);
  }

  // -------------------------------------------------------------------------
  // Sources
  // -------------------------------------------------------------------------

  /** List all data sources registered with the server. */
  async listSources(opts?: RequestOptions): Promise<DataSourceDescriptor[]> {
    return this.fetchJson<DataSourceDescriptor[]>(
      "/api/sources",
      undefined,
      this.metadataTimeoutMs,
      opts,
    );
  }

  /** Get a single DataSourceDescriptor by source_id. */
  async getSource(sourceId: string, opts?: RequestOptions): Promise<DataSourceDescriptor> {
    return this.fetchJson<DataSourceDescriptor>(
      `/api/sources/${encodeURIComponent(sourceId)}`,
      undefined,
      this.metadataTimeoutMs,
      opts,
    );
  }

  /**
   * Get the parsed OME-NGFF metadata for a source.
   * Returns an empty object if the source has no metadata.
   *
   * Concurrent callers for the same source share one request. Selecting a
   * source asks twice in the same tick -- the metadata panel and the channel-name
   * loader are independent components -- and a MicroManager per-frame blob is
   * 13.8 MB, so the second ask is a duplicate download and a duplicate parse.
   * Nothing is retained past the response: a re-indexed source still reports
   * whatever it reports now.
   */
  async getSourceMetadata(sourceId: string, opts?: RequestOptions): Promise<Record<string, unknown>> {
    const path = `/api/sources/${encodeURIComponent(sourceId)}/metadata`;
    // A caller with its own signal gets its own request: aborting a shared one
    // would cancel it out from under whoever else is waiting on it.
    if (opts?.signal) {
      return this.fetchJson<Record<string, unknown>>(path, undefined, this.metadataTimeoutMs, opts);
    }

    const pending = this.metadataInFlight.get(sourceId);
    if (pending) return pending;

    const request = this.fetchJson<Record<string, unknown>>(
      path,
      undefined,
      this.metadataTimeoutMs,
      opts,
    );
    this.metadataInFlight.set(sourceId, request);
    const release = () => {
      if (this.metadataInFlight.get(sourceId) === request) this.metadataInFlight.delete(sourceId);
    };
    // Both arms, so a failure clears the slot and a rejection is never left
    // unhandled on this branch of the chain.
    request.then(release, release);
    return request;
  }

  /**
   * Execute SQL query against server's source metadata database.
   *
   * @param sql SQL query (e.g., "SELECT source_id FROM sources WHERE source_type='ome-zarr'")
   * @returns Query result with rows and truncation metadata
   * @throws {TensorApiError} on validation error or timeout
   */
  async querySources(sql: string, opts?: RequestOptions): Promise<QuerySourcesResult> {
    const { data, headers } = await this.fetchJsonWithHeaders<Record<string, unknown>[]>(
      "/api/sources/query",
      { sql },
      this.metadataTimeoutMs,
      opts,
    );

    const totalSources = parseInt(headers.get("X-Total-Sources") ?? "0", 10);
    const returnedSources = parseInt(headers.get("X-Returned-Sources") ?? String(data.length), 10);
    const truncated = headers.get("X-Truncated") === "true";

    return { rows: data, totalSources, returnedSources, truncated };
  }

  // -------------------------------------------------------------------------
  // Admin (config read/write, status, restart)
  // -------------------------------------------------------------------------

  /** Read the on-disk config, its path, and the JSON Schema (`GET /api/config`). */
  async getAdminConfig(opts?: RequestOptions): Promise<AdminConfigResponse> {
    return this.fetchJson<AdminConfigResponse>(
      "/api/config",
      undefined,
      this.metadataTimeoutMs,
      opts,
    );
  }

  /**
   * Validate and write the config (`PUT /api/config`). Does NOT restart.
   *
   * On a schema-validation failure the server returns `422` and `fetchJson`
   * throws a {@link TensorApiError} whose `.detail` is the
   * {@link AdminConfigValidationBody} (`{detail, errors}`); callers render
   * `error.detail.errors` inline.
   *
   * Takes no cancellation signal, unlike the read methods: aborting a write
   * leaves the caller unable to say whether the server applied it, which is a
   * worse position than waiting.
   */
  async putAdminConfig(config: Record<string, unknown>): Promise<AdminConfigSaveResult> {
    return this.fetchJson<AdminConfigSaveResult>(
      "/api/config",
      { method: "PUT", body: JSON.stringify(config) },
      this.metadataTimeoutMs,
    );
  }

  /** Backend health merged with process facts (`GET /api/admin/status`). */
  async getAdminStatus(opts?: RequestOptions): Promise<AdminStatus> {
    return this.fetchJson<AdminStatus>(
      "/api/admin/status",
      undefined,
      this.metadataTimeoutMs,
      opts,
    );
  }

  /**
   * List one directory of the server's filesystem (`GET /api/admin/browse`).
   *
   * Local-mode only (biopb/biopb#244): in remote mode the server returns 404 and
   * `fetchJson` throws a {@link TensorApiError}. Callers gate the UI on
   * {@link AdminStatus.local} so this is only invoked when available. A blank
   * `path` starts at the server user's home directory.
   */
  async browse(path?: string, opts?: RequestOptions): Promise<BrowseResponse> {
    const qs = path ? `?path=${encodeURIComponent(path)}` : "";
    return this.fetchJson<BrowseResponse>(
      `/api/admin/browse${qs}`,
      undefined,
      this.metadataTimeoutMs,
      opts,
    );
  }

  // -------------------------------------------------------------------------
  // Slice
  // -------------------------------------------------------------------------

  /**
   * Fetch a sub-region of a tensor as raw bytes.
   *
   * The server returns C-contiguous numpy bytes; shape, dtype, and dim labels
   * are in response headers ``X-Shape``, ``X-Dtype``, ``X-Dim-Labels``.
   *
   * @throws {TensorApiError} on HTTP error or timeout.
   * @throws {TensorAbortError} if `opts.signal` fired.
   */
  async slice(req: SliceRequest, opts?: RequestOptions): Promise<TypedNdArray> {
    return this.fetchBinary("/api/slice", req, this.chunkTimeoutMs, opts, readNdArray);
  }

  // -------------------------------------------------------------------------
  // Tiles
  // -------------------------------------------------------------------------

  /**
   * Grid, pyramid levels and selectable axes for a tensor
   * (`GET /api/tile_info`).
   *
   * Addressed by `array_id` alone (`source_id` for a single-tensor source,
   * `source_id/field` otherwise) -- the tensor identity policy, not the
   * deprecated `(source_id, tensor_id)` pair the slice/render routes still take.
   *
   * Fetch once per tensor and keep it: `tile_size` comes from the stored
   * `chunk_shape`, so it varies per tensor and must not be assumed.
   */
  async tileInfo(arrayId: string, opts?: RequestOptions): Promise<TileInfo> {
    return this.fetchJson<TileInfo>(
      `/api/tile_info/${encodeArrayId(arrayId)}`,
      undefined,
      this.metadataTimeoutMs,
      opts,
    );
  }

  /**
   * One tile as raw bytes (`GET /api/tile`, `fmt=raw`).
   *
   * A cacheable GET, so a tile already seen is served by the browser cache
   * without reaching the network -- which is the point of addressing pixels this
   * way rather than through `slice()`. Pass `opts.signal` for tiles that may
   * leave the viewport before they land: aborting also lets the server skip the
   * read if it has not started it.
   *
   * @throws {TensorAbortError} if `opts.signal` fired.
   */
  async tile(req: TileRequest, opts?: RequestOptions): Promise<TileResult> {
    return this.fetchGet(this.tilePath(req, {}), this.chunkTimeoutMs, opts, async (res) => ({
      ...(await readNdArray(res)),
      tileSize: parseInt(res.headers.get("X-Tile-Size") ?? "0", 10),
      level: parseInt(res.headers.get("X-Tile-Level") ?? "0", 10),
      col: parseInt(res.headers.get("X-Tile-Col") ?? "0", 10),
      row: parseInt(res.headers.get("X-Tile-Row") ?? "0", 10),
    }));
  }

  /**
   * One tile rendered server-side (`GET /api/tile`, `fmt=png|jpeg`).
   *
   * The same tile as {@link tile}, with appearance baked in: far fewer bytes,
   * at the cost of making contrast part of the cache key. Intended for slow
   * links and high channel counts, not as a separate rendering path.
   */
  async tileImage(req: TileImageRequest, opts?: RequestOptions): Promise<Blob> {
    const { fmt = "jpeg", lo, hi, color, use_min_max } = req;
    return this.fetchGet(
      this.tilePath(req, { fmt, lo, hi, color, use_min_max }),
      this.chunkTimeoutMs * 2,
      opts,
      (res) => res.blob(),
    );
  }

  /** Build a tile URL. Every parameter that decides the bytes is in it, by design. */
  private tilePath(req: TileRequest, extra: Record<string, unknown>): string {
    const qs = new URLSearchParams();
    const params: Record<string, unknown> = {
      level: req.level,
      col: req.col,
      row: req.row,
      t: req.t,
      z: req.z,
      c: req.c,
      reduction_method: req.reduction_method,
      ...extra,
    };
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) qs.set(k, String(v));
    }
    const query = qs.toString();
    return `/api/tile/${encodeArrayId(req.array_id)}${query ? `?${query}` : ""}`;
  }

  private async fetchGet<T>(
    path: string,
    timeoutMs: number | undefined,
    opts: RequestOptions | undefined,
    consume: (res: Response) => Promise<T>,
  ): Promise<T> {
    return this.send(path, { method: "GET", headers: this.headers() }, timeoutMs, opts, consume);
  }

  // -------------------------------------------------------------------------
  // Render (backend image rendering)
  // -------------------------------------------------------------------------

  /**
   * Fetch a rendered image from the backend.
   *
   * Uses server-side VTK/PIL rendering to produce PNG/JPEG output.
   * For raw format, returns RGBA bytes (4 bytes per pixel, uint8).
   * This is an alternative to slice() + frontend rendering.
   *
   * @throws {TensorApiError} on HTTP error, timeout, or if rendering not enabled.
   * @throws {TensorAbortError} if `opts.signal` fired.
   */
  async render(req: RenderRequest, opts?: RequestOptions): Promise<RenderResult> {
    // Use longer timeout for rendering (may be slower than raw slice)
    return this.fetchBinary("/api/render", req, this.chunkTimeoutMs * 2, opts, async (res) => {
      const width = parseInt(res.headers.get("X-Image-Width") ?? "0", 10);
      const height = parseInt(res.headers.get("X-Image-Height") ?? "0", 10);
      const percentileLoValue = parseFloat(res.headers.get("X-Percentile-Lo-Value") ?? "0");
      const percentileHiValue = parseFloat(res.headers.get("X-Percentile-Hi-Value") ?? "1");
      const format = res.headers.get("X-Image-Format") ?? req.output_format ?? "jpeg";

      // For raw format, use arrayBuffer; for png/jpeg, use blob
      const blob = format === "raw" ? await res.arrayBuffer() : await res.blob();

      return { blob, width, height, percentileLoValue, percentileHiValue, format };
    });
  }

  // -------------------------------------------------------------------------
  // Diagnostics
  // -------------------------------------------------------------------------

  async diagnostics(opts?: RequestOptions): Promise<DiagnosticsSnapshot> {
    return this.fetchJson<DiagnosticsSnapshot>(
      "/api/diagnostics",
      undefined,
      this.metadataTimeoutMs,
      opts,
    );
  }
}

/**
 * Percent-encode an array_id for a path, keeping its `/` separators.
 *
 * `encodeURIComponent` would turn the field separator into `%2F`. The server
 * decodes that back, so it happens to work, but it makes two spellings of one
 * tile -- and therefore two browser-cache entries. Encode per segment instead.
 */
function encodeArrayId(arrayId: string): string {
  return arrayId.split("/").map(encodeURIComponent).join("/");
}

/** Read the shape/dtype/label headers the binary routes share into an ndarray. */
async function readNdArray(res: Response): Promise<TypedNdArray> {
  const shape = (res.headers.get("X-Shape") ?? "").split(",").filter(Boolean).map(Number);
  const dtype = res.headers.get("X-Dtype") ?? "";
  const dimLabels = (res.headers.get("X-Dim-Labels") ?? "").split(",").filter(Boolean);
  return { buffer: await res.arrayBuffer(), shape, dtype, dimLabels };
}
