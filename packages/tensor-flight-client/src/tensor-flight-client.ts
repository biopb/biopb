/**
 * High-level facade mirroring the Python TensorFlightClient API.
 *
 * Usage:
 *   const client = new TensorFlightClient("http://localhost:8816", token);
 *   const sources = await client.listSources();
 *   const arr = client.getTensor("my-source", "tensor-0");
 *   const data = await arr.compute({ z: 5, c: 0, scaleHint: [1,1,1,8,8] });
 */

import { TensorHttpClient } from "./client.js";
import { TensorArray, buildAxisMap, isAxisMapAmbiguous } from "./tensor-array.js";
import type { DataSourceDescriptor } from "./types.js";

export class TensorFlightClient {
  private readonly _http: TensorHttpClient;
  /** Source cache populated by listSources(). */
  private _sources: Map<string, DataSourceDescriptor> = new Map();

  /**
   * @param apiBase  Base URL of the FastAPI HTTP sidecar, e.g.
   *                 "http://localhost:8816".
   * @param token    Website access token (null/empty for dev-mode bypass).
   */
  constructor(apiBase = "http://localhost:8816", token: string | null = null) {
    this._http = new TensorHttpClient(apiBase, token);
  }

  /** Expose the underlying HTTP client for direct use. */
  get http(): TensorHttpClient {
    return this._http;
  }

  // -------------------------------------------------------------------------
  // API
  // -------------------------------------------------------------------------

  /** List all data sources from the server. */
  async listSources(): Promise<DataSourceDescriptor[]> {
    const sources = await this._http.listSources();
    this._sources = new Map(sources.map((s) => [s.source_id, s]));
    return sources;
  }

  /**
   * Get source-level OME/vendor metadata as a plain JS object.
   * Returns {} if no metadata is available.
   */
  async getSourceMetadata(sourceId: string): Promise<Record<string, unknown>> {
    return this._http.getSourceMetadata(sourceId);
  }

  /**
   * Return a lazy TensorArray for the given source + tensor.
   *
   * If the source has already been fetched (via listSources), the descriptor
   * is resolved from the local cache.  Otherwise a single getSource() call
   * is made to populate it.
   *
   * This method is synchronous-first for the cache-hit path; the returned
   * TensorArray only issues network requests when .compute() is called.
   */
  getTensor(sourceId: string, tensorId: string): TensorArray {
    const cached = this._sources.get(sourceId);
    if (cached) {
      const td = cached.tensors.find((t) => t.array_id === tensorId);
      if (td) return new TensorArray(this._http, sourceId, td);
    }
    // Return a "pending" proxy — actual descriptor resolved lazily
    return new LazyTensorArray(this._http, sourceId, tensorId, this._sources);
  }
}

// ---------------------------------------------------------------------------
// LazyTensorArray: resolves descriptor on first compute()
// ---------------------------------------------------------------------------

/**
 * TensorArray whose descriptor is fetched lazily on the first .compute().
 * Used when getTensor() is called before listSources().
 */
class LazyTensorArray extends TensorArray {
  /** Single shared resolution promise — prevents concurrent duplicate getSource() calls. */
  private _resolvePromise: Promise<void> | null = null;
  private readonly _pendingSourceId: string;
  private readonly _pendingTensorId: string;
  private readonly _sourceCache: Map<string, DataSourceDescriptor>;

  constructor(
    client: TensorHttpClient,
    sourceId: string,
    tensorId: string,
    sourceCache: Map<string, DataSourceDescriptor>,
  ) {
    // Placeholder descriptor — replaced on first compute() via _doResolve()
    super(client, sourceId, {
      array_id: tensorId,
      dim_labels: [],
      shape: [],
      chunk_shape: [],
      dtype: "uint8",
    });
    this._pendingSourceId = sourceId;
    this._pendingTensorId = tensorId;
    this._sourceCache = sourceCache;
  }

  override async compute(options = {}): Promise<import("./types.js").TypedNdArray> {
    this._resolvePromise ??= this._doResolve();
    await this._resolvePromise;
    return super.compute(options);
  }

  private async _doResolve(): Promise<void> {
    const source = await this._client.getSource(this._pendingSourceId);
    this._sourceCache.set(source.source_id, source);
    const td = source.tensors.find((t) => t.array_id === this._pendingTensorId);
    if (!td) {
      throw new Error(
        `Tensor '${this._pendingTensorId}' not found in source '${this._pendingSourceId}'`,
      );
    }
    this._descriptor = td;
    this._axisMap = buildAxisMap(td.dim_labels);
    this._axisMapAmbiguous = isAxisMapAmbiguous(td.dim_labels);
  }
}
