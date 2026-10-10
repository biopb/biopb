/**
 * High-level facade mirroring the Python TensorFlightClient API.
 *
 * Usage:
 *   const client = new TensorFlightClient("http://localhost:8816", token);
 *   const { sources } = await client.listSourcesPage(20000);
 *   const arr = client.getTensor("my-source/tensor-0");
 *   const data = await arr.compute({ z: 5, c: 0, scaleHint: [1,1,1,8,8] });
 */

import { TensorHttpClient } from "./client.js";
import { TensorArray, buildAxisMap, isAxisMapAmbiguous } from "./tensor-array.js";
import type { DataSourceDescriptor, SourceListing } from "./types.js";

export class TensorFlightClient {
  private readonly _http: TensorHttpClient;
  /** Source cache populated by listSourcesPage(). */
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

  /** List at most `limit` data sources, and whether the catalog is longer. */
  async listSourcesPage(limit: number): Promise<SourceListing> {
    const listing = await this._http.listSourcesPage(limit);
    this._sources = new Map(listing.sources.map((s) => [s.source_id, s]));
    return listing;
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
   * If the source has already been fetched (via listSourcesPage), the descriptor
   * is resolved from the local cache.  Otherwise the tensor describes itself
   * on the first .compute() (one tileInfo() call).
   *
   * This method is synchronous-first for the cache-hit path; the returned
   * TensorArray only issues network requests when .compute() is called.
   */
  getTensor(arrayId: string): TensorArray {
    const cached = this._sources.get(sourceOf(arrayId));
    if (cached) {
      const td = descriptorIn(cached, arrayId);
      if (td) return new TensorArray(this._http, td);
    }
    // Return a "pending" proxy — actual descriptor resolved lazily
    return new LazyTensorArray(this._http, arrayId);
  }
}

// ---------------------------------------------------------------------------
// array_id resolution
// ---------------------------------------------------------------------------

/** The routing prefix of an array_id: everything before the first `/`. */
function sourceOf(arrayId: string): string {
  return arrayId.split("/", 1)[0]!;
}

/**
 * The descriptor `arrayId` names within `source`, or undefined.
 *
 * A bare source_id resolves only when the source holds exactly one tensor,
 * which is what the identity policy says its array_id *is*. On a multi-tensor
 * source it stays unresolved rather than guessing tensors[0] (biopb/biopb#75).
 * This is a lookup in a listing already in hand, with no server answer to defer
 * to — unlike the sidecar, which asks and reports what it is told.
 */
function descriptorIn(source: DataSourceDescriptor, arrayId: string) {
  const exact = source.tensors.find((t) => t.array_id === arrayId);
  if (exact) return exact;
  if (arrayId === source.source_id && source.tensors.length === 1) {
    return source.tensors[0];
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// LazyTensorArray: resolves descriptor on first compute()
// ---------------------------------------------------------------------------

/**
 * TensorArray whose descriptor is fetched lazily on the first .compute().
 * Used when getTensor() is called before listSourcesPage().
 */
class LazyTensorArray extends TensorArray {
  /** Single shared resolution promise — prevents concurrent duplicate tileInfo() calls. */
  private _resolvePromise: Promise<void> | null = null;
  private readonly _pendingArrayId: string;

  constructor(client: TensorHttpClient, arrayId: string) {
    // Placeholder descriptor — replaced on first compute() via _doResolve()
    super(client, {
      array_id: arrayId,
      dim_labels: [],
      shape: [],
      dtype: "uint8",
    });
    this._pendingArrayId = arrayId;
  }

  override async compute(options = {}): Promise<import("./types.js").TypedNdArray> {
    this._resolvePromise ??= this._doResolve();
    await this._resolvePromise;
    return super.compute(options);
  }

  private async _doResolve(): Promise<void> {
    // The tensor describes itself: array_id is the whole address, so there is
    // no source to look up and no guessing among its tensors.
    const info = await this._client.tileInfo(this._pendingArrayId);
    const td = {
      array_id: this._pendingArrayId,
      dim_labels: info.dim_labels,
      shape: info.shape,
      dtype: info.dtype,
    };
    this._descriptor = td;
    this._axisMap = buildAxisMap(td.dim_labels);
    this._axisMapAmbiguous = isAxisMapAmbiguous(td.dim_labels);
  }
}
