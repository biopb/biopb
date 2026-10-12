/**
 * High-level facade mirroring the Python TensorFlightClient API.
 *
 * Usage:
 *   const client = new TensorFlightClient("http://localhost:8816", token);
 *   const arr = client.getTensor("my-source/tensor-0");
 *   const data = await arr.compute({ z: 5, c: 0, scaleHint: [1,1,1,8,8] });
 */

import { TensorHttpClient } from "./client.js";
import { TensorArray, buildAxisMap, isAxisMapAmbiguous } from "./tensor-array.js";

export class TensorFlightClient {
  private readonly _http: TensorHttpClient;

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

  /**
   * Get source-level OME/vendor metadata as a plain JS object.
   * Returns {} if no metadata is available.
   */
  async getSourceMetadata(sourceId: string): Promise<Record<string, unknown>> {
    return this._http.getSourceMetadata(sourceId);
  }

  /**
   * Return a lazy TensorArray for the given tensor.
   *
   * The tensor describes itself on the first .compute() (one tileInfo() call);
   * this method issues no request.
   */
  getTensor(arrayId: string): TensorArray {
    return new LazyTensorArray(this._http, arrayId);
  }
}

// ---------------------------------------------------------------------------
// LazyTensorArray: resolves descriptor on first compute()
// ---------------------------------------------------------------------------

/**
 * TensorArray whose descriptor is fetched lazily on the first .compute().
 * Returned by getTensor().
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
