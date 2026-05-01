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
  DataSourceDescriptor,
  DiagnosticsSnapshot,
  ReadyzSnapshot,
  SliceRequest,
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
    super(`TensorApi ${status}: ${message}`);
    this.name = "TensorApiError";
  }
}

// ---------------------------------------------------------------------------
// Timeout helpers
// ---------------------------------------------------------------------------

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const id = setTimeout(
      () => reject(new TensorApiError(408, `Timeout after ${ms}ms (${label})`)),
      ms,
    );
    promise.then(
      (v) => { clearTimeout(id); resolve(v); },
      (e) => { clearTimeout(id); reject(e); },
    );
  });
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export class TensorHttpClient {
  private readonly base: string;
  private readonly token: string | null;

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
  ): Promise<T> {
    const url = `${this.base}${path}`;
    const promise = fetch(url, {
      ...options,
      headers: { ...this.headers(), ...(options?.headers as Record<string, string> ?? {}) },
    }).then(async (res) => {
      if (!res.ok) {
        let detail: unknown;
        try { detail = await res.json(); } catch { /* ignore */ }
        throw new TensorApiError(res.status, res.statusText, detail);
      }
      return res.json() as Promise<T>;
    });
    return timeoutMs != null
      ? withTimeout(promise, timeoutMs, path)
      : promise;
  }

  private async fetchBinary(
    path: string,
    body: unknown,
    timeoutMs?: number,
  ): Promise<Response> {
    const url = `${this.base}${path}`;
    const promise = fetch(url, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    }).then((res) => {
      if (!res.ok) {
        return res.json().then(
          (detail) => { throw new TensorApiError(res.status, res.statusText, detail); },
          () => { throw new TensorApiError(res.status, res.statusText); },
        );
      }
      return res;
    });
    return timeoutMs != null
      ? withTimeout(promise, timeoutMs, path)
      : promise;
  }

  // -------------------------------------------------------------------------
  // Health (no auth required)
  // -------------------------------------------------------------------------

  async livez(): Promise<{ status: string; timestamp: string }> {
    return this.fetchJson("/livez", undefined, this.metadataTimeoutMs);
  }

  async readyz(): Promise<ReadyzSnapshot> {
    return this.fetchJson("/readyz", undefined, this.metadataTimeoutMs);
  }

  // -------------------------------------------------------------------------
  // Sources
  // -------------------------------------------------------------------------

  /** List all data sources registered with the server. */
  async listSources(): Promise<DataSourceDescriptor[]> {
    return this.fetchJson<DataSourceDescriptor[]>(
      "/api/sources",
      undefined,
      this.metadataTimeoutMs,
    );
  }

  /** Get a single DataSourceDescriptor by source_id. */
  async getSource(sourceId: string): Promise<DataSourceDescriptor> {
    return this.fetchJson<DataSourceDescriptor>(
      `/api/sources/${encodeURIComponent(sourceId)}`,
      undefined,
      this.metadataTimeoutMs,
    );
  }

  /**
   * Get the parsed OME-NGFF metadata for a source.
   * Returns an empty object if the source has no metadata.
   */
  async getSourceMetadata(sourceId: string): Promise<Record<string, unknown>> {
    return this.fetchJson<Record<string, unknown>>(
      `/api/sources/${encodeURIComponent(sourceId)}/metadata`,
      undefined,
      this.metadataTimeoutMs,
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
   */
  async slice(req: SliceRequest): Promise<TypedNdArray> {
    const res = await this.fetchBinary("/api/slice", req, this.chunkTimeoutMs);

    const shapeHeader = res.headers.get("X-Shape") ?? "";
    const dtype = res.headers.get("X-Dtype") ?? "uint8";
    const dimLabels = (res.headers.get("X-Dim-Labels") ?? "")
      .split(",")
      .filter(Boolean);

    const shape = shapeHeader.split(",").filter(Boolean).map(Number);
    const buffer = await res.arrayBuffer();

    return { buffer, shape, dtype, dimLabels };
  }

  // -------------------------------------------------------------------------
  // Diagnostics
  // -------------------------------------------------------------------------

  async diagnostics(): Promise<DiagnosticsSnapshot> {
    return this.fetchJson<DiagnosticsSnapshot>(
      "/api/diagnostics",
      undefined,
      this.metadataTimeoutMs,
    );
  }
}
