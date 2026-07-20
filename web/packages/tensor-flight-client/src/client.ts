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
    const controller = new AbortController();
    const timeoutId = timeoutMs != null
      ? setTimeout(() => controller.abort(), timeoutMs)
      : null;
    try {
      const res = await fetch(url, {
        ...options,
        headers: { ...this.headers(), ...(options?.headers as Record<string, string> ?? {}) },
        signal: controller.signal,
      });
      if (!res.ok) {
        // Check if response is HTML (nginx error page during startup)
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
      return res.json() as Promise<T>;
    } catch (e) {
      if (e instanceof Error && e.name === "AbortError") {
        throw new TensorApiError(408, `Timeout after ${timeoutMs}ms (${path})`);
      }
      throw e;
    } finally {
      if (timeoutId !== null) clearTimeout(timeoutId);
    }
  }

  private async fetchBinary(
    path: string,
    body: unknown,
    timeoutMs?: number,
  ): Promise<Response> {
    const url = `${this.base}${path}`;
    const controller = new AbortController();
    const timeoutId = timeoutMs != null
      ? setTimeout(() => controller.abort(), timeoutMs)
      : null;
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok) {
        // Check if response is HTML (nginx error page during startup)
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
      return res;
    } catch (e) {
      if (e instanceof Error && e.name === "AbortError") {
        throw new TensorApiError(408, `Timeout after ${timeoutMs}ms (${path})`);
      }
      throw e;
    } finally {
      if (timeoutId !== null) clearTimeout(timeoutId);
    }
  }

  private async fetchJsonWithHeaders<T>(
    path: string,
    body: unknown,
    timeoutMs?: number,
  ): Promise<{ data: T; headers: Headers }> {
    const url = `${this.base}${path}`;
    const controller = new AbortController();
    const timeoutId = timeoutMs != null
      ? setTimeout(() => controller.abort(), timeoutMs)
      : null;
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok) {
        // Check if response is HTML (nginx error page during startup)
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
      const data = await res.json() as T;
      return { data, headers: res.headers };
    } catch (e) {
      if (e instanceof Error && e.name === "AbortError") {
        throw new TensorApiError(408, `Timeout after ${timeoutMs}ms (${path})`);
      }
      throw e;
    } finally {
      if (timeoutId != null) clearTimeout(timeoutId);
    }
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

  /**
   * Execute SQL query against server's source metadata database.
   *
   * @param sql SQL query (e.g., "SELECT source_id FROM sources WHERE source_type='ome-zarr'")
   * @returns Query result with rows and truncation metadata
   * @throws {TensorApiError} on validation error or timeout
   */
  async querySources(sql: string): Promise<QuerySourcesResult> {
    const { data, headers } = await this.fetchJsonWithHeaders<Record<string, unknown>[]>(
      "/api/sources/query",
      { sql },
      this.metadataTimeoutMs,
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
  async getAdminConfig(): Promise<AdminConfigResponse> {
    return this.fetchJson<AdminConfigResponse>(
      "/api/config",
      undefined,
      this.metadataTimeoutMs,
    );
  }

  /**
   * Validate and write the config (`PUT /api/config`). Does NOT restart.
   *
   * On a schema-validation failure the server returns `422` and `fetchJson`
   * throws a {@link TensorApiError} whose `.detail` is the
   * {@link AdminConfigValidationBody} (`{detail, errors}`); callers render
   * `error.detail.errors` inline.
   */
  async putAdminConfig(config: Record<string, unknown>): Promise<AdminConfigSaveResult> {
    return this.fetchJson<AdminConfigSaveResult>(
      "/api/config",
      { method: "PUT", body: JSON.stringify(config) },
      this.metadataTimeoutMs,
    );
  }

  /** Backend health merged with process facts (`GET /api/admin/status`). */
  async getAdminStatus(): Promise<AdminStatus> {
    return this.fetchJson<AdminStatus>(
      "/api/admin/status",
      undefined,
      this.metadataTimeoutMs,
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
  async browse(path?: string): Promise<BrowseResponse> {
    const qs = path ? `?path=${encodeURIComponent(path)}` : "";
    return this.fetchJson<BrowseResponse>(
      `/api/admin/browse${qs}`,
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
    const dtype = res.headers.get("X-Dtype") ?? "";
    const dimLabels = (res.headers.get("X-Dim-Labels") ?? "")
      .split(",")
      .filter(Boolean);

    const shape = shapeHeader.split(",").filter(Boolean).map(Number);
    const buffer = await res.arrayBuffer();

    return { buffer, shape, dtype, dimLabels };
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
   */
  async render(req: RenderRequest): Promise<RenderResult> {
    // Use longer timeout for rendering (may be slower than raw slice)
    const res = await this.fetchBinary("/api/render", req, this.chunkTimeoutMs * 2);

    const width = parseInt(res.headers.get("X-Image-Width") ?? "0", 10);
    const height = parseInt(res.headers.get("X-Image-Height") ?? "0", 10);
    const percentileLoValue = parseFloat(res.headers.get("X-Percentile-Lo-Value") ?? "0");
    const percentileHiValue = parseFloat(res.headers.get("X-Percentile-Hi-Value") ?? "1");
    const format = res.headers.get("X-Image-Format") ?? req.output_format ?? "jpeg";

    // For raw format, use arrayBuffer; for png/jpeg, use blob
    const blob = format === "raw" ? await res.arrayBuffer() : await res.blob();

    return { blob, width, height, percentileLoValue, percentileHiValue, format };
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
