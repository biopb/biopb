/**
 * Unit tests for client.ts: TensorHttpClient and TensorApiError.
 *
 * Uses vitest's vi.stubGlobal to replace the global `fetch` so no real
 * HTTP calls are made.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { TensorHttpClient, TensorApiError } from "./client.js";
import type { DataSourceDescriptor, TypedNdArray } from "./types.js";

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

const BASE = "http://localhost:8816";
const TOKEN = "test-token-abcdef1234";

const SOURCE: DataSourceDescriptor = {
  source_id: "src0",
  source_url: "/data/src0",
  source_type: "zarr",
  metadata_json: null,
  tensors: [
    {
      array_id: "t0",
      dim_labels: ["z", "y", "x"],
      shape: [10, 128, 256],
      chunk_shape: [1, 64, 64],
      dtype: "uint16",
    },
  ],
};

// ---------------------------------------------------------------------------
// fetch mock helpers
// ---------------------------------------------------------------------------

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function binaryResponse(
  data: ArrayBuffer,
  shape: number[],
  dtype: string,
  dimLabels: string[],
  status = 200,
): Response {
  return new Response(data, {
    status,
    headers: {
      "Content-Type": "application/octet-stream",
      "X-Shape": shape.join(","),
      "X-Dtype": dtype,
      "X-Dim-Labels": dimLabels.join(","),
    },
  });
}

function errorResponse(status: number, detail: string): Response {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

// ---------------------------------------------------------------------------
// Setup / teardown
// ---------------------------------------------------------------------------

let mockFetch: ReturnType<typeof vi.fn>;

beforeEach(() => {
  mockFetch = vi.fn();
  vi.stubGlobal("fetch", mockFetch);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// TensorApiError
// ---------------------------------------------------------------------------

describe("TensorApiError", () => {
  it("sets status, message, and name", () => {
    const err = new TensorApiError(404, "Not found", { detail: "missing" });
    expect(err.status).toBe(404);
    expect(err.name).toBe("TensorApiError");
    expect(err.message).toContain("404");
    expect(err.detail).toEqual({ detail: "missing" });
  });

  it("is instanceof Error", () => {
    const err = new TensorApiError(500, "oops");
    expect(err).toBeInstanceOf(Error);
  });
});

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

describe("TensorHttpClient construction", () => {
  it("strips trailing slash from base", () => {
    const c = new TensorHttpClient("http://localhost:8816/", TOKEN);
    // Internal base is private but we can verify requests go to correct URL
    mockFetch.mockResolvedValueOnce(jsonResponse({ status: "ok", timestamp: "2024-01-01T00:00:00Z" }));
    c.livez();
    const calledUrl: string = mockFetch.mock.calls[0]![0] as string;
    expect(calledUrl).not.toContain("//livez");
    expect(calledUrl).toContain("/livez");
  });

  it("accepts null token", () => {
    expect(() => new TensorHttpClient(BASE, null)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Health endpoints
// ---------------------------------------------------------------------------

describe("TensorHttpClient.livez", () => {
  it("GETs /livez and returns parsed JSON", async () => {
    const body = { status: "ok", timestamp: "2025-05-01T00:00:00Z" };
    mockFetch.mockResolvedValueOnce(jsonResponse(body));
    const c = new TensorHttpClient(BASE, TOKEN);
    const result = await c.livez();
    expect(result.status).toBe("ok");
    const [url, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${BASE}/livez`);
    expect((opts.headers as Record<string, string>)["Authorization"]).toBe(`Bearer ${TOKEN}`);
  });

  it("throws TensorApiError on non-OK response", async () => {
    mockFetch.mockResolvedValueOnce(errorResponse(503, "server down"));
    const c = new TensorHttpClient(BASE, TOKEN);
    await expect(c.livez()).rejects.toBeInstanceOf(TensorApiError);
  });
});

describe("TensorHttpClient.readyz", () => {
  it("GETs /readyz", async () => {
    const body = {
      status: "ok", timestamp: "", ready: true,
      dev_mode: false, service: "biopb-tensor-web", version: "0.1.0",
    };
    mockFetch.mockResolvedValueOnce(jsonResponse(body));
    const c = new TensorHttpClient(BASE, TOKEN);
    const r = await c.readyz();
    expect(r.ready).toBe(true);
  });

  it("surfaces the backend freshness fields (progressive discovery)", async () => {
    const body = {
      status: "ok", timestamp: "", ready: true,
      dev_mode: false, service: "biopb-tensor-web", version: "0.1.0",
      source_count: 3,
      backend_health: {
        status: "SERVING",
        source_count: 3,
        full_scan_in_progress: true,
        last_full_scan_finished_at: null,
      },
    };
    mockFetch.mockResolvedValueOnce(jsonResponse(body));
    const c = new TensorHttpClient(BASE, TOKEN);
    const r = await c.readyz();
    expect(r.backend_health?.full_scan_in_progress).toBe(true);
    expect(r.backend_health?.last_full_scan_finished_at).toBeNull();
    expect(r.source_count).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

describe("TensorHttpClient.listSources", () => {
  it("GETs /api/sources and returns array", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse([SOURCE]));
    const c = new TensorHttpClient(BASE, TOKEN);
    const sources = await c.listSources();
    expect(sources).toHaveLength(1);
    expect(sources[0]!.source_id).toBe("src0");
  });

  it("sends Authorization header", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse([]));
    const c = new TensorHttpClient(BASE, TOKEN);
    await c.listSources();
    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect((opts.headers as Record<string, string>)["Authorization"]).toBe(`Bearer ${TOKEN}`);
  });

  it("does NOT send Authorization when token is null", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse([]));
    const c = new TensorHttpClient(BASE, null);
    await c.listSources();
    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect((opts.headers as Record<string, string>)["Authorization"]).toBeUndefined();
  });

  it("throws TensorApiError on 401", async () => {
    mockFetch.mockResolvedValueOnce(errorResponse(401, "Invalid or missing token"));
    const c = new TensorHttpClient(BASE, "wrong");
    const err = await c.listSources().catch((e) => e);
    expect(err).toBeInstanceOf(TensorApiError);
    expect((err as TensorApiError).status).toBe(401);
  });
});

describe("TensorHttpClient.getSource", () => {
  it("encodes source_id in path", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(SOURCE));
    const c = new TensorHttpClient(BASE, TOKEN);
    await c.getSource("path/with spaces");
    const [url] = mockFetch.mock.calls[0] as [string];
    expect(url).toContain(encodeURIComponent("path/with spaces"));
  });

  it("throws 404 TensorApiError for missing source", async () => {
    mockFetch.mockResolvedValueOnce(errorResponse(404, "Source not found"));
    const c = new TensorHttpClient(BASE, TOKEN);
    const err = await c.getSource("nope").catch((e) => e);
    expect(err).toBeInstanceOf(TensorApiError);
    expect((err as TensorApiError).status).toBe(404);
  });
});

describe("TensorHttpClient.getSourceMetadata", () => {
  it("GETs /api/sources/{id}/metadata", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ ome_ngff: { version: "0.4" } }));
    const c = new TensorHttpClient(BASE, TOKEN);
    const meta = await c.getSourceMetadata("src0");
    expect(meta["ome_ngff"]).toBeDefined();
    const [url] = mockFetch.mock.calls[0] as [string];
    expect(url).toContain("/metadata");
  });
});

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

describe("TensorHttpClient.slice", () => {
  function makeBuffer(bytes: number): ArrayBuffer {
    return new ArrayBuffer(bytes);
  }

  it("POSTs /api/slice and returns TypedNdArray", async () => {
    const buf = makeBuffer(10 * 128 * 256 * 2); // uint16
    mockFetch.mockResolvedValueOnce(
      binaryResponse(buf, [10, 128, 256], "uint16", ["z", "y", "x"]),
    );
    const c = new TensorHttpClient(BASE, TOKEN);
    const result: TypedNdArray = await c.slice({
      source_id: "src0",
      tensor_id: "t0",
    });
    expect(result.shape).toEqual([10, 128, 256]);
    expect(result.dtype).toBe("uint16");
    expect(result.dimLabels).toEqual(["z", "y", "x"]);
    expect(result.buffer.byteLength).toBe(buf.byteLength);
  });

  it("uses POST method", async () => {
    const buf = makeBuffer(4);
    mockFetch.mockResolvedValueOnce(binaryResponse(buf, [1, 1, 1], "uint8", []));
    const c = new TensorHttpClient(BASE, TOKEN);
    await c.slice({ source_id: "s", tensor_id: "t" });
    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(opts.method).toBe("POST");
  });

  it("serialises request body as JSON", async () => {
    const buf = makeBuffer(4);
    mockFetch.mockResolvedValueOnce(binaryResponse(buf, [1, 1, 1], "uint8", []));
    const c = new TensorHttpClient(BASE, TOKEN);
    const req = {
      source_id: "src0",
      tensor_id: "t0",
      slice_start: [0, 0, 0],
      slice_stop: [1, 10, 10],
    };
    await c.slice(req);
    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(opts.body as string);
    expect(body.source_id).toBe("src0");
    expect(body.slice_start).toEqual([0, 0, 0]);
  });

  it("throws TensorApiError on 502", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Flight error: RuntimeError" }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const c = new TensorHttpClient(BASE, TOKEN);
    const err = await c.slice({ source_id: "s", tensor_id: "t" }).catch((e) => e);
    expect(err).toBeInstanceOf(TensorApiError);
    expect((err as TensorApiError).status).toBe(502);
  });

  it("handles empty X-Dim-Labels gracefully", async () => {
    const buf = makeBuffer(4);
    mockFetch.mockResolvedValueOnce(binaryResponse(buf, [1, 2], "float32", []));
    const c = new TensorHttpClient(BASE, TOKEN);
    const result = await c.slice({ source_id: "s", tensor_id: "t" });
    expect(result.dimLabels).toEqual([]);
  });

  it("does not force uint8 when custom slice headers are unavailable", async () => {
    const buf = makeBuffer(8);
    mockFetch.mockResolvedValueOnce(
      new Response(buf, {
        status: 200,
        headers: { "Content-Type": "application/octet-stream" },
      }),
    );

    const c = new TensorHttpClient(BASE, TOKEN);
    const result = await c.slice({ source_id: "s", tensor_id: "t" });

    expect(result.shape).toEqual([]);
    expect(result.dtype).toBe("");
    expect(result.dimLabels).toEqual([]);
    expect(result.buffer.byteLength).toBe(8);
  });
});

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

describe("TensorHttpClient.diagnostics", () => {
  it("GETs /api/diagnostics and returns snapshot", async () => {
    const snap = {
      status: "ok", timestamp: "", dev_mode: false,
      connection_state: "connected", degraded_mode: false,
      pixel_budget: null, cache_hit_rate: 0.8,
      latency_p50_ms: 12, latency_p95_ms: 45,
      last_error_code: null, last_error_message: null,
      metrics_ready: true,
    };
    mockFetch.mockResolvedValueOnce(jsonResponse(snap));
    const c = new TensorHttpClient(BASE, TOKEN);
    const d = await c.diagnostics();
    expect(d.connection_state).toBe("connected");
    expect(d.metrics_ready).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Admin (config read/write, status, restart)
// ---------------------------------------------------------------------------

describe("TensorHttpClient.getAdminConfig", () => {
  it("GETs /api/config and returns path, config, schema", async () => {
    const body = {
      path: "/home/u/.config/biopb/biopb.json",
      config: { server: { port: 8815 }, keep_me: { x: 1 } },
      schema: { properties: {} },
    };
    mockFetch.mockResolvedValueOnce(jsonResponse(body));
    const c = new TensorHttpClient(BASE, TOKEN);
    const r = await c.getAdminConfig();
    expect(mockFetch.mock.calls[0]![0]).toBe(`${BASE}/api/config`);
    expect(r.path).toBe(body.path);
    expect(r.config.keep_me).toEqual({ x: 1 });
    expect(r.schema).toHaveProperty("properties");
  });
});

describe("TensorHttpClient.putAdminConfig", () => {
  it("PUTs /api/config with the config body and returns the save result", async () => {
    const result = { saved: true, restart_required: true, path: "/cfg/biopb.json" };
    mockFetch.mockResolvedValueOnce(jsonResponse(result));
    const c = new TensorHttpClient(BASE, TOKEN);
    const r = await c.putAdminConfig({ server: { port: 9000 } });
    const [url, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${BASE}/api/config`);
    expect(opts.method).toBe("PUT");
    expect(JSON.parse(opts.body as string)).toEqual({ server: { port: 9000 } });
    expect(r.restart_required).toBe(true);
  });

  it("throws TensorApiError carrying the 422 validation errors on detail", async () => {
    const body = {
      detail: "Config failed schema validation",
      errors: [{ path: ["pyramid", "downscale_factor"], message: "1 is less than the minimum of 2" }],
    };
    mockFetch.mockResolvedValueOnce(jsonResponse(body, 422));
    const c = new TensorHttpClient(BASE, TOKEN);
    await c.putAdminConfig({ pyramid: { downscale_factor: 1 } }).then(
      () => { throw new Error("should have thrown"); },
      (e: unknown) => {
        expect(e).toBeInstanceOf(TensorApiError);
        const err = e as TensorApiError;
        expect(err.status).toBe(422);
        const detail = err.detail as typeof body;
        expect(detail.errors[0]!.path).toEqual(["pyramid", "downscale_factor"]);
      },
    );
  });
});

describe("TensorHttpClient.getAdminStatus", () => {
  it("GETs /api/admin/status and returns merged health + process facts", async () => {
    const body = {
      running: true, pid: 123, version: "0.4.7", config_path: "/cfg/biopb.json",
      health: "SERVING", source_count: 7, writable: false, uptime_seconds: 42,
      full_scan_in_progress: true, last_full_scan_finished_at: null,
    };
    mockFetch.mockResolvedValueOnce(jsonResponse(body));
    const c = new TensorHttpClient(BASE, TOKEN);
    const r = await c.getAdminStatus();
    expect(mockFetch.mock.calls[0]![0]).toBe(`${BASE}/api/admin/status`);
    expect(r.running).toBe(true);
    expect(r.full_scan_in_progress).toBe(true);
    expect(r.source_count).toBe(7);
  });
});

describe("TensorHttpClient.restartServer", () => {
  it("POSTs /api/admin/restart and returns the 202 body", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ restarting: true }, 202));
    const c = new TensorHttpClient(BASE, TOKEN);
    const r = await c.restartServer();
    const [url, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${BASE}/api/admin/restart`);
    expect(opts.method).toBe("POST");
    expect(r.restarting).toBe(true);
  });
});

describe("TensorHttpClient.browse", () => {
  it("GETs /api/admin/browse with no query when path omitted", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ path: "/home/u", parent: "/home", entries: [], truncated: false }),
    );
    const c = new TensorHttpClient(BASE, null);
    const r = await c.browse();
    expect(mockFetch.mock.calls[0]![0]).toBe(`${BASE}/api/admin/browse`);
    expect(r.path).toBe("/home/u");
    expect(r.parent).toBe("/home");
  });

  it("URL-encodes the path query parameter", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        path: "/data/my images",
        parent: "/data",
        entries: [{ name: "a.zarr", is_dir: true }],
        truncated: false,
      }),
    );
    const c = new TensorHttpClient(BASE, null);
    const r = await c.browse("/data/my images");
    expect(mockFetch.mock.calls[0]![0]).toBe(
      `${BASE}/api/admin/browse?path=%2Fdata%2Fmy%20images`,
    );
    expect(r.entries[0]!.is_dir).toBe(true);
  });

  it("throws TensorApiError when browsing is unavailable (remote mode 404)", async () => {
    mockFetch.mockResolvedValueOnce(errorResponse(404, "File browsing is available only in local mode"));
    const c = new TensorHttpClient(BASE, TOKEN);
    await expect(c.browse("/data")).rejects.toThrow(TensorApiError);
  });
});
