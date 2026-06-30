/** Mirror of biopb.tensor.TensorDescriptor (JSON form from FastAPI). */
export interface TensorDescriptor {
  array_id: string;
  dim_labels: string[];
  /** Full array shape (per dimension). */
  shape: number[];
  chunk_shape: number[];
  /** NumPy-style dtype string, e.g. "uint8", "float32". */
  dtype: string;
}

/** Mirror of biopb.tensor.DataSourceDescriptor (JSON form from FastAPI). */
export interface DataSourceDescriptor {
  source_id: string;
  source_url: string;
  source_type: string;
  /** Raw OME-NGFF JSON string, or null. */
  metadata_json: string | null;
  tensors: TensorDescriptor[];
}

/** Parameters for a single array-slice request. */
export interface SliceRequest {
  source_id: string;
  tensor_id: string;
  /** Per-dimension start indices (inclusive). */
  slice_start?: number[];
  /** Per-dimension stop indices (exclusive). */
  slice_stop?: number[];
  /** Per-dimension integer downsampling factors, e.g. [1, 8, 8]. */
  scale_hint?: number[];
  /** "nearest" | "area" | "linear" (server also accepts "stride", "decimate", "mean"). */
  reduction_method?: string;
  /** Informational: current viewport pixel budget (stored in diagnostics). */
  pixel_budget?: number;
}

/** A typed multi-dimensional array returned by a slice request. */
export interface TypedNdArray {
  /** Raw C-contiguous bytes exactly as returned by numpy.tobytes(). */
  buffer: ArrayBuffer;
  /** Actual shape of the returned slice (may differ from request if edge chunk). */
  shape: number[];
  /** NumPy dtype string, e.g. "uint8", "float32". */
  dtype: string;
  /** Semantic axis labels, e.g. ["t","z","y","x"]. Empty if not available. */
  dimLabels: string[];
}

/** Parsed OME-NGFF multiscales metadata (minimal subset). */
export interface OmeNgffMultiscales {
  axes?: Array<{ name: string; type?: string; unit?: string }>;
  datasets?: Array<{ path: string; coordinateTransformations?: unknown[] }>;
  [key: string]: unknown;
}

export interface DiagnosticsSnapshot {
  status: string;
  timestamp: string;
  dev_mode: boolean;
  connection_state: string;
  degraded_mode: boolean;
  pixel_budget: number | null;
  cache_hit_rate: number | null;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  last_error_code: string | null;
  last_error_message: string | null;
  metrics_ready: boolean;
}

/** The tensor server's `health` action payload, forwarded verbatim by /readyz. */
export interface BackendHealth {
  status?: string;
  source_count?: number;
  metadata_db_enabled?: boolean;
  writable?: boolean;
  uptime_seconds?: number;
  /** Progressive discovery: whether a full catalog scan is running right now. */
  full_scan_in_progress?: boolean;
  /** Epoch seconds of the last successful full scan, or null until the first. */
  last_full_scan_finished_at?: number | null;
}

export interface ReadyzSnapshot {
  status: string;
  timestamp: string;
  ready: boolean;
  dev_mode: boolean;
  service: string;
  version: string;
  /** source_count from the backend health (0/absent on older servers). */
  source_count?: number;
  /** Full backend health dict, including the freshness fields above. */
  backend_health?: BackendHealth | null;
}

export interface QuerySourcesResult {
  rows: Record<string, unknown>[];
  totalSources: number;
  returnedSources: number;
  truncated: boolean;
}

// ---------------------------------------------------------------------------
// Admin endpoint (GET/PUT /api/config, /api/admin/status, /api/admin/restart)
// ---------------------------------------------------------------------------

/** Response of `GET /api/config`: the on-disk config plus its path and schema. */
export interface AdminConfigResponse {
  /** Absolute path of the config file on the server. */
  path: string;
  /** The raw config dict, exactly as it sits on disk (round-trippable). */
  config: Record<string, unknown>;
  /** The JSON Schema (build_config_schema output) describing the config. */
  schema: Record<string, unknown>;
}

/** One schema-validation failure from a rejected `PUT /api/config` (422 body). */
export interface AdminConfigError {
  /** JSON path to the offending field, e.g. ["sources", 0, "url"]. */
  path: (string | number)[];
  message: string;
}

/** Body of a `422` from `PUT /api/config`. Carried on `TensorApiError.detail`. */
export interface AdminConfigValidationBody {
  detail: string;
  errors: AdminConfigError[];
}

/** Response of `PUT /api/config` on success (200). */
export interface AdminConfigSaveResult {
  saved: boolean;
  restart_required: boolean;
  path: string;
}

/** Response of `GET /api/admin/status`: backend health merged with process facts. */
export interface AdminStatus {
  running: boolean;
  pid: number;
  version: string;
  config_path: string | null;
  health: string | null;
  source_count: number | null;
  writable: boolean | null;
  uptime_seconds: number | null;
  full_scan_in_progress: boolean | null;
  last_full_scan_finished_at: number | null;
}

/** Parameters for backend rendering request. */
export interface RenderRequest {
  source_id: string;
  tensor_id: string;
  slice_start?: number[];
  slice_stop?: number[];
  scale_hint?: number[];
  reduction_method?: string;
  percentile_lo?: number;
  percentile_hi?: number;
  color?: string;  // preset name or hex (#rrggbb)
  channel_name?: string;  // for auto color resolution
  use_min_max?: boolean;
  output_format?: "png" | "jpeg" | "raw";  // raw = uncompressed RGBA bytes
  pixel_budget?: number;
}

/** Result of backend rendering request. */
export interface RenderResult {
  /** Image blob (PNG/JPEG) or ArrayBuffer (raw). */
  blob: Blob | ArrayBuffer;
  /** Width of rendered image. */
  width: number;
  /** Height of rendered image. */
  height: number;
  /** Actual computed lo percentile value. */
  percentileLoValue: number;
  /** Actual computed hi percentile value. */
  percentileHiValue: number;
  /** Output format used (from X-Image-Format header). */
  format?: string;
}
