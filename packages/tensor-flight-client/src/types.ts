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

export interface ReadyzSnapshot {
  status: string;
  timestamp: string;
  ready: boolean;
  dev_mode: boolean;
  service: string;
  version: string;
}
