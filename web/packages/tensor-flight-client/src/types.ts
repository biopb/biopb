/** Mirror of biopb.tensor.TensorDescriptor (JSON form from FastAPI). */
export interface TensorDescriptor {
  array_id: string;
  dim_labels: string[];
  /** Full array shape (per dimension). */
  shape: number[];
  /**
   * Transfer grid. EMPTY inside a `DataSourceDescriptor.tensors` entry: a source
   * listing is structural, and the grid is answered per resolved tensor. Use
   * `GET /api/tile_info` (`TileInfo.chunk_shape` / `tile_size`) when you need
   * one -- an empty array is not a usable grid.
   */
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
  /** Structural entry per tensor: array_id, dim_labels, shape, dtype. */
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
  /** "nearest" | "area" | "precompute" (server also accepts "stride", "decimate", "mean"). */
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
  /**
   * Why `backend_health` is null: `connect failed: …` (never reached Flight) or
   * `health check failed: …` (connected, then the health action threw). Null
   * when the backend answered. Absent on servers older than biopb#755, where a
   * null `backend_health` could also mean "no request has connected yet."
   */
  backend_error?: string | null;
}

export interface QuerySourcesResult {
  rows: Record<string, unknown>[];
  totalSources: number;
  returnedSources: number;
  truncated: boolean;
}

// ---------------------------------------------------------------------------
// Admin endpoint (GET/PUT /api/config, /api/admin/status)
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
  /** True when the biopb control owns/supervises this data plane. The admin UI
   * then routes a restart through the control (POST /api/data_plane/restart)
   * instead of the sidecar self-restart, which would conflict with supervision
   * (biopb/biopb#418). Absent on an older sidecar → treated as false. */
  supervised?: boolean;
  config_path: string | null;
  health: string | null;
  source_count: number | null;
  writable: boolean | null;
  uptime_seconds: number | null;
  full_scan_in_progress: boolean | null;
  last_full_scan_finished_at: number | null;
  /** True in local mode (no token enforced, loopback-only). The admin UI shows
   * the server-side file/dir chooser only when this is true — in local mode the
   * server's filesystem is the user's own machine (biopb/biopb#244). Absent on an
   * older sidecar → treated as false. */
  local?: boolean;
}

/** One entry in a `GET /api/admin/browse` directory listing. */
export interface BrowseEntry {
  name: string;
  is_dir: boolean;
}

/** Response of `GET /api/admin/browse`: one directory level of the server's FS.
 * Local-mode only; `parent` is null at the filesystem root (biopb/biopb#244). */
export interface BrowseResponse {
  /** Absolute path of the listed directory. */
  path: string;
  /** Absolute path of the parent directory, or null at the FS root. */
  parent: string | null;
  entries: BrowseEntry[];
  /** True when the listing hit the server's per-directory entry cap. */
  truncated: boolean;
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

// ---------------------------------------------------------------------------
// Tiles
// ---------------------------------------------------------------------------

/** One pyramid level of a tensor's tile grid. */
export interface TileLevel {
  /** 0 is FULL resolution (Viv `PixelSource[]` order, not map-tile z order). */
  level: number;
  /** Downsample factor applied to Y/X at this level, i.e. 2**level. */
  scale: number;
  height: number;
  width: number;
  /** Grid extent at this level; the last row/column may be a short tile. */
  cols: number;
  rows: number;
}

/** Everything needed to address a tensor as a tile grid (`GET /api/tile_info`). */
export interface TileInfo {
  array_id: string;
  dim_labels: string[];
  shape: number[];
  chunk_shape: number[];
  dtype: string;
  /**
   * Square tile edge in pixels. Derived server-side from `chunk_shape` so a tile
   * nests inside a stored chunk -- do NOT assume a constant across tensors.
   */
  tile_size: number;
  /** Wire indices of the display plane; `s` is set only for interleaved RGB(A). */
  plane: { y: number; x: number; s: number | null };
  /** Wire index of each *named* slider axis, or null when nothing names it. */
  selectable: { t: number | null; z: number | null; c: number | null };
  /**
   * Non-plane axes with extent > 1 that `t`/`z`/`c` cannot *name*.
   *
   * An unlabelled axis, a TIFF sequence's opaque file axis (`i`), or the second
   * of two sharing a label. Empty for an ordinary TCZYX tensor.
   *
   * Naming is not addressing: these are selectable, via `TileRequest.sel`. What
   * the list says is that they must be reached positionally, and that there is
   * no semantic title for the slider — so show `label`, the name the source
   * itself gave, rather than deriving `Z` from the axis's position. That
   * derivation is a guess, and the server declines to make it for a reason
   * (`biopb-tensor-server/biopb_tensor_server/core/axes.py`); making it here
   * instead only moves it somewhere less visible.
   */
  sel_axes: TileAxis[];
  levels: TileLevel[];
}

/** A non-plane axis addressed by its wire index, with the source's own name. */
export interface TileAxis {
  /** Wire index, i.e. position in `dim_labels`/`shape`. */
  axis: number;
  /** The source's label for it. May be empty, or shared with another axis. */
  label: string;
  extent: number;
}

/** Address of one tile. Omitted selection axes default to index 0. */
export interface TileRequest {
  /**
   * The tensor's globally-unique id: `source_id` for a single-tensor source,
   * `source_id/field` otherwise. The whole address -- there is no separate
   * tensor_id.
   */
  array_id: string;
  level?: number;
  col?: number;
  row?: number;
  t?: number;
  z?: number;
  c?: number;
  /**
   * Axes selected by wire index: `[[0, 154]]` becomes `?sel=0:154`.
   *
   * For the axes `t`/`z`/`c` cannot name — everything in
   * `TileInfo.sel_axes`. An axis the server *does* name must be sent under that
   * name instead; sending it both ways is refused (422), because one axis with
   * two spellings in one URL is two cache entries for one tile.
   */
  sel?: Array<[number, number]>;
  reduction_method?: string;
}

/** A rendered tile: appearance is baked server-side, so contrast is part of the key. */
export interface TileImageRequest extends TileRequest {
  fmt?: "png" | "jpeg";
  lo?: number;
  hi?: number;
  color?: string;
  use_min_max?: boolean;
}

/** A raw tile plus the grid the server actually served it from. */
export interface TileResult extends TypedNdArray {
  /** Echoed from `X-Tile-*`, so a client can check the grid it assumed. */
  tileSize: number;
  level: number;
  col: number;
  row: number;
}
