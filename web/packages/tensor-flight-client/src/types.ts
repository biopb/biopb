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
  /**
   * The tensor's whole address: `source_id` for a single-tensor source,
   * `source_id/field` otherwise. Not a `(source_id, tensor_id)` pair — that
   * split had to be rejoined server-side before every read.
   */
  array_id: string;
  /** Per-dimension start indices (inclusive). */
  slice_start?: number[];
  /** Per-dimension stop indices (exclusive). */
  slice_stop?: number[];
  /** Per-dimension integer downsampling factors, e.g. [1, 8, 8]. */
  scale_hint?: number[];
  /**
   * Hand the scale decision to the server instead of naming one.
   *
   * `"volume"` reads at the single scale the server keeps a whole 3-D volume
   * warm at — the level napari 3-D and `XR3DLayer` upload as one texture.
   * There is no way to compute it client-side short of reimplementing the
   * server's pyramid planner, and a guess one rung off misses every warmed
   * chunk. `TileInfo.volume` says what it will resolve to for a given tensor.
   *
   * Mutually exclusive with `scale_hint`; sending both is a 422.
   */
  scale_policy?: "volume";
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
  /**
   * Per-axis scale the server actually read at (`X-Scale-Hint`).
   *
   * Echoed on every slice, and load-bearing under `scale_policy`: there the
   * caller did not choose, so this is the only statement of what it got. Empty
   * against a server predating the header.
   */
  scaleHint: number[];
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
  /**
   * What a 3-D read of this tensor gets, or why there isn't one.
   *
   * Not a rung of `levels`: a 3-D renderer takes one whole volume rather than
   * tiles, so it leaves the tile ladder entirely (`XR3DLayer` has no `loader`
   * prop). It rides this response because this is the one call a viewer
   * already makes before it can address the tensor at all.
   *
   * Absent against a server predating it — treat that as unavailable.
   */
  volume?: VolumeInfo;
}

/** `TileInfo.volume`: the plan for a `scale_policy: "volume"` read. */
export type VolumeInfo = VolumeAvailable | VolumeUnavailable;

export interface VolumeUnavailable {
  available: false;
  /**
   * Why, in a sentence meant to be shown: no z axis, a z extent of 1, an
   * interleaved samples axis. A fact about the tensor, so it will not change
   * on a retry.
   */
  reason: string;
}

export interface VolumeAvailable {
  available: true;
  reason: null;
  /** Wire indices of the three volume axes. */
  axes: { z: number; y: number; x: number };
  /** Per-axis scale the server will read at. Full length, wire order. */
  scale_hint: number[];
  /** Extent of the returned volume along z / y / x, after `scale_hint`. */
  depth: number;
  height: number;
  width: number;
  /**
   * Wire size of the volume at its own dtype. NOT the VRAM it costs: Viv casts
   * every volume to Float32 on upload, so that is 4 bytes per voxel regardless.
   */
  bytes: number;
  /**
   * Physical extent of one voxel **of the returned volume** (source physical
   * size already multiplied by `scale_hint`), or null when the source declares
   * none — render isotropic then.
   *
   * All three are in the *same* unit, reconciled server-side: `physical_unit`
   * is per-axis and adapters do not all normalise (NIfTI reports mm, the EM
   * readers nm), so comparing them raw would stretch the volume by whatever the
   * conversion factor was. Axes whose units differ and cannot be placed on a
   * common scale come back null rather than as a plausible wrong ratio.
   */
  spacing: { z: number; y: number; x: number } | null;
  /**
   * Unit `spacing` is expressed in — "µm" whenever the server could convert.
   * Null when the axes agree on a unit it cannot name, which is still a valid
   * ratio. Not needed to render: `spacing` is self-consistent either way.
   */
  unit: string | null;
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

/** A raw tile plus the grid the server actually served it from. */
export interface TileResult extends TypedNdArray {
  /** Echoed from `X-Tile-*`, so a client can check the grid it assumed. */
  tileSize: number;
  level: number;
  col: number;
  row: number;
}
