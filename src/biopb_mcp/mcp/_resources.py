"""MCP resource content for the developer guide system.

Each constant is served as an MCP resource that the agent reads on demand.
"""

GUIDE = """\
# biopb-mcp Automation Guide

## Tools
- `take_screenshot(canvas_only=True)` — capture viewer as PNG image
- `execute_code(python_code)` — run Python with the namespace below
- `inspect_object(object_path)` — reflect on any object (e.g. "viewer.layers")
- `server_status()` — system load, memory, dask, tensor server, kernel state
- `interrupt_kernel()` — SIGINT the running execution (best-effort stop)
- `restart_kernel()` — kill + respawn the kernel (guaranteed stop; rebuilds viewer)

## Namespace (available in execute_code)
| Name | Type | Description |
|------|------|-------------|
| `client` | TensorFlightClient or None | Client instance (None if not connected to data server) for browsing/retrieving catalog data |
| `viewer` | napari.Viewer | The active viewer instance that user sees |
| `np` | module | numpy |
| `da` | module | dask.array |
| `ops` | dict[str, callable] | biopb.image ProcessImage operations from configured servers (may be empty) |

Browse the catalog through `client` — `client.query_sources(sql)` (server-side
DuckDB, complete) or `client.list_sources()` (capped by the server for large
catalogs).

## Image Processing Ops (`ops`)
`ops` maps op name -> a thin callable that runs one `biopb.image.ProcessImage`
op (segmentation, denoising, etc.) on a configured server. Discover and
inspect them before use — each carries a docstring with its server, labels,
input-shape hints, and default kwargs:
```python
list(ops)                          # available op names
inspect_object("ops['op_name']")   # docstring, default kwargs, server
```
Call signature: `ops["name"](image, dim_labels=None, **kwargs)`
* `image` as an `np.ndarray` -> sent inline (eager) -> returns an `np.ndarray`.
* `image` as a tensor-server **source_id str** -> sent as a lazy reference (the
  op server pulls pixels straight from the tensor server, no kernel
  round-trip) -> the result is uploaded back to the tensor server and a new
  **source_id str** is returned, so ops chain lazily on large data:
```python
labels = ops["cellpose_cyto2"](arr)          # ndarray -> ndarray
seg_id = ops["cellpose_cyto2"]("raw_data_id") # id -> id (lazy, large data)
viewer.load_tensor(seg_id)                    # view the result
```

## Execution environment
Code runs in a child Jupyter kernel (a real IPython kernel) with napari
integrated via `%gui qt`. Imports are allowed and variables persist across
`execute_code` calls until the kernel is restarted.

* The viewer is a live desktop window; mutations show up immediately.
* Data from `TensorFlightClient` are lazy, thread-safe, picklable dask arrays.
* Prefer lazy dask operations and only `.compute()` the final result.
* If computing the final result would OOM, upload directly to the server with
  `client.upload_array()` instead of `arr.compute()`.

## Stopping runaway code
* `interrupt_kernel()` sends SIGINT — it frees pure-Python loops, but blocking
  C calls (gRPC tensor fetches, native dask compute) ignore it.
* `restart_kernel()` is the guaranteed stop: it kills the kernel process group
  and respawns a fresh kernel + viewer. All previously defined variables are
  lost. A long execution returns a `timeout` message after `execute_timeout`
  seconds and is auto-interrupted.

## Domain Guides
Read these resources for detailed operations:
- `napari://tensor` — dataset access and upload
- `napari://viewer` — layers, camera, dims, display
- `napari://annotations` — segmentations, points, shapes, labels

## Quick Examples
```python
# Check what data is on the viewer
print([(l.name, type(l).__name__, type(l.data).__name__) for l in viewer.layers])

# Get data from the catalog
np_arr = client.get_tensor("my_source_id").compute()

# Take action then screenshot to verify
viewer.dims.ndisplay = 3
```

## Iterative Workflow for _very_ large data
```python
# 1. Load source data as dask array (lazy)
arr = client.get_tensor("raw_data_id")

# 2. Process
mask_arr = arr > 0.5

# 3. Upload Will call compute() chunk by chunk under the hood.
source_id = client.upload_array(mask_arr, "cache:thresholded_v1")

# 4. Display in viewer for user inspection and approval before next step
layer_name = viewer.load_tensor(source_id)
```
"""

VIEWER = """\
# Viewer Operations

## Layers
```python
# List all layers
for layer in viewer.layers:
    print(f"{layer.name}: {type(layer).__name__} {layer.data.shape} {layer.data.dtype}")

# Get specific layer
layer = viewer.layers["image_name"]

# Remove layer
viewer.layers.remove(viewer.layers["name"])

# Load data to viewer; auto-handles pyramid. Accepts any source_id, including
# those beyond the list_sources() cap (fetched from the server on demand).
layer_name = viewer.load_tensor(source_id="source_id", tensor_id=None, name=None)

# Layer properties
layer = viewer.layers["name"]
layer.visible = False
layer.opacity = 0.7
layer.colormap = "viridis"
layer.contrast_limits = [0, 255]
layer.blending = "additive"         # "translucent", "additive", "minimum", "opaque"
```

## Dimensions (sliders)
```python
# Set slider position (e.g. time axis=0 to frame 50)
viewer.dims.set_point(axis=0, value=50)

# Get current position
print(viewer.dims.point)    # tuple of current positions
```

## Layer types
Image, Labels, Points, Shapes, Vectors, Surface, Tracks
Use `inspect_object("viewer.add_image")` for full signatures.
"""

TENSOR = """\
# Tensor Data Operations

## Check Connection
```python
if client is None:
    print("Not connected. Open Tensor Browser widget and connect first.")
else:
    print(client.health_check())
```

## Browse Sources
For large catalogs the server caps `list_sources()`, so prefer the SQL query
to discover source_ids; there is no pre-built `sources` dict in the namespace.
```python
# Preferred: server-side DuckDB query (complete, not truncated).
# The sources table has columns: source_id, source_url, source_type, metadata_json
arrow_table = client.query_sources("SELECT source_id FROM sources WHERE source_type='ome-zarr'")
print(arrow_table.to_pandas())

# Convenience listing (NOTE: capped by the server for large catalogs)
for sid, src in client.list_sources().items():
    tensors = [(t.array_id, list(t.shape), t.dtype) for t in src.tensors]
    print(f"{sid}: {src.source_url} ({src.source_type}) tensors={tensors}")

# Detailed metadata (OME_JSON) for one source
meta = client.get_source_metadata("source_id")
print(meta)
```

## Load into Viewer
```python
# Auto-handles the multiscale pyramid for large images.
layer_name = viewer.load_tensor("source_id")                   # single-tensor source
layer_name = viewer.load_tensor("source_id", tensor_id="t1")   # multi-tensor source

# Or get a lazy dask array directly, without adding a layer:
arr = client.get_tensor("source_id", tensor_id=None)           # tensor_id optional if single
```

## Upload to Server
Use `"cache:my_result"` as destination for ephemeral results that don't
need to be persisted long-term.
```python
source_id = client.upload_array(arr, "cache:my_result")
```
"""

ANNOTATIONS = """\
## Labels
```python
# Create empty labels layer for painting
shape = viewer.layers["image_name"].data.shape[-2:]  # match y,x of image
viewer.add_labels(np.zeros(shape, dtype=np.int32), name="annotations")

# Create labels from mask
mask = (image_data > threshold).astype(np.int32)
viewer.add_labels(mask, name="segmentation")

# Read labels
labels_data = viewer.layers["annotations"].data
unique_labels = np.unique(labels_data)
print(f"Labels present: {unique_labels}")
```
For points, shapes, and other layer types use `inspect_object` to discover
the full API:
```python
inspect_object("viewer.add_points")
inspect_object("viewer.add_shapes")
```
"""
