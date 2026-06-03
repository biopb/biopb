"""MCP resource content for the developer guide system.

Each constant is served as an MCP resource that the agent reads on demand.
"""

GUIDE = """\
# biopb-mcp Automation Guide

## Domain Guides
Read these resources for detailed operations:
- `guide://tensor` — dataset access and upload
- `guide://viewer` — layers, camera, dims, display
- `guide://annotations` — segmentations, points, shapes, labels
- `guide://ops` — server-side image processing operations

## Execution environment
Code submitted via `execute_code` runs in a child Jupyter kernel (a real IPython kernel)
with napari integrated via `%gui qt`. Imports are allowed and variables persist across
`execute_code` calls until the kernel is restarted.

### Namespace (available in execute_code)
| Name | Type | Description |
|------|------|-------------|
| `client` | TensorFlightClient or None | Client instance (None if not connected to data server) for browsing/retrieving catalog data |
| `viewer` | napari.Viewer | The active viewer instance that user sees |
| `np` | module | numpy |
| `da` | module | dask.array |
| `ops` | dict[str, callable] | biopb.image ProcessImage operations from configured servers (may be empty) |

* The viewer is a live desktop window; mutations show up immediately.
* Data from `TensorFlightClient` are lazy, thread-safe, picklable dask arrays.
* `ops` maps op name -> a inspectable callable that runs dedicated image processing logics.

## Operation Guardrails
* All data are from `client` or `viewer`. Avoid direct accessing file systems unless specifically requested by user.
* Prefer browsing the catalog through `client.query_sources(sql)` (server-side DuckDB, complete) rather than `client.list_sources()` (capped by the server for large catalogs).
* Prefer lazy dask operations and only `.compute()` the final result.
* Intermediate results should be put back on viewer to be validated by user before next step.
* Do _not_ assume. Ask the user to clarify uncertainties - they know the data best.
* After accomplishing a task, ask the user if a skill should be added to the agent's toolbox for future use.

## Quick Examples
```python
# Check what data is on the viewer
print([(l.name, type(l).__name__, type(l.data).__name__) for l in viewer.layers])

# Get data from the catalog and convert to np.ndarray
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

## Canvas mouse events
You can detect user clicks/drags/moves on the canvas — this works reliably in
this kernel (verified end to end). Use napari's viewer-model API, not the raw Qt
widget or vispy canvas. You cannot see the cursor live: install a callback, let
the user interact, then read back what you captured.

```python
# Also: mouse_move_callbacks, mouse_double_click_callbacks. The event has
# .button, .modifiers, .position (world coords), and .pos (canvas pixels).
def on_click(viewer, event):
    layer = viewer.layers.selection.active
    if layer is not None:
        coord = layer.world_to_data(event.position)  # full-ndim data coords (…,z,y,x)
        print(event.button, list(event.modifiers), coord)

viewer.mouse_drag_callbacks.append(on_click)   # mutate in place (see below)
```

If a callback "doesn't fire", it is one of these — NOT a session/setup bug, and
do **not** instrument vispy to investigate (that is the trap, see point 2):

1. **Reassigning instead of mutating the list.** It is an evented pydantic
   field; `viewer.mouse_drag_callbacks = [...]` raises `"Viewer" object has no
   field`. Use `.append()` / `.remove()`.
2. **Tapping the vispy emitter** (`canvas._scene_canvas.events.*`). napari runs
   it with `ignore_callback_errors=False` and vispy `connect()` defaults to
   `position='first'`, so a tap landing ahead of napari's handler that raises
   halts the chain and suppresses napari's callbacks — a working setup looks
   broken. Stay on `viewer.*_callbacks`; if you must, use `position='last'` + try/except.
3. **Window not focused** — click once to focus it, then interact.
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

OPS = """\
## Image Processing Ops (`ops`)
`ops` maps op name -> a thin callable that runs one `biopb.image.ProcessImage`
Discover and inspect them before use — each carries a docstring with its server, labels,
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
"""
