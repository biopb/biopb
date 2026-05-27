"""MCP resource content for the developer guide system.

Each constant is served as an MCP resource that the agent reads on demand.
"""

GUIDE = """\
# napari-biopb MCP Automation Guide

## Tools
- `take_screenshot(canvas_only=True)` — capture viewer as PNG image
- `execute_code(python_code)` — run Python with the namespace below
- `inspect_object(object_path)` — reflect on any object (e.g. "viewer.layers")

## Namespace (available in execute_code)
| Name | Type | Description |
|------|------|-------------|
| `sources` | dict | Cached {source_id: DataSourceDescriptor} data catalog |
| `client` | TensorFlightClient or None | Client instance (None if not connected to data server) for retrieving catalog data |
| `viewer` | napari.Viewer | The active viewer instance that user sees |
| `np` | module | numpy |
| `da` | module | dask.array |

## Threading & dask
Code runs on the Qt main thread, with a preconfigured dask scheduler for background computation.
Avoid long-running non-dask operations to prevent UI blocking.

Data from `TensorFlightClient` is always returned as thread-safe, serializable
dask arrays — safe to use with any scheduler including distributed clusters.
Prefer lazy dask operations and only `.compute()` the final result. If compute final results would
OOM, directly upload array to server with `client.upload_array()` with a ephemeral destination (e.g.
`"cache:my_result"`), instead of running `arr.compute()`.

## Domain Guides
Read these resources for detailed operations:
- `napari://tensor` — dataset access and upload
- `napari://viewer` — layers, camera, dims, display
- `napari://annotations` — points, shapes, labels

## Quick Examples
```python
# List layers
print([(l.name, type(l).__name__, l.data.shape) for l in viewer.layers])

# Load a tensor dataset (connect in Tensor Browser widget first)
viewer.load_tensor("my_source_id")

# Take action then screenshot to verify
viewer.dims.ndisplay = 3
```

## Iterative Workflow for _very_ large data
```python
# 1. Load source data as dask array (lazy)
arr = client.get_tensor("raw_data_id")

# 2. Process
mask_arr = arr > 0.5

# 3. Upload for next step. Will call compute chunk by chunk under the hood.
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

# Load data from catalog to viewer; auto-handles pyramid for large images
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
```python
# List all sources
for sid, src in sources.items():
    tensors = [(t.array_id, list(t.shape), t.dtype) for t in src.tensors]
    print(f"{sid}: {src.source_url} ({src.source_type}) tensors={tensors}")

# Detailed metadata (OME_JSON)
meta = client.get_source_metadata("source_id")
print(meta)

# Query metadata with DuckDB SQL. The sources table has columns: source_id, source_url, source_type, metadata_json
arrow_table = client.query_sources("SELECT source_id FROM sources WHERE source_type='ome-zarr'")
print(arrow_table.to_pandas())
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
