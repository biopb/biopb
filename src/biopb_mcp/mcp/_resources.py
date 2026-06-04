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
| `run_on_main` | callable | `run_on_main(fn)` runs `fn` on the Qt main thread and returns its result; required for viewer mutations from a job thread (no-op on the main thread) |
| `cancelled` | callable | `cancelled()` -> True if the running job has been asked to cancel; poll it in long loops for cooperative cancellation |

* The viewer is a live desktop window. Reads are safe from a job thread, but mutations must run on the main thread — wrap them in `run_on_main()`. `viewer.load_tensor()` and the `viewer.add_*()` family are already wrapped; everything else (layer properties, `viewer.dims`, `viewer.camera`, callback registration) is not.
* Data from `TensorFlightClient` are lazy, thread-safe, picklable dask arrays.
* `ops` maps op name -> an inspectable callable that runs dedicated image-processing logic.

## Long-running jobs & cancellation
A slow `execute_code` call runs in a background thread and returns a `job-N` handle;
watch it with `poll_job` / `take_screenshot` / `server_status`, stop it with `cancel_job`
(graceful) or `restart_kernel` (guaranteed, kills the kernel). To stay stoppable:
* **A blocking `.compute()` is interruptible** — `cancel_job` cancels the in-flight dask
  tasks, so the `.compute()` raises and the job ends. No special pattern needed.
* **Your own long loops** (per-chunk / per-file) have no dask futures to cancel, so poll
  `cancelled()` and `break` yourself.
* **Progress + responsive cancel on a big graph:** submit with the distributed client
  (`_dask_client`, present only under the distributed scheduler) and consume results as
  they land — this also gives a live processed count via `poll_job`:
  ```python
  from dask.distributed import as_completed
  futs = _dask_client.compute(list_of_dask_results)   # list of Futures, non-blocking
  done = []
  for fut in as_completed(futs):
      if cancelled():
          _dask_client.cancel(futs); break
      done.append(fut.result())
      print(f"{len(done)}/{len(futs)} done", flush=True)   # visible via poll_job
  ```

## Operation Guardrails **IMPORTANT**
* All data should be from `client` or `viewer`. Avoid directly accessing the file system unless specifically requested by the user.
* Prefer browsing the catalog through `client.query_sources(sql)` (server-side DuckDB, complete) than `client.list_sources()` (capped by the server for large catalogs).
* Prefer lazy dask operations and only `.compute()` the final result.
* Intermediate results should be put back on `viewer` to be validated by user at each step.
* Do _not_ assume. Ask the user to clarify uncertainties - they know the data better than you do.
* After accomplishing a task, ask the user if a skill should be added to the agent's toolbox for future use.

## Quick Examples
```python
# Check what data is on the viewer
print([(l.name, type(l).__name__, type(l.data).__name__) for l in viewer.layers])

# Get data from the catalog and convert to np.ndarray
dask_arr = client.get_tensor("my_source_id") # lazy, thread-safe, picklable
np_arr = dask_arr.compute() # in memory

# Take action then screenshot to verify
run_on_main(lambda: setattr(viewer.dims, "ndisplay", 3))
```

## Iterative Workflow for _very_ large data
```python
# 1. Load source data as dask array (lazy)
arr = client.get_tensor("raw_data_id")

# 2. Process
mask_arr = arr > 0.5

# 3. Upload. Calls compute() chunk by chunk under the hood.
source_id = client.upload_array(mask_arr, "cache:thresholded_v1")

# 4. Display in viewer for user inspection and approval before next step
layer_name = viewer.load_tensor(source_id)
```
"""

VIEWER = """\
# Viewer Operations

**Threading:** reads are safe from a job thread, but every *mutation* must run on
the Qt main thread. `viewer.load_tensor()` and the `viewer.add_*()` family are
already wrapped for you. For anything else — layer properties, `viewer.dims`,
`viewer.layers.remove()`, `viewer.camera`, registering callbacks — wrap the
mutation in `run_on_main()` (a no-op when already on the main thread). Batch
related mutations into one `run_on_main()` call.

## Layers
```python
# List all layers (read — no run_on_main needed)
for layer in viewer.layers:
    print(f"{layer.name}: {type(layer).__name__} {layer.data.shape} {layer.data.dtype}")

# Get specific layer (read)
layer = viewer.layers["image_name"]

# Remove layer (mutation — wrap it)
run_on_main(lambda: viewer.layers.remove(viewer.layers["name"]))

# Load data to viewer; auto-handles pyramid. Accepts any valid source_id.
# (already wrapped — call directly)
layer_name = viewer.load_tensor(source_id="source_id", tensor_id=None, name=None)

# Layer properties (mutations — batch into one run_on_main call)
def _style():
    layer = viewer.layers["name"]
    layer.visible = False
    layer.opacity = 0.7
    layer.colormap = "viridis"
    layer.contrast_limits = [0, 255]
    layer.blending = "additive"     # "translucent", "additive", "minimum", "opaque"
run_on_main(_style)
```

## Dimensions (sliders)
```python
# Set slider position (mutation — wrap it; e.g. time axis=0 to frame 50)
run_on_main(lambda: viewer.dims.set_point(axis=0, value=50))

# Get current position (read)
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

# Register from the main thread (mutation), and mutate the list in place (below)
run_on_main(lambda: viewer.mouse_drag_callbacks.append(on_click))
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
arr = client.get_tensor("source_id", tensor_id="t1")           # tensor_id optional if single
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
`ops` maps op name -> a thin callable that runs one `biopb.image.ProcessImage` op.
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
