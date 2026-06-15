"""MCP resource content for the developer guide system.

Each constant is served as an MCP resource that the agent reads on demand.
"""

GUIDE = """\
# biopb-mcp IPython Kernel Guide

**Operation Guardrails** are in session `instructions`. Apply on every turn — follow them throughout.

## Namespace
| Name | Type | Description |
|------|------|-------------|
| `client` | TensorFlightClient or None | Connection to the data server for browsing/retrieving image data |
| `viewer` | napari.Viewer | The active viewer instance that user sees and manipulates |
| `np` | module | numpy |
| `da` | module | dask.array |
| `ops` | dict[str, callable] | biopb.image ProcessImage operations from configured servers (may be empty) |
| `run_on_main` | callable | `run_on_main(fn)` runs `fn` on the Qt main thread and returns its result (no-op on the main thread). Rarely needed — the `viewer` already auto-marshals every mutation. Use it only to **batch** many viewer mutations into one main-thread hop, or to touch raw Qt (`viewer.window`). |
| `cancelled` | callable | `cancelled()` -> True if the running job has been asked to cancel; poll it in long loops for cooperative cancellation |

* The viewer is a live desktop window, and the `viewer` handle is **thread-safe**: every mutation (`viewer.dims`, `viewer.camera`, layer properties, `viewer.layers.remove()`, the `add_*()` family, …) is automatically marshaled to the Qt main thread, so just mutate it directly from job code. Two caveats: raw Qt (`viewer.window`) still requires the main thread — off-thread access raises a clear error, so wrap it in `run_on_main()`; and to apply many mutations in one main-thread hop, batch them in a single `run_on_main()`.
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

## Quick Examples
```python
# Check what data is on the viewer
print([(l.name, type(l).__name__, type(l.data).__name__) for l in viewer.layers])

# Get data from the catalog and convert to np.ndarray
dask_arr = client.get_tensor("my_source_id") # lazy, thread-safe, picklable
np_arr = dask_arr.compute() # in memory

# Take action then screenshot to verify (mutations auto-marshal — call directly)
viewer.dims.ndisplay = 3
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
layer_name = viewer.add_tensor(source_id)
```
"""

VIEWER = """\
# Viewer Operations

**Threading:** the `viewer` is thread-safe — every mutation (layer properties,
`viewer.dims`, `viewer.layers.remove()`, `viewer.camera`, the `add_*()` family)
is automatically marshaled to the Qt main thread, so mutate it directly from job
code. `run_on_main()` is optional: use it to **batch** many mutations into one
main-thread hop (one round-trip instead of one per mutation), or to touch raw Qt
(`viewer.window`), which still requires the main thread and otherwise raises a
clear error off-thread.

**If the user closes the napari window**, the kernel is torn down to idle and
any running job is stopped. `server_status` then reports the kernel `not
started`, attributing it to the window close, and the kernel-dependent tools
return the same. Call `start_kernel` to rebuild the viewer. (Briefly, before
the teardown completes, a tool may instead see `window: CLOSED` with a note to
`restart_kernel` — either recovers it.)

## Layers
```python
# List all layers (read)
for layer in viewer.layers:
    print(f"{layer.name}: {type(layer).__name__} {layer.data.shape} {layer.data.dtype}")

# Get specific layer (read)
layer = viewer.layers["image_name"]

# Remove layer (auto-marshaled — call directly)
viewer.layers.remove(viewer.layers["name"])

# Load data to viewer; auto-handles pyramid. Accepts any valid source_id.
layer_name = viewer.add_tensor(source_id="source_id", tensor_id=None, name=None)

# Layer properties (auto-marshaled — set directly; each runs on the main thread)
layer = viewer.layers["name"]
layer.visible = False
layer.opacity = 0.7
layer.colormap = "viridis"
layer.contrast_limits = [0, 255]
layer.blending = "additive"     # "translucent", "additive", "minimum", "opaque"

# To apply many at once in a single main-thread hop, batch with run_on_main:
def _style():
    layer = viewer.layers["name"]
    layer.visible, layer.opacity, layer.colormap = False, 0.7, "viridis"
run_on_main(_style)
```

## Dimensions (sliders)
```python
# Set slider position (auto-marshaled — call directly; e.g. time axis=0 to frame 50)
viewer.dims.set_point(axis=0, value=50)

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

# Register by mutating the callback list in place — append, do not reassign (below)
viewer.mouse_drag_callbacks.append(on_click)
```

If a callback "doesn't fire", it is one of these — NOT a session/setup bug, and
do **not** instrument vispy to investigate (that is the trap, see point 2):

1. **Reassigning instead of mutating the list.** It is not a reassignable
   attribute; `viewer.mouse_drag_callbacks = [...]` raises `"Viewer" object has
   no field`. Use `.append()` / `.remove()`.
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
# The sources table columns: source_id, source_url, source_type, dtype,
# indexed_at, metadata_json, shape_summary
df = client.query_sources("SELECT source_id FROM sources WHERE source_type='ome-zarr'", format="pandas")
print(df)

# Convenience listing (NOTE: capped by the server for large catalogs)
for sid, src in client.list_sources().items():
    tensors = [(t.array_id, list(t.shape), t.dtype) for t in src.tensors]
    print(f"{sid}: {src.source_url} ({src.source_type}) tensors={tensors}")

# Detailed metadata (OME_JSON) for one source
meta = client.get_source_metadata("source_id")
print(meta)
```

## Cloud / unresolved sources
Some sources (cloud / synced-folder, e.g. OneDrive "Files-On-Demand") are
catalogued by URL only: their shape/dtype/fields are *unknown* until first read.
They list with `data_resident == False` and an empty `tensors`, and reading one
(`get_tensor`/`add_tensor`) raises until you resolve it. Resolving asks the
server to **download the whole file** (slow, uses disk, fails offline), so it is
explicit -- never triggered by browsing.
```python
src = client.list_sources()["source_id"]
if not src.data_resident:                    # unresolved / not local
    src = client.resolve("source_id")        # downloads + resolves (may take minutes)
    tensors = [(t.array_id, list(t.shape)) for t in src.tensors]  # now populated
```

## Load into Viewer
```python
# Auto-handles the multiscale pyramid for large images.
layer_name = viewer.add_tensor("source_id")                   # single-tensor source
layer_name = viewer.add_tensor("source_id", tensor_id="t1")   # multi-tensor source

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
viewer.add_tensor(seg_id)                    # view the result
```
"""
