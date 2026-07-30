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

* The viewer is a live desktop window, and the `viewer` handle is **thread-safe**: every mutation (`viewer.dims`, `viewer.camera`, layer properties, `viewer.layers.remove()`, the `add_*()` family, …) is automatically marshaled to the Qt main thread, so just mutate it directly from job code. Two caveats: raw Qt (`viewer.window`) still requires the main thread — off-thread access raises a clear error, so wrap it in `run_on_main()`; and to apply many mutations in one main-thread hop, batch them in a single `run_on_main()`.
* Data from `TensorFlightClient` are lazy, thread-safe, picklable dask arrays.
* **Server data, viewer data and your own arrays are three different things** —
  different axis order, different resolution levels, different laziness. Read
  `guide://data` before writing code that reads pixels off a layer.
* `ops` maps op name -> an inspectable callable that runs dedicated image-processing logic.

**User plugins may add more names** beyond this table (biopb-mcp#92): top-level
definitions from `*.py` files in `~/.config/biopb/kernel/`, and installed
`biopb_mcp.namespace` packages, are loaded into this namespace at kernel start.
They're lab-specific helpers, not built-ins — so if you see an unfamiliar name,
it's likely one of these. Discover and read them by introspection:
```python
[n for n in dir() if not n.startswith("_")]   # everything actually in scope
inspect_object("<name>")                        # its signature + docstring
```

## Skill requirements
A curated skill from `find_skills` carries a `requires:` list. **Resolve it before you start
the skill** — one that assumes a plugin or package it doesn't have fails partway through,
after the user has already waited.

One `server_status` call answers every token but a third-party `pkg:`:

| token | read it from | if it's missing |
|---|---|---|
| `viewer` | `## Viewer` | see **headless / closed window** below |
| `tensor` | `## Tensor Server` | the section names the reason; see below |
| `dask` | `## Dask` | never missing — `da` is always bound; the scheduler is a performance property, not a blocker |
| `ops:<kind>` | `## Ops` | see below |
| `plugin:<name>` | `## Kernel plugins` | see below — this is the **only** place it can be read |
| `pkg:biopb-mcp` | `## Versions` — the version installed in **this kernel's** interpreter, the one that will run the skill | the install is older than the skill; the fix is upgrading biopb |
| `pkg:<other>` | `import <name>` (`import skimage; skimage.__version__`) | see below |

`plugin:<name>` has no other source: a plugin file contributes its function names, not its
own name, so `dir()` can't answer it, and a file that failed to load is still on disk, so a
directory listing can't either. The report is the loader's own record of what survived.

### When something is missing
**Diagnose, tell the user, let them choose.** Installing, seeding and restarting are all
theirs to authorize — but a named gap usually beats abandoning the skill, and several of
these have a fix worth offering.

**`plugin:<name>` — three different causes, in this order:**
1. **Is `pkg:biopb-mcp>=X` in the same `requires:` also unmet?** Then this install simply
   predates the plugin. Seeding cannot conjure it — say so, and point at upgrading biopb
   (rerun the installer). Stop there.
2. **Else check the file:** `ls ~/.config/biopb/kernel/<name>.py`. Present, but absent from
   the report → **it failed to load.** The traceback is in the session log (`log_file:` in
   `## System`). Show the user the error; this is a bug to report, not something to retry.
3. **Absent → never seeded.** `biopb-mcp-seed-plugins` installs the built-ins, then the
   kernel must restart to load them. **Ask before restarting** — it takes the namespace and
   every layer with it.

**`pkg:<name>` — offer three options and let the user pick:**
1. **They install it** — quote the exact command `## Versions` prints (it names *this*
   interpreter). Never a bare `pip install`: it targets whatever env their shell has active,
   which can succeed while the import here still fails.
2. **You install it for them** — same command, run from `execute_code` via `subprocess`,
   **only after they say yes.** Then `importlib.invalidate_caches()` and import again; if the
   module was already half-imported, `restart_kernel` (ask — layers are lost).
3. **The skill's degraded path**, if it names one. Often the right answer for a one-off run:
   nothing to undo, and it is the option that survives an upgrade.

Whichever of the first two they pick, if `## Versions` says the env is uv-managed, say so and
name `~/.config/biopb/extra-packages.txt`: the install lands now but is gone at the next biopb
upgrade unless the requirement is in that file.

**`viewer` — two different failures.** *Headless* means no window exists this session: run
the skill's numeric checks and report numbers, never a screenshot. *`window: CLOSED`* means
data and compute still work but nothing displays; `restart_kernel` restores it (ask — layers
are lost). Neither is a reason to stop if the skill's results are numbers.

**`tensor`.** The section names the cause (not connected / auth / still starting). Check
`biopb control status` with the user, or point at `$BIOPB_TENSOR_URL` if the data lives on a
server the control doesn't own. Do **not** proceed as if the catalog were empty — that reads
to the user as "no data" rather than "not connected".

**`ops:<kind>`.** `## Ops` lists what the servers *do* offer, so say whether one of them
covers the same need — but **ask before substituting** an op the skill didn't name, since a
different model is a different result. Otherwise the user adds a server to
`services.process_image_servers`.

## Long-running jobs
A slow `execute_code` call runs in a background thread and returns a `job-N` handle;
watch it with `poll_job` / `take_screenshot` / `server_status`, stop it with `interrupt_kernel`
(best-effort, raises KeyboardInterrupt into the job and cancels in-flight dask tasks) or
`restart_kernel` (guaranteed, kills the kernel). Notes:
* **A blocking `.compute()` is interruptible** — `interrupt_kernel` cancels the in-flight
  dask tasks, so the `.compute()` raises and the job ends. No special pattern needed.
* **Your own long loops** (per-chunk / per-file) are stopped by `interrupt_kernel`, which
  raises `KeyboardInterrupt` into the loop at the next iteration — no cooperative check needed.
* **Progress on a big graph:** submit with the distributed client
  (`_dask_client`, present only under the distributed scheduler) and consume results as
  they land — this gives a live processed count via `poll_job`:
  ```python
  from dask.distributed import as_completed
  futs = _dask_client.compute(list_of_dask_results)   # list of Futures, non-blocking
  done = []
  for fut in as_completed(futs):
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

DATA = """\
# Where array data lives, and what it actually is

Three places in this session hold pixels, and they are **not interchangeable**.
Most of the wrong answers this session can produce start with code that treats
them as if they were — a napari habit that is right in a plain napari script and
wrong here, because the arrays come off a tensor server, lazily, in a pyramid.

| Source | You get it with | What you get |
|---|---|---|
| **Tensor server** | `client.get_tensor(array_id)` | A lazy dask array. **Source axis order** (the tensor's `dim_labels`), full resolution, nothing materialized yet |
| **Viewer layer** | `layer.data` | Whatever napari is *displaying*: possibly a **list** of pyramid levels, each a proxy rather than a dask array, in **display order** `[..., Z, Y, X]` |
| **Kernel** | your own variables | Exactly what you made — numpy if you computed it, dask if you didn't. It carries no physical scale unless you carried it |

`viewer.add_tensor(source_id)` is a *conversion between the first two*, not a
window onto the first. Everything below follows from that.

## The traps

**1. `layer.data` is a list when the layer is multiscale.** `add_tensor` builds a
resolution pyramid for anything large, and then `layer.data` is
`[full_res, half, quarter, ...]`. `layer.data.shape` raises `AttributeError`,
`np.asarray(layer.data)` tries to stack levels of different shapes, and anything
that "works" on it is working on the wrong thing. Always branch:

```python
arr = layer.data[0] if layer.multiscale else layer.data   # level 0 = full res
```

**2. Level 0 is not the only level, and the others are different data.** A
measurement taken on level 2 is off by the downsample factor, silently and
plausibly. Never pair arrays from two different levels (a label layer's level 0
with an image layer's level 1 is the common version of this).

**3. `layer.data` is not a dask array.** `add_tensor` wraps each level in a
`_ViewerArray` proxy that pins napari's own slice reads to a single-process
scheduler (issue #8). It behaves like a dask array — `.shape`, `.dtype`,
`.compute()`, slicing, `arr > 0`, ufuncs — and operators return plain dask, so
ordinary work is unaffected. Two things do differ: `isinstance(arr, da.Array)` is
`False` (test for `da.Array` on `client.get_tensor` output, not on layer data),
and a bare `np.asarray(arr)` materializes the **whole** array in the main process
rather than on the cluster. Slice first, or `.compute()` explicitly.

**4. The viewer's axes are not the source's axes.** Levels are canonicalized to
`[..., Z, Y, X]` (plus a trailing samples axis for interleaved RGB) regardless of
how the source is laid out. So:

* `client.get_physical_scale(array_id)` is in **source** order — it pairs with
  `client.get_tensor(array_id)`, never with `layer.data`.
* `layer.scale` is in **layer** order — it pairs with `layer.data`, and it is
  what `regionprops(..., spacing=)` wants.

Mixing the two transposes your spacing onto the wrong axes, which changes every
number without changing any shape.

**5. A 2-D source is 3-D in the viewer.** Canonicalization *inserts* a singleton
Z when the tensor has none, so a `[Y, X]` source loads as `[1, Y, X]`, with a
3-element `layer.scale`. Read `ndim` off the array you are actually about to use;
do not infer it from the source.

**6. Lazy means the bill arrives at the end.** Everything from the server is
lazy, so `.shape` and `.dtype` are free while the pixels are not there yet. A
scikit-image call, `np.asarray`, or a plain `for` loop over the array
materializes it in full, in one go, with no progress and no chunking — which is
how a session ends up allocating a volume it cannot hold. Crop or slice first,
keep the lazy chain, and `.compute()` once at the end; past `promote_after` that
compute is a job you can watch and cancel (`guide://kernel`).

**7. A layer you built yourself carries no geometry.** `add_labels(np_array)` /
`add_image(np_array)` store exactly that array, with `scale` defaulting to all
ones and no pyramid. A segmentation added beside a calibrated image therefore
measures in pixels unless you copy the image layer's `scale` onto it.

**8. Upload does not carry labels or physical size.** `client.upload_array(arr,
"cache:name")` takes a **dask** array (wrap numpy with `da.from_array`) and
stores shape, dtype and chunks. Axis labels and pixel size travel only if you
pass `dim_labels=` / `ome_metadata=` — otherwise the round trip through the
server quietly drops them, and `add_tensor` on the result gives an uncalibrated
layer.

## Reading a layer safely

```python
layer = viewer.layers[NAME]
arr = layer.data[0] if layer.multiscale else layer.data   # traps 1, 2
print(arr.shape, arr.dtype, layer.scale, layer.multiscale)
sub = arr[0, 100:600, 100:600].compute()                  # trap 3, 6: crop, then compute
```

`print([(l.name, type(l).__name__, l.multiscale, l.data[0].shape if l.multiscale
else l.data.shape) for l in viewer.layers])` is the one-liner worth running
before you plan any work off the viewer.
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

**Layer data is not a plain array.** A layer loaded by `add_tensor` holds a
pyramid (`layer.data` is a *list*), in display axis order, wrapped so napari's
slice reads stay in-process. `guide://data` has the full set of traps; the rule
of thumb is `layer.data[0] if layer.multiscale else layer.data`.

## Layers
```python
# List all layers (read)
for layer in viewer.layers:
    arr = layer.data[0] if layer.multiscale else layer.data   # multiscale => list
    print(f"{layer.name}: {type(layer).__name__} {arr.shape} {arr.dtype}")

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

CLIENT = """\
# The `client` Handle: Catalog and Tensor Data

Arrays from here are lazy dask arrays in the tensor's own axis order, and a
tensor loaded into the viewer stops being either of those — see `guide://data`
before moving pixels between the server, a layer, and your own variables.

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
# indexed_at, metadata_json, shape_summary, data_resident, and `tensors`
# (a LIST of STRUCT(array_id, dim_labels, shape, chunk_shape, dtype) -- one
# per tensor; `dtype`/`shape_summary` are just the first-tensor projection).
df = client.query_sources("SELECT source_id FROM sources WHERE source_type='ome-zarr'", format="pandas")
print(df)

# Per-tensor queries (multi-field / HCS sources): use the nested `tensors`
# column with UNNEST or list_filter -- the scalar dtype/shape_summary only
# describe tensors[0].
client.query_sources(  # every tensor, one row each
    "SELECT source_id, t.array_id, t.shape, t.dtype "
    "FROM sources, UNNEST(tensors) AS u(t)", format="pandas")
client.query_sources(  # sources having ANY uint16 tensor
    "SELECT source_id FROM sources "
    "WHERE len(list_filter(tensors, t -> t.dtype = 'uint16')) > 0", format="pandas")

# Convenience listing (NOTE: capped by the server for large catalogs)
for sid, src in client.list_sources().items():
    tensors = [(t.array_id, list(t.shape), t.dtype) for t in src.tensors]
    print(f"{sid}: {src.source_url} ({src.source_type}) tensors={tensors}")

# Detailed metadata (OME_JSON) for one source
meta = client.get_source_metadata("source_id")
print(meta)
```

## Cloud / unresolved sources (experimental)
Cloud / remote source support is **experimental** and may change.
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
Hydrate-ahead (optional): `resolve()` fetches a multi-file source's *metadata*
only -- the bulk data files (e.g. zarr/ome-zarr chunks) still recall one-by-one,
slowly, the first time a read touches them, which makes the first pass over a big
image stall repeatedly. If you're about to work through the whole source, warm it
up front so the server pulls every member file resident in one go (server-side;
no pixels cross to the kernel). It's idempotent and reports progress:
```python
done = client.warm("source_id",
                   on_progress=lambda p: print(f"{p.files_done}/{p.files_total}"))
# Long-running; interrupt_kernel cancels it (the stream closes, the server stops).
# Single-file sources are a no-op (resolve already recalled the one file).
```
Filter footgun: an unresolved source has NULL `dtype`/`shape_summary` in the
`sources` table, so `query_sources("... WHERE dtype='uint8'")` silently *drops*
it -- it's hidden for being unresolved, not for not matching. The table carries
a `data_resident` column so you can filter on residency on purpose:
```python
# what hasn't been resolved (downloaded) yet?
client.query_sources("SELECT source_id, source_url FROM sources WHERE NOT data_resident",
                     format="pandas")
```

## Load into Viewer
```python
# Auto-handles the multiscale pyramid for large images.
layer_name = viewer.add_tensor("source_id")                   # single-tensor source
layer_name = viewer.add_tensor("source_id", tensor_id="t1")   # multi-tensor source

# Or get a lazy dask array directly, without adding a layer (address it by its
# array_id: "source_id/t1" for a multi-tensor source, "source_id" for a single one):
arr = client.get_tensor("source_id/t1")
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
