"""MCP resource content for the developer guide system.

Each constant is served as an MCP resource that the agent reads on demand.
"""

GUIDE = """\
# biopb-mcp IPython Kernel Guide

**Operation Guardrails** are in session `instructions`. Apply on every turn — follow them throughout.

Kernel is the main execution context for the agent. It is a live Python interpreter with a napari
viewer window, and it has access to the TensorFlightClient for browsing and retrieving image data.
The kernel is **stateful**: variables, imports, and viewer state persist across turns. Agent code
runs in a background thread to keep the main Qt thread responsive.

## Namespace

| Name | Type | Description |
|------|------|-------------|
| `client` | TensorFlightClient or None | Connection to the data server for browsing/retrieving image data. Marshaled and thread-safe. |
| `viewer` | napari.Viewer | The active viewer instance that user sees and manipulates |
| `np/da` | module | imported packages: numpy and dask.array |
| `ops` | dict[str, callable] | biopb.image ProcessImage operations from configured servers (may be empty) |
| `run_on_main` | callable | runs `fn` on the Qt main thread and returns its result. Use it to **batch** many viewer mutations into one main-thread hop, or to touch raw Qt (`viewer.window`). |

- The `viewer` is a live napari window, made **thread-safe** by marshaling known
  mutations (`viewer.dims`, `viewer.camera`, layer properties, `viewer.layers.remove()`,
  the `add_*()` family, …) to the Qt main thread. One caveat: raw Qt (`viewer.window`)
  still requires the main thread — off-thread access raises a clear error, so wrap it in
  `run_on_main()`. See `guide://viewer` for the full set of viewer operations, including
  mouse events.
- The `client` represents a `TensorFlightClient` instance. Data from the client are
  lazy, thread-safe, picklable dask arrays. See `guide://client` for the full set of client
  operations, including browsing sources, reading tensors, and uploading results; and see
  `guide://data` for the traps when moving pixels between the server, a layer, and your own
  variables.
- `ops` maps op name -> an inspectable callable that runs dedicated image-processing logic.
  The callable is a thin wrapper around a `biopb.image.ProcessImage` gRPC service on a configured
  server. The callable can take either a numpy array (eager) or a tensor-server array_id string
  (lazy). See `guide://ops` for details.
- **Only `np` and `da` are pre-imported.** Everything else needs an explicit
  import, and these are guaranteed installed: `pandas`, `skimage`, `scipy`,
  `sklearn`, `matplotlib`, `cv2` (opencv-headless), `ome_zarr`, `napari`. The
  kernel is stateful, so one import per session is enough.

## Kernel plugins

**User plugins may add more names** beyond the table above: each `*.py` file in
`~/.config/biopb/kernel/`, and each installed `biopb_mcp.namespace` package, is
loaded at kernel start and bound as **one module, named after the file** —
`rolling_ball.py` becomes `rolling_ball`, and its functions are called as
`rolling_ball.subtract_background(...)`. They're lab-specific helpers, not
built-ins, so an unfamiliar module is likely one of these.

- **Which plugins loaded** — `server_status`, section `## Kernel plugins`. It
  lists the files and the packages that actually loaded; the loader is fail-open
  per unit, so a `*.py` sitting in that directory but missing from the report
  failed on load and the session log says why. The section reads
  `(disabled — services.namespace_enabled)` where plugins are switched off. The
  name it reports is the name bound in the namespace.
- **What a plugin offers** — introspection. `inspect_object` on the module prints
  its docstring plus every public callable with its signature and summary line:

```python
[n for n in dir() if not n.startswith("_")]   # everything actually in scope
inspect_object("rolling_ball")                 # the module: docstring + callables
```

A plugin function works inside `da.map_blocks` / `client.submit` like any other —
plugin modules are registered for by-value pickling, so a dask worker does not
need the plugin dir.

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

## You are not the only writer of this namespace
The user can run their own code in this kernel, from the observe web page. It is the
same namespace and the same viewer, so their cells can rebind a variable you set, add
or remove a layer, or import something you did not.

* **You will be told, after the fact.** When user cells have run since your last call,
  a note listing them (`job-N (status)`) is appended to your `execute_code` / `poll_job`
  / `server_status` result. Read them with `poll_job`, and re-check what you rely on
  (`dir()`, `viewer.layers`, `inspect_object`) instead of trusting what you last saw.
* **One job at a time, for both of you.** If the user is running a cell, your
  `execute_code` is rejected as busy — wait and poll, do not try to clear it.
* **Their cell is not yours to stop.** `interrupt_kernel` refuses a user job (it stops
  only your own). Do not reach for `restart_kernel` to get around that: it would
  destroy the user's variables and layers along with yours.

Reading pixels, moving them between the server / a layer / your own variables,
and the round trip for data too large to hold: `guide://data`.
"""

# Appended to GUIDE only when the skills catalog is enabled
# (``services.skills_enabled``, on by default; see _server.get_kernel_guide).
# An install with skills off has no `checklist:` to resolve and no
# `find_skills` to get them from, so this section would describe a tool the
# agent cannot call -- the same reasoning as _SKILLS_INSTRUCTIONS.
SKILL_REQUIREMENTS = """\

## Skill requirements
A curated skill from `find_skills` carries a `checklist:`. **Resolve it before you start the
skill** — one that assumes a plugin or package it doesn't have fails partway through, after
the user has already waited.

`checklist:` informs you; it does not gate the skill. A gap is a fact to tell the user and work
around, not a stop sign: take the body's fallback where it names one, and where it doesn't,
say what you are substituting and why before you spend their time on it. Most tokens have a
workaround — a missing `tensor` plane means data comes from the viewer instead, a missing
package usually has a slower or cruder equivalent in scipy/skimage. What you must not do is
proceed silently: the user cannot judge a result whose method they were never told changed.

One `server_status` call answers every token but a third-party `pkg:`, which is the agent's
responsibility to resolve. Each token names its section: `viewer` → `## Viewer`, `tensor` →
`## Tensor Server`, `dask` → `## Dask`, `ops:<kind>` → `## Ops`, `plugin:<name>` →
`## Kernel plugins` (the only place it can be read — a plugin file contributes its function
names, not its own name, so `dir()` cannot answer it, and a file that failed to load is still
on disk, so a directory listing cannot either), `pkg:biopb-mcp` → `## Versions` (the version
in **this kernel's** interpreter, the one that will run the skill).

A third-party `pkg:<name>` you resolve here, in two steps: `import <name>` answers whether it
is present, and `importlib.metadata.version("<name>")` answers *which version* — never the
module's `__version__` attribute, which is hand-maintained and drifts (`laptrack` ships
`__version__ = "0.17.0"` in its 0.17.1 release, so the attribute fails a `>=0.17.1` token on a
correctly installed package). The token carries a bound at both ends when it names a version,
so an install *newer* than the range is unmet too, and the fix is not another install:
say so and offer the degraded path.

### When something is missing
**Diagnose, tell the user, let them choose.** Installing, seeding and restarting are all
theirs to authorize — but a named gap usually beats abandoning the skill, and several of
these have a fix worth offering.

**`plugin:<name>` — three different causes, in this order:**
1. **Is `pkg:biopb-mcp>=X` in the same `checklist:` also unmet?** Then this install simply
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
"""

DATA = """\
# Where array data lives, and what it actually is

Three places hold pixels here, and they are **not interchangeable**. Treating
them as one is the common source of wrong answers — a plain-napari habit that
breaks because these arrays come off a tensor server, lazily, in a pyramid.

| Source | You get it with | What you get |
|---|---|---|
| **Tensor server** | `client.get_tensor(array_id)` | Lazy dask array, **canonical order** `[..., Z, Y, X]` (`S` last for interleaved colour), full resolution |
| **Viewer layer** | `layer.data` | What napari is *displaying*: a **list** of pyramid levels when multiscale, each a proxy rather than a dask array, at the same order and rank |
| **Kernel** | your own variables | Exactly what you made, carrying no physical scale unless you carried it |

`viewer.add_tensor()` is a *conversion between the first two*, not a window onto
the first. The traps follow from that. The conversion is traceable in one direction:
a layer it loaded records its origin as `layer.metadata['array_id']` — the same id
`client.get_tensor()` takes, so you can always go back to the full-resolution
source-order array (the layer *name* is not reliable for this; it is a display stem
the user may rename). A layer the agent built with `add_image`/`add_labels` has no such
entry.

## The traps

**1. `layer.data` is a list when `layer.multiscale`** — `[full_res, half,
quarter, ...]`, so `layer.data.shape` raises and anything that appears to work
on the list is working on the wrong thing. Branch, take level 0 unless you mean
otherwise, and never pair arrays from two different levels:

```python
arr = layer.data[0] if layer.multiscale else layer.data
```

**2. Layer data is not a dask array.** Each level is wrapped in a `_ViewerArray`
proxy that pins napari's slice reads to a single-process scheduler.
It behaves like a dask array — `.shape`, `.compute()`, slicing, ufuncs — so
ordinary work is unaffected. But `isinstance(arr, da.Array)` is `False`, and a
bare `np.asarray(arr)` materializes the **whole** array in the main process
instead of on the cluster. Slice first, or `.compute()` explicitly.

**3. `layer.scale` is not positional.** A layer carries the source's axes
unchanged, so each size sits on the axis it describes — which means a 3-D
`[C, Y, X]` layer has **no Z**, and `layer.scale[-3]` is its channel axis, not
depth. Read `layer.metadata['dim_labels']` (it matches
`client.get_descriptor(array_id).dim_labels`) rather than counting from the end.
For interleaved colour napari does not count `S`, so `layer.scale` is one shorter
than the array and than `client.get_physical_scale()`: `layer.scale[-1]` is X,
the array's last axis is the colour count.

**4. Lazy means the bill arrives at the end.** `.shape` and `.dtype` are free
while the pixels are not there yet; a scikit-image call, `np.asarray`, or a
`for` loop over the array materializes all of it at once, unchunked and without
progress — which is how a session allocates a volume it cannot hold. Crop first,
keep the chain lazy, `.compute()` once; past `promote_after` that compute is a
job you can watch and cancel (`guide://kernel`).

**5. A layer you built carries no geometry.** `add_labels(arr)` /
`add_image(arr)` store exactly that array: no pyramid, `scale` all ones. A
segmentation added beside a calibrated image measures in pixels until you copy
the image's `scale` onto it.

**6. Upload does not carry labels or physical size.** `client.upload_array`
takes a **dask** array (wrap numpy with `da.from_array`) and stores shape, dtype
and chunks; axis labels and pixel size travel only via `dim_labels=` /
`ome_metadata=`. Otherwise the round trip drops them and `add_tensor` gives back
an uncalibrated layer.

## Reading a layer safely

```python
# What is on the viewer, and in what shape -- worth running before planning any
# work off it.
print([(l.name, type(l).__name__, l.multiscale,
        (l.data[0] if l.multiscale else l.data).shape) for l in viewer.layers])

layer = viewer.layers[NAME]
arr = layer.data[0] if layer.multiscale else layer.data   # trap 1
print(arr.shape, arr.dtype, layer.scale)
sub = arr[0, 100:600, 100:600].compute()                  # traps 2, 5: crop, then compute
```

## The round trip, for data too large to hold

Nothing is materialized until step 3, and step 3 never holds the whole result at
once, so this works on data far larger than memory:

```python
# 1. Off the server: lazy, nothing read yet
arr = client.get_tensor("raw_data_id")

# 2. Build the graph: still lazy, still nothing read
mask_arr = arr > 0.5

# 3. Upload -- the eager step. Computes and sends chunk by chunk.
array_id = client.upload_array(mask_arr, "cache:thresholded_v1")

# 4. Back onto the viewer for the user to check (pyramid-shaped again: trap 1)
layer_name = viewer.add_tensor(array_id)
```

Uploading is also what makes a result *shareable* — an array in the kernel is
visible to nothing else and dies with it. `guide://client` has the upload
arguments, including the ones that carry axis labels and pixel size (trap 6).
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

# Load a tensor as a layer; auto-handles the pyramid. Returns the layer name.
# Addressed by array_id, like client.get_tensor. name defaults from the URL.
layer_name = viewer.add_tensor("source_id")                  # single-tensor source
layer_name = viewer.add_tensor("source_id/t1", name="my_layer")

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

# 2-D <-> 3-D rendering
viewer.dims.ndisplay = 3
```

## Layer types
Image, Labels, Points, Shapes, Vectors, Surface, Tracks
Use `inspect_object("viewer.add_image")` for full signatures.

## Annotation layers (Labels, Points, Shapes)
A layer you build here holds exactly the array you pass — no pyramid, and
`scale` defaults to all ones, so it does **not** inherit the geometry of the
image it was derived from (`guide://data`, trap 6). Copy the source layer's
`scale` across, or every measurement off it comes out in pixels.

```python
# Empty labels layer for the user to paint into, matched to an image layer.
# The branch is on the *image*: that one came off the server and may be a
# pyramid. What you add is plain.
img = viewer.layers["image_name"]
arr = img.data[0] if img.multiscale else img.data
lab = viewer.add_labels(np.zeros(arr.shape[-2:], dtype=np.int32), name="annotations")
lab.scale = img.scale[-2:]                             # else measurements are in pixels

# Labels from a mask you computed
viewer.add_labels((image_data > threshold).astype(np.int32), name="segmentation")

# Read back what the user painted -- a plain array, no branch needed
print(f"Labels present: {np.unique(viewer.layers['annotations'].data)}")
```

The exception is a segmentation that made the round trip — uploaded, then
reloaded with `add_tensor`. That one is a server layer like any other, pyramid
and all, so branch on `multiscale` whenever you did not add the array yourself.

Points and shapes follow the same shape; read the signatures rather than
guessing them:
```python
inspect_object("viewer.add_points")
inspect_object("viewer.add_shapes")
```

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

Arrays from here are lazy dask arrays in the canonical `[..., Z, Y, X]` axis
order the server guarantees, and a tensor loaded into the viewer stops being
lazy — see `guide://data` before moving pixels between the server, a layer,
and your own variables.

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
Arrays are referenced by their `array_id`: `"source_id/t1"` for a tensor within a
multi-tensor source, a bare `"source_id"` for a single-tensor one.
```python
# As a layer -- auto-handles the multiscale pyramid for large images.
layer_name = viewer.add_tensor("source_id")
layer_name = viewer.add_tensor("source_id/t1")

# Or as a lazy dask array, without adding a layer:
arr = client.get_tensor("source_id/t1")
```

## Upload to Server
Use `"cache:my_result"` as destination for ephemeral results that don't
need to be persisted long-term.
```python
array_id = client.upload_array(arr, "cache:my_result")
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
* `image` as a tensor-server **array_id str** -> sent as a lazy reference (the
  op server pulls pixels straight from the tensor server, no kernel
  round-trip) -> the result is uploaded back to the tensor server and a new
  **array_id str** is returned, so ops chain lazily on large data:
```python
labels = ops["cellpose_cyto2"](arr)          # ndarray -> ndarray
seg_id = ops["cellpose_cyto2"]("raw_data_id") # id -> id (lazy, large data)
viewer.add_tensor(seg_id)                    # view the result
```
"""
