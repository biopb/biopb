---
description: The napari window, where the session has one — reading a layer, layers, camera, dims, annotation layers.
---

# The napari window

**This is the napari window, and a session need not have one.** It is the
display surface that can show an array without uploading it, and the one
`take_screenshot` captures — but it needs a display, and a headless session or a
closed window has none. `server_status`'s `## Viewer` says which you are
in; [[web-viewer]] is the route that works either way. Everything below assumes
a window.

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
pyramid (`layer.data` is napari's `MultiScaleData` sequence of levels), in
display axis order, wrapped so napari's slice reads stay in-process. Read the
pixels with **`viewer.tensor(layer)`** — a plain full-resolution dask array from
any layer — rather than indexing `.data`, where `data[0]` means level 0 on a
multiscale layer and plane 0 on a single-scale one. The next section has the
whole story.

## Reading a layer's pixels

**`viewer.tensor(layer)`.** A plain, lazy `da.Array` at the layer's full
resolution, in canonical `[..., Z, Y, X]` order, on either kind of layer — the
array already in hand, unwrapped, with no server round trip, so it works with
the server disconnected. For a *fresh* read instead, `client.get_tensor(id)`
([[tensor-server-client]]).

```python
arr = viewer.tensor(layer)        # or viewer.tensor("layer name")
```

**`layer.data` is the renderer's packaging, not your array**, and reading pixels
off it is the most expensive mistake here because it does not raise. On a
multiscale layer it is napari's `MultiScaleData`, a sequence of levels: `data[0]`
is level 0, while on a single-scale layer the same expression is *plane* 0 — a
different rank, silently. Worse, `np.asarray(layer.data)` and anything numpy
touches return the **lowest** level, so a filter runs at pyramid bottom with no
error and no warning. Each level is also a `_ViewerArray` proxy rather than a
real dask array, so `isinstance(arr, da.Array)` is False and library code that
type-checks its input breaks.

`.shape`, `.ndim` and `.dtype` are safe on both branches and report level 0, so
those never need a branch. Everything else does, and `viewer.tensor` is that
branch written once. (`.unwrap()` gets a real dask array out of a proxy you
already hold.)

**Going back to the server** is a different question: `layer.metadata['array_id']`
is the id `client.get_tensor` takes, for when you want a fresh read of a source
re-indexed since the layer loaded. The layer *name* is not a reliable origin —
it is a display stem the user may rename — and a layer built with
`add_image`/`add_labels` has no `array_id` entry at all.

Worth running before planning any work off a layer:

```python
print([(l.name, type(l).__name__, l.multiscale, l.data.shape)
       for l in viewer.layers])

layer = viewer.layers[NAME]
arr = viewer.tensor(layer)
print(arr.shape, arr.dtype, layer.scale)
sub = arr[0, 100:600, 100:600].compute()     # crop, then compute
```

## Geometry a layer does and does not carry

**`layer.scale` is not positional.** A layer carries the source's axes unchanged,
so each size sits on the axis it describes: a 3-D `[C, Y, X]` layer has **no Z**,
and `layer.scale[-3]` is its channel axis, not depth. Read
`layer.metadata['dim_labels']` rather than counting from the end. For interleaved
colour napari does not count `S`, so `layer.scale` is one shorter than the array
— `layer.scale[-1]` is X, and the array's last axis is the colour count.

**A layer you built carries none of it.** `add_image(arr)` / `add_labels(arr)`
store exactly that array: no pyramid, `scale` all ones. A segmentation added
beside a calibrated image measures in pixels until you copy the image's `scale`
onto it.

## Layers
```python
# List all layers (read). .shape is safe on either branch; for pixels use
# viewer.tensor (above), never .data.
for layer in viewer.layers:
    print(f"{layer.name}: {type(layer).__name__} "
          f"{layer.data.shape} {layer.data.dtype}")

# The pixels, as a plain lazy dask array -- no multiscale branch, no proxy
arr = viewer.tensor("image_name")

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
image it was derived from. Copy the source layer's
`scale` across, or every measurement off it comes out in pixels.

```python
# Empty labels layer for the user to paint into, matched to an image layer.
# .shape reports full resolution whether or not the image is a pyramid, so the
# new labels array matches it. What you add is plain either way.
img = viewer.layers["image_name"]
mask = np.zeros(img.data.shape[-2:], dtype=np.int32)
lab = viewer.add_labels(mask, name="annotations")
lab.scale = img.scale[-2:]                             # else measurements are in pixels

# Labels from a mask you computed
viewer.add_labels((image_data > threshold).astype(np.int32), name="segmentation")

# Read back what the user painted -- a plain array you added, so .data is it
print(f"Labels present: {np.unique(viewer.layers['annotations'].data)}")
```

The exception is a segmentation that made the round trip — uploaded, then
reloaded with `add_tensor`. That one is a server layer like any other, pyramid
and all, so read it with `viewer.tensor()` rather than `.data` whenever you did
not add the array yourself.

**A mask can live on the server.** A *label set* is a tensor of its image, named
`<image array_id>/@labels/<name>`, so a segmentation is not necessarily something
a client made and holds:

```python
client.label_sets("src0")            # -> ['src0/@labels/@ome', 'src0/@labels/nuclei']
viewer.add_tensor("src0/@labels/nuclei")   # a Labels layer, not an Image one
```

`add_tensor` reads the name and builds the right kind of layer, with the image's
pyramid, scale and axes — so unlike a layer you built, it measures in physical
units already. It is added on its own, never alongside its image; add both when
you want both. A `@`-prefixed name is the server's own (`@ome` is rasterized
from an OME-TIFF's masks); it is read-only, and so is every set — replacing one
means uploading a new name.

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
