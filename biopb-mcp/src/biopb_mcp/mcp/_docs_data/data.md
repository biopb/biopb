---
kind: reference
description: How array data is represented here — pyramids, laziness, axis order, rank — and the traps.
---

# Where array data lives, and what it actually is

Three places hold pixels here, and they are **not interchangeable**. Treating
them as one is the common source of wrong answers — a plain-napari habit that
breaks because these arrays come off a tensor server, lazily, in a pyramid.

The middle row exists only where the session has a napari window ([[viewer]]).
Without one, pixels come from the server and from your own variables, and
showing the user a result means putting it *back* on the server ([[web-viewer]]).

| Source | You get it with | What you get |
|---|---|---|
| **Tensor server** | `client.get_tensor(array_id)` | Lazy dask array, **canonical order** `[..., Z, Y, X]` (`S` last for interleaved colour), full resolution |
| **Viewer layer** | `layer.data` | What napari is *displaying*: napari's **`MultiScaleData`** sequence of pyramid levels when multiscale, each level a proxy rather than a dask array, at the same order and rank |
| **Kernel** | your own variables | Exactly what you made, carrying no physical scale unless you carried it |

`viewer.add_tensor()` is a *conversion between the first two*, not a window onto
the first. The traps follow from that. **`viewer.tensor(layer)` undoes the
packaging** (trap 1).

Going back to the *server* is a different question: `layer.metadata['array_id']`
is the same id `client.get_tensor()` takes, for when you want a **fresh** read —
a source re-indexed since the layer loaded. (The layer *name* is not a reliable
origin; it is a display stem the user may rename. A layer the agent built with
`add_image`/`add_labels` has no `array_id` entry at all.)

## The traps

**1. Read a layer's pixels with `viewer.tensor(layer)`, not `layer.data`.**

```python
arr = viewer.tensor(layer)   # or viewer.tensor("layer name")
```

A plain, lazy `da.Array` at the layer's full resolution — the shape
`layer.data.shape` reports — in canonical `[..., Z, Y, X]` order, on either kind
of layer. It is the array already in hand, unwrapped: level 0 of the pyramid for
a layer `add_tensor` loaded, and exactly what you passed for one you built with
`add_image`/`add_labels`. No server round trip, so it works with the server
disconnected. For a *fresh* read instead, `client.get_tensor(array_id)`.

`layer.data` is what napari is *displaying*, and it is packaged for the
renderer, not for you:

```python
layer.data            # multiscale: napari's MultiScaleData, a sequence of levels
layer.data[0]         # multiscale: level 0, full rank.  single-scale: plane 0.
layer.data[0, 0:64]   # multiscale: TypeError (list indices ... not tuple)
layer.data.compute()  # multiscale: AttributeError -- MultiScaleData has none
np.asarray(layer.data)  # multiscale: the *lowest* level. No error. See trap 4.
```

`.shape`, `.ndim` and `.dtype` do work on both branches and report level 0, so
those never need a branch — but `data[0]` returning a different *rank* on each
branch, silently, is why reading pixels off `.data` needs one. `viewer.tensor`
is that branch, written once.

**2. Layer data is not a dask array.** Each level is wrapped in a `_ViewerArray`
proxy that pins napari's slice reads to a single-process scheduler.
It behaves like a dask array — `.shape`, `.compute()`, slicing, ufuncs — so
ordinary work is unaffected. But `isinstance(arr, da.Array)` is `False`, which
breaks library code that type-checks its input, and a bare `np.asarray(arr)`
materializes the **whole** array in the main process instead of on the cluster.
Slice first, or `.compute()` explicitly. `viewer.tensor()` returns a real dask
array; `.unwrap()` gets one out of a proxy you already hold.

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
`for` loop over *an array* materializes all of it at once, unchunked and without
progress — which is how a session allocates a volume it cannot hold. Crop first,
keep the chain lazy, `.compute()` once; past `promote_after` that compute is a
job you can watch and cancel ([[kernel]]).

Hand those same calls a **multiscale `layer.data`** and they fail the other way,
silently: `MultiScaleData.__array__` returns the *lowest* level, so
`np.asarray(layer.data)` and `gaussian(layer.data)` give a result from the
bottom of the pyramid — no error, no warning, wrong resolution — and
`for x in layer.data` iterates levels rather than planes. Resolve the layer to
one array (trap 1) before anything numpy touches it.

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
# work off it. `.shape` reports full resolution on either branch, so no branch.
print([(l.name, type(l).__name__, l.multiscale, l.data.shape)
       for l in viewer.layers])

layer = viewer.layers[NAME]
arr = viewer.tensor(layer)                                # trap 1: real dask array
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

# 3. Declare, then upload -- the eager step. Computes and sends chunk by chunk.
desc = client.create_tensor("cache:thresholded_v1", mask_arr)
client.upload_array(desc, mask_arr)
array_id = desc.array_id

# 4. Back onto the viewer for the user to check. It arrives pyramid-shaped, so
#    read it back with viewer.tensor(), not .data (trap 1).
layer_name = viewer.add_tensor(array_id)
check = viewer.tensor(layer_name)   # the layer's pixels, as a plain dask array
```

Uploading is also what makes a result *shareable* — an array in the kernel is
visible to nothing else and dies with it. [[upload]] has the rest — the other
two kinds, the naming rules, and the arguments that carry axis labels and pixel
size (trap 6).
