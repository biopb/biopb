---
description: Put a result on the server — tensors, label sets and ROI annotations, and what each refuses.
---

# Uploading a result

What you computed lives in this kernel and dies with it. Putting it on the
server is what makes it outlast the session, reachable by the next one, and
visible in the browser — [[web-viewer]] can only show what the server holds.

Three kinds, and the choice matters more than the mechanics:

| what you have | upload it as | shown by |
|---|---|---|
| an image, or any array | a **tensor** — `"<scheme>://<source_id>/@fields/<name>"` | `id=<array_id>` |
| a segmentation, mask or instance labelling | a **label set** — `"zarr://<image array_id>/@labels/<name>"` | `lb=<its array_id>` over the image |
| points, boxes, polygons, scribbles | **ROI annotations** — `put_rois` | `rs=<set_name>` |

**Prefer a label set to a plain tensor for anything derived from an image.** Its
id is the image's own plus `/@labels/<name>`, so the two stay registered and one
link shows both; the same labels uploaded as a tensor of your own are a separate
image the user has to line up by eye.

## Declare, then fill

**An upload adds a tensor to a source that already exists** and creates none.
A result of your own belongs to no source the server discovered, so it goes on
the **scratch source** — one per writable server, at the fixed id `scratch`,
always there:

```python
desc = client.add_tensor("zarr://scratch/@fields/my_result", arr)  # shape, dtype, grid from arr
client.upload_array(desc, arr)                        # writes every chunk, seals it
array_id = desc.array_id                              # "scratch/@fields/my_result"
```

Nothing is registered first and nothing comes back to remember: `scratch` is
the id, every time, on every writable server.

**The scratch source is a temp store, and says so.** Every tensor on it gets a
deadline — the server's cap (a day, typically) when you ask for nothing, or
`ttl_seconds=` when you want less. `desc.ttl_seconds` on the answer is the
lifetime you actually got. Past it the tensor is discarded as if you had
discarded it. So scratch is for an intermediate result of a chain, not for the
finished thing a user will come back to next week.

**The `@fields/` segment is not optional.** A bare `"<source_id>/<name>"` is
what a *format* calls its own tensors, so the upload path refuses it — that is
what keeps your result's id from colliding with one of a file's own.

`add_tensor` takes anything with `.shape` and `.dtype` as the template; a dask
array also supplies the chunk grid, and `chunk_shape=` overrides it — which you
need to get more than one chunk out of a numpy template. The grid is a request,
not a promise: the returned descriptor is the server's echo, and the grid on it
is the server's, which may be coarser than the one you asked for.

**The scheme names the store format and nothing else.** Both survive a restart;
they differ in what a read costs. `zarr://` writes an OME-Zarr image group —
compressed, readable by anything that opens zarr, and the default for anything
to keep. `cache://` stores each chunk as the Arrow batch you sent and serves
that batch back with no decode step, on the grid you uploaded it on; it is the
one to reach for when the result is written once and read hot. The answered
`array_id` carries no scheme — the format is a property of the stored tensor,
not of its name.

**Metadata is the source's, not a tensor's.** Axis labels ride on
`add_tensor` (`dim_labels=`), but the pixel size, units and channel names are
the source's and a tensor inherits them — a tensor that carries its own is
refused rather than silently stripped. The scratch source has none to give:
the tensors on it have nothing to do with each other. So for anything whose
scale matters, upload it onto the image's own source, or say so in the ROI or
label set that refers back to the calibrated image. [[napari-viewer]] is this
failure from the reading end, where an uncalibrated layer measures in
pixels.

**A field is taken while its tensor is served** — at any of the three states
below — and adding under it again is refused (`FlightServerError`). Only the
server's reclaim sweep frees one, after a discarded upload's `upload_ttl`.
**Re-running a cell therefore needs a new field name.**

`upload_array` asks the server for the chunk grid, rechunks onto it, sends
every block and seals it. It raises `ValueError` when the array does not match
the declared shape or dtype, and `UploadRefused` once the upload is over.
`upload_array(desc, arr, slice_hint=(slice(0, 4), ...))` uploads that region
only and does **not** seal — say so yourself with `set_upload_status`.

Writing the grid yourself is `upload_chunk(desc, bounds, data)` per chunk, then
`set_upload_status(desc, "READY")`. **The server decides what a chunk is**:
`bounds` must be one whole cell of its grid, and anything else is refused with
the cell it falls in. `get_upload_status(array_id)` reports `state`,
`expected_chunks` and `uploaded_chunks` while it is in flight.

## The three states

An upload is **PENDING** until you move it, and `set_upload_status` is what
moves it. The ladder only climbs:

| state | reads | writes |
|---|---|---|
| `PENDING` | refused — a missing chunk is one still in flight | accepted |
| `READY` | served; **a chunk never uploaded reads as zeros** | refused |
| `DISCARDED` | refused | refused |

**`"READY"` publishes and seals in one move.** Once a consumer can read it, it
takes no more chunks — so a partial upload is published by declaring it done,
not by leaving it open. A chunk you never sent reads back as zeros, which is
how a sparse result (one labelled frame of a thousand) costs one frame.

The reason the two are one moment: **a chunk already read is not read again.**
A `chunk_id` names fixed bytes everywhere else in this system, so both the
server and the client cache on it. A region a consumer saw as background would
stay background for that consumer even after a later write filled it, and
nothing could tell them otherwise. Sealing at publish removes the case.

**`"DISCARDED"` is the only delete there is**, and it works from every state —
including READY. It takes the server-minted store with it (a `zarr://` member's
directory, a label set's sidecar) and takes the tensor out of its source's
listing; the source itself stays, ready for the next one. Pass a `reason`: it is
what a poller waiting on the result reads back. The field frees after the
server's reclaim sweep, not immediately, so a re-run under the same name has to
wait for it either way.

## The round trip, for data too large to hold

Nothing is materialized until the upload, and the upload never holds the whole
result at once, so this works on data far larger than memory:

```python
arr = client.get_tensor("raw_data_id")   # lazy, nothing read yet
mask = arr > 0.5                         # still lazy, still nothing read

desc = client.add_tensor("zarr://scratch/@fields/thresholded_v1", mask)
client.upload_array(desc, mask)          # the eager step: chunk by chunk
```

## Label sets

Same two calls, a different name:

```python
desc = client.add_tensor(f"zarr://{image_id}/@labels/nuclei", labels)
client.upload_array(desc, labels)
```

A label set names the image it belongs to rather than a source, so it lands
beside an image the server **discovered** just as readily as beside one on
scratch — and it inherits that image's axes and scale.

A set is **unsigned-integer**, spans its image's non-channel axes at full
length, and its all-zero chunks are skipped by `upload_array` — so a sparse mask
is cheap to send.

Its `array_id` is the request's own minus the scheme, and its descriptor
carries an NGFF `image-label` block naming the image it belongs to.

- `client.label_sets(image_array_id)` lists what an image has, sorted.
- `client.get_tensor(set_id)` reads one back like any other tensor.
- `client.set_upload_status(set_id, "DISCARDED", "replaced")` removes an
  *uploaded* set and its sidecar. It leaves alone a set the image's own file
  carries and a server-owned one (a name under `@`) — those are not yours to
  delete, and it answers `UNKNOWN` rather than raising for anything it does not
  reach.

## ROI annotations

Vectors, not pixels, and stored per tensor rather than as a tensor:

```python
from biopb.image import ROI, Point, RoiAnnotation

client.put_rois(image_id, [
    RoiAnnotation(array_id=image_id, set_name="counts", label="focus",
                  roi=ROI(point=Point(x=512, y=384)))
])
```

- **Level-0 pixel coordinates.** A shape measured on a downsampled level has to
  be scaled up first; nothing does it for you.
- **The 2-D vector arms only**: point, rectangle, ellipse, polygon, polyline
  (a scribble's `width` is geometry and widens its bounding box). A mask or mesh
  is refused on purpose — instance segmentation belongs in a label set.
- **`set_name` is the layer**, and it is what `rs=` matches in a viewer link.
- An empty `roi_id` creates (the server mints one); naming an existing one
  updates. The batch applies in a single transaction; `check_rev=True` turns it
  conditional and returns mismatches in `conflicts` instead of applying them.

Read them back with `list_rois(array_id, set_name="")` — no plane or bbox
filter, since a client hit-tests the resident set — and drop them with
`delete_rois(array_id, roi_ids=(), set_name="")`, which without ids deletes
every annotation on the tensor, narrowed by `set_name` when given.

ROIs are private data gated by their tensor's source, so they are **not** on the
SQL browse surface; `list_rois` is the only way to read them.

## Then show it

An upload is not a display. Once it lands, put the link in front of the user
([[web-viewer]]) or the tensor on the window (`viewer.add_tensor(array_id)`,
[[napari-viewer]]) — the upload only made that possible.

Uploading costs a round trip over the data they are about to look at, so on a
session that has a napari window, the window is cheaper for a quick
intermediate. It is still the only route to anything that has to outlive the
session.

## Related

- [[tensor-server-client]] — the read side: browsing the catalog and loading a tensor.
- [[tensor-server-client]] — what an `array_id` addresses, and the axis order an
  upload expects.
