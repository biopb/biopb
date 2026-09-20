---
kind: reference
description: Put a result on the server — tensors, label sets and ROI annotations, and what each refuses.
---

# Uploading a result

What you computed lives in this kernel and dies with it. Putting it on the
server is what makes it outlast the session, reachable by the next one, and
visible in the browser — [[web-viewer]] can only show what the server holds.

Three kinds, and the choice matters more than the mechanics:

| what you have | upload it as | shown by |
|---|---|---|
| an image, or any array | a **tensor** — `"cache:<name>"` | `id=<array_id>` |
| a segmentation, mask or instance labelling | a **label set** — `"<image array_id>/labels/<name>"` | `lb=<its array_id>` over the image |
| points, boxes, polygons, scribbles | **ROI annotations** — `put_rois` | `rs=<set_name>` |

**Prefer a label set to a plain tensor for anything derived from an image.** Its
id is the image's own plus `/labels/<name>`, so the two stay registered and one
link shows both; the same labels uploaded as `cache:` are a separate image the
user has to line up by eye.

## Declare, then fill

```python
desc = client.create_tensor("cache:my_result", arr)   # shape, dtype, grid from arr
client.upload_array(desc, arr)                        # writes every chunk, seals it
array_id = desc.array_id
```

`create_tensor` takes anything with `.shape` and `.dtype` as the template; a
dask array also supplies the chunk grid, and `chunk_shape=` overrides it — which
you need to get more than one chunk out of a numpy template. The returned
descriptor is the server's echo and is what every later call takes.

**An upload carries shape, dtype and chunks, and nothing else.** Axis labels and
pixel size travel only if you pass them — `dim_labels=` and `ome_metadata=` on
`create_tensor` — so a result uploaded without them comes back uncalibrated and
measures in pixels. Copy them off the input's descriptor rather than writing
them out; [[data]] trap 6 is this failure from the reading end.

**The destination prefix decides where it lands.** `"cache:<name>"` is
cache-backed and right for something the user is only going to look at;
`"ome_zarr:<name>"` is zarr-backed and persists as files. `"cache:"` with no
name has the server mint one, which is the fix for the rule below.

**A name is taken while its source exists** — pending, finished *or* discarded —
and creating under it again is refused (`FlightServerError`). Only the server's
reclaim sweep frees one, after a discarded upload's `upload_ttl`. **Re-running a
cell therefore needs a new name, or a bare `"cache:"`.**

`upload_array` rechunks onto the declared grid, sends every block and finishes.
It raises `ValueError` when the array does not match the declared shape or
dtype, and `UploadRefused` once the upload is over. Writing the grid yourself is
`upload_chunk(desc, bounds, data)` per chunk plus `finish_upload(desc)` — which
is the only route to READY, the state a consumer waiting on the result polls
for. `get_upload_status(array_id)` reports `state`, `expected_chunks` and
`uploaded_chunks` while it is in flight.

## Label sets

Same two calls, a different name:

```python
desc = client.create_tensor(f"{image_id}/labels/nuclei", labels)
client.upload_array(desc, labels)
```

A set is **unsigned-integer**, spans its image's non-channel axes at full
length, and its all-zero chunks are skipped — so a sparse mask is cheap to send.
Its `array_id` is the request's own rather than a minted `source_id`, and its
descriptor carries an NGFF `image-label` block naming the image it belongs to.

- `client.label_sets(image_array_id)` lists what an image has, sorted.
- `client.get_tensor(set_id)` reads one back like any other tensor.
- `client.delete_labels(set_id)` removes an *uploaded* set and frees the name at
  once. It refuses a set the image's own file carries and a server-owned one
  (a name under `@`) — those are not yours to delete.

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
[[viewer]]) — the upload only made that possible.

Uploading costs a round trip over the data they are about to look at, so on a
session that has a napari window, the window is cheaper for a quick
intermediate. It is still the only route to anything that has to outlive the
session.

## Related

- [[client]] — the read side: browsing the catalog and loading a tensor.
- [[data]] — what an `array_id` addresses, and the axis order an upload expects.
