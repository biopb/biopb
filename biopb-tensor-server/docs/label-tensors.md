# Label tensors

**Experimental.** The label-tensor model may still change without notice.

Scope: `biopb-tensor-server`, the Python SDK, and the two viewers only as far
as naming what they consume. Companion to [upload-model.md](upload-model.md), which covers
`label_sets` / `label_uploads` / `attached_fields` as the shared attachment
mechanism; this is the deep dive on label sets specifically.

## Goal

Serve pixel-wise labels for an image -- a segmentation, a mask, a class map --
as tensors alongside the image.

## Model

A label set is one integer tensor whose `array_id` is

```
<image array_id>/@labels/<name>
```

`src_ab12/@labels/nuclei` on a single-tensor source, `src_ab12/Image:0/@labels/nuclei`
or `src_ab12/A/1/@labels/nuclei` on a multi-tensor one. It mirrors the NGFF
layout (`labels/<name>` under the image group) with the segment **marked**.

**The marker is on the id, not on disk.** The group inside an OME-Zarr store is
`labels/` and stays that way. What the marker buys is that an attached tensor's id
can never collide with a native one. See `core/labels.py`, `core/attached.py`.

**Reserved names.** A name starting with `@` is server-owned, the same as
`@ome`: the set rasterized from an OME-TIFF's masks. Clients can read a
reserved set and never create or delete one.

**Dtype.** Unsigned integer; `0` is background. The upload refuses anything
else. `uint32` is the recommendation, `uint16` is fine for small counts.

### Extent

A set's axes are the image's canonical axes with the channel axis dropped,
each at the image's full length. A label pixel and its image pixel share an
index -- the same contract the ROI store runs on ("level-0 pixels, the server
never rescales geometry") -- which is what lets a viewer overlay a set with no
transform.

The server states the mapping the rule implies: `biopb.labels.image_axes`, the
image axis each axis of the set indexes, alongside the NGFF `image-label`
block. `/api/tile_info` surfaces the same list as `image_axes` for a label set.

The rule is checked twice: the upload refuses a set that would not span its
image at create, and `SourceAdapter.label_sets` checks every set again where
the origins meet (`extent_mismatch`, on normalized descriptors). Mismatched labels
are dropped with a warning.

## Three different origins

| origin | backing | writable | content_version |
|---|---|---|---|
| NGFF `labels/<name>` in an OME-Zarr source | the array in place, own pyramid if it ships one | no | the parent's |
| OME-TIFF `<Mask>` shapes | rasterized on read from the metadata dict | no | the parent's |
| uploaded | a zarr array the server minted under `write_dir` | write-once | its own, minted at create |

The first two derive from the file, so they share its stat-signature
`content_version` and go stale with it. An uploaded set is its own content: it
mints a random token at create, persists it in the sidecar's attrs, so a name
reused after a delete won't hit a stale cache entry.

**Never inside the source.** An uploaded set is not written into the user's
OME-Zarr. The sidecar lives at

```
<write_dir>/labels/<parent source_id>/<name>.zarr
```

one NGFF label group per set (`.zattrs` with `multiscales` + `image-label`, a
`0/` array), the `.zattrs` also carrying a `biopb` block: the upload marker
and `labels: {image_field, content_version}`. `write_dir` must not be a
discovery root, otherwise the label is listed twice under two ids.

### Attachment to the parent

Sets are tensors *of the parent source*, not sources. `SourceAdapter` owns the
concept via a hook: `get_embedded_labels()` that an adapter class overrides
(`OmeZarrAdapter` reads its NGFF `labels/` group there); `attach_label_set`
/ `detach_label_set` are what the registry's `on_register` hook (`sidecar_attacher`)
and the upload kind (at READY; discard) use.

`label_uploads` is the second, smaller index: sets the upload path is still
filling, and the tombstones of ones it gave up on. Routable but never listed,
and what the DoPut boundary looks an upload up in and the reclaim sweep walks.

`resolve_tensor(tensor_id)` and `resolve_chunk_adapter(field)` are the two
lookups the serve path uses (`get_flight_info`, `do_get`, the precache): a
`.../@labels/<name>[/<level>]` field answers from `label_sets`, everything
else delegates to the format. `catalog_tensors` appends the sets after
`list_tensor_descriptors`, so a source's first tensor -- what every listing
reads as its picture -- is never a set.

A set's adapter is `LabelSetAdapter` (`adapters/labels.py`): `OmeZarrAdapter`
opened on the label group, bound under the parent's `source_id` with the set's
field as its tensor name, so its chunk ids and native levels ride under
`<image>/@labels/<name>`.

## Reads

Every set reads through the ordinary path and the ordinary chunk cache. The
chunk id is a pure function of `array_id`, bounds, scale and method under the
content_version wrapper, so a set has its own cache namespace for free, and
the cache is what makes rasterizing or downsampling a large set affordable.

**Nearest, per tensor.** Averaging label ids produces ids that exist nowhere,
so a set's adapter forces `nearest` on every computed level
(`_advertised_pyramid`), and the precache and clients follow the same ladder.
A native NGFF set that ships its own multiscales serves them as `precompute`.

**Rasterizing OME masks.** A `<Mask>` is `x, y, width, height`, optional
`TheZ/TheT/TheC` pins, and a `BinData` bitmap. The `@ome` set for an image is
one tensor over the image's extent (`adapters/ome_masks.py`); the label value
of a mask is the 1-based index of its ROI in the image's `roi_refs` order, so
overlaps paint later-wins for free. A chunk read decodes the masks whose
bounding box and pin intersect the chunk and paints them; decoded bitmaps are
memoized per adapter, and the ordinary chunk cache holds the painted output.
`TheC` is inert here (channel distinction is never carried), as is a pin
naming an axis the image does not have.

OME-TIFF only, via `OmeTiffAdapter.get_embedded_labels`; a bioio-backed format
carrying OME-XML does not rasterize its masks. The fast metadata path
(`_fast_ome_metadata`) base64-encodes a `Mask`'s `bin_data.value`
(`_b64_encode_mask_bindata`) before dumping the metadata dict, since a raw
non-UTF-8 bitmap otherwise crashes the dump and costs the file its entire OME
metadata; `strip_mask_bindata` then drops `bin_data.value` unconditionally in
`sync_source_added`, so a real bitmap never reaches the SQL-queryable
`metadata_json`.

**Precache.** A set is on the ladder like any tensor -- coarse levels of a
mostly empty set are small and nearest over zeros is trivial, so the precache
is not special-cased.

## Upload

The upload SDK is reused as it stands, with no verb of its own:

```python
desc = client.create_tensor("src_ab12/@labels/nuclei", labels, chunk_shape=...)
client.upload_array(desc, labels)        # skips all-zero chunks for this kind
client.set_upload_status("src_ab12/@labels/nuclei", "DISCARDED", "replaced")
```

Server side this is a third upload kind, selected by the request `array_id`
carrying no `cache:` / `ome_zarr:` prefix and a `/@labels/` segment. Unlike the
other two, the request's `array_id` *is* the final one. The kind:

- resolves the parent by splitting at the last `/@labels/` and refuses if the
  parent is absent, unresolved, or does not serve pixels;
- refuses a non-unsigned-integer dtype, a reserved name, a name that would not
  stay inside the sidecar directory (`unsafe_store_name`), or a shape /
  `dim_labels` that is not the parent's canonical non-channel extent -- a
  request naming no `dim_labels` is filled in from the image rather than
  refused, since the extent rule leaves exactly one legal answer;
- refuses a name already attached, finished or pending (biopb/biopb#1054,
  per parent);
- creates the sidecar array with the pending marker and the minted
  content_version, and registers the set as **pending** on the parent's
  attachment: routable for `get_flight_info` (so the status poll works from
  create) but not listed, and not readable.

Reaching **READY** clears the pending marker, lists the set
(`attach_label_set`) and re-syncs the parent's catalog row, in that order --
the catalog must not name a set a restart would sweep away.

**Skipping zeros is per kind.** `upload_array` may drop all-zero chunks only
for the label kind, where an unwritten chunk reads as fill; a `cache:` source
answers a read of an unwritten chunk with "holds no chunk", so the skip is not
general SDK behaviour.

**Replacement.** A set is replaced by uploading under a new name, or by
deleting the old one first. Delete is the one action beyond create/write.

## Lifecycle

Durable uploads (`ome_zarr:` and label sidecars) get the full lifecycle:
discard removes the store under the progress lock (mark DISCARDED, delete the
array, drop the attachment or catalog row), and `reap` covers them the same as
any other upload -- a quiet pending set is discarded after `upload_ttl`, its
tombstone reclaimed after another. Upload state is persisted in the store's
attrs, so a crash-pending sidecar is deleted at boot rather than served. A
set's lifetime otherwise follows its parent's *identity*, not its liveness:
`source_id` is a hash of the resolved URL, so a moved image no longer
resolves any sidecar filed under its old id -- exactly how it loses its
ROIs. Unlike a ROI row, though, a **finished** sidecar has no catalog row of
its own, so there is nothing for the ROI orphan clock (`unseen_rois` /
`prune_unseen`) to act on: like an orphaned field, it is kept on disk
indefinitely, and a source that returns to the same URL re-attaches it. Only
a **crash-pending** sidecar is removed, at the next boot's sweep -- the same
rule every unfinished store gets.

## Discovery

The catalog is the only browse surface, and the identity policy requires it
and `get_flight_info` to carry identical `array_id`s. A set is an ordinary
entry in `sources.tensors`, found by its path:

```sql
SELECT t.array_id FROM sources, UNNEST(tensors) AS u(t)
WHERE t.array_id LIKE 'src_ab12/@labels/%'
```

The typed signal rides in the descriptor `get_flight_info` returns: an NGFF
`image-label` block in `metadata_json`, with `source.image` set to the
parent's `array_id` (the sidecar is not adjacent to the image, so NGFF's
relative path would be meaningless). No proto change.

**Authorization.** A set is an attached tensor, so it carries its own read
capability or none at all -- a grant on its image does not reach it. Create,
write and every state transition are full access like every other mutation.

## Clients

- **SDK**: `create_tensor` / `upload_array` / `set_upload_status` as above; a
  `label_sets(image_array_id)` convenience over the catalog query.

### The SPA

The web viewer needs no dedicated route and no new wire field. A set is an
ordinary tensor, listed by `/api/sources` and read through
`/api/tile_info` + `/api/tile` like any other; the browser adds three things.

**The path is what says a tensor is a set.** There is no `role` column, and a
listing's per-tensor entry carries no `metadata_json` -- so
`splitLabelArrayId` (`@biopb/tensor-flight-client`, a mirror of
`core/labels.py::split_label_field`) is the one place the rule lives. The tree
groups on it (`groupTensors`), and a set's row sits under the image its id
names.

**The overlay is a second Viv image layer, not a second data path.** The same
`createTensorPixelSources` the image uses, a `MultiscaleImageLayer` drawn
*under* the annotation layers so line work is not covered by a categorical
fill. Nearest sampling comes from both ends: the server forces it on every
computed level of a set, and Viv's `interpolation: "nearest"` keeps the GPU
from blending two ids into a third.

**Colour is a pure function of the id.** `LabelPaletteExtension` claims
`DECKGL_MUTATE_COLOR` and rotates hue by the golden-ratio conjugate, so
consecutive ids land far apart rather than running a gradient; `0` is
background and fully transparent. `contrastLimits = [0, 1]` is left at
identity, which is what delivers the stored id to the palette.

A set spans the image's non-channel extent, so `labelSelection` (in
`@biopb/tensor-flight-client`) reads the server's `image_axes` rather than
re-deriving it, falling back to the extent rule only against a server that
does not state it.

Playback paces on both layers landing: paced on the image alone, a set whose
read is slower (no native pyramid, every coarse tile a full-resolution read
reduced on the way out) falls out of step for the rest of playback. The
overlay is held transparent until its own `onViewportLoad` confirms it holds
the plane the image has landed on, and fails open on a set that errored or
none at all (`PLAY_STALL_MS` is the backstop). A set that 404s costs the
overlay a badge, never the image.

A link carries the overlay as `lb=<the set's array_id>` and its alpha as `lo`
-- the whole address, not the bare name, so a link naming one image's set and
another image's `id` draws nothing.

### MCP and napari

Same shape as the SPA: no new route, no new call. The routing sits in the one
shared pipeline (`_tensor_utils.add_tensor_layer`), so the Tensor Browser and
the MCP `add_tensor` cannot disagree about what a set is, and nothing is added
implicitly -- a set becomes a layer when it is asked for, never alongside its
image.

**The path is what says a tensor is a set**, here as in the browser:
`biopb.tensor._labels` is the Python mirror of `core/labels.py` (the SDK's own
copy, since biopb-tensor-server is not an installable dependency of a
client). The Tensor Browser's `_group_tensors` files each set under its
image.

**The pyramid is the server's**, exactly as for an image -- safe for ids
because a set's computed levels are advertised `nearest`, and napari only
*picks* a level, never downsamples the data itself. A multiscale `Labels`
layer is `editable = False`, matching a write-once set.

**The set is given the image's rank before it is added.** napari aligns
layers of differing rank from the right, so a `T Z Y X` set added beside a
`T C Z Y X` image would otherwise land its `T` on the image's `C`. The channel
axes the set does not have are inserted from `image_axes` and broadcast to the
image's length rather than left singleton -- a singleton axis would put the
layer outside its own extent at every channel but the first, blanking as the
channel slider moves; a broadcast axis is a view onto the one underlying chunk,
so the mask shows on every channel for a single read. Alignment inserts and
never permutes; a mapping that would need a transpose is refused rather than
mislaid.
