# Label tensors — backend design

Status: steps 1–4 implemented; 5 (clients) remains. The
sequence is biopb/biopb#1059. Companion to `roi-annotations.md`, which
scoped instance segmentation *out* of the annotation store and into "a label
tensor the server already serves as pixels". This is that tensor.

Scope: `biopb-tensor-server`, the Python SDK, and the two viewers only as far as
naming what they consume.

## Goal

Serve pixel-wise labels for an image — a segmentation, a mask, a class map — as
tensors alongside the image, from three origins:

- **file-embedded**: an OME-TIFF's `<Mask>` ROI shapes, today counted and
  dropped by the ROI import (`_ome_rois.py`, `dropped_masks`); an OME-Zarr's
  NGFF `labels/` group, today not enumerated at all;
- **uploaded**: a client sends a whole label set over the existing upload path;
- and **several of each per image**: the file's own set and a user's alternative
  are both readable, side by side, under different names.

## Non-goals

**No per-instance edits.** A label set is written once, whole, and replaced
whole. There is no chunk-level mutation of a finished set, no undo, and no
merge. This is what keeps every set an ordinary write-once tensor and keeps the
one-name-one-adapter rule from biopb/biopb#1048 without exceptions.

**No sub-extent labels.** A set always spans the image (see Extent). Positioning
a smaller array would need an origin the descriptor does not carry, a proto
change, and a transform in every viewer; the fill value already makes a sparse
set cost nothing, so the shape rule is enforced instead.

**No volatile flavour.** One durable kind. A label set is small next to its
image, and one lifecycle is simpler than two.

**No label semantics.** The server carries integer ids and, where the origin
has them, NGFF `image-label` colours and properties. What an id *means* is the
producer's.

## Model

A label set is one integer tensor whose `array_id` is

```
<image array_id>/labels/<name>
```

`src_ab12/labels/nuclei` on a single-tensor source, `src_ab12/Image:0/labels/nuclei`
or `src_ab12/A/1/labels/nuclei` on a multi-tensor one. This is the NGFF layout
(`labels/<name>` under the image group), so an on-disk OME-Zarr set, an
uploaded set and a rasterized OME-TIFF set all present identically, and the
tensor identity policy already allows it (a field may contain `/`; HCS uses
`well/field`). The parent image is recovered by splitting at the **last**
`/labels/`; `name` is therefore non-empty and slash-free.

**Reserved names.** A name starting with `@` is server-owned, in the same
spirit as the `@ome` ROI set: `labels/@ome` is the set rasterized from an
OME-TIFF's masks, and a native NGFF set keeps its on-disk name. Clients can read
a reserved set and never create or delete one.

**Dtype.** Unsigned integer; `0` is background. The upload refuses anything
else. `uint32` is the recommendation, `uint16` is fine for small counts.

### Extent

A set's axes are the image's canonical axes with the channel axis dropped, each
at the image's full length. A label pixel and its image pixel share an index,
which is the contract the ROI store already runs on ("level-0 pixels, the
server never rescales geometry") and what lets a viewer overlay a set with no
transform.

Sparse coverage — one labelled frame of a thousand — is a storage question, not
a shape one, and every layer already answers it with zeros:

- the sidecar zarr is created with fill value `0` and never materializes an
  unwritten chunk, so a frame-0-only set on a 1000-frame timelapse costs one
  frame on disk;
- `finish` seals whatever landed (it logs uploaded-of-expected; it does not
  require the grid), so the client sends only the chunks it has;
- an unwritten zarr chunk reads back as zeros at no I/O; the rasterized OME set
  yields a zero block for any chunk no mask touches.

The one relaxation this leaves open — a set carrying a *subset* of the image's
axes, absent meaning broadcast — is additive but not in the first version: a
plain 2-D array on a timelapse is ambiguous between "frame 0" and "every
frame", and the fill rule makes the explicit full-extent upload cheap enough
that the ambiguity is not worth resolving yet.

A mask pinned to one channel (OME `TheC`) rasterizes into the shared set; the
channel distinction is not carried.

The rule is checked twice: the upload refuses a set that would not span its
image at create, and `SourceAdapter.label_sets` checks every set again where
the origins meet (`extent_mismatch`, on normalized descriptors), so a native
set of another shape, a sidecar that reached the directory by other means, or
a store whose image changed shape under it is dropped with a warning rather
than served misaligned.

## Three origins, one tensor shape

| origin | backing | writable | content_version |
|---|---|---|---|
| NGFF `labels/<name>` in an OME-Zarr source | the array in place, own pyramid if it ships one | no | the parent's |
| OME-TIFF `<Mask>` shapes | rasterized on read from the metadata dict | no | the parent's |
| uploaded | a zarr array the server minted under `write_dir` | write-once | its own, minted at create |

The first two derive from the file, so they share its stat-signature
`content_version` and go stale with it. An uploaded set is its own content: it
mints a random token at create, persists it in the sidecar's attrs, and its
adapter wraps chunk ids with it — the gen-token pattern `cache:` uploads use —
so a name reused after a delete can never hit a stale cache entry.

**The name is a path, so it is checked as one.** A store's directory is named
after what the client asked for, and discard removes that directory whole, so
the name must stay inside the directory the server chose: `ome_zarr:../../x`
otherwise mints and later deletes a store two levels above `write_dir`. Both
separators and `:` are refused on every platform, not just the host's, and the
name is refused rather than sanitized — it is an identity as well as a path
(it is what `create_tensor` refuses a collision on), so folding two requests
onto one store would trade a traversal for a mix-up. `cache:` needs none of
this: it hashes the name into a `source_id` and never reaches a filesystem.

**Never inside the source.** An uploaded set is not written into the user's
OME-Zarr even when it could be: discovery rescans the tree, and the
reconciler's stat-based `content_version` would flip and invalidate the
*image's* cache. The sidecar lives at

```
<write_dir>/labels/<parent source_id>/<name>.zarr
```

one NGFF label group per set (`.zattrs` with `multiscales` + `image-label`, a
`0/` array), the `.zattrs` also carrying a `biopb` block: the upload marker and
`labels: {image_field, content_version}`. `write_dir`
must not be a discovery root: `ZarrAdapter.claim` takes any `.zarr` with a
`.zattrs`, and a sidecar claimed as a source of its own would be listed twice
under two ids. The `ome_zarr:` kind has the same exposure today; step 1 below
makes it a documented configuration rule and a startup check.

### Attachment to the parent

Sets are tensors *of the parent source*, not sources. The registry keeps one
entry per source, so the parent adapter has to answer for them — and it does so
through the base class, not a wrapper (the codebase already carries one
forwarding wrapper at that seam, `NormalizingAdapter`, and a second layer of
`__getattr__` forwarding is the objection). `SourceAdapter` owns the concept;
the format's own `list_tensor_descriptors` / `get_tensor_adapter` stay
ignorant of sidecars:

- `get_embedded_labels()` is the hook a format overrides, beside
  `get_embedded_rois` (`OmeZarrAdapter` reads its NGFF `labels/` group there);
  `attach_label_set` / `detach_label_set` are what the registry's
  `on_register` hook (finished sidecars, `sidecar_attacher`) and the upload
  kind (at `finish`; `delete`) use; `label_sets` is the merged view, every
  set normalized like any tensor and checked against the image it binds to
  (`label_binding_error`, which the upload's create calls too, so one rule
  answers for every origin).
- `label_uploads` is the second, smaller index: sets the upload path is still
  filling, and the tombstones of ones it gave up on. Routable but never
  listed — the bytes have not all arrived, or are gone — and it is what the
  DoPut boundary looks an upload up in and what the reclaim sweep walks.
- `resolve_tensor(tensor_id)` and `resolve_chunk_adapter(field)` are the two
  lookups the serve path uses (`get_flight_info`, `do_get`, the precache): a
  `.../labels/<name>[/<level>]` field answers from `label_sets`, everything
  else delegates to the format. A label-shaped id the source has no set for is
  handed to the format anyway — a proxy's upstream may serve it.
- `catalog_tensors` appends the sets after `list_tensor_descriptors`. The
  catalog's scalar `dtype` / `shape_summary` columns describe `tensors[0]`, so
  a set is never first.

A set's adapter is `LabelSetAdapter` (`adapters/labels.py`): `OmeZarrAdapter`
opened on the label group, bound under the parent's `source_id` with the set's
field as its tensor name, so its chunk ids and native levels ride under
`<image>/labels/<name>` (a level's name and `content_version` compose from its
adapter's, for images and sets alike).

## Reads

Every set reads through the ordinary path and the ordinary chunk cache. The
chunk id is a pure function of `array_id`, bounds, scale and method under the
content_version wrapper, so a set has its own cache namespace for free, and the
cache is what makes rasterizing or downsampling a large set affordable.

**Nearest, per tensor.** The advertised pyramid takes its reduction method from
the server-wide `PyramidConfig`; averaging label ids produces ids that exist
nowhere. A set's adapter forces `nearest` on every computed level in
`_advertised_pyramid`, so the precache and clients follow the same ladder. A
native NGFF set that ships its own multiscales serves them as `precompute`.

**Rasterizing OME masks.** A `<Mask>` is `x, y, width, height`, optional
`TheZ/TheT/TheC` pins, and a `BinData` bitmap (1 bit per pixel, row-major,
possibly zlib/bzip2-compressed). The `@ome` set for an image is one tensor
over the image's extent (`adapters/ome_masks.py`); the label value of a mask
is the 1-based index of its ROI in the image's `roi_refs` order, which is
stable because it comes from the metadata. Overlaps: later wins, which
`roi_refs` order gives for free — masks paint in that order. A chunk read
decodes the masks whose bounding box and pin intersect the chunk and paints
them; decoded bitmaps are memoized per adapter, and the ordinary chunk cache
holds the painted output. `TheC` is inert here (design, "Extent" — the channel
distinction is never carried), and a pin naming an axis the image does not
have is inert the same way.

OME-TIFF only, via `OmeTiffAdapter.get_embedded_labels`; a bioio-backed format
carrying OME-XML (a legacy multi-file OME-TIFF) does not rasterize its masks
in this step. The fast metadata path (`_fast_ome_metadata`) used to crash
outright on a real (non-UTF-8) bitmap — `model_dump(mode="json")` decodes a
`bytes` field as UTF-8, and there is no partial dump — silently costing the
file its *entire* OME metadata, not just the mask; `_b64_encode_mask_bindata`
base64-encodes a `Mask`'s `bin_data.value` in place before the dump so it
survives. That base64 form is also what makes stripping it cheap: a real
bitmap must never reach the SQL-queryable `metadata_json` regardless of
whether ROI *annotations* import ran (masks are not annotations, so they are
outside that gate) — `strip_mask_bindata` drops `bin_data.value` unconditionally
in `sync_source_added`, leaving the rest of the shape (extent, pins) in place.

**Precache.** A set is on the ladder like any tensor. Coarse levels of a mostly
empty set are small and nearest over zeros is trivial, so the precache is not
special-cased.

## Upload

The step 7 SDK from biopb/biopb#1048 is reused as it stands, plus one verb:

```python
desc = client.create_tensor("src_ab12/labels/nuclei", labels, chunk_shape=...)
client.upload_array(desc, labels)        # skips all-zero chunks for this kind
client.delete_labels("src_ab12/labels/nuclei")   # frees the name again
```

Server side this is a third upload kind, selected by the request `array_id`
carrying no `cache:` / `ome_zarr:` prefix and a `/labels/` segment. Unlike the
other two, the request's `array_id` *is* the final one. The kind:

- resolves the parent by splitting at the last `/labels/` and refuses if the
  parent is absent, unresolved, or does not serve pixels;
- refuses a non-unsigned-integer dtype, a reserved name, a name that would not
  stay inside the sidecar directory (`unsafe_store_name` — the name becomes a
  directory the server creates and, on discard, removes whole), or a shape /
  `dim_labels` that is not the parent's canonical non-channel extent — a
  request naming no `dim_labels` is filled in from the image rather than
  refused, since the extent rule leaves exactly one legal answer;
- refuses a name already attached, finished or pending — the biopb/biopb#1054
  rule, now per parent;
- creates the sidecar array with the pending marker and the minted
  content_version, and registers the set as **pending** on the parent's
  attachment: routable for `get_flight_info` (so the poll to READY works from
  create) but not listed.

`finish` seals the pending marker, lists the set (`attach_label_set`) and
re-syncs the parent's catalog row (`sync_source_added` is an upsert; the ROI
re-import it triggers is already idempotent), in that order — the catalog must
not name a set a restart would sweep away. The status and TTL machinery apply
as they stand, with two plumbing changes in `UploadManager`: `status` /
`finish` / `discard` / `write_chunk` receive a set's `array_id` where they
receive a `source_id` today, and resolve it through the parent's
`label_uploads` (`_locate`); and
`reap` walks that index on every source as well as the registered upload
sources — a quiet pending set is discarded with its sidecar and unlisted, and
its tombstone detached a TTL later, which is what frees the name.

**Skipping zeros is per kind.** `upload_array` may drop all-zero chunks only for
the label kind, where an unwritten chunk reads as fill. A `cache:` source
answers a read of an unwritten chunk with "holds no chunk", so the skip is not
a general SDK behaviour.

**Replacement.** A set is replaced by uploading under a new name, or by
deleting the old one first. Delete is the one new action.

## Lifecycle: durable uploads get the whole of it

Before step 1 of biopb/biopb#1059 the `durable` flag gated three things:
`discard` raised for a durable upload, the reap sweep skipped it, and the
catalog row was written at create. An abandoned `ome_zarr:` upload was
therefore PENDING for the life of the server, listed, with a partial store on
disk — and after a restart the partial store came back through discovery as an
ordinary source with no upload state at all. A label set would have inherited
all of that, with the sidecar as its only backing.

biopb/biopb#1048 recorded the refusal's reason: a `.zarr` on disk and a catalog
row "are not this call's to release", and deleting a directory "is a genuinely
destructive act whose auth story is its own decision". Both hold for a user's
file. Neither holds for a store the server minted under `write_dir`, which is
where the `ome_zarr:` uploads and the label sidecars both live: the server owns
those bytes, the act is bounded to what it created, and the auth is the one
every other mutation already has — `do_put` and `do_action` are full access.

So, for both kinds:

- **Discard removes the store.** Under the progress lock: mark DISCARDED,
  delete the array, drop it from the listing (the attachment, or the catalog
  row for `ome_zarr:`). The tombstone then behaves exactly as the cache kind's,
  and `reap` stops skipping durable uploads — a quiet pending set is discarded
  after `upload_ttl`, and its tombstone reclaimed after another.
- **The upload state is persisted.** A pending marker is written into the
  store's attrs at create and cleared at `finish`. At startup, a sidecar still
  carrying it is a crashed upload and is deleted rather than served; the
  `ome_zarr:` equivalent is deleted before discovery can claim it, together
  with its catalog row — a persisted catalog outlives the process, and
  `write_dir` is outside every discovery root, so the reconciler never sees
  that id and nothing else would drop it. A sidecar has no row of its own: it
  is a tensor of its parent, whose row is rebuilt when that parent registers.
  The cache kind never needed any of this because nothing of it survives a
  restart.
- **`delete_labels`** (`do_action`, full access) removes a *finished* uploaded
  set: drop it from the attachment, delete the sidecar, re-sync the parent's
  catalog row. Refused for a reserved or native set. Its cache chunks become
  unreachable and fall to LRU, as after a reindex. The name is free again at
  once, safely, because the next set under it mints a new content_version.

A set's lifetime otherwise follows its parent's. `source_id` is a hash of the
resolved URL, so a moved image loses its sidecar sets exactly as it loses its
ROIs today; the `unseen_rois` / `prune_unseen` orphan clock extends to sidecar
directories whose parent has not been seen.

## Discovery

The catalog is the surface. It is the only browse surface, the SPA source tree
is built from it, and the identity policy requires the catalog and
`get_flight_info` to carry identical `array_id`s. A set is an ordinary entry in
`sources.tensors`, found by its path:

```sql
SELECT t.array_id FROM sources, UNNEST(tensors) AS u(t)
WHERE t.array_id LIKE 'src_ab12/labels/%'
```

The typed signal rides in the descriptor `get_flight_info` returns: an NGFF
`image-label` block in the tensor's `metadata_json`, which is documented as
NGFF-compatible, with `source.image` set to the parent's `array_id` (the sidecar
is not adjacent to the image, so NGFF's relative path would be meaningless).
No proto change, no Java build. A `role` column in the `tensors` struct is a
later addition only if filtering on the path proves fragile.

**Authorization.** A set is part of its parent object: a read capability on
the parent covers its sets, and nothing else changes. Create, write, finish and
delete are full access like every other mutation.

## Clients

- **SDK**: `create_tensor` / `upload_array` / `finish_upload` as above;
  `delete_labels(array_id)`; a `label_sets(image_array_id)` convenience over
  the catalog query.
- **MCP / napari**: `add_tensor` builds a `Labels` layer when the descriptor
  carries `image-label`; `viewer.tensor(layer)` plus `upload_array` is the
  round trip. The guide text stops describing masks as a client-only artefact.
- **SPA**: the source tree lists a tensor's sets under it; one is drawn as an
  overlay with nearest sampling and a categorical colormap.

## Implementation order

Each step is independently mergeable.

1. **Durable upload lifecycle** on `ome_zarr:`: discard removes the store,
   reap covers durable, persisted pending marker and boot cleanup, `write_dir`
   outside discovery roots as a checked rule. Revisits the #1048 decision.
2. **Label attachment**: wrapper at the registration seam, sidecar
   enumeration, routing, listing after finish and image-first ordering,
   per-set content_version, nearest ladder, `image-label` metadata. The native
   NGFF `labels/` group in `OmeZarrAdapter` rides on this.
3. **Labels upload kind**: create / write / finish / delete through
   `UploadManager` keyed by `array_id`; `reap` over attachments; SDK zero-skip
   and `delete_labels`.
4. **OME-TIFF masks**: rasterizing adapter, the `@ome` set, `BinData` stripped
   from `metadata_json`.
5. **Clients**: MCP `add_tensor` and guide text; SPA tree and overlay.
