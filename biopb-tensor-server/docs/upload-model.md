# Uploads and the scratch source

Scope: `biopb-tensor-server`, the Python and Java SDKs, and the MCP upload
doc.

## Adding a tensor

`add_tensor` adds a tensor to a source that already exists; nothing on the
upload path creates one. Two forms, the scheme naming only the **store
format** (`zarr://` or `cache://`, one lifecycle for both):

| request | adds |
|---|---|
| `<scheme>://<source_id>/@fields/<name>` | a tensor to a source, discovered or `scratch` |
| `zarr://<array_id>/@labels/<name>` | a label set to the tensor (binding and extent rules: `label-tensors.md`) |

Both carry a **marked segment**, whatever kind the source is: a bare field
is a native tensor id, which only a format mints, so the upload path never
answers one and refuses a request for one (Names).

Each is refused if the parent doesn't exist, the name is taken, or it fails
the rules in Names. The answered `array_id` carries no scheme -- format is a
property of the stored tensor, read off its directory at adoption -- and a
bare id with no `/` is never an upload.

The write side is ticket-based (Wire): `add_tensor` creates the tensor,
`GetFlightInfo` plans it -- answered even while the tensor is PENDING, so a
partial or resumed upload can ask for exactly the tickets it still needs --
and DoPut takes the ticket that plan minted. DoGet takes the same ticket
once the tensor is READY. One planner decides what a chunk is, on both
transports and both store formats; neither SDK tiles or decodes a ticket
itself.

## The scratch source

A result that belongs to no source of the user's needs somewhere to go. A
writable server serves one **scratch source**, at the fixed id `scratch`
(`adapters/scratch.py`), constructed at startup and put on the catalog. It
is a source adapter with no bytes **and no tensors** of its own: what is
added to it is an uploaded field like any other, under
`<write_dir>/fields/scratch/` (Fields). The class is therefore the three
methods `SourceAdapter` declares abstract plus the cap below, and the base
resolves, lists and checks its tensors as it does for any source.

- **The id is fixed, which is the point.** A producer writes
  `zarr://scratch/@fields/<name>` without a round trip first. There is
  nothing to mint, nothing for a client to remember, and nothing to adopt
  at boot -- its tensors come back through the `on_register` hook that
  gives every source its uploaded fields (`fields_attacher`), which is the
  whole of this source's adoption.
- **It keeps no directory of its own**, so there is nothing under
  `write_dir` naming it and nothing to sweep. `content_version` is None,
  the base's word for content this adapter does not serve; every member
  carries its own token.
- **It caps every lifetime on it** (`ServerConfig.scratch_ttl`, a day by
  default), including an upload that asked for none. That is what makes it
  a temp store rather than a shared permanent one; 0 turns the cap off and
  makes it permanent, deliberately.
- **It is not reclaimable, and does not need to be.** Empty is its resting
  state, and it leaves no directory and no orphan row behind.
- **Metadata is source-scoped, and it has none.** A scrap heap describes no
  acquisition -- the tensors on it have nothing to do with each other -- so
  a tensor that needs a physical scale is uploaded onto the source that has
  one. `add_tensor` carries shape, dtype, grid and dim labels only; a
  request that also sets `metadata_json` on a tensor is refused, not
  silently dropped. The one exception is a label set's own NGFF
  `image-label` block, which stays on its `add_tensor` request.
- **`write_dir` is never discovered.** The reconciler owns the user's
  directories; the upload subsystem is the sole registrar for everything
  under `write_dir`. The second line of defence, for a `write_dir`
  misplaced inside a root, is that both zarr claims decline a directory
  carrying the member block (`is_upload_subsystem_store`).

## Store formats

Two member layouts for an uploaded field, chosen by the scheme on
`add_tensor` and recognized from the member directory at adoption
(`member_marker`, `open_any_member`):

| | `zarr://` | `cache://` |
|---|---|---|
| member layout | OME-Zarr image group: `.zattrs` (multiscales, upload marker), `0/` | `descriptor.json` (shape, dtype, chunk grid, dim labels, upload marker), `seg_NNNN.arrow` + `seg_NNNN.idx` |
| bytes on disk | zarr chunks, compressed | one Arrow batch per chunk, exactly as uploaded |
| a read | slice the array; zarr decodes the chunks it overlaps | look the ticket up in the index and hand back the stored batch -- no decode |
| an unwritten chunk | fill value, zeros | not in the index: a zero batch is synthesized |
| transfer grid | a whole multiple of the store's chunk, as any zarr | the uploaded grid, exactly -- the index knows those bounds and no others |
| localhost fast path | default `locate_chunk`: resolve into the chunk cache, answer its byte range | override: the chunk's byte range in its own sealed segment, no copy into the chunk cache |

`cache://` is the file cache's own segment format
(`cache/segment_store.py`, `cache/segment_index.py`), sealed the same way
-- a `.idx` sidecar written at seal -- so the writer and the boot index are
shared code with the chunk cache: code that writes and reads, nothing that
deletes. A member's segments are the tensor, not a cache of it: they sit
outside `max_total_bytes`, the eviction sweep and the retention classes.

The index keys by **bounds**, never by chunk id -- a chunk id carries the
server's serving-semantics epoch, which moves on an upgrade that changes
what the bytes mean, while the bytes on disk do not. At adoption the
adapter mints this build's chunk ids over the indexed bounds and keeps that
map in memory; only an epoch bump rebuilds it. READY closes the open
segment and writes its sidecar -- no segment is open on a READY member.

## Uploaded fields

`<scheme>://<source_id>/@fields/<name>` puts the tensor in
`<write_dir>/fields/<source_id>/<name>/` -- a member directory in either
store format -- whatever kind the source is. One layout, because neither
kind has anywhere of its own to put it: a discovered source's bytes are the
user's, and the scratch source holds none. It is attached by the same
`on_register` hook that attaches label sidecars, and listed after the
format's own tensors (of which the scratch source has none).

A field and a label set are both **attached tensors**: `SourceAdapter`
holds one `field -> adapter` index (`_attached_tensors`), and `label_sets`,
`label_uploads` and `attached_fields` are checked views over it --
`label_sets` is the attached tensors whose field parses as a set, each
checked against `label_binding_error`. A field differs from a label set in
binding to nothing, decoding nothing, and mapping to no axes.

A set may bind to an uploaded field, since a field is a tensor of its
source like any other. An orphaned field -- its discovery root gone -- is
**kept**, not swept, since it is the only copy of what a user uploaded; a
source that returns re-attaches it.

## Names

An attached tensor's id carries a **marked segment** naming what attached
it -- `@labels` for a set, `@fields` for a field -- rather than a reserved
bare word. `@` is the server's marker namespace elsewhere too (`@ome`, the
label set rasterized from an OME-TIFF's embedded masks); an uploaded field
or set name may not open with it.

Marking, not reserving, is what makes an attached id unable to collide with
a native one: `<source_id>/<field>` is exactly the shape of a native
tensor id, so a field or scene named `0` would otherwise shadow one of the
file's own, while `<source_id>/@fields/0` cannot. The rule has no exception
-- the upload path mints no bare field on any kind of source -- so a bare
id always names a tensor some format produced. The disk layout is
unaffected -- NGFF's group is still `labels/` -- only the wire id is
marked, and the parse is right-to-left, so a set literally named `labels`
(`.../@labels/labels`) is still legal.

A name is a path component on every platform that may ever serve this
`write_dir`:

| refused | why |
|---|---|
| a leading `@` (uploaded field/set name) | the marker namespace above |
| `CON`, `PRN`, `AUX`, `NUL`, `COM1`-`COM9`, `LPT1`-`LPT9` | Windows device names, reserved with any extension |
| a trailing space or `.` | Windows strips them silently; two identities fold onto one store |
| `<`, `>`, `"`, `\|`, `?`, `*`, control characters | forbidden on NTFS |
| a leading `.` | the hidden-file convention (zarr's own metadata is `.zattrs` etc.) |

Collision checks fold case (NTFS, APFS and HFS+ are case-insensitive by
default) and NFC-normalize (HFS+ stores NFD), so one name is one identity
on every platform. The length check counts UTF-8 bytes, the 255-byte limit
ext4, APFS and NTFS share.

## Lifecycle

Three states, one action: `set_upload_status(array_id, state, reason)`
moves a tensor PENDING -> READY, or -> DISCARDED from either; nothing
returns it to PENDING yet.

- **Per tensor, not per source.** The marker sits on the member group; a
  member a crash left PENDING is removed by the boot sweep, and its source
  (if any) is untouched.
- **READY is one moment: it seals writes and opens reads.** PENDING
  refuses reads outright -- a hole is indistinguishable from a chunk still
  in flight -- so no state lets a read and a write both land, and no cache
  entry on either side of the wire can predate READY.
- **`content_version` is minted per store, not sampled**, and stamped once
  at create. It separates uploads, not edits: a name reclaimed after a
  discard mints chunk ids that cannot collide with the tombstone's still in
  the cache. Nothing rotates at READY.
- **DISCARDED on a READY tensor removes it**: unlists it and releases its
  store. A tensor adopted from an earlier life holds no progress record, so
  discarding it is store removal alone; a native tensor is never
  discardable.
- **A deadline is swept at any state; idleness only at PENDING.** A
  tensor whose `ttl_seconds` has run out is discarded wherever it is on the
  ladder -- a lifetime that stopped applying at READY would be no lifetime,
  since READY is where a finished result spends its life. Idleness is
  PENDING's alone: past it nothing is expected to make progress. The
  deadline is recorded in the member's marker as an absolute wall-clock
  time, so it survives a restart, and one that ran out while the server was
  down is swept at boot rather than adopted and reaped later.
- **A tombstone is detached once it has stood for `upload_ttl`**, which is
  what frees the name. So a discarded tensor takes two sweeps to vanish and
  is a tombstone in between, reachable to whoever is polling it.

## Wire

| action | body |
|---|---|
| `add_tensor` | `TensorDescriptor` in, `TensorDescriptor` out (`array_id` minus scheme, the store's own chunk grid, and `ttl_seconds` as the lifetime granted) |
| `GetFlightInfo` | `FlightRequest.tensor_read`; answered for a PENDING tensor too; `slice_hint` limits the plan |
| DoPut | `PutCommand.chunk_ticket` in the descriptor command, the chunk's batch as the stream |
| `set_upload_status` | `SetUploadStatus`; READY and DISCARDED are the only settable states |
| `upload_status` | unchanged |

Java (`TensorUploads`, `TensorFlightClient`) and the TS client mirror the
same actions and ticket flow, with no client-side tiling or decoding. There
is no `create_tensor` or `register_source` action, `ome_zarr:` source, or
volatile `cache:` source in this model -- every tensor is added to a source
by `add_tensor`, and every source is either discovered or the one the
server serves itself.
