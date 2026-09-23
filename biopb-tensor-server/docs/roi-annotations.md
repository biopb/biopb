# ROI annotations

**Experimental.** The ROI annotation schema and wire types may still change without notice.

A user draws 2-D ROIs (points, polygons, rectangles, ellipses, polylines) on
a tensor in the viewer SPA; they persist across restarts in a `rois` table
in the tensor server's DuckDB catalog, reached over a small Flight action
set and re-exposed by the HTTP sidecar.

Scope of this doc: the backend (store, wire types, API surfaces). The SPA's
draw tooling is [roi-annotations-ui.md](../../web/docs/roi-annotations-ui.md).

## Model

An **annotation** is a geometry plus the metadata that makes it findable. The
geometry is `biopb.image.ROI` (`proto/biopb/image/roi.proto`), reused as-is;
`ROI` carries geometry only (no id, no label), so the record around it,
`RoiAnnotation`, is new.

Only the 2-D vector arms are accepted: `point`, `rectangle`, `ellipse`,
`polygon`, `polyline`. A `mask` (a `BinData` bitmap, hundreds of KB, past
"annotation scale") or `mesh` (3-D, where plane pinning has no meaning) is
rejected but stays in the proto, so accepting one later needs no wire
change.

- **One row per ROI, not one row per set** — useful for `WHERE label =
  'mitotic'`, per-field object counts, or a join against `sources`. A
  layer's atomicity comes from `set_name` plus a batched `put_rois` in one
  transaction, not from an opaque row.
- **Coordinates are level-0 pixels**, floating point — a shape drawn on a
  downsampled level is scaled up client-side before the write.
- **Plane pinning is a sparse map, keyed by wire axis index** — `{2: 12}`,
  not `{"z": 12}`. A missing key means "every index".
- **Annotations are not source metadata** — not merged into `sources` table,
  but live in a sibling table (`rois`) in the same catalog DB.

## Reserved sets

A `set_name` starting with `@` is server-owned: a cache of the source file,
filled by an adapter's `get_embedded_rois` and replaced wholesale on
re-registration (today's importer files OME-embedded ROIs into `@ome`). The
store refuses a client write addressed to a reserved set, by name or by one
of its ids, since an edit landing there would be silently destroyed at the
next re-import -- so **editing is a clone, done client-side**: read the set,
mint fresh ids, write them into a set of your own. `list_rois` without a
`set_name` returns only client-owned sets; naming a reserved one is the only
way to read it. A reserved set shares the `sources` lifecycle (cleared and
rewritten alongside a source's own row) and sits outside the orphan
machinery entirely, since pruning a cache of the source file would reclaim
nothing a re-import doesn't already rebuild; the import itself can never
fail a registration and is exempt from `max_rois_per_tensor`, since a user
should not be pushed against the cap by rows they did not author.

## Schema

The `rois` table, in `MetadataDatabase._create_schema()`:

```sql
CREATE TABLE rois (
    roi_id     TEXT NOT NULL,           -- uuid4 hex; client-supplied or server-minted
    array_id   TEXT NOT NULL,           -- the anchor, unversioned
    source_id  TEXT NOT NULL,           -- array_id split on the first '/'; joins + authz
    source_url TEXT,                    -- last-seen catalog URL; names an orphan report
                                         -- and anchors re-attach. NULL until first seen
    set_name   TEXT NOT NULL DEFAULT 'default',
    label      TEXT,                    -- user class/name
    shape_kind TEXT NOT NULL,           -- point|rectangle|ellipse|polygon|polyline
    plane      MAP(UINTEGER, UINTEGER), -- axis -> index; absent key = all indices
    bbox       DOUBLE[4],               -- [x0,y0,x1,y1], level-0 px, derived server-side
    geometry   TEXT NOT NULL,           -- biopb.image.ROI as canonical proto3 JSON
    props_json TEXT,                    -- free-form client JSON (color, score, author)
    drawn_against_version TEXT,         -- content_version at write time, or NULL
    rev        BIGINT NOT NULL,         -- per-roi, monotonic
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    last_seen_at TIMESTAMP,             -- source's last observed-in-catalog time; NULL = never seen
    PRIMARY KEY (array_id, roi_id)
);
CREATE INDEX idx_rois_array ON rois(array_id);
```

`bbox` is derived server-side from `geometry` and is what makes the SQL
surface useful ("annotations overlapping this region," per-label counts
over a plate) though the viewer never reads it, always fetching a tensor's
whole set instead. `geometry` is proto3 JSON **text**, not a serialized
blob, so the sidecar hands it to the SPA verbatim while keeping the row
legible to SQL. The primary key is composite because `roi_id` is unique
*within a tensor* only -- two tensors both choosing `"roi-1"` is ordinary
(a batch naming one twice is refused) -- and a client-supplied id is
length-bounded (128 bytes) and comma-free, since the sidecar's delete route
addresses ids by a comma-separated list.

`rois` is in `MetadataDatabase.ALLOWED_TABLES`, SELECT-only like every other
allowed table -- `client.query_sources(...)` and the MCP catalog surface can
read it for analysis, but the viewer always uses the typed read below.
`sync_source_removed()` does not cascade into `rois`, since a rescan or a
transient unregister must not destroy a user's work: orphaned rows re-attach
if the same path is registered again, and deleting is always an explicit
call (see Staleness). `put_rois` wraps its batch in one transaction, since
DuckDB otherwise autocommits per statement and a failure partway through
would leave partial rows a concurrent `list_rois` could watch appear one at
a time. `annotations.max_rois_per_tensor` (default 5000) is the line
between "annotation store" and "object store" -- a million-object
segmentation belongs in a label tensor.

## Wire types

`RoiAnnotation` (`proto/biopb/image/annotation.proto`, the `biopb.image`
package) carries the fields described above -- `roi_id`, `array_id`,
`set_name`, `label`, `roi` (the geometry), `plane`, `props_json`,
`drawn_against_version`, `rev`, and two timestamps; field 6 is reserved
(a dropped `dim_label`-keyed pin, never to be reused). It wraps
`biopb.image.ROI` directly rather than importing `biopb.tensor` for it,
since `biopb.image` already imports `biopb.tensor` and the reverse would
cycle. `RoiPutResult`/`RoiDeleteResult`/`RoiListResult`/`RoiSetInfo` carry a
batch's outcome; `RoiPruneRequest`/`RoiUnseen`/`RoiPruneResult` are the
`roi_prune` action's own types; `biopb/tensor/ticket.proto` adds
`RoiRead`/`RoiPut`/`RoiDelete` as the flight's addresses
(`TensorTicket.roi_read`, `PutCommand.roi_put`, `PutCommand.roi_delete`).
See the proto files for exact field numbers.

A read takes no plane or bbox filter, since the client filters the resident
set in memory; `set_name` scopes it (empty returns client-owned sets, a
reserved set only comes back when named), and `sets` always covers the
whole tensor by counting stored rows, so a client can see a reserved set
exists. Concurrency is per-ROI optimistic: the server bumps `rev` on every
write, and with `check_rev` set, a request whose `rev` doesn't match the
stored one is reported in `conflicts` and not applied while the rest of the
batch still lands -- a client that doesn't care leaves `check_rev` false
and gets last-writer-wins.

## API surfaces

**The `roi` flight** is the authoritative surface — the HTTP sidecar is a
separate client that reaches the server over gRPC and cannot touch the
DuckDB catalog directly:

| Verb | Wire | Stream | Reply |
|---|---|---|---|
| DoGet | `TensorTicket.roi_read {array_id, set_name}` | ROI rows | `truncated` + `sets` (JSON) in the stream's schema metadata |
| DoPut | `PutCommand.roi_put {array_id, check_rev}` | ROI rows | `RoiPutResult` in the put's app_metadata |
| DoPut | `PutCommand.roi_delete {array_id, set_name}` | one `roi_id` column, or empty | `RoiDeleteResult` in the put's app_metadata |

One row schema serves both directions (`biopb.image._roi_rows.ROI_ROW_SCHEMA`)
-- what the server streams on a read is what it accepts on a write, and
`TensorFlightClient.put_rois()` / `list_rois()` / `delete_rois()` rebuild the
result messages from that row stream, which is also what gives
napari-via-MCP the same feature with no second store. Orphan cleanup is
separate: `roi_prune` runs against the catalog directly, not a tensor (an
orphaned row has no live source to authorize against), as a server-token
action reporting unless `apply` is set.

**Why a flight and not the SQL surface.** A read or write names one tensor
and is authorized on its source, exactly like a pixel read; a SQL query
names no source, so `rois` stays writable only through the flight (readable
via SQL for analysis). Each action authorizes on the source first: a
feature-level refusal (annotations disabled, no metadata DB) raises
`FlightUnavailableError`, a rejected request (bad geometry, cap breached)
raises `FlightServerError` -- the split is what lets the sidecar answer 501
vs 422 without parsing messages. None are gated on `--writable`, since an
annotation writes no pixels; the auth boundary is the token (required in
remote mode) plus `annotations.enabled` (default true).

**HTTP sidecar** (`serving/http_server.py`), the SPA's path via the control's
`/data_plane/*` proxy:

```
GET    /api/rois/{array_id:path}?set=
POST   /api/rois/{array_id:path}        body: {"rois": [RoiAnnotation…], "check_rev": bool}
DELETE /api/rois/{array_id:path}?ids=a,b&set=nuclei
```

The mutating routes call `ctx.check_token(request)` then
`_require_same_origin(request)`, and JSON bodies are canonical proto3 JSON
on both ends. Filling `drawn_against_version` is the caller's job (cheaper
than a server-side describe round trip per save), and the handler strips
the `@version` token wherever one can enter -- the path and each
annotation's own `array_id` -- since responses carry versioned ids and a
read-edit-write round trip hands one straight back.

## Persistence

The whole catalog is file-backed (`MetadataDatabase(store_path=...)` opens a
DuckDB file instead of `":memory:"`, shared with `sources` and
`decode_rates`), but annotations are the only rows no rescan can reproduce.
Reloaded rows re-attach with no fixup, since `array_id` is deterministic
across restarts for file-backed sources (upload/scratch sources are the
exception, their URL synthesized per run), and a file that won't open, or
whose `rois` shape this build doesn't understand, is fatal
(`AnnotationStoreError`) after a short retry -- neither renaming it aside
nor falling back to an in-memory catalog is safe, since the latter would
keep the server up while every ROI drawn on it silently stops surviving a
restart. A person choosing `catalog.persist = false` gets a session-only
store on purpose (`annotations_persisted: false` in `health`, not an error).
`rois` alone carries a real schema-migration ladder -- `sources` is dropped
and recreated every open and `decode_rates` just drops and re-collects on a
version bump, but nothing can reproduce an annotation, so an older file is
migrated and a newer-build file refused rather than misread.

## Staleness

**Content staleness** (pixels changed under the annotation) is handled by
`drawn_against_version`: a client compares it against the current
descriptor and warns. Nothing to prune.

**Referential staleness** -- the tensor is gone and the row points at
nothing -- needs a policy since there's no existence oracle: the catalog is
routinely an incomplete picture (discovery is progressive, an unresolved
cloud source is absent and entirely intact), and only file-backed sources
have a path to stat at all. So the rule is: never assert deletion, only
measure elapsed time since last observed presence -- `mark_sources_seen()`
stamps `last_seen_at = now()` for every ROI whose source is currently in the
catalog, run at the end of each full scan, so a source offline for a week
simply never advances it.

Deletion is a policy over age (`annotations.prune_unseen_days`, default `0`
= never), not a claim about the world -- `prune_unseen(before)` applies it
and `unseen_rois(before)` reports what it would take using the same
predicate, so a dry run and the real thing can't drift, reachable against a
running server (`biopb tensor prune-annotations --days N [--apply]`) or
directly against a stopped one's catalog file. Auto-pruning stays opt-in:
these are hand-drawn rows, so the safe default is to surface orphans and let
a person confirm.

### Known limitation: a move orphans annotations

Identity is path-derived, so `mv` gives a local file a new `source_id`, a
new `array_id`, and the tensor appears freshly unannotated while the old
rows go orphan. Re-attachment, not pruning, is the fix -- not yet
implemented, though the stored `source_url` plus `drawn_against_version`
(which survives a plain `mv`, since it's `mtime_ns:size`) is enough to offer
"these annotations were drawn on a file of the same name, size and mtime --
re-attach?" instead of silently losing them. A **proxied** source is the
opposite case by design: its id is built from `(alias, upstream_source_id)`
and carries no endpoint, so moving the upstream file leaves annotations
attached -- the equivalent event there is renaming the `alias`, see
[remote-tensor-cache.md](remote-tensor-cache.md).
