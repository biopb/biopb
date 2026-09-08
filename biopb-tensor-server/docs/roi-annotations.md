# ROI annotations — backend design

Lets a user draw 2-D ROIs (points, polygons, rectangles, ellipses) on a tensor in
the viewer SPA and have them persist across restarts. The store
is a new table in the tensor server's DuckDB catalog, reached over a small Flight
action set and re-exposed by the HTTP sidecar.

Scope of this doc: the backend (store, wire types, API surfaces). The SPA's draw
tooling is `docs/roi-annotations-ui.md`.

## Non-goals

**Instance segmentation is a separate tensor, not an annotation set.** A
segmentation's objects belong in a label tensor the server already serves as
pixels — not as 10⁴–10⁶ rows in the catalog. This is the load-bearing
scope decision: it is what lets the read path be a single whole-set fetch over
DoAction, keeps the per-tensor cap human-scale, and confines the store to the
2-D vector shapes. If bulk objects ever need a home here, revisit storage and
channel together rather than growing this table.

Also out: undo/history, cross-tensor annotation search, and export to
OME-XML / GeoJSON (the geometry is proto, so an exporter is additive).

## Model

An **annotation** is a geometry plus the metadata that makes it findable. The
geometry is `biopb.image.ROI` (`proto/biopb/image/roi.proto`), reused as-is, and
Python exports it from `biopb.image`. `ROI` carries geometry *only* — no id, no
label — so the record around it is new (`RoiAnnotation`, below).

**Only the 2-D vector arms are accepted:** `point`, `rectangle`, `ellipse`,
`polygon`, `polyline`. A `mask` or `mesh` is rejected with a clear error. This
follows from instance segmentation being a separate tensor (see Non-goals):
`Mask` carries a `BinData` bitmap, so a single ROI can be hundreds of KB and a
few thousand of them stop being "annotation scale" in the one dimension the cap
is trying to bound — and `Mesh` is 3-D, where plane pinning has no meaning. The
proto keeps both arms, so accepting them later is additive with no wire change.

**`Polyline` is new in `roi.proto`** — the scribble / freehand stroke, which the
existing shapes could not express. Its vertex list looks like a `Polygon`'s, but
the two are not interchangeable: a polygon is a closed region where "inside" is
meaningful, a polyline encloses nothing and marks the band of pixels under the
brush. OME-XML draws the same distinction, so this is the shared geometry
vocabulary catching up rather than a local extension — it goes in `roi.proto`,
where a detection response can return one too, not in `annotation.proto` where
only this store would know it.

Its `width` is part of the geometry, not styling. A scribble labels the pixels
the brush covered, so the width decides what the ROI *means* — and concretely,
the bbox extends `width/2` past the vertex extent. Had the width lived in
`props_json` (opaque to the server) the derived bbox would silently
under-report every fat stroke.

Five decisions fix the semantics:

**One row per ROI, not one row per set.** A set-per-row store — `(array_id,
set_name) -> blob of all its ROIs` — is simpler to write, but it makes the
catalog a key-value box: the row is opaque, every read is by primary key, and
nothing in the design still argues for DuckDB over a plain dict. Row-per-ROI is
what makes the premise pay off — `WHERE label = 'mitotic'`, per-field object
counts, a bbox overlap, a join against `sources` — and at the cap here the row
count is trivial. What set-per-row wins is atomicity of a whole layer; row-per-ROI
gets that back from `set_name` plus a batched `roi_put` applied in one real
transaction (see below — the write lock alone does not provide that). Per-set attributes (display colour, visibility) have no table yet on
purpose — they are client display state today, and a `roi_sets` table is additive
if they ever need to be shared.

**Anchor to `array_id`, unversioned — and enforced, not assumed.** A tensor is identified by `array_id` alone
(see the identity policy in `descriptor.proto`), so that is the foreign key —
not `source_id` (a multi-field source has many tensors) and not the sidecar's
`source@token/field` HTTP form, which changes whenever the file's
`content_version` changes. Annotations must outlive an in-place edit of the
image, so the sidecar strips the version token before the Flight call exactly as
the tile route does (`_split_array_version`). What *is* stored is the
`content_version` observed at write time (`drawn_against_version`), so a client
can say "this image changed since it was annotated" instead of silently drawing
stale outlines.

The store *refuses* a versioned id rather than trusting callers to strip. This is
worth a rule because of where the existing sidecar gets its correctness from:
every read route handles the token as a side effect of resolving a descriptor
(`_tensor_desc_by_array_id` strips and validates in one step), so the invariant
had no independent existence. A store keyed by `array_id` resolves nothing, so a
missed strip was silent — the annotation filed itself under a phantom tensor
whose id is never minted again once content changes, `source_id` matched no
catalog row, and a bare read returned nothing. Loud beats silent: a versioned id
is not a different tensor, it is a caller that forgot. Only the source half is
checked, since `@` is legal payload in a field name.

**Coordinates are level-0 pixels.** Always full-resolution, in the tensor's own
Y/X axes, floating point. A polygon drawn on a downsampled level is scaled up by
the client before the write. The server never rescales geometry.

**Plane pinning is a sparse map, not t/z/c fields.** A 2-D ROI lives on one plane
of an N-D tensor, and the axes are per-tensor, so the pin is a map — and a
missing key means "applies at every index of that dimension", which is how a
user gets an ROI that follows a z-stack or a time course without duplicating it
per plane.

**The map is keyed by wire axis index, not by dim_label** — `{2: 12, 0: 0}`, not
`{"z": 12, "t": 0}`. The axis is a 0-based position in the tensor's own
`dim_labels`, and so is the index on it. Labels were the first design and are wrong for a reason
that only shows up on real data: a label is neither guaranteed present nor
guaranteed unique. A TIFF sequence's opaque file axis has none, and two axes of
one tensor may share one. A label-keyed pin cannot address those axes *at all*,
so an annotation on them silently broadcasts across every index — usually every
field of view, which is the worst case to get wrong quietly. Position addresses
every axis, and it is the basis the geometry already uses (coordinates are in
the tensor's own Y/X axes, found positionally).

The label is still the right thing to *show*; any client holding the descriptor
can turn an index back into one. It is deliberately not stored alongside: two
spellings of one axis could disagree, and then neither would be authoritative.

**Annotations are not source metadata.** They do not go in `sources.metadata_json`
and are not merged into `GET /api/sources/{id}/metadata`. That column is
adapter-produced and rewritten by the `INSERT OR REPLACE` in `sync_source_added()`
on every re-registration, so an annotation parked there would be destroyed by the
next rescan. "Read as metadata" here means a sibling table in the same catalog DB,
queryable next to `sources` — not a field inside a source row.

## Reserved set names

A `set_name` beginning with `@` is server-owned. Nothing fills one yet; the
namespace exists so the rule is in place before an importer can write to it
(biopb/biopb#951, which imports OME-embedded ROIs into `@ome`).

Such a set is a **cache of the source file**: filled from it, and replaced
wholesale when the file changes. So a client edit landing there is not merely
filed in the wrong layer — it is destroyed at the next re-import, silently. The
store refuses the write rather than trusting clients to honour the convention,
the same posture it takes toward a versioned `array_id`.

A prefix rather than a fixed name, so a second importer (ImageJ overlays, a
GeoJSON sidecar) becomes read-only with no client release. Punctuation because
the namespace must be one no existing catalog can already be using — a user set
called `ome` is plausible where `@ome` is not.

**Editing is a clone, and it is client-side**: read the set, mint fresh ids, write
them into a set of your own. No server operation, and clients tell mutable from
immutable by the name alone.

Fresh ids are not a style preference. `_put_rois_locked` looks up existing rows by
`(array_id, roi_id)` — `set_name` is not in the key, and it *is* one of the
columns an update writes. A clone reusing an imported id would therefore take the
update branch and **move** the row out of the reserved set rather than copy it,
with no error. That write is refused too, from the other side.

Deleting is refused when a reserved set is addressed, by name or by one of its
ids — a delete that silently dropped part of what it was handed would report
success for work it did not do. The exception is an unqualified "clear this
tensor", which scopes reserved sets out instead: nothing in that request was
addressed to them, they are not the caller's data, and re-import would restore
them anyway.

**A reserved set is outside the orphan machinery too** — `unseen_rois` skips it
and `prune_unseen` will not delete it. The clock exists to give hand-drawn work a
long grace period before anything removes it; a cache of the source file has
nothing to protect, and pruning it reclaims nothing a re-import would not
rebuild. Both share `_UNSEEN_PREDICATE` rather than repeating the SQL, because
the report is the dry run for the delete and a difference between them would show
a person one set of rows and remove another.

This one is load-bearing rather than tidy: `prune_unseen` deletes with raw SQL,
not through `delete_rois`, so it is a second deletion path that the guard above
does not cover. Without the exclusion the CLI would delete rows the API refuses
to touch, and `prune-annotations` would count imported copies in the total it
asks a person to confirm.

### They share the `sources` lifecycle

An imported set is **scan output**, exactly like a `sources` row — derived from a
file, rewritten when that file is re-registered. So it gets that lifecycle rather
than machinery of its own: cleared in `_create_schema` alongside
`DROP TABLE IF EXISTS sources`, written by `sync_source_added` in the same
transaction as the upsert, removed by `sync_source_removed` with the row.

That is deliberately *not* a watermark table. `sync_source_added` is already the
replace-on-rescan mechanism — re-registration is driven by the same stat
signature `content_version` comes from — so there is no separate freshness
question to track, nothing to keep in sync with the rows, no "have I imported an
empty set?" ambiguity, and no reaper to write. Deriving there is also free:
`get_metadata()` has just been called, so the import is a dict walk rather than a
second parse.

Two consequences. A **rebuildable subset now lives in the table whose whole
justification is being unrebuildable** — which cuts the useful way, since
`_ROI_MIGRATIONS` may delete and re-derive reserved rows instead of migrating
them. And **availability couples to registration**: imported ROIs exist only
while their source does, and appear progressively as discovery proceeds. Correct
rather than unfortunate — a tensor that is not registered cannot be opened.

The `rois` key is stripped from `metadata` before it is serialised into
`sources.metadata_json`. Once the store owns them a second copy is duplicated
bulk, and it would keep annotations visible on
`GET /api/sources/{id}/metadata` — the surface this design says they do not
appear on. Safe because the derivation reads the adapter's fresh
`get_metadata()` return, never the stored column.

**The format decides, through the adapter API.** `SourceAdapter.get_embedded_rois`
returns `({}, None)` by default; `OmeTiffAdapter` and `_BioioAdapterBase` (hence
every vendor subclass) implement it. The server does not police what
`get_metadata()` returns, so a `rois` key means whatever that format meant by it
— reading an EMD's `original_metadata` or an OME-Zarr's `.zattrs` as OME-XML
would invent annotations.

Not keyed on `source_type` either. That is a name, and it lies in both
directions: `ome-zarr` carries NGFF rather than OME-XML, while `zeiss`,
`leica`, `nikon` and the rest are ome-types dumps through bioio. A hook also
generalises the way this design expects — ImageJ overlays or a GeoJSON sidecar
are a method on their own adapter, not another branch here.

The metadata dict and the tensor list are passed *in* rather than recomputed:
the caller holds both, and `get_metadata` is a documented pure producer that
would re-parse.

`annotations.enabled = false` skips the parse entirely. Those rows would be
unreadable through every surface — the SQL one drops `rois` from
`allowed_tables` too — so the work and the storage buy nothing, and `rois` stays
in `metadata_json` because nothing read it.

The open-time clear runs **after** `_reconcile_roi_schema`, not beside the
`DROP TABLE sources` that motivates it. That check can refuse the catalog, and
it promises "The file is untouched" when it does — a delete before it would make
that a lie, and would run against a schema this build has not established it
understands.

**The import cannot fail a registration.** It runs on a file nobody here wrote,
and it sits inside `sync_source_added` — so an unguarded raise would cost a
source its pixels over an annotation, which is backwards: an imported set is
disposable (the next registration rebuilds it, and open clears it anyway) where
a source that will not register is an outage.

Three layers, because the raises are of three kinds. A malformed *shape* — a
coordinate that is not a number, a `TheZ` outside uint32 — is dropped on its own
and counted, so one bad shape cannot cost a file its other forty. A malformed
*structure* (`union` that is not a mapping, `rois` that is not a list) is skipped
the same way. And the whole read is wrapped, since reaching that handler means
something the module does not model at all.

The write is likewise after the source row commits, not inside its transaction,
and swallows its own failures. `_prepare_roi` stays the single authority on what
is storable — the importer skips rows it rejects rather than carrying a second
copy of the rules.

The one thing not dropped on a failure is `sources.metadata_json`'s `rois` key:
it is stripped only when the read completed, so a wholesale failure leaves the
last copy in place instead of losing it silently. Per-shape drops are counted in
the log line.

Imported rows are also outside `max_rois_per_tensor`. The cap exists to keep this
an annotation store rather than a segmentation store; a user should not be pushed
toward it by rows they did not author, and cloning an imported set — which is how
editing one works — is exactly what would trip it.

## Schema

New table in `MetadataDatabase._create_schema()`:

```sql
CREATE TABLE rois (
    roi_id     TEXT NOT NULL,           -- uuid4 hex; client-supplied or server-minted
    array_id   TEXT NOT NULL,           -- the anchor, unversioned
    source_id  TEXT NOT NULL,           -- array_id split on the first '/'; joins + authz
    source_url TEXT,                    -- catalog URL at the last sighting; names the image
                                        -- in an orphan report, and anchors re-attach after a
                                        -- move. NULL only until the source is first seen
    set_name   TEXT NOT NULL DEFAULT 'default',
    label      TEXT,                    -- user class/name
    shape_kind TEXT NOT NULL,           -- point|rectangle|ellipse|polygon|polyline
    plane      MAP(UINTEGER, UINTEGER), -- axis -> index; absent key = all indices
    bbox       DOUBLE[4],               -- [x0,y0,x1,y1], level-0 px, derived server-side
    geometry   TEXT,                    -- biopb.image.ROI as canonical proto3 JSON
    props_json TEXT,                    -- free-form client JSON (color, score, author)
    drawn_against_version TEXT,         -- content_version at write time, or NULL
    rev        BIGINT NOT NULL,         -- per-roi, monotonic
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    last_seen_at TIMESTAMP,             -- last time the source was observed in the catalog;
                                        -- orphan age = now() - last_seen_at, NULL = never seen
    PRIMARY KEY (array_id, roi_id)
);
CREATE INDEX idx_rois_array ON rois(array_id);
```

`bbox` is derived by the server from the geometry. It is not used by the viewer read path — which fetches a
tensor's whole set — but it is what makes the SQL surface useful: "annotations
overlapping this region", "objects per field", per-label counts over a plate.

`geometry` is proto3 JSON **text**, not a serialized blob. The proto stays the
schema — both ends still validate through `json_format` / protobuf-es — but text
buys two things a blob does not: the sidecar hands the stored geometry to the SPA
verbatim, with no decode/re-encode on the hot path, and the row stays legible to
the SQL surface (`SELECT geometry FROM rois`, or reach inside with
`geometry->'$.polygon.points'`). At annotation scale the ~2× size over a blob is
not worth optimising.

**The key is composite, and has to be.** `roi_id` is unique *within a tensor*,
not globally: a client may name its own ids, so two tensors independently
choosing `"roi-1"` is ordinary rather than a conflict. With `roi_id` alone as the
primary key this silently destroyed data — the create-or-update lookup is scoped
by `array_id` and saw no conflict, so it treated the write as a create, while
`INSERT OR REPLACE` hit the global key and overwrote the *other* tensor's row.
Every other query here is already array_id-scoped; the key now matches. A batch
naming one `roi_id` twice is refused (the writes would collapse), and a
client-supplied id is length-bounded since it becomes half the key.

A client-supplied `roi_id` is stripped, length-bounded, and may not contain a
comma — the sidecar deletes by a comma-separated `?ids=` list, so such an id
could be created but never addressed, and the delete would report zero removals
with no error.

`set_name` groups annotations into a layer ("nuclei", "hand-drawn"). It is in the
first version deliberately — it is what makes bulk delete and layer toggling
possible, and retrofitting a grouping key later is a migration of every stored row.

**Read via SQL — as an analysis affordance, not the client read path.** Add
`"rois"` to `MetadataDatabase.ALLOWED_TABLES` so `client.query_sources(...)` and
the MCP catalog surface can count labels, join against `sources` and sweep the
whole catalog. It costs one set entry. The viewer never composes SQL — it uses
the typed read below, which builds parameterized SQL server-side.
`FORBIDDEN_KEYWORDS` is unchanged: the SQL surface stays SELECT-only, and every
write goes through the typed methods.

**Lifetime.** `sync_source_removed()` does *not* cascade into `rois`. A rescan or
a transient unregister must not destroy a user's work; orphaned rows are tiny,
and they re-attach if the same path is registered again (`source_id` is a SHA-256
of the resolved path, so it is stable). Deleting annotations is an explicit call.
See Persistence and staleness for why absence from the catalog is never on its
own a reason to delete.

**A batch is a transaction, not just a locked section.** `put_rois` wraps its
whole body in `BEGIN`/`COMMIT` with a rollback on any failure. The write lock
only serializes writers — DuckDB autocommits each statement, so without this a
failure partway through left the rows written so far behind, and a reader
(`list_rois` deliberately takes no lock, using its own cursor) watched a layer
appear row by row. Cursors observe the pre-commit snapshot, so one transaction
buys both all-or-nothing recovery and an all-or-nothing view. The rollback is
guarded so a dead connection cannot mask the original error, and so an open
`BEGIN` can never poison the shared connection for later writers.

**Cap.** One config knob, `annotations.max_rois_per_tensor` (default 5 000): a
write that would exceed it fails with a clear error. The number is deliberately
human-scale — it is the line between "annotation store" and "object store", and
it is what lets the read path stay a single whole-set fetch (see below). A
million-object segmentation belongs in a label tensor, not here (Non-goals).

## Wire types

`proto/biopb/image/annotation.proto`, in the `biopb.image` package. It belongs
there rather than in `biopb.tensor` for two reasons: it wraps `biopb.image.ROI`,
and `biopb.image` already imports `biopb.tensor` (`image_data.proto`), so the
reverse import is a package cycle — `buf lint` rejects it. Nothing is lost:
`array_id` crosses as a plain string, so the record needs nothing from
`biopb.tensor`.

```proto
message RoiAnnotation {
  string roi_id = 1;              // empty on create -> server mints a uuid4
  string array_id = 2;            // unversioned
  string set_name = 3;            // empty -> "default"
  string label = 4;
  biopb.image.ROI roi = 5;        // geometry, level-0 pixel coords
  reserved 6;                     // was the dim_label-keyed pin
  map<uint32, uint32> plane = 12; // wire axis index -> index; absent = all
  string props_json = 7;
  optional bytes drawn_against_version = 8;
  int64 rev = 9;                  // server-assigned; echo it back to write safely
  int64 created_at_unix_ms = 10;
  int64 updated_at_unix_ms = 11;
}

message RoiPutRequest  { string array_id = 1; repeated RoiAnnotation rois = 2; bool check_rev = 3; }
message RoiConflict    { string roi_id = 1; int64 stored_rev = 2; }
message RoiPutResult   { repeated RoiAnnotation stored = 1; repeated RoiConflict conflicts = 2; }
message RoiDeleteRequest { string array_id = 1; repeated string roi_ids = 2; string set_name = 3; }
message RoiDeleteResult  { repeated string deleted = 1; }

message RoiListRequest { string array_id = 1; string set_name = 2; }
message RoiListResult  { repeated RoiAnnotation rois = 1; bool truncated = 2; }
```

`RoiListRequest` takes no plane or bbox filter on purpose. The client fetches a
tensor's whole annotation set once and filters in memory: it needs every ROI
resident anyway to hit-test, drag a vertex and re-render, and a viewport-filtered
fetch would make the ROI you are mid-edit disappear on a pan.

Concurrency is per-ROI optimistic: the server bumps `rev` on every write and
returns the stored record. With `check_rev` set, a request whose `rev` does not
match the stored one is reported in `conflicts` and not applied — the rest of the
batch still lands. Two viewer panes on one tensor is already the normal case, so
this is worth having from the start; a client that does not care leaves
`check_rev` false and gets last-writer-wins.

## API surfaces

**Flight actions** (`server.do_action`, listed in `list_actions`) — the
authoritative surface, because the HTTP sidecar is a separate client that reaches
the server over gRPC and cannot touch the DuckDB catalog directly:

| Action | Body | Result |
|---|---|---|
| `roi_put` | `RoiPutRequest` | `RoiPutResult` |
| `roi_list` | `RoiListRequest` | `RoiListResult` |
| `roi_delete` | `RoiDeleteRequest` | `RoiDeleteResult` |

**Why DoAction and not DoGet/DoPut.** The catalog's other read surface
(`__metadata_query__`) goes over DoGet because it answers an arbitrary *query*
whose result set is unbounded — that is what earns the ticket, the Arrow stream
and the backpressure. ROI reads are not a query: the request is "the annotation
set for this tensor", the answer is bounded by the per-tensor cap, and the client
wants the whole thing. A ticket + record-batch stream for a few hundred rows is
ceremony that buys nothing, and it splits authorization across two calls
(GetFlightInfo mints a ticket that then travels on its own) where DoAction
authorizes the one call that does the work. Writes are small structured commands
with a structured reply (`stored` + `conflicts`), which is DoAction's shape;
DoPut's only response channel is a single app-metadata blob, and it is gated on
`--writable`, which annotations deliberately are not.

**What would flip this to DoPut/DoGet** is bulk object import — ruled out in
Non-goals, and this is one of the things that decision buys.

Each calls `self._authorize_source(context, array_id.split("/")[0])` first, like
`chunk_locate`. A feature-level refusal (annotations disabled, or a server with
no metadata DB) raises `FlightUnavailableError`; a rejected request (bad
geometry, mismatched array_id, cap breached) raises `FlightServerError`. That
split is what lets the sidecar answer 501 vs 422 without parsing messages. They are *not* gated on `--writable`: that flag governs creating
tensor stores, and an annotation writes no pixels. The auth boundary is the token
(required in remote mode), plus `annotations.enabled` (default true) for a
deployment that wants a strictly read-only catalog.

Mirrored on `TensorFlightClient` as `put_rois()` / `list_rois()` / `delete_rois()`,
which is also what gives napari-via-MCP the same feature without a second store.

**HTTP sidecar** (`serving/http_server.py`), the SPA's path via the control's
`/data_plane/*` proxy:

```
GET    /api/rois/{array_id:path}?set=
POST   /api/rois/{array_id:path}        body: {"rois": [RoiAnnotation…], "check_rev": bool}
DELETE /api/rois/{array_id:path}?ids=a,b&set=nuclei
```

`/api/rois/*` is its own namespace, so it does not collide with the greedy
`/api/sources/{source_id:path}` catch-all. The mutating routes call
`ctx.check_token(request)` then `_require_same_origin(request)`, following
`PUT /api/config`. Filling `drawn_against_version` is the caller's job, not the
sidecar's: the SPA already holds the tensor's descriptor, and doing it here would
cost a describe round trip on every save. JSON bodies are canonical proto3 JSON
(`google.protobuf.json_format` server-side, protobuf-es `toJson`/`fromJson` in the
SPA) so one schema serves both ends and neither hand-writes a DTO. The handler strips the
`@version` token wherever one can enter — the path *and* each annotation's own
`array_id` in the body — because responses carry versioned ids, so a read-edit-write
round trip hands one straight back and the store only ever sees bare ids.

## Persistence

The whole catalog is file-backed: `MetadataDatabase(store_path=...)` opens a
DuckDB file instead of `":memory:"`. `sources` comes along for the ride because
DuckDB has one database per connection and the sandbox blocks `ATTACH` — but it
is truncated on open, so the only content the file really carries is the
annotations. They are the only rows here no rescan can reproduce.

Reloaded rows re-attach with no fixup: `array_id` is deterministic across
restarts for file-backed sources (`generate_source_id` is a SHA-256 of the
resolved path). The exception is upload/scratch sources whose URL is synthesized
per run.

**Why not a separate SQLite mirror**, keeping the catalog in memory. It was the
first plan, on the theory that authored data deserves its own file in a format
that outlives DuckDB versions. Two of the three arguments for it turned out to
be wrong on measurement: a durable single-row write is 1329 µs in SQLite against
2317 µs in DuckDB (both just fsync), and the multi-process case does not exist —
a Flight server is a singleton with respect to one set of data. What survives is
file-format longevity, which is insurance rather than function, and the escape
hatch stays open: an exporter is additive and needs no migration.

### Three things happen on open

- **`TRUNCATE sources`.** It is scan output, and nothing prunes rows for a
  source deleted while the server was down; a stale row would advertise a tensor
  no read path can serve. Discovery repopulates it.
- **`shape_kind` and `bbox` are recomputed** from each row's stored `geometry`.
  Those two are `_roi_bbox` output cached in columns; `geometry` is the
  annotation. Deriving them again is what lets the formula change without a
  migration — adding `rotation` to `Ellipse` (biopb#935) moved every rotated
  ellipse's box, and this is the difference between that landing on restart and
  it needing a backfill. A row whose geometry will not parse keeps its stored
  values rather than being dropped.
- **`sources` is dropped and recreated**, not truncated. It is scan output, so
  rebuilding it also keeps its columns current and exempts it from schema
  versioning entirely.
- **A file that will not open, or whose `rois` shape this build does not
  understand, is fatal** (`AnnotationStoreError`), after a short retry. See
  below — these are the parts of the on-open pass that are policy rather than
  mechanism.

### Which file

Nothing at all when `annotations.enabled` is false. DuckDB's lock is exclusive,
so a server holding the catalog open would block `prune-annotations` and every
other reader for a feature it is not serving — and, since an unopenable store is
fatal, could refuse to start over annotations it was told not to serve. Being
disabled drops `rois` from the SQL surface too: empty rows would be the wrong
answer, because the table is unserved rather than unpopulated and a result set
cannot say which.

Otherwise `annotations.store_path` when set, else `state_dir()/catalogs/<digest
of the resolved config path>.duckdb`. A **relative** `store_path` anchors on the
config file's directory, never on the cwd: a server is started by the control
plane, by systemd, or by hand from wherever the user was standing, so a
cwd-relative store would mean one config silently naming a different catalog per
launch — and the one place it appears to work is the developer's own shell.
(`SourceConfig.local_path` still has this bug for source urls: biopb#947.) Keyed by config path because the singleton is
per data set: two servers on two `biopb.json` files are ordinary, and a shared
file would have them take turns clearing each other's `sources`. A server
started with no config file has nothing to derive a name from and stays in
memory, which is also what `annotations.persist = false` selects.

State tree, not cache: `cache_dir()` is documented as safe for a janitor to
empty, and half this file is not.

### Schema versioning

Persistence is what created this problem. The schema used to be rebuilt every
boot, so changing it cost nothing; now the file outlives the code, and
`CREATE TABLE IF NOT EXISTS` against an older file is a **silent no-op** — the
server starts, reports SERVING, and every annotation read and write then fails
on a missing column.

The split is the same one that runs through the rest of this design:

- **`sources` is not versioned at all.** It is scan output, so it is dropped and
  recreated on open. A change to its columns needs nothing.
- **`rois` carries `_ROI_SCHEMA_VERSION`**, stamped in a `catalog_meta` key/value
  table (not in `ALLOWED_TABLES`, so the SQL surface cannot reach it). Older
  files run the `_ROI_MIGRATIONS` ladder; a file from a **newer** build is
  refused rather than misread; a version with no migration to reach it is
  refused rather than skipped. A `rois` table with no marker is version 1 by
  definition — the marker shipped in the same release.

**The expected column set is built from `_ROIS_DDL` in a throwaway in-memory
database**, not written out by hand. The version marker only helps if every
schema change remembers to bump it, and the edit that forgets is the same edit
that would not have updated a hand-written list either. Deriving it means a
forgotten bump surfaces as a refusal at startup instead of a `Binder Error` on
the first annotation.

### A store that will not open is fatal

Neither of the two tempting recoveries is safe.

**Not "rename it aside and start clean."** DuckDB raises the same `IOException`
for a corrupt file and for one another process holds, so the two are
indistinguishable at the point of decision — and guessing wrong on a lock is the
worse error by a distance. The rename *succeeds* while the other server has the
file open; it keeps writing to the renamed inode, this one starts a fresh
catalog at the original path, and the annotations split across two files with
nothing anywhere to say so. That is the exact failure the per-config path exists
to prevent, reintroduced by the error handler.

**Not "fall back to memory."** `annotations.persist` is a promise about
durability. Serving anyway keeps the server up while every ROI drawn on it goes
to a catalog that vanishes at the next restart — loss discovered a day later,
with the work already gone. A health flag does not fix that; nobody reads health
before they start tracing.

So the server refuses to start, with a message naming the file and the four
things it can be. All four are decisions for a person — restore the file, fix
the permissions, match the DuckDB version that wrote it, or stop the other
server — and the operator who genuinely wants a session-only store says
`annotations.persist = false`, which is not an error at all.

**The retry comes first** (3 attempts, 0.5 s apart), and it is the only
distinction available between the four: a lock held by a server on its way down
clears within a second, and a restart race is the one open failure that resolves
itself. Nothing else does.

`health` reports `annotations_persisted` — false only for a deliberately
session-only server, since the accidental case cannot start. A client can say so
before someone spends a morning tracing.

## Staleness

### Two kinds of stale

**Content staleness** — the tensor still exists but its pixels changed under the
annotation. Already handled: `drawn_against_version` holds the `content_version`
observed at write time, so a client compares and warns. Nothing to prune.

**Referential staleness** — the tensor is gone, and the rows point at nothing.
This is what a persisted table accumulates, and what needs a policy.

### There is no existence oracle

Two tempting rules both fail.

**"Delete rows whose `array_id` is not in `sources`."** The catalog is routinely
an incomplete picture of what exists: discovery is progressive (`SERVING` does
not imply a complete catalog — that is why `health` publishes
`full_scan_in_progress` and `last_full_scan_finished_at`), an unmounted drive or
an unresolved cloud source is absent and entirely intact, and a rescan's
unregister/re-register window is absence too. A sweep racing the first scan
deletes everything.

**"Then stat `source_url` and prove it is gone."** Only file-backed sources have
a path to stat. A remote tensor-server proxy source, an `s3://` or synced-folder
source, an upload/scratch source with a synthesized URL — none of them are on a
local filesystem, and probing a proxy means an RPC to an upstream whose being
down is ordinary operation, not evidence of deletion. Building a per-source-type
existence probe would mean an oracle for every adapter, each with its own
unknown-vs-gone ambiguity, and no adapter instance even exists for a source the
catalog has dropped.

So: **never assert deletion.** Record presence when it is observed, and let
absence be measured in elapsed time rather than judged.

### The orphan clock

`MetadataDatabase.mark_sources_seen()` — one set-based statement:

```sql
UPDATE rois SET source_url = COALESCE(NULLIF(s.source_url, ''), rois.source_url),
                last_seen_at = now()
  FROM sources s WHERE rois.source_id = s.source_id;
```

Matching on `source_id`, not `array_id`, is deliberate: an unresolved cloud
source is in the catalog with an **empty** `tensors` list, so an array_id join
would score its annotations as unseen while the source is sitting right there.

That is the whole mechanism — no I/O, no per-adapter probe, uniform across every
source type, and it degrades correctly: a drive offline for a week simply does
not advance `last_seen_at`. It also refreshes `source_url` — backfilling one a
write could not resolve, and following a source whose display url moved without
its identity moving (an `alias` re-root, a `dnd://` stamp).

**The seam is scan completion, not a timer.** `SourceManager._mark_catalog_complete()`
is where the three paths that can finish a full scan converge — a forced full
rescan, an upstream re-list pass, and a static-only config with nothing to walk.
The gate the sweep needs (`full_scan_in_progress == false` plus a
`last_full_scan_finished_at` from this process) is that method's postcondition,
so it is held rather than checked; a separate long-interval worker would be
re-deriving an edge from a level and could not tell "complete and fresh" from
"complete an hour ago, and two mounts have dropped since". The cadence comes
free: `full_rescan_interval`, an hour by default.

### Deleting, which is a separate decision

Deletion is a policy over age (`annotations.prune_unseen_days`, default `0` =
never) rather than a claim about the world. `prune_unseen(before)` applies it and
`unseen_rois(before)` reports what it would take, per tensor and named by
`source_url` — one predicate, so a dry run and the real thing cannot drift.

Both are exposed on demand, because the automatic path is off by default and,
when on, will not fire until the server has been up for the whole threshold.

**`biopb tensor prune-annotations --days N`** is the one to reach for. It needs
nothing the server does not already expose: `rois` is in `ALLOWED_TABLES`, so
the SQL surface reads it, and `roi_delete` takes explicit ids — two ordinary
client calls against a *running* server, through the usual `_data_plane`
endpoint and credential resolution. Deleting by id, per tensor, means a delete
issued from a report can only ever remove what the report listed.

**`biopb-tensor-server prune-annotations <config> --days N`** is the same thing
against the file, for when there is no server to dial. It **requires the server
to be stopped**, which is worth stating because the intuition runs the other
way: DuckDB's lock on the catalog is exclusive for *readers* as well as writers
— `read_only=True` is refused too — so nothing can open the file while the
server has it. The command recognises that case and says so.

Both report unless given `--apply`.

Two conditions in `_mark_catalog_complete` are load-bearing:

- **The sweep runs before the delete, in the same pass.** `last_seen_at` does
  not advance while the server is off, so after a week down every annotation
  looks a week unseen; deleting first would take out rows whose images are
  sitting right there.
- **Deleting arms only once this process has been up longer than the
  threshold.** The sweep is the only thing that advances `last_seen_at`, so a
  row can be *known* unseen for N days only if the server has been running the
  sweep for N days; anything shorter reads a gap the server was not present for
  as evidence about the world, and a restart is when that gap is largest.

  Gating on the first scan instead — the obvious version — is not enough. It
  assumes whatever makes a source visible is reachable at boot, and a proxy
  upstream that is down then is usually still down an hour later, so the second
  completed scan would delete its annotations. Uptime does not care why a source
  was absent.

  The cost is that a server restarted more often than the threshold never
  auto-prunes. That is the intended trade, and it is why the reporting path
  exists: a person can act on `unseen_rois` whenever they like.

The clock is `time.monotonic()`, so an NTP correction cannot age the server into
deleting.

Age is `COALESCE(last_seen_at, created_at)`. A row written before its source
ever reached the catalog has no sighting to measure from, and reading that as
"infinitely fresh" would make the strongest orphan the one row a prune can never
reach.

Both catalog-derived columns are written **only when the catalog answered** —
and a write states that structurally rather than by reconstruction. A create is
an `INSERT`; an update is an `UPDATE` naming only the columns a client owns, so
identity, `created_at`, `source_url` and `last_seen_at` are preserved by *not
being mentioned*. (A full-row `INSERT OR REPLACE` had to re-derive all sixteen
columns on every write, and an edit made while the source was briefly absent
duly wiped a good `source_url` and stamped `last_seen_at` as though the tensor
had just been seen — resetting the orphan clock for an image that may really be
gone.)

Presence is recorded once per call, for the whole source:

```sql
UPDATE rois SET source_url = COALESCE(?, source_url), last_seen_at = ?
 WHERE source_id = ?;
```

The catalog's answer wins, which both backfills a URL that is still NULL —
closing the window where annotations were written before discovery caught up —
and follows a rename. A catalog row that names nothing (NULL, or the `""` the
descriptor path writes for a source with no url of its own) leaves the stored
label alone: an unnamed source is not a rename. And this is the *same statement
the prune sweep runs*: the sweep adds only the catalog-completeness gate, which a
presence observation does not need (only a conclusion about absence does).
Absence writes nothing at all.

Because the write happens **only on presence**, the column freezes by itself at
the last sighting — the name to show for a source that is gone. It used to
freeze at the *first*, so a source renamed and later unmounted was reported under
a name the catalog had abandoned long before it went away, at the one moment the
column is load-bearing. Nothing matches on `source_url` (both statements join on
`source_id`), so refreshing it cannot change which rows are stamped or when.

A write against a source the catalog does not know is still stored — refusing it
would turn a rescan window into lost work, and absence proves nothing. It lands
with a NULL `source_url`, which is precisely the unreportable row above; the
sweep, or the next write that can resolve the URL, backfills it.

Auto-delete stays **opt-in** — `prune_unseen_days` defaults to 0. These are
hand-drawn user data; the safe default is to surface orphans, sorted by
`last_seen_at` and named by `source_url`, and let a person confirm. `source_url`
is stored for exactly this: `array_id` is a SHA-256 and cannot be inverted, so
without it an orphan report can only say "annotations for `zarr_a3f2b1c4`",
which no one can act on — and a row written before its source was ever in the
catalog is the one orphan nobody can be told about.

### Known limitation: a move orphans annotations

Identity is path-derived, so `mv` gives a file a new `source_id`, a new
`array_id`, and a tensor that appears freshly unannotated while the old rows go
orphan. Pruning is not the fix — re-attachment is, and it is a separate problem.
The stored `source_url` plus `drawn_against_version` (`mtime_ns:size`, which
survives a plain `mv`) is enough to offer "these annotations were drawn on a file
of the same name, size and mtime — re-attach?" rather than silently losing them.

A **proxied** source is the deliberate opposite: its id is built from
`(alias, upstream_source_id)` and carries no endpoint, so moving the upstream
leaves annotations attached. The equivalent event there is renaming the `alias`,
which is part of the identity by design — see *remote-tensor-cache.md*.

## Implementation order

1. `annotation.proto` + `buf generate` (Python and Java only — `buf.gen.yaml`
   emits no TypeScript, and the SPA hand-writes its JSON codec rather than
   generating one; see `docs/roi-annotations-ui.md`).
2. `MetadataDatabase`: `rois` table, `put_rois` / `list_rois` / `delete_rois`,
   bbox derivation, shape-arm rejection, `ALLOWED_TABLES += {"rois"}`; unit tests
   against the DB alone.
3. Three Flight actions + `TensorFlightClient` methods; round-trip test.
4. Three sidecar routes; test through the app with a mocked client.
5. Config: `annotations.enabled`, `annotations.max_rois_per_tensor` in
   `config_schema.py`. (`source_url` is written from step 2, since rows without
   it can never be reported or re-attached.)
6. Docs: this file linked from `docs/http-server.md` and `ARCHITECTURE.md`.
7. Persistence: `annotations.persist` / `annotations.store_path`, the file-backed
   connection and its on-open pass.
8. The orphan clock: `mark_sources_seen` / `unseen_rois` / `prune_unseen`, driven
   from `SourceManager._mark_catalog_complete`. No UI yet — `unseen_rois` is the
   query an admin route or `biopb roi prune --dry-run` would render.
