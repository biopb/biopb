# ROI annotations — backend design

Lets a user draw 2-D ROIs (points, polygons, rectangles, ellipses) on a tensor in
the viewer SPA and have them persist for the life of the server process. The store
is a new table in the tensor server's DuckDB catalog, reached over a small Flight
action set and re-exposed by the HTTP sidecar.

Scope of this doc: the backend (store, wire types, API surfaces). The SPA's draw
tooling is a separate design.

## Non-goals

**Instance segmentation is a separate tensor, not an annotation set.** A
segmentation's objects belong in a label tensor the server already serves as
pixels — not as 10⁴–10⁶ rows in an in-memory catalog. This is the load-bearing
scope decision: it is what lets the read path be a single whole-set fetch over
DoAction, keeps the per-tensor cap human-scale, and confines the store to the
2-D vector shapes. If bulk objects ever need a home here, revisit storage and
channel together rather than growing this table.

Also out: undo/history, cross-tensor annotation search, and export to
OME-XML / GeoJSON (the geometry is proto, so an exporter is additive).

## Model

An **annotation** is a geometry plus the metadata that makes it findable. The
geometry is `biopb.image.ROI` (`proto/biopb/image/roi.proto`), reused as-is —
TypeScript is already generated for the SPA
(`web/.../gen/biopb/image/roi_pb.ts`) and Python exports it from `biopb.image`.
`ROI` carries geometry *only* — no id, no label — so the record around it is new
(`RoiAnnotation`, below).

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
gets that back from `set_name` plus a batched `roi_put` applied in one
transaction. Per-set attributes (display colour, visibility) have no table yet on
purpose — they are client display state today, and a `roi_sets` table is additive
if they ever need to be shared.

**Anchor to `array_id`, unversioned.** A tensor is identified by `array_id` alone
(see the identity policy in `descriptor.proto`), so that is the foreign key —
not `source_id` (a multi-field source has many tensors) and not the sidecar's
`source@token/field` HTTP form, which changes whenever the file's
`content_version` changes. Annotations must outlive an in-place edit of the
image, so the sidecar strips the version token before the Flight call exactly as
the tile route does (`_split_array_version`). What *is* stored is the
`content_version` observed at write time (`drawn_against_version`), so a client
can say "this image changed since it was annotated" instead of silently drawing
stale outlines.

**Coordinates are level-0 pixels.** Always full-resolution, in the tensor's own
Y/X axes, floating point. A polygon drawn on a downsampled level is scaled up by
the client before the write. The server never rescales geometry.

**Plane pinning is a sparse map, not t/z/c fields.** A 2-D ROI lives on one plane
of an N-D tensor, and `dim_labels` are per-tensor, so the pin is
`map<dim_label, index>` — `{"z": 12, "t": 0}`. A missing key means "applies at
every index of that dimension", which is how a user gets an ROI that follows a
z-stack or a time course without duplicating it per plane.

**Annotations are not source metadata.** They do not go in `sources.metadata_json`
and are not merged into `GET /api/sources/{id}/metadata`. That column is
adapter-produced and rewritten by the `INSERT OR REPLACE` in `sync_source_added()`
on every re-registration, so an annotation parked there would be destroyed by the
next rescan. "Read as metadata" here means a sibling table in the same catalog DB,
queryable next to `sources` — not a field inside a source row.

## Schema

New table in `MetadataDatabase._create_schema()`:

```sql
CREATE TABLE rois (
    roi_id     TEXT NOT NULL,           -- uuid4 hex; client-supplied or server-minted
    array_id   TEXT NOT NULL,           -- the anchor, unversioned
    source_id  TEXT NOT NULL,           -- array_id split on the first '/'; joins + authz
    source_url TEXT,                    -- catalog URL at write time; names the image in an
                                        -- orphan report, and anchors re-attach after a move.
                                        -- NULL only until the source is first seen
    set_name   TEXT NOT NULL DEFAULT 'default',
    label      TEXT,                    -- user class/name
    shape_kind TEXT NOT NULL,           -- point|rectangle|ellipse|polygon|polyline
    plane      MAP(VARCHAR, BIGINT),    -- sparse pin; absent key = all indices
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
  map<string, int64> plane = 6;   // dim_label -> index; absent = all
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

## Persistence and staleness

The catalog is `duckdb.connect(":memory:")`, so annotations die with the process.
Accepted for now — and worth noting that while that holds, **there is no
staleness problem at all**: the table cannot outlive the catalog it references.
Everything below is the cost of closing the gap, not work for the first version.

Three things keep the fill-in cheap:

- every write funnels through three `MetadataDatabase` methods, so a write-behind
  has exactly three call sites;
- the row is flat and serializable — a per-catalog `rois.jsonl` or a file-backed
  DuckDB attachment stores it verbatim;
- `array_id` is deterministic across restarts for file-backed sources
  (`generate_source_id` is a SHA-256 of the resolved path), so reloaded rows
  re-attach to the same tensors with no fixup. The exception is upload/scratch
  sources whose URL is synthesized per run.

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

### Pruning, when it is needed

One set-based statement, run on a long interval and only while the catalog is
known-complete (`full_scan_in_progress == false` and a
`last_full_scan_finished_at` from this process):

```sql
UPDATE rois SET last_seen_at = now()
 WHERE source_id IN (SELECT source_id FROM sources);
```

Matching on `source_id`, not `array_id`, is deliberate: an unresolved cloud
source is in the catalog with an **empty** `tensors` list, so an array_id join
would score its annotations as unseen while the source is sitting right there.

That is the whole mechanism — no I/O, no per-adapter probe, uniform across every
source type, and it degrades correctly: a drive offline for a week simply does
not advance `last_seen_at`. Deletion is then a policy over age
(`annotations.prune_unseen_days`, default `0` = never) rather than a claim about
the world.

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
UPDATE rois SET source_url = COALESCE(source_url, ?), last_seen_at = ?
 WHERE source_id = ?;
```

`COALESCE` backfills a URL that is still NULL — closing the window where
annotations were written before discovery caught up — while leaving an existing
one alone. And this is the *same statement the prune sweep runs*: the sweep adds
only the catalog-completeness gate, which a presence observation does not need
(only a conclusion about absence does). Absence writes nothing at all.

A write against a source the catalog does not know is still stored — refusing it
would turn a rescan window into lost work, and absence proves nothing. It lands
with a NULL `source_url`, which is precisely the unreportable row above, and the
next write that *can* resolve the URL backfills it.

Auto-delete stays **opt-in** — the default above is off. These are hand-drawn
user data; the safe default is to surface orphans, sorted by `last_seen_at` and
named by `source_url`, and let a person confirm via an admin route or a
`biopb roi prune --dry-run`. `source_url` is stored for exactly this: `array_id`
is a SHA-256 and cannot be inverted, so without it an orphan report can only say
"annotations for `zarr_a3f2b1c4`", which no one can act on. **That is the part
that must be decided now** — rows written today without a URL can never be
reported or re-attached later.

### Known limitation: a move orphans annotations

Identity is path-derived, so `mv` gives a file a new `source_id`, a new
`array_id`, and a tensor that appears freshly unannotated while the old rows go
orphan. Pruning is not the fix — re-attachment is, and it is a separate problem.
The stored `source_url` plus `drawn_against_version` (`mtime_ns:size`, which
survives a plain `mv`) is enough to offer "these annotations were drawn on a file
of the same name, size and mtime — re-attach?" rather than silently losing them.

## Implementation order

1. `annotation.proto` + `buf generate` (Python, Java; the TS gen for the SPA is
   a separate `protoc-gen-es` run, not wired into `buf.gen.yaml`).
2. `MetadataDatabase`: `rois` table, `put_rois` / `list_rois` / `delete_rois`,
   bbox derivation, shape-arm rejection, `ALLOWED_TABLES += {"rois"}`; unit tests
   against the DB alone.
3. Three Flight actions + `TensorFlightClient` methods; round-trip test.
4. Three sidecar routes; test through the app with a mocked client.
5. Config: `annotations.enabled`, `annotations.max_rois_per_tensor` in
   `config_schema.py`. (`prune_unseen_days` lands with persistence, not now — but
   `source_url` is written from step 2, since rows without it can never be
   reported or re-attached.)
6. Docs: this file linked from `docs/http-server.md` and `ARCHITECTURE.md`.
