# Retiring DataSourceDescriptor from the wire

Status: **proposed** — nothing below is implemented. Five coupled changes that
finish what Flight protocol v2 (#1018) started: the catalog row becomes the only
representation of a source on the wire, and `DataSourceDescriptor` becomes an
SDK-side view built from rows.

## What v2 already changed

`list_flights` no longer builds a descriptor per source. It advertises one flight
per catalog table, each carrying that table's Arrow schema and a
`SELECT * FROM <table>` ticket; browsing sources is a catalog query. Nothing in
`serving/server.py` calls `get_source_descriptor()` any more.

The SDK followed: `list_sources` / `get_source` run SQL and build the descriptor
**client-side** from the row (`_session.py:471,485` → `_catalog_rows.py`).

What did not follow is everything below. Each item is v1 residue that still works.

## 1. The two encodings are already identical

`catalog_entry()` (`core/adapter_base.py:161`) projects a tensor onto exactly
`{array_id, dim_labels, shape, dtype}`, and `get_source_descriptor` sets
`metadata_json=""`. `descriptor_from_row` builds exactly those four per tensor,
plus `source_id` / `source_url` / `source_type` / `data_resident`, and hardcodes
`metadata_json=""`.

So the proto `resolve` puts on the wire carries **precisely the row's
information**. It is a second encoding of the same projection.

## 2. `resolve` and `add_source` should return rows

Both actions return descriptors today: `ResolveStreamMessage.result`
(`serving/server.py:977`) and `AddSourceResult.added` (`:1223`, filled from
`sources/source_manager.py:1359`).

The row already exists when either returns. Resolution's `on_resolved` callback
(`adapters/unresolved.py:241` → `sources/reconciler.py:870`) calls
`sync_source_added`, which "overwrites the source's NULL shape/dtype row with the
concrete descriptor"; `_catalog_sync_added` (`serving/server.py:506`) does the
same before the add tally is assembled. The server builds a *second*
representation from the adapter instead of returning the one it just wrote.

**Decision.** Both return Arrow rows with `SOURCE_ROW_COLUMNS`. The SDK feeds them
to `descriptor_from_row`, so the public return type is unchanged — the proto stops
being a wire type.

This also removes a divergence: the adapter answers live (`is_resident()` is
documented VOLATILE, "evaluate at the moment of use and never cache"), the row is
a snapshot, so `resolve()` then `list_sources()` can report different
`data_resident` for one source. Returning the row makes that impossible.

Resolve's *reason* for returning anything stays valid: an unresolved source has an
empty `tensors` list, and the post-resolution enumeration is not otherwise
obtainable without hitting the `list_sources` cap (`_session.py:842`).

## 3. `get_source_descriptor()` then has no callers

Three today:

| Caller | Needs a descriptor? |
|---|---|
| `resolve()` (`core/adapter_base.py:514`) | no, once §2 lands |
| `_descriptor_for` → `AddSourceResult.added` | no, once §2 lands |
| `metadata_db.sync_source_added:1148` | **never did** |

The upsert reads only `.source_url`, `.source_type`, `.data_resident`, `.tensors`
and destructures them straight back into columns. All four are on the adapter
already. It built a proto because v1's listing path had one.

**Decision.** Delete `get_source_descriptor` and `_descriptor_for`.

**Carry the invariant with it.** `get_source_descriptor` is the enforcement point
for #812 (no `chunk_shape` on a catalog entry) via its `catalog_entry()` map, and
with ListFlights gone the upsert is its *only* remaining caller. The row's
`tensors` struct is built inline at `serving/metadata_db.py:1162` and does not
apply it. The projection must move there — e.g. a `catalog_tensors(adapter)`
helper — or the invariant evaporates silently.

## 4. Deprecate `list_sources` / `get_source`, drop the SDK store

Both are wrappers around `query_sources`. The repo's own guidance already steers
away from them: `biopb-mcp/.../mcp/_server.py:714` tells agents to prefer
`query_sources` because `list_sources` is "server-capped for large catalogs", and
the tensor browser switches to server-side query above
`SERVER_QUERY_THRESHOLD` (`_connection.py:326`, `tensor_browser/_widget.py:2384`).

`_state.sources` is a cache, not state — every read falls back
(`_session.py:577` re-lists, `:754` calls `get_source`). Both first-party
consumers keep their own: `TensorConnection.sources` (with its own documented
rebind-never-mutate threading contract) and the SPA's `useAppStore.sources`. It is
a third copy neither reads.

It also changes semantics rather than just caching: `get_physical_scale`
(`_session.py:633`) raises the unresolved error **only if the source happens to be
cached** — "Only catches sources already in the catalog cache; a never-listed id
still falls through". Same call, different behaviour depending on whether
`list_sources()` ran earlier.

**Decision.** Deprecate both; drop `_state.sources`; make the unresolved check
explicit. Keep `_state.descriptors` — the per-array_id structural cache from
GetFlightInfo (#795) is addressing facts, a different thing.

**Consumers to migrate.** Two in the sidecar — `http_server.py:1680`
(`GET /api/sources`, which feeds the SPA and the web client's `listSources()`) and
`http_server.py:660`, where `_tensor_candidates` lists the *whole* catalog to name
the alternatives in one 404 — and Java's `TensorFlightClient.listSources()`
(`TensorFlightClient.java:219`), a separate implementation with its own store.

## 5. `get_source_metadata` should read the column

The name is right; the call is a workaround. It resolves `tensors[0].array_id`
from the store and issues a tensor-bound GetFlightInfo — but the server fills
`metadata_json` from `self._metadata_db.get_metadata_json(source_id)`
(`serving/server.py:1388`), a lookup **by source_id**. The array_id contributes
nothing.

`get_metadata()`'s contract already says so (`core/adapter_base.py:429`): called
once at registration to populate `sources.metadata_json`, and "the serve path
reads it back from the catalog, never by recomputing here" (#253).
`RemoteTensorAdapter` already reads that column directly
(`adapters/remote_tensor.py:631`).

**This one returns a wrong answer, not just a slow one.** After the source-level
row, the server overlays `tensor_adapter.get_tensor_metadata()` — a per-tensor
delta (an OME-Zarr HCS field's own OME block, an EMD signal's
`original_metadata`) — and wraps it with field 0's `dim_label`. On a multi-field
source, "the source's metadata" comes back contaminated with one arbitrary
field's extras.

**Decision.** `get_source_metadata(source_id)` reads `sources.metadata_json` via
`query_sources`. The per-tensor overlay stays where it belongs, on a tensor-bound
read's `with_metadata`. This is what #253 said the scheme was.

Note this call is *simplified* by §4: its only uses of the store are
`tensors[0].array_id` and the empty-tensors unresolved check, both of which
become one SQL row.

## Open questions

1. **`AddSourceResult.added` shape.** Rows do not nest in a proto. Either `added`
   becomes an id list (consistent with `already_present` / `refreshed` /
   `removed`, but costs the caller a round-trip after a many-file drag-drop), or
   the result carries an Arrow batch beside the proto.
2. **Where `catalog_entry()` lives** once its current home is deleted (§3).
3. **Java parity** — same treatment, or an explicit "python first" note.

## Stale text to fix regardless

Three comments describe the path v2 already deleted:

- `core/adapter_base.py:491` — "every path into the DuckDB row **and into
  ListFlights** goes through here". The ListFlights half is gone.
- `core/adapter_base.py:495` — `metadata_json=""  # filled by GetFlightInfo()`.
  Nothing in `serving/` fills it on a descriptor; the column is written by the
  upsert.
- `tests/catalog_grid_ownership_test.py:95` — "the only path into the DuckDB row
  and into the **adapter-fallback ListFlights**". There is no adapter-fallback
  ListFlights in v2.

## Sequencing

§5 and the stale text are independent and can land first. §1–3 are one change
(the wire swap and the deletion are not separable). §4 is independent of §1–3 but
should follow §5, which removes two of the store's uses.

`serving/metadata_db.sync_source_added` duck-types its argument — "several
adapters supply that surface without inheriting the base"
(`serving/metadata_db.py:1188`) — so a replacement must stay duck-typed. Four test
stubs hand-define `get_source_descriptor` (`tests/test_metadata_db.py:33,185,218,854`).
