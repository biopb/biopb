# Retiring DataSourceDescriptor

Status: **implemented**, in two passes. Five coupled changes (#1025) finished
what Flight protocol v2 (#1018) started: the catalog row became the only
representation of a source on the wire, and `DataSourceDescriptor` an SDK-side
view built from rows. Then #1032 retired the message from that view too — §6.

## What v2 already changed

`list_flights` no longer builds a descriptor per source. It advertises one flight
per catalog table, each carrying that table's Arrow schema and a
`SELECT * FROM <table>` ticket; browsing sources is a catalog query.

The SDK followed: `list_sources` / `get_source` run SQL and build the descriptor
**client-side** from the row (`_session.py` → `_catalog_rows.py`).

What did not follow is everything below. Each item was v1 residue that worked.

## 1. The two encodings were already identical

`catalog_entry()` projects a tensor onto exactly
`{array_id, dim_labels, shape, dtype}`, and `get_source_descriptor` set
`metadata_json=""`. `descriptor_from_row` builds exactly those four per tensor,
plus `source_id` / `source_url` / `source_type` / `data_resident`, and hardcodes
`metadata_json=""`.

So the proto `resolve` put on the wire carried **precisely the row's
information**. It was a second encoding of the same projection.

## 2. `resolve` and `add_source` return rows

Both actions returned descriptors: `ResolveStreamMessage.result` and
`AddSourceResult.added`.

The row already exists when either returns. Resolution's `on_resolved` callback
(`adapters/unresolved.py` → `sources/reconciler.py`) calls `sync_source_added`,
which overwrites the source's NULL shape/dtype row with the concrete one;
`_catalog_sync_added` does the same before the add tally is assembled. The
server built a *second* representation from the adapter instead of returning the
one it had just written.

**`resolve` returns the row**, as an Arrow IPC stream of one `sources` row with
`SOURCE_ROW_COLUMNS` (`ResolveStreamMessage.source_row`). Each SDK decodes it
with the same `descriptor_from_row` / `descriptorsFromRows` that backs
`list_sources`, so the public return type is unchanged and the proto stops being
a wire type.

That also removes a divergence: the adapter answers live (`is_resident()` is
documented VOLATILE, "evaluate at the moment of use and never cache") while the
row is a snapshot, so `resolve()` then a browse could report different
`data_resident` for one source. Returning the row makes that impossible.

Resolve's *reason* for returning anything stays valid: an unresolved source has
an empty `tensors` list, and the post-resolution enumeration is not otherwise
obtainable without hitting the browse cap.

**`add_source` returns source_ids** (`AddSourceResult.added`, now
`repeated string`), consistent with its `already_present` / `refreshed` /
`removed` siblings. The descriptors it used to carry were read by nobody: the
one consumer, the tensor browser's drop worker, took `len()`. Registration has
already written each row, so anything more is one `query_sources` away.

**A server that owns its catalog re-syncs on resolve.** The backfill above is
the SourceManager's `on_resolved`; a server holding its own catalog has no such
callback wired, and would return the unresolved placeholder row it wrote at
registration. `_handle_resolve` closes that, and only when the adapter actually
hydrated — re-syncing an already-resolved source would re-parse metadata that
`release_registration_cache` has dropped.

## 3. `get_source_descriptor()` is gone

It had three callers, none of which needed a descriptor: `resolve()`,
`_descriptor_for` → `AddSourceResult.added` (both answered by §2), and
`metadata_db.sync_source_added` — which never did. The upsert read only
`.source_url`, `.source_type`, `.data_resident`, `.tensors` and destructured them
straight back into columns; all four are on the adapter already. It built a
proto because v1's listing path had one.

So the adapter surface is the four reads: `catalog_url` (new, the display form
of `source_url` that `_catalog_url` used to override inside the descriptor),
`source_type`, `is_resident()` and `list_tensor_descriptors()`. `resolve()`
returns nothing.

**The invariant moved with it.** `get_source_descriptor` was the enforcement
point for #812 (no `chunk_shape` on a catalog entry) via its `catalog_entry()`
map. `catalog_tensors(adapter)` is that point now, and is the one path into the
row's `tensors` column.

## 4. `list_sources` / `get_source` are deprecated, and the SDK keeps no catalog

Both are wrappers around `query_sources` that inherit the server's query row cap
— so a browse, which is exactly where the cap matters, came back silently
truncated. The repo's own guidance already steered away from them. They still
work; they warn, and point at `query_sources` plus a row decoder (now exported,
in both SDKs, as the migration path).

`_state.sources` is gone. It was a cache, not state — every read fell back — and
both first-party consumers keep their own (`TensorConnection.sources`, the SPA's
`useAppStore.sources`). It also changed semantics rather than just caching:
`get_physical_scale` raised the unresolved error **only if the source happened to
be cached**. That check is now the server's, on the probe the call already
makes, so it holds for every id.

`_state.descriptors` stays: the per-array_id structural cache from GetFlightInfo
(#795) is addressing facts, a different thing.

Migrated consumers: the sidecar's `GET /api/sources`, `GET /api/sources/{id}`
and `_tensor_candidates` (which listed the *whole* catalog to name alternatives
in one 404); the MCP `TensorConnection`'s snapshot; and Java's `getTensor`
resolution path, which now does one addressed row query. `RemoteTensorAdapter`'s
`list_sources()` fallback was deleted rather than migrated — it fired when
`query_sources` failed, and in v2 `list_sources` *is* that query, so it could
only fail identically.

## 5. `get_source_metadata` reads the column

The name was right; the call was a workaround. It resolved `tensors[0].array_id`
and issued a tensor-bound GetFlightInfo — but the server fills `metadata_json`
from `get_metadata_json(source_id)`, a lookup **by source_id**. The array_id
contributed nothing.

`get_metadata()`'s contract already said so: called once at registration to
populate `sources.metadata_json`, and "the serve path reads it back from the
catalog, never by recomputing here" (#253).

**This one returned a wrong answer, not just a slow one.** After the
source-level row, the server overlays `tensor_adapter.get_tensor_metadata()` — a
per-tensor delta (an OME-Zarr HCS field's own OME block, an EMD signal's
`original_metadata`). On a multi-field source, "the source's metadata" came back
contaminated with one arbitrary field's extras.

It now reads `sources.metadata_json` via a one-row `query_sources`. The
per-tensor overlay stays where it belongs, on a tensor-bound read's
`with_metadata`.

## 6. The SDK-side view is no longer a proto either (#1032)

§1–5 left the message off the wire but kept it as what each SDK decoded a row
*into*. That was the expensive half. The message is generated code, so every
catalog column a client wanted cost a `.proto` edit and a `buf generate` across
every binding — for a struct no message carries. `is_resolved` (#1030) made the
price concrete: one boolean, a full codegen cycle.

The "one schema, every language" argument for paying it was already void.
TypeScript never shared the type — `web/packages/tensor-flight-client/src/types.ts`
hand-rolls the same shape — and `http_server.py`'s `/api/sources` decodes rows to
plain dicts. Two of three consumers had already converged on a hand-written
struct.

The fix is not a nicer struct — it is **no struct**. Replacing a proto the SDK
picked with a dataclass the SDK picked keeps the shape of the mistake, just
cheaper: the next field still lands in a type the caller did not choose. So
the SDK stops choosing. `query_sources` already answers in `records` / `arrow`
/ `pandas`; `resolve` returns the single row it just wrote, in the same
`records` shape (Java: a `VectorSchemaRoot`, the type `querySources` already
returns). Every client can decode a row — that is the one thing they all
share — and what they decode it into is theirs.

biopb-mcp is the worked example: it wants attribute access for a Qt widget
that reads these fields across a few hundred lines, so it defines its own
frozen dataclass in `_catalog.py`. That is a caller's choice, made in caller
code, which is exactly what the original design intended.

**The proto decoders stay, deprecated.** `descriptor_from_row` /
`descriptors_from_rows` / `descriptorsFromRows` keep their names, their
signatures and their exact output, as do `list_sources` / `get_source` — the
two stable APIs that return `DataSourceDescriptor`. Nothing an SDK user calls
changes shape under them; the deprecation is the message, and what it says is
that this path cannot grow. There is deliberately **no replacement decoder to
point at** — the row is the data structure, and the warning says so.

**`resolve()` does change, because it can.** Cloud support is experimental, so
it returns the row now — a `records`-shaped dict in Python, a
`VectorSchemaRoot` in Java. The question it raises is why it does not mirror
`warm()`, which returns a terminal `WarmProgress`. The two look symmetric and
are not:

- `warm` changes **residency**, which is deliberately not a durable catalog
  fact (#1035). Its `files_total` / `files_done` / `bytes_done` exist nowhere
  else — no query recovers them — so returning them is the only way they reach
  a client. `files_total == 0` is load-bearing on exactly that basis: it is
  how a caller learns the source had nothing to warm.
- `resolve` changes **`is_resolved` and `tensors`**, which are columns. Its
  result is durable by construction, so the return is a shortcut to data that
  now exists rather than the only channel to it.

And a terminal `ResolveProgress` would carry `elapsed_seconds` (the caller can
time it) plus `target_name` / `target_bytes` (derivable from `source_url`) —
the reconstructible half — while dropping the tensor list, the half that is
the point. That is backwards from `warm`, where the returned counts are the
irreplaceable part. So: a status where there is only a status, a result where
there is a result.

The nearest real alternative was `resolve() -> None`, matching §2's
`add_source`: the row is written, and one addressed `query_sources` away.
Returning it wins on ergonomics — the caller's next move is to read the
tensors — and it closes the transition in one call, with no window in which a
rescan re-registers the source. (§2's stated reason, that the enumeration is
"not otherwise obtainable without hitting the browse cap", is not the right
one: `WHERE source_id = ...` returns a single row and no cap ever bites.)

**Absent beats empty.** §1 noted `descriptor_from_row` hardcoded
`metadata_json=""`, and §3's invariant is that a catalog entry carries no
`chunk_shape`. A proto could only express those as *empty*: present, testable,
and permanently meaningless. A row simply has no such column, which deleted an
unreachable `if tensor_desc.chunk_shape:` branch in the napari widget and
turned several tests from "asserts empty" into "asserts the column is not
there".

**And it fixed a bug.** With no `is_resolved` to read, both the napari
`tensor_browser` (`_is_unresolved`) and the Java `getTensor` path inferred
"unresolved" from `len(tensors) == 0` — which also describes a source that
resolved cleanly and held nothing readable. Both now read the flag, and both say
something different for the two cases. The `biopb tensor query` CLI prints
`<unresolved>` rather than `<no tensors>` for the first.

Each moved off the deprecated decoder because it needed the field, which is the
only reason to move — and each picked its own landing place. The CLI reads the
row's keys directly; biopb-mcp decodes into its own dataclass;
`TensorConnection.resolve_source` decodes `resolve()`'s row the same way, so
the widget's next decision — whether to warm — reads the flag off the type that
package chose.

The proto message stays in `descriptor.proto`, unused: removing it would break
codegen for consumers outside this repo, and it costs nothing to leave.

## Wire compatibility

Deliberately breaking. The SDK refuses a pre-v2 server at connect, so the v1
fallbacks in both clients (a bare serialized descriptor as the resolve terminal;
the empty-body heartbeat convention) went with it. Field numbers 2
(`ResolveStreamMessage.result`) and 1 (`AddSourceResult.added`) are reserved.

§6 changes no wire bytes. It deprecates a decoder without adding one, and
leaves every stable signature alone; the single return-type change is
`resolve()`, which cloud support's experimental status allows. It does add
`is_resolved` to `SOURCE_ROW_COLUMNS`, so an SDK from after #1032 against a
server from before #1033 fails the SELECT rather than degrading; that is the
same coupling `data_resident` already had.
