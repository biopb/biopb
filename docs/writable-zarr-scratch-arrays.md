# Plan: Writable, readable, ephemeral zarr arrays on the tensor server

Status: proposal
Scope: `biopb-tensor-server` (server + Python SDK client)
Related: `biopb/biopb#8` (read-grid decoupling), the upload facility
(`create_source` / `do_put` / `upload_status`).

## 1. Goal

Let a client create an **ephemeral scratch array** on the tensor server and
write values into it by region — i.e. treat a server-side source as a writable
*and* readable zarr array:

```python
z = client.create_array(shape=(64, 2048, 2048), dtype="uint16",
                        chunks=(1, 512, 512))   # -> RemoteZarr handle, source_id assigned
z[10, 0:512, 0:512] = plane                     # arbitrary-region write
patch = z[10, 0:256, 0:256]                      # random-access read-back (lazy dask)
z.delete()                                       # tear down the scratch array
```

The array must support arbitrary-region writes (including sub-chunk regions),
full random-access reads through the existing lazy-dask path, and a real
teardown — with no permanent artifact left behind.

## 2. Current state (what exists)

The server already exposes two server-created source primitives behind the same
`create_source` Flight action + `do_put` chunk upload + `upload_status`
lifecycle (`biopb-tensor-server/biopb_tensor_server/server.py:1006-1180`):

| | `cache:` (`CachedSourceAdapter`) | `ome_zarr:` (`OmeZarrAdapter`) |
|---|---|---|
| Backing | CacheManager only (memory/file) | real zarr array on disk in `_write_dir` |
| Write bounds | arbitrary (`cached_source.py:153`) | chunk-grid-aligned only (`server.py:1144-1162`) |
| Read back | only exact written bounds (`cached_source.py:212`); `get_data(arbitrary)` raises (`cached_source.py:136`) | full random access (normal zarr adapter) |
| Persistence | ephemeral, evictable | permanent on disk, no cleanup |
| Client API | `upload_array()` / `create_source` + `upload_chunk` (`client.py:1967, 2103, 2146`) | same |

Neither primitive delivers the target UX:

- **`cache:`** is ephemeral and takes arbitrary writes, but is *not a readable
  array* — you can only read back the identical chunk bounds you wrote, and they
  can be evicted. The lazy-dask / viewer read path calls `get_data(bounds)`,
  which raises for cache sources.
- **`ome_zarr:`** is a readable random-access zarr array, but it is permanent
  (no delete, no TTL, requires `write_dir`) and its writes are chunk-aligned and
  **destructive** (sub-chunk writes zero-pad the rest of the chunk —
  `zarr.py:175-181`).

## 3. Gaps to close

1. **Arbitrary-region (sub-chunk) writes.** `ZarrAdapter.write_chunk`
   (`zarr.py:162-183`) zero-pads any partial chunk, erasing the rest of that
   chunk. A true `z[sl] = data` needs server-side **read-modify-write** of
   boundary chunks. No client region-assignment API exists (only whole-array
   `upload_array`).
2. **Ephemeral lifecycle + teardown.** No `delete_source` / `drop` Flight action
   exists — only an internal `unregister_source()` (`server.py:267`), not
   exposed. Scratch arrays need a temp dir, a client-callable delete, and
   optionally TTL/idle eviction.
3. **Write concurrency safety.** Read-modify-write on a zarr v2 chunk from
   concurrent `do_put`s racing the same chunk is unsafe; needs a write lock.
4. **Client writable-array handle.** No object supports `z[sl] = data` / `z[sl]`
   against a server source.
5. **Read-after-create exposure & empty-region semantics.** Confirm a
   freshly-created source is readable mid-session (it is `register_source`'d) and
   that the precache/pyramid path tolerates a new, possibly-empty array. Settle
   that unwritten regions read as `fill_value` (zarr → 0) rather than erroring
   (the `cache:` behavior).

## 4. Design

Add a **zarr-backed ephemeral source type** that combines `cache:` ephemerality
with zarr random read/write. New `array_id` prefix `scratch:` handled in
`_create_source` (`server.py:1006`), alongside `cache:` and `ome_zarr:`.

A `scratch:` source:
- creates a zarr array in a **temp scratch dir** (new `_scratch_dir`, default a
  subdir of the OS temp / cache dir — *not* the permanent `_write_dir`),
- accepts **arbitrary-region writes** via server-side read-modify-write under a
  per-source write lock,
- is fully readable through the existing lazy-dask `get_tensor` path (normal
  `ZarrAdapter` reads; unwritten regions return `fill_value`),
- is torn down by a new **`delete_source`** Flight action (and optional TTL).

Rationale for a new prefix rather than extending `cache:`: `cache:` semantics
(write-only, evictable, exact-bounds read) are relied on by the lazy-input
compute framework; overloading it risks regressions. A distinct `scratch:`
keeps the contract explicit. (Open question 7.1 — revisit if we'd rather make
`cache:` zarr-backed wholesale.)

### 4.1 Server changes

**`server.py`**
- `_create_source`: add `elif array_id.startswith("scratch:")` branch. Create
  zarr array in `_scratch_dir / f"{name}.zarr"` (mode `w`), `fill_value=0`,
  register a `ScratchZarrAdapter`, `initialize_upload(...)`. Mark the source as
  ephemeral (track in a `self._scratch_sources: dict[source_id, path]`).
- `_handle_chunk_upload`: route scratch adapters through a new
  **arbitrary-bounds RMW path** (do *not* enforce the chunk-alignment checks at
  `server.py:1144-1162`). Acquire the source's write lock, call
  `adapter.write_region(bounds, data)`.
- Add `delete_source` to `list_actions` (`server.py:514-518`) and a handler:
  validate ownership/source exists, `unregister_source(source_id)`, `rmtree` the
  scratch path, drop bookkeeping. Idempotent.
- Optional: background TTL sweeper that deletes scratch sources idle beyond a
  configurable window.

**`adapters/zarr.py` / new `ScratchZarrAdapter`**
- `write_region(bounds, data)`: write `self.zarr_array[slices] = data` directly
  for the exact bounds. Zarr handles the chunk RMW internally for unaligned
  bounds **as long as bounds are computed correctly** — i.e. assign by slice,
  not by chunk index. This avoids the zero-pad bug by never going through
  `write_chunk(chunk_idx, ...)`. Validate bounds ⊆ shape and dtype match.
- Reads inherit `ZarrAdapter.get_data` unchanged.
- Hold a `threading.Lock` per adapter for write serialization (reads are fine
  concurrently; zarr v2 reads during writes of *other* chunks are safe — lock
  only writes, or use a finer per-chunk lock if contention warrants).

### 4.2 Client changes (`src/main/python/biopb/tensor/client.py`)

- `create_array(shape, dtype, chunks=None, dim_labels=None, ome_metadata=None,
  ephemeral=True) -> RemoteZarr`: thin wrapper over `create_source` with the
  `scratch:` prefix (or `ome_zarr:` when `ephemeral=False`). Returns a handle.
- `class RemoteZarr`:
  - `__setitem__(self, key, value)`: normalize `key` to a bounds tuple, then
    decompose the assignment into `upload_chunk(source_id, bounds, data)` calls.
    For the first cut, send the whole region as one `do_put` when it fits under
    `MAX_ARROW_BATCH_BYTES`; otherwise tile along non-spatial axes (reuse
    `chunk.py` splitting logic). Server-side RMW handles partial chunks.
  - `__getitem__(self, key)`: delegate to the existing lazy `get_tensor` /
    dask-array slicing path.
  - `delete()`: call the new `delete_source` action.
  - context-manager support (`__enter__`/`__exit__ -> delete()`) for scoped
    scratch arrays.

### 4.3 Proto

No new messages required: `ChunkUpload` (source_id + `ChunkBounds`) already
carries arbitrary bounds; `delete_source` takes a UTF-8 source_id in the action
body and returns a small JSON ack, matching `upload_status`’s convention.

## 5. Implementation steps

1. `ScratchZarrAdapter` + `write_region` (RMW by slice) + per-source lock.
   Unit test: aligned write, sub-chunk write preserves neighbors, read-back,
   unwritten region returns fill_value. (`tests/upload_test.py` style.)
2. `_create_source` `scratch:` branch + `_scratch_dir` config + bookkeeping.
3. `do_put` routing for scratch (arbitrary bounds, no alignment enforcement,
   under lock).
4. `delete_source` action + handler + `list_actions` entry; idempotent rmtree.
   Test: create → write → delete → source gone, dir removed.
5. Client `create_array` + `RemoteZarr.__setitem__/__getitem__/delete` +
   context manager. Integration test: create, region writes (aligned +
   unaligned), read-back equality, delete.
6. (Optional) TTL/idle sweeper + config knob.
7. Docs: extend `biopb-tensor-server/CLAUDE.md` (source types table) and the
   data-plane guide; note the `cache:` vs `scratch:` vs `ome_zarr:` distinction.

## 6. Testing

- Server unit: `ScratchZarrAdapter.write_region` RMW correctness (sub-chunk
  writes must not zero neighbors — the regression that the existing
  `write_chunk` zero-pad would cause), bounds validation, fill_value reads,
  concurrent-write lock.
- Server integration: create scratch source → arbitrary-region uploads →
  full-array read-back via client dask → values match; `delete_source` removes
  source + on-disk dir; double-delete is a no-op.
- Client: `RemoteZarr` setitem decomposition (single batch vs tiled), getitem
  round-trip, context-manager teardown.

## 7. Open questions

1. New `scratch:` prefix vs. making `cache:` zarr-backed wholesale. (Leaning
   `scratch:` to avoid disturbing the lazy-input compute contract.)
2. Full read-modify-write for unaligned writes vs. documenting an aligned-only
   restriction for v1. (RMW-by-slice via zarr is nearly free, so prefer full
   RMW.)
3. TTL/eviction policy and default scratch-dir location + disk-budget cap.
4. Authorization: reuse the per-source capability token
   (`CachedSourceAdapter.token`) so only the creator can write/delete a scratch
   source.
5. Crash cleanup: orphaned scratch dirs after an unclean shutdown — sweep on
   startup using a manifest or dir naming convention.
