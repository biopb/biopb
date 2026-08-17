# The localhost read path (client side)

How `biopb.tensor.TensorFlightClient` gets chunk bytes when the tensor server is
on the same machine, and how the chunks it gets are cached. The *server* half —
the `chunk_locate` action, segment files, byte ranges recorded at write time, and
the local-disk gate on the file cache — is in
[`biopb-tensor-server/ARCHITECTURE.md`](../biopb-tensor-server/ARCHITECTURE.md).

## The cache-file mmap fast path (biopb/biopb#9)

The server's file cache already holds every decoded chunk as an Arrow IPC message
in a segment file, so instead of re-sending those bytes through the loopback
`do_get` socket the client asks for the chunk's on-disk byte range
(`locate_entry`), `mmap`s the segment, reads just that message, and hands out a
**zero-copy view** onto the mapping (biopb/biopb#571). The client closes its own
`MemoryMappedFile` handle at once, but Arrow refcounts the mapping so the returned
array keeps it alive (`ndarray → pyarrow.Buffer → MemoryMappedFile`), and untouched
chunk pages are never faulted in — a partial read is nearly free.

This beats the socket because the bytes are already warm in the page cache (the
server wrote them for caching anyway), it skips the loopback gRPC overhead, and it
skips the whole-chunk copy the socket cannot: `.copy()` there fell off glibc's
32 MiB `mmap`-threshold cliff at a 64 MB chunk. Both paths return a **read-only**
view, so mutability is one uniform contract.

**Safety** rests on the server never truncating a mapped segment inode: segment
ids are strictly monotonic and eviction only `unlink`s, so the one truncating
`"wb"` open always targets a fresh path. An NFS `cache_dir` would break that and
wants an explicit gate.

**Cost — a disk leak.** While a client holds a view the server can't reclaim that
segment's blocks even after eviction `unlink`s it (the inode survives to last
close), so a client pinning many segments keeps the server's `cache_dir` above
budget. The client bounds this with **pinned-segment accounting**: it tracks the
on-disk size of the distinct segments it keeps mapped (refcounted by inode,
released by a `weakref.finalize` on the backing Arrow buffer) and, once over
`BIOPB_CACHEFILE_PIN_LIMIT` (**off by default** — a size like `16GiB` enables it),
copies the chunk out and drops the mapping instead of pinning another segment —
still off the warm mmap, no `do_get`. The hot path stays cheap: the gate is a
lock-free int compare, the segment size reuses the `stat` the fast path already
does, and only the view branch pays a lock plus one finalizer.

It runs on **all platforms, not just POSIX** (biopb/biopb#582). The old gate
assumed a client mmap blocks the server's segment `unlink`, but the client keeps
only a mapped *view*, not an open handle — so Windows removes the name at once and
the view keeps the pages valid until munmap, delete-on-last-close exactly as on
POSIX. (The biopb/biopb#5 concern was about an open *handle*.)

`BIOPB_CACHEFILE_TRANSFER_DISABLED=1` forces the socket. The client falls back to
`do_get` whenever a chunk can't be located — memory backend, old server, evicted
segment — which is the designed floor of this path. This replaced an earlier
`/dev/shm` `shm_transfer` path that was *slower* than the socket because it
allocated a fresh POSIX segment per chunk.

## The client chunk cache is two-tier

The read path (`biopb.tensor._pool`, `_fetch_chunk_distributed`) routes each
fetched chunk to one of two per-process caches, split by what the chunk actually
costs:

| Chunk kind | Cache | Why |
|---|---|---|
| mmap **view** from the fast path | **weak** (`WeakValueDictionary`) | shared OS page-cache pages, ~0 private RAM; free, uncounted, self-evicting — GC releases the entry *and* the server-side pin the instant the last holder drops the array |
| **copy** (`do_get` result, or an over-pin-budget copy) | **strong** (`cachey`) | real private RAM, so it is bounded by `_resolve_cache_bytes` (the requested size, `0` disables) |

This is why there is no longer a localhost "off by default" gate
(`BIOPB_CACHE_LOCAL` was removed): the old objection — a redundant second RAM copy
replicated per dask worker — never applied to weak views (shared page cache,
nothing replicated), and copies, rare on localhost, are what the budget bounds.
`biopb-mcp` sizes the copy budget cluster-wide via `dask_cache_budget` (split
`budget // n_workers` by a worker-init plugin).

The weak cache dedups *overlapping-lifetime* reads of one chunk; a chunk re-read
after it was fully dropped simply misses and re-runs the cheap (~1 ms) localhost
fast path.

**Caveat measured (viewer):** the viewer's *serial* plane reads scatter across
workers — dask's locality scheduler keys on tracked task *dependencies*, not an
opaque per-worker cache side-effect — so the clean viewer fix is a single-process
scheduler with one shared cache (biopb/biopb-mcp#8).

## Read amplification — chunk size is conflated with access granularity

The server sizes chunks to a fixed transfer cap (`MAX_ARROW_BATCH_BYTES = 64 MB`
in `chunk.py`, splitting non-spatial axes first and keeping the Y-X plane whole),
and that same grid is the *access* unit the client reads. A consumer reading a
small sub-region — the napari viewer scrubbing one Z plane (~2.75 MB) out of a
~63 MB chunk — transfers the whole chunk, a ~23× amplification.

The capability to read arbitrary sub-bounds cheaply already exists
(`adapter.get_data(bounds)` decodes only the requested planes); only the chunk
grid / `chunk_id` forces whole-chunk transfers. Decoupling the read grid from the
64 MB transfer cap (client-selectable granularity) is the structural fix —
biopb/biopb#8.
