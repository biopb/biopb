# The localhost read path (client side)

How `biopb.tensor.TensorFlightClient` gets chunk bytes when the tensor server is
on the same machine, and how the chunks it gets are cached. The *server* half —
the `chunk_locate` action, segment files, byte ranges recorded at write time, and
the local-disk gate on the file cache — is in
[`biopb-tensor-server/ARCHITECTURE.md`](../biopb-tensor-server/ARCHITECTURE.md).

## The cache-file mmap fast path

The server's file cache already holds every decoded chunk as an Arrow IPC
message in a segment file, so instead of re-sending those bytes through the
loopback `do_get` socket the client asks for the chunk's on-disk byte range
(`locate_entry`), `mmap`s the segment, reads just that message, and hands out a
**zero-copy view** onto the mapping. The client closes its own
`MemoryMappedFile` handle at once, but Arrow refcounts the mapping so the
returned array keeps it alive (`ndarray → pyarrow.Buffer → MemoryMappedFile`),
and untouched chunk pages are never faulted in — a partial read is nearly free.
Both paths return a **read-only** view, so mutability is one uniform contract.

This beats the socket because the bytes are already warm in the page cache, it
skips the loopback gRPC overhead, and it skips the whole-chunk copy the socket
can't avoid (`.copy()` there falls off glibc's 32 MiB `mmap`-threshold cliff at
a 64 MB chunk).

**Safety** rests on the server never truncating a mapped segment inode: segment
ids are strictly monotonic and eviction only `unlink`s, so the one truncating
`"wb"` open always targets a fresh path. An NFS `cache_dir` breaks that and
needs an explicit gate.

Codec consistency is checked exactly once, at `GetFlightInfo`
(`_check_wire_protocol` in `_session.py`, comparing the `chunk_wire_protocol`
schema-metadata key to `TENSOR_WIRE_PROTOCOL_VERSION`), not on every fetch.
Neither `locate_entry`'s mmap read nor `do_get` re-checks it — each just
decodes a chunk from its own embedded Arrow IPC schema — so both are
schema-safe only because a `chunk_id` cannot exist without having already
passed a `GetFlightInfo` call.

**Cost — a disk leak.** While a client holds a view the server can't reclaim
that segment's blocks even after eviction `unlink`s it (the inode survives to
last close), so a client pinning many segments keeps the server's `cache_dir`
above budget. The client bounds this with **pinned-segment accounting**: it
tracks the on-disk size of the distinct segments it keeps mapped (refcounted by
inode, released by a `weakref.finalize` on the backing Arrow buffer) and, once
over `BIOPB_CACHEFILE_PIN_LIMIT` (**off by default** — a size like `16GiB`
enables it), copies the chunk out and drops the mapping instead of pinning
another segment — still off the warm mmap, no `do_get`. The hot path stays
cheap: the gate is a lock-free int compare, the segment size reuses the `stat`
the fast path already does, and only the view branch pays a lock plus one
finalizer.

It runs on **all platforms, not just POSIX** — the client keeps only a mapped
*view*, not an open handle, so Windows removes the name at once and the view
keeps the pages valid until munmap, delete-on-last-close exactly as on POSIX.

`BIOPB_CACHEFILE_TRANSFER_DISABLED=1` forces the socket. The client falls back
to `do_get` whenever a chunk can't be located — memory backend, old server,
evicted segment — which is the designed floor of this path.

## The client chunk cache is two-tier

The read path (`biopb.tensor._pool`, `_fetch_chunk_distributed`) routes each
fetched chunk to one of two per-process caches, split by what the chunk
actually costs:

| Chunk kind | Cache | Why |
|---|---|---|
| mmap **view** from the fast path | **weak** (`WeakValueDictionary`) | shared OS page-cache pages, ~0 private RAM; free, uncounted, self-evicting — GC releases the entry *and* the server-side pin the instant the last holder drops the array |
| **copy** (`do_get` result, or an over-pin-budget copy) | **strong** (`cachey`) | real private RAM, bounded by `_resolve_cache_bytes` (the requested size, `0` disables) |

There is no localhost "off by default" gate on this cache — the weak cache
holds nothing replicated across workers, and the strong cache's budget bounds
the copies that remain. The weak cache dedups *overlapping-lifetime* reads of
one chunk; a chunk re-read after it was fully dropped simply misses and re-runs
the cheap (~1 ms) localhost fast path.

## Read amplification cost

The server sizes chunks to a fixed transfer cap (`MAX_ARROW_BATCH_BYTES = 64 MB`
in `chunk.py`, splitting non-spatial axes first and keeping the Y-X plane
whole), and that same grid is the *access* unit the client reads. A consumer
reading a small sub-region transfers the whole chunk. The mmap path is what makes
such amplification **cheap**, because no copy of the data was made during the read.
