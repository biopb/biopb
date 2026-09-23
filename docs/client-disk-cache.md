# A client-side disk chunk cache

**Experimental.** Off by default and may still change without notice.

Scope: `biopb.tensor` SDK (`_pool.py`, `_diskcache.py`). Complements
[localhost-fast-path.md](localhost-fast-path.md) (the server-local mmap path) and
[remote-tensor-cache.md](../biopb-tensor-server/docs/remote-tensor-cache.md)
(the same problem solved by running a local server).

## What it does

On a remote (non-localhost) `do_get` miss, the client writes the fetched chunk
to a local file and mmaps it back on the next read, instead of re-fetching over
the network. A localhost read never uses this cache — it already has the
server's own mmap fast path ([localhost-fast-path.md](localhost-fast-path.md)); `_is_localhost_location`
is the discriminator. This turns the OS page cache into a cross-process,
cross-session shared cache: N dask workers reading the same chunk share one
file instead of holding N private RAM copies.

Off by default: set `BIOPB_CHUNK_CACHE` to a size (e.g. `2GiB`) to enable it. A
cache with no budget is disabled, and unset / `0` / unparsable all read as
disabled.

## Keying

`sha256(chunk_id)`, under a directory named for the location:
`<cache_root>/<format>/<hash(location)>/<ab>/<digest>.arrow` — sharded two hex
characters deep so `getdents` stays cheap. `FORMAT` (currently `v1`) names the
entry layout and stored encoding (one Arrow IPC stream, one record batch, the
server's unified chunk schema); bump it on any change to either and the old
tree becomes unreachable rather than misread — no migration code needed.

`chunk_id` is already content-versioned (`content_version` is prepended into it
at mint time), so hashing it verbatim is sufficient: a re-registered source
with new bytes mints different chunk_ids, so a stale entry becomes
un-lookupable rather than mis-served. A server-side *reading* change (not a
data change) rides `CHUNK_SEMANTICS_EPOCH` in the same header, so a bump
re-keys every chunk_id and this cache misses like any other cache. Hashing the
raw token — never parsing it — is what keeps this inside the SDK's
client-opacity contract for `chunk_id`.

`chunk_id` embeds `array_id`, which is server-local, not server-unique, so the
location has to be part of the key. `biopb.tensor._location` canonicalizes it
(e.g. `grpc+tcp://` and `grpc://` fold to one spelling) before hashing;
`localhost`/`127.0.0.1` are deliberately not resolved to each other, and
neither ever reaches this cache anyway.

The staleness bet is the one the server's own persistent file cache already
takes: `content_version` is a `mtime_ns:size` signature, so a size-preserving
write that also preserves mtime is invisible. This design inherits that risk
rather than adding to it.

## The security boundary is the OS user, not the token

Two tokens against one server share cached chunks here — the key is location +
chunk_id only, unlike the in-process caches, which key on `(location, token)`.
That's sound because the tree is owner-only (`0o700` directories, `0o600`
files): the unit of separation is the OS account, not the credential. Keying on
the token instead would add no real boundary (anyone who can read the
directory already has the OS account) and would break the cache on every token
rotation.

A cache root on shared or writable-by-others storage is a real risk: an
account that can write into the tree can plant a well-formed Arrow file at a
key this process will read as pixels. `BIOPB_CHUNK_CACHE_DIR` pointed at such a
directory is warned about at startup, not refused — the mode check only looks
at the root, so a tree created before this hardening keeps its old, permissive
modes. There is no migration; the fix is to delete and let it refill:

```sh
rm -rf ~/.cache/biopb/chunks     # %LOCALAPPDATA%\biopb\Cache\chunks on Windows
```

On Windows, `os.chmod` only toggles the read-only attribute; the default root
under `%LOCALAPPDATA%` is already owner-only by inherited ACL, and the mode
check itself is POSIX-only.

A root detected as `tmpfs`, a network mount, or a cloud-synced folder is
refused outright (`biopb._fs_detect`, shared with the server's own `cache_dir`
demotion logic) — the read path degrades to the in-memory fallback like any
other disk-cache failure. `/tmp` is excluded for the same reason: it is often
`tmpfs`, which would turn "unbounded disk" into unevictable RAM.

## Layout and location

Root: `cache_dir() / "chunks"` — `~/.cache/biopb/chunks`,
`%LOCALAPPDATA%\biopb\Cache\chunks` on Windows (sized for tens of GB, unlike
the SDK's other kilobyte-scale state dirs). `BIOPB_CHUNK_CACHE_DIR` relocates
it. Writes go to a temp file in the same directory, then `os.rename` (atomic on
the same filesystem), so a reader never sees a torn file; there is no
write-path lockfile, so concurrent workers racing the same miss just pay N
redundant `do_get`s, same as without this cache.

## Eviction

`unlink()` on a mapped file is safe on POSIX — the mapping survives to the last
`munmap` — so a sweeper deletes cold files while readers hold them, with no
reader coordination and no read-path locking: a process that believed a chunk
was on disk gets `FileNotFoundError` and falls through to `do_get`.

Coordination is needed only for policy, so N processes don't over-evict by a
factor of N. `ExclusiveFileLock` on `sweep.lock` (`biopb._lifecycle.file_lock`
— stdlib-only, held on an fd so it releases if the holder dies) makes exactly
one worker sweep; the rest see a failed non-blocking lock and move on. A
process attempts the lock only after it has itself written
`BIOPB_CHUNK_CACHE`-budget/16 bytes (or 64 MiB, whichever is larger) since its
last attempt, so sweep rate tracks write rate rather than process count.

There's no free `atime` signal to read (default `relatime`, or hosts mounted
`noatime`), so recency is hand-rolled: the reader already `stat`s the file
before mmap and bumps its mtime if it's more than ~1h stale. Eviction is
oldest-mtime-first, which is also scan-resistant for free — a single large
scan's chunks keep their original write mtime and are always the oldest, so
they're evicted before an interactive session's revisited working set.

**A TTL, but a generous one** (`_TTL_DEFAULT` = 7 days,
`BIOPB_CHUNK_CACHE_TTL` overrides it). The byte budget bounds the working set;
the TTL's only job is reclaiming what a budget alone would hold onto forever —
a dataset the user finished with in March. It's enforced lazily on read (free
off the existing `stat`) and by the sweep.

**A free-space floor, independent of budget.** Each sweep also checks
`statvfs` and evicts hard when free space falls below a floor (default 10 GiB,
`BIOPB_CHUNK_CACHE_MIN_FREE` overrides), so a bad budget guess can't fill
someone's disk.

## Performance

A warm lookup is flat and small — ~65 µs regardless of chunk size (a stat, an
mmap, two sha256s, an Arrow header read) and independent of how many files are
already in the cache (ext4's htree absorbs the directory lookup). That beats
the localhost `chunk_locate` round trip this path stands in for on remote
(~290 µs) — a hit here has no RPC in it at all.

Writes are never `fsync`ed — a chunk file is regenerable from the server, so
there's no durability requirement, and crash recovery is "unlink anything that
doesn't parse." A 64 MB chunk (the transfer cap) writes at roughly 1.8 GB/s in
a burst and degrades to roughly 0.5 GB/s under sustained streaming once the
write outruns the kernel's dirty-page budget — quote the sustained number for
a bulk scan and the burst number for an interactive miss, never the same one
for both. Both are well under this box's DRAM ceiling, so "a memcpy into page
cache" is the wrong model for the cost; the device is what's slow, not the
copy.

Writing a file per chunk, rather than appending into a shared segment the way
the server's own file cache does, costs roughly 10-40% of write throughput but
keeps the design simple: the key is a pure path lookup with no index, eviction
is a plain `unlink`, and a torn write is recovered by deleting one file.

The write's cost lands entirely on the process that pays it — the chunk it
wrote gains it nothing, since it already holds the decoded array. The payoff
is cross-process and cross-session (dask fan-out, the next session); a
single-process one-shot script sees only cost, never a speedup.

## Relationship to the in-memory LRU

The per-process `cachey` LRU is kept, demoted to a fallback for when the disk
cache is unavailable (tmpfs rejection, ENOSPC, a bad path). Once a remote
`do_get` result is written to disk it's read back as an mmap view and routed to
the weak `_view_cache` rather than the strong LRU, so the two layers never
double-buffer the same chunk — nothing needs to check for this, since a
healthy disk cache simply stops `put`ting into the strong cache on the remote
path.

## Known gaps

- No config surface beyond the four `BIOPB_CHUNK_CACHE*` env vars — nothing
  plumbs this through biopb-mcp's `dask.*` config.
- A second SDK (e.g. Java) sharing this tree is undecided and not built.
- `UnresolvedSourceAdapter` (the URL-only cloud model) sets no
  `_content_version` and therefore cannot be safely served via this path.
