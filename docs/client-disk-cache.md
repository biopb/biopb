# A client-side disk chunk cache

**Status:** implemented, **off by default** — set `BIOPB_CHUNK_CACHE` to a size
to enable. Written 2026-08-28; landed on `feat/client-disk-cache`.
**Component:** `biopb.tensor` SDK (`_pool.py`); needs nothing new from the server.
**Related:** [`localhost-fast-path.md`](localhost-fast-path.md) (the sibling path
this one complements), [`biopb-tensor-server/docs/remote-tensor-cache.md`](../biopb-tensor-server/docs/remote-tensor-cache.md)
(the same problem solved by running a local server).

## Why

The SDK's chunk cache is a per-process `cachey` LRU. That does not compose with
the multi-process dask workflows the SDK exists to feed: N workers fetching the
same chunk do N `do_get`s and hold N private copies of the same bytes.

Two answers already exist, and both assume infrastructure. A local tensor server
turns those N reads into one via the `chunk_locate` mmap fast path; a local
server pointed at a remote one (`remote-tensor-cache.md`) extends that to remote
data. Both require standing up a server, which is exactly what a user reaching
for the bare SDK is trying to avoid.

That the per-process LRU is the wrong shape is already conceded downstream:
biopb-mcp carries a bespoke dask `WorkerPlugin` (`_make_cache_plugin`) whose only
job is to divide one budget across workers so the replicated cache stays bounded —
glue whose docstring names the exact scenario, "the MCP kernel talks directly to a
**remote** tensor server under the multi-process distributed cluster, where each
worker would otherwise replicate the client cache".

The proposal: **the client writes each fetched chunk to a local cache dir and
mmaps it back**, calling `do_get` only when the file is absent. The filesystem is
the database, and the OS page cache is the cross-process shared cache that a
per-process LRU can never be. It makes that budget-splitting glue mostly moot,
since the shared copy stops being replicated at all.

## The shape

Every piece of the read half is already built and shipped for
`localhost-fast-path.md`: `_try_cachefile_transfer` mmaps a file, reads one Arrow
IPC message at an offset, and hands out a zero-copy view that Arrow keeps alive
past the local `mm.close()`; `_view_cache_put` weak-caches it. This design changes
only **who wrote the file**. `_decode_unified_batch` and the view cache are reused
verbatim.

So the two paths collapse into one concept — *a chunk lives in a file I can mmap;
either the server wrote it or I did* — with a clean split of regimes:

| Location | Who writes the file | Path |
|---|---|---|
| localhost | the server's segment cache | existing `chunk_locate` fast path |
| remote | this client | proposed, below |

**Remote-only is load-bearing, not tidiness** — see *What the write costs*. On
localhost the server's copy is already warm and writing a second one is pure loss.
`_should_try_cachefile` / `_is_localhost_location` already supply the
discriminator.

### Keying

`sha256(chunk_id)` under a directory named for the location. That is the whole key.

It is sufficient because **`chunk_id` is already content-versioned**:
biopb/biopb#178's `wrap_content_version` prepends a `[0xFF][fmt][len][cv]` header
on the read-plan mint path, `cache_key_for_chunk_id` deliberately keeps it, and
every file-backed adapter sets `_content_version` from `content_version_from_path`
(`ome_zarr` and `tifffile_adapter` inherit it). A re-registered source with new
bytes mints different chunk_ids, so a stale entry becomes un-lookupable rather
than mis-served. The client needs no new descriptor field and no new wire data —
and hashing an opaque token is not parsing it, so this stays inside the
client-opacity contract that biopb/biopb#346 was reverted to protect.

Two notes on why this exact key:

- **Location scoping is not optional.** `chunk_id` embeds `array_id`, which is
  server-local, not server-unique. Today that is covered incidentally, because
  `_CACHE_POOL` is keyed `(location, token)` and so the flat `chunk_id.hex()`
  cache key never needed it. A persistent dir must put it back, as the directory
  name.
- **Hash the whole `chunk_id`, not `cache_key_for_chunk_id`'s normalisation.**
  That function is server-side by design and unavailable here. Hashing the raw
  token over-keys relative to it (a legacy trailing method suffix keys
  distinctly) — and over-keying costs a miss, while under-keying serves wrong
  pixels. That is the correct failure direction for a client.

The staleness bet — `content_version` is a `mtime_ns:size` stat signature, so a
size-preserving write that also preserves mtime is invisible — is the same bet the
server's own persistent file cache already takes across restarts. This design does
not add staleness risk; it inherits a decision made once, in #178.

### The security boundary is the OS user, not the token

The key drops something the in-process caches keep: `_CACHE_POOL`, `_VIEW_CACHE`
and `_CALL_OPTS_POOL` are all keyed `(location, token)`, while this cache is
keyed on location and chunk_id alone. **Two tokens used by one OS account
against one server share cached chunks.**

That is deliberate, and it is sound only because the isolation moves one level
down, to the filesystem. The tree is created **owner-only** — `0o700`
directories, `0o600` files — so the unit of separation is the OS account. Under a
default umask it would otherwise be `0o755`/`0o644`, and on the many distributions
that ship `0o755` home directories every cached pixel would be readable by any
other local account. That is not a hypothetical umask nitpick; it was the state
of the first implementation, and the mode is now asserted in tests.

Keying on the token instead would be worse on both counts:

- **It adds no boundary.** A second principal able to read this directory already
  has the OS account, and can read the files whatever they are named. The key
  only separates principals that the filesystem does not, and biopb has no such
  sub-account boundary.
- **It breaks the cache.** Tokens rotate; content does not. Keying a *persistent*
  store on a credential orphans every entry at each re-authentication and
  re-fetches the whole working set. In-process that cost is invisible, because a
  process outlives its token — persisted, it is a cache-invalidation bug.

Hashing a bearer credential into a path name that appears in directory listings,
backups, and error messages is also worth avoiding on its own.

**A shared cache directory breaks the model**, and the interesting risk is
integrity rather than confidentiality: an account that can *write* into the tree
can plant a well-formed Arrow file at a key this process will read, and it will
be decoded as pixels. `BIOPB_CHUNK_CACHE_DIR` pointed at shared scratch is
therefore warned about loudly at startup — warned rather than refused, because a
shared filesystem used by one person is legitimate and indistinguishable from a
genuinely shared one, and rather than silently tightened, because a directory the
user handed us may not be ours to chmod.

Windows is covered by location instead of mode bits: `os.chmod` there only
toggles the read-only attribute, and the default root under `%LOCALAPPDATA%` is
already owner-only by inherited ACL. The mode check is POSIX-only, matching
`_credentials`.

**A tree created before this hardening keeps its old modes.** The startup check
looks at the root only, and nothing walks the existing subdirectories to
retighten them, so a cache dir first written by an earlier build stays at the
umask default even though new directories under it are `0o700`. There is no
migration: delete the tree and let it refill. It holds nothing but regenerable
chunks, which is the same property that makes `unlink`-anything-that-does-not-
parse the whole recovery story.

```sh
rm -rf ~/.cache/biopb/chunks     # %LOCALAPPDATA%\biopb\Cache\chunks on Windows
```

### Layout and location

`<cache_root>/<hash(location)>/<ab>/<cdef…>.arrow`, one Arrow IPC message per
file, written to `.tmp.<pid>.<tid>` then `os.rename` (atomic on the same
filesystem) so a reader never sees a torn file. No lockfiles on the write path: N
workers racing the same miss cost N `do_get`s, which is exactly what happens today
anyway.

Two levels of hash-prefix sharding keep `getdents` cheap and let the sweeper work
over sampled shards instead of globally ordering every file.

**`/tmp` is the wrong default.** On many systems it is `tmpfs`, which would turn
"unbounded disk" into unevictable RAM plus an mmap that is also RAM — strictly
worse than the LRU being replaced.

The root is `cache_dir() / "chunks"` — `~/.cache/biopb`, and
`%LOCALAPPDATA%\biopb\Cache` on Windows. That Windows split is the only place
`_locations.py` diverges from its one-layout-everywhere rule, and this tree is
why: the other three hold kilobytes, while this one is sized for tens of GB,
which is exactly what `%LOCALAPPDATA%` exists to keep out of roaming profiles and
Folder Redirection. `BIOPB_CHUNK_CACHE_DIR` relocates it.

A root on `tmpfs`, a network mount, or a cloud-synced folder is **refused**:
`load_settings` returns None with a warning naming the reason, and the read path
degrades to the in-memory fallback like any other disk-cache failure. The
classification is `biopb._fs_detect` — the same probe the server uses to demote a
bad `cache_dir` to its memory backend (biopb/biopb#571), moved down into the core
SDK so both tenants share one parser instead of the SDK growing a second.

RAM-backed rejection is opt-in there (`reject_memory=True`) because the two
tenants genuinely disagree: the server's answer to an unusable `cache_dir` *is*
its memory backend, so demoting a `tmpfs` dir would trade RAM for the same RAM.
This cache has no such fallback and exists precisely to keep bytes out of RAM.

## Eviction

The load-bearing property is that **`unlink()` on a mapped file is safe on
POSIX** — the mapping survives to last `munmap`, and Arrow already refcounts that
for views handed out. So a sweeper deletes cold files *while readers hold them*,
with no reader coordination and no locking on the read path. A process that
believed a chunk was on disk gets `FileNotFoundError` and falls through to
`do_get`; nothing needs invalidating and no signal is broadcast.

What *does* need coordination is policy: N processes must not all scan-and-evict
at once, or they over-evict by a factor of N, each seeing the pre-sweep size.

**One mutual-exclusion token is the entire coordination requirement**, and the
primitive is already in the core SDK: `biopb/_lifecycle/file_lock.py` — stdlib
-only, cross-platform, and held on an fd so **the OS releases it when the holder
dies**, which matters when the N processes are dask workers that get killed. A
sweep is `file_lock(cache_dir/"sweep.lock", timeout=0)`; the winner sweeps, the
losers pay one failed `flock` and move on. No daemon.

**Trigger on work, not wall-clock.** A process attempts the lock only once it has
itself written K bytes since its last attempt (`_SWEEP_AFTER_BYTES_MIN`, or a
sixteenth of the budget, whichever is larger). Sweep rate then tracks write rate
rather than process count — 32 idle workers never sweep, and a burst of misses is
swept by whichever worker crosses the threshold first.

The byte counter is not optional, and an earlier draft of this doc was wrong to
say the TTL removes it. The TTL removes the need for a *timer* — nothing has to
wake up on a schedule to reclaim idle bytes. But the free-space floor is one
`statvfs` while the byte budget cannot be known without a full scan, so the scan
still needs something to trigger it. Written-bytes is that trigger.

### There is no free recency signal

This is where "the filesystem is the database" actually bites. `atime` is
unusable: default `relatime` updates it only when it is older than `mtime` or
older than 24h, and SSD-tuned hosts often mount `noatime`. The filesystem will not
tell you which chunks are hot.

The cheap recovery is **hand-rolled relatime**: the reader already `stat`s the file
before mmap, so `os.utime()` it only when its mtime is older than ~1h. That is one
metadata write per chunk per hour instead of per hit, and it recovers most of
LRU's value over plain FIFO. Eviction is then oldest-mtime-first.

### A TTL, but a generous one

Byte budget and TTL bound different things, and only the budget bounds the one
that matters: a TTL's ceiling is `fetch_rate × TTL`, so at even 12 MB/s a 24-hour
TTL admits ~1 TB, and at 1 GbE ~9.7 TB. Any TTL long enough to be useful for
revisits admits far more than a laptop has.

It is also the wrong shape for image access specifically. Pyramids make the
access distribution extremely skewed — the overview level is touched on every
navigation and its miss blanks the viewport — and a TTL is the one policy blind
to skew, expiring the hot set on the same schedule as a chunk seen once. Worse,
it discards under *no pressure*: your session's chunks die because you went to
lunch, even with the cache 3% full. Budget eviction is relative, so it only
discards when something is actually competing for the space. And the working set
here is denominated in bytes (the demand-tier warm extent is ~20 GB on a
timelapse), not in seconds — a budget's parameter falls out of the workload, a
TTL's does not.

So the TTL is not doing hit-rate work. Its job is the one thing a budget cannot
do: a budget is a target the cache rises to and *stays at*, so without a TTL
biopb permanently occupies N GB of a dataset the user finished with in March.
`_TTL_DEFAULT` is 7 days — comfortably past any plausible revisit, so it never
fires inside a session and cannot do the skew-blind damage above. It only
collects abandoned datasets.

It is enforced in both places, and needs to be. Lazily on the read path, free off
the `stat` that precedes the mmap — but that only reclaims files someone touches,
and the files never touched again are exactly a single-pass scan's garbage. So
the pressure sweep applies it too, which costs nothing since it is stat-ing
everything anyway.

### A free-space floor, not just a budget

Whatever byte budget is configured will be wrong on someone's laptop. Each sweep
also checks `statvfs` and evicts down hard when free space falls below a floor,
regardless of budget. This is what keeps a bad guess from turning into *the user's
disk is full and nothing else on the box works* — the failure mode that would
otherwise be unattributable to biopb.

## What a hit costs

Measured 2026-08-29 on the dev box (Ryzen 5 5600X, ext4 on an LVM volume spanning
a SATA MX500 and an NVMe, pyarrow 24.0.0), medians over the cache's own
`read_batch`/`write_batch`. **Warm** = the file's pages are resident; **cold** =
`sync` then `POSIX_FADV_DONTNEED` on that file immediately before the read.

| per chunk | 256 KB | 1 MB | 8 MB | 64 MB |
|---|---|---|---|---|
| `read_batch` (the lookup), warm | 64 µs | 63 µs | 66 µs | 118 µs |
| + decode and touch every byte, warm | 150 µs | 282 µs | 1.5 ms | 12 ms |
| + decode and touch every byte, cold | 1.2 ms | 1.7 ms | 6.2 ms | 98 ms |
| `write_batch`, on the miss path | 0.5 ms | 1.5 ms | 7.3 ms | 93 ms |

**The lookup is flat and small** — ~66 µs, near-independent of chunk size, because
the mmap is lazy and nothing here touches a pixel: 4 µs of keying (two sha256s),
2 µs `stat`, 12 µs `memory_map`, 42 µs Arrow `open_stream`/`read_next_batch`.

That is ~4x cheaper than the localhost `chunk_locate` round trip this path stands
in for on remote (~290 µs, `localhost-fast-path.md`). On a hit the client's own
file is not merely competitive with the server fast path — it is the cheaper of
the two, because it has no RPC in it.

Size appears only when the caller faults pages in. Cold reads run 0.5–1.4 GB/s;
that spread is this box's LV striping two devices of different speeds, not a
property of the code, so read the cold row as an order of magnitude.

### A hit does not care how big the cache is

`sha256` → path → `open` does no directory listing, and ext4's htree absorbs the
rest. Timed against a tree filled to N entries, re-reading one target chunk:

| files in cache | 1 | 1 000 | 10 000 | 100 000 | 500 000 |
|---|---|---|---|---|---|
| warm `read_batch` | 69 µs | 68 µs | 67 µs | 65 µs | 68 µs |
| `_scan` (the sweeper's full walk) | 0.2 ms | 11 ms | 67 ms | 765 ms | 5.0 s |

**The sweeper is the O(N) piece**, ~10 µs per file, and `note_written` runs it
synchronously inside `write_batch`. Amortised it is self-limiting and independent
of both knobs: scanning `budget/chunk` files once per `budget/16` bytes written is
**~160 µs per miss** at any budget and any chunk size — invisible beside the
`do_get` that miss already paid for.

The exposure is the tail, not the mean. Whichever worker crosses the threshold
eats the entire walk, so a 64 GB budget of 256 KB chunks (256k files) stalls that
worker ~2.5 s. `maybe_sweep` takes the lock with `timeout=0.0`, so only one worker
pays and the others skip — that bounds the blast radius without shrinking it. If
file counts ever reach that range the fix is sampling shards instead of ordering
every file globally, which is what the one byte of sharding in the layout is for.

## What the write costs

Less than it looks, and not what it looks like.

**The write is never `fsync`ed.** A chunk file is regenerable from the server, so
there is no durability requirement at all — crash recovery is "unlink anything
that does not parse". Writeback is deferred to the kernel and off the fetch path.

One `MAX_ARROW_BATCH_BYTES` (64 MB) chunk, disk side measured (*What a hit
costs*), network side still arithmetic from the nominal link rate:

| | 64 MB chunk |
|---|---|
| `do_get` over 1 GbE | ~570 ms (arithmetic) |
| `do_get` over 10 GbE | ~58 ms (arithmetic) |
| `write_batch` | **93 ms (measured)** |
| localhost `chunk_locate` + mmap | ~1 ms (measured, `localhost-fast-path.md`) |

**The write costs about ten times what the memcpy argument above predicts.**
Measured throughput is ~1 GB/s (0.7–1.4 GB/s across sizes, plus ~0.4 ms fixed per
write), not the 33–36 GB/s DRAM ceiling, and it is flat from 1 MB up — so this is
per-byte cost in Arrow serialisation and the `write` copy, not dirty-page
throttling, and not something a smaller chunk escapes.

The conclusion survives but with a thinner margin than claimed: the write is
**~15% of a 1 GbE miss**, not 1%, and over 10 GbE it is *larger* than the miss it
rides on — a 64 MB chunk would cost 58 ms to fetch and 93 ms to persist. Remote-
only scoping is still load-bearing (on localhost the write is a ~90x regression
against the existing fast path), but on a fast LAN "nearly free" is the wrong
word for it. What justifies the write there is the cross-process payoff below,
not its size.

**Rejected: populate on second miss.** Knowing a miss is the second one requires a
persistent record of the first — itself a write, so the cost being avoided is paid
anyway — and per-process bookkeeping does not compose across N workers that each
see the chunk once.

### The residue is pollution, not bandwidth

The real cost of a single-pass 500 GB scan is not the bytes written; it is that it
fills the dir with chunks that will never hit and drives the sweeper to evict an
interactive session's working set.

That mostly dissolves into the recency mechanism already required. With
mtime-bump-on-hit, a re-read chunk gets a fresh mtime while single-pass chunks keep
their original write mtime forever — so **oldest-mtime-first eviction is already
scan-resistant**. The streaming run's chunks are always the oldest and are always
evicted first; the revisited interactive set survives. That is the property
SIEVE/CLOCK exist to provide, falling out of a signal needed anyway.

### Who benefits

The process taking the miss gains nothing from the write — it already holds the
decoded array. The payoff is entirely **cross-process and cross-session**: dask
fan-out, and the next session. A single-process one-shot script sees only cost.
This should not be sold as a general speedup.

One refinement makes that cost zero copies over the status quo: write the file
from the `do_get` Arrow buffer, then mmap it back and return a view onto the
mapping, dropping the in-memory buffer. Same number of copies as returning the
in-memory array does today, and it lands every chunk in **one** representation — a
weak-cacheable mmap view — regardless of which of the three paths fetched it.

## Relationship to the in-memory LRU

**Keep it, demoted to a fallback.**

Today the strong `cachey` cache holds exactly two things: `do_get` results, and
over-pin-budget copies from the localhost fast path. Under this design remote
`do_get` results become mmap views and route to the weak `_view_cache` instead, so
its main population source disappears. Two cases survive:

1. **Over-budget localhost copies** (`_pin_budget_exhausted()` → `is_view=False`),
   untouched by this work.
2. **The disk cache being unavailable at runtime** — `tmpfs` rejection, read-only
   or full filesystem, `ENOSPC` mid-write, the free-space floor, a misbehaving
   Windows path.

(2) is the whole argument for keeping it. This subsystem has more runtime failure
modes than an LRU does and they are environmental; without an in-memory fallback,
the user's disk filling up degrades to *every chunk re-fetched over the network on
every dask task*. The LRU is what makes that a footnote instead of a cliff.

**But the two layers must not both hold the same chunk.** Strong-caching a chunk
that is also in a local file is double-buffering: a private RAM copy of bytes
already in the page cache, N private copies against one shared one across N
workers. Returning that RAM to the page cache is strictly better use of the same
bytes. This needs no new knob — it falls out for free, because once remote
`do_get` results are views, nothing `put`s to the strong cache on the healthy
remote path and its budget simply goes unconsumed.

**Do not lower the default 1 GB budget.** `cachey` holds only what is `put`, so the
budget is a ceiling, not a reservation; lowering it only hurts the degraded case it
now exists to cover.

**Foot-gun to document:** `configure_cache(..., cache_bytes=0)` pins the strong
cache off tri-state (`None` in `_CACHE_POOL`, deliberately not recreated). Today
that means "always `do_get`" — bad, but bounded and expected. Once the strong cache
is the fallback layer, `0` means "no fallback", and a disk-cache failure under it
becomes the cliff above.

This is reachable, not hypothetical: biopb-mcp's `dask.cache_budget` documents
"`0` disables" and `_register_cache_plugin` passes `budget // n_workers` straight
into `configure_cache`. Keep the semantic rather than special-casing it, but say so
plainly at the config surface.

## Not yet done

- **Config surface.** Environment variables only so far; nothing plumbs this
  through biopb-mcp's `dask.*` config the way `cache_budget` is plumbed.

## Open questions

- The `do_get` rows are still arithmetic. The disk-side rows are measured now
  (*What a hit costs*), but no real remote fetch has been timed end to end, so
  the 10 GbE conclusion rests on a nominal link rate rather than a wire.
- `UnresolvedSourceAdapter` (the URL-only cloud model in
  `cloud-storage-support.md`) sets no `_content_version`. Confirm it never serves
  chunks; if it can, its chunk_ids are unversioned and must not be persisted.
- Windows: `unlink`-while-mapped is delete-on-last-close there too
  (biopb/biopb#582 established this for the localhost path), but the sweeper's
  behaviour when a *name* is still open wants checking separately.
- Is the byte budget per-location or global across the cache root? Per-location is
  simpler to sweep; global is what a user actually means by "use 50 GB".
