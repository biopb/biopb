"""Client-side on-disk chunk cache -- the filesystem as the shared cache.

The SDK's in-process ``cachey`` LRU does not compose with the multi-process dask
workflows it exists to feed: N workers fetching one chunk do N ``do_get``s and
hold N private copies. A local tensor server fixes that (the ``chunk_locate``
mmap fast path turns N reads into one), but standing one up is exactly what a
user reaching for the bare SDK is avoiding.

So the client writes each fetched chunk to a file and mmaps it back, calling
``do_get`` only on a miss. The OS page cache is then the cross-process shared
cache a per-process LRU can never be. Design rationale, measurements, and the
policy comparison behind every constant here: ``docs/client-disk-cache.md``.

Three properties carry the design:

- **A chunk file is regenerable**, so it is never ``fsync``ed and crash recovery
  is "unlink whatever does not parse". The on-path write cost is a memcpy into
  page cache; the writeback is the kernel's problem, off the fetch path.
- **``chunk_id`` is already content-versioned** (biopb/biopb#178 prepends a
  ``content_version`` header on the mint path, and ``cache_key_for_chunk_id``
  keeps it), so re-registered source bytes mint different ids and a stale entry
  becomes un-lookupable rather than mis-served. The key needs nothing else --
  and hashing an opaque token is not parsing it, so this stays inside the
  client-opacity contract biopb/biopb#346 was reverted to protect.
- **``unlink`` on a mapped file is safe**, so the sweeper deletes cold files
  while readers hold them, with no reader coordination and nothing to invalidate.
  A process that believed a chunk was on disk gets a miss and refetches.

**Remote locations only.** On localhost the server's own segment already holds
the chunk warm and the existing fast path reads it in ~1 ms; writing a second
copy would cost more than the miss it saves. ``_pool`` owns that gate.

Security boundary: the OS user, not the token
---------------------------------------------
The cache key is ``sha256(chunk_id)`` under a *location* digest. The bearer
token is deliberately NOT part of it, unlike the in-process pools, which are
keyed ``(location, token)``. So two tokens used by one OS user against one
server share cached chunks.

That is sound only because the isolation is enforced one level down, by the
filesystem: the tree is created owner-only (``0o700``), so the unit of
separation is the OS account. Anything that breaks that assumption breaks the
model -- which is why a root that is group- or world-accessible is warned about
loudly rather than quietly used (see :func:`_ensure_owner_only`).

Keying on the token instead would be worse on both counts. It would not add a
boundary: a second principal able to read this cache dir already has the OS
account, and could read the files whatever they are named. And it would break
the cache, because *tokens rotate and content does not* -- re-authenticating
would orphan every entry and re-fetch the whole working set. In-process that
cost is invisible (a process outlives its token); persisted, it is a
cache-invalidation bug. Hashing a bearer credential into a path name that shows
up in directory listings, backups, and error messages is also a thing to avoid
on its own.
"""

import hashlib
import logging
import os
import shutil
import stat
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pyarrow as pa
from dask.utils import parse_bytes

from biopb._fs_detect import unsafe_cache_dir_reason
from biopb._lifecycle.file_lock import ExclusiveFileLock
from biopb._locations import cache_dir

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================
#
# Opt-in, like the pinned-segment bound (BIOPB_CACHEFILE_PIN_LIMIT): this spends
# a user's disk, and the payoff is entirely cross-process/cross-session -- the
# process taking the miss gains nothing, since it already holds the decoded
# array. A single-process one-shot script would see only the cost, so the default
# is off until a caller says otherwise.

ENV_BUDGET = "BIOPB_CHUNK_CACHE"  # size string; enables the cache
ENV_DIR = "BIOPB_CHUNK_CACHE_DIR"  # relocate off the default cache tree
ENV_TTL = "BIOPB_CHUNK_CACHE_TTL"  # seconds; idle reclamation
ENV_MIN_FREE = "BIOPB_CHUNK_CACHE_MIN_FREE"  # size string; free-space floor

# A TTL's only job here is collecting abandoned datasets, NOT managing hit rate:
# image access is revisit-heavy at second-to-hour scale (scrubbing Z, panning
# back, an agent retrying a region) and a pyramid's overview level is touched on
# every navigation. An aggressive TTL would evict that hot set on a fixed
# schedule with no pressure to justify it -- pure loss, since eviction here is
# supposed to be *relative* (see _sweep). Seven days is comfortably past any
# plausible revisit, so the TTL never fires inside a session.
_TTL_DEFAULT = 7 * 24 * 3600.0

# Evict hard below this much free space regardless of budget. Whatever budget a
# user configures will be wrong on someone's laptop, and the failure this guards
# is not a slow cache -- it is "the disk is full and nothing else on the box
# works", which no one would attribute to biopb.
_MIN_FREE_DEFAULT = 10 * 1024**3

# Hand-rolled relatime. The filesystem will not tell us what is hot: default
# `relatime` updates atime only when it is older than mtime or older than 24h,
# and SSD-tuned hosts mount `noatime` outright. So a reader bumps mtime itself --
# but only when it is already stale, making this one metadata write per chunk per
# hour instead of one per hit. Eviction is then oldest-mtime-first, which is
# *scan-resistant for free*: a re-read chunk carries a fresh mtime while a
# single-pass scan's chunks keep their original write time forever, so the scan's
# own garbage is always the first thing evicted and an interactive working set
# survives it.
_MTIME_BUMP_AFTER = 3600.0

# Bytes this process must write before it attempts a sweep. The free-space floor
# is one statvfs, but the byte budget cannot be known without a full scan, so the
# scan needs a trigger. Work-proportional rather than timed: sweep rate tracks
# write rate instead of process count, so idle workers never sweep and a burst of
# misses is swept by whichever worker crosses the threshold first.
_SWEEP_AFTER_BYTES_MIN = 64 * 1024**2

_SUFFIX = ".arrow"
_LOCK_NAME = "sweep.lock"


class Settings:
    """Resolved cache policy for this process. Immutable; env is read once."""

    __slots__ = ("root", "budget", "ttl", "min_free", "sweep_after")

    def __init__(self, root: Path, budget: int, ttl: float, min_free: int):
        self.root = root
        self.budget = budget
        self.ttl = ttl
        self.min_free = min_free
        # Scan at least every 1/16th of the budget, so a small budget is swept
        # proportionally often rather than overrunning by a fixed 64 MiB.
        self.sweep_after = max(_SWEEP_AFTER_BYTES_MIN, budget // 16)


# Mode for every directory this module creates. Not subject to umask trouble: a
# umask only clears bits, and 0o700 has no group/other bits to clear.
_DIR_MODE = 0o700
_FILE_MODE = 0o600


def _ensure_owner_only(root: Path) -> None:
    """Create *root* owner-only, and complain if an existing one is not.

    The directory bit is what actually enforces the boundary: with the root at
    ``0o700`` nobody else can traverse into it, whatever the individual files are
    set to. A default umask would otherwise leave this ``0o755`` under a
    ``0o755`` home -- i.e. every cached pixel readable by any local account.

    A pre-existing wide root is warned about, not fixed and not refused. Not
    fixed because tightening a directory the user pointed us at (via
    ``BIOPB_CHUNK_CACHE_DIR``) may not be ours to tighten; not refused because a
    shared scratch filesystem used by one person is legitimate, and we cannot
    tell it apart from a genuinely shared one. The warning names the risk that
    matters most, which is not the read: another account able to write here can
    plant a well-formed Arrow file at a key we will read, and we would decode it
    as pixels.

    Windows has no POSIX mode bits (``os.chmod`` there only toggles read-only),
    and the default root under ``%LOCALAPPDATA%`` is already owner-only by
    inherited ACL, so the check is POSIX-only -- matching ``_credentials``.
    """
    existed = root.is_dir()
    root.mkdir(parents=True, mode=_DIR_MODE, exist_ok=True)
    if sys.platform == "win32" or not existed:
        return
    try:
        mode = stat.S_IMODE(root.stat().st_mode)
    except OSError:
        return
    if mode & 0o077:
        logger.warning(
            "Chunk cache dir %s is mode %o, not owner-only. Cached pixel data is "
            "readable by other local accounts, and any account that can write "
            "here can plant a chunk this process will decode as image data. "
            "Use a directory only you can access (chmod 700).",
            root,
            mode,
        )


def _parse_bytes_or(raw: Optional[str], default: Optional[int]) -> Optional[int]:
    if raw is None or not raw.strip():
        return default
    try:
        return int(parse_bytes(raw))
    except (ValueError, TypeError):
        logger.warning("Unparseable size %r; using default", raw)
        return default


def _parse_seconds_or(raw: Optional[str], default: float) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning("Unparseable duration %r; using default", raw)
        return default


def load_settings(env=None) -> Optional[Settings]:
    """Resolve the cache policy from the environment, or None when disabled.

    Disabled is the default and is also how an explicit ``0`` budget reads --
    there is no separate on/off flag, because a cache with no budget is off.
    """
    env = os.environ if env is None else env
    budget = _parse_bytes_or(env.get(ENV_BUDGET), None)
    if not budget or budget <= 0:
        return None

    raw_dir = env.get(ENV_DIR)
    root = Path(raw_dir) if raw_dir and raw_dir.strip() else cache_dir() / "chunks"

    # Refuse storage that would make this cache a liability rather than a win.
    # RAM-backed (tmpfs) is the trap worth naming: it turns "unbounded disk" into
    # unevictable RAM plus a mapping that is also RAM -- strictly worse than the
    # in-memory LRU this is meant to relieve, and invisible unless we say so.
    # Network and cloud-synced dirs break the mmap semantics the read path needs
    # (an unlinked-but-mapped inode, a page that never vanishes under the reader)
    # and stall on dehydrated-file recall.
    #
    # Loud, not silent: the user asked for this cache by setting a budget, so a
    # refusal has to say why. Disabling degrades to the in-memory fallback, which
    # is the designed floor for every other disk-cache failure too.
    reason = unsafe_cache_dir_reason(root, reject_memory=True)
    if reason:
        logger.warning(
            "Chunk cache disabled: %s is %s. Set %s to a directory on local disk.",
            root,
            reason,
            ENV_DIR,
        )
        return None

    try:
        _ensure_owner_only(root)
    except OSError as e:
        logger.warning("Chunk cache disabled: cannot create %s: %s", root, e)
        return None

    return Settings(
        root=root,
        budget=budget,
        ttl=_parse_seconds_or(env.get(ENV_TTL), _TTL_DEFAULT),
        min_free=_parse_bytes_or(env.get(ENV_MIN_FREE), _MIN_FREE_DEFAULT)
        or _MIN_FREE_DEFAULT,
    )


# ==============================================================================
# Layout
# ==============================================================================
#
# <root>/<location digest>/<ab>/<full digest>.arrow
#
# The location digest is not optional: chunk_id embeds array_id, which is
# server-*local*, not server-*unique*. In-process that is covered incidentally --
# _CACHE_POOL is keyed (location, token), so the flat chunk_id.hex() cache key
# never needed it -- but a persistent directory must put it back.
#
# The chunk digest is sha256 of the WHOLE opaque chunk_id, deliberately not the
# server's cache_key_for_chunk_id normalisation (which is server-side by design
# and unavailable here). Hashing the raw token over-keys relative to it: a legacy
# trailing method suffix keys distinctly. Over-keying costs a miss; under-keying
# serves wrong pixels.
#
# One byte of sharding keeps getdents cheap on a large cache and lets a future
# sweeper sample shards instead of ordering every file globally.


def location_key(location: str) -> str:
    return hashlib.sha256(location.encode("utf-8")).hexdigest()[:16]


def chunk_path(root: Path, location: str, chunk_id: bytes) -> Path:
    digest = hashlib.sha256(chunk_id).hexdigest()
    return root / location_key(location) / digest[:2] / f"{digest}{_SUFFIX}"


# ==============================================================================
# Read / write
# ==============================================================================


def read_batch(
    settings: Settings, location: str, chunk_id: bytes
) -> Optional[pa.RecordBatch]:
    """Read a cached chunk as an Arrow batch aliasing an mmap, or None on a miss.

    The returned batch's buffers alias the mapping, and Arrow refcounts it, so
    closing our own handle here is safe -- the munmap waits for the last Buffer,
    which the caller's decoded array keeps alive. Untouched pages are never
    faulted in, so a partial read stays cheap. Decoding is the caller's job
    (``_pool`` owns the unified-schema decode), which also keeps this module free
    of an import cycle.

    Enforces the TTL lazily off the ``stat`` the read needs anyway, and bumps a
    stale mtime so eviction has a recency signal. Any failure is a miss: a file
    that does not parse is one the caller refetches, which is the whole recovery
    story.
    """
    path = chunk_path(settings.root, location, chunk_id)
    try:
        st = os.stat(path)
    except OSError:
        return None

    now = time.time()
    age = now - st.st_mtime
    if age > settings.ttl:
        _unlink_quietly(path)
        return None

    if age > _MTIME_BUMP_AFTER:
        try:
            os.utime(path, None)
        except OSError:
            pass  # read-only or racing eviction; recency is best-effort

    try:
        mm = pa.memory_map(str(path), "r")
        try:
            batch = pa.ipc.open_stream(mm).read_next_batch()
        finally:
            mm.close()
        return batch
    except (OSError, pa.ArrowInvalid, StopIteration) as e:
        # Truncated by a crash mid-write, or written by an incompatible Arrow.
        logger.debug("disk-cache read failed for %s, refetching: %s", path.name, e)
        _unlink_quietly(path)
        return None


def write_batch(
    settings: Settings, location: str, chunk_id: bytes, batch: pa.RecordBatch
) -> Optional[pa.RecordBatch]:
    """Persist a freshly fetched chunk, then hand back an mmap-backed batch.

    Written to a per-(pid, tid) temp name and ``os.rename``d into place -- atomic
    on the same filesystem, so a reader never sees a torn file and no lock is
    needed on the write path. N workers racing one miss cost N ``do_get``s, which
    is what happens today anyway.

    Returning the *re-read* batch is what makes this free rather than a tax: the
    caller drops its private in-memory buffer and holds a view onto the shared
    mapping instead, so the copy count is unchanged and every chunk ends up in
    one representation regardless of which path fetched it. None on any failure,
    leaving the caller to use what it already has.
    """
    path = chunk_path(settings.root, location, chunk_id)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        # One level at a time: Path.mkdir applies `mode` only to the FINAL
        # component, so a parents=True call would leave the location dir at the
        # umask default. The 0o700 root blocks traversal either way, but the
        # tree should not depend on one directory's mode for the whole boundary.
        shard = path.parent
        shard.parent.mkdir(parents=True, mode=_DIR_MODE, exist_ok=True)
        shard.mkdir(mode=_DIR_MODE, exist_ok=True)
        with pa.OSFile(str(tmp), "wb") as sink:
            with pa.ipc.new_stream(sink, batch.schema) as writer:
                writer.write_batch(batch)
        # Defence in depth behind the 0o700 root: one chmod on a path that just
        # wrote megabytes, and it keeps the file owner-only even if the root's
        # mode is later widened. POSIX only -- see _ensure_owner_only.
        if sys.platform != "win32":
            os.chmod(tmp, _FILE_MODE)
        os.rename(tmp, path)
    except OSError as e:
        # ENOSPC, a read-only mount, a vanished parent: the cache is a bonus, so
        # a write failure degrades to the in-memory path rather than raising.
        logger.debug("disk-cache write failed for %s: %s", path.name, e)
        _unlink_quietly(tmp)
        return None

    note_written(settings, batch.nbytes)
    return read_batch(settings, location, chunk_id)


def _unlink_quietly(path: Path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# ==============================================================================
# Sweeping
# ==============================================================================
#
# Correctness needs no coordination -- unlink-while-mapped is safe and a stale
# path self-heals into a miss. Policy needs exactly one thing: mutual exclusion,
# so N processes do not all scan-and-evict at once and over-evict by a factor of
# N, each of them seeing the same pre-sweep total.
#
# ExclusiveFileLock is that token, and it is already the SDK's: stdlib-only,
# cross-platform, and held on a descriptor, so the OS drops it when the holder
# *dies*. That matters here precisely because the N processes are dask workers
# that get killed -- a lockfile with a pid record would need reaping, this needs
# none.

_written_since_sweep = 0
_written_lock = threading.Lock()


def note_written(settings: Settings, nbytes: int) -> None:
    """Account a write and sweep once this process has written enough."""
    global _written_since_sweep
    with _written_lock:
        _written_since_sweep += nbytes
        if _written_since_sweep < settings.sweep_after:
            return
        _written_since_sweep = 0
    maybe_sweep(settings)


def reset_write_accounting() -> None:
    """Drop the sweep trigger's accumulator (at-fork, and for tests)."""
    global _written_since_sweep
    with _written_lock:
        _written_since_sweep = 0


def maybe_sweep(settings: Settings) -> bool:
    """Sweep if no other process is already doing it. True if this one swept."""
    lock = ExclusiveFileLock(settings.root / _LOCK_NAME)
    try:
        if not lock.acquire(timeout=0.0):
            return False
    except OSError:
        return False
    try:
        _sweep(settings)
    except OSError as e:
        logger.debug("disk-cache sweep failed: %s", e)
    finally:
        lock.release()
    return True


def _free_bytes(path: Path) -> Optional[int]:
    try:
        st = os.statvfs(path)
    except (OSError, AttributeError):  # AttributeError: no statvfs on Windows
        return _free_bytes_portable(path)
    return st.f_bavail * st.f_frsize


def _free_bytes_portable(path: Path) -> Optional[int]:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def _scan(
    root: Path, ttl: float, now: float
) -> Tuple[List[Tuple[float, int, Path]], int]:
    """Collect ``(mtime, size, path)`` for every live entry, expiring by TTL.

    TTL is applied here as well as on the read path because a lazy check only
    reclaims files someone touches -- and the files never touched again are
    exactly a single-pass scan's garbage, the case the TTL most needs to collect.
    Free, since the sweep is stat-ing everything regardless.
    """
    entries: List[Tuple[float, int, Path]] = []
    total = 0
    for path in root.rglob(f"*{_SUFFIX}"):
        try:
            st = path.stat()
        except OSError:
            continue
        if now - st.st_mtime > ttl:
            _unlink_quietly(path)
            continue
        entries.append((st.st_mtime, st.st_size, path))
        total += st.st_size
    return entries, total


def _sweep(settings: Settings) -> None:
    """Expire by TTL, then evict oldest-first until under budget and above floor.

    Two independent pressures, one eviction pass: the byte budget bounds what the
    cache *occupies*, the free-space floor bounds what it leaves for everything
    else on the machine. Whichever demands more bytes back sets the target.
    """
    root = settings.root
    if not root.is_dir():
        return

    now = time.time()
    entries, total = _scan(root, settings.ttl, now)

    over_budget = total - settings.budget
    free = _free_bytes(root)
    under_floor = 0 if free is None else settings.min_free - free
    # Evicting N bytes both lowers the total and raises free space by N, so one
    # target satisfies both constraints.
    to_free = max(over_budget, under_floor)
    if to_free <= 0:
        return

    entries.sort(key=lambda e: e[0])  # oldest mtime first
    freed = 0
    for _mtime, size, path in entries:
        if freed >= to_free:
            break
        _unlink_quietly(path)
        freed += size

    logger.debug(
        "disk-cache sweep: freed %d B of %d B target (total %d, budget %d)",
        freed,
        to_free,
        total,
        settings.budget,
    )
