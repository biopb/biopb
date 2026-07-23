"""Pickle-safe worker-side connection/cache pool and chunk-fetch subsystem.

Extracted from :mod:`biopb.tensor.client` (issue #278 item C): the dask arrays
returned by ``TensorFlightClient`` fetch chunks lazily, so the leaf tasks must be
picklable and reconnect per worker process. This module owns exactly that
subsystem and holds no reference to ``TensorFlightClient``:

- **Connection/cache pools** keyed by ``(location, token)``: a per-thread
  ``FlightClient`` for lock-free access, plus a cross-thread ``Cache`` and
  ``FlightCallOptions``. Fork-safe (each entry records the pid that created it).
- **The localhost cache-file fast path** (issue #9): read a chunk straight from
  the server's on-disk segment via ``chunk_locate`` + mmap instead of ``do_get``.
- **The chunk-fetch leaf functions and dask-array builder**: pickle-safe because
  they close over no ``FlightClient`` -- connections/caches/call options are
  recreated lazily per worker from the module-level pools above.

``client.py`` re-exports these names, so ``biopb.tensor.client.<name>`` stays a
stable import surface for existing callers (e.g. biopb-mcp's ``configure_cache``
worker plugin, the cachefile / connection-pool tests, and the benchmarks).
"""

import atexit
import json
import logging
import os
import threading
import weakref
from functools import cache
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from cachey import Cache
from dask.base import tokenize
from dask.blockwise import BlockwiseDep, BlockwiseDepDict, blockwise as _blockwise
from dask.delayed import delayed
from dask.highlevelgraph import HighLevelGraph

from biopb.tensor.ticket_pb2 import TensorTicket

logger = logging.getLogger(__name__)


# ==============================================================================
# Cache-file transfer optimization (localhost fast path, issue #9)
# ==============================================================================
#
# On localhost the tensor server's file cache already holds every decoded chunk
# as an Arrow IPC message in a segment file. Instead of re-sending those bytes
# through the loopback gRPC socket (do_get), the client asks the server to
# locate the chunk (chunk_locate action), then mmaps the segment file and reads
# just that message -- handing out a zero-copy view onto the mapping (Option C,
# biopb/biopb#571). The client closes its own MemoryMappedFile handle at once,
# but Arrow refcounts the mapping so the returned array keeps it alive (the
# munmap waits for the last Buffer); untouched chunk pages are never faulted, so
# a partial read is nearly free. Gated to POSIX (Windows file-mmap blocks
# segment unlink -- see biopb/biopb#5). Safety rests on the server never
# truncating a mapped segment inode (see file_backend `_create_segment`); an
# NFS cache_dir would violate that and wants an explicit gate before this path
# runs there.
#
# A held view also *pins* its segment's disk on the server (an evicted/unlinked
# segment's blocks survive to the last close), so the process caps the total
# mapped-segment size and copies the chunk out once over budget -- see the
# pinned-segment accounting below (disk-leak workaround, biopb/biopb#571).

# Highest on-disk segment format version this client can parse. The client
# reads server-written segment bytes directly, so the layout is a cross-process
# contract: chunk_locate reports the server's CACHE_FILE_FORMAT_VERSION, and we
# decline the fast path (fall back to do_get) for anything newer than this
# rather than risk misreading the mmap. Bump in lockstep with the server when
# this client learns to parse a newer format.
_CACHEFILE_SUPPORTED_FORMAT = 1

# Per-location capability cache: dask workers are separate processes, so each
# memoizes independently after its first probe. None = unknown, False = the
# server doesn't support chunk_locate (old server) so don't retry.
_cachefile_support: Dict[str, bool] = {}
_cachefile_support_lock = threading.Lock()


def _is_cachefile_disabled_by_env() -> bool:
    """Whether the cache-file fast path is explicitly disabled via env var."""
    return os.environ.get("BIOPB_CACHEFILE_TRANSFER_DISABLED", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _cachefile_supported(location: str) -> Optional[bool]:
    with _cachefile_support_lock:
        return _cachefile_support.get(location)


def _set_cachefile_supported(location: str, supported: bool) -> None:
    with _cachefile_support_lock:
        _cachefile_support[location] = supported


def _cache_local_enabled() -> bool:
    """Whether to keep a client-side chunk cache for a *localhost* server.

    Default ``False``: on localhost the tensor server already caches its data
    (and the cache-file fast path makes a re-fetch cheap), so a per-process client
    cache is mostly a redundant second copy -- and under a multi-process dask
    cluster it is replicated in every worker, multiplying memory use. Set
    ``BIOPB_CACHE_LOCAL=1`` to opt back in (e.g. a slow loopback proxy).
    """
    return os.environ.get("BIOPB_CACHE_LOCAL", "").lower() in ("1", "true", "yes")


def _resolve_cache_bytes(location: str, requested: int) -> int:
    """Effective per-process chunk-cache size for a connection.

    Single point that decides whether (and how big) a client-side cache is.
    Folds in the localhost rule: a localhost server gets no cache (returns 0)
    unless explicitly re-enabled; otherwise the requested size is honored.
    """
    if requested <= 0:
        return 0
    if not _cache_local_enabled() and _is_localhost_location(location):
        return 0
    return requested


@cache
def _is_localhost_location(location: str) -> bool:
    """Check if location points to localhost.

    Parses location URI and checks hostname against localhost variants.
    Uses socket.getaddrinfo() to resolve hostname to loopback.

    Memoized: this is called on every chunk fetch (cache-policy + fast-path checks)
    and may do a DNS lookup, so the per-location result is cached for the life
    of the process.

    Args:
        location: Flight server location string (e.g., "grpc://localhost:8815")

    Returns:
        True if location resolves to loopback address
    """
    import re
    import socket

    # Parse location URI - handle various formats
    # grpc://hostname:port, grpc+tls://hostname:port, hostname:port
    # IPv6 format: grpc://[::1]:port
    match = re.match(
        r"^(?:grpc(?:\+tls)?://)?(?:\[([^\]]+)\]|([^:]+))(?:\:\d+)?$", location
    )
    if not match:
        return False

    # IPv6 bracketed or regular hostname
    hostname = match.group(1) or match.group(2)

    # Direct localhost matches
    localhost_names = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
    if hostname.lower() in localhost_names or hostname in localhost_names:
        return True

    # Resolve hostname via getaddrinfo
    try:
        addrinfo = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in addrinfo:
            # Check if address is loopback
            if (
                family == socket.AF_INET
                and sockaddr[0] == "127.0.0.1"
                or family == socket.AF_INET6
                and sockaddr[0] == "::1"
            ):
                return True
    except socket.gaierror:
        # Hostname resolution failed, not localhost
        pass

    return False


def _should_try_cachefile(location: str) -> bool:
    """Whether to attempt the localhost cache-file fast path for this location.

    Requires: not disabled by env, a POSIX host (Windows file-mmap blocks the
    server's segment unlink -- biopb/biopb#5), a loopback server (shared
    filesystem), and a server not already known to lack chunk_locate.
    """
    if _is_cachefile_disabled_by_env():
        return False
    if os.name != "posix":
        return False
    if not _is_localhost_location(location):
        return False
    return _cachefile_supported(location) is not False


def _decode_unified_batch(batch: pa.RecordBatch) -> Tuple[np.ndarray, pa.Buffer]:
    """Decode the server's unified cache schema into a read-only array *view*
    and the Arrow buffer backing it.

    The file cache stores each chunk as ``[data: binary, shape: list<int64>,
    dtype: string, ...]`` where ``data`` is the raw C-contiguous bytes of the
    raveled array. Raw bytes are reinterpreted via the dtype string, so
    endianness round-trips (biopb/biopb#293).

    The returned array aliases ``data_buf`` (numpy holds it through the buffer
    protocol), so the two share a lifetime. A caller that must keep the buffer's
    backing alive -- or anchor a finalizer to it, as the pinned-segment
    accounting does -- uses the returned buffer, which is the exact object the
    array references (not a fresh wrapper, whose lifetime would be independent).
    """
    dtype = np.dtype(batch.column("dtype")[0].as_py())
    shape = tuple(batch.column("shape").to_pylist()[0])
    count = int(np.prod(shape)) if shape else 0
    # binary array buffers = [validity, offsets, data]; data holds the blob.
    data_buf = batch.column("data").buffers()[2]
    arr = np.frombuffer(data_buf, dtype=dtype, count=count).reshape(shape)
    arr.flags.writeable = False
    return arr, data_buf


def _array_from_unified_batch(
    batch: pa.RecordBatch, *, copy: bool = True
) -> np.ndarray:
    """Reconstruct a numpy array from the server's unified cache schema.

    The result is **read-only** (``writeable=False``) regardless of ``copy``, so
    the contract a consumer sees is uniform whether a chunk arrived over
    ``do_get`` or the localhost mmap fast path -- otherwise the same user code
    would succeed or raise depending on platform and deployment
    (biopb/biopb#571). A chunk is a shared, cached view of server-owned bytes;
    mutating one in place is never safe, so callers must copy before writing.

    ``copy=False`` (Option C, biopb/biopb#571) hands out a zero-copy view of the
    batch's Arrow buffer, kept alive through the buffer protocol -- the default
    for both production paths. The underlying buffer is already immutable, so the
    former unconditional ``.copy()`` was pure overhead, and for a 64 MB chunk it
    fell off glibc's 32 MiB mmap-threshold cliff.

    ``copy=True`` (the default here) is the owned-copy fallback for callers that
    must *not* alias server memory -- e.g. the mmap fast path once it is over its
    pinned-segment budget (release the mapping now, don't pin more server disk),
    or a cache backing that can be truncated under the reader (NFS). It allocates
    a fresh buffer and freezes it read-only to match.
    """
    arr, _ = _decode_unified_batch(batch)
    if copy:
        arr = arr.copy()
        arr.flags.writeable = False
    return arr


# ------------------------------------------------------------------------------
# Pinned-segment accounting (disk-leak workaround, biopb/biopb#571)
# ------------------------------------------------------------------------------
#
# A fast-path view keeps its segment file's mapping alive, so the server cannot
# reclaim that segment's disk blocks while the client holds the array -- even
# after eviction unlinks the file (the inode survives to the last close). A
# client that holds many views can thus keep the server's cache_dir above its
# configured budget.
#
# Bound it: track the on-disk size of the distinct segments this process keeps
# mapped, and once that crosses a threshold, copy the chunk out and let the
# mapping go (copy=True) instead of handing out another view -- so no further
# segment is pinned until some views drop. Refcounted per inode: many chunks
# from one segment pin it once; a segment un-pins when the last view of it is
# garbage-collected (a weakref.finalize on the backing Arrow buffer).
#
# Kept cheap on the hot read path: the gate is a lock-free read of a plain int;
# only the view branch pays a lock + one weakref.finalize (per chunk actually
# mapped), and the segment size is taken from the stat the fast path already does.

_PIN_LIMIT_DEFAULT = 16 * 1024 * 1024 * 1024  # 16 GiB mapped before copying


def _pin_limit_bytes() -> int:
    """Max on-disk segment bytes this process keeps mmap-pinned before the fast
    path falls back to copying. ``BIOPB_CACHEFILE_PIN_LIMIT_BYTES`` overrides;
    ``0`` forces every fast-path read to copy (no view is ever handed out); a
    negative or unparseable value uses the default."""
    raw = os.environ.get("BIOPB_CACHEFILE_PIN_LIMIT_BYTES")
    if raw is None:
        return _PIN_LIMIT_DEFAULT
    try:
        val = int(raw)
    except ValueError:
        return _PIN_LIMIT_DEFAULT
    return val if val >= 0 else _PIN_LIMIT_DEFAULT


class _SegmentPin:
    __slots__ = ("size", "refs")

    def __init__(self, size: int):
        self.size = size
        self.refs = 0


_pinned_lock = threading.Lock()
_pinned_segments: Dict[int, _SegmentPin] = {}  # segment inode -> pin record
_pinned_total = 0  # sum of sizes of currently-pinned segments (bytes)


def _pin_budget_exhausted() -> bool:
    """Whether the process is at/above its pinned-segment budget.

    Lock-free by design: ``_pinned_total`` is a plain int (atomic under the GIL)
    and the bound is a heuristic, so a momentarily stale read is acceptable and
    the localhost hot read path stays clear of lock traffic.
    """
    return _pinned_total >= _pin_limit_bytes()


def _register_segment_pin(inode: int, size: int, anchor: object) -> None:
    """Charge a fast-path view against its segment, and release the charge when
    the view is collected.

    ``anchor`` must be the Arrow buffer the returned array actually holds (its
    ``base`` chain), so the finalizer fires exactly when the last array derived
    from this read -- and thus the last hold on the mapping -- is gone.
    Refcounted by inode so many chunks read from one segment count its disk once.
    """
    global _pinned_total
    with _pinned_lock:
        pin = _pinned_segments.get(inode)
        if pin is None:
            pin = _pinned_segments[inode] = _SegmentPin(size)
            _pinned_total += size
        pin.refs += 1
    weakref.finalize(anchor, _release_segment_pin, inode)


def _release_segment_pin(inode: int) -> None:
    """Drop one view's charge against ``inode``; un-pin the segment on the last."""
    global _pinned_total
    with _pinned_lock:
        pin = _pinned_segments.get(inode)
        if pin is None:
            return
        pin.refs -= 1
        if pin.refs <= 0:
            _pinned_total -= pin.size
            del _pinned_segments[inode]


def _try_cachefile_transfer(
    client: flight.FlightClient,
    location: str,
    token: Optional[str],
    chunk_id: bytes,
    call_options: flight.FlightCallOptions,
) -> Optional[np.ndarray]:
    """Attempt the cache-file fast path for a chunk.

    Asks the server to locate the chunk on disk (chunk_locate), then mmaps the
    segment file and reads the single IPC message. Normally hands out a zero-copy
    view onto the mapping (Option C); once the process is over its pinned-segment
    budget it copies the chunk out and releases the mapping instead. Returns the
    array, or None to fall back to do_get (server too old, chunk not
    cached/locatable, or any read failure).
    """
    ticket = TensorTicket(chunk_id=chunk_id)
    action = flight.Action("chunk_locate", ticket.SerializeToString())

    try:
        results = client.do_action(action, options=call_options)
        payload = next(results).body.to_pybytes().decode("utf-8")
    except flight.FlightError as e:
        # An old server doesn't know the action -- stop probing this location.
        if "Unknown action" in str(e):
            _set_cachefile_supported(location, False)
        else:
            logger.debug(f"chunk_locate failed, falling back to do_get: {e}")
        return None
    except Exception as e:
        logger.debug(f"chunk_locate failed, falling back to do_get: {e}")
        return None

    _set_cachefile_supported(location, True)

    try:
        info = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if not info.get("available"):
        return None

    # The segment layout is a cross-process contract; refuse to parse a format
    # newer than we understand. The server's format won't change mid-session,
    # so stop probing this location.
    if int(info.get("format_version", 1)) > _CACHEFILE_SUPPORTED_FORMAT:
        logger.debug(
            "chunk_locate reports segment format %s > supported %s; using do_get",
            info.get("format_version"),
            _CACHEFILE_SUPPORTED_FORMAT,
        )
        _set_cachefile_supported(location, False)
        return None

    try:
        segment_path = info["segment_path"]
        byte_offset = int(info["byte_offset"])
        generation_id = int(info["generation_id"])
        # Detect a segment evicted and recreated at the same path before we map.
        # Reuse this stat's st_size as the segment's pinned-disk cost below, so
        # the accounting adds no extra syscall.
        st = os.stat(segment_path)
        if st.st_ino != generation_id:
            return None
        mm = pa.memory_map(segment_path, "r")
        try:
            schema = pa.ipc.open_stream(mm).schema
            mm.seek(byte_offset)
            msg = pa.ipc.read_message(mm)
            batch = pa.ipc.read_record_batch(msg, schema)
            # Option C (biopb/biopb#571): hand out a zero-copy view onto the
            # mapping instead of copying out of it. The IPC-decoded data buffer
            # aliases the mmap, and Arrow refcounts the mapping -- ``mm.close()``
            # below drops *this* handle but the munmap waits for the last Buffer,
            # which the returned array keeps alive through
            #   ndarray -> pyarrow.Buffer -> MemoryMappedFile -> fd + mapping.
            # So closing here is still correct. This makes partial reads nearly
            # free: untouched chunk pages are never faulted in.
            #
            # The view keeps the server from reclaiming this segment's disk while
            # we hold it, so once the process is over its pinned-segment budget we
            # copy the chunk out and let the mapping go instead -- still off the
            # warm page cache (no do_get), just bounding the disk-leak.
            if _pin_budget_exhausted():
                arr = _array_from_unified_batch(batch, copy=True)
            else:
                arr, data_buf = _decode_unified_batch(batch)
                _register_segment_pin(generation_id, st.st_size, data_buf)
        finally:
            mm.close()
        logger.debug(f"_try_cachefile_transfer: read {arr.nbytes} bytes via mmap")
        return arr
    except (OSError, ValueError, KeyError, pa.ArrowInvalid) as e:
        logger.debug(f"cache-file read failed, falling back to do_get: {e}")
        return None


# ==============================================================================
# Module-level pools for worker-local connection caching (pickle-safe)
# ==============================================================================
#
# FlightClient connections are stored per-thread for lock-free access.
# Cache and CallOptions remain shared across threads for cross-thread cache hits.
#
# Fork-safety: Each thread stores (pid, client) to detect forked processes.
# When a process forks, child threads detect pid mismatch and create fresh
# connections (inherited gRPC sockets are broken).
#

# Per-thread storage: thread gets its own FlightClient per (location, token)
_THREAD_LOCAL = threading.local()

# Global registry for cleanup: thread_id -> {(location, token): FlightClient}
_CONNECTION_REGISTRY: Dict[
    int, Dict[Tuple[str, Optional[str]], flight.FlightClient]
] = {}
_REGISTRY_LOCK = threading.Lock()

# Shared pools for cache and call options (cross-thread cache hits enabled)
#
# Cache pool value is tri-state per (location, token):
#   - absent          -> never configured; first fetch creates a default cache
#   - (pid, Cache)    -> pinned/created with that budget
#   - (pid, None)     -> deliberately pinned OFF by configure_cache(); a later
#                        fetch must honor this and NOT recreate a cache.
_CACHE_POOL: Dict[Tuple[str, Optional[str]], Tuple[int, Optional[Cache]]] = {}
_CALL_OPTS_POOL: Dict[Tuple[str, Optional[str]], flight.FlightCallOptions] = {}
_POOL_LOCK = threading.Lock()


@atexit.register
def _cleanup_connection_pool():
    """Clean up all pooled FlightClient connections on process exit."""
    with _REGISTRY_LOCK:
        for _thread_id, clients in _CONNECTION_REGISTRY.items():
            for _key, client in clients.items():
                try:
                    client.close()
                except Exception:
                    pass
        _CONNECTION_REGISTRY.clear()

    with _POOL_LOCK:
        _CACHE_POOL.clear()
        _CALL_OPTS_POOL.clear()


def _evict_dead_threads():
    """Close connections from threads that have died."""
    for thread_id, clients in list(_CONNECTION_REGISTRY.items()):
        # Check if thread is alive using threading._active (tracks all live threads)
        if thread_id not in threading._active:
            # Thread died - close all its connections
            for _key, client in clients.items():
                try:
                    client.close()
                except Exception:
                    pass
            del _CONNECTION_REGISTRY[thread_id]


def _get_thread_client(location: str, token: Optional[str]) -> flight.FlightClient:
    """Get thread-local FlightClient (no lock for read access).

    Creates FlightClient lazily on first call per thread. Thread-safe via
    thread-local storage. Fork-safe: stale connections from parent process
    are detected and cleaned up before use.

    Args:
        location: Flight server location string
        token: Bearer token (or None for no auth)

    Returns:
        FlightClient for this thread and location
    """
    key = (location, token)
    current_pid = os.getpid()

    # Fast path: thread already has client for this location (no lock)
    local_pool = getattr(_THREAD_LOCAL, "clients", None)
    if local_pool is None:
        local_pool = {}
        _THREAD_LOCAL.clients = local_pool
    elif key in local_pool:
        pid_client = local_pool[key]
        if isinstance(pid_client, tuple):
            pid, client = pid_client
            if pid == current_pid:
                return client
            # Forked child - inherited gRPC socket is broken, create new
            try:
                client.close()
            except Exception:
                pass
            del local_pool[key]
        else:
            # Legacy format (shouldn't happen, but handle gracefully)
            return pid_client

    # Slow path: create new client with gRPC options tuned for 64MB chunks, register for cleanup
    # 80MB max message size (slightly above 64MB chunk threshold)
    client = flight.FlightClient(
        location,
        generic_options=[
            ("grpc.max_send_message_size", 80 * 1024 * 1024),
            ("grpc.max_receive_message_size", 80 * 1024 * 1024),
        ],
    )
    local_pool[key] = (current_pid, client)

    thread_id = threading.current_thread().ident
    if thread_id is not None:
        with _REGISTRY_LOCK:
            # Eviction: clean up dead threads
            _evict_dead_threads()
            # Register this thread's connection
            if thread_id not in _CONNECTION_REGISTRY:
                _CONNECTION_REGISTRY[thread_id] = {}
            _CONNECTION_REGISTRY[thread_id][key] = client

    return client


def _get_shared_cache(
    location: str, token: Optional[str], cache_bytes: int
) -> Optional[Cache]:
    """Get shared Cache for cross-thread cache hits, or None if caching is off.

    The requested size is run through :func:`_resolve_cache_bytes`, so a
    localhost server (default) yields ``None`` and no Cache is ever allocated.

    Args:
        location: Flight server location string
        token: Bearer token (or None for no auth)
        cache_bytes: Requested cache size for worker-local cache

    Returns:
        Cache instance shared across threads, or None when caching is disabled.
    """
    key = (location, token)
    current_pid = os.getpid()

    # An already-pooled entry wins: it may have been pinned by configure_cache()
    # at worker startup, in which case the per-fetch cache_bytes is irrelevant.
    # The stored cache may be None -- a sentinel meaning "pinned OFF" -- which we
    # return as-is so the decision is not undone by this fetch's cache_bytes.
    with _POOL_LOCK:
        if key in _CACHE_POOL:
            pool_pid, cache = _CACHE_POOL[key]
            if pool_pid == current_pid:
                return cache
            del _CACHE_POOL[key]  # fork-safety: inherited stale entry

    # Not pooled yet: resolve the requested size and create lazily (or skip).
    effective = _resolve_cache_bytes(location, cache_bytes)
    if effective <= 0:
        return None

    with _POOL_LOCK:
        if key not in _CACHE_POOL:
            _CACHE_POOL[key] = (current_pid, Cache(available_bytes=effective))
        return _CACHE_POOL[key][1]


def _get_shared_call_options(
    location: str, token: Optional[str]
) -> flight.FlightCallOptions:
    """Get shared FlightCallOptions.

    Args:
        location: Flight server location string
        token: Bearer token (or None for no auth)

    Returns:
        FlightCallOptions for this connection
    """
    key = (location, token)

    with _POOL_LOCK:
        if key not in _CALL_OPTS_POOL:
            if token:
                _CALL_OPTS_POOL[key] = flight.FlightCallOptions(
                    headers=[(b"authorization", f"Bearer {token}".encode())]
                )
            else:
                _CALL_OPTS_POOL[key] = flight.FlightCallOptions()

    return _CALL_OPTS_POOL[key]


def _get_worker_resources(location: str, token: Optional[str], cache_bytes: int):
    """Get cached FlightClient, Cache, and CallOptions for a connection namespace.

    Creates resources lazily on first call per (location, token) key.
    FlightClient is per-thread for lock-free access. Cache and CallOptions
    are shared across threads for cross-thread cache hits.

    Each worker process has its own pool after unpickle. If a process
    forks after pool was populated, the child detects pid mismatch and
    creates fresh connections.

    Args:
        location: Flight server location string
        token: Bearer token (or None for no auth)
        cache_bytes: Cache size for worker-local cache

    Returns:
        Tuple of (FlightClient, Optional[Cache], FlightCallOptions). The cache
        is None when caching is disabled for this connection (e.g. localhost).
    """
    client = _get_thread_client(location, token)
    cache = _get_shared_cache(location, token, cache_bytes)
    call_options = _get_shared_call_options(location, token)

    return (client, cache, call_options)


def configure_cache(location: str, token: Optional[str], cache_bytes: int) -> int:
    """Pin this process's chunk-cache budget for a connection, authoritatively.

    Sets the per-process chunk cache to ``cache_bytes`` and keeps it there: every
    later fetch honors this budget regardless of the ``cache_bytes`` it requests.
    Idempotent. Call it once per worker process (e.g. from a dask worker-init
    plugin) to fix the budget deterministically across a dynamically-sized
    cluster.

    A localhost server (the default) or ``cache_bytes <= 0`` pins the cache OFF;
    later fetches then skip caching rather than recreating one of their own.

    Args:
        location: Flight server location string
        token: Bearer token (or None for no auth)
        cache_bytes: Requested per-process cache size in bytes

    Returns:
        The effective (resolved) cache size that was pinned, in bytes.
    """
    # Unlike the lazy first-touch creation in _get_shared_cache, this (re)sizes
    # the pooled cache now and records a None sentinel for the disabled case, so
    # a later fetch can't undo the decision from its own cache_bytes.
    effective = _resolve_cache_bytes(location, cache_bytes)
    key = (location, token)
    current_pid = os.getpid()

    with _POOL_LOCK:
        existing = _CACHE_POOL.get(key)
        if effective <= 0:
            # Pin OFF authoritatively: store a None-cache sentinel instead of
            # deleting, so a later fetch's _get_shared_cache honors the decision
            # rather than recreating a cache from its own cache_bytes.
            _CACHE_POOL[key] = (current_pid, None)
            return 0
        # (Re)create when absent, inherited from a fork, or a different size.
        if (
            existing is None
            or existing[0] != current_pid
            or existing[1].available_bytes != effective
        ):
            _CACHE_POOL[key] = (current_pid, Cache(available_bytes=effective))

    return effective


def _fetch_chunk_distributed(
    location: str,
    token: Optional[str],
    chunk_id: bytes,
    bounds_start: Tuple[int, ...],
    bounds_stop: Tuple[int, ...],
    cache_bytes: int,
    schema_metadata: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Fetch a chunk from Flight server using worker-local resources.

    This function is pickle-safe because it has no closure references to
    non-serializable objects (FlightClient). Connection and cache are
    obtained from module-level pools at runtime.

    For POSIX localhost connections, attempts the cache-file fast path
    (chunk_locate + mmap) first before falling back to do_get().

    Args:
        location: Flight server location string
        token: Bearer token (or None for no auth)
        chunk_id: Chunk identifier bytes
        bounds_start: Chunk start coordinates as tuple
        bounds_stop: Chunk stop coordinates as tuple
        cache_bytes: Cache size for worker-local cache
        schema_metadata: Optional schema metadata dict. Not used by the
            cache-file fast path (support is probed via chunk_locate); retained
            for signature compatibility with the chunk-fetch call sites.

    Returns:
        numpy array with chunk data
    """
    client, cache, call_options = _get_worker_resources(location, token, cache_bytes)

    # Cache lookup (cache is None when caching is disabled, e.g. localhost)
    cache_key = chunk_id.hex()
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug(f"fetch_chunk_distributed: cache hit for {cache_key[:16]}")
            return cached

    logger.debug(f"fetch_chunk_distributed: fetching {cache_key[:16]} from server")

    arr = None

    # Try the localhost cache-file fast path if all conditions met (issue #9)
    if _should_try_cachefile(location):
        arr = _try_cachefile_transfer(client, location, token, chunk_id, call_options)

    # Fallback to do_get if the fast path wasn't attempted or failed
    if arr is None:
        ticket = TensorTicket(chunk_id=chunk_id)
        reader = client.do_get(
            flight.Ticket(ticket.SerializeToString()), options=call_options
        )
        # do_get returns a single-row unified binary batch [data, shape, dtype];
        # decode it exactly like the cache-file fast path (raw bytes reinterpreted
        # via the dtype string, so endianness round-trips -- biopb/biopb#293).
        # copy=False: the batch's Arrow buffer is in-memory and numpy keeps it
        # alive, so a view is safe here and skips a needless whole-chunk copy
        # (biopb/biopb#571). The mmap fast path above hands out a view too, but
        # for a different reason: there the buffer aliases the segment *mapping*,
        # which Arrow refcounts so it outlives that path's local ``mm.close()``.
        # Both paths return a read-only array.
        arr = _array_from_unified_batch(reader.read_all().to_batches()[0], copy=False)

    # Cache the result (skipped when caching is disabled)
    if cache is not None:
        cache.put(cache_key, arr, cost=arr.nbytes)

    return arr


def _fetch_chunk_block(
    dep: Tuple[bytes, Tuple[int, ...], Tuple[int, ...]],
    location: str,
    token: Optional[str],
    cache_bytes: int,
    schema_metadata: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Single-``Blockwise``-layer callback that fetches one block.

    The regular-grid construction (see ``_build_dask_array_from_chunk_map``)
    routes here. ``dep`` is the block's *own* ``(chunk_id, bounds_start,
    bounds_stop)`` triple, delivered per block by a ``BlockwiseDepDict`` so that
    only this block's chunk id/bounds ride in its task -- not the whole array's
    chunk table. That keeps a partial read's graph size O(blocks read) instead of
    O(blocks total): slicing one plane out of an N-chunk tensor no longer drags
    all N chunk ids through the graph (biopb/biopb#278). Pickle-safe for the same
    reason as :func:`_fetch_chunk_distributed`: no closure over a FlightClient.
    """
    chunk_id, bounds_start, bounds_stop = dep
    return _fetch_chunk_distributed(
        location,
        token,
        chunk_id,
        tuple(bounds_start),
        tuple(bounds_stop),
        cache_bytes,
        schema_metadata,
    )


def _regular_grid_chunks(
    chunk_map: Dict[Tuple[int, ...], Tuple[bytes, Any]],
    grid_shape: Tuple[int, ...],
    shape: Tuple[int, ...],
) -> Optional[Tuple[Tuple[int, ...], ...]]:
    """Return the dask ``chunks`` tuple iff the grid is a regular tiling.

    A regular tiling is a full Cartesian product of blocks whose per-axis sizes
    are *separable* (the extent along axis *d* depends only on the block's index
    along *d*) and contiguous, covering the full shape. That is exactly the
    precondition for representing the array as a single ``Blockwise`` (map_blocks)
    layer instead of an O(n_chunks)-layer ``da.block`` graph.

    Returns the per-axis chunk-size tuple (suitable for ``da.map_blocks``'s
    ``chunks=`` argument), or ``None`` if the grid is ragged/sparse and the
    caller must fall back to ``da.block``.
    """
    ndim = len(shape)
    axis_chunks: List[Tuple[int, ...]] = []
    for axis in range(ndim):
        # index along this axis -> (start, stop), verified consistent
        extents: Dict[int, Tuple[int, int]] = {}
        for chunk_idx, (_chunk_id, bounds) in chunk_map.items():
            i = chunk_idx[axis]
            extent = (int(bounds.start[axis]), int(bounds.stop[axis]))
            if extents.setdefault(i, extent) != extent:
                return None  # ragged: same index, different extent
        if len(extents) != grid_shape[axis]:
            return None  # missing indices along this axis
        sizes: List[int] = []
        expected_start = 0
        for i in range(grid_shape[axis]):
            if i not in extents:
                return None
            start, stop = extents[i]
            if start != expected_start:
                return None  # non-contiguous tiling
            sizes.append(stop - start)
            expected_start = stop
        if expected_start != shape[axis]:
            return None  # does not cover the full extent
        axis_chunks.append(tuple(sizes))

    n_blocks = 1
    for g in grid_shape:
        n_blocks *= g
    if len(chunk_map) != n_blocks:
        return None  # sparse grid (Cartesian product not fully populated)

    return tuple(axis_chunks)


def _build_dask_array_from_chunk_map(
    chunk_map: Dict[Tuple[int, ...], Tuple[bytes, Any]],
    grid_shape: Tuple[int, ...],
    shape: Tuple[int, ...],
    dtype: np.dtype,
    location: str,
    token: Optional[str],
    cache_bytes: int,
    schema_metadata: Optional[Dict[str, str]],
) -> da.Array:
    """Build the lazy chunk-fetching dask array from a chunk-index map.

    Shared by ``tensor_from_pb`` and ``_build_dask_array``. For a regular chunk
    grid (the common case) this emits a *single* ``Blockwise`` layer, so slicing
    one chunk culls to O(1) tasks and graph optimization is O(1) rather than
    O(n_chunks). Each block's ``chunk_id`` and bounds are delivered per block via
    a ``BlockwiseDepDict`` -- not broadcast as one whole-array literal -- so a
    partial read's graph *size* is O(blocks read), not O(blocks total): a
    single-plane slice of an N-chunk tensor carries one chunk id, not N. This
    makes serial single-plane reads (napari scrubbing a large T axis) and partial
    computes dramatically cheaper without changing per-chunk fetch behavior or
    leaf-task parallelism. Ragged/sparse grids fall back to the
    ``da.block``-of-``from_delayed`` construction.
    """
    if not chunk_map:
        raise ValueError("No chunks found")

    chunks = _regular_grid_chunks(chunk_map, grid_shape, shape)
    if chunks is not None:
        # Per-block dependency: block-index -> (chunk_id, bounds_start,
        # bounds_stop). A BlockwiseDepDict hands each output block *only* its own
        # entry, so culling a slice drops the rest and the graph stays small.
        dep_map = {
            chunk_idx: (
                chunk_id,
                tuple(int(s) for s in bounds.start),
                tuple(int(s) for s in bounds.stop),
            )
            for chunk_idx, (chunk_id, bounds) in chunk_map.items()
        }
        numblocks = tuple(len(c) for c in chunks)
        # Name the layer from the chunk ids alone, not the whole dep_map.
        # tokenize recurses every entry, and each block's (start, stop) is already
        # determined by `chunks` (the grid), so hashing the bounds is redundant
        # O(n_chunks) work -- ~1s at 80k chunks (biopb/biopb#346). Each chunk_id
        # encodes its array_id + bounds (+ any scale suffix), so the id tuple is an
        # injective fingerprint of the array: distinct arrays -> distinct ids ->
        # distinct name, and a false cache hit is impossible.
        chunk_ids = tuple(cid for cid, _start, _stop in dep_map.values())
        name = "biopb-tensor-chunk-" + tokenize(
            chunk_ids, location, token, cache_bytes, schema_metadata, dtype, chunks
        )
        dep = BlockwiseDepDict(mapping=dep_map, numblocks=numblocks)
        return _regular_blockwise_array(
            name, dep, chunks, dtype, location, token, cache_bytes, schema_metadata
        )

    # Fallback: ragged/sparse grid -> one delayed task per chunk.
    blocks = np.empty(grid_shape, dtype=object)
    for chunk_idx, (chunk_id, bounds) in chunk_map.items():
        chunk_shape = tuple(
            stop - start for start, stop in zip(bounds.start, bounds.stop, strict=True)
        )
        blocks[chunk_idx] = da.from_delayed(
            delayed(_fetch_chunk_distributed)(
                location,
                token,
                chunk_id,
                tuple(bounds.start),
                tuple(bounds.stop),
                cache_bytes,
                schema_metadata,
            ),
            shape=chunk_shape,
            dtype=dtype,
        )
    return da.block(blocks.tolist())


def _regular_blockwise_array(
    name: str,
    dep: BlockwiseDep,
    chunks: Tuple[Tuple[int, ...], ...],
    dtype: np.dtype,
    location: str,
    token: Optional[str],
    cache_bytes: int,
    schema_metadata: Optional[Dict[str, str]],
) -> da.Array:
    """Wrap a per-block ``BlockwiseDep`` in a single ``Blockwise`` (map_blocks) layer.

    Used by the regular-grid builder ``_build_dask_array_from_chunk_map`` with a
    materialized ``BlockwiseDepDict`` dep. Built directly rather than via
    ``da.map_blocks`` because the dep must be indexed like the output so each task
    gets only its own entry (biopb/biopb#278).
    """
    out_ind = tuple(range(len(chunks)))
    layer = _blockwise(
        _fetch_chunk_block,
        name,
        out_ind,
        dep,
        out_ind,
        location,
        None,
        token,
        None,
        cache_bytes,
        None,
        schema_metadata,
        None,
        numblocks={},
    )
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[])
    return da.Array(graph, name, chunks, dtype=dtype)
