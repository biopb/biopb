"""What a cache miss costs: declared for a scaled chunk, measured for a raw one.

The class is written into a cache segment, so it must not depend on whoever
stored the chunk first: the precache and a client can ask for the same coarse
chunk_id, and either order must give the same answer.

A **scaled** chunk is declared: cheap only when its scale is off the advertised
ladder, i.e. when nobody is expected to return to it. That answer is a pure
function of the chunk_id, which is what lets the class be stored.

A **full-resolution** chunk has no ladder to be judged against, so a declared
rule can only ever call it normal. What separates one worth keeping from one
worth dropping is what it cost to decode, so that is measured -- see
:class:`DecodeRates`. The measurement is per ``array_id`` rather than per chunk,
which keeps the stored class a property of the array, and it is off by default:
until an operator has read the table and set a threshold, it classifies nothing.

A stamp is never revised. A segment is opened per ``(class, size)`` pool and
fills with chunks from whatever arrays are being read, so a segment has no one
array to re-derive a class from at recovery -- the class can only be applied
where it routes a chunk to a pool, which is at write time. The consequence is
that an array's warmup chunks keep the class they were stored with, and so do
chunks stored while an array was faster than it is now. Both err in the
direction of keeping bytes the cache could have dropped, which is why this is
documented rather than defended against.

The ladder lives here rather than on the adapter so the pyramid config need not
be threaded down the read path: adapters are built without the server's config,
and a per-chunk call cannot carry one through every override of
``resolve_chunk_data``. The server installs it once instead.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import (
    TYPE_CHECKING,
    Dict,
    FrozenSet,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from biopb_tensor_server.core.chunk import compute_pyramid_scale_hints
from biopb_tensor_server.core.config import PyramidConfig

if TYPE_CHECKING:
    from biopb_tensor_server.cache import RetentionClass

logger = logging.getLogger(__name__)

# The server's ladder knobs, set once at construction (TensorFlightServer). A
# default instance keeps a direct caller -- a test, a script driving an adapter
# with no server -- on the shipped ladder rather than on none.
_active_pyramid_config = PyramidConfig()


def set_active_pyramid_config(config: Optional[PyramidConfig]) -> None:
    """Install the ladder every retention decision is judged against.

    One process serves one config, so this is process state rather than an
    argument on the read path. A second server in the same process (tests) wins
    it, and an adapter that already memoized its ladder keeps the old one --
    neither matters while the config is read once at server start.
    """
    global _active_pyramid_config
    _active_pyramid_config = config or PyramidConfig()


def computed_ladder(
    shape: Sequence[int], dim_labels: Optional[Sequence[str]]
) -> FrozenSet[Tuple[int, ...]]:
    """The computed pyramid's scales for one tensor, under the active config.

    ``PyramidConfig`` is the single source of truth for the levels the server
    advertises *and* the ones the precache warms, so a decision made from this
    set cannot drift from either.
    """
    return frozenset(
        tuple(int(s) for s in level)
        for level in compute_pyramid_scale_hints(
            list(shape),
            list(dim_labels) if dim_labels else None,
            **_active_pyramid_config.level_kwargs(),
        )
    )


def retention_for_scale(
    scale: Tuple[int, ...], ladder: FrozenSet[Tuple[int, ...]]
) -> RetentionClass:
    """Classify one scaled chunk against its tensor's ``computed_ladder``.

    A rung of the ladder is what a client returns to on every open, and what the
    precache warms. Any other scale is a one-off, asked for in passing and
    regenerable from the full-resolution chunks under it.
    """
    return "normal" if scale in ladder else "cheap"


# --- measured cost: per-array decode throughput -----------------------------
#
# The declared rule above answers a scaled chunk. A full-resolution chunk has no
# ladder to be judged against -- every one of them is "normal" by declaration --
# so the only thing that can separate a full-res chunk worth keeping from one
# worth dropping is what it actually cost to decode. That is measured here, and
# only here: the sample is taken on the unscaled read, which is the one read
# that cannot be served out of the cache (a scaled read can, since #965, so
# timing one would measure the cache's own state -- see resolve_chunk_data).


# Throughput is an EMA rather than a mean so a source whose cost changes (a
# handle warms, a network mount degrades) is eventually described by its recent
# reads rather than by every read since boot. Ten-ish samples to converge.
_EMA_ALPHA = 0.25

# How stale a persisted row may be. The table is written through on a debounce
# rather than per sample: a read is a few milliseconds and a rewrite of the
# whole table is comparable, so per-sample persistence would cost more than the
# thing it measures. A minute bounds both the staleness a live query sees and
# what a hard kill forfeits.
_FLUSH_SECONDS = 60.0


class DecodeRateStore(Protocol):
    """Where the table is kept between runs -- the catalog DuckDB database.

    Named as a protocol rather than imported so the read path does not depend on
    the catalog: an embedded server built without one simply never attaches, and
    measures for its own lifetime.
    """

    def load_decode_rates(self) -> Dict[str, Tuple[float, int]]: ...

    def save_decode_rates(self, rows: Mapping[str, Tuple[float, int]]) -> None: ...


class DecodeRates:
    """Per-``array_id`` decode throughput in MB/s, and the class it implies.

    One row per tensor, keyed by the chunk_id's ``array_id`` -- which is the
    native pyramid level's own id where there is one, so a compressed low level
    and a raw level 0 are measured separately rather than averaged into a number
    describing neither.

    The rate is what a rebuild costs per byte, not what the format can sustain:
    a read is timed as the read path issues it, so an array with small chunks
    reports lower MB/s than the same format would over large ones, because its
    per-call cost is amortized over fewer bytes. That is the right number for
    deciding what to keep, and the reason the threshold has to be read off the
    spread of a particular server's table rather than reasoned about.

    ``cheap_mbps`` of 0 (the default) measures everything and classifies
    nothing: the table is a diagnostic until an operator has looked at it and
    picked a threshold. There is no defensible shipped default -- "fast enough
    that rebuilding beats keeping" is a property of the machine's disk and the
    formats on it, which is what the table is for.

    Once attached to a store the table both survives a restart and *is* the
    client surface: it is a `decode_rates` table in the catalog database, read
    with ``client.query_sources``. It is deliberately not kept beside the cache
    segments -- the cache directory is the operator's to delete, and clearing it
    should cost the bytes, not a run's worth of measurement.
    """

    __slots__ = ("_lock", "_rates", "_counts", "_store", "_next_flush", "cheap_mbps")

    def __init__(self, cheap_mbps: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._rates: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._store: Optional[DecodeRateStore] = None
        self._next_flush = 0.0
        self.cheap_mbps = float(cheap_mbps)

    def attach(self, store: DecodeRateStore) -> None:
        """Persist through ``store``, starting from the rows it already holds.

        The stored rows lose to anything measured since this object was built,
        so attaching late cannot undo a live measurement. Best-effort: a table
        that will not load leaves this run measuring from scratch, which is one
        warmup, not a failure worth refusing to serve over.
        """
        try:
            stored = store.load_decode_rates()
        except Exception:  # noqa: BLE001 - diagnostic data, never load-bearing
            logger.warning("Could not load persisted decode rates", exc_info=True)
            stored = {}
        with self._lock:
            for array_id, (mbps, samples) in stored.items():
                if mbps > 0.0 and samples > 0:
                    self._rates.setdefault(array_id, mbps)
                    self._counts.setdefault(array_id, samples)
            self._store = store
            self._next_flush = time.monotonic() + _FLUSH_SECONDS

    def record(self, array_id: str, nbytes: int, seconds: float) -> None:
        """Fold one full-resolution read into ``array_id``'s rate.

        Every read counts, however small. A small chunk's MB/s is dominated by
        per-call fixed cost -- opening a handle, re-mmapping (biopb/biopb#816),
        a seek -- rather than by the format's decode speed, but that cost is
        part of what rebuilding the chunk would take, which is the question
        being asked. Discarding those reads would have left an array whose
        chunks are all small permanently unmeasured, and so permanently
        "normal", when some of them are the cheapest things in the cache.

        Only a duration the clock could not resolve is dropped, because it
        cannot be divided by.
        """
        if seconds <= 0.0:
            return
        mbps = nbytes / 1e6 / seconds
        with self._lock:
            previous = self._rates.get(array_id)
            self._rates[array_id] = (
                mbps if previous is None else previous + _EMA_ALPHA * (mbps - previous)
            )
            self._counts[array_id] = self._counts.get(array_id, 0) + 1
            due = self._store is not None and time.monotonic() >= self._next_flush
        if due:
            self.flush()

    def rate(self, array_id: str) -> Optional[float]:
        """``array_id``'s MB/s, or None if it has never been read at full resolution."""
        with self._lock:
            return self._rates.get(array_id)

    def retention_for_array(self, array_id: str) -> RetentionClass:
        """Classify a full-resolution chunk of ``array_id`` by measured cost.

        "cheap" only ever *widens* what eviction may take: an array with no
        rate yet, or a rate under the threshold, keeps the declared answer,
        which is what every full-resolution chunk got before this existed.
        """
        if self.cheap_mbps <= 0.0:
            return "normal"
        measured = self.rate(array_id)
        if measured is None:
            return "normal"
        return "cheap" if measured >= self.cheap_mbps else "normal"

    def snapshot(self) -> Dict[str, Tuple[float, int]]:
        """The whole table, as ``array_id -> (mbps, samples)``."""
        with self._lock:
            return {
                array_id: (rate, self._counts.get(array_id, 0))
                for array_id, rate in sorted(self._rates.items())
            }

    def flush(self) -> None:
        """Write the table through to the attached store. A no-op without one.

        Swallows a store failure, here rather than in the store: this is called
        from the read path, and losing a write costs one warmup's worth of
        measurement -- not worth failing a read or a clean shutdown over.
        """
        with self._lock:
            store = self._store
            self._next_flush = time.monotonic() + _FLUSH_SECONDS
        if store is None:
            return
        try:
            store.save_decode_rates(self.snapshot())
        except Exception:  # noqa: BLE001 - diagnostic data, never load-bearing
            logger.warning("Could not persist decode rates", exc_info=True)


# Process state, like the ladder above and for the same reason: the read path
# cannot carry it, and one process serves one cache.
_active_decode_rates = DecodeRates()


def set_active_decode_rates(rates: Optional[DecodeRates]) -> None:
    """Install the table every full-resolution chunk is classified against."""
    global _active_decode_rates
    _active_decode_rates = rates or DecodeRates()


def active_decode_rates() -> DecodeRates:
    """The installed table (never None -- an un-configured one measures only)."""
    return _active_decode_rates


def record_decode(array_id: str, nbytes: int, seconds: float) -> None:
    """Fold one full-resolution read into the active table."""
    _active_decode_rates.record(array_id, nbytes, seconds)


def retention_for_array(array_id: str) -> RetentionClass:
    """The active table's class for a full-resolution chunk of ``array_id``."""
    return _active_decode_rates.retention_for_array(array_id)
