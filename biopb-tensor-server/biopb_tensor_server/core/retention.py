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

import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, FrozenSet, Optional, Sequence, Tuple

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

# Below this a read is dominated by per-call fixed cost -- opening a handle,
# re-mmapping (biopb/biopb#816), a seek -- not by decode throughput, and its
# MB/s describes the call overhead instead of the format. Such reads are dropped
# rather than damped: an array whose chunks are all smaller than this simply
# never accumulates a rate, and stays "normal".
_MIN_SAMPLE_BYTES = 1 << 20

# An EMA is meaningless on its first sample and noisy for a few after it. Until
# an array has this many, it has no rate at all and classifies "normal" -- the
# conservative direction, and the reason an unmeasured array needs no special
# case.
_MIN_SAMPLES = 4


class DecodeRates:
    """Per-``array_id`` decode throughput in MB/s, and the class it implies.

    One row per tensor, keyed by the chunk_id's ``array_id`` -- which is the
    native pyramid level's own id where there is one, so a compressed low level
    and a raw level 0 are measured separately rather than averaged into a number
    describing neither.

    ``cheap_mbps`` of 0 (the default) measures everything and classifies
    nothing: the table is a diagnostic until an operator has looked at it and
    picked a threshold. There is no defensible shipped default -- "fast enough
    that rebuilding beats keeping" is a property of the machine's disk and the
    formats on it, which is what the table is for.
    """

    __slots__ = ("_lock", "_rates", "_counts", "cheap_mbps")

    def __init__(self, cheap_mbps: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._rates: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self.cheap_mbps = float(cheap_mbps)

    def record(self, array_id: str, nbytes: int, seconds: float) -> None:
        """Fold one full-resolution read into ``array_id``'s rate.

        Silently ignores a read too small to measure or too fast to have a
        meaningful duration -- both describe the clock, not the format.
        """
        if nbytes < _MIN_SAMPLE_BYTES or seconds <= 0.0:
            return
        mbps = nbytes / 1e6 / seconds
        with self._lock:
            previous = self._rates.get(array_id)
            self._rates[array_id] = (
                mbps if previous is None else previous + _EMA_ALPHA * (mbps - previous)
            )
            self._counts[array_id] = self._counts.get(array_id, 0) + 1

    def rate(self, array_id: str) -> Optional[float]:
        """``array_id``'s MB/s, or None while it has too few samples to have one."""
        with self._lock:
            if self._counts.get(array_id, 0) < _MIN_SAMPLES:
                return None
            return self._rates.get(array_id)

    def retention_for_array(self, array_id: str) -> RetentionClass:
        """Classify a full-resolution chunk of ``array_id`` by measured cost.

        "cheap" only ever *widens* what eviction may take: an array with no
        rate, too few samples, or a rate under the threshold keeps the declared
        answer, which is what every full-resolution chunk got before this
        existed.
        """
        if self.cheap_mbps <= 0.0:
            return "normal"
        measured = self.rate(array_id)
        if measured is None:
            return "normal"
        return "cheap" if measured >= self.cheap_mbps else "normal"

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """The whole table, for the ``decode_rates`` action and the CLI."""
        with self._lock:
            return {
                array_id: {"mbps": rate, "samples": self._counts.get(array_id, 0)}
                for array_id, rate in sorted(self._rates.items())
            }

    def load(self, path: Path) -> None:
        """Merge a persisted table in, so a restart does not re-serve its warmup.

        Best-effort in both directions: an absent, truncated or hand-edited file
        leaves the table empty, which is exactly the state a first run is in.
        """
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        with self._lock:
            for array_id, row in raw.items():
                try:
                    rate = float(row["mbps"])
                    count = int(row["samples"])
                except (TypeError, ValueError, KeyError):
                    continue
                if rate > 0.0 and count > 0:
                    self._rates[array_id] = rate
                    self._counts[array_id] = count

    def save(self, path: Path) -> None:
        """Persist the table beside the segments it describes.

        Written atomically, and swallowed on failure: losing it costs one
        warmup, and this runs on the shutdown path.
        """
        snapshot = self.snapshot()
        if not snapshot:
            return
        tmp = path.parent / (path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(snapshot))
            os.replace(tmp, path)
        except OSError:
            logger.debug("could not persist decode rates to %s", path, exc_info=True)
            tmp.unlink(missing_ok=True)


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
