"""What a start spends its time on, and how big a catalog row is, by source type.

Registering a source opens and parses its file; on a large site that is hours.
This records, per source type, how long each step took and how large what it
produced is, so the cost of a start (and of anything built on a row's contents)
is read from a log line rather than guessed. Counts are cheap; nothing here
reads a file.
"""

from __future__ import annotations

import logging
import threading
from array import array
from collections import defaultdict
from typing import Dict, List, NamedTuple, Sequence

logger = logging.getLogger(__name__)


class SyncCost(NamedTuple):
    """What writing one source's catalog row cost, returned by
    ``MetadataDatabase.sync_source_added``."""

    metadata_s: float  # get_metadata() and the tensor listing
    upsert_s: float  # the row write and the embedded-ROI import
    metadata_bytes: int  # metadata_json, as stored
    tensors_bytes: int  # the tensors column, as JSON
    descriptor_bytes: int  # the tensors' lean descriptors, serialized


_STEPS = ("create_s", "normalize_s", "metadata_s", "upsert_s")
_SIZES = ("metadata_bytes", "tensors_bytes", "descriptor_bytes", "members")


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


class RegistrationStats:
    """Per-source-type samples of registration steps and row sizes, plus the time
    each adapter's ``claim`` spent in the walk."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # 8 bytes a sample, not a boxed float: a site of 100k sources keeps nine
        # per source for the life of the process.
        self._samples: Dict[str, Dict[str, array]] = defaultdict(
            lambda: defaultdict(lambda: array("d"))
        )
        # adapter -> [probes, seconds, claimed]. Updated without the lock: the
        # walk is one thread, and a count off by a race would not matter.
        self._claims: Dict[str, List[float]] = defaultdict(lambda: [0, 0.0, 0])
        self._reported = 0

    def record_claim(self, adapter: str, seconds: float, claimed: bool) -> None:
        entry = self._claims[adapter]
        entry[0] += 1
        entry[1] += seconds
        entry[2] += 1 if claimed else 0

    def record_registration(
        self,
        source_type: str,
        *,
        create_s: float,
        normalize_s: float,
        members: int,
        cost: SyncCost | None,
    ) -> None:
        values = {"create_s": create_s, "normalize_s": normalize_s, "members": members}
        if cost is not None:
            values.update(cost._asdict())
        with self._lock:
            samples = self._samples[source_type]
            for key, value in values.items():
                samples[key].append(float(value))

    def summary_lines(self) -> List[str]:
        """One line per source type, then one per adapter that probed a path."""
        with self._lock:
            by_type = {
                t: {k: list(v) for k, v in s.items()} for t, s in self._samples.items()
            }
        lines: List[str] = []
        for source_type in sorted(by_type):
            samples = by_type[source_type]
            count = len(samples.get("create_s", ()))
            parts = [f"{source_type}: n={count}"]
            for step in _STEPS:
                values = samples.get(step)
                if values:
                    parts.append(
                        f"{step[:-2]} p50={_percentile(values, 0.5) * 1000:.0f}ms "
                        f"p95={_percentile(values, 0.95) * 1000:.0f}ms "
                        f"total={sum(values):.1f}s"
                    )
            for size in _SIZES:
                values = samples.get(size)
                if values:
                    parts.append(
                        f"{size} mean={sum(values) / len(values):.0f} "
                        f"max={max(values):.0f}"
                    )
            lines.append("  " + "; ".join(parts))
        claims = {a: list(v) for a, v in self._claims.items()}
        for adapter in sorted(claims, key=lambda a: -claims[a][1]):
            probes, seconds, claimed = claims[adapter]
            lines.append(
                f"  claim {adapter}: probes={int(probes)} claimed={int(claimed)} "
                f"total={seconds:.1f}s"
            )
        return lines

    def log_summary(self) -> None:
        """Log the summary once per new registration (a no-op when nothing was
        registered since the last call)."""
        with self._lock:
            total = sum(len(s.get("create_s", ())) for s in self._samples.values())
            if total == self._reported:
                return
            self._reported = total
        logger.info(
            "Registration cost by source type (%d sources):\n%s",
            total,
            "\n".join(self.summary_lines()),
        )
