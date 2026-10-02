"""Stat-derived facts about a path: its change signature and whether it is quiet.

Shared by the claim gate (``SourceManager._should_claim``), the removal shield and
the rebuild trigger (``Reconciler``), so "has this stopped changing" and "did it
change" are each answered in one place. Everything here reads a ``stat_result``;
nothing keeps state between calls.
"""

from __future__ import annotations

from typing import Any, Tuple


def build_entry_signature(
    stat_result: Any,
    is_directory: bool,
    cloud: bool = False,
) -> Tuple[Any, ...]:
    """Build a stable signature tuple for a file or directory.

    Under a ``cloud = true`` root the signature is **residency-invariant** --
    keyed on identity (``st_dev``, ``st_ino``) only. Hydrating a placeholder
    (a consented recall) bumps ``st_size``/``st_mtime_ns``/``st_ctime_ns``;
    including those would make the next rescan see the just-resolved source as
    "changed" and destructively remove+re-add it (and re-dehydration/eviction
    would flap it). Archived cloud data is stable and cloud mtime is
    untrustworthy anyway, so identity is the right key there. Non-cloud
    entries keep the full mtime/size-sensitive signature.

    Identity-only, so a zeroed cloud inode degrades it to a constant ``(0, 0)``
    that is still residency-invariant and per-path.
    """
    if cloud:
        return (stat_result.st_dev, stat_result.st_ino)
    if is_directory:
        return (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_mtime_ns,
            stat_result.st_ctime_ns,
        )
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def entry_change_time(stat_result: Any, now: float) -> float:
    """Return the best available filesystem change timestamp for an entry."""
    mtime_ns = getattr(stat_result, "st_mtime_ns", None)
    ctime_ns = getattr(stat_result, "st_ctime_ns", None)
    if mtime_ns is not None and ctime_ns is not None:
        return min(now, max(mtime_ns, ctime_ns) / 1_000_000_000)

    mtime = getattr(stat_result, "st_mtime", None)
    ctime = getattr(stat_result, "st_ctime", None)
    if mtime is not None and ctime is not None:
        return min(now, max(float(mtime), float(ctime)))

    return now


def entry_is_quiet(last_changed: float, now: float, stability_window: float) -> bool:
    """Has this path been unchanged long enough to be acted on?

    The single stability predicate, asked from two sides that must keep getting
    the same answer:

    * the claim gate (``SourceManager._should_claim``), to decide whether a path
      may be claimed -- claiming a half-written file registers a wrong
      descriptor, and for a format whose type marker is written last (OME-TIFF) a
      wrong ``source_type``, hence a different ``source_id``;
    * the removal shield (``Reconciler._claim_is_quiet``), to decide whether a
      claim missing from a walk is really gone or merely churning.

    ``last_changed`` is ``entry_change_time``'s value -- max(mtime, ctime),
    clamped to now.
    """
    return now - last_changed >= stability_window
