"""Per-entry syscall baseline for the monitored-tree walks — biopb/biopb#56.

#56 is an umbrella for *constant-factor* reductions in full-scan cost. Its premise
is that each filesystem entry costs ~20-30 metadata syscalls per forced rescan,
spread across the two walks the rescan runs back to back, and that most of those
syscalls are redundant:

- the **state walk** ``SourceManager._scan_tree_state`` — captures a stat-signature
  per entry for change/stability tracking;
- the **claim walk** — asks the adapters to claim each stable entry. Post-#56 item 4
  this no longer re-walks the filesystem: ``discover_sources_from_entries`` drives the
  claim protocol straight off the state walk's snapshot, so the second walk's per-entry
  ``stat``/``lstat``/``resolve``/``get_file_identity`` are gone and each adapter reads
  ``is_dir``/``is_file`` from the cached flag instead of re-stat'ing.

The issue is explicit that we **measure first** before optimizing. This module is
that measurement: it drives one forced rescan over a representative tree and counts
the syscall-bearing primitives, attributed per walk and per primitive. It started as
the pre-optimization baseline; with the state-walk rewrite (items 1+2: ``os.scandir``
+ one reused stat per entry) and the single-traversal claim phase (items 3+4) landed,
it now also serves as the **regression guard** locking those wins in — ``logical_resolve``
is ~1 (not ~1/entry), the state walk costs ~1 stat/entry (not ~14-15), and the claim
column's ``stat``/``scandir``/``logical_resolve`` collapse to ~0 (only genuine adapter
content probes remain). **It changes no walk code.**

Two complementary mechanisms:

1. A Python-boundary counter (primary; deterministic, cross-platform) that tallies
   the os-level primitives (``os.stat``/``lstat``/``scandir``/``readlink``/``open``)
   the walks issue, plus the *logical* high-level calls (``Path.resolve``,
   ``get_file_identity``) whose redundancy was the point. ``Path.resolve()`` fans out
   into several ``lstat``/``readlink`` syscalls (O(depth)), so logical-call count and
   syscall count are reported side by side. ``DirEntry.stat()`` calls C ``fstatat``
   directly and so escapes an ``os.stat`` patch — the post-#56 per-entry stat — so it
   is counted via a scandir proxy (``direntry_stat``) to keep the total honest.
2. A ``strace -f -c`` ground-truth pass (secondary; Linux + strace only, else
   skipped) that validates the Python counter against real kernel counts.

Each bucket maps to the #56 inventory item it relates to:

    logical Path.resolve() (was >1x/entry, now ~1)  -> item 1 (drop redundant resolve)
    direntry_stat ~1x/entry (was stat+lstat fan-out) -> items 1-2 (reuse stat; scandir)
    claim-walk stat/is_dir/is_file per probe (was    -> items 3-4 (cache is_dir; drive
        ~16x/entry, now ~0)                                claims from the snapshot)
    builtin open() per stable file                   -> item 5 (scope the append probe)

Run:
    pytest benchmarks/scan_syscalls_test.py -s -v
    pytest benchmarks/scan_syscalls_test.py -s -v --benchmark-only   # also time it
"""

import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.discovery import (
    DiscoveryState,
    discover_sources_from_entries,
)
from biopb_tensor_server.source_manager import SourceManager

from benchmarks.utils import generate_synthetic_hcs_plate, generate_synthetic_tiff


# Same tree "scale" knobs as discovery_scan_test.py so the two benchmarks describe
# the *same* tree from different angles (probe count there, syscall count here).
SCALES = {
    "small": dict(wells=8, fields=2, shape=(512, 512), chunks=(64, 64)),
    "medium": dict(wells=24, fields=4, shape=(512, 512), chunks=(32, 32)),
    "large": dict(wells=48, fields=4, shape=(1024, 1024), chunks=(32, 32)),
}


def _count_interior_files(root: Path) -> int:
    return sum(len(files) for _, _, files in os.walk(root))


# ---------------------------------------------------------------------------
# Python-boundary syscall counter
# ---------------------------------------------------------------------------


class _CountingDirEntry:
    """Wrap an ``os.DirEntry`` to count the ``stat()`` it issues.

    ``DirEntry.stat()`` calls C ``fstatat`` directly, *bypassing* the ``os.stat``
    Python callable — so patching ``os.stat`` cannot see it. After #56 the state
    walk's one-stat-per-entry IS a ``DirEntry.stat()``, so without this proxy the
    counter would silently under-report by ~1 syscall/entry. Everything else
    delegates to the real entry untouched."""

    __slots__ = ("_e", "_c")

    def __init__(self, entry, counts):
        self._e = entry
        self._c = counts

    def stat(self, *args, **kwargs):
        self._c["direntry_stat"] += 1
        return self._e.stat(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._e, name)


class _CountingScandir:
    """Context-manager/iterator wrapper around ``os.scandir`` that yields
    :class:`_CountingDirEntry` proxies and is otherwise transparent."""

    def __init__(self, real, counts):
        self._real = real
        self._c = counts

    def __iter__(self):
        for entry in self._real:
            yield _CountingDirEntry(entry, self._c)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _SyscallCounter:
    """Count the syscall-bearing primitives the walks issue, by patching them.

    Wrappers only count and delegate — they never change behavior — so running a
    walk inside the counter is observably identical, just instrumented. Patches
    the os module (true syscalls, including ``DirEntry.stat`` via a scandir proxy)
    and a couple of high-level entry points (``Path.resolve``, ``get_file_identity``)
    whose *logical* call count is the redundancy signal #56 item 1 was about — now
    a regression signal: after #56 ``logical_resolve`` is ~1, not ~1/entry.
    """

    # os-level primitives = true syscalls (a Path method that issues one shows up
    # here too, since the patched os function is what it ultimately calls).
    # ``scandir`` and ``direntry_stat`` are tracked separately (scandir needs a
    # proxy to also catch DirEntry.stat), but all count toward the syscall total.
    _OS_PRIMS = ("stat", "lstat", "readlink", "open")
    _TRUE_SYSCALL_BUCKETS = _OS_PRIMS + ("scandir", "direntry_stat")

    def __init__(self):
        self.counts = {name: 0 for name in self._TRUE_SYSCALL_BUCKETS}
        # Logical high-level calls (each fans out into several os-level syscalls).
        self.counts["logical_resolve"] = 0
        self.counts["logical_get_file_identity"] = 0
        self.counts["builtin_open"] = 0
        self._saved = {}

    @contextmanager
    def measure(self):
        import builtins
        import pathlib
        import biopb_tensor_server.discovery as discovery
        import biopb_tensor_server.source_manager as source_manager

        c = self.counts

        # --- os-level (true syscalls) ---
        for name in self._OS_PRIMS:
            orig = getattr(os, name)
            self._saved[("os", name)] = orig

            def make(orig=orig, name=name):
                def wrapper(*args, **kwargs):
                    c[name] += 1
                    return orig(*args, **kwargs)

                return wrapper

            setattr(os, name, make())

        # --- os.scandir: count the call AND wrap so DirEntry.stat is counted ---
        orig_scandir = os.scandir
        self._saved[("os", "scandir")] = orig_scandir

        def scandir_wrapper(*args, **kwargs):
            c["scandir"] += 1
            return _CountingScandir(orig_scandir(*args, **kwargs), c)

        os.scandir = scandir_wrapper

        # --- builtin open (the append probe, _can_open_for_append) ---
        orig_open = builtins.open
        self._saved[("builtins", "open")] = orig_open

        def open_wrapper(*args, **kwargs):
            c["builtin_open"] += 1
            return orig_open(*args, **kwargs)

        builtins.open = open_wrapper

        # --- logical Path.resolve (fans out into lstat/readlink under the hood) ---
        orig_resolve = pathlib.Path.resolve
        self._saved[("Path", "resolve")] = orig_resolve

        def resolve_wrapper(self, *args, **kwargs):
            c["logical_resolve"] += 1
            return orig_resolve(self, *args, **kwargs)

        pathlib.Path.resolve = resolve_wrapper

        # --- logical get_file_identity (now reuses the caller's stat; #56) ---
        # Both walks call it, but each module that does `from ... import
        # get_file_identity` holds its own binding, so patch every such module —
        # patching only discovery would miss the state walk's calls (source_manager).
        # Accept the optional stat_result arg #56 added.
        orig_identity = discovery.get_file_identity

        def identity_wrapper(path, *args, **kwargs):
            c["logical_get_file_identity"] += 1
            return orig_identity(path, *args, **kwargs)

        for mod in (discovery, source_manager):
            self._saved[(mod.__name__, "get_file_identity")] = mod.get_file_identity
            mod.get_file_identity = identity_wrapper

        try:
            yield self
        finally:
            for name in self._OS_PRIMS:
                setattr(os, name, self._saved[("os", name)])
            os.scandir = self._saved[("os", "scandir")]
            builtins.open = self._saved[("builtins", "open")]
            pathlib.Path.resolve = self._saved[("Path", "resolve")]
            for mod in (discovery, source_manager):
                mod.get_file_identity = self._saved[(mod.__name__, "get_file_identity")]

    @property
    def total_syscalls(self) -> int:
        """Sum of the true-syscall buckets (os-level + DirEntry.stat)."""
        return sum(self.counts[name] for name in self._TRUE_SYSCALL_BUCKETS)


def _make_manager(root: Path) -> SourceManager:
    """A SourceManager wired to scan ``root`` and probe everything immediately.

    ``stability_window=0`` + ``stable_rescans_required=0`` make every entry pass the
    discovery gate on the first pass (so the claim walk actually probes the whole
    tree); ``probe_open_files=True`` exercises the per-file append probe (#56 item 5);
    ``full_rescan_interval=0`` forces the full sweep. ``server=None`` is safe: the two
    measured methods (``_refresh_entry_state`` and the snapshot-driven claim phase)
    never touch it — only reconcile/registration would, and we don't run that here.
    """
    return SourceManager(
        server=None,
        registry=get_default_registry(),
        discovery_state=DiscoveryState(),
        watcher=None,
        monitored_dirs={root},
        stability_window=0.0,
        probe_open_files=True,
        full_rescan_interval=0.0,
        stable_rescans_required=0,
    )


def _run_state_walk(manager: SourceManager) -> None:
    """Phase 1: the stat-signature sweep. publish=True so the claim walk's
    ``_should_scan_path`` sees the freshly captured entry state."""
    manager._refresh_entry_state(force_full=True, publish=True)


def _run_claim_walk(manager: SourceManager, root: Path):
    """Phase 2: adapter discovery driven from the state walk's published snapshot —
    exactly how ``_handle_rescan`` drives it post-#56 item 4 (no second filesystem
    walk). ``_run_state_walk`` published ``_entry_state``/``_skipped_stable_dirs``, so
    this consumes them just like the runtime rescan does."""
    return discover_sources_from_entries(
        (
            (path_str, entry[0], entry[1])
            for path_str, entry in manager._entry_state.items()
        ),
        manager._registry,
        path_filter=manager._should_scan_resolved,
        skipped_dirs=manager._skipped_stable_dirs,
    )


def run_both_walks(root) -> int:
    """Drive one forced rescan (state walk then claim walk) over ``root``.

    Standalone (no counter, no pytest) so the strace subprocess can call it. Returns
    the number of sources discovered.
    """
    root = Path(root)
    manager = _make_manager(root)
    _run_state_walk(manager)
    state = _run_claim_walk(manager, root)
    return len(state.claims)


# ---------------------------------------------------------------------------
# strace ground-truth (Linux + strace only)
# ---------------------------------------------------------------------------

_STRACE_SYSCALLS = "stat,lstat,newfstatat,statx,openat,open,readlink,getdents64"


def _strace_total_syscalls(root: Path) -> int:
    """Total traced syscalls for run_both_walks(root), minus an import-only baseline.

    Subtracting a baseline run that imports the harness but does nothing strips the
    interpreter-startup/import syscalls, leaving (approximately) the walk's own.
    """
    repo = Path(__file__).resolve().parent.parent  # biopb-tensor-server/
    walk_script = (
        "from benchmarks.scan_syscalls_test import run_both_walks;"
        f"run_both_walks({str(root)!r})"
    )
    baseline_script = "import benchmarks.scan_syscalls_test"  # import only

    def traced(script: str) -> int:
        proc = subprocess.run(
            ["strace", "-f", "-c", "-e", f"trace={_STRACE_SYSCALLS}",
             sys.executable, "-c", script],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
        # strace -c writes its summary to stderr; the last column of the totals
        # row (or the sum of per-syscall "calls" columns) is what we want.
        total = 0
        for line in proc.stderr.splitlines():
            m = re.match(r"\s*\d+\.\d+\s+\S+\s+\S+\s+(\d+)\s+\d*\s+(\w+)\s*$", line)
            if m and m.group(2) in _STRACE_SYSCALLS.split(","):
                total += int(m.group(1))
        return total

    return max(0, traced(walk_script) - traced(baseline_script))


# ---------------------------------------------------------------------------
# Fixtures + reporting
# ---------------------------------------------------------------------------


# Module-scoped so each tree (the large one is ~200k files) is built once and
# shared by both the counter test and the strace test, not rebuilt per test.
@pytest.fixture(scope="module", params=list(SCALES), ids=list(SCALES))
def scan_tree(request, tmp_path_factory):
    """One HCS OME-Zarr plate (chunk-file heavy) + sibling tiffs — same tree shape
    as discovery_scan_test.py. Returns (scale_id, root, interior_file_count)."""
    scale = request.param
    spec = SCALES[scale]
    root = tmp_path_factory.mktemp(f"syscalls_{scale}")

    generate_synthetic_hcs_plate(
        str(root),
        wells=spec["wells"],
        fields=spec["fields"],
        shape=spec["shape"],
        chunks=spec["chunks"],
    )
    for i in range(3):
        extra = root / f"extra_{i}"
        extra.mkdir()
        generate_synthetic_tiff(str(extra), shape=(256, 256))

    return scale, root, _count_interior_files(root)


def _report(scale: str, n_files: int, state_c: _SyscallCounter, claim_c: _SyscallCounter):
    total = state_c.total_syscalls + claim_c.total_syscalls
    per_entry = total / n_files if n_files else 0.0
    print(f"\n=== #56 syscall baseline: {scale} ===")
    print(f"interior files (entries): {n_files}")
    print(f"{'primitive':<28}{'state walk':>12}{'claim walk':>12}{'total':>10}")
    for k in state_c.counts:
        sv, cv = state_c.counts[k], claim_c.counts[k]
        print(f"{k:<28}{sv:>12}{cv:>12}{sv + cv:>10}")
    print(f"{'TRUE SYSCALLS (os-level)':<28}"
          f"{state_c.total_syscalls:>12}{claim_c.total_syscalls:>12}{total:>10}")
    print(f"--> syscalls per entry: {per_entry:.1f}")


def _write_baseline(record: dict) -> Path:
    """Persist one scale's numbers under .benchmarks/ so the baseline survives the
    test run and can be pasted into the #56 PR / issue comment."""
    import json

    out_dir = Path(__file__).resolve().parent.parent / ".benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scan_syscalls_{record['scale']}.json"
    out_path.write_text(json.dumps(record, indent=2))
    return out_path


class TestSyscallBaseline:
    """Confirm and record the per-entry syscall cost #56 sets out to reduce."""

    def test_per_entry_syscalls(self, scan_tree):
        scale, root, n_files = scan_tree

        # Measure each walk in its own counter for per-walk attribution. The state
        # walk must run first (and publish) so the claim walk's stability gate sees
        # the entry state, exactly as _handle_rescan sequences them.
        state_c = _SyscallCounter()
        manager = _make_manager(root)
        with state_c.measure():
            _run_state_walk(manager)

        claim_c = _SyscallCounter()
        with claim_c.measure():
            n_sources = len(_run_claim_walk(manager, root).claims)

        _report(scale, n_files, state_c, claim_c)

        total = state_c.total_syscalls + claim_c.total_syscalls
        per_entry = total / n_files if n_files else 0.0

        # Persist the baseline; this — not an assertion — is the deliverable.
        out_path = _write_baseline(
            dict(
                scale=scale,
                interior_files=n_files,
                sources=n_sources,
                state_walk_syscalls=state_c.total_syscalls,
                claim_walk_syscalls=claim_c.total_syscalls,
                total_syscalls=total,
                syscalls_per_entry=round(per_entry, 2),
                state_walk_breakdown=dict(state_c.counts),
                claim_walk_breakdown=dict(claim_c.counts),
            )
        )
        print(f"--> wrote {out_path}")

        # Records the numbers; the asserts below are regression guards locking in
        # the #56 state-walk optimization (items 1+2), not the original baseline.
        assert n_files > 0
        assert total > 0
        # Post-#56 the state walk no longer resolve()s every entry — only the rare
        # symlink and the root — so resolve() count is far below the entry count.
        # A regression that reintroduces the per-entry resolve fan-out trips this.
        logical_resolves = (
            state_c.counts["logical_resolve"] + claim_c.counts["logical_resolve"]
        )
        assert logical_resolves < n_files
        # And the walks now cost ~1 stat/entry (DirEntry.stat), not the ~14-15 the
        # baseline measured. Guard a generous ceiling so a future regression is loud.
        assert per_entry < 5

    @pytest.mark.skipif(
        sys.platform != "linux" or shutil.which("strace") is None,
        reason="strace ground-truth requires Linux with strace installed",
    )
    def test_strace_ground_truth_agrees(self, scan_tree):
        """The Python-boundary counter should not wildly diverge from real kernel
        counts. strace sees a superset (getdents, statx variants, syscalls below the
        Python boundary), so we only require the same order of magnitude.

        Run on the ``small`` tree only: strace re-runs the whole walk in two
        subprocesses, which is prohibitively slow on the 200k-file ``large`` tree and
        adds nothing — the counter↔kernel agreement is scale-independent."""
        scale, root, n_files = scan_tree
        if scale != "small":
            pytest.skip("strace validation runs on the small tree only (speed)")

        py_c = _SyscallCounter()
        with py_c.measure():
            run_both_walks(root)
        py_total = py_c.total_syscalls

        strace_total = _strace_total_syscalls(root)
        print(f"\n[strace] {scale}: python-counter={py_total} "
              f"strace={strace_total} (entries={n_files})")

        assert strace_total > 0
        # Same order of magnitude: strace >= python (ground truth sees a superset)
        # and not more than ~5x larger (sanity against a parsing or baseline bug).
        assert py_total * 0.5 <= strace_total <= py_total * 5
