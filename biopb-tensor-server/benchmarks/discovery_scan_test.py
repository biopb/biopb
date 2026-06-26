"""Full-scan (discovery walk) benchmark — biopb/biopb#55.

`discover_sources` walks a directory tree asking adapters to claim paths. Before
#55 the walk had no claim feedback: once a container store (e.g. an OME-Zarr
plate) was claimed at its root, the walk still recursed into it and probed every
interior **chunk file** for a claim that can never fire. That made full-scan cost
scale with the number of chunk files rather than with the number of logical
sources.

This benchmark builds a tree dominated by chunk files and times the full scan two
ways on the *same* tree:

- ``naive``  — the pre-#55 behavior: descend into claimed directories too
  (probe every interior path).
- ``pruned`` — the shipped behavior: stop descending below a directory-level
  claim (what ``discover_sources`` now does).

Both produce the identical set of claims; the gap between them is exactly the
work #55 removed. Each benchmark records the interior file count, the number of
adapter probes, and the number of sources found in ``extra_info`` so the
probe-count reduction is visible alongside the wall-clock number.

Run:
    pytest benchmarks/discovery_scan_test.py --benchmark-only
"""

import os
from pathlib import Path

import pytest
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.discovery import (
    ClaimContext,
    DiscoveryState,
    discover_sources,
    walk_with_identity_tracking,
)

from benchmarks.utils import generate_synthetic_hcs_plate, generate_synthetic_tiff

# Tree "scale": (wells, fields, chunks). Smaller chunks => more chunk files per
# field array, which is what amplifies the pre-#55 cost. Logical source count is
# fixed (one plate + a few sibling tiffs) across scales, so any growth in scan
# cost under `naive` comes purely from chunk-file fan-out.
SCALES = {
    "small": dict(wells=8, fields=2, shape=(512, 512), chunks=(64, 64)),
    "medium": dict(wells=24, fields=4, shape=(512, 512), chunks=(32, 32)),
    "large": dict(wells=48, fields=4, shape=(1024, 1024), chunks=(32, 32)),
}


class _CountingRegistry:
    """Wraps the default registry, counting every adapter-probe (claim) call.

    The probe count is a clean, deterministic proxy for the per-entry syscall
    storm #55 describes: each probe fans out to ~8 adapters, each doing its own
    is_dir()/is_file()/exists() stats.
    """

    def __init__(self):
        self._registry = get_default_registry()
        self.probes = 0

    def get_claims_for_path(self, ctx, state):
        self.probes += 1
        return self._registry.get_claims_for_path(ctx, state)

    def get_adapter_for_type(self, source_type):
        return self._registry.get_adapter_for_type(source_type)


def _scan(root: Path, registry, *, prune: bool) -> DiscoveryState:
    """Run a full discovery scan, toggling the #55 claim-feedback.

    With ``prune=True`` this mirrors the shipped ``discover_sources`` (don't
    descend below a directory-level claim). With ``prune=False`` it reproduces
    the pre-#55 behavior (descend into claimed stores and probe every interior
    file). Both share the identical root-claim short-circuit and per-path claim
    loop, so the only difference measured is the descent policy.
    """
    state = DiscoveryState()

    try:
        state.visited_identities.add(_identity(root))
    except OSError:
        return state

    # Root-claim short-circuit (identical to discover_sources).
    ctx = ClaimContext(root)
    claims = registry.get_claims_for_path(ctx, state)
    if claims:
        state.add_claim(claims[0])
        return state

    should_descend = (
        (lambda p: not state.is_path_claimed(str(p))) if prune else None
    )
    for path in walk_with_identity_tracking(
        root, state.visited_identities, should_descend=should_descend
    ):
        path_str = str(path)
        if state.is_path_claimed(path_str):
            continue
        claims = registry.get_claims_for_path(ClaimContext(path), state)
        if claims:
            state.add_claim(claims[0])
    return state


def _identity(path: Path) -> str:
    from biopb_tensor_server.discovery import get_file_identity

    return get_file_identity(path)


def _count_interior_files(root: Path) -> int:
    return sum(len(files) for _, _, files in os.walk(root))


@pytest.fixture(params=list(SCALES), ids=list(SCALES))
def scan_tree(request, tmp_path_factory):
    """Build a tree: one HCS OME-Zarr plate (many chunk files) + sibling tiffs."""
    spec = SCALES[request.param]
    root = tmp_path_factory.mktemp(f"scan_{request.param}")

    generate_synthetic_hcs_plate(
        str(root),
        wells=spec["wells"],
        fields=spec["fields"],
        shape=spec["shape"],
        chunks=spec["chunks"],
    )
    # A handful of unrelated leaf sources so the scan isn't a single root claim.
    for i in range(3):
        extra = root / f"extra_{i}"
        extra.mkdir()
        generate_synthetic_tiff(str(extra), shape=(256, 256))

    return root, _count_interior_files(root)


class TestFullScan:
    """Time the full discovery scan; compare pre-#55 vs shipped behavior."""

    def test_scan_pruned(self, benchmark, scan_tree):
        """Shipped behavior: scan cost is independent of chunk-file count."""
        root, n_files = scan_tree

        def run():
            reg = _CountingRegistry()
            state = _scan(root, reg, prune=True)
            return reg.probes, len(state.claims)

        probes, n_sources = benchmark(run)

        benchmark.extra_info.update(
            interior_files=n_files, probes=probes, sources=n_sources, mode="pruned"
        )
        # The claimed plate's chunk files are never probed: probe count is on the
        # order of the directory/leaf-source count, far below the file count.
        assert probes < n_files

    def test_scan_naive(self, benchmark, scan_tree):
        """Pre-#55 behavior: scan probes every interior chunk file (the regression)."""
        root, n_files = scan_tree

        def run():
            reg = _CountingRegistry()
            state = _scan(root, reg, prune=False)
            return reg.probes, len(state.claims)

        probes, n_sources = benchmark(run)

        benchmark.extra_info.update(
            interior_files=n_files, probes=probes, sources=n_sources, mode="naive"
        )

    def test_pruned_and_naive_find_same_sources(self, scan_tree):
        """Sanity: the optimization changes cost, not the discovered catalog."""
        root, _ = scan_tree

        pruned = _scan(root, get_default_registry(), prune=True)
        naive = _scan(root, get_default_registry(), prune=False)

        assert {c.primary_path for c in pruned.claims.values()} == {
            c.primary_path for c in naive.claims.values()
        }


class TestDiscoverSourcesScan:
    """Time the real ``discover_sources`` entry point (shipped path end-to-end)."""

    def test_discover_sources(self, benchmark, scan_tree):
        root, n_files = scan_tree
        registry = get_default_registry()

        state = benchmark(lambda: discover_sources(root, registry))

        benchmark.extra_info.update(
            interior_files=n_files, sources=len(state.claims)
        )
