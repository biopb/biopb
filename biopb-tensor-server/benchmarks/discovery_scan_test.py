"""One-shot discovery scan benchmark -- biopb/biopb#55, #344.

`discover_under` is the walk behind a drag-drop and a ``monitor = false``
directory: one :class:`TreeScanner` snapshot of the root, then claims off it. Its
cost is the number of *entries* stat-ed, so a tree dominated by the chunk files of
an OME-Zarr plate is the worst case -- the claim phase prunes the plate's
interior (#55), but the snapshot has already paid a stat for every chunk by then.

Two shapes, on the same plate:

- ``tree``  -- a folder holding the plate and a few sibling tiffs: the whole
  interior is stat-ed.
- ``root``  -- the plate itself dropped: the root is claimed before any walk, so
  the cost does not depend on the chunk count at all.

Each records the interior file count, the adapter probes and the sources found in
``extra_info`` so the probe-count reduction is visible beside the wall clock.

Run:
    pytest benchmarks/discovery_scan_test.py --benchmark-only
"""

import os
from pathlib import Path

import pytest
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.sources.scan_root import discover_under

from benchmarks.utils import generate_synthetic_hcs_plate, generate_synthetic_tiff

# Tree "scale": (wells, fields, chunks). Smaller chunks => more chunk files per
# field array. Logical source count is fixed (one plate + a few sibling tiffs)
# across scales, so any growth in scan cost comes purely from chunk-file fan-out.
SCALES = {
    "small": {"wells": 8, "fields": 2, "shape": (512, 512), "chunks": (64, 64)},
    "medium": {"wells": 24, "fields": 4, "shape": (512, 512), "chunks": (32, 32)},
    "large": {"wells": 48, "fields": 4, "shape": (1024, 1024), "chunks": (32, 32)},
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


def _count_interior_files(root: Path) -> int:
    return sum(len(files) for _, _, files in os.walk(root))


@pytest.fixture(params=list(SCALES), ids=list(SCALES))
def scan_tree(request, tmp_path_factory):
    """Build a folder: one HCS OME-Zarr plate (many chunk files) + sibling tiffs.

    Returns ``(folder, plate, interior_file_count)``.
    """
    spec = SCALES[request.param]
    root = tmp_path_factory.mktemp(f"scan_{request.param}")

    plate, _, _ = generate_synthetic_hcs_plate(
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

    return root, Path(plate), _count_interior_files(root)


class TestOneShotScan:
    def test_scan_a_folder(self, benchmark, scan_tree):
        """The whole tree is stat-ed; only the claim phase prunes the interior."""
        root, _, n_files = scan_tree

        def run():
            reg = _CountingRegistry()
            state = discover_under(root, reg)
            return reg.probes, len(state.claims)

        probes, n_sources = benchmark(run)

        benchmark.extra_info.update(
            interior_files=n_files, probes=probes, sources=n_sources, shape="tree"
        )
        # The claimed plate's chunk files are never probed: probe count is on the
        # order of the directory/leaf-source count, far below the file count.
        assert probes < n_files

    def test_scan_a_dataset_root(self, benchmark, scan_tree):
        """A claimed root is never walked: cost is independent of chunk count."""
        _, plate, n_files = scan_tree

        def run():
            reg = _CountingRegistry()
            state = discover_under(plate, reg)
            return reg.probes, len(state.claims)

        probes, n_sources = benchmark(run)

        benchmark.extra_info.update(
            interior_files=n_files, probes=probes, sources=n_sources, shape="root"
        )
        assert (probes, n_sources) == (1, 1)
