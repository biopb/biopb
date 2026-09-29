"""Pin the load-bearing adapter registration order (claim precedence).

``AdapterRegistry.get_claims_for_path`` calls ``claim()`` in registration order
and stops at the FIRST adapter that returns a claim (callers take ``claims[0]``).
So registration order IS priority, and several correctness properties ride on it
purely by convention in ``adapters/__init__.py::get_default_registry`` -- with no
executable check until now. These tests make that ordering an invariant:

- ``ome-tiff`` before ``micromanager-legacy`` before ``tiff-sequence`` -- an
  ``.ome.tif`` must become its own OME-TIFF source (OmeTiffAdapter makes a
  file-level claim and consumes multi-file siblings) rather than being welded into
  a plain directory sequence, and a valid MicroManager dataset must be claimed +
  subtree-pruned before the generic sequence adapter ever probes it.
- ``ome-zarr`` before ``zarr`` -- a multiscales store must be claimed by the
  specific adapter, not the generic Zarr fallback.

If someone reorders ``get_default_registry`` these break, instead of the bug
surfacing only as mis-claimed data in the field.
"""

import importlib
import importlib.abc
import sys

import pytest
from biopb_tensor_server.adapters import (
    OmeTiffAdapter,
    OmeZarrAdapter,
    TiffSequenceAdapter,
    ZarrAdapter,
    get_default_registry,
)
from biopb_tensor_server.adapters.tiff import MicroManagerLegacyAdapter


def _order_indices(*adapter_classes):
    """Return each class's index in the default registry's probe order.

    Uses the FIRST occurrence to stay robust if an adapter class ever occupies
    more than one slot; its earliest slot is the one that governs precedence.
    """
    adapters = get_default_registry()._adapters
    return [
        min(i for i, c in enumerate(adapters) if c is cls) for cls in adapter_classes
    ]


def test_ome_tiff_before_micromanager_before_tiff_sequence():
    """The TIFF trio's claim precedence is ome-tiff > micromanager > sequence."""
    ome, mm, seq = _order_indices(
        OmeTiffAdapter, MicroManagerLegacyAdapter, TiffSequenceAdapter
    )
    assert ome < mm < seq


def test_ome_zarr_before_plain_zarr():
    """Specific OME-Zarr adapter must out-prioritize the generic Zarr fallback."""
    ome_zarr, plain_zarr = _order_indices(OmeZarrAdapter, ZarrAdapter)
    assert ome_zarr < plain_zarr


def test_all_three_tiff_adapters_are_registered():
    """Guard against a trio member silently dropping out of the default registry."""
    adapters = get_default_registry()._adapters
    for cls in (OmeTiffAdapter, MicroManagerLegacyAdapter, TiffSequenceAdapter):
        assert cls in adapters, f"{cls.__name__} missing from the default registry"


class TestMissingReaderIsNotClaimed:
    """A reader package that is not installed must leave its extension unclaimed.

    ``adapters/__init__.py`` gates each optional adapter on importing its reader
    (``readlif``, ``mrc``, ``nd2``, ``bioio``), not just its own module -- every
    one of those adapters imports the reader lazily, inside the method that uses
    it, so the module import alone succeeds with the package absent. Without the
    probe the adapter registers, claims its extension, and fails at read instead.

    That distinction became load-bearing when bioio left the default install
    (biopb/biopb#799): an install without it must decline a remote CZI, not
    claim one.
    """

    @staticmethod
    def _rebuild_without(package, attr):
        """Rebuild the default registry with *package* unimportable.

        Returns ``(registered_names, attr_value)`` snapshotted while the block is
        in force -- ``importlib.reload`` mutates the module in place, so the
        restoring reload in ``finally`` would otherwise undo what we assert on.
        """

        class _Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == package or fullname.startswith(package + "."):
                    raise ImportError(f"blocked: {fullname}")
                return None

        import biopb_tensor_server.adapters as adapters_mod

        blocker = _Blocker()
        saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == package}
        for name in saved:
            del sys.modules[name]
        sys.meta_path.insert(0, blocker)
        try:
            reloaded = importlib.reload(adapters_mod)
            registered = {
                cls.__name__ for cls in reloaded.get_default_registry()._adapters
            }
            return registered, getattr(reloaded, attr)
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(saved)
            # Restore the real module bindings for every later test in the run.
            importlib.reload(adapters_mod)

    @pytest.mark.parametrize(
        "package,attr",
        [
            ("readlif", "LifAdapter"),
            ("mrc", "DeltaVisionAdapter"),
            ("nd2", "Nd2Adapter"),
            ("bioio", "ZeissAdapter"),
        ],
    )
    def test_adapter_is_not_registered_without_its_reader(self, package, attr):
        pytest.importorskip(package)
        registered, adapter = self._rebuild_without(package, attr)

        assert adapter is None
        assert attr not in registered
