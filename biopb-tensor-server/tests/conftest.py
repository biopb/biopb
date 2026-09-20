"""Test fixtures for biopb-tensor-server tests.

Imports fixture factory functions from fixtures module and wraps them as pytest fixtures.
"""

import os
import tempfile
import threading
from pathlib import Path

# Neutralise terminal colour before anything builds a Rich Console. Several
# tests assert on the *text* a CLI prints, and Rich emits ANSI escapes into
# captured output whenever it believes colour is wanted -- turning
# `assert "Deleted 1" in captured` into a comparison against "\x1b[3m...".
#
# `FORCE_COLOR` is the one that bites: Rich checks it *before* `NO_COLOR`, so
# setting `NO_COLOR` alone does not help, and several terminal tools and agent
# harnesses export it. CI has neither set, which is why these tests pass there
# and fail on a developer's machine -- the worst place for a test to disagree
# with CI. Duplicated from the repo-root conftest because this package sets its
# own `[tool.pytest.ini_options]`, which makes it pytest's rootdir, so the root
# conftest is never collected for these tests.
os.environ.pop("FORCE_COLOR", None)
os.environ["NO_COLOR"] = "1"
os.environ.setdefault("TERM", "dumb")

import pytest
from biopb.tensor.client import TensorFlightClient
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.config import CacheConfig
from biopb_tensor_server.fixtures import (
    create_5d_6d_micromanager_dataset,
    create_companion_ome_dataset,
    create_hdf5_dataset,
    create_multi_series_ome_tiff,
    create_multifile_micromanager_dataset,
    create_multifile_ome_dataset,
    create_multiresolution_ome_zarr,
    create_tiled_ome_tiff,
    create_zarr_array,
)

from tests import catalog_server

# =============================================================================
# pytest fixtures using the factory functions
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_upstream_client_pool():
    """Isolate the process-wide upstream client pool (biopb/biopb#266 B1).

    Pooled clients are keyed by ``(endpoint, token)`` and live until eviction or
    process exit, so a client dialed at a random port in one test could be
    handed back in a later test that happens to reuse the port. Clear the pool
    around every test to keep them independent (and to close leaked clients)."""
    from biopb_tensor_server.adapters.remote_tensor import _clear_client_pool

    _clear_client_pool()
    yield
    _clear_client_pool()


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def multifile_mm_dataset(temp_dir):
    """Complete Micro-Manager multi-file dataset."""
    return create_multifile_micromanager_dataset(temp_dir)


@pytest.fixture
def multifile_mm_dataset_incomplete(temp_dir):
    """Incomplete Micro-Manager multi-file dataset (missing one channel)."""
    return create_multifile_micromanager_dataset(temp_dir, complete=False)


@pytest.fixture
def multifile_5d_6d_mm_dataset(temp_dir):
    """Full 5D/6D MicroManager dataset with position, time, channel, and z dimensions."""
    return create_5d_6d_micromanager_dataset(
        temp_dir,
        n_positions=2,
        n_times=3,
        n_channels=2,
        n_z=4,
    )


@pytest.fixture
def multifile_ome_dataset(temp_dir):
    """Complete multi-file OME-TIFF dataset with OME-XML metadata."""
    return create_multifile_ome_dataset(temp_dir)


@pytest.fixture
def multifile_ome_dataset_incomplete(temp_dir):
    """Incomplete multi-file OME-TIFF dataset (missing one file)."""
    return create_multifile_ome_dataset(temp_dir, complete=False)


@pytest.fixture
def multires_ome_zarr(temp_dir):
    """Multi-resolution OME-Zarr dataset."""
    return create_multiresolution_ome_zarr(temp_dir)


@pytest.fixture
def tiled_ome_tiff(temp_dir):
    """Tiled OME-TIFF file."""
    return create_tiled_ome_tiff(temp_dir)


@pytest.fixture
def multi_series_ome_tiff(temp_dir):
    """Multi-series OME-TIFF file (multiple fields/positions)."""
    return create_multi_series_ome_tiff(temp_dir)


@pytest.fixture
def companion_ome_dataset(temp_dir):
    """Companion OME dataset with .companion.ome file."""
    return create_companion_ome_dataset(temp_dir)


@pytest.fixture
def hdf5_dataset(temp_dir):
    """HDF5 dataset with chunked array."""
    return create_hdf5_dataset(temp_dir)


@pytest.fixture
def simple_zarr_array(temp_dir):
    """Simple Zarr array for basic tests."""
    return create_zarr_array(temp_dir)


@pytest.fixture
def transfer_target(monkeypatch):
    """Set the transfer-size target in bytes for the duration of a test.

    Chunk size is the only knob on the transfer grid (biopb/biopb#809): there is
    no minimum-endpoint floor, so a tiny fixture is one chunk at the 8 MB
    default. A test whose subject is multi-chunk behaviour -- cache entries per
    region, grid snapping under a slice, endpoint counts -- lowers the target
    rather than relying on a fixture that happens to sit above a floor.
    """

    def _set(nbytes: int) -> int:
        from biopb_tensor_server.core import chunk

        monkeypatch.setattr(chunk, "PREFERRED_ARROW_BATCH_BYTES", int(nbytes))
        return int(nbytes)

    return _set


@pytest.fixture
def epoch(monkeypatch):
    """Set the serving-semantics epoch for the duration of a test.

    The epoch (biopb/biopb#1076) re-keys every chunk_id, so a test about
    invalidation bumps it rather than contriving a content change.
    """

    def _set(value: int) -> int:
        from biopb_tensor_server.core import chunk

        monkeypatch.setattr(chunk, "CHUNK_SEMANTICS_EPOCH", int(value))
        return int(value)

    return _set


@pytest.fixture
def writable_server(tmp_path):
    """A live, writable server on an OS-assigned port.

    No wait after `serve()`: `FlightServerBase` binds and starts serving in
    `__init__`, so the port is live before this returns -- the thread only parks
    on it.
    """
    CacheManager.reset()
    CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
    server = catalog_server(
        location="grpc://localhost:0", writable=True, write_dir=Path(tmp_path)
    )
    server.mark_ready()
    threading.Thread(target=server.serve, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        CacheManager.reset()


@pytest.fixture
def client(writable_server):
    """A client connected to `writable_server`."""
    c = TensorFlightClient(f"grpc://localhost:{writable_server.port}")
    yield c
    c.close()


@pytest.fixture
def cache(tmp_path):
    """A file-backed cache with the scaled-read knob on.

    The backend matters: `resolve_chunk_data` caches *unscaled* chunks only on
    the file backend, so it is the one where a full-resolution read leaves
    anything for a probe to find.
    """
    manager = CacheManager(
        CacheConfig(
            file_cache_dir=tmp_path / "cache",
            source_scaled_reads=True,
        )
    )
    try:
        yield manager
    finally:
        manager.close()
