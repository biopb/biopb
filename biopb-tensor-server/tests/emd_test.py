"""Tests for the EMD electron-microscopy adapter."""

import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from biopb_tensor_server.adapters.emd import EmdAdapter
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

# EMD is read via rosettasciio (the [em] extra), backed by h5py; skip the whole
# module when either is absent rather than erroring at collection.
pytest.importorskip("rsciio")
pytest.importorskip("h5py")

from biopb.tensor.ticket_pb2 import ChunkBounds  # noqa: E402


def create_synthetic_emd(
    path: Path,
    shape: tuple = (2, 3, 8, 8),
    dtype=np.uint16,
    chunks: tuple = (1, 1, 8, 8),
):
    """Write a minimal Berkeley/NCEM EMD file (HDF5) rosettasciio can read.

    Returns the numpy array written (native, pre-transpose order).
    """
    import h5py

    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    with h5py.File(path, "w") as f:
        f.attrs["version_major"] = 0
        f.attrs["version_minor"] = 2
        g = f.create_group("data/datacube_000")
        g.attrs["emd_group_type"] = 1
        d = g.create_dataset("data", data=data, chunks=chunks)
        for i, nm in enumerate(["dim1", "dim2", "dim3", "dim4"][: len(shape)]):
            ax = g.create_dataset(nm, data=np.arange(d.shape[i], dtype=np.float32))
            ax.attrs["name"] = nm
            ax.attrs["units"] = "nm"
    return data


def _emd_expected(path):
    """Ground truth: what rosettasciio itself reads (post axis-transpose)."""
    from rsciio.emd import file_reader

    return file_reader(str(path), lazy=False)[0]["data"]


class TestEmdAdapterClaim:
    """Tests for EmdAdapter.claim()."""

    def test_claim_emd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.emd"
            create_synthetic_emd(p)
            claim = EmdAdapter.claim(ClaimContext(p), DiscoveryState())
            assert claim is not None
            assert claim.source_type == "emd"
            assert claim.primary_path == str(p)

    def test_claim_non_emd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.h5"
            p.write_bytes(b"\x89HDF\r\n\x1a\n")
            assert EmdAdapter.claim(ClaimContext(p), DiscoveryState()) is None

    def test_claim_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert (
                EmdAdapter.claim(ClaimContext(Path(tmpdir)), DiscoveryState()) is None
            )


class TestEmdAdapter:
    """Tests for EmdAdapter functionality (multi-tensor source)."""

    def test_list_tensors_and_native_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.emd"
            create_synthetic_emd(p, shape=(2, 3, 8, 8), chunks=(1, 1, 8, 8))
            adapter = EmdAdapter.create_from_config(SourceConfig(url=str(p)))

            descs = adapter.list_tensor_descriptors()
            assert len(descs) == 1
            d = descs[0]
            # array_id is source_id/field
            assert d.array_id == f"{adapter.source_id}/0"
            # rsciio reverses axis order; chunk_shape is the native HDF5 grid,
            # reversed to match (native (1,1,8,8) -> (8,8,1,1)).
            assert list(d.chunk_shape) == [8, 8, 1, 1]
            assert list(d.shape) == [8, 8, 3, 2]
            assert d.dtype == np.dtype("uint16").str

    def test_get_tensor_adapter_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.emd"
            create_synthetic_emd(p)
            adapter = EmdAdapter.create_from_config(SourceConfig(url=str(p)))
            expected = _emd_expected(p)

            field = adapter._within_source_field(
                adapter.list_tensor_descriptors()[0].array_id
            )
            ta = adapter.get_tensor_adapter(field)
            assert ta.get_tensor_descriptor().array_id == f"{adapter.source_id}/0"

            stop = list(expected.shape)
            sub = ta.get_data(
                ChunkBounds(start=[0, 0, 0, 0], stop=[s // 2 or 1 for s in stop])
            )
            exp = np.asarray(expected)[tuple(slice(0, s // 2 or 1) for s in stop)]
            np.testing.assert_array_equal(sub, exp)

    def test_source_level_get_data_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.emd"
            create_synthetic_emd(p)
            adapter = EmdAdapter.create_from_config(SourceConfig(url=str(p)))
            with pytest.raises(ValueError):
                adapter.get_data(ChunkBounds(start=[0, 0, 0, 0], stop=[1, 1, 1, 1]))

    def test_unknown_signal_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.emd"
            create_synthetic_emd(p)
            adapter = EmdAdapter.create_from_config(SourceConfig(url=str(p)))
            with pytest.raises(ValueError):
                adapter.get_tensor_adapter("99")


# NOTE: Velox/ThermoFisher eager-fallback (rsciio's Velox 4D-STEM lazy is a TODO)
# is exercised only by a real Velox fixture; not synthesizable via plain h5py.
# The code path (non-dask -> da.from_array wrap with a warning) is covered by
# manual verification against a real .emd; add a fixture-backed test if one lands.


class TestEmdAdapterIntegration:
    """Server -> client -> dask round-trip for one EMD signal."""

    def test_server_client_roundtrip(self):
        from biopb.tensor import TensorFlightClient
        from biopb_tensor_server import TensorFlightServer

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.emd"
            create_synthetic_emd(p, shape=(2, 3, 16, 16), chunks=(1, 1, 16, 16))
            adapter = EmdAdapter.create_from_config(SourceConfig(url=str(p)))
            source_id = adapter.source_id
            expected = np.asarray(_emd_expected(p))
            array_id = adapter.list_tensor_descriptors()[0].array_id

            server = TensorFlightServer("grpc://localhost:0")
            server.register_source(source_id, adapter)
            server.mark_ready()
            t = threading.Thread(target=server.serve, daemon=True)
            t.start()
            time.sleep(1)
            try:
                client = TensorFlightClient(
                    f"grpc://localhost:{server.port}", cache_bytes=10_000_000
                )
                assert source_id in client.list_sources()
                darr = client.get_tensor(array_id)  # source_id/field
                assert tuple(darr.shape) == tuple(expected.shape)
                np.testing.assert_array_equal(darr.compute(), expected)
                client.close()
            finally:
                server.shutdown()
