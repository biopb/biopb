"""Tests for the EMD electron-microscopy adapter."""

import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from biopb_tensor_server.adapters.emd import EmdAdapter
from biopb_tensor_server.core.adapter_base import strip_source_prefix
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

# EMD is read via rosettasciio (the [em] extra), backed by h5py; skip the whole
# module when either is absent rather than erroring at collection.
pytest.importorskip("rsciio")
pytest.importorskip("h5py")

from biopb.tensor.ticket_pb2 import ChunkBounds  # noqa: E402

from tests import catalog_server, register_and_catalog, source_ids


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


@contextmanager
def _emd_adapter(path):
    """An adapter over *path*, closed on exit: Windows cannot delete the file of
    a test's temp directory while the adapter holds it open."""
    adapter = EmdAdapter.create_from_config(SourceConfig(url=str(path)))
    try:
        yield adapter
    finally:
        adapter.close()


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
            with _emd_adapter(p) as adapter:
                descs = adapter.list_tensors()
                assert len(descs) == 1
                d = descs[0]
                # array_id is source_id/field
                assert d.array_id == f"{adapter.source_id}/0"
                assert list(d.shape) == [8, 8, 3, 2]

                # chunk_shape is the transfer grid (biopb/biopb#809), answered by the
                # signal-bound adapter: seeded by the native HDF5 blocks and reversed
                # with the axes like everything else rsciio reports: native
                # (1,1,8,8) -> (8,8,1,1), then grown in whole blocks because one
                # 128-byte block is far below the transfer target.
                signal = adapter.get_tensor_adapter(d.array_id)
                grid = list(signal.get_tensor_descriptor().chunk_shape)
                assert [grid[0], grid[1]] == [8, 8]
                assert all(
                    g % n == 0 and g <= s
                    for g, n, s in zip(grid, [8, 8, 1, 1], [8, 8, 3, 2], strict=True)
                )
                assert d.dtype == np.dtype("uint16").str

    def test_get_tensor_adapter_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.emd"
            create_synthetic_emd(p)
            with _emd_adapter(p) as adapter:
                expected = _emd_expected(p)

                field = strip_source_prefix(
                    adapter.source_id, adapter.list_tensors()[0].array_id
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
            with _emd_adapter(p) as adapter:
                with pytest.raises(ValueError):
                    adapter.get_data(ChunkBounds(start=[0, 0, 0, 0], stop=[1, 1, 1, 1]))

    def test_unknown_signal_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.emd"
            create_synthetic_emd(p)
            with _emd_adapter(p) as adapter:
                with pytest.raises(ValueError):
                    adapter.get_tensor_adapter("99")


def _file_is_held(path) -> bool:
    """Whether this process still has the HDF5 file open (h5py refuses a second
    open in another mode, and opening for append does not truncate)."""
    import h5py

    try:
        with h5py.File(path, "a"):
            return False
    except OSError:
        return True


class TestEmdAdapterHandle:
    """The file is held between reads, released on close or when idle, reopened by
    the next read. rosettasciio 0.15+ keeps the file open under ``lazy=True``."""

    _FIRST = ChunkBounds(start=[0, 0, 0, 0], stop=[4, 4, 1, 1])

    @pytest.fixture(autouse=True)
    def _needs_a_lazy_reader_that_holds_the_file(self):
        # Older rosettasciio reads eagerly, so there is no file to hold.
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "probe.emd"
            create_synthetic_emd(p)
            with _emd_adapter(p):
                if not _file_is_held(p):
                    pytest.skip("rosettasciio reads EMD eagerly here")

    def _read(self, adapter):
        return adapter.get_tensor_adapter("0").get_data(self._FIRST)

    def test_close_releases_the_file_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.emd"
            create_synthetic_emd(p)
            adapter = EmdAdapter.create_from_config(SourceConfig(url=str(p)))
            self._read(adapter)
            adapter.close()
            adapter.close()
            assert not _file_is_held(p)

    def test_a_read_after_close_reopens_the_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.emd"
            create_synthetic_emd(p)
            with _emd_adapter(p) as adapter:
                before = self._read(adapter)
                adapter.close()
                assert not _file_is_held(p)
                np.testing.assert_array_equal(self._read(adapter), before)
                assert _file_is_held(p)

    def test_a_source_rebuilt_from_its_row_holds_no_file_until_its_first_read(self):
        from tests.payload_equivalence import hydrate

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.emd"
            create_synthetic_emd(p)
            source = SourceConfig(url=str(p), type="emd", source_id="src")
            parsed = EmdAdapter.create_from_config(source)
            before = self._read(parsed)
            rebuilt = hydrate(parsed, source)
            parsed.close()
            try:
                assert rebuilt.list_tensors()
                assert not _file_is_held(p)  # built from the row, nothing opened
                np.testing.assert_array_equal(self._read(rebuilt), before)
                assert _file_is_held(p)
                rebuilt.close()
                assert not _file_is_held(p)
                np.testing.assert_array_equal(self._read(rebuilt), before)
                assert _file_is_held(p)  # released and reopened like any other
            finally:
                rebuilt.close()

    def test_an_idle_file_is_released_by_the_reaper_and_reopened_on_read(
        self, monkeypatch
    ):
        from biopb_tensor_server.adapters import _handle_reaper, emd

        reaper = _handle_reaper.IdleHandleReaper(0.01, "emd-test", max_handles=8)
        monkeypatch.setattr(emd, "_handle_reaper", reaper)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.emd"
            create_synthetic_emd(p)
            with _emd_adapter(p) as adapter:
                before = self._read(adapter)
                assert _file_is_held(p)
                time.sleep(0.05)
                reaper._sweep()
                assert not _file_is_held(p)
                np.testing.assert_array_equal(self._read(adapter), before)
                assert _file_is_held(p)


# NOTE: Velox/ThermoFisher eager-fallback (rsciio's Velox 4D-STEM lazy is a TODO)
# is exercised only by a real Velox fixture; not synthesizable via plain h5py.
# The code path (non-dask -> da.from_array wrap with a warning) is covered by
# manual verification against a real .emd; add a fixture-backed test if one lands.


class TestEmdAdapterIntegration:
    """Server -> client -> dask round-trip for one EMD signal."""

    def test_server_client_roundtrip(self):
        from biopb.tensor import TensorFlightClient

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.emd"
            create_synthetic_emd(p, shape=(2, 3, 16, 16), chunks=(1, 1, 16, 16))
            with _emd_adapter(p) as adapter:
                source_id = adapter.source_id
                expected = np.asarray(_emd_expected(p))
                array_id = adapter.list_tensors()[0].array_id

                server = catalog_server("localhost:0")
                register_and_catalog(server, source_id, adapter)
                server.mark_ready()
                t = threading.Thread(target=server.serve, daemon=True)
                t.start()
                time.sleep(1)
                try:
                    client = TensorFlightClient(
                        f"grpc://localhost:{server.port}", cache_bytes=10_000_000
                    )
                    assert source_id in source_ids(client)
                    darr = client.get_tensor(array_id)  # source_id/field
                    assert tuple(darr.shape) == tuple(expected.shape)
                    np.testing.assert_array_equal(darr.compute(), expected)
                    client.close()
                finally:
                    server.shutdown()
