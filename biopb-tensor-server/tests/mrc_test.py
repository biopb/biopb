"""Tests for the MRC electron-microscopy adapter."""

import struct
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from biopb_tensor_server.adapters.mrc import MrcAdapter
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

# The MRC header is parsed by rosettasciio (the [em] extra); skip the whole
# module when it is not installed rather than erroring at collection.
pytest.importorskip("rsciio")

from biopb.tensor.ticket_pb2 import ChunkBounds  # noqa: E402

# numpy dtype -> MRC MODE code
_MODE = {
    np.dtype("int8"): 0,
    np.dtype("int16"): 1,
    np.dtype("float32"): 2,
    np.dtype("uint16"): 6,
    np.dtype("float16"): 12,
}


def create_synthetic_mrc(
    path: Path,
    shape: tuple = (4, 8, 8),  # (nz, ny, nx)
    dtype=np.float32,
    cell: tuple = (10.0, 10.0, 10.0),  # (x, y, z) cell size in Angstrom
):
    """Write a minimal standard MRC-2014 file with deterministic data.

    Returns the numpy array written (order z, y, x) for round-trip comparison.
    """
    nz, ny, nx = shape
    dtype = np.dtype(dtype)
    data = np.arange(nz * ny * nx, dtype=dtype).reshape(nz, ny, nx)

    hdr = bytearray(1024)
    struct.pack_into("<3i", hdr, 0, nx, ny, nz)  # NX NY NZ (NX fastest)
    struct.pack_into("<i", hdr, 12, _MODE[dtype])  # MODE
    struct.pack_into("<3i", hdr, 28, nx, ny, nz)  # MX MY MZ
    struct.pack_into("<3f", hdr, 40, *cell)  # cell dims Xlen Ylen Zlen (A)
    struct.pack_into("<3i", hdr, 64, 1, 2, 3)  # MAPC MAPR MAPS
    hdr[208:212] = b"MAP "  # map stamp
    struct.pack_into("<i", hdr, 212, 0x00004144)  # little-endian machine stamp

    with open(path, "wb") as f:
        f.write(bytes(hdr))
        f.write(data.tobytes())
    return data


class TestMrcAdapterClaim:
    """Tests for MrcAdapter.claim()."""

    @pytest.mark.parametrize("ext", [".mrc", ".mrcs", ".rec", ".st", ".map"])
    def test_claim_mrc_family(self, ext):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / f"vol{ext}"
            create_synthetic_mrc(p)
            claim = MrcAdapter.claim(ClaimContext(p), DiscoveryState())
            assert claim is not None
            assert claim.source_type == "mrc"
            assert claim.primary_path == str(p)

    def test_claim_non_mrc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.txt"
            p.write_text("not mrc")
            assert MrcAdapter.claim(ClaimContext(p), DiscoveryState()) is None

    def test_claim_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert (
                MrcAdapter.claim(ClaimContext(Path(tmpdir)), DiscoveryState()) is None
            )


class TestMrcAdapter:
    """Tests for MrcAdapter functionality."""

    def _adapter(self, tmpdir, **kw):
        p = Path(tmpdir) / "vol.mrc"
        data = create_synthetic_mrc(p, **kw)
        return MrcAdapter.create_from_config(SourceConfig(url=str(p))), data

    def test_descriptor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter, data = self._adapter(tmpdir, shape=(4, 8, 8))
            desc = adapter.get_tensor_descriptor()
            assert list(desc.shape) == [4, 8, 8]
            assert desc.dtype == np.dtype("float32").str
            assert list(desc.dim_labels) == ["z", "y", "x"]
            assert list(desc.chunk_shape) == [4, 8, 8]  # single chunk

    def test_uses_memmap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter, _ = self._adapter(tmpdir)
            assert adapter._mm is not None  # own memmap, not the dask fallback

    def test_get_data_subregion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter, data = self._adapter(tmpdir, shape=(4, 8, 8))
            sub = adapter.get_data(ChunkBounds(start=[1, 2, 3], stop=[3, 6, 7]))
            np.testing.assert_array_equal(sub, data[1:3, 2:6, 3:7])

    def test_get_data_full(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter, data = self._adapter(tmpdir, shape=(3, 5, 7), dtype=np.int16)
            sub = adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=[3, 5, 7]))
            np.testing.assert_array_equal(sub, data)
            assert sub.dtype == np.int16

    def test_get_data_out_of_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter, _ = self._adapter(tmpdir, shape=(4, 8, 8))
            with pytest.raises(ValueError):
                adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=[99, 8, 8]))

    def test_physical_scale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # cell (x,y,z)=(10,10,10) A over MX,MY,MZ=(8,8,4) -> per-voxel nm:
            # x=y=(10/8)/10=0.125, z=(10/4)/10=0.25
            adapter, _ = self._adapter(tmpdir, shape=(4, 8, 8), cell=(10, 10, 10))
            scale, unit = adapter._physical_scale()
            assert scale == pytest.approx([0.25, 0.125, 0.125])
            assert unit[1] == "nm" and unit[2] == "nm"

    def test_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter, _ = self._adapter(tmpdir)
            meta = adapter.get_metadata()
            assert meta["format"] == "mrc"
            assert "std_header" in meta


class TestMrcAdapterIntegration:
    """Server -> client -> dask round-trip."""

    def test_server_client_roundtrip(self):
        from biopb.tensor import TensorFlightClient
        from biopb_tensor_server import TensorFlightServer

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "vol.mrc"
            data = create_synthetic_mrc(p, shape=(4, 16, 16))
            adapter = MrcAdapter.create_from_config(SourceConfig(url=str(p)))
            source_id = adapter.source_id

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
                darr = client.get_tensor(
                    source_id
                )  # single-tensor: array_id == source_id
                assert tuple(darr.shape) == (4, 16, 16)
                np.testing.assert_array_equal(darr.compute(), data)
                client.close()
            finally:
                server.shutdown()
