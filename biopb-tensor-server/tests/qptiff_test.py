"""Tests for the QPTIFF (Akoya PhenoImager) adapter."""

import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from biopb_tensor_server.adapters.qptiff import QptiffAdapter
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

# QPTIFF is read via tifffile (core) but its compressed tiles decode through
# imagecodecs; the synthetic fixtures below are uncompressed, but skip the module
# when imagecodecs is absent to match the registration gate.
pytest.importorskip("imagecodecs")

import tifffile  # noqa: E402
from biopb.tensor.descriptor_pb2 import TensorDescriptor  # noqa: E402
from biopb.tensor.ticket_pb2 import ChunkBounds  # noqa: E402

_QPI_DESC = (
    "<PerkinElmer-QPI-ImageDescription>"
    "<Objective>x20</Objective><Name>{name}</Name>"
    "</PerkinElmer-QPI-ImageDescription>"
)


def create_synthetic_qptiff(
    path: Path,
    n_channels: int = 3,
    base: int = 512,
    n_levels: int = 3,
    markers=("DAPI", "CD8", "PanCK"),
    tile: int = 256,
):
    """Write a minimal pyramidal multichannel QPTIFF tifffile can read.

    Level 0 is (n_channels, base, base); each further level halves Y/X. Returns the
    full-resolution numpy array (c, y, x) for round-trip comparison.
    """
    data = np.arange(n_channels * base * base, dtype=np.uint16).reshape(
        n_channels, base, base
    )
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        for lvl in range(n_levels):
            step = 2**lvl
            arr = data[:, ::step, ::step]
            opts = dict(photometric="minisblack", tile=(tile, tile))
            if lvl == 0:
                # Marker name in the first channel page's vendor XML.
                tw.write(
                    arr,
                    subifds=n_levels - 1,
                    description=_QPI_DESC.format(name=markers[0]),
                    **opts,
                )
            else:
                tw.write(arr, subfiletype=1, **opts)
    return data


def _adapter(path: Path) -> QptiffAdapter:
    return QptiffAdapter.create_from_config(SourceConfig(url=str(path)))


class TestQptiffAdapterClaim:
    def test_claim_qptiff_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "slide.qptiff"
            create_synthetic_qptiff(p)
            claim = QptiffAdapter.claim(ClaimContext(p), DiscoveryState())
            assert claim is not None
            assert claim.source_type == "qptiff"
            assert claim.primary_path == str(p)

    def test_claim_tif_with_vendor_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "slide.tif"  # QPTIFF saved as .tif
            create_synthetic_qptiff(p)
            claim = QptiffAdapter.claim(ClaimContext(p), DiscoveryState())
            assert claim is not None
            assert claim.source_type == "qptiff"

    def test_decline_plain_tif(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "plain.tif"
            tifffile.imwrite(p, np.zeros((16, 16), np.uint8))  # no vendor XML
            assert QptiffAdapter.claim(ClaimContext(p), DiscoveryState()) is None

    def test_decline_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert (
                QptiffAdapter.claim(ClaimContext(Path(tmp)), DiscoveryState()) is None
            )


class TestQptiffAdapter:
    def test_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "slide.qptiff"
            create_synthetic_qptiff(p, n_channels=3, base=512, tile=256)
            desc = _adapter(p).get_tensor_descriptor()
            assert list(desc.shape) == [3, 512, 512]
            assert list(desc.dim_labels) == ["c", "y", "x"]
            assert desc.dtype == np.dtype("uint16").str
            # Native tile grid as the advertised access chunk.
            assert list(desc.chunk_shape) == [1, 256, 256]

    def test_get_data_subregion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "slide.qptiff"
            data = create_synthetic_qptiff(p)
            sub = _adapter(p).get_data(
                ChunkBounds(start=[0, 10, 20], stop=[2, 100, 200])
            )
            np.testing.assert_array_equal(sub, data[0:2, 10:100, 20:200])

    def test_get_data_out_of_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "slide.qptiff"
            create_synthetic_qptiff(p, base=512)
            with pytest.raises(ValueError):
                _adapter(p).get_data(ChunkBounds(start=[0, 0, 0], stop=[3, 999, 512]))

    def test_native_pyramid_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "slide.qptiff"
            create_synthetic_qptiff(p, n_channels=3, base=512, n_levels=3)
            adapter = _adapter(p)
            assert adapter.has_native_pyramid()
            levels = adapter.get_native_pyramid_levels()
            assert [list(lv.shape) for lv in levels] == [
                [3, 512, 512],
                [3, 256, 256],
                [3, 128, 128],
            ]
            # Level 0 = [1,1,1]; each further level doubles the Y/X factor.
            assert [list(lv.scale_hint) for lv in levels] == [
                [1, 1, 1],
                [1, 2, 2],
                [1, 4, 4],
            ]
            assert all(
                lv.native and lv.reduction_method == "precompute" for lv in levels
            )

    def test_read_plan_routes_to_native_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "slide.qptiff"
            create_synthetic_qptiff(p, n_channels=3, base=512, n_levels=3)
            adapter = _adapter(p)
            # Request the [1,2,2] level via precompute.
            req = TensorDescriptor(
                array_id=adapter.array_id,
                dim_labels=["c", "y", "x"],
                shape=[3, 512, 512],
                dtype=np.dtype("uint16").str,
                scale_hint=[1, 2, 2],
                reduction_method="precompute",
            )
            plan = adapter.get_read_plan(req)
            # Chunk_ids carry the level suffix so DoGet dispatches to the level.
            assert plan.chunk_endpoints
            level_adapter = adapter.get_level_adapter("1")
            assert level_adapter.get_tensor_descriptor().array_id == (
                f"{adapter.source_id}/1"
            )
            assert list(level_adapter.get_tensor_descriptor().shape) == [3, 256, 256]

    def test_read_plan_unknown_scale_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "slide.qptiff"
            create_synthetic_qptiff(p)
            req = TensorDescriptor(
                array_id="x",
                shape=[3, 512, 512],
                dim_labels=["c", "y", "x"],
                dtype=np.dtype("uint16").str,
                scale_hint=[1, 3, 3],  # no such level
                reduction_method="precompute",
            )
            with pytest.raises(ValueError):
                _adapter(p).get_read_plan(req)

    def test_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "slide.qptiff"
            create_synthetic_qptiff(p, markers=("DAPI", "CD8", "PanCK"))
            meta = _adapter(p).get_metadata()
            assert meta["format"] == "qptiff"
            # First channel page carries the marker name in the vendor XML.
            assert "DAPI" in meta.get("channels", [])


class TestQptiffAdapterIntegration:
    """Server -> client -> dask round-trip, incl. a native-level scaled read."""

    def test_server_client_roundtrip(self):
        from biopb.tensor import TensorFlightClient
        from biopb_tensor_server import TensorFlightServer

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "slide.qptiff"
            data = create_synthetic_qptiff(p, n_channels=2, base=512, n_levels=3)
            adapter = _adapter(p)
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
                # Full-resolution read (single-tensor: array_id == source_id).
                darr = client.get_tensor(source_id)
                assert tuple(darr.shape) == (2, 512, 512)
                np.testing.assert_array_equal(darr.compute(), data)

                # Native-pyramid overview: a "precompute" read of the /4 scale must
                # return the on-disk level verbatim (subsampled), NOT an area
                # recompute of full res -- that is the whole point of #135.
                lvl2 = client.get_tensor(
                    source_id, scale_hint=[1, 4, 4], reduction_method="precompute"
                )
                assert tuple(lvl2.shape) == (2, 128, 128)
                np.testing.assert_array_equal(lvl2.compute(), data[:, ::4, ::4])
                client.close()
            finally:
                server.shutdown()
