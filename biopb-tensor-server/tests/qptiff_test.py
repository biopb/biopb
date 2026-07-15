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
    baseline_marker: str = "DAPI",
    tile: int = 256,
):
    """Write a minimal pyramidal multichannel QPTIFF tifffile can read.

    Level 0 is (n_channels, base, base); each further level halves Y/X. Returns the
    full-resolution numpy array (c, y, x) for round-trip comparison.

    Only the *baseline* page (channel 0) carries a vendor <Name>. A real QPTIFF
    names every channel page, but tifffile's writer can't reproduce that here: the
    moment its pages carry differing ImageDescriptions it splits them into separate
    series, which would collapse the (c, y, x) tensor the read path needs -- and a
    single multichannel write attaches the description to page 0 only. So this
    fixture exercises "one named channel, the rest unnamed", which is exactly the
    positional/gap case get_metadata must not truncate.
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
                tw.write(
                    arr,
                    subifds=n_levels - 1,
                    description=_QPI_DESC.format(name=baseline_marker),
                    **opts,
                )
            else:
                tw.write(arr, subfiletype=1, **opts)
    return data


def _adapter(path: Path) -> QptiffAdapter:
    return QptiffAdapter.create_from_config(SourceConfig(url=str(path)))


def _wait_until_serving(port: int, timeout: float = 5.0) -> None:
    """Block until the server reports SERVING, or raise past the deadline.

    More reliable than a fixed sleep: the port binds the moment serve() runs, but
    a read races if it lands before mark_ready(), and a fixed sleep is both slower
    on a warm machine and flakier under CI load. Retry on FlightUnavailableError
    (port not up yet) and STARTING until health flips to SERVING.
    """
    import json

    from pyarrow import flight

    deadline = time.monotonic() + timeout
    with flight.FlightClient(f"grpc://localhost:{port}") as probe:
        while True:
            try:
                (raw,) = list(probe.do_action(flight.Action("health", b"")))
                if json.loads(raw.body.to_pybytes()).get("status") == "SERVING":
                    return
            except flight.FlightUnavailableError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"server on :{port} did not reach SERVING in time")
            time.sleep(0.02)


class TestQptiffAdapterClaim:
    def test_claim_qptiff_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "slide.qptiff"
            create_synthetic_qptiff(p)
            claim = QptiffAdapter.claim(ClaimContext(p), DiscoveryState())
            assert claim is not None
            assert claim.source_type == "qptiff"
            assert claim.primary_path == str(p)

    def test_decline_tif_even_with_vendor_xml(self):
        # Claim is suffix-only (biopb/biopb#135): a QPTIFF saved as .tif is NOT
        # claimed -- sniffing the vendor XML on the claim path is disabled, so it
        # falls through to the generic bioio adapter. Use type: qptiff to force
        # the native-pyramid path on a .tif-named QPTIFF.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "slide.tif"  # QPTIFF saved as .tif
            create_synthetic_qptiff(p)
            assert QptiffAdapter.claim(ClaimContext(p), DiscoveryState()) is None

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

    def test_single_level_advertises_computed_not_native_pyramid(self):
        # A QPTIFF with no on-disk pyramid (1 level) must NOT advertise a native
        # pyramid: a pyramid needs >=2 levels, so has_native_pyramid is False and
        # get_native_pyramid_levels is None, dropping it onto the server's computed
        # path (like OME-Zarr single-level). The precache worker keys on
        # has_native_pyramid, so a wrong True here would suppress overview warming.
        from biopb_tensor_server.core.config import PyramidConfig

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "flat.qptiff"
            data = create_synthetic_qptiff(p, n_channels=2, base=256, n_levels=1)
            adapter = _adapter(p)
            assert adapter._n_levels() == 1
            assert adapter.has_native_pyramid() is False
            assert adapter.get_native_pyramid_levels() is None

            desc = adapter.get_tensor_descriptor()
            # Forced low threshold -> the computed plan has >=2 levels, all
            # native=False / reduction=area (on-the-fly downsample from level 0).
            cfg = PyramidConfig(
                reduction_method="area",
                threshold=64,
                downscale_factor=4,
                pixel_budget_cubic_root=512,
            )
            levels = adapter._advertised_pyramid(desc, cfg)
            assert len(levels) >= 2
            assert all(not lv.native for lv in levels)
            assert [list(lv.scale_hint) for lv in levels] == [[1, 1, 1], [1, 4, 4]]
            # Full read still round-trips through the base (non-precompute) path.
            full = adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=list(desc.shape)))
            np.testing.assert_array_equal(full, data)

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

    def test_level_adapter_is_a_full_backend(self):
        # The level adapter must be a genuine source+tensor adapter (like
        # OME-Zarr's ZarrAdapter level), not a partial TensorAdapter that
        # hardcodes array_id -- so code treating it as a full backend (metadata-DB
        # sync, source-level ops) finds source_id/array_id/source_url.
        from biopb_tensor_server.core.base import SourceAdapter, TensorAdapter

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "slide.qptiff"
            create_synthetic_qptiff(p, n_channels=3, base=512, n_levels=3)
            adapter = _adapter(p)
            lvl = adapter.get_level_adapter("1")

            assert isinstance(lvl, SourceAdapter) and isinstance(lvl, TensorAdapter)
            # Identity is built by the base array_id property from source_id +
            # _tensor_name, not hardcoded into the descriptor.
            assert lvl.source_id == adapter.source_id
            assert lvl._tensor_name == "1"
            assert lvl.array_id == f"{adapter.source_id}/1"
            # Source-level surface resolves without AttributeError, and carries the
            # real file url + this format (not ZarrAdapter's synthetic-store repr).
            assert lvl.source_url == adapter._source_url
            assert lvl.source_type == "qptiff"
            src_desc = lvl.get_source_descriptor()
            assert src_desc.source_id == adapter.source_id
            assert [d.array_id for d in lvl.list_tensor_descriptors()] == [
                f"{adapter.source_id}/1"
            ]
            # Cached: repeated calls return the same instance.
            assert adapter.get_level_adapter("1") is lvl

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
            create_synthetic_qptiff(p, n_channels=3, baseline_marker="DAPI")
            meta = _adapter(p).get_metadata()
            assert meta["format"] == "qptiff"
            # channels is positional (one entry per channel): the named baseline
            # plus None gaps -- NOT collapsed to ["DAPI"], which would misalign.
            assert meta["channels"] == ["DAPI", None, None]
            # Vendor XML is returned whole, not sliced to a byte cap.
            assert meta["image_description"].endswith(
                "</PerkinElmer-QPI-ImageDescription>"
            )


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
            _wait_until_serving(server.port)
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
