"""Conformance tests for the compact-grid GetFlightInfo response (biopb/biopb#346).

When the client sets ``TensorReadOption.compact_grid_ok`` and the plan is a
regular chunk grid, the server omits the per-chunk endpoint list and instead
sets ``TensorDescriptor.chunk_array_id`` (plus the realized ``slice_hint``), so
the client can regenerate every ``(chunk_id, bounds)`` arithmetically.

The contract these tests lock: **reconstructing from the compact descriptor
reproduces the explicit endpoint list byte-for-byte** -- same chunk_ids, same
logical bounds, same order -- for plain, sliced, scaled, and precomputed-pyramid
reads. ``_reconstruct`` here is the reference the Python client (PR-C) mirrors.
"""

import numpy as np
import pytest
from biopb.tensor.descriptor_pb2 import FlightCmd, TensorDescriptor, TensorReadOption
from biopb.tensor.ticket_pb2 import ChunkBounds, TensorTicket
from biopb_tensor_server import TensorFlightServer
from biopb_tensor_server.core.chunk import (
    encode_chunk_id,
    encode_chunk_id_with_scale,
)

try:
    import pyarrow.flight as flight

    _HAVE_FLIGHT = True
except ImportError:  # pragma: no cover
    _HAVE_FLIGHT = False

pytestmark = pytest.mark.skipif(not _HAVE_FLIGHT, reason="pyarrow.flight unavailable")


def _ceil_div(a, b):
    return -(-a // b)


def _reconstruct(desc):
    """Reference client-side reconstruction of the explicit plan from a compact
    descriptor. Returns ``[(chunk_id, logical_start, logical_stop), ...]`` in the
    server's ``np.ndindex`` (row-major) chunk order.
    """
    ndim = len(desc.shape)
    logical_chunk = tuple(int(c) for c in desc.chunk_shape)
    scale = tuple(int(s) for s in desc.scale_hint) or (1,) * ndim
    scaled = bool(len(desc.scale_hint))
    method = desc.reduction_method
    chunk_array_id = desc.chunk_array_id
    # Realized virtual-coordinate bounds carried by slice_hint (always set in
    # compact mode).
    rstart = tuple(int(s) for s in desc.slice_hint.start)
    rstop = tuple(int(s) for s in desc.slice_hint.stop)
    # virtual chunk size = logical chunk size * scale (exact: the server's
    # virtual_chunk_size is a multiple of scale, and logical = virtual // scale).
    vcs = tuple(logical_chunk[d] * scale[d] for d in range(ndim))
    n = tuple(_ceil_div(rstop[d] - rstart[d], vcs[d]) for d in range(ndim))

    out = []
    for idx in np.ndindex(*n):
        vstart = tuple(rstart[d] + idx[d] * vcs[d] for d in range(ndim))
        vstop = tuple(min(vstart[d] + vcs[d], rstop[d]) for d in range(ndim))
        vbounds = ChunkBounds(start=list(vstart), stop=list(vstop))
        if scaled:
            cid = encode_chunk_id_with_scale(chunk_array_id, vbounds, scale, method)
            lstart = tuple((vstart[d] - rstart[d]) // scale[d] for d in range(ndim))
            lstop = tuple(
                _ceil_div(vstop[d] - rstart[d], scale[d]) for d in range(ndim)
            )
        else:
            cid = encode_chunk_id(chunk_array_id, vbounds)
            lstart = tuple(vstart[d] - rstart[d] for d in range(ndim))
            lstop = tuple(vstop[d] - rstart[d] for d in range(ndim))
        out.append((cid, lstart, lstop))
    return out


def _explicit_endpoints(info):
    """Parse ``(chunk_id, logical_start, logical_stop)`` from a FlightInfo's
    explicit endpoint list, exactly as the client does today."""
    out = []
    for ep in info.endpoints:
        ticket = TensorTicket.FromString(ep.ticket.ticket)
        bounds = ChunkBounds.FromString(ep.app_metadata)
        out.append((ticket.chunk_id, tuple(bounds.start), tuple(bounds.stop)))
    return out


def _flight_info(server, source_id, read_opt):
    cmd = FlightCmd(source_id=source_id, tensor_read=read_opt)
    desc = flight.FlightDescriptor.for_command(cmd.SerializeToString())
    # A tokenless source's _authorize_source ignores context, so None is fine.
    return server.get_flight_info(None, desc)


def _assert_compact_matches_explicit(server, source_id, base_read_opt):
    """The heart of the contract: for the same read, the compact response
    reconstructs to the explicit response, byte-identical and in order."""
    explicit_opt = TensorReadOption()
    explicit_opt.CopyFrom(base_read_opt)
    explicit_opt.compact_grid_ok = False
    compact_opt = TensorReadOption()
    compact_opt.CopyFrom(base_read_opt)
    compact_opt.compact_grid_ok = True

    explicit_info = _flight_info(server, source_id, explicit_opt)
    compact_info = _flight_info(server, source_id, compact_opt)

    explicit = _explicit_endpoints(explicit_info)
    assert explicit, "expected a non-empty explicit plan for this read"

    # Compact response carries no endpoints, but does carry chunk_array_id.
    assert list(compact_info.endpoints) == []
    compact_desc = TensorDescriptor.FromString(compact_info.descriptor.command)
    assert compact_desc.chunk_array_id != ""
    assert len(compact_desc.slice_hint.start) == len(compact_desc.shape)

    reconstructed = _reconstruct(compact_desc)
    assert reconstructed == explicit


def _serve(source_id, adapter):
    server = TensorFlightServer("grpc://localhost:0")
    server.register_source(source_id, adapter)
    return server


class TestCompactGridConformance:
    @pytest.mark.skipif(
        "not __import__('importlib').util.find_spec('zarr')",
        reason="zarr not available",
    )
    def test_plain_full_read(self, simple_zarr_array):
        import zarr
        from biopb_tensor_server import ZarrAdapter

        zarr_path, shape, chunks = simple_zarr_array
        arr = zarr.open_array(zarr_path, mode="r")
        server = _serve("plain", ZarrAdapter(arr, "plain", ["y", "x"]))
        _assert_compact_matches_explicit(server, "plain", TensorReadOption())

    @pytest.mark.skipif(
        "not __import__('importlib').util.find_spec('zarr')",
        reason="zarr not available",
    )
    def test_sliced_read(self, simple_zarr_array):
        import zarr
        from biopb.tensor.descriptor_pb2 import SliceHint
        from biopb_tensor_server import ZarrAdapter

        zarr_path, shape, chunks = simple_zarr_array
        arr = zarr.open_array(zarr_path, mode="r")
        server = _serve("sliced", ZarrAdapter(arr, "sliced", ["y", "x"]))
        # An off-grid sub-region: the server snaps to the chunk grid.
        opt = TensorReadOption(slice_hint=SliceHint(start=[10, 20], stop=[100, 110]))
        _assert_compact_matches_explicit(server, "sliced", opt)

    @pytest.mark.skipif(
        "not __import__('importlib').util.find_spec('zarr')",
        reason="zarr not available",
    )
    def test_scaled_virtual_read(self, simple_zarr_array):
        import zarr
        from biopb_tensor_server import ZarrAdapter

        zarr_path, shape, chunks = simple_zarr_array
        arr = zarr.open_array(zarr_path, mode="r")
        server = _serve("scaled", ZarrAdapter(arr, "scaled", ["y", "x"]))
        opt = TensorReadOption(scale_hint=[2, 2], reduction_method="area")
        _assert_compact_matches_explicit(server, "scaled", opt)

    @pytest.mark.skipif(
        "not __import__('importlib').util.find_spec('zarr')",
        reason="zarr not available",
    )
    def test_precompute_pyramid_read(self, multires_ome_zarr):
        """The precompute path is why chunk_array_id exists: the plan targets a
        level adapter whose chunk_ids carry ``source_id/{level}``, while the
        descriptor's array_id is reset to the base tensor."""
        import zarr
        from biopb_tensor_server import OmeZarrAdapter

        zarr_path, level_paths, zattrs = multires_ome_zarr
        root = zarr.open_group(zarr_path, mode="r")
        server = _serve("ome", OmeZarrAdapter(root["0"], "ome"))
        opt = TensorReadOption(scale_hint=[2, 2], reduction_method="precompute")

        # Sanity: chunk_array_id must be the level id, not the base array_id.
        compact_opt = TensorReadOption()
        compact_opt.CopyFrom(opt)
        compact_opt.compact_grid_ok = True
        info = _flight_info(server, "ome", compact_opt)
        desc = TensorDescriptor.FromString(info.descriptor.command)
        assert desc.chunk_array_id != desc.array_id

        _assert_compact_matches_explicit(server, "ome", opt)


class TestCompactGridOptOut:
    @pytest.mark.skipif(
        "not __import__('importlib').util.find_spec('zarr')",
        reason="zarr not available",
    )
    def test_default_still_returns_explicit_endpoints(self, simple_zarr_array):
        """An old / non-opted-in client (compact_grid_ok defaults False) still
        gets the full explicit endpoint list and no chunk_array_id."""
        import zarr
        from biopb_tensor_server import ZarrAdapter

        zarr_path, shape, chunks = simple_zarr_array
        arr = zarr.open_array(zarr_path, mode="r")
        server = _serve("legacy", ZarrAdapter(arr, "legacy", ["y", "x"]))

        info = _flight_info(server, "legacy", TensorReadOption())
        assert len(info.endpoints) > 0
        desc = TensorDescriptor.FromString(info.descriptor.command)
        assert desc.chunk_array_id == ""
