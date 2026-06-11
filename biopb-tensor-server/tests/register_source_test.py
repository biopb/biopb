"""register_source contract: source_id must be non-empty and slash-free.

The tensor identity policy (proto/biopb/tensor/descriptor.proto) makes source_id
slash-free a load-bearing invariant: the internal chunk-route id is
"source_id/array_id" and is decoded by splitting on the first "/", so a "/" in
source_id would make the (source_id, array_id) pair undecodable. register_source
is the single chokepoint every registration path funnels through, so the guard
lives there.
"""

import pytest

from biopb_tensor_server import TensorFlightServer


class _StubAdapter:
    """Minimal stand-in; register_source validates source_id before it ever
    touches the adapter, so no real backend is needed."""

    source_id = "stub"


class TestRegisterSourceValidation:
    @staticmethod
    def _server():
        # Bind to an ephemeral port but never serve(); we only exercise the
        # in-process registration guard.
        return TensorFlightServer("grpc://localhost:0")

    def test_rejects_source_id_with_slash(self):
        server = self._server()
        with pytest.raises(ValueError, match="must not contain '/'"):
            server.register_source("plate/well", _StubAdapter())
        # Nothing should have been stored.
        assert "plate/well" not in server._sources

    def test_rejects_empty_source_id(self):
        server = self._server()
        with pytest.raises(ValueError, match="must be non-empty"):
            server.register_source("", _StubAdapter())

    def test_accepts_slash_free_source_id(self):
        server = self._server()
        adapter = _StubAdapter()
        server.register_source("aics_73f53c400a63", adapter)
        assert server._sources["aics_73f53c400a63"] is adapter

    def test_accepts_hierarchical_looking_but_slash_free_id(self):
        # Underscores / colons are fine -- only "/" is forbidden (it is the
        # chunk-route delimiter).
        server = self._server()
        server.register_source("zarr_a3f2b1c4d5e6", _StubAdapter())
        server.register_source("Image:0_source", _StubAdapter())
        assert "zarr_a3f2b1c4d5e6" in server._sources
        assert "Image:0_source" in server._sources
