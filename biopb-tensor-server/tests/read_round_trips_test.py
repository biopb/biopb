"""What one read costs in round trips.

Counted rather than asserted indirectly: a pre-resolve added back to the read
path would fail no correctness test while the tile path quietly paid a catalog
query per tile.

The id shape matters as much as the slice shape. A zarr source's ``array_id``
*is* its ``source_id``, so a bare id echoes back unchanged; scene-based adapters
(nd2, lif, czi, bioio, ome_zarr, ome_tiff) qualify their ids even for a
single-scene file, so there a bare id comes back changed and the #75 check has
to ask the catalog which case it was.
"""

import threading

import pytest
import zarr
from biopb.tensor.client import TensorFlightClient
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb_tensor_server.adapters.zarr import ZarrAdapter

from tests import catalog_server, register_and_catalog
from tests.multifield_test import MockMultifieldAdapter


class _QualifiedIdAdapter(MockMultifieldAdapter):
    """A scene adapter that obeys the identity policy.

    ``array_id`` is the full ``source_id[/field]`` form everywhere
    (``proto/biopb/tensor/descriptor.proto``), as the real adapters emit it.
    ``MockMultifieldAdapter`` publishes the bare field name, which makes a
    qualified read miss its own catalog row.
    """

    def list_tensor_descriptors(self):
        return [
            TensorDescriptor(
                array_id=f"{self.source_id}/{tensor_id}",
                shape=list(shape),
                chunk_shape=list(shape),
                dtype=dtype,
            )
            for tensor_id, shape, dtype in self.tensor_specs
        ]


@pytest.fixture
def counted_server(tmp_path):
    """Three sources covering the id shapes a read can address.

    Its own server rather than conftest's ``writable_server``, which registers
    no catalog rows and only one source.
    """
    path = tmp_path / "t.zarr"
    arr = zarr.open_array(
        str(path), mode="w", shape=(64, 64), chunks=(16, 16), dtype="uint8"
    )
    arr[:] = 5

    server = catalog_server("grpc://localhost:0")
    # array_id == source_id: a bare id echoes back unchanged.
    register_and_catalog(
        server,
        "flat",
        ZarrAdapter(zarr.open_array(str(path), mode="r"), "flat", ["y", "x"]),
    )
    # One scene, but a qualified array_id -- the single-scene nd2/czi/lif shape.
    register_and_catalog(
        server,
        "scened",
        _QualifiedIdAdapter("scened", [("Position:0", (64, 64), "uint8")]),
    )
    # Several scenes: the #75 refusal.
    register_and_catalog(
        server,
        "plate",
        _QualifiedIdAdapter(
            "plate", [(f"Position:{i}", (64, 64), "uint8") for i in range(4)]
        ),
    )
    server.mark_ready()
    threading.Thread(target=server.serve, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()


class _CountingFlight:
    """Wraps a FlightClient, counting the two calls a read can make.

    ``get_flight_info`` plans the read; ``do_get`` queries the catalog. Chunk
    fetches do not show up here -- they go through the thread-local pool client.
    """

    def __init__(self, inner):
        self._inner = inner
        self.plans = 0
        self.catalog_queries = 0

    def get_flight_info(self, *args, **kwargs):
        self.plans += 1
        return self._inner.get_flight_info(*args, **kwargs)

    def do_get(self, *args, **kwargs):
        self.catalog_queries += 1
        return self._inner.do_get(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture
def counted(counted_server):
    """A client whose round trips are counted; yields ``(client, counter)``."""
    client = TensorFlightClient(f"grpc://localhost:{counted_server.port}")
    # Touch the connection first: the property runs a one-time protocol health
    # check, which is a round trip and not one of the ones under test.
    assert client._state.client is not None
    counter = _CountingFlight(client._state.raw_client)
    client._state.raw_client = counter
    try:
        yield client, counter
    finally:
        client.close()


BOUNDED = (slice(0, 16), slice(0, 16))
OPEN_ENDED = (slice(0, 16), slice(None))


class TestReadRoundTrips:
    def test_a_tile_burst_costs_one_call_per_tile(self, counted):
        """Qualified id, bounded start/stop: the shape the HTTP tile route
        issues, and the one that must stay at one RPC."""
        client, counter = counted
        for i in range(8):
            client.get_tensor(
                "scened/Position:0", slice_hint=(slice(0, 16), slice(i * 8, i * 8 + 8))
            )

        assert counter.plans == 8
        assert counter.catalog_queries == 0

    def test_a_bare_id_that_echoes_back_unchanged_asks_nothing(self, counted):
        """array_id == source_id, so the server took no default to check."""
        client, counter = counted
        for _ in range(8):
            client.get_tensor("flat", slice_hint=BOUNDED)

        assert (counter.plans, counter.catalog_queries) == (8, 0)

    def test_a_bare_id_the_server_qualified_asks_once_per_read(self, counted):
        """The cost this file exists to keep visible.

        A bare read of a scene-based source comes back changed, and only the
        catalog can say whether that was a one-tensor default or a silent pick
        among many (#75). Nothing amortises it, so it is once per read -- cheap
        (a ``len(tensors)`` count, not the struct column) but not free. A caller
        reading one tensor repeatedly should pass the qualified array_id.
        """
        client, counter = counted
        for _ in range(8):
            client.get_tensor("scened", slice_hint=BOUNDED)

        assert counter.plans == 8
        assert counter.catalog_queries == 8

    def test_an_open_ended_stop_resolves_once_not_twice(self, counted):
        """``slice(None)`` has no stop, and only the tensor knows where it ends.

        The resolve that fills it reads the catalog row, which settles the #75
        question too, so the post-response check must not re-ask.
        """
        client, counter = counted
        for _ in range(8):
            client.get_tensor("scened", slice_hint=OPEN_ENDED)

        assert (counter.plans, counter.catalog_queries) == (8, 8)

    def test_a_full_read_asks_nothing_extra(self, counted):
        """No slice_hint at all: nothing to fill, and a qualified id to route."""
        client, counter = counted
        assert client.get_tensor("scened/Position:0").shape == (64, 64)

        assert (counter.plans, counter.catalog_queries) == (1, 0)

    def test_the_ambiguous_bare_id_is_still_refused(self, counted):
        """The refusal the post-response check exists for, so a change that
        makes the counts above cheaper by dropping it fails here."""
        client, _ = counted
        with pytest.raises(ValueError, match="multiple tensors"):
            client.get_tensor("plate", slice_hint=BOUNDED)
