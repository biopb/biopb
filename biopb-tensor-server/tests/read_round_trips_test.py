"""What one read costs in round trips.

The SDK caches no descriptors, so the thing that keeps a tile burst cheap is
that planning a read does not *need* one: ``GetFlightInfo`` both plans the read
and reports the tensor it bound, and the addressing refusals are taken from that
answer rather than from a catalog lookup done beforehand.

These count the calls because nothing else will notice if that stops being true.
Re-introducing a pre-resolve would not fail a correctness test -- every other
test here would still pass, and the tile path would quietly pay a catalog query
per tile. That is the regression this file exists for.
"""

import threading

import pytest
import zarr
from biopb.tensor.client import TensorFlightClient
from biopb_tensor_server.adapters.zarr import ZarrAdapter

from tests import catalog_server, register_and_catalog


@pytest.fixture
def tile_server(tmp_path):
    path = tmp_path / "t.zarr"
    arr = zarr.open_array(
        str(path), mode="w", shape=(64, 64), chunks=(16, 16), dtype="uint8"
    )
    arr[:] = 5
    server = catalog_server("grpc://localhost:0")
    register_and_catalog(
        server,
        "solo",
        ZarrAdapter(zarr.open_array(str(path), mode="r"), "solo", ["y", "x"]),
    )
    server.mark_ready()
    threading.Thread(target=server.serve, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()


class _CountingFlight:
    """Wraps a FlightClient, counting the two calls a read can make.

    ``get_flight_info`` plans the read; ``do_get`` is how the catalog is
    queried, so a non-zero count is a catalog lookup on the read path.
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


def _counted(client):
    # Touch the connection first: the property runs a one-time protocol health
    # check, which is a round trip and not one of the ones under test.
    assert client._state.client is not None
    counter = _CountingFlight(client._state.raw_client)
    client._state.raw_client = counter
    return counter


class TestReadRoundTrips:
    def test_a_tile_burst_costs_one_call_per_tile(self, tile_server):
        """Bounded start/stop and a qualified-or-single-tensor id: the shape the
        HTTP tile route issues, and the one that must stay at one RPC.

        With the descriptor cache this cost an extra catalog query on the first
        tile -- the cache then hid it from the rest of the burst. Not planning
        against a descriptor at all removes the query instead of memoizing it.
        """
        client = TensorFlightClient(f"grpc://localhost:{tile_server.port}")
        try:
            counter = _counted(client)
            for i in range(8):
                client.get_tensor(
                    "solo", slice_hint=(slice(0, 16), slice(i * 8, i * 8 + 8))
                )

            assert counter.plans == 8
            assert counter.catalog_queries == 0
        finally:
            client.close()

    def test_an_open_ended_stop_is_the_one_shape_that_asks(self, tile_server):
        """``slice(None)`` has no stop, and only the tensor knows where it ends.

        This is the single remaining pre-RPC resolve, and it is charged to the
        request that needs it rather than to every read.
        """
        client = TensorFlightClient(f"grpc://localhost:{tile_server.port}")
        try:
            counter = _counted(client)
            client.get_tensor("solo", slice_hint=(slice(0, 16), slice(None)))

            assert (counter.plans, counter.catalog_queries) == (1, 1)
        finally:
            client.close()

    def test_a_full_read_asks_nothing_extra(self, tile_server):
        """No slice_hint at all: nothing to fill, so nothing to look up."""
        client = TensorFlightClient(f"grpc://localhost:{tile_server.port}")
        try:
            counter = _counted(client)
            assert client.get_tensor("solo").shape == (64, 64)

            assert (counter.plans, counter.catalog_queries) == (1, 0)
        finally:
            client.close()
