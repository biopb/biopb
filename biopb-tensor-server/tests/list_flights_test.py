"""ListFlights advertises the public catalog: one flight per queryable table.

A stock Flight client can list, read each table's schema, and DoGet the
whole table from the advertised ticket without any biopb proto. Sources are
not flights -- browsing them is a catalog query, reading them is addressed.
"""

import pyarrow.flight as flight
import pytest
from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor
from biopb.tensor.ticket_pb2 import TensorTicket
from biopb_tensor_server.core.metadata_db import MetadataDatabase
from biopb_tensor_server.serving.server import TensorFlightServer


class _CatalogAdapter:
    """Adapter double that syncs cleanly into the metadata DB."""

    capability_token = None

    def __init__(self, source_id):
        self.source_id = source_id

    def get_source_descriptor(self):
        return DataSourceDescriptor(
            source_id=self.source_id,
            source_url=f"file:///{self.source_id}",
            source_type="zarr",
            data_resident=True,
            tensors=[
                TensorDescriptor(
                    array_id=self.source_id,
                    shape=[10, 10],
                    chunk_shape=[10, 10],
                    dtype="uint8",
                )
            ],
        )

    def get_metadata(self):
        return {}


class _Context:
    """A ServerCallContext double carrying one presented bearer."""

    def __init__(self, token=None):
        self._token = token

    def get_middleware(self, name):
        from biopb_tensor_server.serving.server import _AuthMiddleware

        return _AuthMiddleware(self._token)


def _server(db, token=None):
    return TensorFlightServer(
        location="grpc://localhost:0", metadata_db=db, token=token
    )


def test_one_flight_per_public_table_with_its_schema():
    db = MetadataDatabase()
    db.sync_source_added("a", _CatalogAdapter("a"))
    server = _server(db)

    infos = list(server.list_flights(_Context(), b""))
    by_path = {"/".join(p.decode() for p in i.descriptor.path): i for i in infos}
    assert set(by_path) == set(db.allowed_tables) == {"sources", "decode_rates"}
    assert "rois" not in by_path  # private data is not a catalog flight

    sources = by_path["sources"]
    assert sources.schema.names[:3] == ["source_id", "source_url", "source_type"]
    assert "tensors" in sources.schema.names


def test_the_advertised_ticket_reads_the_whole_table():
    db = MetadataDatabase()
    for sid in ("b", "a"):
        db.sync_source_added(sid, _CatalogAdapter(sid))
    server = _server(db)

    (sources,) = [
        i
        for i in server.list_flights(_Context(), b"")
        if i.descriptor.path == [b"sources"]
    ]
    ticket = TensorTicket.FromString(sources.endpoints[0].ticket.ticket)
    assert ticket.WhichOneof("payload") == "catalog_query"
    assert ticket.catalog_query.sql == "SELECT * FROM sources"

    stream = server.do_get(_Context(), sources.endpoints[0].ticket)
    table = stream.to_reader().read_all() if hasattr(stream, "to_reader") else None
    if table is None:
        pytest.skip("RecordBatchStream is not readable in-process on this pyarrow")
    assert sorted(table.column("source_id").to_pylist()) == ["a", "b"]


def test_a_path_descriptor_names_a_table():
    db = MetadataDatabase()
    db.sync_source_added("a", _CatalogAdapter("a"))
    server = _server(db)

    info = server.get_flight_info(
        _Context(), flight.FlightDescriptor.for_path("sources")
    )
    assert info.schema.names[:1] == ["source_id"]
    with pytest.raises(flight.FlightServerError, match="disallowed table"):
        server.get_flight_info(_Context(), flight.FlightDescriptor.for_path("rois"))


def test_a_bare_server_advertises_its_own_catalog():
    """A server built without a catalog makes one and keeps it in step with
    its registry, so direct registration is browsable too."""
    server = TensorFlightServer(location="grpc://localhost:0")
    server.register_source("x", _CatalogAdapter("x"))
    paths = {tuple(i.descriptor.path) for i in server.list_flights(_Context(), b"")}
    assert (b"sources",) in paths
    rows = server._metadata_db.query("SELECT source_id FROM sources").to_pylist()
    assert [r["source_id"] for r in rows] == ["x"]
    server.unregister_source("x")
    assert server._metadata_db.query("SELECT source_id FROM sources").num_rows == 0


def test_the_catalog_tier_is_the_server_token():
    db = MetadataDatabase()
    db.sync_source_added("a", _CatalogAdapter("a"))
    server = _server(db, token="secret")

    with pytest.raises(flight.FlightUnauthenticatedError):
        list(server.list_flights(_Context(), b""))
    with pytest.raises(flight.FlightUnauthenticatedError):
        list(server.list_flights(_Context("wrong"), b""))
    assert len(list(server.list_flights(_Context("secret"), b""))) == 2
