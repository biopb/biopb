"""Tests for TensorFlightServer.list_flights() error resilience.

A single source whose descriptor build fails (e.g. an aicsimageio source whose
scene-switching fallback raises) must not abort the whole listing — it should be
skipped while the remaining healthy sources are still returned.
"""

from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor
from biopb_tensor_server.core.metadata_db import MetadataDatabase
from biopb_tensor_server.serving.server import TensorFlightServer


class _HealthyAdapter:
    """Minimal adapter stub whose descriptor build succeeds."""

    capability_token = None

    def __init__(self, source_id):
        self.source_id = source_id

    def get_source_descriptor(self):
        return DataSourceDescriptor(
            source_id=self.source_id,
            source_url=f"file:///{self.source_id}",
            source_type="tiff",
            tensors=[
                TensorDescriptor(
                    array_id=self.source_id,
                    shape=[10, 10],
                    dtype="uint8",
                    chunk_shape=[10, 10],
                )
            ],
        )


class _FailingAdapter:
    """Adapter stub whose descriptor build raises, mimicking aicsimageio."""

    capability_token = None

    def __init__(self, source_id):
        self.source_id = source_id

    def get_source_descriptor(self):
        raise RuntimeError("scene switching failed")


def _command_source_id(flight_info):
    """Decode the source_id out of a FlightInfo's command descriptor."""
    desc = DataSourceDescriptor()
    desc.ParseFromString(flight_info.descriptor.command)
    return desc.source_id


def test_list_flights_skips_failing_source():
    """A source that raises during descriptor build is skipped, not fatal."""
    server = TensorFlightServer(location="grpc://localhost:0")

    server.sources.replace(
        {
            "good-1": _HealthyAdapter("good-1"),
            "bad": _FailingAdapter("bad"),
            "good-2": _HealthyAdapter("good-2"),
        }
    )

    infos = list(server.list_flights(None, b""))

    returned_ids = {_command_source_id(info) for info in infos}
    assert returned_ids == {"good-1", "good-2"}


def test_list_flights_all_healthy():
    """All sources returned when none fail."""
    server = TensorFlightServer(location="grpc://localhost:0")

    server.sources.replace(
        {
            "good-1": _HealthyAdapter("good-1"),
            "good-2": _HealthyAdapter("good-2"),
        }
    )

    infos = list(server.list_flights(None, b""))

    returned_ids = {_command_source_id(info) for info in infos}
    assert returned_ids == {"good-1", "good-2"}


# --- DuckDB-catalog-backed path (biopb/biopb#265) ---------------------------


class _CatalogAdapter:
    """Adapter double that syncs cleanly into the metadata DB."""

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


def test_list_flights_served_from_catalog_not_adapters():
    """With a metadata DB, ListFlights reflects the catalog, not the adapter
    registry: a source only in the DB shows up; a source only in ``sources``
    does not. This is the single-source-of-truth switch."""
    db = MetadataDatabase()
    db.sync_source_added("in-db", _CatalogAdapter("in-db"))

    server = TensorFlightServer(location="grpc://localhost:0", metadata_db=db)
    # Present in the adapter registry but NOT in the catalog -> must be invisible.
    server.sources.replace({"only-adapter": _HealthyAdapter("only-adapter")})

    returned_ids = {_command_source_id(i) for i in server.list_flights(None, b"")}
    assert returned_ids == {"in-db"}


def test_list_flights_catalog_truncation_signaled():
    """The cap and truncation schema metadata carry over to the DuckDB path."""
    db = MetadataDatabase()
    for sid in ("a", "b", "c"):
        db.sync_source_added(sid, _CatalogAdapter(sid))

    server = TensorFlightServer(
        location="grpc://localhost:0", metadata_db=db, max_list_flights_results=2
    )

    infos = list(server.list_flights(None, b""))
    assert len(infos) == 2
    meta = infos[0].schema.metadata
    assert meta[b"total_sources"] == b"3"
    assert meta[b"returned_sources"] == b"2"
    assert meta[b"truncated"] == b"True"
    # Deterministic order: first two source_ids.
    assert [_command_source_id(i) for i in infos] == ["a", "b"]
