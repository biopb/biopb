"""Tests for TensorFlightServer.list_flights() error resilience.

A single source whose descriptor build fails (e.g. an aicsimageio source whose
scene-switching fallback raises) must not abort the whole listing — it should be
skipped while the remaining healthy sources are still returned.
"""

import pyarrow.flight as flight
import pytest
from biopb.tensor.descriptor_pb2 import (
    DataSourceDescriptor,
    TensorCriteria,
    TensorDescriptor,
)
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


def _command_descriptor(flight_info):
    desc = DataSourceDescriptor()
    desc.ParseFromString(flight_info.descriptor.command)
    return desc


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
    # The lean listing carries the transfer grid: it is the adapter's stable
    # per-tensor choice, not a per-request one (biopb/biopb#809).
    assert all(
        list(_command_descriptor(info).tensors[0].chunk_shape) == [10, 10]
        for info in infos
    )


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

    infos = list(server.list_flights(None, b""))
    returned_ids = {_command_source_id(i) for i in infos}
    assert returned_ids == {"in-db"}
    listed = _command_descriptor(infos[0]).tensors[0]
    assert list(listed.shape) == [10, 10]
    # Structural entry only. The adapter double hands the catalog a grid; the
    # catalog does not store one and ListFlights does not publish one -- the
    # transfer grid is GetFlightInfo's to answer, per the tensor it binds
    # (biopb/biopb#812).
    assert list(listed.chunk_shape) == []


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


class _TokenedAdapter(_HealthyAdapter):
    """A source readable only by a caller presenting its capability token.

    These exist only on servers built without a metadata DB (the embedded
    image-base result cache), which is why the adapter path is the one that has
    to keep them hidden.
    """

    capability_token = "secret"


class TestAddressedByCriteria:
    """ListFlights narrowed to one source_id.

    The addressed form of a browse: the catalog is asked a WHERE instead of
    being streamed whole for the caller to search it. What it must NOT do is
    widen what is visible -- naming a source is not authority to see one the
    listing would decline.
    """

    @staticmethod
    def _for(source_id):
        return TensorCriteria(source_id=source_id).SerializeToString()

    def test_criteria_narrows_the_catalog_to_one_row(self):
        db = MetadataDatabase()
        for sid in ("a", "b", "c"):
            db.sync_source_added(sid, _CatalogAdapter(sid))
        server = TensorFlightServer(location="grpc://localhost:0", metadata_db=db)

        infos = list(server.list_flights(None, self._for("b")))

        assert [_command_source_id(i) for i in infos] == ["b"]

    def test_empty_criteria_is_still_the_whole_listing(self):
        # What every client predating the field sends, and what the pyarrow
        # default sends. It must keep meaning "everything".
        db = MetadataDatabase()
        for sid in ("a", "b"):
            db.sync_source_added(sid, _CatalogAdapter(sid))
        server = TensorFlightServer(location="grpc://localhost:0", metadata_db=db)

        assert len(list(server.list_flights(None, b""))) == 2

    def test_an_unknown_id_lists_nothing(self):
        db = MetadataDatabase()
        db.sync_source_added("a", _CatalogAdapter("a"))
        server = TensorFlightServer(location="grpc://localhost:0", metadata_db=db)

        assert list(server.list_flights(None, self._for("nope"))) == []

    def test_a_source_past_the_cap_is_still_addressable(self):
        # The bug this fixes. The cap bounds a browse; it was also silently
        # bounding lookup, so a perfectly readable source answered "not found"
        # purely because of where it sorted.
        db = MetadataDatabase()
        for sid in ("a", "b", "c"):
            db.sync_source_added(sid, _CatalogAdapter(sid))
        server = TensorFlightServer(
            location="grpc://localhost:0", metadata_db=db, max_list_flights_results=2
        )

        browsed = {_command_source_id(i) for i in server.list_flights(None, b"")}
        assert "c" not in browsed

        addressed = list(server.list_flights(None, self._for("c")))
        assert [_command_source_id(i) for i in addressed] == ["c"]

    def test_a_filtered_answer_is_never_reported_as_truncated(self):
        # `total` counts what matched, not the catalog, so one row out of many
        # is a complete answer rather than a clipped listing.
        db = MetadataDatabase()
        for sid in ("a", "b", "c"):
            db.sync_source_added(sid, _CatalogAdapter(sid))
        server = TensorFlightServer(
            location="grpc://localhost:0", metadata_db=db, max_list_flights_results=2
        )

        info = next(iter(server.list_flights(None, self._for("a"))))

        assert info.schema.metadata[b"truncated"] == b"False"
        assert info.schema.metadata[b"total_sources"] == b"1"

    def test_criteria_narrows_the_adapter_path_too(self):
        server = TensorFlightServer(location="grpc://localhost:0")
        server.sources.replace(
            {"one": _HealthyAdapter("one"), "two": _HealthyAdapter("two")}
        )

        infos = list(server.list_flights(None, self._for("two")))

        assert [_command_source_id(i) for i in infos] == ["two"]

    def test_naming_a_tokened_source_does_not_reveal_it(self):
        # The whole point of excluding them from enumeration is that knowing the
        # id is not enough. A filter that skipped the token check would hand
        # that back.
        server = TensorFlightServer(location="grpc://localhost:0")
        server.sources.replace({"secret-src": _TokenedAdapter("secret-src")})

        assert list(server.list_flights(None, self._for("secret-src"))) == []

    def test_unparseable_criteria_is_refused_not_ignored(self):
        # Dropping a filter it could not read would answer "give me one source"
        # with the entire catalog, and look like it worked.
        db = MetadataDatabase()
        db.sync_source_added("a", _CatalogAdapter("a"))
        server = TensorFlightServer(location="grpc://localhost:0", metadata_db=db)

        with pytest.raises(flight.FlightServerError, match="TensorCriteria"):
            list(server.list_flights(None, b"\xff\xff\xff\xff"))

    def test_an_id_with_sql_in_it_is_data_not_syntax(self):
        # source_ids are partly caller-supplied (a drag-dropped path becomes
        # one), so the filter is bound as a parameter rather than interpolated.
        db = MetadataDatabase()
        db.sync_source_added("a", _CatalogAdapter("a"))
        server = TensorFlightServer(location="grpc://localhost:0", metadata_db=db)

        assert list(server.list_flights(None, self._for("' OR 1=1 --"))) == []
        assert len(list(server.list_flights(None, b""))) == 1
