"""Tests for TensorFlightServer.list_flights() error resilience.

A single source whose descriptor build fails (e.g. an aicsimageio source whose
scene-switching fallback raises) must not abort the whole listing — it should be
skipped while the remaining healthy sources are still returned.
"""

from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor
from biopb_tensor_server.server import TensorFlightServer


class _HealthyAdapter:
    """Minimal adapter stub whose descriptor build succeeds."""

    token = None

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

    token = None

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

    server._sources = {
        "good-1": _HealthyAdapter("good-1"),
        "bad": _FailingAdapter("bad"),
        "good-2": _HealthyAdapter("good-2"),
    }

    infos = list(server.list_flights(None, b""))

    returned_ids = {_command_source_id(info) for info in infos}
    assert returned_ids == {"good-1", "good-2"}


def test_list_flights_all_healthy():
    """All sources returned when none fail."""
    server = TensorFlightServer(location="grpc://localhost:0")

    server._sources = {
        "good-1": _HealthyAdapter("good-1"),
        "good-2": _HealthyAdapter("good-2"),
    }

    infos = list(server.list_flights(None, b""))

    returned_ids = {_command_source_id(info) for info in infos}
    assert returned_ids == {"good-1", "good-2"}
