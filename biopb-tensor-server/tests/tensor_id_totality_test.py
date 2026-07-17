"""Issue #378: total field resolution + a correct Flight error taxonomy.

Two coupled defects on the read path (``get_flight_info`` ->
``SourceAdapter.get_tensor_adapter``):

1. ``get_tensor_adapter`` was not total -- an unknown nonempty field either
   silently returned the base (wrong) tensor (single-tensor sources) or raised a
   bare ``ValueError`` (HCS / scene sources).
2. That miss was mapped to the wrong Flight status -- a bare ``ValueError`` ->
   ``FlightInternalError`` (INTERNAL = "server bug, don't retry") for what is a
   client error (NOT_FOUND / INVALID_ARGUMENT).

These tests pin the fix: every adapter kind rejects an unknown nonempty field
with a typed domain error, the ``#44`` no-field defaults still resolve, and the
Flight boundary maps the taxonomy to a *terminal* (not INTERNAL) Flight error
carrying the canonical code + machine reason in ``extra_info``.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
import pytest
from biopb.tensor.descriptor_pb2 import FlightCmd, TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server import TensorFlightServer
from biopb_tensor_server.core.base import BackendAdapter
from biopb_tensor_server.core.errors import (
    InvalidTensorId,
    SourceResolveRetriableError,
    SourceUnresolvedError,
    TensorNotFound,
    TensorResolutionError,
)
from biopb_tensor_server.serving.server import to_flight_error


# --------------------------------------------------------------------------- #
# Minimal adapters
# --------------------------------------------------------------------------- #
class _SingleTensorAdapter(BackendAdapter):
    """A single-tensor source whose sole tensor is addressed by ``source_id``."""

    @classmethod
    def claim(cls, path, visited_identities):  # pragma: no cover - not discovered
        return None

    @classmethod
    def create_from_config(cls, source, credentials_config=None):  # pragma: no cover
        raise NotImplementedError

    def __init__(self, source_id: str, shape=(4, 4)):
        self.source_id = source_id
        self._shape = shape
        self._source_url = f"mock://{source_id}"
        self._source_type = "mock-single"

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=["y", "x"],
            shape=list(self._shape),
            chunk_shape=list(self._shape),
            dtype="uint8",
        )

    def list_tensor_descriptors(self):
        return [self.get_tensor_descriptor()]

    def get_metadata(self) -> dict:
        return {}

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        super().get_data(bounds)
        shape = tuple(int(s - a) for a, s in zip(bounds.start, bounds.stop))
        return np.zeros(shape, dtype="uint8")


class _NamedTensorAdapter(_SingleTensorAdapter):
    """Single-tensor source whose lone tensor carries a name (aicsimageio's
    ``Image:0``), so ``source_id/Image:0`` is a *valid* read of the default."""

    def __init__(self, source_id: str, tensor_name: str = "Image:0", shape=(4, 4)):
        super().__init__(source_id, shape)
        self._tensor_name = tensor_name


# --------------------------------------------------------------------------- #
# 1. Domain taxonomy -> Flight boundary mapping
# --------------------------------------------------------------------------- #
class TestBoundaryMapping:
    def test_not_found_is_terminal_not_internal_with_code(self):
        err = to_flight_error(TensorNotFound("nope", reason="unknown_field"))
        # Terminal FlightServerError, NOT the "server bug" FlightInternalError.
        assert isinstance(err, flight.FlightServerError)
        assert not isinstance(err, flight.FlightInternalError)
        assert json.loads(err.extra_info) == {
            "code": "NOT_FOUND",
            "reason": "unknown_field",
        }

    def test_invalid_tensor_id_maps_to_invalid_argument(self):
        err = to_flight_error(InvalidTensorId("bad", reason="malformed_tensor_id"))
        assert isinstance(err, flight.FlightServerError)
        assert not isinstance(err, flight.FlightInternalError)
        assert json.loads(err.extra_info) == {
            "code": "INVALID_ARGUMENT",
            "reason": "malformed_tensor_id",
        }

    def test_unresolved_maps_to_unavailable_retriable(self):
        # A retriable/unresolved source is UNAVAILABLE (retry), never terminal.
        assert isinstance(
            to_flight_error(SourceUnresolvedError("x")), flight.FlightUnavailableError
        )
        assert isinstance(
            to_flight_error(SourceResolveRetriableError("y")),
            flight.FlightUnavailableError,
        )

    def test_taxonomy_subclasses_valueerror_for_graceful_degradation(self):
        # Like SourceUnresolvedError: existing ``except ValueError`` guards catch it.
        assert issubclass(TensorResolutionError, ValueError)
        assert issubclass(TensorNotFound, TensorResolutionError)
        assert issubclass(InvalidTensorId, TensorResolutionError)


# --------------------------------------------------------------------------- #
# 2. Single-tensor base totality (the silent-false-positive case)
# --------------------------------------------------------------------------- #
class TestSingleTensorTotality:
    def test_default_forms_resolve_to_the_sole_tensor(self):
        a = _SingleTensorAdapter("src")
        for tid in (None, "", "src"):
            assert a.get_tensor_adapter(tid) is a

    def test_unknown_nonempty_field_raises_not_found(self):
        a = _SingleTensorAdapter("src")
        with pytest.raises(TensorNotFound) as ei:
            a.get_tensor_adapter("src/@nonexistent")
        assert ei.value.grpc_code == "NOT_FOUND"
        assert ei.value.reason == "unknown_field"
        # A bare unknown field faults too (not just the source-qualified form).
        with pytest.raises(TensorNotFound):
            a.get_tensor_adapter("@nonexistent")

    def test_named_sole_tensor_is_addressable(self):
        a = _NamedTensorAdapter("src", "Image:0")
        # #44 defaults + the tensor's own name all resolve to the one tensor.
        for tid in (None, "", "src", "Image:0", "src/Image:0"):
            assert a.get_tensor_adapter(tid) is a
        with pytest.raises(TensorNotFound):
            a.get_tensor_adapter("src/Image:9")


# --------------------------------------------------------------------------- #
# 3. Multi-scene (OME-TIFF / bioio share ``_scene_index_for_field``)
# --------------------------------------------------------------------------- #
class TestSceneTotality:
    def test_ome_tiff_unknown_scene_raises_not_found(self, tmp_path):
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter
        from biopb_tensor_server.fixtures import create_multi_series_ome_tiff

        path, _, _ = create_multi_series_ome_tiff(str(tmp_path), n_series=3)
        adapter = OmeTiffAdapter(path, "idx")
        adapter.list_tensor_descriptors()

        with pytest.raises(TensorNotFound) as ei:
            adapter.get_tensor_adapter("idx/Image:999")
        assert ei.value.grpc_code == "NOT_FOUND"
        assert ei.value.reason == "unknown_field"
        # Still catchable as ValueError (backward-compatible guards / old tests).
        with pytest.raises(ValueError):
            adapter._scene_index_for_field("Image:999")


# --------------------------------------------------------------------------- #
# 4. HCS OME-Zarr (well / field parsing) -- no real store needed
# --------------------------------------------------------------------------- #
class TestHcsTotality:
    def _hcs_stub(self):
        from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter

        obj = OmeZarrAdapter.__new__(OmeZarrAdapter)
        obj.source_id = "plate"
        obj._is_hcs_plate = True
        obj._hcs_well_paths = {"A01": "A/1"}
        obj._hcs_well_metadata = {"A01": {"well": {"images": [{"path": "0"}]}}}
        return obj

    def test_unknown_well_raises_not_found(self):
        obj = self._hcs_stub()
        with pytest.raises(TensorNotFound) as ei:
            obj.get_tensor_adapter("plate/Z99/0")
        assert ei.value.grpc_code == "NOT_FOUND"
        assert ei.value.reason == "unknown_field"

    def test_field_index_out_of_range_raises_not_found(self):
        obj = self._hcs_stub()
        with pytest.raises(TensorNotFound) as ei:
            obj.get_tensor_adapter("plate/A01/7")
        assert ei.value.grpc_code == "NOT_FOUND"

    def test_malformed_tensor_id_raises_invalid_argument(self):
        obj = self._hcs_stub()
        with pytest.raises(InvalidTensorId) as ei:
            obj.get_tensor_adapter("plate/A01")  # not 'well/field'
        assert ei.value.grpc_code == "INVALID_ARGUMENT"
        assert ei.value.reason == "malformed_tensor_id"

    def test_non_integer_field_index_raises_invalid_argument(self):
        obj = self._hcs_stub()
        with pytest.raises(InvalidTensorId) as ei:
            obj.get_tensor_adapter("plate/A01/x")
        assert ei.value.grpc_code == "INVALID_ARGUMENT"


# --------------------------------------------------------------------------- #
# 5. End-to-end at the Flight verb: get_flight_info maps the miss correctly
# --------------------------------------------------------------------------- #
def _flight_info_for(server, source_id, tensor_id):
    cmd = FlightCmd(source_id=source_id)
    cmd.tensor_read.tensor_id = tensor_id
    descriptor = flight.FlightDescriptor.for_command(cmd.SerializeToString())
    # Sources carry no capability token here, so _authorize_source never touches
    # the (None) context -- call the verb directly, no socket needed.
    return server.get_flight_info(None, descriptor)


class TestGetFlightInfoBoundary:
    def test_unknown_field_is_terminal_not_internal_with_code(self):
        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("src", _SingleTensorAdapter("src"))

        with pytest.raises(flight.FlightServerError) as ei:
            _flight_info_for(server, "src", "src/@nonexistent")
        # The core regression: a typo'd array_id must NOT read as a server bug.
        assert not isinstance(ei.value, flight.FlightInternalError)
        assert json.loads(ei.value.extra_info)["code"] == "NOT_FOUND"

    def test_unknown_source_is_terminal_not_found(self):
        server = TensorFlightServer("grpc://localhost:0")
        with pytest.raises(flight.FlightServerError) as ei:
            _flight_info_for(server, "ghost", "ghost/x")
        assert not isinstance(ei.value, flight.FlightInternalError)
        assert json.loads(ei.value.extra_info)["code"] == "NOT_FOUND"

    def test_valid_default_still_resolves(self):
        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("src", _SingleTensorAdapter("src"))
        info = _flight_info_for(server, "src", "")  # bare/default -> sole tensor
        assert isinstance(info, flight.FlightInfo)
