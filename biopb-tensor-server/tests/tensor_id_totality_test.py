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
import pyarrow.flight as flight
import pytest
from biopb.tensor.descriptor_pb2 import FlightCmd, TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server import TensorFlightServer
from biopb_tensor_server.core.base import TensorAdapter
from biopb_tensor_server.core.errors import (
    InvalidTensorId,
    SourceResolveRetriableError,
    SourceUnresolvedError,
    TensorNotFound,
    TensorResolutionError,
    UnknownResolutionError,
)
from biopb_tensor_server.serving.server import (
    _adapter_lookup_error,
    to_flight_error,
)


# --------------------------------------------------------------------------- #
# Minimal adapters
# --------------------------------------------------------------------------- #
class _SingleTensorAdapter(TensorAdapter):
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
        shape = tuple(
            int(s - a) for a, s in zip(bounds.start, bounds.stop, strict=True)
        )
        return np.zeros(shape, dtype="uint8")


class _NamedTensorAdapter(_SingleTensorAdapter):
    """Single-tensor source whose lone tensor carries a name (aicsimageio's
    ``Image:0``), so ``source_id/Image:0`` is a *valid* read of the default."""

    def __init__(self, source_id: str, tensor_name: str = "Image:0", shape=(4, 4)):
        super().__init__(source_id, shape)
        self._tensor_name = tensor_name


class _LegacyMissAdapter(_SingleTensorAdapter):
    """An adapter that predates the typed taxonomy: raises a *bare* ``ValueError``
    for an unknown field (as several not-yet-converted adapters still do). The read
    verbs' fallback must coerce this to NOT_FOUND, not leak it as INTERNAL."""

    def get_tensor_adapter(self, tensor_id):
        field = self._within_source_field(tensor_id)
        if field and field != self.source_id:
            raise ValueError(f"legacy: unknown field {field!r}")
        return self


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
        # Both unresolved variants -> retriable FlightUnavailableError, and the
        # extra_info code AGREES with that class (UNAVAILABLE) -- no class-says-
        # retry / code-says-terminal mismatch. The message carries "unresolved"
        # so the Python client's resolve-steering (substring match) still fires.
        for exc in (SourceUnresolvedError("x"), SourceResolveRetriableError("y")):
            err = to_flight_error(exc)
            assert isinstance(err, flight.FlightUnavailableError)
            assert json.loads(err.extra_info)["code"] == "UNAVAILABLE"
            assert "unresolved" in str(err).lower()

    def test_taxonomy_subclasses_valueerror_for_graceful_degradation(self):
        # Like SourceUnresolvedError: existing ``except ValueError`` guards catch it.
        assert issubclass(TensorResolutionError, ValueError)
        assert issubclass(TensorNotFound, TensorResolutionError)
        assert issubclass(InvalidTensorId, TensorResolutionError)


# --------------------------------------------------------------------------- #
# 1b. Shared adapter-lookup fallback (get_flight_info / do_get / chunk_locate)
# --------------------------------------------------------------------------- #
class TestAdapterLookupFallback:
    """Every read verb maps an adapter-lookup miss through _adapter_lookup_error,
    so the same exception surfaces identically no matter which verb hit it -- and an
    unclassified bare exception from an adapter that predates the typed taxonomy is
    coerced to a terminal UNKNOWN (not a fabricated NOT_FOUND that misblames the
    caller, and never a "server bug" FlightInternalError)."""

    def test_bare_valueerror_coerced_to_unknown(self):
        err = _adapter_lookup_error(ValueError("legacy adapter boom"), "ctx")
        assert isinstance(err, flight.FlightServerError)
        assert not isinstance(err, flight.FlightInternalError)
        assert json.loads(err.extra_info) == {
            "code": "UNKNOWN",
            "reason": "unclassified",
        }

    def test_bare_keyerror_coerced_to_unknown(self):
        err = _adapter_lookup_error(KeyError("missing"), "ctx")
        assert isinstance(err, flight.FlightServerError)
        assert not isinstance(err, flight.FlightInternalError)
        assert json.loads(err.extra_info)["code"] == "UNKNOWN"

    def test_bare_attributeerror_and_typeerror_coerced_to_unknown(self):
        # A None.split-style crash (AttributeError) or an int(None) (TypeError) in
        # a not-yet-converted multi-tensor resolver is unclassified -- not a client
        # "field not found" -- so it maps to a terminal UNKNOWN, not INTERNAL.
        for exc in (AttributeError("'NoneType' has no attr split"), TypeError("x")):
            err = _adapter_lookup_error(exc, "ctx")
            assert isinstance(err, flight.FlightServerError)
            assert not isinstance(err, flight.FlightInternalError)
            assert json.loads(err.extra_info)["code"] == "UNKNOWN"

    def test_unknown_resolution_error_subclasses_the_taxonomy(self):
        # Rides the same ValueError-based taxonomy, so existing except-guards catch
        # it and to_flight_error picks its terminal FlightServerError class.
        assert issubclass(UnknownResolutionError, TensorResolutionError)
        assert UnknownResolutionError("x", reason="unclassified").grpc_code == "UNKNOWN"

    def test_typed_taxonomy_passes_through_with_its_code(self):
        assert (
            json.loads(
                _adapter_lookup_error(
                    InvalidTensorId("bad", reason="malformed_tensor_id"), "ctx"
                ).extra_info
            )["code"]
            == "INVALID_ARGUMENT"
        )
        assert (
            json.loads(
                _adapter_lookup_error(
                    TensorNotFound("nope", reason="unknown_field"), "ctx"
                ).extra_info
            )["code"]
            == "NOT_FOUND"
        )

    def test_unresolved_stays_retriable(self):
        err = _adapter_lookup_error(SourceUnresolvedError("not hydrated"), "ctx")
        assert isinstance(err, flight.FlightUnavailableError)
        assert "unresolved" in str(err).lower()

    def test_do_get_bare_valueerror_is_terminal_not_internal(self):
        # End-to-end at the verb: a stale ticket naming an unknown field on a
        # legacy adapter (bare ValueError) must NOT read as a server bug -- do_get
        # coerces it the same way get_flight_info does (issue #378).
        from biopb.tensor.ticket_pb2 import TensorTicket
        from biopb_tensor_server.core.base import encode_chunk_id

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("legacy", _LegacyMissAdapter("legacy"))
        chunk_id = encode_chunk_id(
            "legacy/@bad", ChunkBounds(start=[0, 0], stop=[4, 4])
        )
        ticket = flight.Ticket(TensorTicket(chunk_id=chunk_id).SerializeToString())
        with pytest.raises(flight.FlightServerError) as ei:
            server.do_get(None, ticket)
        assert not isinstance(ei.value, flight.FlightInternalError)
        # A bare ValueError from a legacy adapter is unclassified -> terminal UNKNOWN.
        assert json.loads(ei.value.extra_info)["code"] == "UNKNOWN"

    def test_do_get_unregistered_source_ticket_is_terminal_with_code(self):
        # Finding #3: a stale ticket whose source is no longer registered ->
        # _get_adapter_for_chunk returns None -> the adapter-is-None fallthrough now
        # rides the taxonomy (terminal NOT_FOUND *with a code*), matching
        # get_flight_info's tensor_adapter-is-None sibling, rather than a bare
        # FlightServerError carrying no extra_info code (issue #378).
        from biopb.tensor.ticket_pb2 import TensorTicket
        from biopb_tensor_server.core.base import encode_chunk_id

        server = TensorFlightServer("grpc://localhost:0")
        chunk_id = encode_chunk_id("ghost", ChunkBounds(start=[0, 0], stop=[4, 4]))
        ticket = flight.Ticket(TensorTicket(chunk_id=chunk_id).SerializeToString())
        with pytest.raises(flight.FlightServerError) as ei:
            server.do_get(None, ticket)
        assert not isinstance(ei.value, flight.FlightInternalError)
        assert json.loads(ei.value.extra_info) == {
            "code": "NOT_FOUND",
            "reason": "unknown_source",
        }

    def test_chunk_locate_unregistered_source_is_terminal_with_code(self):
        # Same fallthrough on the cache-file locate path (finding #3).
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.core.base import encode_chunk_id
        from biopb_tensor_server.core.config import CacheConfig

        # _handle_chunk_locate short-circuits to {"available": False} when no cache
        # manager exists, so give it one to reach the adapter-is-None fallthrough.
        CacheManager.initialize(CacheConfig(backend="memory"))
        try:
            server = TensorFlightServer("grpc://localhost:0")
            chunk_id = encode_chunk_id("ghost", ChunkBounds(start=[0, 0], stop=[4, 4]))
            with pytest.raises(flight.FlightServerError) as ei:
                server._handle_chunk_locate(chunk_id)
            assert not isinstance(ei.value, flight.FlightInternalError)
            assert json.loads(ei.value.extra_info)["code"] == "NOT_FOUND"
        finally:
            CacheManager.reset()


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
        # __init__ is bypassed here, so mirror the per-instance caches it sets.
        obj._field_adapters = {}
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
# 4b. EMD (signal index parsing) -- a real synthetic store
# --------------------------------------------------------------------------- #
class TestEmdTotality:
    """The EMD field is the signal's integer index (``source_id/0``, ...): a
    non-integer field is structurally malformed (``InvalidTensorId``), an integer
    outside the signal range names no signal (``TensorNotFound``)."""

    def _emd_adapter(self, tmp_path):
        # EMD reads via rosettasciio (the [em] extra) over h5py; skip this test
        # when either is absent rather than failing at use.
        pytest.importorskip("rsciio")
        h5py = pytest.importorskip("h5py")
        from biopb_tensor_server.adapters.emd import EmdAdapter
        from biopb_tensor_server.core.config import SourceConfig

        # One datacube -> one signal, so index 0 is the only valid field.
        path = tmp_path / "test.emd"
        data = np.arange(2 * 3 * 8 * 8, dtype=np.uint16).reshape(2, 3, 8, 8)
        with h5py.File(path, "w") as f:
            f.attrs["version_major"] = 0
            f.attrs["version_minor"] = 2
            g = f.create_group("data/datacube_000")
            g.attrs["emd_group_type"] = 1
            d = g.create_dataset("data", data=data, chunks=(1, 1, 8, 8))
            # rsciio's NCEM reader parses one axis dataset per data dim.
            for i, nm in enumerate(["dim1", "dim2", "dim3", "dim4"]):
                ax = g.create_dataset(nm, data=np.arange(d.shape[i], dtype=np.float32))
                ax.attrs["name"] = nm
                ax.attrs["units"] = "nm"
        return EmdAdapter.create_from_config(SourceConfig(url=str(path)))

    def test_valid_signal_index_resolves(self, tmp_path):
        adapter = self._emd_adapter(tmp_path)
        field = adapter._within_source_field(
            adapter.list_tensor_descriptors()[0].array_id
        )
        ta = adapter.get_tensor_adapter(field)
        assert ta.get_tensor_descriptor().array_id == f"{adapter.source_id}/0"

    def test_out_of_range_signal_raises_not_found(self, tmp_path):
        adapter = self._emd_adapter(tmp_path)
        with pytest.raises(TensorNotFound) as ei:
            adapter.get_tensor_adapter(f"{adapter.source_id}/99")
        assert ei.value.grpc_code == "NOT_FOUND"
        assert ei.value.reason == "unknown_field"

    def test_non_integer_signal_raises_invalid_argument(self, tmp_path):
        adapter = self._emd_adapter(tmp_path)
        with pytest.raises(InvalidTensorId) as ei:
            adapter.get_tensor_adapter(f"{adapter.source_id}/xyz")
        assert ei.value.grpc_code == "INVALID_ARGUMENT"
        assert ei.value.reason == "malformed_tensor_id"


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

    def test_bare_source_id_resolves_default_single_tensor(self):
        # A bare source_id (the #44 back-compat default) resolves to the sole
        # tensor, same as the empty form -- both reduce to field None.
        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("src", _SingleTensorAdapter("src"))
        info = _flight_info_for(server, "src", "src")
        assert isinstance(info, flight.FlightInfo)

    def test_bare_source_id_resolves_default_multi_tensor(self, tmp_path):
        # The chokepoint substitutes the default (first) tensor for a no-field
        # request on a *multi-tensor* source too (#44): a bare source_id -> field
        # None -> the first descriptor's field, rather than forwarding None to the
        # scene resolver (which faulted -- NOT_FOUND for OME-TIFF/bioio, an
        # uncaught None.split for OME-Zarr HCS -> INTERNAL) pre-#508.
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter
        from biopb_tensor_server.fixtures import create_multi_series_ome_tiff

        path, _, _ = create_multi_series_ome_tiff(str(tmp_path), n_series=3)
        server = TensorFlightServer("grpc://localhost:0")
        adapter = OmeTiffAdapter(path, "idx")
        adapter.list_tensor_descriptors()
        server.register_source("idx", adapter)

        # bare source_id and empty both resolve; the first tensor's array_id is
        # what the descriptor reports.
        for tid in ("idx", ""):
            info = _flight_info_for(server, "idx", tid)
            assert isinstance(info, flight.FlightInfo)
        # an explicit unknown field on the same multi-tensor source still faults
        # terminally (not INTERNAL), so the default-substitution did not mask misses.
        with pytest.raises(flight.FlightServerError) as ei:
            _flight_info_for(server, "idx", "idx/Image:999")
        assert not isinstance(ei.value, flight.FlightInternalError)
        assert json.loads(ei.value.extra_info)["code"] == "NOT_FOUND"
