"""The GetFlightInfo field mask: nothing is implied by omission.

Replaces three ad-hoc ``with_*`` bools, one of which -- ``with_read_plan`` --
defaulted to *true*. So the cheap call used to be the one you had to ask for,
and the O(chunks) enumeration was what you got by saying nothing. That is
inverted here, which is the property most of these tests pin.

Residency is the reason it had to be a mask rather than a fourth bool: filled
unconditionally it would put a stat walk of the source on every tensor open
(biopb/biopb#1048).
"""

from pathlib import Path

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb.tensor.descriptor_pb2 import FlightRequest, TensorReadOption
from biopb_tensor_server.core.read_mask import READ_MASK_PATHS, read_mask
from google.protobuf.field_mask_pb2 import FieldMask


class TestReadMaskReader:
    """``read_mask`` -- the one place the legal paths are defined."""

    def test_empty_is_empty(self):
        """An empty mask is a describe, not 'everything'. The opposite reading
        would restore the old default by the back door."""
        assert read_mask(TensorReadOption(array_id="x")) == frozenset()

    def test_known_paths_pass_through(self):
        opt = TensorReadOption(
            array_id="x", fields=FieldMask(paths=["endpoints", "pyramid"])
        )
        assert read_mask(opt) == {"endpoints", "pyramid"}

    def test_an_unknown_path_is_refused_not_dropped(self):
        """Silently ignoring it would surface later as a field that never
        arrived, with nothing to explain why -- and a typo reads exactly like
        the server choosing not to fill it."""
        opt = TensorReadOption(array_id="x", fields=FieldMask(paths=["pyramyd"]))
        with pytest.raises(flight.FlightServerError, match="pyramyd"):
            read_mask(opt)

    def test_the_refusal_names_what_is_legal(self):
        opt = TensorReadOption(array_id="x", fields=FieldMask(paths=["nope"]))
        with pytest.raises(flight.FlightServerError, match="is_resident"):
            read_mask(opt)

    def test_one_bad_path_refuses_the_whole_request(self):
        """Not a partial answer: a caller that asked for four parts and got
        three has no way to tell which."""
        opt = TensorReadOption(
            array_id="x", fields=FieldMask(paths=["pyramid", "bogus"])
        )
        with pytest.raises(flight.FlightServerError):
            read_mask(opt)

    def test_the_path_set_matches_the_constants(self):
        """A tripwire on adding a path to one and not the other."""
        assert {
            "endpoints",
            "metadata_json",
            "pyramid",
            "upload_status",
            "is_resident",
        } == READ_MASK_PATHS


def _zarr_source(writable_server, tmp_path, name="plain", value=3):
    import zarr
    from biopb_tensor_server.adapters.zarr import ZarrAdapter

    path = tmp_path / f"{name}.zarr"
    arr = zarr.open_array(
        str(path), mode="w", shape=(4, 4), chunks=(2, 2), dtype="uint8"
    )
    arr[:] = value
    writable_server.register_source(
        name, ZarrAdapter(zarr.open_array(str(path), mode="r"), name, ["y", "x"])
    )
    return name


class TestResidencyIsOptIn:
    """The field this mask exists for."""

    def test_unset_when_nobody_asked(self, client, writable_server, tmp_path):
        sid = _zarr_source(writable_server, tmp_path)
        desc = client.get_descriptor(sid)
        assert not desc.HasField("is_resident")

    def test_answered_when_asked(self, client, writable_server, tmp_path):
        sid = _zarr_source(writable_server, tmp_path)
        desc = client.get_descriptor(sid, with_residency=True)
        assert desc.HasField("is_resident")
        assert desc.is_resident is True  # a real local zarr

    def test_a_plain_read_does_not_ask(self, client, writable_server, tmp_path):
        """The regression this guards: residency filled on every open would be
        the catalog-wide stat walk again, one source at a time."""
        sid = _zarr_source(writable_server, tmp_path)
        arr = client.get_tensor(sid)
        assert arr.shape == (4, 4)
        assert not client.get_descriptor(sid).HasField("is_resident")

    def test_every_answer_is_freshly_asked_for(self, client, writable_server, tmp_path):
        """descriptor.proto says outright that residency is not to be cached by
        a client: a synced folder re-dehydrates with nothing to notify anyone,
        so there is no moment at which a stored answer stays true. The SDK holds
        no descriptor, so a stored one cannot exist -- and asking once does not
        make the next caller, who did not ask, see it."""
        sid = _zarr_source(writable_server, tmp_path)
        assert client.get_descriptor(sid, with_residency=True).is_resident is True
        assert not client.get_descriptor(sid).HasField("is_resident")


class TestUnknownPathOverTheWire:
    def test_the_server_refuses_it(self, client, writable_server, tmp_path):
        """End to end, not just the reader: a bad path must not reach an
        adapter and come back as a confusing partial response."""
        sid = _zarr_source(writable_server, tmp_path)
        cmd = FlightRequest(
            tensor_read=TensorReadOption(
                array_id=sid, fields=FieldMask(paths=["not_a_path"])
            )
        )
        fd = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        with pytest.raises(flight.FlightError, match="not_a_path"):
            client._state.client.get_flight_info(fd)


class TestEndpointsAreOptIn:
    def test_a_describe_gets_no_endpoints(self, client, writable_server, tmp_path):
        sid = _zarr_source(writable_server, tmp_path)
        cmd = FlightRequest(tensor_read=TensorReadOption(array_id=sid))
        fd = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        info = client._state.client.get_flight_info(fd)
        assert list(info.endpoints) == []

    def test_asking_for_them_gets_them(self, client, writable_server, tmp_path):
        sid = _zarr_source(writable_server, tmp_path)
        cmd = FlightRequest(
            tensor_read=TensorReadOption(
                array_id=sid, fields=FieldMask(paths=["endpoints"])
            )
        )
        fd = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        info = client._state.client.get_flight_info(fd)
        # Count is the server's transfer grid, not the store's chunking; the
        # property here is that a plan arrived at all.
        assert list(info.endpoints)

    def test_the_read_path_still_reads(self, client, writable_server, tmp_path):
        """The SDK had to start asking for `endpoints` explicitly; if it ever
        stops, a read returns an empty plan rather than an error."""
        sid = _zarr_source(writable_server, tmp_path, value=7)
        assert np.asarray(client.get_tensor(sid))[0, 0] == 7


class TestEveryConstructionSetsAMask:
    """A source-level guard, because the failure is silent.

    A read path that forgets the mask does not error -- it gets an empty plan
    and reads nothing. That is how `SerializableTensorImg` was missed when the
    `with_*` bools were replaced: it built a bare `TensorReadOption`, relying on
    `with_read_plan` defaulting to true, and its tests never exercise the
    network path so nothing failed.

    So this scans the sources rather than the behaviour. Crude, but it is the
    shape of the mistake: every construction must say what it wants, and a new
    one that does not is exactly what this catches.
    """

    ROOT = Path(__file__).resolve().parents[2]

    def _sources(self, pattern, *dirs):
        for d in dirs:
            base = self.ROOT / d
            if base.exists():
                yield from (p for p in base.rglob(pattern) if "build" not in p.parts)

    def test_java_builders_set_fields(self):
        import re

        offenders = []
        for p in self._sources("*.java", "src/main/java"):
            text = p.read_text()
            for m in re.finditer(
                r"TensorReadOption\.newBuilder\(\)(.{0,400}?);", text, re.S
            ):
                if ".setFields(" not in m.group(1):
                    offenders.append(f"{p.name}:{text[: m.start()].count(chr(10)) + 1}")
        assert not offenders, (
            "TensorReadOption built without a field mask -- a read path that "
            f"omits `endpoints` silently returns no plan: {offenders}"
        )

    def test_python_builders_set_fields(self):
        """The SDK funnels every construction through ``_read_option``; the
        server's proxy extends the mask on the next line. Anything else is a
        site that has not been considered."""
        import re

        allowed = {"_session.py", "remote_tensor.py"}
        offenders = []
        for p in self._sources(
            "*.py", "src/main/python", "biopb-tensor-server/biopb_tensor_server"
        ):
            if p.name in allowed:
                continue
            text = p.read_text()
            for m in re.finditer(r"TensorReadOption\(", text):
                offenders.append(f"{p.name}:{text[: m.start()].count(chr(10)) + 1}")
        assert not offenders, f"unreviewed TensorReadOption construction: {offenders}"
