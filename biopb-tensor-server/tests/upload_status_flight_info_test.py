"""Upload status rides the descriptor, not an action (biopb/biopb#1048 step 2).

The fast-return protocol is *hold a capability, poll until READY, then read*.
Since `do_action` takes full access, an action could not serve the one caller
that needs it -- a holder of a per-source read capability and nothing else. So
the status is a field on the descriptor `get_flight_info` already returns, on a
call that is already source-scoped and capability-authorized.

The test that carries the point is `TestCapabilityHolderCanPoll`: everything
else here is mechanism.
"""

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb.tensor.client import TensorFlightClient
from biopb.tensor.descriptor_pb2 import UploadStatus as UploadStatusPb
from biopb.tensor.ticket_pb2 import ChunkBounds


def _make(client, name="cache:status", shape=(4, 4), chunk=(2, 2)):
    return client.create_tensor(
        name, np.empty(shape, dtype=np.uint16), chunk_shape=chunk
    )


def _put(client, desc, start, stop):
    data = np.full(
        [b - a for a, b in zip(start, stop, strict=True)], 7, dtype=np.uint16
    )
    client.upload_chunk(desc, ChunkBounds(start=list(start), stop=list(stop)), data)


class TestOnTheDescriptor:
    def test_a_fresh_upload_reports_pending_with_its_grid(self, client):
        desc = client.get_descriptor(_make(client).array_id, with_upload_status=True)
        assert desc.HasField("upload_status")
        assert desc.upload_status.state == UploadStatusPb.PENDING
        assert desc.upload_status.expected_chunks == 4  # 4x4 in 2x2
        assert desc.upload_status.uploaded_chunks == 0

    def test_it_advances_as_chunks_land(self, client):
        desc = _make(client)
        _put(client, desc, (0, 0), (2, 2))
        desc = client.get_descriptor(desc.array_id, with_upload_status=True)
        assert desc.upload_status.uploaded_chunks == 1
        assert desc.upload_status.state == UploadStatusPb.PENDING

    def test_an_ordinary_source_leaves_it_unset(
        self, client, writable_server, tmp_path
    ):
        """Unset, not PENDING-with-zeroes.

        A file-backed source is not an upload that has made no progress, and a
        client has to be able to tell -- a zeroed status would read as "still
        arriving" on a source nothing is writing to.
        """
        import zarr
        from biopb_tensor_server.adapters.zarr import ZarrAdapter

        path = tmp_path / "plain.zarr"
        arr = zarr.open_array(
            str(path), mode="w", shape=(4, 4), chunks=(2, 2), dtype="uint8"
        )
        arr[:] = 3
        writable_server.register_source(
            "plain",
            ZarrAdapter(zarr.open_array(str(path), mode="r"), "plain", ["y", "x"]),
        )

        desc = client.get_descriptor("plain", with_upload_status=True)
        assert desc.shape == [4, 4]  # a real, readable source
        assert not desc.HasField("upload_status")


class TestCapabilityHolderCanPoll:
    """The reason this is not an action.

    A source carrying a capability token is reachable by that token alone --
    which is exactly the fast-return handle. Under `do_action`-is-full-access,
    an action-shaped status would leave such a caller able to read the result
    but unable to learn when it was readable.
    """

    def test_the_capability_opens_the_status(self, writable_server):
        desc = _make(TensorFlightClient(f"grpc://localhost:{writable_server.port}"))
        adapter = writable_server.sources.get(desc.array_id)
        adapter.capability_token = "cap-token"

        holder = TensorFlightClient(
            f"grpc://localhost:{writable_server.port}", token="cap-token"
        )
        try:
            desc = holder.get_descriptor(desc.array_id, with_upload_status=True)
            assert desc.upload_status.state == UploadStatusPb.PENDING
        finally:
            holder.close()

    def test_a_stranger_is_refused(self, writable_server):
        desc = _make(TensorFlightClient(f"grpc://localhost:{writable_server.port}"))
        writable_server.sources.get(desc.array_id).capability_token = "cap-token"

        stranger = TensorFlightClient(
            f"grpc://localhost:{writable_server.port}", token="wrong"
        )
        try:
            with pytest.raises(flight.FlightUnauthenticatedError):
                stranger.get_descriptor(desc.array_id)
        finally:
            stranger.close()


class TestDiscarded:
    def test_a_tombstone_still_answers_with_its_reason(self, client, writable_server):
        """Describe is not a chunk read, so it never reaches
        `_refuse_if_discarded` -- a poller learns why instead of meeting a dead
        call. Its bytes stay unreadable; only the status is."""
        desc = _make(client, shape=(2, 2), chunk=(2, 2))
        writable_server.uploads.discard(desc.array_id, "job died")

        desc = client.get_descriptor(desc.array_id, with_upload_status=True)
        assert desc.upload_status.state == UploadStatusPb.DISCARDED
        assert desc.upload_status.reason == "job died"


class TestTheActionIsGone:
    def test_upload_status_is_not_advertised(self, client):
        advertised = {a.type for a in client._state.client.list_actions()}
        assert "upload_status" not in advertised
        assert "create_tensor" in advertised  # the listing itself still works

    def test_calling_it_fails(self, client):
        """Deleted rather than gated: the policy is satisfied by removing the
        action, not by refusing it."""
        with pytest.raises(flight.FlightError):
            list(
                client._state.client.do_action(
                    flight.Action("upload_status", b"cache_whatever")
                )
            )


class TestNotCached:
    def test_there_is_no_descriptor_cache_to_keep_it_in(self, client):
        """A cached PENDING would shadow the READY a later poll came for --
        turning the one field whose purpose is freshness into the stalest thing
        in the session. The SDK keeps no descriptor at all now, so there is
        nowhere for a stale status to live."""
        desc = _make(client, shape=(2, 2), chunk=(2, 2))
        client.get_descriptor(desc.array_id)

        assert not hasattr(client._state, "descriptors")
        assert not hasattr(client, "_descriptors")

    def test_a_second_poll_sees_new_progress(self, client):
        """The end-to-end consequence: polling is live, not memoized."""
        desc = _make(client, shape=(2, 2), chunk=(2, 2))
        assert client.get_upload_status(desc.array_id)["uploaded_chunks"] == 0
        _put(client, desc, (0, 0), (2, 2))
        status = client.get_upload_status(desc.array_id)
        assert status["uploaded_chunks"] == 1
        # Still PENDING with its grid full: the count reports progress and
        # decides nothing, since `finish` is the only route to READY
        # (biopb/biopb#1048 step 5).
        assert status["state"] == "PENDING"


class TestSdkDictShape:
    def test_unknown_for_a_source_with_no_upload(self, client, writable_server):
        """Distinct from PENDING: no amount of polling moves it, which is what
        a caller's poll loop stops on at once.

        Also guards a seam: the descriptor probe under this now restates a
        not-found id as a ValueError rather than letting the Flight error
        through, and this caller asked a narrower question than `describe` does
        -- it must still answer UNKNOWN rather than raise.
        """
        status = client.get_upload_status("cache_does_not_exist")
        assert status["state"] == "UNKNOWN"
        assert status["expected_chunks"] == 0

    def test_an_unresolved_source_still_raises(self, client, monkeypatch):
        """The other half of that seam: swallowing every ValueError would turn
        the resolve steer into a bland UNKNOWN, which tells a cloud-source owner
        nothing about what to do next. Only the addressing error is absorbed."""
        from biopb.tensor._session import _unresolved_source_error

        def _unresolved(*args, **kwargs):
            raise _unresolved_source_error("cloud_x")

        monkeypatch.setattr(client._catalog, "_fetch_tensor_descriptor", _unresolved)
        with pytest.raises(ValueError, match=r"client\.resolve"):
            client.get_upload_status("cloud_x")

    def test_the_dict_keeps_its_shape(self, client):
        """The wire moved; the SDK's answer did not. `biopb_image_base` mirrors
        this dict in-process, and the two should stay the same shape."""
        desc = _make(client)
        status = client.get_upload_status(desc.array_id)
        assert set(status) == {
            "source_id",
            "state",
            "expected_chunks",
            "uploaded_chunks",
            "reason",
        }
        assert status["source_id"] == desc.array_id
        assert status["state"] == "PENDING"
