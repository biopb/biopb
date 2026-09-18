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
    return client.create_source(
        source_name=name, shape=shape, dtype="<u2", chunk_shape=chunk
    )


def _put(client, handle, start, stop):
    data = np.full(
        [b - a for a, b in zip(start, stop, strict=True)], 7, dtype=np.uint16
    )
    client.upload_chunk(handle, ChunkBounds(start=list(start), stop=list(stop)), data)


class TestOnTheDescriptor:
    def test_a_fresh_upload_reports_pending_with_its_grid(self, client):
        handle = _make(client)
        source_id = handle.source_id
        desc = client.get_descriptor(source_id, with_upload_status=True)
        assert desc.HasField("upload_status")
        assert desc.upload_status.state == UploadStatusPb.PENDING
        assert desc.upload_status.expected_chunks == 4  # 4x4 in 2x2
        assert desc.upload_status.uploaded_chunks == 0

    def test_it_advances_as_chunks_land(self, client):
        handle = _make(client)
        source_id = handle.source_id
        _put(client, handle, (0, 0), (2, 2))
        desc = client.get_descriptor(source_id, with_upload_status=True)
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
        source_id = _make(
            TensorFlightClient(f"grpc://localhost:{writable_server.port}")
        ).source_id
        adapter = writable_server.sources.get(source_id)
        adapter.capability_token = "cap-token"

        holder = TensorFlightClient(
            f"grpc://localhost:{writable_server.port}", token="cap-token"
        )
        try:
            desc = holder.get_descriptor(source_id, with_upload_status=True)
            assert desc.upload_status.state == UploadStatusPb.PENDING
        finally:
            holder.close()

    def test_a_stranger_is_refused(self, writable_server):
        source_id = _make(
            TensorFlightClient(f"grpc://localhost:{writable_server.port}")
        ).source_id
        writable_server.sources.get(source_id).capability_token = "cap-token"

        stranger = TensorFlightClient(
            f"grpc://localhost:{writable_server.port}", token="wrong"
        )
        try:
            with pytest.raises(flight.FlightUnauthenticatedError):
                stranger.get_descriptor(source_id)
        finally:
            stranger.close()


class TestDiscarded:
    def test_a_tombstone_still_answers_with_its_reason(self, client, writable_server):
        """Describe is not a chunk read, so it never reaches
        `_refuse_if_discarded` -- a poller learns why instead of meeting a dead
        call. Its bytes stay unreadable; only the status is."""
        handle = _make(client, shape=(2, 2), chunk=(2, 2))
        source_id = handle.source_id
        writable_server.uploads.discard(source_id, "job died")

        desc = client.get_descriptor(source_id, with_upload_status=True)
        assert desc.upload_status.state == UploadStatusPb.DISCARDED
        assert desc.upload_status.reason == "job died"


class TestTheActionIsGone:
    def test_upload_status_is_not_advertised(self, client):
        advertised = {a.type for a in client._state.client.list_actions()}
        assert "upload_status" not in advertised
        assert "create_source" in advertised  # the listing itself still works

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
    def test_the_descriptor_cache_does_not_keep_the_status(self, client):
        """A cached PENDING would shadow the READY a later poll came for --
        turning the one field whose purpose is freshness into the stalest thing
        in the session."""
        handle = _make(client, shape=(2, 2), chunk=(2, 2))
        source_id = handle.source_id
        client.get_descriptor(source_id)  # seeds the structural cache

        cached = client._state.descriptors.get(source_id)
        assert cached is not None, "the probe should have seeded the cache"
        assert not cached.HasField("upload_status")

    def test_a_second_poll_sees_new_progress(self, client):
        """The end-to-end consequence: polling is live, not memoized."""
        handle = _make(client, shape=(2, 2), chunk=(2, 2))
        source_id = handle.source_id
        assert client.get_upload_status(source_id)["uploaded_chunks"] == 0
        _put(client, handle, (0, 0), (2, 2))
        status = client.get_upload_status(source_id)
        assert status["uploaded_chunks"] == 1
        # Still PENDING with its grid full: the count reports progress and
        # decides nothing, since `finish` is the only route to READY
        # (biopb/biopb#1048 step 5).
        assert status["state"] == "PENDING"


class TestSdkDictShape:
    def test_unknown_for_a_source_with_no_upload(self, client, writable_server):
        """Distinct from PENDING: no amount of polling moves it, which is what
        `wait_for_upload_ready` rejects on the first pass."""
        status = client.get_upload_status("cache_does_not_exist")
        assert status["state"] == "UNKNOWN"
        assert status["expected_chunks"] == 0

    def test_the_dict_keeps_its_shape(self, client):
        """The wire moved; the SDK's answer did not. `biopb_image_base` mirrors
        this dict in-process, and the two should stay the same shape."""
        handle = _make(client)
        source_id = handle.source_id
        status = client.get_upload_status(source_id)
        assert set(status) == {
            "source_id",
            "state",
            "expected_chunks",
            "uploaded_chunks",
            "reason",
        }
        assert status["source_id"] == source_id
        assert status["state"] == "PENDING"
