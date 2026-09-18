"""Upload sessions: who may write, and when writing stops.

Three rules (biopb/biopb#1048 steps 4 and 5):

- a write names the **attempt** it belongs to, not just the source, so the
  writer displaced by a second ``create_source`` for the same name learns it
  was displaced instead of filling a stranger's source;
- ``finish`` is the **only** route to READY, replacing a chunk count that was
  never a completeness check;
- reading is **not** gated on either: an early read of a half-filled upload is
  allowed, and an unwritten chunk refuses by itself.
"""

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb.tensor.ticket_pb2 import ChunkBounds


def _handle(client, name="cache:sessioned", shape=(4, 4), chunk=(2, 2)):
    return client.create_source(
        source_name=name, shape=shape, dtype="<u2", chunk_shape=chunk
    )


def _put(client, handle, start, stop, fill=7):
    data = np.full(
        [b - a for a, b in zip(start, stop, strict=True)], fill, dtype=np.uint16
    )
    client.upload_chunk(handle, ChunkBounds(start=list(start), stop=list(stop)), data)


def _fill(client, handle, shape=(4, 4), chunk=(2, 2)):
    """Write every chunk of the declared grid, and nothing more."""
    for y in range(0, shape[0], chunk[0]):
        for x in range(0, shape[1], chunk[1]):
            _put(client, handle, (y, x), (y + chunk[0], x + chunk[1]))


class TestSessionIdentity:
    """Step 4: ``source_id`` names the object, ``session_id`` names the attempt."""

    def test_a_create_hands_back_a_session(self, client):
        handle = _handle(client)
        assert handle.source_id
        assert handle.session_id

    def test_two_creates_of_the_same_name_differ_by_session(self, client):
        first = _handle(client, name="cache:same-name")
        second = _handle(client, name="cache:same-name")
        # Deterministic id for a named cache upload -- that is the whole reason
        # supersession is possible at all.
        assert second.source_id == first.source_id
        assert second.session_id != first.session_id

    def test_the_displaced_writer_is_refused_as_superseded(self, client):
        first = _handle(client, name="cache:taken")
        _handle(client, name="cache:taken")

        with pytest.raises(flight.FlightCancelledError, match="superseded"):
            _put(client, first, (0, 0), (2, 2))

    def test_the_current_session_is_unaffected(self, client):
        _handle(client, name="cache:taken-2")
        second = _handle(client, name="cache:taken-2")

        _put(client, second, (0, 0), (2, 2))  # no raise
        assert client.get_upload_status(second.source_id)["uploaded_chunks"] == 1

    def test_a_write_naming_no_session_is_refused(self, client):
        """An unnamed write is what a build from before sessions sends, and what
        a caller reusing a bare ``source_id`` would send. It must not be read as
        "whichever session is current"."""
        from biopb.tensor._upload import UploadHandle

        handle = _handle(client, name="cache:anonymous")
        with pytest.raises(flight.FlightCancelledError, match="superseded"):
            _put(client, UploadHandle(handle.source_id, ""), (0, 0), (2, 2))

    def test_status_cannot_tell_the_displaced_writer_anything(self, client):
        """Why the refusal message is load-bearing rather than a nicety.

        Status is keyed by ``source_id``, so the displaced producer polling it
        reads its *successor's* progress -- a healthy PENDING that says nothing
        about its own attempt being over. The error on the write is the only
        channel it has.
        """
        first = _handle(client, name="cache:invisible")
        second = _handle(client, name="cache:invisible")
        _put(client, second, (0, 0), (2, 2))

        status = client.get_upload_status(first.source_id)
        assert status["state"] == "PENDING"
        assert status["uploaded_chunks"] == 1  # the successor's chunk, not its own


class TestFinishSeals:
    """Step 5: completion is declared, not counted."""

    def test_writing_every_chunk_does_not_make_it_ready(self, client):
        """The retired rule. ``expected_chunks`` is ceil(shape/chunk_shape), but
        a cache-backed source takes arbitrary bounds -- so reaching the count
        never meant the array was covered, and this is the behaviour that
        encoded the confusion."""
        handle = _handle(client)
        _fill(client, handle)

        status = client.get_upload_status(handle.source_id)
        assert status["state"] == "PENDING"
        assert status["uploaded_chunks"] == status["expected_chunks"] == 4

    def test_finish_makes_it_ready(self, client):
        handle = _handle(client)
        _fill(client, handle)

        assert client.finish_upload(handle).state == 2  # READY
        assert client.get_upload_status(handle.source_id)["state"] == "READY"

    def test_finish_does_not_require_a_full_grid(self, client):
        """The producer decides what complete means; the count only reports.

        A source written on bounds of its own choosing is as finishable as one
        that tiled the declared grid -- which is precisely what the count could
        not express.
        """
        handle = _handle(client)
        _put(client, handle, (0, 0), (2, 2))

        status = client.finish_upload(handle)
        assert status.state == 2
        assert status.uploaded_chunks == 1
        assert status.expected_chunks == 4

    def test_a_write_after_finish_is_refused_as_sealed(self, client):
        handle = _handle(client)
        _put(client, handle, (0, 0), (2, 2))
        client.finish_upload(handle)

        with pytest.raises(flight.FlightCancelledError, match="already finished"):
            _put(client, handle, (2, 2), (4, 4))

    def test_the_sealed_source_keeps_what_it_had(self, client):
        """A refused write changes nothing -- the seal is not a rollback."""
        handle = _handle(client)
        _put(client, handle, (0, 0), (2, 2))
        client.finish_upload(handle)
        with pytest.raises(flight.FlightCancelledError):
            _put(client, handle, (2, 2), (4, 4))

        assert client.get_upload_status(handle.source_id)["uploaded_chunks"] == 1

    def test_finish_is_idempotent_for_its_own_session(self, client):
        handle = _handle(client)
        _put(client, handle, (0, 0), (2, 2))

        client.finish_upload(handle)
        assert client.finish_upload(handle).state == 2  # not an error

    def test_a_superseded_session_cannot_finish(self, client):
        """Otherwise a straggler could publish a source it did not write."""
        first = _handle(client, name="cache:finish-race")
        second = _handle(client, name="cache:finish-race")
        _put(client, second, (0, 0), (2, 2))

        with pytest.raises(flight.FlightCancelledError, match="superseded"):
            client.finish_upload(first)
        assert client.get_upload_status(second.source_id)["state"] == "PENDING"

    def test_a_discarded_upload_cannot_be_finished(self, writable_server, client):
        handle = _handle(client, name="cache:doomed")
        writable_server.uploads.discard(handle.source_id, "job died")

        with pytest.raises(flight.FlightCancelledError, match="job died"):
            client.finish_upload(handle)

    def test_a_finished_upload_can_still_be_discarded(self, writable_server, client):
        """Sealing publishes a result; it does not make it permanent."""
        handle = _handle(client)
        _put(client, handle, (0, 0), (2, 2))
        client.finish_upload(handle)

        status = writable_server.uploads.discard(handle.source_id, "withdrawn")
        assert status["state"] == "DISCARDED"

    def test_finishing_something_that_is_not_an_upload_is_an_error(
        self, writable_server
    ):
        """Unlike ``discard``, which is total: a ``finish`` that names nothing
        means the caller believes it has been writing somewhere it has not."""
        with pytest.raises(flight.FlightServerError, match="not an upload"):
            writable_server.uploads.finish("no-such-source", "no-such-session")


class TestReadingIsNotGatedOnFinish:
    """Sealing bounds the *writes*, not the reads.

    An early read of a half-filled upload stays allowed: a consumer that knows
    which region has landed has a real use for it, and the server has no way to
    know it does not. What an unwritten chunk gets is what it always got -- a
    per-chunk refusal naming the bounds, from the adapter that would have had to
    invent the data.
    """

    def test_a_written_chunk_reads_before_finish(self, client):
        handle = _handle(client)
        _fill(client, handle)

        arr = np.asarray(client.get_tensor(handle.source_id))
        assert (arr == 7).all()

    def test_an_unwritten_chunk_still_refuses_by_itself(self, client):
        """The failure stays scoped to the chunk that is missing -- which is
        what makes a partial read usable rather than all-or-nothing."""
        handle = _handle(client)
        _put(client, handle, (0, 0), (2, 2))

        with pytest.raises(flight.FlightError, match="holds no chunk"):
            client.get_tensor(handle.source_id).compute()

    def test_a_status_poll_answers_while_pending(self, client):
        handle = _handle(client)
        _put(client, handle, (0, 0), (2, 2))

        desc = client.get_descriptor(handle.source_id, with_upload_status=True)
        assert desc.upload_status.state == 1  # PENDING
        assert desc.upload_status.uploaded_chunks == 1

    def test_a_discarded_source_reports_its_reason(self, writable_server, client):
        """Disposal is the one terminal state a reader does meet: the adapter
        stays as a tombstone so the reason reaches it."""
        handle = _handle(client)
        _put(client, handle, (0, 0), (2, 2))
        writable_server.uploads.discard(handle.source_id, "client went away")

        with pytest.raises(flight.FlightError, match="client went away"):
            client.get_tensor(handle.source_id).compute()

    def test_finishing_changes_nothing_for_a_reader(self, client):
        """Stated as a test because the reverse was briefly built: a sealed
        source reads exactly as the same source read a moment earlier."""
        handle = _handle(client)
        _fill(client, handle)
        before = np.asarray(client.get_tensor(handle.source_id))

        client.finish_upload(handle)

        np.testing.assert_array_equal(
            before, np.asarray(client.get_tensor(handle.source_id))
        )
