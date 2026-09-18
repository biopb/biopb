"""An upload's lifecycle: one name, one adapter, sealed by ``finish``.

Three rules (biopb/biopb#1048 steps 4 and 5):

- a name is **single-use** for the life of the server: ``create_source``
  refuses a collision -- pending, finished or discarded alike -- so
  ``source_id`` alone names an attempt, and a writer's chunks can only land
  in the source it created;
- ``finish`` is the **only** route to READY, replacing a chunk count that was
  never a completeness check;
- reading is **not** gated on either: an early read of a half-filled upload is
  allowed, and an unwritten chunk refuses by itself.
"""

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb.tensor.ticket_pb2 import ChunkBounds


def _make(client, name="cache:lifecycle", shape=(4, 4), chunk=(2, 2)):
    return client.create_source(
        source_name=name, shape=shape, dtype="<u2", chunk_shape=chunk
    )


def _put(client, source_id, start, stop, fill=7):
    data = np.full(
        [b - a for a, b in zip(start, stop, strict=True)], fill, dtype=np.uint16
    )
    client.upload_chunk(
        source_id, ChunkBounds(start=list(start), stop=list(stop)), data
    )


def _fill(client, source_id, shape=(4, 4), chunk=(2, 2)):
    """Write every chunk of the declared grid, and nothing more."""
    for y in range(0, shape[0], chunk[0]):
        for x in range(0, shape[1], chunk[1]):
            _put(client, source_id, (y, x), (y + chunk[0], x + chunk[1]))


class TestOneNameOneAdapter:
    """Step 4: a name is taken by whoever created it, for the life of the server.

    A named ``cache:`` upload has a deterministic id. Were a second create to
    replace the first's adapter, the first writer's chunks -- and its
    ``finish`` -- would land in a source that is no longer its own, with
    nothing to tell it so: status is keyed by the id it still holds. Refusing
    the collision is what lets ``source_id`` name the attempt, with no second
    token to carry.
    """

    def test_a_second_create_under_a_pending_name_is_refused(self, client):
        _make(client, name="cache:taken")

        with pytest.raises(flight.FlightServerError, match="already exists"):
            _make(client, name="cache:taken")

    def test_the_first_upload_is_untouched_by_the_refusal(
        self, writable_server, client
    ):
        source_id = _make(client, name="cache:mine")
        adapter = writable_server.sources.get(source_id)
        with pytest.raises(flight.FlightServerError):
            _make(client, name="cache:mine")

        _put(client, source_id, (0, 0), (2, 2))  # no raise
        assert writable_server.sources.get(source_id) is adapter
        assert client.get_upload_status(source_id)["uploaded_chunks"] == 1

    def test_a_finished_name_is_refused_too(self, client):
        """Sealed is not free. A name that could be reclaimed by finishing is
        a name a straggler can be raced for."""
        source_id = _make(client, name="cache:done")
        _put(client, source_id, (0, 0), (2, 2))
        client.finish_upload(source_id)

        with pytest.raises(flight.FlightServerError, match="already exists"):
            _make(client, name="cache:done")
        assert client.get_upload_status(source_id)["state"] == "READY"

    def test_the_refusal_leaves_nothing_behind(self, writable_server, client):
        """The refused adapter is never registered -- the registry's count is
        what shows it."""
        _make(client, name="cache:once")
        before = len(writable_server.sources)

        with pytest.raises(flight.FlightServerError):
            _make(client, name="cache:once")
        assert len(writable_server.sources) == before

    def test_the_refusal_says_how_to_proceed(self, client):
        """It names the requested id and the minted-name form, since the caller
        holding a fixed name has to change something."""
        _make(client, name="cache:advice")

        with pytest.raises(flight.FlightServerError, match="cache:advice.*'cache:'"):
            _make(client, name="cache:advice")

    def test_an_unnamed_create_never_collides(self, client):
        first = _make(client, name="cache:")
        second = _make(client, name="cache:")
        assert first != second


class TestFinishSeals:
    """Step 5: completion is declared, not counted."""

    def test_writing_every_chunk_does_not_make_it_ready(self, client):
        """The retired rule. ``expected_chunks`` is ceil(shape/chunk_shape), but
        a cache-backed source takes arbitrary bounds -- so reaching the count
        never meant the array was covered, and this is the behaviour that
        encoded the confusion."""
        source_id = _make(client)
        _fill(client, source_id)

        status = client.get_upload_status(source_id)
        assert status["state"] == "PENDING"
        assert status["uploaded_chunks"] == status["expected_chunks"] == 4

    def test_finish_makes_it_ready(self, client):
        source_id = _make(client)
        _fill(client, source_id)

        assert client.finish_upload(source_id).state == 2  # READY
        assert client.get_upload_status(source_id)["state"] == "READY"

    def test_finish_does_not_require_a_full_grid(self, client):
        """The producer decides what complete means; the count only reports.

        A source written on bounds of its own choosing is as finishable as one
        that tiled the declared grid -- which is precisely what the count could
        not express.
        """
        source_id = _make(client)
        _put(client, source_id, (0, 0), (2, 2))

        status = client.finish_upload(source_id)
        assert status.state == 2
        assert status.uploaded_chunks == 1
        assert status.expected_chunks == 4

    def test_a_write_after_finish_is_refused_as_sealed(self, client):
        source_id = _make(client)
        _put(client, source_id, (0, 0), (2, 2))
        client.finish_upload(source_id)

        with pytest.raises(flight.FlightCancelledError, match="already finished"):
            _put(client, source_id, (2, 2), (4, 4))

    def test_the_sealed_source_keeps_what_it_had(self, client):
        """A refused write changes nothing -- the seal is not a rollback."""
        source_id = _make(client)
        _put(client, source_id, (0, 0), (2, 2))
        client.finish_upload(source_id)
        with pytest.raises(flight.FlightCancelledError):
            _put(client, source_id, (2, 2), (4, 4))

        assert client.get_upload_status(source_id)["uploaded_chunks"] == 1

    def test_finish_is_idempotent(self, client):
        source_id = _make(client)
        _put(client, source_id, (0, 0), (2, 2))

        client.finish_upload(source_id)
        assert client.finish_upload(source_id).state == 2  # not an error

    def test_a_discarded_upload_cannot_be_finished(self, writable_server, client):
        source_id = _make(client, name="cache:doomed")
        writable_server.uploads.discard(source_id, "job died")

        with pytest.raises(flight.FlightCancelledError, match="job died"):
            client.finish_upload(source_id)

    def test_a_finished_upload_can_still_be_discarded(self, writable_server, client):
        """Sealing publishes a result; it does not make it permanent."""
        source_id = _make(client)
        _put(client, source_id, (0, 0), (2, 2))
        client.finish_upload(source_id)

        status = writable_server.uploads.discard(source_id, "withdrawn")
        assert status["state"] == "DISCARDED"

    def test_finishing_something_that_is_not_an_upload_is_an_error(
        self, writable_server
    ):
        """Unlike ``discard``, which is total: a ``finish`` that names nothing
        means the caller believes it has been writing somewhere it has not."""
        with pytest.raises(flight.FlightServerError, match="not an upload"):
            writable_server.uploads.finish("no-such-source")


class TestReadingIsNotGatedOnFinish:
    """Sealing bounds the *writes*, not the reads.

    An early read of a half-filled upload stays allowed: a consumer that knows
    which region has landed has a real use for it, and the server has no way to
    know it does not. What an unwritten chunk gets is what it always got -- a
    per-chunk refusal naming the bounds, from the adapter that would have had to
    invent the data.
    """

    def test_a_written_chunk_reads_before_finish(self, client):
        source_id = _make(client)
        _fill(client, source_id)

        arr = np.asarray(client.get_tensor(source_id))
        assert (arr == 7).all()

    def test_an_unwritten_chunk_still_refuses_by_itself(self, client):
        """The failure stays scoped to the chunk that is missing -- which is
        what makes a partial read usable rather than all-or-nothing."""
        source_id = _make(client)
        _put(client, source_id, (0, 0), (2, 2))

        with pytest.raises(flight.FlightError, match="holds no chunk"):
            client.get_tensor(source_id).compute()

    def test_a_status_poll_answers_while_pending(self, client):
        source_id = _make(client)
        _put(client, source_id, (0, 0), (2, 2))

        desc = client.get_descriptor(source_id, with_upload_status=True)
        assert desc.upload_status.state == 1  # PENDING
        assert desc.upload_status.uploaded_chunks == 1

    def test_a_discarded_source_reports_its_reason(self, writable_server, client):
        """Disposal is the one terminal state a reader does meet: the adapter
        stays as a tombstone so the reason reaches it."""
        source_id = _make(client)
        _put(client, source_id, (0, 0), (2, 2))
        writable_server.uploads.discard(source_id, "client went away")

        with pytest.raises(flight.FlightError, match="client went away"):
            client.get_tensor(source_id).compute()

    def test_finishing_changes_nothing_for_a_reader(self, client):
        """Stated as a test because the reverse was briefly built: a sealed
        source reads exactly as the same source read a moment earlier."""
        source_id = _make(client)
        _fill(client, source_id)
        before = np.asarray(client.get_tensor(source_id))

        client.finish_upload(source_id)

        np.testing.assert_array_equal(before, np.asarray(client.get_tensor(source_id)))
