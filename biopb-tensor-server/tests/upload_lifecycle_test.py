"""An upload's lifecycle: one name, one adapter, one ladder to climb.

Three rules (biopb/biopb#1048 steps 4 and 5):

- a name is **single-use** while its source is registered: ``create_tensor``
  refuses a collision -- at any state alike -- so ``source_id`` alone names an
  attempt, and a writer's chunks can only land in the source it created
  (step 6's sweep, ``upload_reclaim_test.py``, is what frees a discarded name);
- ``set_upload_status`` is the **only** thing that moves an upload, replacing a
  chunk count that was never a completeness check;
- the ladder has **two gates, not one**: READY opens reads and FINISHED closes
  writes, so a producer can publish a result before it has finished writing it,
  and what it has not written reads as zeros.
"""

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb.tensor import UploadRefused
from biopb.tensor.ticket_pb2 import ChunkBounds


def _make(client, name="cache:lifecycle", shape=(4, 4), chunk=(2, 2)):
    """Declare a tensor; the descriptor is what every write takes."""
    return client.create_tensor(
        name, np.empty(shape, dtype=np.uint16), chunk_shape=chunk
    )


def _put(client, desc, start, stop, fill=7):
    data = np.full(
        [b - a for a, b in zip(start, stop, strict=True)], fill, dtype=np.uint16
    )
    client.upload_chunk(desc, ChunkBounds(start=list(start), stop=list(stop)), data)


def _fill(client, desc, shape=(4, 4), chunk=(2, 2)):
    """Write every chunk of the declared grid, and nothing more."""
    for y in range(0, shape[0], chunk[0]):
        for x in range(0, shape[1], chunk[1]):
            _put(client, desc, (y, x), (y + chunk[0], x + chunk[1]))


def _set(client, desc, state, reason=""):
    return client.set_upload_status(desc, state, reason)


class TestOneNameOneAdapter:
    """Step 4: a name is taken by whoever created it, for as long as it stands.

    A named ``cache:`` upload has a deterministic id. Were a second create to
    replace the first's adapter, the first writer's chunks -- and its
    transitions -- would land in a source that is no longer its own, with
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
        desc = _make(client, name="cache:mine")
        adapter = writable_server.sources.get(desc.array_id)
        with pytest.raises(flight.FlightServerError):
            _make(client, name="cache:mine")

        _put(client, desc, (0, 0), (2, 2))  # no raise
        assert writable_server.sources.get(desc.array_id) is adapter
        assert client.get_upload_status(desc.array_id)["uploaded_chunks"] == 1

    def test_a_finished_name_is_refused_too(self, client):
        """Sealed is not free. A name that could be reclaimed by finishing is
        a name a straggler can be raced for."""
        desc = _make(client, name="cache:done")
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "FINISHED")

        with pytest.raises(flight.FlightServerError, match="already exists"):
            _make(client, name="cache:done")
        assert client.get_upload_status(desc.array_id)["state"] == "FINISHED"

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
        assert first.array_id != second.array_id


class TestTheLadder:
    """Step 5: the state is declared, not counted -- and it only climbs."""

    def test_writing_every_chunk_does_not_move_it(self, client):
        """The retired rule. ``expected_chunks`` is ceil(shape/chunk_shape), but
        a cache-backed source takes arbitrary bounds -- so reaching the count
        never meant the array was covered, and this is the behaviour that
        encoded the confusion."""
        desc = _make(client)
        _fill(client, desc)

        status = client.get_upload_status(desc.array_id)
        assert status["state"] == "PENDING"
        assert status["uploaded_chunks"] == status["expected_chunks"] == 4

    def test_ready_publishes_and_finished_seals(self, client):
        desc = _make(client)
        _fill(client, desc)

        assert _set(client, desc, "READY")["state"] == "READY"
        assert client.get_upload_status(desc.array_id)["state"] == "READY"
        assert _set(client, desc, "FINISHED")["state"] == "FINISHED"
        assert client.get_upload_status(desc.array_id)["state"] == "FINISHED"

    def test_finished_from_pending_skips_a_rung(self, client):
        """One call for the ordinary case: a producer that never wanted a
        partial read publishes and seals at once."""
        desc = _make(client)
        _fill(client, desc)

        assert _set(client, desc, "FINISHED")["state"] == "FINISHED"
        assert (np.asarray(client.get_tensor(desc.array_id)) == 7).all()

    def test_no_state_requires_a_full_grid(self, client):
        """The producer decides what complete means; the count only reports.

        A source written on bounds of its own choosing is as finishable as one
        that tiled the declared grid -- which is precisely what the count could
        not express.
        """
        desc = _make(client)
        _put(client, desc, (0, 0), (2, 2))

        status = _set(client, desc, "FINISHED")
        assert status["state"] == "FINISHED"
        assert status["uploaded_chunks"] == 1
        assert status["expected_chunks"] == 4

    def test_a_write_after_ready_still_lands(self, client):
        """READY is not a seal. This is the whole point of splitting it off."""
        desc = _make(client)
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "READY")

        _put(client, desc, (2, 2), (4, 4))  # no raise
        assert client.get_upload_status(desc.array_id)["uploaded_chunks"] == 2

    def test_a_write_after_finished_is_refused_as_sealed(self, client):
        desc = _make(client)
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "FINISHED")

        with pytest.raises(UploadRefused) as exc:
            _put(client, desc, (2, 2), (4, 4))
        assert exc.value.state == "FINISHED"
        assert exc.value.source_id == desc.array_id

    def test_the_sealed_source_keeps_what_it_had(self, client):
        """A refused write changes nothing -- the seal is not a rollback."""
        desc = _make(client)
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "FINISHED")
        with pytest.raises(UploadRefused):
            _put(client, desc, (2, 2), (4, 4))

        assert client.get_upload_status(desc.array_id)["uploaded_chunks"] == 1

    def test_setting_the_state_it_is_in_is_a_no_op(self, client):
        desc = _make(client)
        _put(client, desc, (0, 0), (2, 2))

        _set(client, desc, "FINISHED")
        assert _set(client, desc, "FINISHED")["state"] == "FINISHED"  # not an error

    def test_the_ladder_does_not_descend(self, client):
        """A published result cannot be unpublished, and a sealed one cannot be
        reopened for writes a reader has already been told are over."""
        desc = _make(client)
        _set(client, desc, "FINISHED")

        with pytest.raises(flight.FlightServerError, match="cannot go back"):
            _set(client, desc, "READY")
        assert client.get_upload_status(desc.array_id)["state"] == "FINISHED"

    def test_pending_is_not_settable(self, client):
        """An upload starts there and nothing returns it there."""
        desc = _make(client)

        with pytest.raises(flight.FlightServerError, match="not a state"):
            _set(client, desc, "PENDING")

    def test_a_discarded_upload_cannot_be_moved(self, writable_server, client):
        desc = _make(client, name="cache:doomed")
        writable_server.uploads.discard(desc.array_id, "job died")

        with pytest.raises(UploadRefused, match="job died") as exc:
            _set(client, desc, "FINISHED")
        assert exc.value.state == "DISCARDED"
        assert exc.value.reason == "job died"

    def test_a_finished_upload_can_still_be_discarded(self, client):
        """Sealing publishes a result; it does not make it permanent. Discard
        is reachable from every rung, which is what makes it the only delete."""
        desc = _make(client)
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "FINISHED")

        assert _set(client, desc, "DISCARDED", "withdrawn")["state"] == "DISCARDED"
        with pytest.raises(flight.FlightError, match="withdrawn"):
            client.get_tensor(desc.array_id).compute()

    def test_climbing_something_that_is_not_an_upload_is_an_error(self, client):
        """Unlike DISCARDED, which is total: a climb that names nothing means
        the caller believes it has been writing somewhere it has not."""
        with pytest.raises(flight.FlightServerError, match="not an upload"):
            client.set_upload_status("no-such-source", "FINISHED")

    def test_discarding_something_that_is_not_an_upload_is_not(self, client):
        """A statement about the end state, not a receipt: a retry after the
        tombstone is reclaimed must not fail."""
        status = client.set_upload_status("no-such-source", "DISCARDED")
        assert status["state"] == "UNKNOWN"


class TestTheReadGate:
    """READY is what makes a hole meaningful, so reads wait for it.

    Before it, a chunk that has not arrived is indistinguishable from one still
    in flight; serving zeros for both would hand a consumer a half-written
    result it cannot tell from a finished one. After it, the producer has said
    the holes are meant to be holes.
    """

    def test_a_pending_upload_refuses_reads(self, client):
        desc = _make(client)
        _fill(client, desc)

        with pytest.raises(flight.FlightError, match="not readable yet"):
            client.get_tensor(desc.array_id).compute()

    def test_a_status_poll_answers_while_pending(self, client):
        """The gate is on the pixels, not on the descriptor -- polling to READY
        is what a consumer waiting on a result does."""
        desc = _make(client)
        _put(client, desc, (0, 0), (2, 2))

        desc = client.get_descriptor(desc.array_id, with_upload_status=True)
        assert desc.upload_status.state == 1  # PENDING
        assert desc.upload_status.uploaded_chunks == 1

    def test_ready_opens_it(self, client):
        desc = _make(client)
        _fill(client, desc)
        _set(client, desc, "READY")

        assert (np.asarray(client.get_tensor(desc.array_id)) == 7).all()

    def test_an_unwritten_chunk_reads_as_zeros(self, client):
        """Not an error: a producer that published a partial result meant the
        rest to be background."""
        desc = _make(client)
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "READY")

        arr = np.asarray(client.get_tensor(desc.array_id))
        assert (arr[:2, :2] == 7).all()
        assert (arr[2:, :] == 0).all()
        assert (arr[:, 2:] == 0).all()

    def test_a_chunk_written_after_publishing_is_served(self, writable_server, client):
        """What READY buys, at the seam that provides it.

        Asserted against the adapter rather than through ``get_tensor``: the
        SDK caches a chunk under its ``chunk_id``, which names fixed bytes
        everywhere else in this system, so a consumer that has already read a
        region keeps what it read. Watching a result fill in means re-reading
        from a process that has not, which is a client-side question; what the
        server owes is that the bytes are there to be read.
        """
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.core.chunk import mint_chunk_id
        from biopb_tensor_server.core.chunk_batch import unpack_chunk_array

        desc = _make(client)
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "READY")

        adapter = writable_server.sources.get(desc.array_id)
        bounds = ChunkBounds(start=[2, 2], stop=[4, 4])
        chunk_id = mint_chunk_id(
            adapter.source_id, bounds, content_version=adapter.content_version
        )
        cache = CacheManager.get_instance()
        assert not unpack_chunk_array(adapter.resolve_chunk_data(chunk_id, cache)).any()

        _put(client, desc, (2, 2), (4, 4))
        assert (
            unpack_chunk_array(adapter.resolve_chunk_data(chunk_id, cache)) == 7
        ).all()

    def test_a_discarded_source_reports_its_reason(self, writable_server, client):
        """Disposal is the one terminal state a reader does meet: the adapter
        stays as a tombstone so the reason reaches it."""
        desc = _make(client)
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "READY")
        writable_server.uploads.discard(desc.array_id, "client went away")

        with pytest.raises(flight.FlightError, match="client went away"):
            client.get_tensor(desc.array_id).compute()

    def test_sealing_changes_nothing_for_a_reader(self, client):
        """FINISHED bounds the writes, not the reads: a sealed source reads
        exactly as the same source read a moment earlier."""
        desc = _make(client)
        _fill(client, desc)
        _set(client, desc, "READY")
        before = np.asarray(client.get_tensor(desc.array_id))

        _set(client, desc, "FINISHED")

        np.testing.assert_array_equal(
            before, np.asarray(client.get_tensor(desc.array_id))
        )
