"""An upload's lifecycle: one name, one adapter, one ladder to climb.

Three rules (biopb/biopb#1048 steps 4 and 5):

- a field is **single-use** while its tensor is served: ``add_tensor``
  refuses a collision -- at any state alike -- so ``array_id`` alone names an
  attempt, and a writer's chunks can only land in the tensor it created
  (the reclaim sweep, ``upload_reclaim_test.py``, is what frees a discarded
  field);
- ``set_upload_status`` is the **only** thing that moves an upload, replacing a
  chunk count that was never a completeness check;
- the ladder has **one gate**: READY seals writes and opens reads together,
  so nothing readable is still writable, and a chunk that was never written
  reads as zeros.
"""

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb.tensor import UploadRefused
from biopb.tensor.ticket_pb2 import ChunkBounds


def _make(client, source, field="lifecycle", shape=(4, 4), chunk=(2, 2)):
    """Declare a tensor on *source*; the descriptor is what every write takes."""
    return client.add_tensor(
        f"cache://{source}/{field}",
        np.empty(shape, dtype=np.uint16),
        chunk_shape=chunk,
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


class TestOneFieldOneAdapter:
    """A field is taken by whoever added it, for as long as it stands.

    Were a second add to replace the first's adapter, the first writer's
    chunks -- and its transitions -- would land in a tensor that is no longer
    its own, with nothing to tell it so: status is keyed by the id it still
    holds. Refusing the collision is what lets ``array_id`` name the attempt,
    with no second token to carry.
    """

    def test_a_second_add_under_a_pending_field_is_refused(self, client, source):
        _make(client, source, field="taken")

        with pytest.raises(flight.FlightServerError, match="already has a tensor"):
            _make(client, source, field="taken")

    def test_the_first_upload_is_untouched_by_the_refusal(self, client, source):
        desc = _make(client, source, field="mine")
        with pytest.raises(flight.FlightServerError):
            _make(client, source, field="mine")

        _put(client, desc, (0, 0), (2, 2))  # no raise
        assert client.get_upload_status(desc.array_id)["uploaded_chunks"] == 1

    def test_a_published_field_is_refused_too(self, client, source):
        """Published is not free. A field that could be reclaimed by publishing
        is a field a straggler can be raced for."""
        desc = _make(client, source, field="done")
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "READY")

        with pytest.raises(flight.FlightServerError, match="already has a tensor"):
            _make(client, source, field="done")
        assert client.get_upload_status(desc.array_id)["state"] == "READY"

    def test_the_refusal_leaves_nothing_behind(self, writable_server, client, source):
        """The refused adapter is never attached -- the source's members are
        what show it."""
        _make(client, source, field="once")
        members = dict(writable_server.sources.get(source).members)

        with pytest.raises(flight.FlightServerError):
            _make(client, source, field="once")
        assert dict(writable_server.sources.get(source).members) == members

    def test_the_refusal_says_how_to_proceed(self, client, source):
        """It names the field it collided with, since the caller holding a
        fixed name has to change something."""
        _make(client, source, field="advice")

        with pytest.raises(flight.FlightServerError, match="'advice'.*another name"):
            _make(client, source, field="advice")

    def test_the_same_field_on_two_sources_never_collides(self, client, source):
        """A field is taken on its own source, not server-wide: that is the
        whole point of adding to a source rather than minting one."""
        other = client.register_source()
        first = _make(client, source, field="same")
        second = _make(client, other, field="same")
        assert first.array_id != second.array_id


class TestTheLadder:
    """Step 5: the state is declared, not counted -- and it only climbs."""

    def test_writing_every_chunk_does_not_move_it(self, client, source):
        """The retired rule. ``expected_chunks`` is ceil(shape/chunk_shape), but
        a cache-backed source takes arbitrary bounds -- so reaching the count
        never meant the array was covered, and this is the behaviour that
        encoded the confusion."""
        desc = _make(client, source)
        _fill(client, desc)

        status = client.get_upload_status(desc.array_id)
        assert status["state"] == "PENDING"
        assert status["uploaded_chunks"] == status["expected_chunks"] == 4

    def test_ready_publishes_and_seals(self, client, source):
        """One call, one moment: the result becomes readable and stops taking
        chunks together."""
        desc = _make(client, source)
        _fill(client, desc)

        assert _set(client, desc, "READY")["state"] == "READY"
        assert client.get_upload_status(desc.array_id)["state"] == "READY"
        assert (np.asarray(client.get_tensor(desc.array_id)) == 7).all()

    def test_no_state_requires_a_full_grid(self, client, source):
        """The producer decides what complete means; the count only reports.

        A source written on bounds of its own choosing is as finishable as one
        that tiled the declared grid -- which is precisely what the count could
        not express.
        """
        desc = _make(client, source)
        _put(client, desc, (0, 0), (2, 2))

        status = _set(client, desc, "READY")
        assert status["state"] == "READY"
        assert status["uploaded_chunks"] == 1
        assert status["expected_chunks"] == 4

    def test_a_write_after_ready_is_refused_as_sealed(self, client, source):
        """The seal is the half of READY that makes its reads cacheable."""
        desc = _make(client, source)
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "READY")

        with pytest.raises(UploadRefused) as exc:
            _put(client, desc, (2, 2), (4, 4))
        assert exc.value.state == "READY"
        assert exc.value.source_id == desc.array_id

    def test_the_sealed_source_keeps_what_it_had(self, client, source):
        """A refused write changes nothing -- the seal is not a rollback."""
        desc = _make(client, source)
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "READY")
        with pytest.raises(UploadRefused):
            _put(client, desc, (2, 2), (4, 4))

        assert client.get_upload_status(desc.array_id)["uploaded_chunks"] == 1

    def test_setting_the_state_it_is_in_is_a_no_op(self, client, source):
        desc = _make(client, source)
        _put(client, desc, (0, 0), (2, 2))

        _set(client, desc, "READY")
        assert _set(client, desc, "READY")["state"] == "READY"  # not an error

    def test_pending_is_not_settable(self, client, source):
        """An upload starts there and nothing returns it there -- which is also
        what keeps the ladder from descending, now that it has one rung to
        climb."""
        desc = _make(client, source)
        _set(client, desc, "READY")

        with pytest.raises(flight.FlightServerError, match="not a state"):
            _set(client, desc, "PENDING")
        assert client.get_upload_status(desc.array_id)["state"] == "READY"

    def test_a_discarded_upload_cannot_be_moved(self, writable_server, client, source):
        desc = _make(client, source, field="doomed")
        writable_server.uploads.discard(desc.array_id, "job died")

        with pytest.raises(UploadRefused, match="job died") as exc:
            _set(client, desc, "READY")
        assert exc.value.state == "DISCARDED"
        assert exc.value.reason == "job died"

    def test_a_published_upload_can_still_be_discarded(self, client, source):
        """Publishing a result does not make it permanent. Discard is reachable
        from every rung, which is what makes it the only delete."""
        desc = _make(client, source)
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "READY")

        assert _set(client, desc, "DISCARDED", "withdrawn")["state"] == "DISCARDED"
        with pytest.raises(flight.FlightError, match="withdrawn"):
            client.get_tensor(desc.array_id).compute()

    def test_climbing_something_that_is_not_an_upload_is_an_error(self, client):
        """Unlike DISCARDED, which is total: a climb that names nothing means
        the caller believes it has been writing somewhere it has not."""
        with pytest.raises(flight.FlightServerError, match="not an upload"):
            client.set_upload_status("no-such-source", "READY")

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

    def test_a_pending_upload_refuses_reads(self, client, source):
        desc = _make(client, source)
        _fill(client, desc)

        with pytest.raises(flight.FlightError, match="not readable yet"):
            client.get_tensor(desc.array_id).compute()

    def test_a_status_poll_answers_while_pending(self, client, source):
        """The gate is on the pixels, not on the descriptor -- polling to READY
        is what a consumer waiting on a result does."""
        desc = _make(client, source)
        _put(client, desc, (0, 0), (2, 2))

        desc = client.get_descriptor(desc.array_id, with_upload_status=True)
        assert desc.upload_status.state == 1  # PENDING
        assert desc.upload_status.uploaded_chunks == 1

    def test_ready_opens_it(self, client, source):
        desc = _make(client, source)
        _fill(client, desc)
        _set(client, desc, "READY")

        assert (np.asarray(client.get_tensor(desc.array_id)) == 7).all()

    def test_an_unwritten_chunk_reads_as_zeros(self, client, source):
        """Not an error: a producer that published a partial result meant the
        rest to be background."""
        desc = _make(client, source)
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "READY")

        arr = np.asarray(client.get_tensor(desc.array_id))
        assert (arr[:2, :2] == 7).all()
        assert (arr[2:, :] == 0).all()
        assert (arr[:, 2:] == 0).all()

    def test_a_read_gap_can_never_be_contradicted(
        self, writable_server, client, source
    ):
        """Why the two gates are one.

        A gap reads as zeros and that answer is cacheable -- here it is served
        from the cache the second time. Nothing may later fill the gap, because
        an entry already handed out under its ``chunk_id`` cannot be recalled,
        least of all from the client's own pool. The seal is what removes the
        case rather than racing it.
        """
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.core.chunk import mint_chunk_id
        from biopb_tensor_server.core.chunk_batch import unpack_chunk_array

        desc = _make(client, source)
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "READY")

        adapter = writable_server.sources.get(source).members["lifecycle"]
        bounds = ChunkBounds(start=[2, 2], stop=[4, 4])
        chunk_id = mint_chunk_id(
            adapter.array_id, bounds, content_version=adapter.content_version
        )
        cache = CacheManager.get_instance()
        assert not unpack_chunk_array(adapter.resolve_chunk_data(chunk_id, cache)).any()

        with pytest.raises(UploadRefused):
            _put(client, desc, (2, 2), (4, 4))
        assert not unpack_chunk_array(adapter.resolve_chunk_data(chunk_id, cache)).any()

    def test_a_discarded_source_reports_its_reason(
        self, writable_server, client, source
    ):
        """Disposal is the one terminal state a reader does meet: the adapter
        stays as a tombstone so the reason reaches it."""
        desc = _make(client, source)
        _put(client, desc, (0, 0), (2, 2))
        _set(client, desc, "READY")
        writable_server.uploads.discard(desc.array_id, "client went away")

        with pytest.raises(flight.FlightError, match="client went away"):
            client.get_tensor(desc.array_id).compute()

    def test_a_repeated_publish_changes_nothing_for_a_reader(self, client, source):
        """Setting READY again is a no-op all the way down, not a re-publish."""
        desc = _make(client, source)
        _fill(client, desc)
        _set(client, desc, "READY")
        before = np.asarray(client.get_tensor(desc.array_id))

        _set(client, desc, "READY")

        np.testing.assert_array_equal(
            before, np.asarray(client.get_tensor(desc.array_id))
        )
