"""The write plan: one planner, one ticket, both transports (step 4).

An upload no longer describes where its bytes go.
``GetFlightInfo`` plans the tensor exactly as it does for a read -- answered
while the upload is still PENDING, since planning is a metadata read -- and
DoPut carries back one of the plan's tickets and the batch for that chunk.

What that buys, and what is tested here: the client stops tiling, so it cannot
disagree with the server's grid; a partial or resumed upload asks for exactly
the tickets it needs; and a write inherits the version and scale checks a read
already has.
"""

import numpy as np
import pytest
from biopb.tensor import _upload
from biopb.tensor.descriptor_pb2 import SliceHint
from biopb.tensor.ticket_pb2 import ChunkBounds


def _make(client, source, field="plan", shape=(4, 4), chunk=(2, 2)):
    return client.add_tensor(
        f"cache://{source}/@fields/{field}",
        np.empty(shape, dtype=np.uint16),
        chunk_shape=chunk,
    )


def _plan(client, array_id, slice_hint=None):
    return _upload._plan_write(client._upload._state, array_id, slice_hint)


class TestThePlanIsTheGrid:
    def test_a_pending_tensor_is_planned_not_read(self, client, source):
        """Planning is a metadata read, so it is answered before READY.

        Nothing else could work: the plan is what the first write needs, and
        the upload cannot be published before it has one.
        """
        desc = _make(client, source)
        assert client.get_upload_status(desc.array_id)["state"] == "PENDING"

        plan = _plan(client, desc.array_id)
        assert [(c.start, c.stop) for c in plan] == [
            ((0, 0), (2, 2)),
            ((0, 2), (2, 4)),
            ((2, 0), (4, 2)),
            ((2, 2), (4, 4)),
        ]

    def test_the_write_and_the_read_hold_the_same_tickets(self, client, source):
        """One planner and one ticket: DoPut takes it while PENDING, DoGet
        takes the very same bytes once READY."""
        desc = _make(client, source)
        writing = {c.ticket for c in _plan(client, desc.array_id)}
        client.upload_array(desc, np.zeros((4, 4), np.uint16))
        reading = {c.ticket for c in _plan(client, desc.array_id)}
        assert writing == reading

    def test_a_slice_asks_for_only_the_chunks_it_touches(self, client, source):
        """The point of the hint: a partial or resumed upload plans its box."""
        desc = _make(client, source)
        plan = _plan(client, desc.array_id, SliceHint(start=[0, 0], stop=[2, 2]))
        assert [(c.start, c.stop) for c in plan] == [((0, 0), (2, 2))]

    def test_a_slice_is_snapped_outward_to_whole_chunks(self, client, source):
        """A chunk is the unit, so a box inside one plans that whole one."""
        desc = _make(client, source)
        plan = _plan(client, desc.array_id, SliceHint(start=[1, 1], stop=[2, 2]))
        assert [(c.start, c.stop) for c in plan] == [((0, 0), (2, 2))]


class TestUploadChunkTakesAChunk:
    def test_bounds_that_are_not_a_chunk_are_refused_with_the_grid(
        self, client, source
    ):
        """Refused by the client, before anything is sent: a write of part of a
        chunk has nowhere to land, and the message names the cell it falls in
        rather than leaving the caller to guess the grid."""
        desc = _make(client, source)
        with pytest.raises(ValueError, match=r"is not one chunk.*\[0, 0\]-\[2, 2\]"):
            client.upload_chunk(
                desc,
                ChunkBounds(start=[0, 0], stop=[1, 1]),
                np.ones((1, 1), np.uint16),
            )
        assert client.get_upload_status(desc.array_id)["uploaded_chunks"] == 0

    def test_a_box_spanning_two_chunks_is_refused_too(self, client, source):
        desc = _make(client, source)
        with pytest.raises(ValueError, match="is not one chunk"):
            client.upload_chunk(
                desc,
                ChunkBounds(start=[0, 0], stop=[4, 2]),
                np.ones((4, 2), np.uint16),
            )

    def test_a_chunk_of_the_grid_lands(self, client, source):
        desc = _make(client, source)
        client.upload_chunk(
            desc, ChunkBounds(start=[2, 2], stop=[4, 4]), np.full((2, 2), 9, np.uint16)
        )
        client.set_upload_status(desc, "READY")
        read = client.get_tensor(desc.array_id).compute()
        assert read[2:, 2:].tolist() == [[9, 9], [9, 9]]
        assert read[:2, :2].tolist() == [[0, 0], [0, 0]]  # never written: zeros


class TestUploadArrayOnAPlan:
    def test_the_whole_array_is_sealed(self, client, source):
        desc = _make(client, source)
        source = np.arange(16, dtype=np.uint16).reshape(4, 4)
        status = client.upload_array(desc, source)
        assert (status["state"], status["uploaded_chunks"]) == ("READY", 4)
        np.testing.assert_array_equal(
            client.get_tensor(desc.array_id).compute(), source
        )

    def test_a_slice_hint_uploads_that_region_and_does_not_seal(self, client, source):
        """A region cannot know the upload is done, so it leaves it PENDING;
        the caller publishes when it has written everything it means to."""
        desc = _make(client, source)
        source = np.arange(16, dtype=np.uint16).reshape(4, 4)
        status = client.upload_array(
            desc, source, slice_hint=(slice(0, 2), slice(0, 2))
        )
        assert (status["state"], status["uploaded_chunks"]) == ("PENDING", 1)

        client.set_upload_status(desc, "READY")
        read = client.get_tensor(desc.array_id).compute()
        np.testing.assert_array_equal(read[:2, :2], source[:2, :2])
        # The rest was never written, so it reads back as the gap it is.
        assert read[2:, 2:].max() == 0

    def test_an_open_ended_slice_stop_comes_from_the_declared_shape(
        self, client, source
    ):
        """The upload path already holds the shape, so unlike a read this
        costs no resolve."""
        desc = _make(client, source)
        source = np.arange(16, dtype=np.uint16).reshape(4, 4)
        status = client.upload_array(
            desc, source, slice_hint=(slice(2, None), slice(None))
        )
        assert status["uploaded_chunks"] == 2

    def test_a_slice_hint_of_the_wrong_rank_is_refused(self, client, source):
        desc = _make(client, source)
        with pytest.raises(ValueError, match="slice_hint has 1 axes"):
            client.upload_array(
                desc, np.zeros((4, 4), np.uint16), slice_hint=(slice(0, 2),)
            )

    def test_the_two_halves_together_fill_the_tensor(self, client, source):
        """Resuming: two plans, disjoint, one tensor."""
        desc = _make(client, source)
        source = np.arange(16, dtype=np.uint16).reshape(4, 4)
        client.upload_array(desc, source, slice_hint=(slice(0, 2), slice(None)))
        client.upload_array(desc, source, slice_hint=(slice(2, 4), slice(None)))
        client.set_upload_status(desc, "READY")
        np.testing.assert_array_equal(
            client.get_tensor(desc.array_id).compute(), source
        )
