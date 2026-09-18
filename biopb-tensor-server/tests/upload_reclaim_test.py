"""Reclamation (biopb/biopb#1048 step 6): a sweep over the registry by
``updated_at``, with one TTL for its two halves.

- a PENDING upload with no write for ``ttl`` seconds is **discarded** -- the job
  that owned it died -- with a reason, so a straggler's write is refused the
  way it would be after an explicit discard;
- a tombstone that has stood for ``ttl`` seconds is **unregistered**: its
  status reads UNKNOWN and its name is free again.

Finished uploads are published results and are never reclaimed. A durable kind
is swept like any other: the store and the catalog row were the server's own
(biopb/biopb#1059), so a discard takes both with it.
"""

import time

import numpy as np
import pytest
from biopb.tensor import UploadRefused
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

TTL = 10.0


def _make(client, name="cache:reclaim", shape=(4, 4), chunk=(2, 2)):
    return client.create_tensor(
        name, np.empty(shape, dtype=np.uint16), chunk_shape=chunk
    )


def _put(client, desc, start=(0, 0), stop=(2, 2), fill=7):
    data = np.full(
        [b - a for a, b in zip(start, stop, strict=True)], fill, dtype=np.uint16
    )
    client.upload_chunk(desc, ChunkBounds(start=list(start), stop=list(stop)), data)


@pytest.fixture
def uploads(writable_server):
    """The manager with a known TTL, and its thread parked so only the test sweeps."""
    manager = writable_server.uploads
    manager.stop_sweep()
    manager.ttl = TTL
    return manager


def _past_ttl(margin=1.0):
    """A ``now`` at which anything touched before this call is past the TTL."""
    return time.monotonic() + TTL + margin


class TestAQuietUploadExpires:
    def test_a_pending_upload_with_no_writes_is_discarded(self, uploads, client):
        desc = _make(client)

        assert uploads.reap(now=_past_ttl()) == (1, 0)

        status = client.get_upload_status(desc.array_id)
        assert status["state"] == "DISCARDED"
        assert "expired" in status["reason"]

    def test_a_late_write_is_refused_with_the_expiry_reason(self, uploads, client):
        """The straggler learns the same way it would after an explicit discard."""
        desc = _make(client)
        uploads.reap(now=_past_ttl())

        with pytest.raises(UploadRefused, match="expired") as exc:
            _put(client, desc)
        assert exc.value.state == "DISCARDED"

    def test_a_write_within_the_ttl_keeps_it_alive(self, uploads, client):
        desc = _make(client)
        _put(client, desc)

        assert uploads.reap(now=time.monotonic() + TTL / 2) == (0, 0)
        assert client.get_upload_status(desc.array_id)["state"] == "PENDING"

    def test_a_finished_upload_is_never_expired(self, uploads, client):
        """READY is a published result; its lifetime is its reader's."""
        desc = _make(client)
        _put(client, desc)
        client.finish_upload(desc)

        assert uploads.reap(now=_past_ttl()) == (0, 0)
        assert client.get_upload_status(desc.array_id)["state"] == "READY"

    def test_a_quiet_durable_upload_is_discarded_and_its_store_removed(
        self, uploads, client, writable_server, tmp_path
    ):
        """A zarr upload's store and catalog row were the server's own, so the
        sweep disposes of both (biopb/biopb#1059)."""
        desc = uploads.create_tensor(
            TensorDescriptor(
                array_id="ome_zarr:quiet",
                shape=[4, 4],
                dtype="uint16",
                chunk_shape=[2, 2],
                dim_labels=["y", "x"],
            )
        )
        assert (tmp_path / "quiet.zarr").is_dir()

        assert uploads.reap(now=_past_ttl()) == (1, 0)

        status = client.get_upload_status(desc.array_id)
        assert status["state"] == "DISCARDED"
        assert "expired" in status["reason"]
        assert not (tmp_path / "quiet.zarr").exists()
        rows = writable_server.metadata_db.query("SELECT source_id FROM sources")
        assert desc.array_id not in rows.column(0).to_pylist()

        assert uploads.reap(now=_past_ttl(margin=2 * TTL)) == (0, 1)
        assert desc.array_id not in writable_server.sources

    def test_a_zero_ttl_disables_the_sweep(self, uploads, client):
        desc = _make(client)
        uploads.ttl = 0

        assert uploads.reap(now=_past_ttl()) == (0, 0)
        assert client.get_upload_status(desc.array_id)["state"] == "PENDING"


class TestATombstoneIsReclaimed:
    def test_an_aged_tombstone_is_unregistered(self, uploads, client, writable_server):
        desc = _make(client)
        uploads.discard(desc.array_id, "gone")

        assert uploads.reap(now=_past_ttl()) == (0, 1)

        assert desc.array_id not in writable_server.sources
        assert client.get_upload_status(desc.array_id)["state"] == "UNKNOWN"

    def test_a_fresh_tombstone_still_answers(self, uploads, client):
        """Until the TTL passes, a straggler still gets the reason."""
        desc = _make(client)
        uploads.discard(desc.array_id, "gone")

        assert uploads.reap(now=time.monotonic() + TTL / 2) == (0, 0)

        status = client.get_upload_status(desc.array_id)
        assert status["state"] == "DISCARDED"
        assert status["reason"] == "gone"

    def test_an_expired_upload_takes_two_sweeps_to_vanish(self, uploads, client):
        """Expiry is a discard, so the tombstone stands for a TTL of its own."""
        desc = _make(client)
        first = _past_ttl()

        assert uploads.reap(now=first) == (1, 0)
        assert uploads.reap(now=first) == (0, 0)
        assert client.get_upload_status(desc.array_id)["state"] == "DISCARDED"

        assert uploads.reap(now=first + TTL + 1) == (0, 1)
        assert client.get_upload_status(desc.array_id)["state"] == "UNKNOWN"

    def test_reclaiming_frees_the_name(self, uploads, client):
        old = _make(client, name="cache:again")
        _put(client, old, fill=1)
        uploads.discard(old.array_id, "gone")
        uploads.reap(now=_past_ttl())

        new = _make(client, name="cache:again")

        assert new.array_id == old.array_id
        assert client.get_upload_status(new.array_id)["state"] == "PENDING"

    def test_the_old_bytes_do_not_resurface_under_the_reused_name(
        self, uploads, client
    ):
        """The tombstone's chunks are still in the cache; the re-created source
        writes and reads in a fresh content_version namespace, so it cannot
        see them."""
        old = _make(client, name="cache:again")
        _put(client, old, fill=1)
        uploads.discard(old.array_id, "gone")
        uploads.reap(now=_past_ttl())
        new = _make(client, name="cache:again")

        with pytest.raises(Exception, match="holds no chunk"):
            client.get_tensor(new.array_id)[:2, :2].compute()

        _put(client, new, fill=2)
        assert (client.get_tensor(new.array_id)[:2, :2].compute() == 2).all()


class TestTheSweepThread:
    def test_the_server_starts_the_sweep(self, writable_server):
        assert writable_server.uploads._thread is not None

    def test_stop_is_idempotent(self, uploads):
        uploads.stop_sweep()
        uploads.stop_sweep()
        assert uploads._thread is None

    def test_a_zero_ttl_starts_no_thread(self, uploads):
        uploads.ttl = 0
        uploads.start_sweep()
        assert uploads._thread is None

    def test_the_thread_reaps_on_its_own(self, uploads, client):
        """The one timing test: a sub-second TTL and a bounded wait."""
        desc = _make(client)
        uploads.discard(desc.array_id, "gone")
        uploads.ttl = 0.2
        uploads.start_sweep()
        try:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if client.get_upload_status(desc.array_id)["state"] == "UNKNOWN":
                    break
                time.sleep(0.1)
            assert client.get_upload_status(desc.array_id)["state"] == "UNKNOWN"
        finally:
            uploads.stop_sweep()
