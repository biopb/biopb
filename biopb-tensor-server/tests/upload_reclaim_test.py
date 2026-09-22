"""Reclamation: a sweep over every upload by ``updated_at``, with one TTL for
its three halves.

- a PENDING upload with no write for ``ttl`` seconds is **discarded** -- the job
  that owned it died -- with a reason, so a straggler's write is refused the
  way it would be after an explicit discard;
- a tombstone that has stood for ``ttl`` seconds is **detached**: its status
  reads UNKNOWN and its field is free again;
- a registered source that has been **empty** for ``ttl`` seconds goes too,
  which is what stops an abandoned ``register_source`` leaving a directory and
  a catalog row nothing can reach.

Finished uploads are published results and are never reclaimed. A store on disk
is swept like any other upload: it and the catalog listing were the server's
own (biopb/biopb#1059), so a discard takes both with it.
"""

import time

import numpy as np
import pytest
from biopb.tensor import UploadRefused
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters.fields import (
    fields_root,
    source_fields_dir,
)
from biopb_tensor_server.core.adapter_base import catalog_tensors
from biopb_tensor_server.core.attached import attached_field

TTL = 10.0


def _make(client, source, field="reclaim", shape=(4, 4), chunk=(2, 2)):
    return client.add_tensor(
        f"cache://{source}/@fields/{field}",
        np.empty(shape, dtype=np.uint16),
        chunk_shape=chunk,
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
    def test_a_pending_upload_with_no_writes_is_discarded(
        self, uploads, client, source
    ):
        desc = _make(client, source)

        assert uploads.reap(now=_past_ttl()) == (1, 0)

        status = client.get_upload_status(desc.array_id)
        assert status["state"] == "DISCARDED"
        assert "expired" in status["reason"]

    def test_a_late_write_is_refused_with_the_expiry_reason(
        self, uploads, client, source
    ):
        """The straggler learns the same way it would after an explicit discard."""
        desc = _make(client, source)
        uploads.reap(now=_past_ttl())

        with pytest.raises(UploadRefused, match="expired") as exc:
            _put(client, desc)
        assert exc.value.state == "DISCARDED"

    def test_a_write_within_the_ttl_keeps_it_alive(self, uploads, client, source):
        desc = _make(client, source)
        _put(client, desc)

        assert uploads.reap(now=time.monotonic() + TTL / 2) == (0, 0)
        assert client.get_upload_status(desc.array_id)["state"] == "PENDING"

    def test_a_published_upload_is_never_expired(self, uploads, client, source):
        """Past PENDING the upload is a published result, and its lifetime is
        its reader's rather than its writer's."""
        desc = _make(client, source)
        _put(client, desc)
        client.set_upload_status(desc, "READY")

        assert uploads.reap(now=_past_ttl()) == (0, 0)
        assert client.get_upload_status(desc.array_id)["state"] == "READY"

    def test_a_quiet_zarr_upload_is_discarded_and_its_store_removed(
        self, uploads, client, writable_server, source, tmp_path
    ):
        """A member's store was the server's own, so the sweep disposes of it
        and takes the tensor out of its source's listing (biopb/biopb#1059)."""
        desc = client.add_tensor(
            f"zarr://{source}/@fields/quiet",
            np.empty((4, 4), dtype=np.uint16),
            chunk_shape=(2, 2),
            dim_labels=["y", "x"],
        )
        parent = writable_server.sources.get(source)
        store = source_fields_dir(fields_root(tmp_path), source) / "quiet"
        assert store.is_dir()

        assert uploads.reap(now=_past_ttl()) == (1, 0)

        status = client.get_upload_status(desc.array_id)
        assert status["state"] == "DISCARDED"
        assert "expired" in status["reason"]
        assert not store.exists()
        assert catalog_tensors(parent) == []

        assert uploads.reap(now=_past_ttl(margin=2 * TTL)) == (0, 1)
        assert attached_field("quiet") not in parent.attached_tensors

    def test_a_zero_ttl_disables_the_sweep(self, uploads, client, source):
        desc = _make(client, source)
        uploads.ttl = 0

        assert uploads.reap(now=_past_ttl()) == (0, 0)
        assert client.get_upload_status(desc.array_id)["state"] == "PENDING"


class TestATombstoneIsReclaimed:
    def test_an_aged_tombstone_is_detached(
        self, uploads, client, writable_server, source
    ):
        desc = _make(client, source)
        uploads.discard(desc.array_id, "gone")

        assert uploads.reap(now=_past_ttl()) == (0, 1)

        assert (
            attached_field("reclaim")
            not in writable_server.sources.get(source).attached_tensors
        )
        assert client.get_upload_status(desc.array_id)["state"] == "UNKNOWN"

    def test_a_fresh_tombstone_still_answers(self, uploads, client, source):
        """Until the TTL passes, a straggler still gets the reason."""
        desc = _make(client, source)
        uploads.discard(desc.array_id, "gone")

        assert uploads.reap(now=time.monotonic() + TTL / 2) == (0, 0)

        status = client.get_upload_status(desc.array_id)
        assert status["state"] == "DISCARDED"
        assert status["reason"] == "gone"

    def test_an_expired_upload_takes_two_sweeps_to_vanish(
        self, uploads, client, source
    ):
        """Expiry is a discard, so the tombstone stands for a TTL of its own."""
        desc = _make(client, source)
        first = _past_ttl()

        assert uploads.reap(now=first) == (1, 0)
        assert uploads.reap(now=first) == (0, 0)
        assert client.get_upload_status(desc.array_id)["state"] == "DISCARDED"

        assert uploads.reap(now=first + TTL + 1) == (0, 1)
        assert client.get_upload_status(desc.array_id)["state"] == "UNKNOWN"

    def test_reclaiming_frees_the_field(self, uploads, client, source):
        old = _make(client, source, field="again")
        _put(client, old, fill=1)
        uploads.discard(old.array_id, "gone")
        uploads.reap(now=_past_ttl())

        new = _make(client, source, field="again")

        assert new.array_id == old.array_id
        assert client.get_upload_status(new.array_id)["state"] == "PENDING"

    def test_the_old_bytes_do_not_resurface_under_the_reused_field(
        self, uploads, client, source
    ):
        """The tombstone's chunks are still in the cache; the re-created tensor
        writes and reads in a fresh content_version namespace, so it cannot
        see them."""
        old = _make(client, source, field="again")
        _put(client, old, fill=1)
        uploads.discard(old.array_id, "gone")
        uploads.reap(now=_past_ttl())
        new = _make(client, source, field="again")
        _put(client, new, fill=2)
        client.set_upload_status(new, "READY")

        # 2s, never the tombstone's 1s: the chunks it left in the cache sit in
        # a namespace this tensor's chunk_ids cannot name.
        assert (client.get_tensor(new.array_id)[:2, :2].compute() == 2).all()


class TestAnEmptySourceIsReclaimed:
    """An abandoned ``register_source`` leaves nothing behind.

    The source is on the same clock as the tensors it holds, measured from the
    last time one arrived or left -- so a create nobody ever added to, and a
    source whose last tensor was discarded and then reclaimed, go the same way.
    """

    def test_a_source_that_never_received_a_tensor_goes(
        self, uploads, client, writable_server
    ):
        source = client.register_source()
        store = writable_server.sources.get(source).store
        assert store.is_dir()

        assert uploads.reap(now=_past_ttl()) == (0, 1)

        assert source not in writable_server.sources
        assert not store.exists()
        rows = writable_server.metadata_db.query("SELECT source_id FROM sources")
        assert source not in rows.column(0).to_pylist()

    def test_a_source_still_being_filled_is_never_taken(
        self, uploads, client, writable_server, source
    ):
        """Pending counts as live: an upload in flight never has the source
        pulled out from under it, however long the filling takes."""
        _make(client, source)

        assert uploads.reap(now=time.monotonic() + TTL / 2) == (0, 0)
        assert source in writable_server.sources

    def test_the_source_outlives_the_tombstone_it_still_holds(
        self, uploads, client, writable_server, source
    ):
        """Two sweeps, never one: a discarded tensor is still reachable to
        whoever is polling it, so its source has to outlive it."""
        desc = _make(client, source)
        uploads.discard(desc.array_id, "gone")

        # The tombstone goes; the source is touched by losing it and stays.
        assert uploads.reap(now=_past_ttl()) == (0, 1)
        assert source in writable_server.sources

        assert uploads.reap(now=_past_ttl(margin=2 * TTL)) == (0, 1)
        assert source not in writable_server.sources


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

    def test_the_thread_reaps_on_its_own(self, uploads, client, source):
        """The one timing test: a sub-second TTL and a bounded wait."""
        desc = _make(client, source)
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
