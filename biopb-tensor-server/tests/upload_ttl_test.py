"""A tensor's lifetime: what ``add_tensor``'s ``ttl_seconds`` buys.

An upload is a temp store for an intermediate result, so how long the result is
worth keeping is the producer's to say. What that means here:

- a **deadline reaches a published tensor**. Idleness is PENDING's alone --
  past it nothing is expected to make progress -- but a lifetime that stopped
  applying at READY would be no lifetime, because READY is where a finished
  result spends its life;
- the deadline is **recorded with the store**, so it outlives the process that
  granted it. A deadline only the uploading server remembered would be none;
- a source may **cap** it (``max_upload_ttl``), including capping an unset
  request, so nothing lands on a scratch source forever by omission. The cap
  can only shorten, which is what makes it a policy rather than a default.

Nothing here changes a tensor with no deadline: it is kept until someone
discards it, which is what an upload onto a source the server discovered gets.
"""

import json
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb_tensor_server.adapters.fields import fields_root, source_fields_dir
from biopb_tensor_server.adapters.labels import labels_root, sidecar_dir
from biopb_tensor_server.adapters.members import MEMBER_DESCRIPTOR
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.adapter_base import catalog_tensors
from biopb_tensor_server.core.attached import attached_field

from tests import catalog_server

TTL = 10.0
SHAPE = (4, 4)
CHUNK = (2, 2)


def _add(client, source, field="temp", ttl=None, scheme="cache"):
    return client.add_tensor(
        f"{scheme}://{source}/@fields/{field}",
        np.empty(SHAPE, dtype=np.uint16),
        chunk_shape=CHUNK,
        ttl_seconds=ttl,
    )


def _publish(client, desc):
    client.upload_array(desc, np.full(SHAPE, 7, dtype=np.uint16))
    return desc


@pytest.fixture
def uploads(writable_server):
    """The manager with its sweep thread parked, so only the test sweeps."""
    manager = writable_server.uploads
    manager.stop_sweep()
    manager.ttl = TTL
    return manager


def _store(tmp_path, source, field="temp"):
    return source_fields_dir(fields_root(tmp_path), source) / field


def _marker(store):
    """The upload block the member recorded, whichever format it is in."""
    path = store / MEMBER_DESCRIPTOR
    if not path.exists():
        path = store / ".zattrs"
    return json.loads(path.read_text())["biopb"]["upload"]


class TestTheAnswerStatesWhatWasGranted:
    def test_a_requested_lifetime_comes_back(self, client, source):
        desc = _add(client, source, ttl=600)
        assert 0 < desc.ttl_seconds <= 600

    def test_no_request_means_no_deadline(self, client, source):
        """The default for a source that caps nothing: kept until discarded."""
        desc = _add(client, source)
        assert not desc.HasField("ttl_seconds")

    def test_zero_is_refused_rather_than_read_as_immediately(self, client, source):
        """Zero is not a lifetime, and guessing which way a caller meant it is
        how a result vanishes the moment it is made."""
        with pytest.raises(flight.FlightServerError, match="must be positive"):
            _add(client, source, ttl=0)


class TestTheDeadlineIsRecorded:
    @pytest.mark.parametrize("scheme", ["zarr", "cache"])
    def test_both_formats_write_it_beside_the_state(
        self, client, source, tmp_path, scheme
    ):
        _add(client, source, field=f"m-{scheme}", ttl=600, scheme=scheme)
        marker = _marker(_store(tmp_path, source, f"m-{scheme}"))

        assert marker["state"] == "pending"
        assert marker["expires_at"] > time.time()

    def test_publishing_does_not_drop_it(self, client, source, tmp_path):
        """``_publish_store`` rewrites the state over what is on disk, and the
        lifetime has to survive that -- READY is where it mostly applies."""
        desc = _add(client, source, ttl=600)
        recorded = _marker(_store(tmp_path, source))["expires_at"]
        _publish(client, desc)

        marker = _marker(_store(tmp_path, source))
        assert marker["state"] == "ready"
        assert marker["expires_at"] == recorded


class TestTheSweepEnforcesIt:
    def test_a_published_tensor_is_discarded_past_its_deadline(
        self, uploads, client, source, writable_server, tmp_path
    ):
        """The whole point. Idleness never reaches a READY upload; a deadline
        does."""
        desc = _publish(client, _add(client, source, ttl=600))
        store = _store(tmp_path, source)
        assert store.is_dir()

        assert uploads.reap(wall_now=time.time() + 601) == (1, 0)

        assert client.get_upload_status(desc.array_id)["state"] == "DISCARDED"
        assert "lifetime" in client.get_upload_status(desc.array_id)["reason"]
        assert not store.exists()
        assert catalog_tensors(writable_server.sources.get(source)) == []

    def test_one_without_a_deadline_survives(self, uploads, client, source):
        """A published result with no lifetime is its reader's, not the
        sweep's -- the behaviour every existing upload keeps."""
        desc = _publish(client, _add(client, source))

        assert uploads.reap(wall_now=time.time() + 10_000) == (0, 0)
        assert client.get_upload_status(desc.array_id)["state"] == "READY"

    def test_it_does_not_fire_early(self, uploads, client, source):
        desc = _publish(client, _add(client, source, ttl=600))

        assert uploads.reap(wall_now=time.time() + 599) == (0, 0)
        assert client.get_upload_status(desc.array_id)["state"] == "READY"

    def test_the_tombstone_ages_out_like_any_other(
        self, uploads, client, source, writable_server
    ):
        """An expired tensor is a discard, so it takes the same second sweep to
        free its name."""
        desc = _publish(client, _add(client, source, ttl=600))
        wall = time.time() + 601
        assert uploads.reap(wall_now=wall) == (1, 0)

        parent = writable_server.sources.get(source)
        assert attached_field("temp") in parent.attached_tensors

        assert uploads.reap(now=time.monotonic() + 2 * TTL, wall_now=wall) == (0, 1)
        assert attached_field("temp") not in parent.attached_tensors
        assert client.get_upload_status(desc.array_id)["state"] == "UNKNOWN"

    def test_a_zero_ttl_disables_deadlines_too(self, uploads, client, source):
        """``ttl <= 0`` means "do not reclaim on this server", and a deadline is
        reclamation."""
        desc = _publish(client, _add(client, source, ttl=600))
        uploads.ttl = 0

        assert uploads.reap(wall_now=time.time() + 10_000) == (0, 0)
        assert client.get_upload_status(desc.array_id)["state"] == "READY"


class TestALabelSetIsAnUploadedTensorToo:
    """A set takes a lifetime like a field, and a cap reaches it like a field.

    It is not a special case that gets to outlive the source's policy: if it
    were, uploading a set would be the way to leave something on a temp store
    for good.
    """

    @staticmethod
    def _image(client, source):
        desc = _publish(client, _add(client, source, field="img", scheme="zarr"))
        return desc

    @staticmethod
    def _labels(client, image, name="nuclei", ttl=None):
        arr = np.zeros(SHAPE, dtype=np.uint32)
        arr[:2, :2] = 3
        return client.add_tensor(
            f"zarr://{image.array_id}/@labels/{name}",
            arr,
            chunk_shape=CHUNK,
            ttl_seconds=ttl,
        )

    def test_a_requested_lifetime_reaches_a_set(self, client, source):
        image = self._image(client, source)

        assert self._labels(client, image, ttl=600).ttl_seconds <= 600

    def test_a_source_cap_fills_an_unset_request_on_a_set(
        self, client, source, writable_server
    ):
        """The hole this closes: the cap used to reach a field and not a set,
        so a set on a capped source was kept for good."""
        writable_server.sources.get(source).max_upload_ttl = 60
        image = self._image(client, source)

        assert 0 < self._labels(client, image).ttl_seconds <= 60

    def test_no_cap_and_no_request_still_means_no_deadline(self, client, source):
        """Unchanged for a set on a source that caps nothing."""
        image = self._image(client, source)

        assert not self._labels(client, image).HasField("ttl_seconds")

    def test_the_sweep_discards_an_expired_set(
        self, uploads, client, source, writable_server
    ):
        image = self._image(client, source)
        desc = self._labels(client, image, ttl=600)
        client.upload_array(desc, np.zeros(SHAPE, dtype=np.uint32))

        assert uploads.reap(wall_now=time.time() + 601)[0] == 1

        assert client.get_upload_status(desc.array_id)["state"] == "DISCARDED"
        assert "lifetime" in client.get_upload_status(desc.array_id)["reason"]
        assert desc.array_id not in writable_server.sources.get(source).label_sets

    def test_the_deadline_is_recorded_in_the_sidecar(
        self, client, source, writable_server, tmp_path
    ):
        """With the store, so it outlives the process -- which is also what
        lets the boot sweep take a set whose lifetime ran out while the server
        was down."""
        image = self._image(client, source)
        self._labels(client, image, ttl=600)

        sidecar = sidecar_dir(labels_root(tmp_path), source) / "nuclei.zarr"
        block = json.loads((sidecar / ".zattrs").read_text())["biopb"]["upload"]
        assert block["expires_at"] > time.time()


class TestASourceMayCapIt:
    def test_a_cap_shortens_a_longer_request(self, client, source, writable_server):
        writable_server.sources.get(source).max_upload_ttl = 60
        desc = _add(client, source, ttl=100_000)

        assert 0 < desc.ttl_seconds <= 60

    def test_a_cap_fills_an_unset_request(self, client, source, writable_server):
        """What stops a scratch source accumulating forever by omission."""
        writable_server.sources.get(source).max_upload_ttl = 60
        desc = _add(client, source)

        assert 0 < desc.ttl_seconds <= 60

    def test_a_shorter_request_is_kept(self, client, source, writable_server):
        """The cap is a ceiling, not the lifetime: a producer may still ask for
        less than a source allows."""
        writable_server.sources.get(source).max_upload_ttl = 10_000
        desc = _add(client, source, ttl=30)

        assert 0 < desc.ttl_seconds <= 30


class TestItSurvivesARestart:
    """A deadline only the uploading server remembered would be no deadline."""

    @staticmethod
    def _serve(tmp_path):
        server = catalog_server(
            location="grpc://localhost:0", writable=True, write_dir=Path(tmp_path)
        )
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        return server

    def test_a_live_deadline_is_read_back_and_still_enforced(
        self, writable_server, client, source, tmp_path
    ):
        """The adopted tensor tracks no upload -- the record died with the
        process that filled it -- so the sweep reads the deadline off the
        adapter, which read it off the marker."""
        _publish(client, _add(client, source, ttl=600))
        client.close()
        writable_server.shutdown()

        second = self._serve(tmp_path)
        try:
            second.uploads.stop_sweep()
            second.uploads.ttl = TTL
            adopted = second.sources.get(source)
            assert [d.array_id for d in catalog_tensors(adopted)] == [
                f"{source}/@fields/temp"
            ]

            # Expired *and* reclaimed in the one step: with no record there is
            # no tombstone to age out, so this sweep is the whole ending.
            assert second.uploads.reap(wall_now=time.time() + 601) == (1, 1)
            assert not _store(tmp_path, source).exists()
        finally:
            second.shutdown()
            CacheManager.reset()

    def test_an_adopted_tensor_leaves_no_tombstone_to_age_out(
        self, writable_server, client, source, tmp_path
    ):
        """A tombstone's age is ``UploadProgress.updated_at``, stamped at the
        discard. An adopted tensor has no record for that stamp to live on, so
        a tombstone there could never age out and the field would stay taken
        for good -- and it would buy nothing, since a tensor with no record
        already polls as UNKNOWN either way.

        So its expiry frees the name in one step, the way an explicit discard
        of an adopted tensor already does (``_delete_adopted_tensor``).
        """
        _publish(client, _add(client, source, ttl=600))
        client.close()
        writable_server.shutdown()

        second = self._serve(tmp_path)
        try:
            second.uploads.stop_sweep()
            second.uploads.ttl = TTL
            wall = time.time() + 601

            second.uploads.reap(wall_now=wall)

            parent = second.sources.get(source)
            assert attached_field("temp") not in parent.attached_tensors
            # ...and the sweep is done with it: no re-expiring it forever.
            assert second.uploads.reap(
                now=time.monotonic() + 100 * TTL, wall_now=wall
            ) == (0, 0)
        finally:
            second.shutdown()
            CacheManager.reset()

    def test_one_that_ran_out_while_down_is_never_served(
        self, writable_server, client, source, tmp_path
    ):
        """Swept at boot rather than adopted and reaped later: adopting it
        would serve expired bytes until the first sweep came round."""
        _publish(client, _add(client, source, ttl=1))
        store = _store(tmp_path, source)
        assert store.is_dir()
        client.close()
        writable_server.shutdown()

        # Backdate the recorded deadline rather than sleeping through it.
        path = store / MEMBER_DESCRIPTOR
        marker = json.loads(path.read_text())
        marker["biopb"]["upload"]["expires_at"] = time.time() - 1
        path.write_text(json.dumps(marker))

        second = self._serve(tmp_path)
        try:
            assert not store.exists()
            assert catalog_tensors(second.sources.get(source)) == []
        finally:
            second.shutdown()
            CacheManager.reset()
