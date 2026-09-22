"""The scratch source: somewhere to put an intermediate result, at a fixed id.

An upload is a temp store. Getting one used to take a round trip -- mint a
container, remember what came back, and hope something reaps it -- which is a
lot of ceremony for a scrap heap, and ceremony that quietly produced permanent
sources. A writable server offers one scratch source instead, always there,
always at ``scratch``, so a producer writes ``zarr://scratch/@fields/<name>``
without asking for anything first.

What is pinned here is what makes that safe to share:

- it is **there before anything can ask**, and only where a tensor can actually
  be put;
- it keeps **no directory of its own**, so what comes back after a restart
  comes back through the pass that re-attaches every source's uploaded fields;
- it is **not reclaimable**: empty is its resting state, not a fault;
- it **caps every lifetime** on it (``ServerConfig.scratch_ttl``), which is the
  one thing that keeps a shared scrap heap from becoming a shared permanent
  store.
"""

import threading
from pathlib import Path

import numpy as np
import pytest
from biopb.tensor import TensorFlightClient
from biopb_tensor_server.adapters.scratch import SCRATCH_SOURCE_ID
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.adapter_base import catalog_tensors

from tests import catalog_server

SHAPE = (4, 4)
CHUNK = (2, 2)


def _serve(tmp_path, **kwargs):
    server = catalog_server(
        location="grpc://localhost:0", write_dir=Path(tmp_path), **kwargs
    )
    server.mark_ready()
    threading.Thread(target=server.serve, daemon=True).start()
    return server


def _add(client, field, ttl=None, scheme="cache"):
    return client.add_tensor(
        f"{scheme}://{SCRATCH_SOURCE_ID}/@fields/{field}",
        np.empty(SHAPE, dtype=np.uint16),
        chunk_shape=CHUNK,
        ttl_seconds=ttl,
    )


def _publish(client, desc):
    client.upload_array(desc, np.full(SHAPE, 7, dtype=np.uint16))
    return desc


class TestItIsThereBeforeAnythingAsks:
    def test_a_writable_server_serves_it(self, writable_server):
        assert writable_server.sources.get(SCRATCH_SOURCE_ID) is not None

    def test_it_is_in_the_catalog(self, writable_server):
        """Registered *and* catalogued: the two steps a source needs to be both
        readable and browsable."""
        assert writable_server.metadata_db.source_row_ipc(SCRATCH_SOURCE_ID) is not None

    def test_a_read_only_server_has_none(self, tmp_path):
        """A source nothing can be added to would be a listing with no use.
        ``writable`` is the switch, not ``write_dir``."""
        server = _serve(tmp_path, writable=False)
        try:
            assert server.sources.get(SCRATCH_SOURCE_ID) is None
        finally:
            server.shutdown()
            CacheManager.reset()

    def test_an_upload_needs_no_round_trip_first(self, client):
        """The whole point: the id is a constant, so a producer writes it down
        rather than asking for it."""
        desc = _publish(client, _add(client, "straight-in"))

        assert desc.array_id == f"{SCRATCH_SOURCE_ID}/@fields/straight-in"
        assert client.get_upload_status(desc.array_id)["state"] == "READY"


class TestItIsEmptyByDefault:
    def test_an_empty_one_is_a_source_with_no_tensors(self, writable_server):
        """Not a degenerate state: it is what a scrap heap looks like between
        uploads, and the catalog already models it."""
        adapter = writable_server.sources.get(SCRATCH_SOURCE_ID)

        assert catalog_tensors(adapter) == []
        assert adapter.get_metadata() == {}

    def test_the_sweep_leaves_it_alone(self, writable_server, client):
        """A registered source was dropped once it stood empty, because an
        abandoned one left a directory and a row nothing would reach. This has
        neither, so nothing reclaims it."""
        desc = _publish(client, _add(client, "brief"))
        client.set_upload_status(desc.array_id, "DISCARDED")

        uploads = writable_server.uploads
        uploads.reap(now=uploads.ttl * 10)
        uploads.reap(now=uploads.ttl * 20)

        assert writable_server.sources.get(SCRATCH_SOURCE_ID) is not None

    def test_asking_an_empty_one_for_a_tensor_is_a_miss_not_an_error(self, client):
        """A resolution miss: asking a source with no tensors for its tensor is
        a caller's mistake about what it holds, not a server fault."""
        with pytest.raises(Exception, match="no tensors yet"):
            client.get_descriptor(SCRATCH_SOURCE_ID)


class TestItKeepsNoDirectoryOfItsOwn:
    def test_its_fields_come_back_after_a_restart(
        self, writable_server, client, tmp_path
    ):
        """Through the pass that re-attaches every source's uploaded fields --
        there is no container to walk, so that pass is the whole of adoption."""
        _publish(client, _add(client, "survivor"))
        client.close()
        writable_server.shutdown()

        second = _serve(tmp_path, writable=True)
        try:
            assert [
                d.array_id
                for d in catalog_tensors(second.sources.get(SCRATCH_SOURCE_ID))
            ] == [f"{SCRATCH_SOURCE_ID}/@fields/survivor"]
        finally:
            second.shutdown()
            CacheManager.reset()

    def test_its_tensors_live_where_every_uploaded_field_does(self, client, tmp_path):
        _publish(client, _add(client, "somewhere"))

        assert (tmp_path / "fields" / SCRATCH_SOURCE_ID / "somewhere").is_dir()


class TestItCapsEveryLifetimeOnIt:
    def test_an_upload_that_asked_for_nothing_still_gets_a_deadline(self, client):
        """What keeps a shared scrap heap from becoming a shared permanent
        store: the default is the cap, not "forever"."""
        desc = _add(client, "undated")

        assert 0 < desc.ttl_seconds <= 86400

    def test_the_cap_comes_from_the_config(self, tmp_path):
        server = _serve(tmp_path, writable=True, scratch_ttl=60)
        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")
            desc = _add(client, "capped")
            assert 0 < desc.ttl_seconds <= 60
            client.close()
        finally:
            server.shutdown()
            CacheManager.reset()

    def test_zero_turns_the_cap_off(self, tmp_path):
        """The knob for a deployment that wants uploads kept until discarded --
        which makes this a permanent store, and is why it is not the default."""
        server = _serve(tmp_path, writable=True, scratch_ttl=0)
        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")
            desc = _add(client, "forever")
            assert not desc.HasField("ttl_seconds")
            client.close()
        finally:
            server.shutdown()
            CacheManager.reset()

    def test_a_shorter_request_is_still_honoured(self, client):
        """A ceiling, not an assignment."""
        desc = _add(client, "brief", ttl=30)

        assert 0 < desc.ttl_seconds <= 30
