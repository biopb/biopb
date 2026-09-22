"""Registered sources: minted on request, adopted at boot.

The gap this closes (biopb/biopb#1048): a store the upload path minted was
registered only in the life that created it. Its catalog row persisted, so
after a restart the row was browsable with nothing behind it. Nothing was
missing from *discovery* -- ``write_dir`` is outside every discovery root by
rule, and stays there -- what was missing was the adoption pass.

So the properties here are: the id is the server's and is recorded, not
derived from the path; the container comes back under that id in the next life;
and a directory that records no id is not a source at all. What it holds is not
in it -- its tensors are uploaded fields, attached at registration like any
other source's.
"""

import json
from pathlib import Path

import pytest
from biopb_tensor_server.adapters._writable import folded_match
from biopb_tensor_server.adapters.registered import (
    create_registered_source,
    read_source_block,
    scan_registered_sources,
    sources_root,
)
from biopb_tensor_server.core.adapter_base import catalog_tensors
from biopb_tensor_server.core.attached import attached_field


@pytest.fixture
def sources_dir(tmp_path):
    return sources_root(tmp_path / "w")


class TestTheIdIsRecorded:
    """Minted here and written into the store, so nothing has to re-derive it
    -- which is what lets ``write_dir`` move."""

    def test_the_store_carries_the_id_it_was_registered_under(self, sources_dir):
        adapter = create_registered_source("plate", None, sources_dir)

        block = read_source_block(adapter.store)
        assert block["source_id"] == adapter.source_id
        assert block["content_version"] == adapter.content_version

    def test_the_id_is_not_derived_from_the_name(self, sources_dir, tmp_path):
        """Two sources of the same name in different write_dirs get different
        ids, and one source keeps its id wherever its directory is moved."""
        first = create_registered_source("plate", None, sources_dir)
        second = create_registered_source(
            "plate", None, sources_root(tmp_path / "other")
        )
        assert first.source_id != second.source_id

    def test_a_moved_write_dir_keeps_the_id(self, sources_dir, tmp_path):
        adapter = create_registered_source("plate", None, sources_dir)
        original = adapter.source_id

        moved = tmp_path / "moved" / "sources"
        moved.parent.mkdir(parents=True)
        sources_dir.rename(moved)

        assert list(scan_registered_sources(moved)) == [original]

    def test_the_content_version_is_minted_not_sampled(self, sources_dir):
        """Two sources registered from the same bytes still differ: the token
        separates registrations, it is not a change signal."""
        a = create_registered_source("a", None, sources_dir)
        b = create_registered_source("b", None, sources_dir)
        assert a.content_version != b.content_version


class TestBootAdoption:
    def test_a_registered_source_comes_back(self, sources_dir):
        adapter = create_registered_source("plate", {"omero": {"n": 1}}, sources_dir)

        adopted = scan_registered_sources(sources_dir)

        assert list(adopted) == [adapter.source_id]
        revived = adopted[adapter.source_id]
        assert revived.content_version == adapter.content_version
        assert revived.get_metadata() == {"omero": {"n": 1}}

    def test_an_empty_source_is_a_source(self, sources_dir):
        """A collection with no member is resolved with an empty tensor list,
        which the catalog already models -- an unresolved source carries one."""
        adapter = create_registered_source("empty", None, sources_dir)

        revived = scan_registered_sources(sources_dir)[adapter.source_id]
        assert revived.is_resolved()
        assert revived.list_tensor_descriptors() == []

    def test_a_directory_with_no_recorded_id_is_removed(self, sources_dir):
        """It has no identity to adopt, so leaving it would accumulate bytes
        nothing can reach or name -- the same call a sidecar that lost its
        token gets, one step harsher because a collection has no parent to
        fall back on."""
        sources_dir.mkdir(parents=True)
        stray = sources_dir / "stray.zarr"
        stray.mkdir()
        (stray / ".zattrs").write_text(json.dumps({"omero": {}}))

        assert scan_registered_sources(sources_dir) == {}
        assert not stray.exists()

    def test_an_unparseable_token_is_removed_too(self, sources_dir):
        adapter = create_registered_source("plate", None, sources_dir)
        zattrs = json.loads((adapter.store / ".zattrs").read_text())
        zattrs["biopb"]["source"]["content_version"] = "not-hex"
        (adapter.store / ".zattrs").write_text(json.dumps(zattrs))

        assert scan_registered_sources(sources_dir) == {}
        assert not adapter.store.exists()

    def test_nothing_to_adopt_is_not_an_error(self, sources_dir):
        assert scan_registered_sources(sources_dir) == {}

    def test_two_directories_recording_one_id_keep_the_first(self, sources_dir):
        """Guessing which is current would be a coin flip, so the second is
        skipped rather than served or deleted."""
        first = create_registered_source("a", None, sources_dir)
        second = create_registered_source("b", None, sources_dir)
        zattrs = json.loads((second.store / ".zattrs").read_text())
        zattrs["biopb"]["source"]["source_id"] = first.source_id
        (second.store / ".zattrs").write_text(json.dumps(zattrs))

        adopted = scan_registered_sources(sources_dir)

        assert list(adopted) == [first.source_id]
        assert adopted[first.source_id].store == first.store
        assert second.store.exists()


class TestTheName:
    def test_an_unportable_name_is_refused(self, sources_dir):
        """Refused before anything is written: the name is checked ahead of
        the directory it would become."""
        with pytest.raises(ValueError, match="device name"):
            create_registered_source("CON", None, sources_dir)
        assert not sources_dir.exists()

    def test_a_case_variant_of_a_taken_name_is_refused(self, sources_dir):
        create_registered_source("Plate", None, sources_dir)
        with pytest.raises(ValueError, match="already exists"):
            create_registered_source("plate", None, sources_dir)

    def test_an_empty_name_is_minted(self, sources_dir):
        adapter = create_registered_source("", None, sources_dir)
        assert adapter.store.exists()
        assert adapter.source_id


class TestTheSourceSurface:
    def test_an_empty_source_has_no_default_tensor(self, sources_dir):
        """Named as a resolution miss, not a server error: asking a source with
        no tensors for its tensor is a mistake about what it holds."""
        from biopb_tensor_server.core.errors import TensorNotFound

        adapter = create_registered_source("plate", None, sources_dir)
        with pytest.raises(TensorNotFound, match="no tensors yet"):
            adapter.get_tensor_adapter(None)

    def test_a_field_is_listed_and_addressable(self, sources_dir):
        """Listed through ``catalog_tensors``, which is where an attached
        tensor joins a source's listing: the container has none of its own."""
        adapter = create_registered_source("plate", None, sources_dir)
        field = attached_field("img")
        member = _FakeMember(f"{adapter.source_id}/{field}")
        adapter.attach_tensor(field, member)

        assert [d.array_id for d in catalog_tensors(adapter)] == [
            f"{adapter.source_id}/{field}"
        ]
        assert adapter.resolve_tensor(f"{adapter.source_id}/{field}") is member
        assert adapter.resolve_tensor(field) is member
        # With no field, the source's default is its first published one.
        assert adapter.get_tensor_adapter(None) is member

    def test_an_unknown_field_is_a_resolution_miss(self, sources_dir):
        from biopb_tensor_server.core.errors import TensorNotFound

        adapter = create_registered_source("plate", None, sources_dir)
        adapter.attach_tensor(attached_field("img"), _FakeMember("x/img"))
        with pytest.raises(TensorNotFound, match="has no tensor"):
            adapter.get_tensor_adapter("nope")

    def test_a_field_collides_folded(self, sources_dir):
        adapter = create_registered_source("plate", None, sources_dir)
        adapter.attach_tensor(attached_field("Nuclei"), _FakeMember("x/Nuclei"))

        assert folded_match(attached_field("nuclei"), adapter.attached_tensors) == (
            attached_field("Nuclei")
        )
        assert (
            folded_match(attached_field("membrane"), adapter.attached_tensors) is None
        )

    def test_disposing_removes_the_collection_whole(self, sources_dir):
        adapter = create_registered_source("plate", None, sources_dir)
        adapter.dispose_store()
        assert not adapter.store.exists()


class _FakeMember:
    """A member stands in for a tensor adapter; step 3 attaches none itself."""

    def __init__(self, array_id):
        from biopb.tensor.descriptor_pb2 import TensorDescriptor

        self._desc = TensorDescriptor(
            array_id=array_id, shape=[2, 2], dtype="<u2", chunk_shape=[2, 2]
        )

    def get_tensor_descriptor(self):
        return self._desc


class TestItIsStillNotDiscovered:
    def test_the_store_is_under_write_dir(self, tmp_path):
        """The rule the whole design rests on: one registrar per subtree, and
        write_dir's is this one. A collection under a discovery root would be
        claimed a second time, and the same bytes would reach the catalog under
        two ids."""
        write_dir = tmp_path / "w"
        adapter = create_registered_source("plate", None, sources_root(write_dir))
        assert Path(adapter.store).is_relative_to(write_dir)


class TestOverTheWire:
    """``register_source`` end to end, and the restart it exists for."""

    def _server(self, tmp_path):
        import threading

        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.core.config import CacheConfig

        from tests import catalog_server

        CacheManager.reset()
        CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
        server = catalog_server(
            location="grpc://localhost:0", writable=True, write_dir=tmp_path / "w"
        )
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        return server

    def test_it_answers_a_minted_id(self, writable_server, client):
        source_id = client.register_source("plate", {"omero": {"channels": []}})

        assert source_id.startswith("registered_")
        assert writable_server.sources.get(source_id) is not None

    def test_it_is_listed_and_empty(self, writable_server, client):
        source_id = client.register_source("plate")

        rows = client.query_sources(
            f"SELECT source_id, tensors FROM sources WHERE source_id = '{source_id}'",
            format="records",
        )
        assert len(rows) == 1
        assert not rows[0]["tensors"]

    def test_a_taken_name_is_refused_over_the_wire(self, writable_server, client):
        import pyarrow.flight as flight

        client.register_source("plate")
        with pytest.raises(flight.FlightServerError, match="already exists"):
            client.register_source("Plate")

    def test_an_unportable_name_is_refused_over_the_wire(self, writable_server, client):
        import pyarrow.flight as flight

        with pytest.raises(flight.FlightServerError, match="device name"):
            client.register_source("CON")

    def test_it_is_advertised_as_an_action(self, writable_server, client):
        assert "register_source" in {
            a.type for a in client._state.client.list_actions()
        }

    def test_it_survives_a_restart(self, tmp_path):
        """The point of the whole step: the id, the metadata and the catalog
        row all come back, because boot adoption re-registers the store -- not
        because anything discovered it."""
        from biopb.tensor import TensorFlightClient

        first = self._server(tmp_path)
        try:
            client = TensorFlightClient(f"grpc://localhost:{first.port}")
            source_id = client.register_source("plate", {"omero": {"channels": []}})
        finally:
            first.shutdown()

        second = self._server(tmp_path)
        try:
            revived = second.sources.get(source_id)
            assert revived is not None
            assert revived.get_metadata() == {"omero": {"channels": []}}

            client = TensorFlightClient(f"grpc://localhost:{second.port}")
            rows = client.query_sources(
                f"SELECT source_id FROM sources WHERE source_id = '{source_id}'",
                format="records",
            )
            assert len(rows) == 1
        finally:
            second.shutdown()

    def test_a_name_freed_by_hand_can_be_reused_with_a_new_id(self, tmp_path):
        """The id is the store's, not the name's: registering the same name
        again after the directory is gone is a different source."""
        from biopb.tensor import TensorFlightClient

        first = self._server(tmp_path)
        try:
            client = TensorFlightClient(f"grpc://localhost:{first.port}")
            original = client.register_source("plate")
        finally:
            first.shutdown()

        import shutil

        shutil.rmtree(tmp_path / "w" / "sources" / "plate.zarr")

        second = self._server(tmp_path)
        try:
            client = TensorFlightClient(f"grpc://localhost:{second.port}")
            assert client.register_source("plate") != original
        finally:
            second.shutdown()
