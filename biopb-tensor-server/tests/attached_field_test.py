"""A tensor uploaded onto a source the server discovered (step 8b).

``docs/upload-model.md``, "Fields on a file source": a discovered source's bytes
are the user's, so a field added to one lives beside it, under
``<write_dir>/fields/<source_id>/<name>/``, and is bound as
``<source_id>/@fields/<name>``.

What the marked segment buys is what most of this file is about: an uploaded
field cannot shadow a scene of the user's own file, however it is named, so the
kind needs no namespace rule of its own -- and `labels`, `0` and `scene1` are all
ordinary field names again.
"""

import json
import threading
from pathlib import Path

import numpy as np
import pyarrow.flight as flight
import pytest
import zarr
from biopb.tensor import TensorFlightClient
from biopb_tensor_server.adapters.fields import fields_root, scan_source_fields
from biopb_tensor_server.adapters.zarr import ZarrAdapter
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.config import CacheConfig

from tests import catalog_server, register_and_catalog

SHAPE = (4, 6)
CHUNK = (2, 3)


def _arr(fill=1, shape=SHAPE):
    return np.full(shape, fill, dtype=np.uint16)


def _their_file(tmp_path, name="theirs", shape=SHAPE):
    """A zarr array of the user's own, outside ``write_dir``'s subtrees."""
    store = tmp_path / "data" / f"{name}.zarr"
    if not store.exists():
        store.parent.mkdir(parents=True, exist_ok=True)
        zarr.create(store=zarr.DirectoryStore(str(store)), shape=shape, dtype="uint16")
    return ZarrAdapter(zarr.open_array(str(store), mode="r"), name)


def _add(client, source, name, scheme="zarr", arr=None, **kw):
    arr = _arr() if arr is None else arr
    return client.add_tensor(
        f"{scheme}://{source}/@fields/{name}", arr, chunk_shape=CHUNK, **kw
    )


def _serve(tmp_path):
    """A fresh server on *tmp_path*'s ``write_dir`` -- a restart, in-process."""
    server = catalog_server(
        location="grpc://localhost:0", writable=True, write_dir=Path(tmp_path)
    )
    server.mark_ready()
    threading.Thread(target=server.serve, daemon=True).start()
    return server


@pytest.fixture
def discovered(writable_server, tmp_path):
    """A source the server discovered, registered and catalogued."""
    register_and_catalog(writable_server, "theirs", _their_file(tmp_path))
    return "theirs"


class TestTheAnsweredId:
    def test_it_carries_the_marked_segment_and_not_the_scheme(self, client, discovered):
        desc = _add(client, discovered, "raw")
        assert desc.array_id == "theirs/@fields/raw"

    @pytest.mark.parametrize("scheme", ["zarr", "cache"])
    def test_both_store_formats_round_trip_their_pixels(
        self, client, discovered, scheme
    ):
        """Both, not zarr alone: a member directory is format-agnostic since
        ``open_any_member``, so a field gets what a member gets."""
        desc = _add(client, discovered, f"px-{scheme}", scheme=scheme)
        client.upload_array(desc, _arr(7))
        np.testing.assert_array_equal(
            client.get_tensor(desc.array_id).compute(), _arr(7)
        )


class TestWhereTheBytesGo:
    def test_beside_the_source_and_never_inside_it(
        self, writable_server, client, discovered, tmp_path
    ):
        """The user's file is untouched: the store is under ``write_dir``."""
        before = sorted(p.name for p in (tmp_path / "data" / "theirs.zarr").iterdir())
        desc = _add(client, discovered, "raw")
        client.upload_array(desc, _arr())

        assert (fields_root(tmp_path) / "theirs" / "raw").is_dir()
        assert (
            sorted(p.name for p in (tmp_path / "data" / "theirs.zarr").iterdir())
            == before
        )

    def test_a_registered_source_is_refused_the_segment(self, client, source):
        """One kind of source, one layout: a collection has a store of its own,
        and a second place for its tensors would be two to adopt and sweep."""
        with pytest.raises(flight.FlightServerError, match="registered source"):
            _add(client, source, "raw")


class TestItCannotShadowTheFilesOwnTensors:
    """The whole reason the segment is marked (``docs/upload-model.md``, Names)."""

    def test_a_field_named_like_a_scene_is_its_own_tensor(
        self, writable_server, client, discovered
    ):
        desc = _add(client, discovered, "0")
        client.upload_array(desc, _arr(3))

        assert desc.array_id == "theirs/@fields/0"
        # The source's own tensor is untouched and still first in the listing.
        rows = writable_server.metadata_db.query(
            "SELECT tensors FROM sources WHERE source_id = 'theirs'"
        )
        listed = [str(t["array_id"]) for t in rows[0][0]]
        assert listed[0] == "theirs"
        assert "theirs/@fields/0" in listed

    def test_labels_is_an_ordinary_field_name(self, client, discovered):
        """Step 8a stopped reserving the word; nothing here takes it back."""
        desc = _add(client, discovered, "labels")
        assert desc.array_id == "theirs/@fields/labels"

    def test_a_name_under_the_marker_is_the_servers(self, client, discovered):
        with pytest.raises(flight.FlightServerError, match="marks a segment"):
            _add(client, discovered, "@ome")


class TestListingAndReading:
    def test_it_is_listed_after_the_sources_own_tensors(
        self, writable_server, client, discovered
    ):
        """A source's first tensor is its picture, and an uploaded field must
        never be that -- the same rule a label set is listed by."""
        desc = _add(client, discovered, "raw")
        client.upload_array(desc, _arr())

        adapter = writable_server.sources.get("theirs")
        from biopb_tensor_server.core.adapter_base import catalog_tensors

        assert [t.array_id for t in catalog_tensors(adapter)] == [
            "theirs",
            "theirs/@fields/raw",
        ]

    def test_pending_is_routable_but_not_listed(
        self, writable_server, client, discovered
    ):
        _add(client, discovered, "filling")
        adapter = writable_server.sources.get("theirs")

        assert "@fields/filling" in adapter.attached_tensors
        assert adapter.attached_fields == {}
        status = client.get_upload_status("theirs/@fields/filling")
        assert status["state"] == "PENDING"
        with pytest.raises(flight.FlightError):
            client.get_tensor("theirs/@fields/filling").compute()

    def test_a_label_set_binds_to_an_uploaded_field(self, client, discovered):
        """A field is a tensor of its source like any other, so a set may span
        one -- which is what putting the fields in ``_normalized_tensors`` buys."""
        desc = _add(client, discovered, "raw")
        client.upload_array(desc, _arr())

        labels = client.add_tensor(
            "zarr://theirs/@fields/raw/@labels/nuclei",
            np.zeros(SHAPE, dtype=np.uint32),
            chunk_shape=CHUNK,
        )
        assert labels.array_id == "theirs/@fields/raw/@labels/nuclei"


class TestWhatItRefuses:
    @pytest.mark.parametrize(
        "name,why",
        [
            ("..", "relative path component"),
            # A slash never reaches the name rule: it is what makes the *field*
            # stop naming one of these, so the grammar refuses it first.
            ("a/b", "slash-free name"),
            ("a\\b", "path separator"),
            ("CON", "Windows device name"),
            ("@labels", "marks a segment"),
        ],
    )
    def test_a_name_that_cannot_be_a_directory(self, client, discovered, name, why):
        with pytest.raises(flight.FlightServerError, match=why):
            _add(client, discovered, name)

    def test_a_taken_name_folded(self, client, discovered):
        _add(client, discovered, "Nuclei")
        with pytest.raises(flight.FlightServerError, match="already exists"):
            _add(client, discovered, "nuclei")

    def test_metadata_is_the_sources(self, client, discovered):
        with pytest.raises(flight.FlightServerError, match="metadata is the source's"):
            _add(client, discovered, "raw", ome_metadata={"name": "mine"})


class TestAcrossARestart:
    """The half that makes an uploaded field worth having: it outlives the
    process that received it, and re-attaches to its source."""

    def test_a_published_field_comes_back(self, writable_server, client, tmp_path):
        register_and_catalog(writable_server, "theirs", _their_file(tmp_path))
        desc = _add(client, "theirs", "raw")
        client.upload_array(desc, _arr(9))
        writable_server.shutdown()
        client.close()

        second = _serve(tmp_path)
        try:
            register_and_catalog(second, "theirs", _their_file(tmp_path))
            again = TensorFlightClient(f"grpc://localhost:{second.port}")
            try:
                np.testing.assert_array_equal(
                    again.get_tensor("theirs/@fields/raw").compute(), _arr(9)
                )
            finally:
                again.close()
        finally:
            second.shutdown()

    def test_a_pending_field_is_swept(self, writable_server, client, tmp_path):
        register_and_catalog(writable_server, "theirs", _their_file(tmp_path))
        _add(client, "theirs", "half")
        store = fields_root(tmp_path) / "theirs" / "half"
        assert store.is_dir()
        writable_server.shutdown()
        client.close()

        second = _serve(tmp_path)
        try:
            assert not store.exists()
        finally:
            second.shutdown()

    def test_an_orphaned_field_is_kept(self, writable_server, client, tmp_path):
        """A discovery root that goes away leaves the field with nothing to
        attach it to. It is the only copy of what someone uploaded, where a
        label set is usually derived, so it is kept rather than swept."""
        register_and_catalog(writable_server, "theirs", _their_file(tmp_path))
        desc = _add(client, "theirs", "raw")
        client.upload_array(desc, _arr())
        writable_server.shutdown()
        client.close()

        second = _serve(tmp_path)  # nothing registers "theirs" this time
        try:
            assert (fields_root(tmp_path) / "theirs" / "raw").is_dir()
            assert "@fields/raw" in scan_source_fields("theirs", fields_root(tmp_path))
        finally:
            second.shutdown()

    def test_a_rescan_does_not_drop_it(self, writable_server, client, tmp_path):
        """The row stays the reconciler's: a rescan that rewrites it must not
        lose what the upload path attached."""
        register_and_catalog(writable_server, "theirs", _their_file(tmp_path))
        desc = _add(client, "theirs", "raw")
        client.upload_array(desc, _arr())

        # What a reconciler pass does: register the source again, from scratch.
        register_and_catalog(writable_server, "theirs", _their_file(tmp_path))
        rows = writable_server.metadata_db.query(
            "SELECT tensors FROM sources WHERE source_id = 'theirs'"
        )
        assert "theirs/@fields/raw" in [str(t["array_id"]) for t in rows[0][0]]


class TestDiscardAndDelete:
    def test_a_discard_takes_the_store_and_the_listing(
        self, writable_server, client, discovered, tmp_path
    ):
        desc = _add(client, discovered, "raw")
        client.upload_array(desc, _arr())
        store = fields_root(tmp_path) / "theirs" / "raw"
        assert store.is_dir()

        client.set_upload_status(desc.array_id, "DISCARDED")

        assert not store.exists()
        adapter = writable_server.sources.get("theirs")
        assert adapter.attached_fields == {}

    def test_a_field_adopted_from_an_earlier_life_deletes(
        self, writable_server, client, tmp_path
    ):
        """It holds no upload record to seal, so the discard is store removal
        alone -- which needs the store path the adoption kept."""
        register_and_catalog(writable_server, "theirs", _their_file(tmp_path))
        desc = _add(client, "theirs", "raw")
        client.upload_array(desc, _arr())
        writable_server.shutdown()
        client.close()

        second = _serve(tmp_path)
        try:
            register_and_catalog(second, "theirs", _their_file(tmp_path))
            again = TensorFlightClient(f"grpc://localhost:{second.port}")
            try:
                again.set_upload_status("theirs/@fields/raw", "DISCARDED")
            finally:
                again.close()
            assert not (fields_root(tmp_path) / "theirs" / "raw").exists()
        finally:
            second.shutdown()


class TestScanSourceFields:
    def test_it_skips_a_pending_store(self, tmp_path):
        root = tmp_path / "fields" / "src"
        (root / "half").mkdir(parents=True)
        (root / "half" / ".zattrs").write_text(
            json.dumps({"biopb": {"upload": {"state": "pending"}}})
        )
        assert scan_source_fields("src", tmp_path / "fields") == {}

    def test_no_directory_is_no_fields(self, tmp_path):
        assert scan_source_fields("src", tmp_path / "fields") == {}


@pytest.fixture(autouse=True)
def _cache(tmp_path):
    """Each test gets its own cache, as the writable-server fixture does."""
    CacheManager.reset()
    CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "chunk-cache"))
    yield
    CacheManager.reset()
