"""Uploading a label set (biopb/biopb#1059 step 3).

The third upload kind: selected by an ``array_id`` with no ``kind:`` prefix
and a ``/labels/`` segment, it creates a tensor of a source that already
exists rather than a source. What that costs the boundary is a second place to
look an upload up (the parent's ``label_uploads``, not the registry) and a
catalog row that is the parent's; what it buys the client is the ordinary
``create_tensor`` / ``upload_array`` / ``finish_upload`` round trip, plus
``delete_labels`` to free the name again.

Design: ``biopb-tensor-server/docs/label-tensors.md``.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb.tensor._session import _parse_flight_endpoints
from biopb_tensor_server.adapters.labels import labels_root, sidecar_dir
from biopb_tensor_server.adapters.zarr import UPLOAD_PENDING, UPLOAD_READY, upload_state
from biopb_tensor_server.core.adapter_base import catalog_tensors
from biopb_tensor_server.core.chunk import content_version_of
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.errors import WriteNotSupportedError
from biopb_tensor_server.fixtures import create_multiresolution_ome_zarr

from tests import register_and_catalog
from tests.label_attachment_test import _write_label_group

SHAPE = (64, 64)
CHUNK = (32, 32)


@pytest.fixture
def image(tmp_path):
    """A plain 2-D OME-Zarr image, two levels, no labels of its own."""
    zarr_path, _, _ = create_multiresolution_ome_zarr(
        str(tmp_path / "img"), n_levels=2, base_shape=SHAPE, chunk_size=CHUNK
    )
    return Path(zarr_path)


def _adapter(root, source_id="oz1"):
    from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter

    return OmeZarrAdapter.create_from_config(
        SourceConfig(url=str(root), type="ome-zarr", source_id=source_id)
    )


@pytest.fixture
def served(writable_server, image):
    register_and_catalog(writable_server, "oz1", _adapter(image))
    return writable_server


def _labels(shape=SHAPE, dtype="uint32"):
    arr = np.zeros(shape, dtype=dtype)
    arr[:8, :8] = 3
    arr[32:40, 32:40] = 5
    return arr


def _create(client, array_id, arr=None, **kw):
    arr = _labels() if arr is None else arr
    return client.create_tensor(array_id, arr, chunk_shape=CHUNK, **kw)


def _tensor_ids(server, source_id="oz1"):
    return (
        server.metadata_db.query(
            "SELECT list_transform(tensors, t -> t.array_id) FROM sources "
            f"WHERE source_id = '{source_id}'"
        )
        .column(0)
        .to_pylist()[0]
    )


class TestTheRoundTrip:
    def test_create_upload_finish_and_read_back(self, served, client):
        labels = _labels()
        desc = _create(client, "oz1/labels/nuclei")
        assert desc.array_id == "oz1/labels/nuclei"

        # Routable from create, so the producer can poll to READY -- but not
        # listed, because none of its bytes have landed.
        assert client.get_upload_status("oz1/labels/nuclei")["state"] == "PENDING"
        assert "oz1/labels/nuclei" not in _tensor_ids(served)

        status = client.upload_array(desc, labels)
        assert status["state"] == "READY"

        read = client.get_tensor("oz1/labels/nuclei")
        assert read.dtype == np.uint32
        np.testing.assert_array_equal(read.compute(), labels)

    def test_finish_lists_it_under_its_image(self, served, client):
        desc = _create(client, "oz1/labels/nuclei")
        client.upload_array(desc, _labels())

        ids = _tensor_ids(served)
        assert ids[0] == "oz1"  # the image is still tensors[0]
        assert ids[-1] == "oz1/labels/nuclei"
        assert client.label_sets("oz1") == ["oz1/labels/nuclei"]

    def test_the_descriptor_names_its_image(self, served, client):
        desc = _create(client, "oz1/labels/nuclei")
        client.upload_array(desc, _labels())

        got = client.get_descriptor(
            "oz1/labels/nuclei", with_metadata=True, with_pyramid=True
        )
        meta = json.loads(got.metadata_json)["metadata"]
        assert meta["image-label"]["source"] == {"image": "oz1"}
        # Averaging ids invents ids: every computed level is nearest.
        assert {lvl.reduction_method for lvl in got.pyramid} == {"nearest"}

    def test_two_sets_coexist_under_one_image(self, served, client):
        for name in ("nuclei", "cells"):
            client.upload_array(_create(client, f"oz1/labels/{name}"), _labels())
        assert client.label_sets("oz1") == ["oz1/labels/cells", "oz1/labels/nuclei"]

    def test_client_metadata_supplies_the_colours(self, served, client):
        colors = [{"label-value": 3, "rgba": [255, 0, 0, 255]}]
        desc = _create(
            client,
            "oz1/labels/nuclei",
            ome_metadata={"image-label": {"colors": colors}},
        )
        client.upload_array(desc, _labels())
        meta = json.loads(
            client.get_descriptor("oz1/labels/nuclei", with_metadata=True).metadata_json
        )["metadata"]
        assert meta["image-label"]["colors"] == colors
        assert meta["image-label"]["source"] == {"image": "oz1"}


class TestWhatTheKindRefuses:
    @pytest.mark.parametrize(
        "array_id,arr,why",
        [
            ("nope/labels/x", None, "names no registered source"),
            ("oz1/labels/@ome", None, "are the server's own"),
            ("oz1/labels/x", np.zeros(SHAPE, "float32"), "unsigned integer"),
            ("oz1/labels/x", np.zeros((32, 32), "uint32"), "does not span"),
            ("oz1/nope", None, "Invalid array_id format"),
        ],
    )
    def test_refusals(self, served, client, array_id, arr, why):
        import pyarrow.flight as flight

        with pytest.raises(flight.FlightServerError, match=why):
            _create(client, array_id, arr=arr)

    @pytest.mark.parametrize("name", ["..", ".", "a\\b", "C:evil", "x" * 260])
    def test_a_name_that_would_escape_the_sidecar_directory_is_refused(
        self, served, client, tmp_path, name
    ):
        """The set's name becomes a directory the server creates and later
        removes whole, so it has to stay inside the one the server chose."""
        import pyarrow.flight as flight

        with pytest.raises(flight.FlightServerError, match="the set's name"):
            _create(client, f"oz1/labels/{name}")
        assert not list(labels_root(Path(tmp_path)).glob("**/*.zarr"))

    def test_a_taken_name_is_refused_until_it_is_deleted(self, served, client):
        import pyarrow.flight as flight

        client.upload_array(_create(client, "oz1/labels/nuclei"), _labels())
        with pytest.raises(flight.FlightServerError, match="already exists"):
            _create(client, "oz1/labels/nuclei")
        # ...and while it is still filling, too.
        _create(client, "oz1/labels/pending")
        with pytest.raises(flight.FlightServerError, match="already exists"):
            _create(client, "oz1/labels/pending")

    def test_a_finished_set_takes_no_further_chunk(self, served, client):
        """Sealed, like any finished upload -- there is no chunk-level edit."""
        from biopb_tensor_server.core.errors import UploadSealedError

        registered = served.sources.get("oz1")
        client.upload_array(_create(client, "oz1/labels/nuclei"), _labels())
        with pytest.raises(UploadSealedError):
            registered.label_sets["labels/nuclei"].put_chunk(None, None, None, None)

    def test_a_set_that_is_not_an_upload_refuses_a_write_outright(self, tmp_path):
        """A sidecar read back at startup tracks no upload at all."""
        from biopb_tensor_server.adapters.labels import open_label_set

        from tests.label_attachment_test import _sidecar

        group = _sidecar(tmp_path / "labels", "oz1", "mine")
        label_set = open_label_set(
            group, source_id="oz1", image_field="", name="mine", content_version=b"t"
        )
        with pytest.raises(WriteNotSupportedError, match="read-only"):
            label_set.put_chunk(None, None, None, None)


class TestTheSidecar:
    def test_it_is_born_pending_and_sealed_by_finish(self, served, client, tmp_path):
        store = sidecar_dir(labels_root(Path(tmp_path)), "oz1") / "nuclei.zarr"
        desc = _create(client, "oz1/labels/nuclei")
        assert upload_state(json.loads((store / ".zattrs").read_text())) == (
            UPLOAD_PENDING
        )
        client.upload_array(desc, _labels())
        assert upload_state(json.loads((store / ".zattrs").read_text())) == UPLOAD_READY

    def test_an_all_zero_chunk_is_never_materialized(self, served, client, tmp_path):
        """The sparse case: only the blocks that carry ids cost anything."""
        store = sidecar_dir(labels_root(Path(tmp_path)), "oz1") / "sparse.zarr"
        sparse = np.zeros(SHAPE, "uint32")
        sparse[:8, :8] = 1  # the (0, 0) chunk alone, of four
        desc = _create(client, "oz1/labels/sparse", arr=sparse)
        status = client.upload_array(desc, sparse)

        assert (status["uploaded_chunks"], status["expected_chunks"]) == (1, 4)
        chunks = {p.name for p in (store / "0").iterdir() if not p.name.startswith(".")}
        assert chunks == {"0.0"}
        np.testing.assert_array_equal(
            client.get_tensor("oz1/labels/sparse").compute(), sparse
        )

    def test_the_zero_skip_is_this_kind_s_alone(self, served, client):
        """A ``cache:`` source answers an unwritten chunk with "holds no chunk",
        so every block of one is sent however empty it is."""
        sparse = np.zeros(SHAPE, "uint32")
        sparse[:8, :8] = 1
        desc = client.create_tensor("cache:sparse", sparse, chunk_shape=CHUNK)
        status = client.upload_array(desc, sparse)
        assert (status["uploaded_chunks"], status["expected_chunks"]) == (4, 4)

    def test_it_survives_a_restart(self, served, client, tmp_path, image):
        """A finished sidecar is re-attached by the registration hook."""
        from tests import catalog_server

        client.upload_array(_create(client, "oz1/labels/nuclei"), _labels())
        served.shutdown()

        fresh = catalog_server(
            location="grpc://localhost:0", writable=True, write_dir=Path(tmp_path)
        )
        try:
            registered = register_and_catalog(fresh, "oz1", _adapter(image))
            assert "labels/nuclei" in registered.label_sets
            assert "oz1/labels/nuclei" in [
                t.array_id for t in catalog_tensors(registered)
            ]
        finally:
            fresh.shutdown()


class TestDelete:
    def test_it_unlists_the_set_and_removes_the_store(self, served, client, tmp_path):
        store = sidecar_dir(labels_root(Path(tmp_path)), "oz1") / "nuclei.zarr"
        client.upload_array(_create(client, "oz1/labels/nuclei"), _labels())

        assert client.delete_labels("oz1/labels/nuclei") == {
            "array_id": "oz1/labels/nuclei",
            "deleted": True,
        }
        assert not store.exists()
        assert client.label_sets("oz1") == []

    def test_the_action_is_advertised(self, served, client):
        assert "delete_labels" in {a.type for a in client._state.client.list_actions()}

    def test_the_name_is_free_again_and_reads_its_own_bytes(self, served, client):
        first, second = _labels(), _labels() * 2
        client.upload_array(_create(client, "oz1/labels/nuclei"), first)
        np.testing.assert_array_equal(
            client.get_tensor("oz1/labels/nuclei").compute(), first
        )
        client.delete_labels("oz1/labels/nuclei")

        client.upload_array(_create(client, "oz1/labels/nuclei"), second)
        np.testing.assert_array_equal(
            client.get_tensor("oz1/labels/nuclei").compute(), second
        )

    @pytest.mark.parametrize("array_id", ["oz1/labels/gone", "oz1", "oz1/labels/half"])
    def test_only_a_finished_uploaded_set_can_be_deleted(
        self, served, client, array_id
    ):
        import pyarrow.flight as flight

        _create(client, "oz1/labels/half")  # pending: not finished, not deletable
        with pytest.raises(flight.FlightServerError, match="not a deletable"):
            client.delete_labels(array_id)

    def test_a_native_set_is_the_file_s(self, writable_server, client, tmp_path):
        """An NGFF ``labels/`` group inside the user's store is never deleted."""
        import pyarrow.flight as flight

        zarr_path, _, _ = create_multiresolution_ome_zarr(
            str(tmp_path / "native"), n_levels=1, base_shape=SHAPE, chunk_size=CHUNK
        )
        group = _write_label_group(Path(zarr_path) / "labels" / "own", levels=1)
        register_and_catalog(
            writable_server, "oz2", _adapter(Path(zarr_path), source_id="oz2")
        )
        assert client.label_sets("oz2") == ["oz2/labels/own"]
        with pytest.raises(flight.FlightServerError, match="not a deletable"):
            client.delete_labels("oz2/labels/own")
        assert group.exists()


class TestTheSweep:
    def test_a_quiet_pending_set_is_discarded_and_its_store_removed(
        self, served, client, tmp_path
    ):
        import time

        quiet = time.monotonic() + 10_000
        store = sidecar_dir(labels_root(Path(tmp_path)), "oz1") / "dead.zarr"
        _create(client, "oz1/labels/dead")
        assert store.is_dir()

        assert served.uploads.reap(now=quiet) == (1, 0)
        assert not store.exists()
        assert client.get_upload_status("oz1/labels/dead")["state"] == "DISCARDED"

        # A second pass, a TTL later, reclaims the tombstone -- which is what
        # frees the name.
        assert served.uploads.reap(now=quiet + 10_000) == (0, 1)
        assert client.get_upload_status("oz1/labels/dead")["state"] == "UNKNOWN"
        client.upload_array(_create(client, "oz1/labels/dead"), _labels())

    def test_a_finished_set_is_not_swept(self, served, client):
        import time

        client.upload_array(_create(client, "oz1/labels/nuclei"), _labels())
        assert served.uploads.reap(now=time.monotonic() + 10_000) == (0, 0)
        assert client.label_sets("oz1") == ["oz1/labels/nuclei"]

    def test_a_crashed_upload_is_removed_at_boot(self, served, client, tmp_path):
        store = sidecar_dir(labels_root(Path(tmp_path)), "oz1") / "crashed.zarr"
        _create(client, "oz1/labels/crashed")
        served.shutdown()

        from tests import catalog_server

        fresh = catalog_server(
            location="grpc://localhost:0", writable=True, write_dir=Path(tmp_path)
        )
        try:
            assert not store.exists()
        finally:
            fresh.shutdown()


class TestContentVersion:
    """What the descriptor publishes is what the tickets were minted with.

    An uploaded set's bytes are its own -- they live in the sidecar store, not
    in the image file, which a set arriving never touches. So it carries its
    own ``content_version`` (biopb/biopb#178), and the copy published on the
    descriptor (biopb/biopb#780) has to be that one and not the image's: they
    are one signal, and a consumer namespacing a cache it cannot key by
    chunk_id is keying it by the published form.
    """

    def _chunk_ids(self, client, array_id):
        info = flight.FlightInfo.deserialize(client.get_tensor_pb(array_id).flight_info)
        return set(_parse_flight_endpoints(info)[0])

    def _minted(self, client, array_id):
        """The content_version the read plan's chunk_ids carry."""
        return {
            content_version_of(chunk_id)
            for chunk_id in self._chunk_ids(client, array_id)
        }

    def test_a_set_publishes_its_own_version_not_its_image_s(self, served, client):
        client.upload_array(_create(client, "oz1/labels/nuclei"), _labels())

        image = client.get_descriptor("oz1").content_version
        labels = client.get_descriptor("oz1/labels/nuclei").content_version

        assert image and labels
        assert labels != image

    @pytest.mark.parametrize("array_id", ["oz1", "oz1/labels/nuclei"])
    def test_the_published_version_is_the_minted_one(self, served, client, array_id):
        client.upload_array(_create(client, "oz1/labels/nuclei"), _labels())

        published = client.get_descriptor(array_id).content_version
        assert self._minted(client, array_id) == {published}

    def test_a_reused_name_publishes_a_new_version(self, served, client):
        """``delete_labels``: "the next set uploaded under it is a distinct
        tensor with its own cache namespace". The image file is untouched by
        both uploads, so an image-derived version could not say so."""
        client.upload_array(_create(client, "oz1/labels/nuclei"), _labels())
        first = client.get_descriptor("oz1/labels/nuclei").content_version
        client.delete_labels("oz1/labels/nuclei")

        client.upload_array(_create(client, "oz1/labels/nuclei"), _labels())
        second = client.get_descriptor("oz1/labels/nuclei").content_version

        assert first != second
        assert client.get_descriptor("oz1").content_version  # unmoved either way

    def test_a_semantics_epoch_bump_leaves_the_published_version_alone(
        self, served, client, epoch
    ):
        """biopb/biopb#1076, over the wire.

        The epoch re-keys chunks; it does not claim the data changed. So the
        chunk_ids move, the published field does not, and -- because the two
        versions are framed separately rather than fused -- the content_version
        INSIDE the new chunk_ids is still the published one. A consumer that
        stamped something with this token (an ROI's ``drawn_against_version``)
        must not read a server upgrade as "your image changed".
        """
        client.upload_array(_create(client, "oz1/labels/nuclei"), _labels())
        names = ("oz1", "oz1/labels/nuclei")
        ids = {a: self._chunk_ids(client, a) for a in names}
        published = {a: client.get_descriptor(a).content_version for a in names}
        assert all(self._minted(client, a) == {published[a]} for a in names)

        epoch(1)

        for array_id in names:
            assert self._chunk_ids(client, array_id).isdisjoint(ids[array_id])
            assert (
                client.get_descriptor(array_id).content_version == published[array_id]
            )
            # Still recoverable from the re-keyed ids: the epoch moved, the
            # content claim did not.
            assert self._minted(client, array_id) == {published[array_id]}
