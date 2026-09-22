"""An upload adds a tensor to a source that already exists (step 5).

``register_source`` mints a container and nothing else; ``add_tensor`` puts
``<scheme>://<source_id>/@fields/<name>`` on it. The scheme names the store
format and nothing else, so the answered ``array_id`` carries none and the
format is read back off the directory at the next registration.

What that buys, and what is tested here: a finished upload survives a restart
(the gap this whole design closes, biopb/biopb#1048), several tensors share one
source's metadata and one catalog row, and a pending one is routable to its own
writer without being visible to a reader.
"""

import json
import threading

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb.tensor import TensorFlightClient
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters.fields import (
    fields_root,
    source_fields_dir,
)
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.adapter_base import catalog_tensors
from biopb_tensor_server.core.config import CacheConfig

from tests import catalog_server

SHAPE = (4, 6)
CHUNK = (2, 3)


def _arr(fill=1, shape=SHAPE):
    return np.full(shape, fill, dtype=np.uint16)


def _add(client, source, field, scheme="zarr", arr=None, **kw):
    arr = _arr() if arr is None else arr
    return client.add_tensor(
        f"{scheme}://{source}/@fields/{field}", arr, chunk_shape=CHUNK, **kw
    )


class TestTheAnsweredId:
    def test_it_is_the_request_minus_the_scheme(self, client, source):
        """The format is a property of the stored tensor, not of its name: a
        reader that later asks for this id says nothing about zarr or cache."""
        desc = _add(client, source, "img")
        assert desc.array_id == f"{source}/@fields/img"

    @pytest.mark.parametrize("scheme", ["zarr", "cache"])
    def test_both_formats_answer_the_same_id(self, client, source, scheme):
        desc = _add(client, source, f"f-{scheme}", scheme=scheme)
        assert desc.array_id == f"{source}/@fields/f-{scheme}"

    @pytest.mark.parametrize("scheme", ["zarr", "cache"])
    def test_both_formats_round_trip_their_pixels(self, client, source, scheme):
        desc = _add(client, source, f"px-{scheme}", scheme=scheme)
        client.upload_array(desc, _arr(7))
        np.testing.assert_array_equal(
            client.get_tensor(desc.array_id).compute(), _arr(7)
        )


class TestOneSourceManyTensors:
    def test_the_source_lists_what_is_published_and_no_more(
        self, writable_server, client, source
    ):
        published = _add(client, source, "done")
        client.upload_array(published, _arr())
        _add(client, source, "filling")  # still PENDING

        listed = [
            d.array_id for d in catalog_tensors(writable_server.sources.get(source))
        ]
        assert listed == [published.array_id]

    def test_a_pending_tensor_is_writable_but_not_readable(self, client, source):
        """The one gate: a hole and a chunk still in flight are the same
        picture, so PENDING refuses reads outright rather than zero-filling."""
        desc = _add(client, source, "filling")
        # On the server's grid, not the one the request asked for: 4x6 uint16
        # is far under the transfer bound, so the whole tensor is one chunk.
        grid = tuple(desc.chunk_shape)
        client.upload_chunk(
            desc, ChunkBounds(start=[0, 0], stop=list(grid)), _arr(5, grid)
        )

        with pytest.raises(flight.FlightError):
            client.get_tensor(desc.array_id).compute()

        client.set_upload_status(desc, "READY")
        assert client.get_tensor(desc.array_id).compute().max() == 5

    def test_the_source_has_one_catalog_row_for_all_of_them(
        self, writable_server, client, source
    ):
        for field in ("a", "b", "c"):
            client.upload_array(_add(client, source, field), _arr())

        rows = (
            writable_server.metadata_db.query("SELECT source_id FROM sources")
            .column(0)
            .to_pylist()
        )
        assert rows.count(source) == 1

    def test_metadata_is_the_sources_and_every_tensor_inherits_it(
        self, writable_server, client
    ):
        """Source-scoped, as the design has it: the physical scale a reader
        sees on a tensor is the one that rode in on ``register_source``."""
        source = client.register_source(
            "calibrated", {"omero": {"channels": [{"label": "dapi"}]}}
        )
        client.upload_array(_add(client, source, "img"), _arr())

        assert writable_server.sources.get(source).get_metadata()["omero"] == {
            "channels": [{"label": "dapi"}]
        }


class TestItSurvivesARestart:
    """Problem 2 of the design: a finished upload used to stop being readable
    after a restart -- its catalog row outlived the adapter behind it, because
    nothing re-registered the store the next life found."""

    @staticmethod
    def _server(tmp_path):
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
        server = catalog_server(
            location="grpc://localhost:0", writable=True, write_dir=tmp_path / "w"
        )
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        return server

    def test_a_published_zarr_tensor_reads_back_in_the_next_life(self, tmp_path):
        first = self._server(tmp_path)
        try:
            client = TensorFlightClient(f"grpc://localhost:{first.port}")
            source = client.register_source("keepme")
            desc = _add(client, source, "img")
            client.upload_array(desc, _arr(9))
            client.close()
        finally:
            first.shutdown()

        second = self._server(tmp_path)
        try:
            client = TensorFlightClient(f"grpc://localhost:{second.port}")
            np.testing.assert_array_equal(
                client.get_tensor(desc.array_id).compute(), _arr(9)
            )
            client.close()
        finally:
            second.shutdown()
            CacheManager.reset()

    def test_the_source_id_survives_the_write_dir_moving(self, tmp_path):
        """Recorded, not derived: a hash of the path would not survive this,
        and the catalog row could then never be matched to its store."""
        first = self._server(tmp_path)
        try:
            client = TensorFlightClient(f"grpc://localhost:{first.port}")
            source = client.register_source("moving")
            client.upload_array(_add(client, source, "img"), _arr())
            client.close()
        finally:
            first.shutdown()

        moved = tmp_path / "elsewhere"
        (tmp_path / "w").rename(moved)
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
        second = catalog_server(
            location="grpc://localhost:0", writable=True, write_dir=moved
        )
        try:
            assert second.sources.get(source) is not None
            assert [
                d.array_id for d in catalog_tensors(second.sources.get(source))
            ] == [f"{source}/@fields/img"]
        finally:
            second.shutdown()
            CacheManager.reset()

    def test_a_store_recording_no_id_is_swept_rather_than_served(self, tmp_path):
        """An id that cannot be read is an id the catalog row cannot be matched
        to, so leaving it would accumulate bytes nothing can reach or name."""
        from biopb_tensor_server.adapters.registered import (
            scan_registered_sources,
            sources_root,
        )

        root = sources_root(tmp_path)
        store = root / "anonymous.zarr"
        store.mkdir(parents=True)
        (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (store / ".zattrs").write_text(json.dumps({"multiscales": []}))

        assert scan_registered_sources(root) == {}
        assert not store.exists()


class TestWhatItRefuses:
    @pytest.mark.parametrize(
        "field,why",
        [
            ("@labels", "server owns"),
            ("CON", "device name"),
            ("..", "relative path component"),
            (".hidden", "starts with"),
        ],
    )
    def test_an_unusable_field(self, client, source, field, why):
        with pytest.raises(flight.FlightServerError, match=why):
            _add(client, source, field)

    def test_a_scheme_that_is_not_a_store_format(self, client, source):
        with pytest.raises(flight.FlightServerError, match="not a store format"):
            _add(client, source, "img", scheme="parquet")

    def test_a_label_set_takes_zarr_only(self, client, source):
        """A set is NGFF, so it has one format and the scheme has to say so."""
        client.upload_array(_add(client, source, "img"), _arr())
        with pytest.raises(flight.FlightServerError, match="a label set is NGFF"):
            client.add_tensor(
                f"cache://{source}/img/@labels/nuclei",
                np.zeros(SHAPE, np.uint32),
                chunk_shape=CHUNK,
            )

    def test_a_source_that_is_not_registered(self, client):
        with pytest.raises(
            flight.FlightServerError, match="names no registered source"
        ):
            _add(client, "registered_nope", "img")

    def test_a_bare_field_is_refused_on_a_discovered_source_too(
        self, writable_server, client, tmp_path
    ):
        """A bare field is a native tensor id, which only a format mints. The
        rule is the source's kind-independent one, and the refusal names the
        grammar that replaces it."""
        import zarr

        store = tmp_path / "theirs.zarr"
        zarr.create(store=zarr.DirectoryStore(str(store)), shape=(4, 4), dtype="uint16")
        from biopb_tensor_server.adapters.zarr import ZarrAdapter

        adapter = ZarrAdapter(zarr.open_array(str(store), mode="r"), "theirs")
        writable_server.register_source("theirs", adapter)

        with pytest.raises(
            flight.FlightServerError, match=r"<source_id>/@fields/<name>"
        ):
            client.add_tensor(
                "zarr://theirs/img", np.zeros(SHAPE, np.uint16), chunk_shape=CHUNK
            )


class TestALabelSetOnAMember:
    """A set belongs to an image, and a member is an image like any other."""

    def test_it_attaches_to_the_member_and_reads_back(self, client, source):
        image = _add(client, source, "img")
        client.upload_array(image, _arr())

        labels = np.zeros(SHAPE, np.uint32)
        labels[:2, :3] = 4
        desc = client.add_tensor(
            f"zarr://{image.array_id}/@labels/nuclei", labels, chunk_shape=CHUNK
        )
        client.upload_array(desc, labels)

        assert desc.array_id == f"{source}/@fields/img/@labels/nuclei"
        assert client.label_sets(image.array_id) == [desc.array_id]
        np.testing.assert_array_equal(
            client.get_tensor(desc.array_id).compute(), labels
        )

    def test_a_set_on_a_pending_member_is_refused(self, client, source):
        """A member is not an image anyone may read until it is published, so
        there is no extent to check a set against yet."""
        image = _add(client, source, "img")  # never published

        with pytest.raises(flight.FlightServerError):
            client.add_tensor(
                f"zarr://{image.array_id}/@labels/nuclei",
                np.zeros(SHAPE, np.uint32),
                chunk_shape=CHUNK,
            )


class TestTheStoreFollowsThePlan:
    def test_a_zarr_member_is_chunked_on_the_grid_it_is_planned_on(
        self, writable_server, client, source, tmp_path
    ):
        """One grid, not two: on disk, on the wire, for reads and for writes,
        and across a restart where nothing remembers what was asked for
        (``_writable.upload_grid``)."""
        import zarr

        desc = _add(client, source, "img")
        store = source_fields_dir(fields_root(tmp_path), source) / "img"
        native = zarr.open_array(str(store / "0"), mode="r").chunks

        assert tuple(desc.chunk_shape) == native

    def test_a_member_carries_its_own_content_version(
        self, writable_server, client, source, tmp_path
    ):
        """Its own, not its source's: publishing one member must not move a
        sibling's chunk ids, and a restart must not move either."""
        from biopb_tensor_server.adapters.registered import read_member_version
        from biopb_tensor_server.adapters.zarr import read_zattrs

        parent = writable_server.sources.get(source)
        parent = writable_server.sources.get(source)
        fields_dir = source_fields_dir(fields_root(tmp_path), source)
        first = _add(client, source, "a")
        second = _add(client, source, "b")
        tokens = {
            field: read_member_version(read_zattrs(fields_dir / field))
            for field in ("a", "b")
        }

        assert None not in tokens.values()
        assert tokens["a"] != tokens["b"]
        assert tokens["a"] != parent.content_version
        assert first.array_id != second.array_id
