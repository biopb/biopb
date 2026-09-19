"""A durable upload gets the whole lifecycle (biopb/biopb#1059 step 1).

An ``ome_zarr:`` upload's store is minted under ``write_dir`` and its catalog
row written at create, so both are the server's own to release: discard removes
the directory and drops the row, the sweep treats the kind like any other, and
a store still carrying the ``pending`` marker when a server starts is a crashed
upload -- deleted at boot, and declined by discovery in the meantime.
"""

import json
import threading
from pathlib import Path

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb.tensor import TensorFlightClient, UploadRefused
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.adapters.zarr import ZarrAdapter, upload_state
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.chunk import encode_chunk_id
from biopb_tensor_server.core.config import CacheConfig
from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState
from biopb_tensor_server.serving.upload_manager import write_dir_under_root

from tests import catalog_server


def _create(client, name="ome_zarr:durable"):
    return client.create_tensor(
        name, np.empty((4, 4), dtype=np.uint16), chunk_shape=(2, 2)
    )


def _put(client, desc, start=(0, 0), stop=(2, 2), fill=7):
    data = np.full(
        [b - a for a, b in zip(start, stop, strict=True)], fill, dtype=np.uint16
    )
    client.upload_chunk(desc, ChunkBounds(start=list(start), stop=list(stop)), data)


def _catalog_ids(db):
    return {
        r["source_id"] for r in db.query("SELECT source_id FROM sources").to_pylist()
    }


def _marker(store: Path):
    return upload_state(json.loads((store / ".zattrs").read_text()))


class TestDiscardReleasesTheStore:
    def test_the_directory_and_the_row_are_gone(
        self, writable_server, client, tmp_path
    ):
        desc = _create(client)
        _put(client, desc)
        store = tmp_path / "durable.zarr"
        assert store.is_dir()
        assert desc.array_id in _catalog_ids(writable_server.metadata_db)

        writable_server.uploads.discard(desc.array_id, "operator said so")

        assert not store.exists()
        assert desc.array_id not in _catalog_ids(writable_server.metadata_db)
        status = client.get_upload_status(desc.array_id)
        assert status["state"] == "DISCARDED"
        assert status["reason"] == "operator said so"

    def test_a_late_write_is_refused_and_does_not_recreate_the_store(
        self, writable_server, client, tmp_path
    ):
        """zarr makes parent directories on write, so a straggler could
        otherwise resurrect a partial store a moment after it was deleted."""
        desc = _create(client)
        writable_server.uploads.discard(desc.array_id, "gone")

        with pytest.raises(UploadRefused, match="gone") as exc:
            _put(client, desc)
        assert exc.value.state == "DISCARDED"
        assert not (tmp_path / "durable.zarr").exists()

    def test_a_read_is_refused_rather_than_answered_with_fill(
        self, writable_server, client
    ):
        """Read once before the discard so the chunk is in the cache: the
        refusal has to come before the cache is consulted, or the reader gets
        the bytes the cache still holds (and zarr's fill for the rest)."""
        desc = _create(client)
        _put(client, desc, fill=3)
        client.finish_upload(desc)
        assert client.get_tensor(desc.array_id)[:2, :2].compute().max() == 3
        adapter = writable_server.sources.get(desc.array_id)
        chunk_id = encode_chunk_id(
            desc.array_id, ChunkBounds(start=[0, 0], stop=[2, 2])
        )

        writable_server.uploads.discard(desc.array_id, "gone")

        with pytest.raises(flight.FlightServerError, match="gone"):
            adapter.resolve_chunk_data(chunk_id, CacheManager.get_instance())

    def test_discard_is_idempotent_on_a_store_already_gone(
        self, writable_server, client
    ):
        desc = _create(client)
        first = writable_server.uploads.discard(desc.array_id, "first")
        second = writable_server.uploads.discard(desc.array_id, "second")
        assert (first["state"], second["state"]) == ("DISCARDED", "DISCARDED")
        assert second["reason"] == "first"

    def test_a_write_in_flight_lands_before_the_store_goes(
        self, writable_server, client, tmp_path
    ):
        """The write lock orders a write that passed the refusal ahead of the
        disposal: whichever wins, no directory is left behind."""
        desc = _create(client)
        adapter = writable_server.sources.get(desc.array_id)
        store = tmp_path / "durable.zarr"
        errors = []

        def write():
            try:
                for _ in range(20):
                    _put(client, desc)
            except UploadRefused:
                pass
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        t = threading.Thread(target=write)
        t.start()
        writable_server.uploads.discard(desc.array_id, "race")
        t.join()

        assert not errors
        assert adapter.upload_status()["state"] == "DISCARDED"
        assert not store.exists()


class TestCreateOwnsItsDirectory:
    def test_an_existing_directory_is_refused_and_untouched(
        self, writable_server, client, tmp_path
    ):
        """Discard removes the directory whole, so create must never adopt one
        it did not make -- a finished upload from an earlier server life, or
        anything else that happens to sit under write_dir."""
        theirs = tmp_path / "taken.zarr"
        theirs.mkdir()
        (theirs / "keep.txt").write_text("not yours")

        with pytest.raises(flight.FlightServerError, match="already exists"):
            client.create_tensor(
                "ome_zarr:taken", np.empty((4, 4), np.uint16), chunk_shape=(2, 2)
            )

        assert (theirs / "keep.txt").read_text() == "not yours"
        assert not (theirs / ".zarray").exists()
        assert writable_server.uploads.status("ome_zarr_x")["state"] == "UNKNOWN"


class TestTheNameCannotEscapeWriteDir:
    """The name becomes a directory the server creates and, on discard,
    removes whole, so it must stay inside the one the server chose."""

    @pytest.mark.parametrize(
        "name", ["../../escaped", "..", ".", "a/b", "a\\b", "C:evil"]
    )
    def test_refused_and_nothing_is_written_anywhere(self, tmp_path, name):
        """write_dir is nested inside tmp_path, so an escape would land under
        tmp_path too -- one glob then covers the whole surface."""
        from biopb.tensor.descriptor_pb2 import TensorDescriptor

        write_dir = tmp_path / "a" / "b" / "w"
        write_dir.mkdir(parents=True)
        desc = TensorDescriptor(
            array_id=f"ome_zarr:{name}",
            shape=[4, 4],
            dtype="uint16",
            chunk_shape=[2, 2],
        )
        with pytest.raises(ValueError, match="cannot name a store"):
            OmeZarrAdapter.create_upload(name, desc, metadata=None, write_dir=write_dir)
        assert not list(tmp_path.glob("**/*.zarr"))

    def test_the_refusal_reaches_the_client(self, writable_server, client, tmp_path):
        with pytest.raises(flight.FlightServerError, match="cannot name a store"):
            client.create_tensor(
                "ome_zarr:..", np.empty((4, 4), np.uint16), chunk_shape=(2, 2)
            )
        assert not list(tmp_path.glob("**/*.zarr"))

    def test_ordinary_names_are_still_accepted(self):
        from biopb_tensor_server.adapters._writable import unsafe_store_name

        for name in ["nuclei", "my data (1)", "run-2026.09.19", "_x", "..hidden"]:
            assert unsafe_store_name(name) is None


class TestTheBootSweepDropsTheRow:
    """A persisted catalog outlives the process, so the row an ``ome_zarr:``
    upload wrote at create is still there when the next server finds its store
    unfinished -- and nothing else will drop it, since write_dir is outside
    every discovery root and the reconciler never sees this id."""

    @staticmethod
    def _manager(write_dir, db):
        from biopb_tensor_server.core.source_registry import SourceRegistry
        from biopb_tensor_server.serving.upload_manager import UploadManager

        return UploadManager(SourceRegistry(), write_dir, db)

    def test_a_crashed_upload_leaves_no_row_behind(self, tmp_path):
        from biopb.tensor.descriptor_pb2 import TensorDescriptor
        from biopb_tensor_server.serving.metadata_db import MetadataDatabase

        write_dir = tmp_path / "w"
        write_dir.mkdir()
        db = MetadataDatabase()
        first = self._manager(write_dir, db)
        for name in ("crashed", "done"):
            first.create_tensor(
                TensorDescriptor(
                    array_id=f"ome_zarr:{name}",
                    shape=[4, 4],
                    dtype="uint16",
                    chunk_shape=[2, 2],
                )
            )
        first.finish(OmeZarrAdapter.upload_source_id(write_dir / "done.zarr"))
        assert len(_catalog_ids(db)) == 2

        # The process dies here. The stores and the rows both survive it.
        assert self._manager(write_dir, db).discard_unfinished_stores() == 1
        assert not (write_dir / "crashed.zarr").exists()
        assert _catalog_ids(db) == {
            OmeZarrAdapter.upload_source_id(write_dir / "done.zarr")
        }


class TestTheMarker:
    def test_ready_is_announced_only_after_the_seal(
        self, writable_server, client, tmp_path
    ):
        """A poller that has seen READY must find the store sealed after a
        crash; so if the seal cannot be written, finish fails and the upload
        stays PENDING for a retry."""
        desc = _create(client)
        _put(client, desc)
        adapter = writable_server.sources.get(desc.array_id)
        store = tmp_path / "durable.zarr"

        def refuse(state):
            raise OSError("disk full")

        adapter._write_upload_state = refuse
        with pytest.raises(flight.FlightServerError, match="could not seal"):
            client.finish_upload(desc)
        assert client.get_upload_status(desc.array_id)["state"] == "PENDING"
        assert _marker(store) == "pending"

        del adapter._write_upload_state
        assert client.finish_upload(desc)["state"] == "READY"
        assert _marker(store) == "ready"

    def test_finish_after_discard_reports_the_discard(self, writable_server, client):
        """The seal on a removed store fails on the missing file; what the
        caller learns is the discard, not the missing file."""
        desc = _create(client)
        writable_server.uploads.discard(desc.array_id, "gone")

        with pytest.raises(UploadRefused, match="gone") as exc:
            client.finish_upload(desc)
        assert exc.value.state == "DISCARDED"

    def test_pending_from_create_and_ready_after_finish(self, client, tmp_path):
        desc = _create(client)
        store = tmp_path / "durable.zarr"
        assert _marker(store) == "pending"

        _put(client, desc)
        client.finish_upload(desc)

        assert _marker(store) == "ready"
        # The rest of the metadata is untouched by the rewrite.
        zattrs = json.loads((store / ".zattrs").read_text())
        assert "multiscales" in zattrs

    def test_a_caller_supplied_metadata_keeps_its_keys(self, client, tmp_path):
        desc = client.create_tensor(
            "ome_zarr:withmeta",
            np.empty((4, 4), dtype=np.uint16),
            chunk_shape=(2, 2),
            ome_metadata={
                "multiscales": [
                    {
                        "version": "0.4",
                        "axes": [
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ],
                        "datasets": [{"path": "0"}],
                    }
                ],
                "omero": {"channels": []},
            },
        )
        zattrs = json.loads((tmp_path / "withmeta.zarr" / ".zattrs").read_text())
        assert zattrs["omero"] == {"channels": []}
        assert _marker(tmp_path / "withmeta.zarr") == "pending"
        assert desc.array_id


class TestAServerRestart:
    def _server(self, tmp_path):
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
        server = catalog_server(
            location="grpc://localhost:0", writable=True, write_dir=tmp_path / "w"
        )
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        return server

    def test_a_pending_store_is_removed_at_boot_and_a_ready_one_kept(self, tmp_path):
        first = self._server(tmp_path)
        try:
            client = TensorFlightClient(f"grpc://localhost:{first.port}")
            crashed = _create(client, "ome_zarr:crashed")
            _put(client, crashed)
            done = _create(client, "ome_zarr:done")
            _put(client, done)
            client.finish_upload(done)
        finally:
            first.shutdown()
        assert (tmp_path / "w" / "crashed.zarr").is_dir()

        second = self._server(tmp_path)
        try:
            assert not (tmp_path / "w" / "crashed.zarr").exists()
            assert _marker(tmp_path / "w" / "done.zarr") == "ready"
        finally:
            second.shutdown()
            CacheManager.reset()

    def test_no_write_dir_means_nothing_to_sweep(self):
        from biopb_tensor_server.core.source_registry import SourceRegistry
        from biopb_tensor_server.serving.upload_manager import UploadManager

        manager = UploadManager(SourceRegistry(), None, None)
        assert manager.discard_unfinished_stores() == 0


class TestDiscoveryDeclinesAPendingStore:
    def test_both_claims_decline_until_finish(self, client, tmp_path):
        """A finished upload store is a bare array, which ``ZarrAdapter``
        claims (``OmeZarrAdapter`` declines a top-level ``.zarray`` by design);
        while pending, neither takes it."""
        desc = _create(client, "ome_zarr:half")
        store = tmp_path / "half.zarr"

        ctx = ClaimContext(store)
        assert OmeZarrAdapter.claim(ctx, DiscoveryState()) is None
        assert ZarrAdapter.claim(ctx, DiscoveryState()) is None

        client.finish_upload(desc)
        claim = ZarrAdapter.claim(ClaimContext(store), DiscoveryState())
        assert claim is not None and claim.source_type == "zarr"


class TestWriteDirPlacement:
    def test_inside_a_root_is_reported(self, tmp_path):
        root = tmp_path / "data"
        (root / "uploads").mkdir(parents=True)
        assert write_dir_under_root(root / "uploads", [root]) == root
        assert write_dir_under_root(root, [root]) == root

    def test_outside_every_root_is_fine(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "uploads").mkdir()
        assert write_dir_under_root(tmp_path / "uploads", [tmp_path / "data"]) is None
        assert write_dir_under_root(None, [tmp_path / "data"]) is None
