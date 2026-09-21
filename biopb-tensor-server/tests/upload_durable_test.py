"""A ``zarr://`` member gets the whole lifecycle (biopb/biopb#1059 step 1).

A member's store is minted under its source's directory in ``write_dir``, so it
is the server's own to release: discard removes the directory and takes the
tensor out of its source's listing, the sweep treats it like any other upload,
and a store still carrying the ``pending`` marker when a server starts is a
crashed upload -- deleted at boot, and declined by discovery in the meantime.
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
from biopb_tensor_server.core.errors import UploadDiscardedReadError
from biopb_tensor_server.serving.upload_manager import write_dir_under_root

from tests import catalog_server


def _create(client, source, field="durable"):
    return client.add_tensor(
        f"zarr://{source}/{field}",
        np.empty((4, 4), dtype=np.uint16),
        chunk_shape=(2, 2),
    )


def _store(server, source, field="durable") -> Path:
    """Where the member keeps its bytes: ``<write_dir>/sources/<name>.zarr/<field>``."""
    return server.sources.get(source).member_store(field)


# The whole 4x4 tensor is one planned chunk: a zarr store is minted on the
# transfer grid, and 4x4 uint16 is far under it (``_writable.upload_grid``).
def _put(client, desc, start=(0, 0), stop=(4, 4), fill=7):
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
    def test_the_directory_goes_and_the_tensor_is_unlisted(
        self, writable_server, client, source
    ):
        desc = _create(client, source)
        _put(client, desc)
        client.set_upload_status(desc, "READY")
        parent = writable_server.sources.get(source)
        store = _store(writable_server, source)
        assert store.is_dir()
        assert [d.array_id for d in parent.list_tensor_descriptors()] == [desc.array_id]

        writable_server.uploads.discard(desc.array_id, "operator said so")

        assert not store.exists()
        assert parent.list_tensor_descriptors() == []
        # The *source* keeps its row: a member has none of its own, and the
        # source is still there to add another tensor to.
        assert source in _catalog_ids(writable_server.metadata_db)
        status = client.get_upload_status(desc.array_id)
        assert status["state"] == "DISCARDED"
        assert status["reason"] == "operator said so"

    def test_a_late_write_is_refused_and_does_not_recreate_the_store(
        self, writable_server, client, source
    ):
        """zarr makes parent directories on write, so a straggler could
        otherwise resurrect a partial store a moment after it was deleted."""
        desc = _create(client, source)
        store = _store(writable_server, source)
        writable_server.uploads.discard(desc.array_id, "gone")

        with pytest.raises(UploadRefused, match="gone") as exc:
            _put(client, desc)
        assert exc.value.state == "DISCARDED"
        assert not store.exists()

    def test_a_read_is_refused_rather_than_answered_with_fill(
        self, writable_server, client, source
    ):
        """Read once before the discard so the chunk is in the cache: the
        refusal has to come before the cache is consulted, or the reader gets
        the bytes the cache still holds (and zarr's fill for the rest)."""
        desc = _create(client, source)
        _put(client, desc, fill=3)
        client.set_upload_status(desc, "READY")
        assert client.get_tensor(desc.array_id)[:2, :2].compute().max() == 3
        adapter = writable_server.sources.get(source).members["durable"]
        chunk_id = encode_chunk_id(
            desc.array_id, ChunkBounds(start=[0, 0], stop=[4, 4])
        )

        writable_server.uploads.discard(desc.array_id, "gone")

        with pytest.raises(UploadDiscardedReadError, match="gone"):
            adapter.resolve_chunk_data(chunk_id, CacheManager.get_instance())

    def test_discard_is_idempotent_on_a_store_already_gone(
        self, writable_server, client, source
    ):
        desc = _create(client, source)
        first = writable_server.uploads.discard(desc.array_id, "first")
        second = writable_server.uploads.discard(desc.array_id, "second")
        assert (first["state"], second["state"]) == ("DISCARDED", "DISCARDED")
        assert second["reason"] == "first"

    def test_a_write_in_flight_lands_before_the_store_goes(
        self, writable_server, client, source
    ):
        """The write lock orders a write that passed the refusal ahead of the
        disposal: whichever wins, no directory is left behind."""
        desc = _create(client, source)
        adapter = writable_server.sources.get(source).members["durable"]
        store = _store(writable_server, source)
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


class TestAddOwnsItsDirectory:
    def test_an_existing_directory_is_refused_and_untouched(
        self, writable_server, client, source
    ):
        """Discard removes the directory whole, so an add must never adopt one
        it did not make -- a crashed upload the boot sweep has yet to reach, or
        anything else that happens to sit under the collection."""
        theirs = _store(writable_server, source, "taken")
        theirs.mkdir()
        (theirs / "keep.txt").write_text("not yours")

        with pytest.raises(flight.FlightServerError, match="already exists"):
            _create(client, source, "taken")

        assert (theirs / "keep.txt").read_text() == "not yours"
        assert not (theirs / ".zarray").exists()
        assert "taken" not in writable_server.sources.get(source).members


class TestTheFieldCannotEscapeItsSource:
    """The field becomes a directory the server creates and, on discard,
    removes whole, so it must stay inside the one the server chose."""

    @pytest.mark.parametrize(
        "field", ["../../escaped", "..", ".", "a/b", "a\\b", "C:evil"]
    )
    def test_refused_and_nothing_is_written_anywhere(
        self, writable_server, client, source, field, tmp_path
    ):
        """write_dir is the whole of tmp_path, so an escape would land under
        it -- one glob then covers the whole surface."""
        with pytest.raises(flight.FlightServerError):
            client.add_tensor(
                f"zarr://{source}/{field}",
                np.empty((4, 4), np.uint16),
                chunk_shape=(2, 2),
            )
        assert not list(tmp_path.glob("**/escaped*"))

    def test_the_refusal_names_the_field(self, writable_server, client, source):
        with pytest.raises(flight.FlightServerError, match="cannot name a tensor"):
            client.add_tensor(
                f"zarr://{source}/..",
                np.empty((4, 4), np.uint16),
                chunk_shape=(2, 2),
            )

    def test_ordinary_names_are_still_accepted(self):
        from biopb_tensor_server.adapters._writable import unsafe_store_name

        for name in ["nuclei", "my data (1)", "run-2026.09.19", "_x", "nuclei2"]:
            assert unsafe_store_name(name) is None


class TestANameIsAPathOnEveryPlatform:
    """A store minted on Linux has to open, and keep its identity, when the
    same ``write_dir`` is later served from Windows or macOS -- so the rules
    are the union of what the three refuse, checked on whichever is running.
    """

    @pytest.mark.parametrize(
        "name,why",
        [
            ("CON", "device name"),
            ("con", "device name"),
            ("COM1.zarr", "device name"),
            ("LPT9", "device name"),
            ("nul", "device name"),
            ("trailing.", "Windows strips"),
            ("trailing ", "Windows strips"),
            (".hidden", "starts with"),
            ("..hidden", "starts with"),
            ('quote"d', "NTFS"),
            ("pipe|d", "NTFS"),
            ("star*", "NTFS"),
            ("q?", "NTFS"),
            ("lt<gt>", "NTFS"),
            ("bell\x07", "control character"),
        ],
    )
    def test_refused(self, name, why):
        from biopb_tensor_server.adapters._writable import unsafe_store_name

        reason = unsafe_store_name(name)
        assert reason is not None, f"{name!r} should be refused"
        assert why in reason

    def test_a_device_name_is_refused_whatever_follows_the_dot(self):
        """Windows reserves them with any extension, and the server appends
        ``.zarr`` -- so ``CON`` alone mints a file that cannot be opened."""
        from biopb_tensor_server.adapters._writable import unsafe_store_name

        assert unsafe_store_name("CON.anything") is not None
        assert unsafe_store_name("CONSTANT") is None  # only the exact stem

    def test_the_length_limit_counts_bytes_not_characters(self):
        """255 is a directory-entry limit, so a non-ASCII name passes a
        character count and still overflows it."""
        from biopb_tensor_server.adapters._writable import unsafe_store_name

        assert unsafe_store_name("a" * 250) is None
        assert unsafe_store_name("é" * 250) is not None
        assert "bytes" in unsafe_store_name("é" * 250)

    def test_the_refusal_reaches_the_client(self, writable_server, client, source):
        with pytest.raises(flight.FlightServerError, match="cannot name a tensor"):
            client.add_tensor(
                f"zarr://{source}/CON",
                np.empty((4, 4), np.uint16),
                chunk_shape=(2, 2),
            )


class TestNamesCollideFolded:
    """Two names that differ only by case or normalization are one directory
    on NTFS, APFS and HFS+ and two on ext4. The refusal folds, so the source
    does not split in two on the next host to serve this ``write_dir``.
    """

    def test_fold_name_equates_case_and_normal_form(self):
        import unicodedata

        from biopb_tensor_server.adapters._writable import fold_name

        assert fold_name("Nuclei") == fold_name("nuclei")
        assert fold_name(unicodedata.normalize("NFD", "café")) == fold_name("café")

    def test_the_name_itself_is_not_folded(self, writable_server, client, source):
        """Only the comparison folds: the store is written under the spelling
        the caller chose, so ``Nuclei`` stays ``Nuclei`` on disk."""
        _create(client, source, "Nuclei")
        collection = writable_server.sources.get(source).store
        assert [d.name for d in collection.iterdir() if d.is_dir()] == ["Nuclei"]

    def test_a_case_variant_of_a_taken_field_is_refused(self, client, source):
        _create(client, source, "Nuclei")
        with pytest.raises(flight.FlightServerError, match="already has a tensor"):
            _create(client, source, "nuclei")

    def test_an_unrelated_name_is_still_free(self, client, source):
        _create(client, source, "Nuclei")
        _create(client, source, "membrane")


class TestLabelsIsReservedAsAField:
    """``labels`` is the NGFF group name and the wire segment that addresses a
    set, so a field of that name would make ``<array_id>/labels/<name>``
    ambiguous. Reserved rather than marked with an ``@``, which would have
    moved every stored ``array_id`` -- ``rois.array_id`` included.
    """

    def test_the_reserved_word_is_refused_whatever_its_case(self):
        from biopb_tensor_server.adapters._writable import unsafe_field_name

        for name in ["labels", "Labels", "LABELS"]:
            assert "reserved" in (unsafe_field_name(name) or "")

    def test_an_ordinary_field_is_accepted(self):
        from biopb_tensor_server.adapters._writable import unsafe_field_name

        for name in ["nuclei", "labelled", "labels2", "my field"]:
            assert unsafe_field_name(name) is None

    def test_a_field_takes_the_store_rules_too(self):
        """It is a path component like any other, minus the extension."""
        from biopb_tensor_server.adapters._writable import unsafe_field_name

        assert unsafe_field_name("CON") is not None
        assert unsafe_field_name("a/b") is not None
        assert unsafe_field_name(".hidden") is not None

    def test_a_set_named_labels_stays_legal(self):
        """The parse is right-to-left, so ``<field>/labels/labels`` is
        unambiguous -- only the *field* is reserved."""
        from biopb_tensor_server.core.labels import split_label_field

        parsed = split_label_field("labels/labels")
        assert parsed is not None
        assert parsed.name == "labels"


class TestTheBootSweepDropsTheLegacyRow:
    """``ome_zarr:`` is gone (``docs/upload-model.md``, Migration), and its
    stores sit directly under ``write_dir`` where nothing can adopt them. A
    persisted catalog still carries the rows they wrote, and nothing else will
    drop them -- write_dir is outside every discovery root, so the reconciler
    never sees these ids."""

    @staticmethod
    def _manager(write_dir, db):
        from biopb_tensor_server.core.source_registry import SourceRegistry
        from biopb_tensor_server.serving.upload_manager import UploadManager

        return UploadManager(SourceRegistry(), write_dir, db)

    @staticmethod
    def _legacy_store(write_dir: Path, name: str, state: str) -> str:
        """A store as the removed kind left it, plus the row it had written."""
        store = write_dir / f"{name}.zarr"
        store.mkdir(parents=True)
        (store / ".zattrs").write_text(
            json.dumps({"biopb": {"upload": {"state": state}}})
        )
        return OmeZarrAdapter.upload_source_id(store)

    def test_the_rows_go_and_only_the_unfinished_bytes_do(self, tmp_path):
        from biopb_tensor_server.serving.metadata_db import MetadataDatabase

        write_dir = tmp_path / "w"
        write_dir.mkdir()
        db = MetadataDatabase()
        crashed = self._legacy_store(write_dir, "crashed", "pending")
        done = self._legacy_store(write_dir, "done", "ready")
        for source_id, store in ((crashed, "crashed"), (done, "done")):
            db._get_cursor().execute(
                "INSERT INTO sources (source_id, source_url, source_type) "
                "VALUES (?, ?, 'ome_zarr')",
                [source_id, str(write_dir / f"{store}.zarr")],
            )
        assert _catalog_ids(db) == {crashed, done}

        assert self._manager(write_dir, db).discard_unfinished_stores() == 1

        # The unfinished store's bytes go; the finished one's are the user's
        # data now, left where they are for a discovery root to pick up.
        assert not (write_dir / "crashed.zarr").exists()
        assert (write_dir / "done.zarr").is_dir()
        # Neither row has an adapter behind it any more, so both go.
        assert _catalog_ids(db) == set()


class TestTheBootSweepRemovesAPendingMember:
    def test_a_crashed_member_is_removed_and_a_published_one_adopted(self, tmp_path):
        """The same walk that removes what a crash left also adopts what it
        did not: the two halves of one pass over ``sources/*/*``."""
        server = _restart_server(tmp_path)
        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")
            source = client.register_source("coll")
            crashed = _create(client, source, "crashed")
            _put(client, crashed)
            done = _create(client, source, "done")
            _put(client, done)
            client.set_upload_status(done, "READY")
            collection = server.sources.get(source).store
        finally:
            server.shutdown()
        assert (collection / "crashed").is_dir()

        second = _restart_server(tmp_path)
        try:
            assert not (collection / "crashed").exists()
            assert _marker(collection / "done") == "ready"
            adopted = second.sources.get(source)
            assert [d.array_id for d in adopted.list_tensor_descriptors()] == [
                f"{source}/done"
            ]
        finally:
            second.shutdown()
            CacheManager.reset()

    def test_no_write_dir_means_nothing_to_sweep(self):
        from biopb_tensor_server.core.source_registry import SourceRegistry
        from biopb_tensor_server.serving.upload_manager import UploadManager

        manager = UploadManager(SourceRegistry(), None, None)
        assert manager.discard_unfinished_stores() == 0


def _restart_server(tmp_path):
    CacheManager.reset()
    CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
    server = catalog_server(
        location="grpc://localhost:0", writable=True, write_dir=tmp_path / "w"
    )
    server.mark_ready()
    threading.Thread(target=server.serve, daemon=True).start()
    return server


class TestTheMarker:
    def test_ready_is_announced_only_after_the_seal(
        self, writable_server, client, source
    ):
        """A reader that has seen READY must find the store kept after a
        crash; so if the marker cannot be written, the transition fails and the
        upload stays PENDING for a retry."""
        desc = _create(client, source)
        _put(client, desc)
        adapter = writable_server.sources.get(source).members["durable"]
        store = _store(writable_server, source)

        def refuse(state):
            raise OSError("disk full")

        adapter._write_upload_state = refuse
        with pytest.raises(flight.FlightServerError, match="could not publish"):
            client.set_upload_status(desc, "READY")
        assert client.get_upload_status(desc.array_id)["state"] == "PENDING"
        assert _marker(store) == "pending"

        del adapter._write_upload_state
        assert client.set_upload_status(desc, "READY")["state"] == "READY"
        assert _marker(store) == "ready"

    def test_a_transition_after_discard_reports_the_discard(
        self, writable_server, client, source
    ):
        """The marker write on a removed store fails on the missing file; what
        the caller learns is the discard, not the missing file."""
        desc = _create(client, source)
        writable_server.uploads.discard(desc.array_id, "gone")

        with pytest.raises(UploadRefused, match="gone") as exc:
            client.set_upload_status(desc, "READY")
        assert exc.value.state == "DISCARDED"

    def test_pending_from_create_and_ready_once_published(
        self, writable_server, client, source
    ):
        desc = _create(client, source)
        store = _store(writable_server, source)
        assert _marker(store) == "pending"

        _put(client, desc)
        client.set_upload_status(desc, "READY")

        assert _marker(store) == "ready"
        # The rest of the metadata is untouched by the rewrite.
        zattrs = json.loads((store / ".zattrs").read_text())
        assert "multiscales" in zattrs

    def test_the_metadata_a_member_carries_is_its_own_ngff(
        self, writable_server, client, source
    ):
        """A member declares shape, dtype, grid and axes and nothing else: the
        OME block is source-scoped and rode in on ``register_source``."""
        desc = _create(client, source)
        zattrs = json.loads((_store(writable_server, source) / ".zattrs").read_text())
        axes = zattrs["multiscales"][0]["axes"]
        assert [a["name"] for a in axes] == ["dim0", "dim1"]
        assert desc.array_id == f"{source}/durable"


class TestDiscoveryDeclinesAPendingStore:
    def test_both_claims_decline_until_published(self, writable_server, client, source):
        """Two lines of defence, and this is the second: nothing under
        ``write_dir`` is discovered at all. If a ``write_dir`` were misplaced
        inside a root, a collection is still a plain ``.zarr`` group and
        nothing in its *shape* would stop a claim taking it -- so the claims
        recognize the subsystem's own block and decline, published or not."""
        desc = _create(client, source)
        collection = writable_server.sources.get(source).store
        member = _store(writable_server, source)

        for store in (collection, member):
            ctx = ClaimContext(store)
            assert OmeZarrAdapter.claim(ctx, DiscoveryState()) is None
            assert ZarrAdapter.claim(ctx, DiscoveryState()) is None

        _put(client, desc)
        client.set_upload_status(desc, "READY")

        # Publishing changes nothing here: the collection is the upload
        # subsystem's for as long as it exists, not just while it is filling.
        assert ZarrAdapter.claim(ClaimContext(collection), DiscoveryState()) is None

    def test_a_users_own_zarr_group_is_unaffected(self, tmp_path):
        """The decline keys on the block the subsystem writes, so an ordinary
        group laid out the same way is still claimed."""
        store = tmp_path / "theirs.zarr"
        store.mkdir()
        (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (store / ".zattrs").write_text(json.dumps({"mine": "not the server's"}))

        assert ZarrAdapter.claim(ClaimContext(store), DiscoveryState()) is not None


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
