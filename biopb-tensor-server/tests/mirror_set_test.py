"""MirrorSet and the bulk row write beneath it: a mirror's row is the upstream's."""

from __future__ import annotations

import pytest
from biopb_tensor_server.adapters.remote_tensor import UpstreamVersion
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.errors import SourceUnresolvedError
from biopb_tensor_server.serving.metadata_db import MetadataDatabase, MirroredRow
from biopb_tensor_server.sources import mirror as mirror_module
from biopb_tensor_server.sources.mirror import MirrorSet
from biopb_tensor_server.sources.roots import Root, RootKind

ROOT_ID = "up1"


def _tensor(array_id="img", shape=(4, 4)):
    return {
        "array_id": array_id,
        "dim_labels": ["y", "x"],
        "shape": list(shape),
        "dtype": "uint8",
    }


def _mirrored(source_id, *, rel=None, metadata_json=None, resolved=True, tensors=None):
    return MirroredRow(
        source_id,
        rel or f"data/{source_id}.tif",
        metadata_json,
        resolved,
        [_tensor(source_id)] if tensors is None else tensors,
    )


def _db():
    db = MetadataDatabase()
    db.ensure_root(ROOT_ID, "grpc://lab")
    return db


def _row(db, source_id):
    (row,) = (
        db.query(
            "SELECT source_url, source_type, metadata_json, is_resolved, tensors, "
            f"indexed_at FROM sources WHERE source_id = '{source_id}'"
        )
        .to_pandas()
        .to_dict("records")
    )
    return row


class TestSyncMirroredRows:
    def test_a_row_is_the_upstreams_beneath_its_root(self):
        db = _db()
        db.sync_mirrored_rows(
            ROOT_ID,
            "tensor-server",
            [_mirrored("lab__a", metadata_json='{"ome": {"k": 1}}')],
        )

        row = _row(db, "lab__a")
        assert row["source_url"] == "grpc://lab/data/lab__a.tif"
        assert row["source_type"] == "tensor-server"
        assert row["metadata_json"] == '{"ome": {"k": 1}}'
        assert bool(row["is_resolved"]) is True
        assert [t["array_id"] for t in row["tensors"]] == ["lab__a"]
        assert db.get_metadata_json("lab__a") == {"ome": {"k": 1}}

    def test_a_batch_may_mix_sources_with_and_without_tensors(self):
        db = _db()
        db.sync_mirrored_rows(
            ROOT_ID,
            "tensor-server",
            [
                _mirrored("lab__a"),
                _mirrored("lab__cloud", resolved=False, tensors=[]),
            ],
        )

        assert len(_row(db, "lab__a")["tensors"]) == 1
        cloud = _row(db, "lab__cloud")
        assert len(cloud["tensors"]) == 0
        assert bool(cloud["is_resolved"]) is False

    def test_a_batch_larger_than_a_statement_is_written_whole(self):
        db = _db()
        n = db._PENDING_CHUNK * 2 + 7
        db.sync_mirrored_rows(
            ROOT_ID,
            "tensor-server",
            [_mirrored(f"lab__s{i}") for i in range(n)],
        )

        assert db.query("SELECT count(*) AS n FROM sources").to_pylist() == [{"n": n}]

    def test_a_row_the_upstream_did_not_change_is_left_alone(self):
        db = _db()
        rows = [_mirrored("lab__a", metadata_json='{"a": 1}')]
        db.sync_mirrored_rows(ROOT_ID, "tensor-server", rows)
        written = _row(db, "lab__a")["indexed_at"]

        db.sync_mirrored_rows(ROOT_ID, "tensor-server", rows)

        assert _row(db, "lab__a")["indexed_at"] == written

    @pytest.mark.parametrize(
        "changed",
        [
            {"metadata_json": '{"a": 2}'},
            {"resolved": False},
            {"tensors": [_tensor("lab__a", shape=(8, 8))]},
            {"rel": "elsewhere/a.tif"},
        ],
    )
    def test_a_row_the_upstream_changed_is_rewritten(self, changed):
        db = _db()
        first = _mirrored("lab__a", metadata_json='{"a": 1}')
        db.sync_mirrored_rows(ROOT_ID, "tensor-server", [first])
        written = _row(db, "lab__a")["indexed_at"]

        db.sync_mirrored_rows(
            ROOT_ID, "tensor-server", [first._replace(**_fields(changed))]
        )

        assert _row(db, "lab__a")["indexed_at"] > written

    def test_nothing_to_write_is_nothing(self):
        _db().sync_mirrored_rows(ROOT_ID, "tensor-server", [])


class TestSyncMirroredRemoved:
    def _ids(self, db):
        return sorted(
            r["source_id"]
            for r in db.query("SELECT source_id FROM sources").to_pylist()
        )

    def test_it_drops_the_named_rows_and_no_others(self):
        db = _db()
        db.sync_mirrored_rows(
            ROOT_ID,
            "tensor-server",
            [_mirrored("lab__a"), _mirrored("lab__b"), _mirrored("lab__c")],
        )

        db.sync_mirrored_removed(["lab__a", "lab__c"])

        assert self._ids(db) == ["lab__b"]

    def test_a_list_longer_than_a_statement_is_dropped_whole(self):
        db = _db()
        n = db._PENDING_CHUNK * 2 + 7
        ids = [f"lab__s{i}" for i in range(n)]
        db.sync_mirrored_rows(
            ROOT_ID, "tensor-server", [_mirrored(i) for i in ids] + [_mirrored("keep")]
        )

        db.sync_mirrored_removed(ids)

        assert self._ids(db) == ["keep"]

    def test_an_id_with_no_row_is_not_an_error(self):
        db = _db()
        db.sync_mirrored_rows(ROOT_ID, "tensor-server", [_mirrored("lab__a")])

        db.sync_mirrored_removed(["nobody", "lab__a"])

        assert self._ids(db) == []

    def test_nothing_to_drop_is_nothing(self):
        _db().sync_mirrored_removed([])


def _fields(changed):
    return {("is_resolved" if k == "resolved" else k): v for k, v in changed.items()}


# ----------------------------------------------------------------- MirrorSet


class _Registry(dict):
    pass


class _Server:
    def __init__(self):
        self.sources = _Registry()
        self.fail_register = False

    def register_source(self, source_id, adapter, evictable=False):
        if self.fail_register:
            raise RuntimeError("no room")
        self.sources[source_id] = adapter
        return adapter

    def unregister_source(self, source_id):
        self.sources.pop(source_id)


class _Upstream:
    """What the module's upstream calls see: ``{upstream id: row}``."""

    def __init__(self, monkeypatch):
        self.rows = {}
        self.fetched = []
        monkeypatch.setattr(mirror_module, "open_upstream_client", lambda *a: object())
        monkeypatch.setattr(mirror_module, "close_upstream_client", lambda c: None)
        monkeypatch.setattr(
            mirror_module, "resolve_upstream_credentials", lambda *a: None
        )
        monkeypatch.setattr(mirror_module, "list_upstream_versions", self._versions)
        monkeypatch.setattr(mirror_module, "fetch_upstream_rows", self._fetch)

    def put(self, upstream_id, *, indexed_at=1, metadata_json=None, resolved=True):
        self.rows[upstream_id] = {
            "source_id": upstream_id,
            "source_url": f"file:///data/{upstream_id}.tif",
            "source_type": "ome-tiff",
            "metadata_json": metadata_json,
            "is_resolved": resolved,
            "tensors": [_tensor(upstream_id)] if resolved else [],
            "indexed_at": indexed_at,
        }

    def _versions(self, client):
        return {
            up: UpstreamVersion(r["indexed_at"], len(r["metadata_json"] or ""))
            for up, r in self.rows.items()
        }

    def _fetch(self, client, ids, sizes):
        self.fetched += list(ids)
        if ids:
            yield [self.rows[i] for i in ids]


@pytest.fixture
def upstream(monkeypatch):
    return _Upstream(monkeypatch)


def _mirrors(db=None, server=None, is_claimed=lambda source_id: False):
    root = Root.from_config(
        SourceConfig(url="grpc://lab:8815", alias="lab"), RootKind.UPSTREAM
    )
    server = server or _Server()
    ensure_root = (
        (lambda r: db.ensure_root(r.root_id, r.root_url))
        if db is not None
        else (lambda r: None)
    )
    return MirrorSet(root, server, db, is_claimed, ensure_root), server


class TestMirrorSet:
    def test_the_first_relist_catalogues_every_source_and_builds_no_adapter(
        self, upstream
    ):
        upstream.put("a", metadata_json='{"ome": 1}')
        upstream.put("b", resolved=False)
        db = MetadataDatabase()
        mirrors, server = _mirrors(db)

        assert mirrors.relist(None) is True

        assert server.sources == {}
        assert mirrors.owns("lab__a") and mirrors.owns("lab__b")
        assert mirrors.unbuilt() == 2
        rows = db.query("SELECT source_id, source_url FROM sources").to_pylist()
        assert {r["source_id"]: r["source_url"] for r in rows} == {
            "lab__a": "grpc://lab/data/a.tif",
            "lab__b": "grpc://lab/data/b.tif",
        }
        assert db.get_metadata_json("lab__a") == {"ome": 1}

    def test_a_mirror_is_built_from_its_row_on_demand(self, upstream):
        upstream.put("a", indexed_at=7)
        db = MetadataDatabase()
        mirrors, server = _mirrors(db)
        mirrors.relist(None)

        assert mirrors.materialize("lab__a") is True

        adapter = server.sources["lab__a"]
        assert adapter.catalog_url == "grpc://lab/data/a.tif"
        assert [t.array_id for t in adapter.list_tensors()] == ["lab__a"]
        assert adapter.is_current(7)
        assert mirrors.unbuilt() == 0
        assert mirrors.materialize("lab__a") is True
        assert server.sources["lab__a"] is adapter  # not built twice

    def test_a_row_the_upstream_has_not_resolved_gets_no_adapter(self, upstream):
        upstream.put("a", resolved=False)
        db = MetadataDatabase()
        mirrors, server = _mirrors(db)
        mirrors.relist(None)
        assert _row(db, "lab__a")["is_resolved"] is False

        with pytest.raises(SourceUnresolvedError):
            mirrors.materialize("lab__a")

        assert server.sources == {}
        assert _row(db, "lab__a")["is_resolved"] is False  # the row's to say

    def test_an_id_the_set_does_not_mirror_is_not_built(self, upstream):
        mirrors, server = _mirrors(MetadataDatabase())

        assert mirrors.materialize("nope") is False
        assert server.sources == {}

    def test_a_failed_build_leaves_the_mirror_unbuilt_and_is_retried(self, upstream):
        upstream.put("a")
        mirrors, server = _mirrors(MetadataDatabase())
        mirrors.relist(None)
        server.fail_register = True

        with pytest.raises(RuntimeError):
            mirrors.materialize("lab__a")
        assert mirrors.unbuilt() == 1

        server.fail_register = False
        assert mirrors.materialize("lab__a") is True

    def test_a_relist_that_finds_nothing_new_fetches_nothing(self, upstream):
        upstream.put("a")
        mirrors, _ = _mirrors(MetadataDatabase())
        mirrors.relist(None)
        upstream.fetched.clear()

        assert mirrors.relist(None) is False
        assert upstream.fetched == []

    def test_a_source_the_upstream_dropped_goes_from_registry_and_catalog(
        self, upstream
    ):
        upstream.put("a")
        upstream.put("b")
        db = MetadataDatabase()
        mirrors, server = _mirrors(db)
        mirrors.relist(None)
        mirrors.materialize("lab__b")

        del upstream.rows["a"]  # never built: nothing to unregister

        assert mirrors.relist(None) is True
        assert not mirrors.owns("lab__a")
        assert list(server.sources) == ["lab__b"]
        assert db.query("SELECT source_id FROM sources").to_pylist() == [
            {"source_id": "lab__b"}
        ]

    def test_a_source_that_will_not_unregister_stays_and_is_tried_again(self, upstream):
        upstream.put("a")
        upstream.put("b")
        db = MetadataDatabase()
        mirrors, server = _mirrors(db)
        mirrors.relist(None)
        mirrors.materialize("lab__a")
        mirrors.materialize("lab__b")
        del upstream.rows["a"]
        del upstream.rows["b"]
        real = server.unregister_source

        def stuck(source_id):
            if source_id == "lab__a":
                raise RuntimeError("busy")
            real(source_id)

        server.unregister_source = stuck
        mirrors.relist(None)

        assert list(server.sources) == ["lab__a"]
        assert db.query("SELECT source_id FROM sources").to_pylist() == [
            {"source_id": "lab__a"}
        ]

        server.unregister_source = real
        assert mirrors.relist(None) is True
        assert list(server.sources) == []

    def test_a_failed_catalog_delete_does_not_fail_the_relist(
        self, upstream, monkeypatch
    ):
        upstream.put("a")
        db = MetadataDatabase()
        mirrors, server = _mirrors(db)
        mirrors.relist(None)
        del upstream.rows["a"]

        def broken(*args):
            raise RuntimeError("disk full")

        monkeypatch.setattr(db, "sync_mirrored_removed", broken)

        assert mirrors.relist(None) is True
        assert list(server.sources) == []

    def test_a_source_the_upstream_re_registered_is_read_again(self, upstream):
        upstream.put("a", indexed_at=1, metadata_json='{"v": 1}')
        db = MetadataDatabase()
        mirrors, server = _mirrors(db)
        mirrors.relist(None)
        mirrors.materialize("lab__a")
        adapter = server.sources["lab__a"]
        upstream.fetched.clear()

        upstream.put("a", indexed_at=2, metadata_json='{"v": 2}')

        assert mirrors.relist(None) is False  # the set did not change
        assert upstream.fetched == ["a"]
        assert db.get_metadata_json("lab__a") == {"v": 2}
        assert server.sources["lab__a"] is adapter
        assert adapter.is_current(2)

    def test_a_mirror_never_built_is_built_at_the_version_it_was_last_synced(
        self, upstream
    ):
        upstream.put("a", indexed_at=1)
        mirrors, server = _mirrors(MetadataDatabase())
        mirrors.relist(None)
        upstream.put("a", indexed_at=2)
        mirrors.relist(None)

        mirrors.materialize("lab__a")

        assert server.sources["lab__a"].is_current(2)

    def test_an_unversioned_upstream_is_read_every_time_but_not_rewritten(
        self, upstream
    ):
        upstream.put("a", indexed_at=None, metadata_json='{"v": 1}')
        db = MetadataDatabase()
        mirrors, _ = _mirrors(db)
        mirrors.relist(None)
        written = _row(db, "lab__a")["indexed_at"]
        upstream.fetched.clear()

        mirrors.relist(None)

        assert upstream.fetched == ["a"]
        assert _row(db, "lab__a")["indexed_at"] == written

    def test_a_source_id_a_claim_holds_is_not_taken_over(self, upstream, caplog):
        upstream.put("a")
        upstream.put("b")
        mirrors, server = _mirrors(is_claimed=lambda source_id: source_id == "lab__a")

        with caplog.at_level("WARNING"):
            mirrors.relist(None)

        assert mirrors.owns("lab__b") and not mirrors.owns("lab__a")
        assert "Not mirroring a" in caplog.text

    def test_a_failed_catalog_write_leaves_nothing_recorded_and_retries(
        self, upstream, monkeypatch
    ):
        upstream.put("a")
        db = MetadataDatabase()
        mirrors, server = _mirrors(db)

        def broken(*args):
            raise RuntimeError("disk full")

        monkeypatch.setattr(db, "sync_mirrored_rows", broken)
        assert mirrors.relist(None) is False
        assert not mirrors.owns("lab__a")

        monkeypatch.delattr(db, "sync_mirrored_rows")
        assert mirrors.relist(None) is True
        assert mirrors.owns("lab__a")

    def test_without_a_catalog_the_sources_are_still_served(self, upstream):
        upstream.put("a")
        mirrors, server = _mirrors(None)

        assert mirrors.relist(None) is True
        assert mirrors.materialize("lab__a") is True
        assert list(server.sources) == ["lab__a"]

    def test_a_mirror_whose_upstream_path_is_unknown_sits_beneath_its_id(
        self, upstream
    ):
        upstream.put("a")
        upstream.rows["a"]["source_url"] = None
        db = MetadataDatabase()
        mirrors, _ = _mirrors(db)

        mirrors.relist(None)

        assert db.query("SELECT source_url FROM sources").to_pylist() == [
            {"source_url": "grpc://lab/a"}
        ]
