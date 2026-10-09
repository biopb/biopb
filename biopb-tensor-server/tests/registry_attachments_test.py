"""Attached tensors belong to the registry: they outlive the adapter that serves them."""

from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb_tensor_server.adapters.scratch import ScratchSource
from biopb_tensor_server.serving.metadata_db import MetadataDatabase
from biopb_tensor_server.sources.source_registry import SourceRegistry, close_adapter

from tests.test_metadata_db import MockAdapter


class _Tensor:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class TestAttachmentsOutliveTheAdapter:
    def test_a_rebuilt_adapter_serves_what_the_old_one_had(self):
        registry = SourceRegistry()
        old = registry.register("s", ScratchSource())
        tensor = _Tensor()
        registry.attach("s", "@fields/a", tensor)
        assert registry.attached("s", "@fields/a") is tensor

        new, displaced = registry.swap("s", ScratchSource())

        assert registry.attached("s", "@fields/a") is tensor
        assert registry.get("s") is new
        assert displaced is old

    def test_closing_a_displaced_adapter_leaves_the_attachments_open(self):
        registry = SourceRegistry()
        registry.register("s", ScratchSource())
        tensor = _Tensor()
        registry.attach("s", "@fields/a", tensor)

        _, displaced = registry.swap("s", ScratchSource())
        close_adapter(displaced)

        assert tensor.closed == 0

    def test_a_source_that_returns_finds_its_attachments(self):
        registry = SourceRegistry()
        registry.register("s", ScratchSource())
        tensor = _Tensor()
        registry.attach("s", "@fields/a", tensor)

        registry.unregister("s")
        registry.register("s", ScratchSource())

        assert registry.attached("s", "@fields/a") is tensor

    def test_tensors_attached_before_registering_are_carried_in(self):
        registry = SourceRegistry()
        tensor = _Tensor()
        registry.attach("s", "@fields/a", tensor)
        registry.register("s", ScratchSource())

        assert registry.attached("s", "@fields/a") is tensor


class TestAdopt:
    def test_adopted_tensors_wait_for_their_source(self):
        registry = SourceRegistry()
        tensor = _Tensor()
        registry.adopt({"later": {"@fields/a": tensor}})

        assert registry.attachments("later") == {"@fields/a": tensor}
        registry.register("later", ScratchSource())
        assert registry.attached("later", "@fields/a") is tensor

    def test_close_all_closes_them_even_without_a_source(self):
        registry = SourceRegistry()
        tensor = _Tensor()
        registry.adopt({"orphan": {"@fields/a": tensor}})

        registry.close_all()

        assert tensor.closed == 1

    def test_detach_returns_what_was_attached(self):
        registry = SourceRegistry()
        tensor = _Tensor()
        registry.adopt({"s": {"@fields/a": tensor}})

        assert registry.detach("s", "@fields/a") is tensor
        assert registry.detach("s", "@fields/a") is None
        assert registry.attachment_snapshot() == []


class TestTheAttachmentListener:
    def test_it_hears_each_change_to_a_sources_attachments(self):
        heard = []
        registry = SourceRegistry()
        registry.set_attachment_listener(heard.append)
        registry.register("s", ScratchSource())

        registry.attach("s", "@fields/a", _Tensor())
        registry.attachment_changed("s")
        registry.detach("s", "@fields/a")

        assert heard == ["s", "s", "s"]

    def test_detaching_nothing_is_not_a_change(self):
        heard = []
        registry = SourceRegistry()
        registry.set_attachment_listener(heard.append)

        assert registry.detach("s", "@fields/a") is None
        assert heard == []

    def test_adopting_what_the_disk_holds_is_not_a_change(self):
        heard = []
        registry = SourceRegistry()
        registry.set_attachment_listener(heard.append)

        registry.adopt({"s": {"@fields/a": _Tensor()}})

        assert heard == []


class _Published:
    """A published field, as the upload path attaches it."""

    def get_tensor_descriptor(self):
        return TensorDescriptor(
            array_id="s/@fields/a", dim_labels=["y", "x"], shape=[4, 4], dtype="uint8"
        )


def _catalogued(registry):
    db = MetadataDatabase()
    db.bind_registry(registry)
    return db


def _source():
    return MockAdapter("s", "file:///d/s.zarr", "zarr", [4, 4], "uint8")


class TestTheCatalogFollowsTheAttachments:
    def _listed(self, db):
        (row,) = db.query("SELECT tensors FROM sources").to_pylist()
        return [t["array_id"] for t in row["tensors"]]

    def test_a_row_lists_what_is_attached_to_its_source(self):
        registry = SourceRegistry()
        db = _catalogued(registry)
        db.sync_source_added("s", registry.register("s", _source()))
        assert self._listed(db) == ["s"]

        registry.attach("s", "@fields/a", _Published())
        assert self._listed(db) == ["s", "s/@fields/a"]

        registry.detach("s", "@fields/a")
        assert self._listed(db) == ["s"]

    def test_a_source_with_no_row_is_not_given_one(self):
        registry = SourceRegistry()
        db = _catalogued(registry)
        registry.register("s", _source())

        registry.attach("s", "@fields/a", _Published())

        assert db.query("SELECT source_id FROM sources").num_rows == 0

    def test_a_failed_catalog_write_does_not_fail_the_change(self, monkeypatch):
        registry = SourceRegistry()
        db = _catalogued(registry)
        registry.register("s", _source())

        def broken(*args):
            raise RuntimeError("disk full")

        monkeypatch.setattr(db, "relist_tensors", broken)
        registry.attach("s", "@fields/a", _Published())

        assert registry.attached("s", "@fields/a") is not None
