"""``Attachments`` judges a label set where things change, and a read is a lookup.

The parent's images are snapshotted by ``rebind`` (the registry calls it on every
registration of the source); the verdict is kept until the images or the
attached fields move.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from biopb_tensor_server.core.adapter_base import TensorEntry
from biopb_tensor_server.core.attachments import Attachments
from biopb_tensor_server.core.errors import AttachedTensorMismatch
from biopb_tensor_server.sources.source_registry import SourceRegistry

SRC = "src"
FIELD = "@fields/raw"
SET = "@labels/n"
FIELD_SET = f"{FIELD}/@labels/n"


class _Parent:
    def __init__(self, shape=(64, 64)):
        self.shape = shape

    def list_tensors(self):
        return [TensorEntry(SRC, ("y", "x"), self.shape, "uint8")]


class _Tensor:
    """What ``Attachments`` asks of an attached tensor."""

    def __init__(self, array_id, shape, readable=True, token=None):
        self.array_id = array_id
        self.shape = shape
        self.capability_token = token
        self.upload = SimpleNamespace(is_readable=readable)

    def get_tensor_descriptor(self):
        return SimpleNamespace(
            array_id=self.array_id,
            dim_labels=["y", "x"],
            shape=list(self.shape),
            dtype="uint32",
        )

    def get_tensor_adapter(self, tensor_id):
        return ("level", tensor_id)


def _bound(parent):
    """An ``Attachments`` whose source's adapter is *parent* (None: not yet)."""
    holder = SimpleNamespace(parent=parent)
    att = Attachments(SRC, lambda: holder.parent)
    att.holder = holder
    return att


def _rebind(att, parent):
    """The registry registered *parent* in place of the adapter."""
    att.holder.parent = parent
    att.rebind()


def _set(shape=(64, 64), **kw):
    return _Tensor(f"{SRC}/{SET}", shape, **kw)


def _field(**kw):
    return _Tensor(f"{SRC}/{FIELD}", (64, 64), **kw)


class TestRebindJudgesTheSets:
    def test_a_set_that_fits_is_readable(self):
        att = _bound(_Parent())
        att.attach(SET, tensor := _set())
        assert att.error(SET) is None
        assert att.route(SET) is tensor

    def test_a_refresh_to_a_different_image_rejudges_it(self):
        att = _bound(_Parent())
        att.attach(SET, _set())

        _rebind(att, _Parent(shape=(32, 32)))

        assert "does not span" in att.error(SET)
        with pytest.raises(AttachedTensorMismatch):
            att.route(SET)
        assert [f for f, _ in att.listed()] == [SET]  # listed, and refuses a read

        _rebind(att, _Parent())  # the file changed back
        assert att.error(SET) is None
        assert att.route(SET) is not None

    def test_a_set_with_no_image_binds_to_nothing(self):
        att = _bound(_Parent())
        att.attach("Image:9/@labels/n", _Tensor(f"{SRC}/Image:9/@labels/n", (64, 64)))
        assert "binds to no tensor" in att.error("Image:9/@labels/n")

    def test_nothing_is_judged_before_there_is_a_parent(self):
        att = _bound(None)
        att.attach(SET, _set(shape=(1, 1)))
        assert att.error(SET) is None
        _rebind(att, _Parent())
        assert att.error(SET) is not None

    def test_registering_is_what_calls_it(self):
        reg = SourceRegistry()
        reg.attach(SRC, SET, _set())
        reg.register(SRC, _Parent(shape=(32, 32)))
        with pytest.raises(AttachedTensorMismatch):
            reg.resolve_tensor(SRC, SET)
        reg.register(SRC, _Parent())  # a refresh, or a rebuild after eviction
        assert reg.resolve_tensor(SRC, SET) is not None

    def test_a_source_that_returns_rejudges_what_stayed(self):
        reg = SourceRegistry()
        reg.register(SRC, _Parent())
        reg.attach(SRC, SET, _set())
        reg.unregister(SRC)  # attachments stay: the id may come back
        reg.register(SRC, _Parent(shape=(32, 32)))
        assert "does not span" in reg.attached_to(SRC).error(SET)

    def test_unregistering_clears_the_verdict(self):
        reg = SourceRegistry()
        reg.register(SRC, _Parent(shape=(32, 32)))
        reg.attach(SRC, SET, _set())
        assert reg.attached_to(SRC).error(SET) is not None
        reg.unregister(SRC)
        assert reg.attached_to(SRC).error(SET) is None

    def test_judging_a_set_does_not_keep_an_evictable_adapter_alive(self):
        import gc
        import weakref

        reg = SourceRegistry()
        parent = _Parent()
        gone = weakref.ref(parent)
        reg.register(SRC, parent, evictable=True)
        reg.attach(SRC, SET, _set())  # reads the parent's images
        assert reg.attached_to(SRC).error(SET) is None

        reg.release_idle(0, now=float("inf"))
        del parent
        gc.collect()
        assert gone() is None


class TestASetBoundToAnUploadedField:
    def test_detaching_the_field_invalidates_the_set_and_attaching_heals_it(self):
        att = _bound(_Parent())
        att.attach(FIELD, _field())
        att.attach(FIELD_SET, _Tensor(f"{SRC}/{FIELD_SET}", (64, 64)))
        assert att.error(FIELD_SET) is None

        att.detach(FIELD)
        assert "binds to no tensor" in att.error(FIELD_SET)
        assert att.get(FIELD_SET) is not None  # marked, not deleted

        att.attach(FIELD, _field())
        assert att.error(FIELD_SET) is None

    def test_a_set_may_arrive_before_its_field(self):
        att = _bound(_Parent())
        att.attach(FIELD_SET, _Tensor(f"{SRC}/{FIELD_SET}", (64, 64)))
        assert att.error(FIELD_SET) is not None
        att.attach(FIELD, _field())
        assert att.error(FIELD_SET) is None

    def test_a_field_still_uploading_is_not_an_image_until_it_is_ready(self):
        att = _bound(_Parent())
        filling = _field(readable=False)
        att.attach(FIELD, filling)
        att.attach(FIELD_SET, _Tensor(f"{SRC}/{FIELD_SET}", (64, 64)))
        assert att.error(FIELD_SET) is not None

        filling.upload.is_readable = True  # the READY transition
        att.revalidate()
        assert att.error(FIELD_SET) is None


class TestConformLabel:
    def test_a_request_with_no_axes_is_given_the_images(self):
        att = _bound(_Parent())
        desc = SimpleNamespace(dim_labels=[], shape=[64, 64])
        assert att.conform_label(SET, desc) is None
        assert desc.dim_labels == ["y", "x"]

    def test_a_set_that_does_not_span_is_refused(self):
        att = _bound(_Parent())
        desc = SimpleNamespace(dim_labels=["y", "x"], shape=[32, 64])
        assert "does not span" in att.conform_label(SET, desc)

    def test_a_field_that_names_no_set_is_refused(self):
        att = _bound(_Parent())
        desc = SimpleNamespace(dim_labels=["y", "x"], shape=[64, 64])
        assert "does not name a label set" in att.conform_label("1", desc)

    def test_a_level_is_not_a_set(self):
        att = _bound(_Parent())
        desc = SimpleNamespace(dim_labels=["y", "x"], shape=[64, 64])
        assert "does not name a label set" in att.conform_label(f"{SET}/1", desc)


class TestRouting:
    def test_a_level_inherits_its_sets_verdict(self):
        att = _bound(_Parent())
        att.attach(SET, _set(shape=(32, 32)))
        with pytest.raises(AttachedTensorMismatch):
            att.route(f"{SET}/1")

    def test_a_level_rides_with_its_set_and_a_bare_field_is_the_formats(self):
        att = _bound(_Parent())
        att.attach(SET, _set())
        assert att.route(f"{SET}/1") == ("level", f"{SRC}/{SET}/1")
        assert att.route("0") is None
        assert att.route(None) is None

    def test_an_upload_in_flight_routes_but_is_not_listed(self):
        att = _bound(_Parent())
        att.attach(SET, tensor := _set(readable=False))
        assert att.route(SET) is tensor
        assert att.listed() == []

    def test_the_token_is_the_owning_tensors(self):
        att = _bound(_Parent())
        att.attach(SET, _set(token="t"))
        assert att.capability_token(f"{SRC}/{SET}") == "t"
        assert att.capability_token(f"{SRC}/{SET}/1") == "t"
        assert att.capability_token(SRC) is None
