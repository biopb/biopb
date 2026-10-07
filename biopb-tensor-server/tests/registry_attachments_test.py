"""Attached tensors belong to the registry: they outlive the adapter that serves them."""

from biopb_tensor_server.adapters.scratch import ScratchSource
from biopb_tensor_server.core.source_registry import SourceRegistry, close_adapter


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
