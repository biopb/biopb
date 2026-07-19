"""Enforce the source-level vs tensor-level adapter interface split.

``base.py`` declares two role interfaces -- ``SourceAdapter`` (discover tensors,
read source metadata, hand out tensor adapters) and ``TensorAdapter`` (read a
specific tensor's data / descriptor / chunks / pyramid / physical scale). Their
*declarations* must stay disjoint so a tensor-scoped method can never silently
land on ``SourceAdapter`` again (the scramble these tests guard against).
``base.py`` already asserts the invariant at import time; these tests re-check it
from the test surface and extend it to the concrete adapter registry.

``TensorAdapter`` **subclasses** ``SourceAdapter`` (biopb/biopb#380): a tensor
adapter is a source that can also serve pixels, which is what every concrete
adapter already is. Disjointness is therefore about where a method is written
(``vars(cls)``), not about what an instance can answer -- these tests assert
exactly that, so an override of a source-scoped method onto ``TensorAdapter``
still fails.
"""

from biopb_tensor_server.core.base import (
    _SOURCE_SCOPED_API,
    _TENSOR_SCOPED_API,
    SourceAdapter,
    TensorAdapter,
    _public_api,
)


def test_role_interfaces_are_disjoint():
    """No public method may belong to both role interfaces."""
    assert _public_api(SourceAdapter).isdisjoint(_public_api(TensorAdapter))
    assert _SOURCE_SCOPED_API.isdisjoint(_TENSOR_SCOPED_API)


def test_tensor_role_nests_inside_the_source_role():
    """A TensorAdapter IS-A SourceAdapter, and inherits (never re-declares) it."""
    assert issubclass(TensorAdapter, SourceAdapter)
    assert not issubclass(SourceAdapter, TensorAdapter)
    # Inherited, not copied: the source-scoped names resolve on TensorAdapter but
    # are declared only on SourceAdapter.
    for name in _SOURCE_SCOPED_API:
        assert hasattr(TensorAdapter, name), name
        assert name not in vars(TensorAdapter), name


def test_declared_scopes_match_the_abcs():
    """Each ABC's public API equals its declared scope set (no undeclared drift)."""
    assert _public_api(SourceAdapter) == _SOURCE_SCOPED_API
    assert _public_api(TensorAdapter) == _TENSOR_SCOPED_API


def test_pyramid_and_scale_methods_are_tensor_scoped():
    """The moved methods live on TensorAdapter and are absent from SourceAdapter."""
    moved = {"plan_flight_info", "get_native_pyramid_levels", "has_native_pyramid"}
    assert moved <= _public_api(TensorAdapter)
    assert moved.isdisjoint(_public_api(SourceAdapter))
    # Not merely inherited from a shared base: declared directly on TensorAdapter.
    for name in moved:
        assert name in vars(TensorAdapter)
        assert name not in vars(SourceAdapter)


def test_has_native_pyramid_derives_from_levels_by_default():
    """The base TensorAdapter derives has_native_pyramid from the levels method."""

    class _NoPyramid(TensorAdapter):
        def get_tensor_descriptor(self):  # abstract
            raise NotImplementedError

        def get_data(self, bounds):  # abstract
            raise NotImplementedError

        # Inherited from the SourceAdapter half of the role (biopb/biopb#380).
        @classmethod
        def create_from_config(cls, source, credentials_config=None):  # abstract
            raise NotImplementedError

        def list_tensor_descriptors(self):  # abstract
            raise NotImplementedError

        def get_metadata(self):  # abstract
            raise NotImplementedError

    class _WithPyramid(_NoPyramid):
        def get_native_pyramid_levels(self):
            return ["level0", "level1"]  # non-None -> has a native pyramid

    assert _NoPyramid().has_native_pyramid() is False
    assert _WithPyramid().has_native_pyramid() is True


def test_registered_adapters_fill_both_roles():
    """Every concrete adapter in the registry is both a SourceAdapter and a
    TensorAdapter -- they serve one combined object, now by inheriting the
    nested ``TensorAdapter(SourceAdapter)`` rather than by multiple
    inheritance."""
    from biopb_tensor_server.adapters import get_default_registry

    registry = get_default_registry()
    adapter_classes = set(registry._adapters)
    assert adapter_classes, "registry should expose concrete adapters"
    for cls in adapter_classes:
        assert issubclass(cls, SourceAdapter), cls
        assert issubclass(cls, TensorAdapter), cls


def test_unresolved_proxy_is_source_only():
    """The unresolved proxy is a SourceAdapter (catalog surface); it does not
    implement the tensor role -- tensor-level calls reach the resolved adapter
    via get_tensor_adapter."""
    from biopb_tensor_server.adapters.unresolved import UnresolvedSourceAdapter

    assert issubclass(UnresolvedSourceAdapter, SourceAdapter)
    assert not issubclass(UnresolvedSourceAdapter, TensorAdapter)
    for name in (
        "plan_flight_info",
        "get_native_pyramid_levels",
        "has_native_pyramid",
    ):
        assert not hasattr(UnresolvedSourceAdapter, name)
