"""Claim-API contract regressions: ClaimContext in, string keys out.

Two shapes of the same bug, both from passing a `Path` where the claim protocol
wants something else. `get_claims_for_path` takes a `ClaimContext` (adapters read
`is_remote` off it), not a bare `Path`; and `DiscoveryState`'s path maps are keyed
by `str`, so a `Path` lookup silently misses and a deleted file never leaves the
catalog. Also pinned here: one adapter raising mid-walk does not abort the others,
and `_type_to_adapter` is filled at register() time rather than back-filled as a
side effect of a claim query.
"""

import os
import tempfile

import numpy as np
import tifffile
from biopb_tensor_server.adapters import get_default_registry


class TestClaimContextIsRequired:
    """`get_claims_for_path` takes a ClaimContext, never a Path."""

    def test_get_claims_for_path_accepts_claim_context(self):
        """Direct test that get_claims_for_path works with ClaimContext.

        This verifies the API contract directly, ensuring adapters receive
        ClaimContext objects with the is_remote property.
        """
        from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a TIFF file
            tiff_path = os.path.join(tmpdir, "test.tif")
            data = np.random.randint(0, 255, (64, 64), dtype=np.uint16)
            tifffile.imwrite(tiff_path, data)

            # Create ClaimContext (local path, so is_remote=False)
            ctx = ClaimContext(tiff_path)
            assert ctx.is_remote is False
            assert ctx.is_file() is True

            # Get claims - should not raise 'PosixPath' has no 'is_remote'
            registry = get_default_registry()
            state = DiscoveryState()
            claims = registry.get_claims_for_path(ctx, state)

            assert len(claims) >= 1
            assert claims[0].source_type is not None


class TestDeleteSourceRegression:
    """Regression tests for source deletion Path vs str bug."""

    def test_path_to_source_mapping_uses_string_keys(self):
        """Test that path_to_source uses string keys, not Path objects.

        This verifies the internal mapping convention that caused the bug.
        get_source_for_path() and remove_claim() expect string keys.
        """
        from biopb_tensor_server.core.discovery import DiscoveryState, SourceClaim

        state = DiscoveryState()

        # Add a claim manually with string path
        claim = SourceClaim(
            source_type="test",
            primary_path="/tmp/test.tif",
            source_id="test_123",
        )
        state.consumed_paths.discard(
            "/tmp/test.tif"
        )  # Remove from consumed to allow add
        state.add_claim(claim)

        # Verify string lookup works
        assert state.get_source_for_path("/tmp/test.tif") == "test_123"

        # Verify Path object lookup does NOT work (demonstrates the bug pattern)
        from pathlib import Path

        assert state.get_source_for_path(Path("/tmp/test.tif")) is None

    def test_remove_claim_requires_string_key(self):
        """Test that remove_claim expects string key.

        This verifies that remove_claim() properly handles string keys.
        """
        from biopb_tensor_server.core.discovery import DiscoveryState, SourceClaim

        state = DiscoveryState()

        # Add a claim manually
        claim = SourceClaim(
            source_type="test",
            primary_path="/tmp/test.tif",
            source_id="test_123",
        )
        state.consumed_paths.discard("/tmp/test.tif")
        state.add_claim(claim)

        assert len(state.claims) == 1

        # Remove with string (correct)
        removed_id = state.remove_claim("/tmp/test.tif")
        assert removed_id == "test_123"
        assert len(state.claims) == 0

    def test_handle_deleted_path_to_str_conversion(self):
        """Test that Path.resolve() result must be converted to str for lookup.

        This regression test directly demonstrates the fix needed in source_manager.
        The _handle_deleted method was passing Path object to get_source_for_path
        and remove_claim, but they expect string keys.
        """
        from pathlib import Path

        from biopb_tensor_server.core.discovery import DiscoveryState, SourceClaim

        state = DiscoveryState()

        # Simulate a path being resolved
        test_path = Path("/tmp/test.tif")
        resolved = test_path.resolve()

        # Add claim with the resolved string path
        claim = SourceClaim(
            source_type="test",
            primary_path=str(resolved),
            source_id="test_456",
        )
        state.consumed_paths.discard(str(resolved))
        state.add_claim(claim)

        # BUG: Passing Path object would fail
        # source_id = state.get_source_for_path(resolved)  # Path object - returns None!
        # assert source_id is None

        # FIX: Convert to string first
        source_id = state.get_source_for_path(str(resolved))  # String - works!
        assert source_id == "test_456"

        # Remove also requires string
        removed = state.remove_claim(str(resolved))
        assert removed == "test_456"


class TestCreatedSourceRegression:
    """Regression tests for source creation using ClaimContext."""

    def test_handle_created_uses_claim_context(self):
        """Test that _handle_created uses ClaimContext for get_claims_for_path.

        Before the fix, _handle_created was passing Path directly to
        get_claims_for_path(), causing 'PosixPath' object has no attribute 'is_remote'.
        """
        from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

        with tempfile.TemporaryDirectory() as tmpdir:
            tiff_path = os.path.join(tmpdir, "test.tif")
            data = np.random.randint(0, 255, (64, 64), dtype=np.uint16)
            tifffile.imwrite(tiff_path, data)

            # Create ClaimContext (the correct API)
            ctx = ClaimContext(tiff_path)
            assert ctx.is_remote is False
            assert ctx.is_file() is True

            # get_claims_for_path must accept ClaimContext, not Path
            registry = get_default_registry()
            state = DiscoveryState()
            claims = registry.get_claims_for_path(ctx, state)

            assert len(claims) >= 1
            assert claims[0].source_type == "tiff"


class _RaisingOnBadPathAdapter:
    @classmethod
    def claim(cls, ctx, state):
        if ctx.name == "bad.dat":
            raise RuntimeError("boom")
        return None


class _ClaimsGoodPathAdapter:
    @classmethod
    def claim(cls, ctx, state):
        from biopb_tensor_server.core.discovery import SourceClaim

        if ctx.is_file() and ctx.name == "good.dat":
            if not state.try_claim_path(ctx.path_str):
                return None
            return SourceClaim(source_type="good", primary_path=ctx.path_str)
        return None


class TestDiscoveryFailureIsolation:
    def test_discover_sources_continues_after_claim_exception_on_one_path(self):
        from pathlib import Path

        from biopb_tensor_server.core.discovery import (
            AdapterRegistry,
            discover_sources as discover_tree_sources,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "bad.dat").write_text("bad")
            (root / "good.dat").write_text("good")

            registry = AdapterRegistry()
            registry.register(_RaisingOnBadPathAdapter)
            registry.register(_ClaimsGoodPathAdapter)

            state = discover_tree_sources(root, registry)

            assert len(state.claims) == 1
            claim = next(iter(state.claims.values()))
            assert claim.primary_path == str(root / "good.dat")
            assert str(root / "bad.dat") not in state.path_to_source


class TestRegistryTypeMap:
    """`_type_to_adapter` is filled at register() time and `get_claims_for_path`
    is a pure query -- not a per-path mutation that back-fills the map as a side
    effect (biopb/biopb#555)."""

    def _registry(self):
        from biopb_tensor_server.core.discovery import AdapterRegistry

        return AdapterRegistry()

    def test_type_resolvable_before_any_claim(self):
        # A typed registration resolves immediately, with no claim first -- the
        # property the lazy-resolve / cloud phase-2 flow relies on.
        registry = self._registry()
        registry.register(_ClaimsGoodPathAdapter, "good")
        assert registry.get_adapter_for_type("good") is _ClaimsGoodPathAdapter

    def test_register_multiple_types_maps_all_to_one_class(self):
        registry = self._registry()
        registry.register(_ClaimsGoodPathAdapter, ["good", "good-alias"])
        assert registry.get_adapter_for_type("good") is _ClaimsGoodPathAdapter
        assert registry.get_adapter_for_type("good-alias") is _ClaimsGoodPathAdapter
        # Registered once, so it occupies a single probe slot (no duplicate claim).
        assert registry._adapters.count(_ClaimsGoodPathAdapter) == 1

    def test_untyped_registration_is_not_resolvable_by_type(self):
        registry = self._registry()
        registry.register(_ClaimsGoodPathAdapter)  # claim-only, no type
        assert registry.get_adapter_for_type("good") is None

    def test_get_claims_for_path_does_not_mutate_type_map(self, tmp_path):
        # The query must not back-fill the type map: an untyped adapter stays
        # unresolvable by type even after it wins a claim.
        from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

        registry = self._registry()
        registry.register(_ClaimsGoodPathAdapter)  # no type

        good = tmp_path / "good.dat"
        good.write_text("good")
        claims = registry.get_claims_for_path(ClaimContext(good), DiscoveryState())

        assert claims and claims[0].source_type == "good"
        assert registry.get_adapter_for_type("good") is None
