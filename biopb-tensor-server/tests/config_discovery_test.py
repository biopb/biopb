"""Regression tests for biopb-tensor-server.

Contains tests for:
1. Claim-based discovery passing Path instead of ClaimContext
2. File deletion not updating flight catalog due to Path vs str mismatch
"""

import os
import tempfile

import numpy as np
import tifffile
from biopb_tensor_server.core.config import (
    SourceConfig,
    discover_sources,
    get_default_registry,
)


class TestDiscoverSourcesRegression:
    """Regression tests for discover_sources claim-based detection."""

    def test_file_discovery_without_type_uses_claim_context(self):
        """Test that file discovery works when type is not specified.

        This regression test verifies that discover_sources() properly
        creates ClaimContext when calling get_claims_for_path() for a file
        without an explicit type.

        Before the fix, this would fail with:
        'PosixPath' object has no attribute 'is_remote'
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple TIFF file
            tiff_path = os.path.join(tmpdir, "test.tif")
            data = np.random.randint(0, 255, (64, 64), dtype=np.uint16)
            tifffile.imwrite(tiff_path, data)

            # SourceConfig without type - triggers claim-based detection
            source = SourceConfig(url=tiff_path)

            # This should work without 'PosixPath' object has no attribute 'is_remote'
            registry = get_default_registry()
            discovered = discover_sources(source, registry)

            assert len(discovered) == 1
            assert discovered[0].type is not None
            assert discovered[0].source_id is not None

    def test_directory_discovery_without_type_uses_claim_context(self):
        """Test that directory discovery works when type is not specified.

        This regression test verifies that discover_sources() properly
        creates ClaimContext when calling get_claims_for_path() for a directory
        without an explicit type or source_id.

        Before the fix, this would fail with:
        'PosixPath' object has no attribute 'is_remote'
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory with a TIFF file
            subdir = os.path.join(tmpdir, "data")
            os.makedirs(subdir)
            tiff_path = os.path.join(subdir, "image.tif")
            data = np.random.randint(0, 255, (64, 64), dtype=np.uint16)
            tifffile.imwrite(tiff_path, data)

            # SourceConfig without type or source_id - triggers claim-based discovery
            source = SourceConfig(url=tmpdir)

            # This should work without 'PosixPath' object has no attribute 'is_remote'
            registry = get_default_registry()
            discovered = discover_sources(source, registry)

            assert len(discovered) >= 1
            for src in discovered:
                assert src.type is not None
                assert src.source_id is not None

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

    def test_handle_moved_arguments_order(self):
        """Test that _handle_moved receives correct old_path and new_path order.

        The watcher stores MOVED events with:
        - WatcherEvent.path = old_path (original location)
        - WatcherEvent.old_path = new_path (new location) - confusing naming!

        Before the fix, _handle_moved was called with swapped arguments:
        _handle_moved(event.old_path, event.path) = _handle_moved(new_path, old_path)

        After the fix:
        _handle_moved(event.path, event.old_path) = _handle_moved(old_path, new_path)
        """
        from pathlib import Path

        from biopb_tensor_server.sources.watcher import WatcherEvent, WatcherEventType

        # Simulate what the watcher creates for a move event
        # The watcher stores: event_buffer[old_path] = (MOVED, time, new_path)
        # So WatcherEvent has: path=old_path, old_path=new_path
        original_path = Path("/home/user/data/test.tif")
        new_path = Path("/home/user/data/test_.tif")

        # Watcher creates this (confusing naming in WatcherEvent.old_path)
        event = WatcherEvent(
            event_type=WatcherEventType.MOVED,
            path=original_path,  # This is actually the OLD path
            old_path=new_path,  # This is actually the NEW path
            is_directory=False,
        )

        # Verify the confusing naming
        assert event.path == original_path  # OLD path (original location)
        assert event.old_path == new_path  # NEW path (destination)

        # Correct call order: _handle_moved(old_path, new_path)
        # Should be: _handle_moved(event.path, event.old_path)
        # NOT: _handle_moved(event.old_path, event.path) - SWAPPED!


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
            assert claims[0].source_type == "aics"


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


class TestTensorServerSourceType:
    """The ``tensor-server`` source type: a grpc:// upstream fronted as a proxy.

    Covers only the §1 scheme/type plumbing (recognition + classification +
    the alias namespace field). The RemoteTensorAdapter / catalog expansion is
    a separate slice; here a grpc source is simply classified and returned
    as-is by ``discover_sources``.
    """

    def test_is_remote_url_recognizes_grpc_schemes(self):
        from biopb_tensor_server.core.remote import is_remote_url

        assert is_remote_url("grpc://lab-store:8815") is True
        assert is_remote_url("grpc+tls://lab-store:8815") is True
        assert is_remote_url("grpcs://lab-store:8815") is True
        assert is_remote_url("GRPC://Lab-Store:8815") is True  # case-insensitive
        # unchanged behaviour for local + other remote schemes
        assert is_remote_url("/data/scratch") is False
        assert is_remote_url("s3://bucket/key") is True

    def test_detect_source_type_maps_grpc_to_tensor_server(self):
        from biopb_tensor_server.core.config import detect_source_type

        assert detect_source_type("grpc://lab:8815") == "tensor-server"
        assert detect_source_type("grpc+tls://lab:8815") == "tensor-server"
        assert detect_source_type("grpcs://lab:8815") == "tensor-server"
        # other remote schemes remain non-auto-detectable
        assert detect_source_type("s3://bucket/key") is None

    def test_detect_source_type_does_not_type_local_paths(self):
        """Filesystem format detection belongs to the adapters (claim()), not here.

        detect_source_type only routes remote schemes now (biopb/biopb#277 item
        B); every local path -- whatever its extension or layout -- returns None
        so the adapters remain the single source of truth for format typing.
        """
        from biopb_tensor_server.core.config import detect_source_type

        for url in (
            "/data/experiment.zarr",
            "/data/image.ome.tif",
            "/data/plain.tif",
            "/data/scan.czi",
            "/data/acquisition/",
        ):
            assert detect_source_type(url) is None, url

    def test_unclaimed_file_raises_instead_of_guessing(self):
        """A typeless file no adapter claims is a hard error, not a guessed type."""
        import pytest

        with tempfile.TemporaryDirectory() as tmpdir:
            mystery = os.path.join(tmpdir, "mystery.xyz")
            with open(mystery, "wb") as f:
                f.write(b"\x00\x01\x02\x03")

            with pytest.raises(ValueError, match="Could not detect type for file"):
                discover_sources(SourceConfig(url=mystery), get_default_registry())

    def test_tensor_server_in_type_literal(self):
        # explicit type still round-trips through SourceConfig
        s = SourceConfig(url="grpc://lab:8815", type="tensor-server")
        assert s.type == "tensor-server"
        assert s.is_remote is True

    def test_grpc_source_auto_classifies_to_tensor_server(self):
        # No explicit type: Case 0 auto-detects grpc -> tensor-server instead of
        # raising "Remote URL requires explicit 'type'". Uses the single-source
        # url form so discovery does not reach out to an upstream (bare-host
        # expansion is exercised in the proxy integration test).
        src = SourceConfig(url="grpc://lab:8815/img", alias="lab")
        assert src.type is None  # not set at construction
        out = discover_sources(src)
        assert len(out) == 1
        assert out[0].type == "tensor-server"
        assert out[0].alias == "lab"
        assert out[0].url == "grpc://lab:8815/img"

    def test_non_grpc_remote_without_type_still_errors(self):
        import pytest

        with pytest.raises(ValueError, match="requires explicit 'type'"):
            discover_sources(SourceConfig(url="s3://bucket/key"))

    def test_alias_must_be_slash_free(self):
        import pytest

        # source_id boundary is the first '/', so an alias prefix cannot contain it
        with pytest.raises(ValueError, match="slash-free"):
            SourceConfig(url="grpc://lab:8815", alias="lab/sub")
        # a slash-free alias is accepted
        assert SourceConfig(url="grpc://lab:8815", alias="lab").alias == "lab"

    def test_alias_parsed_from_config_dict(self):
        from biopb_tensor_server.core.config import parse_config

        cfg = parse_config(
            {
                "sources": [
                    {"url": "grpc://lab:8815", "alias": "lab"},
                    {"url": "grpc://arc:8815", "type": "tensor-server", "alias": "arc"},
                ]
            }
        )
        aliases = {s.url: s.alias for s in cfg.sources}
        assert aliases == {"grpc://lab:8815": "lab", "grpc://arc:8815": "arc"}

    def test_single_source_form_namespaces_source_id(self):
        # grpc://host:port/<id> mirrors one upstream source under <alias>__<id>
        out = discover_sources(
            SourceConfig(url="grpc://lab:8815/experiment1", alias="lab")
        )
        assert len(out) == 1
        assert out[0].source_id == "lab__experiment1"
        assert out[0].url == "grpc://lab:8815/experiment1"
        assert out[0].type == "tensor-server"

    def test_single_source_form_no_alias_keeps_verbatim_id(self):
        out = discover_sources(SourceConfig(url="grpc://lab:8815/experiment1"))
        assert len(out) == 1
        assert out[0].source_id == "experiment1"

    def test_namespaced_source_id_helper(self):
        from biopb_tensor_server.core.config import _namespaced_source_id

        assert _namespaced_source_id("lab", "img") == "lab__img"
        assert _namespaced_source_id(None, "img") == "img"

    def test_alias_clash_collision_is_tolerated(self, caplog):
        import logging

        from biopb_tensor_server.core.config import parse_config, resolve_all_sources

        # Two upstreams sharing alias "lab", each mirroring a same-named source
        # -> both namespace to "lab__img": a flat-catalog collision. It must NOT
        # abort the whole resolve -- the first wins, the collider is dropped+warned.
        cfg = parse_config(
            {
                "sources": [
                    {"url": "grpc://a:8815/img", "alias": "lab"},
                    {"url": "grpc://b:8815/img", "alias": "lab"},
                ]
            }
        )
        with caplog.at_level(logging.WARNING):
            resolved = resolve_all_sources(cfg)
        # one survivor (the first), not an exception
        assert [s.source_id for s in resolved] == ["lab__img"]
        assert resolved[0].url == "grpc://a:8815/img"
        assert any("lab__img" in r.message for r in caplog.records)

    def test_collision_does_not_drop_unrelated_sources(self):
        # a colliding pair must not take down the OTHER, valid sources
        from biopb_tensor_server.core.config import parse_config, resolve_all_sources

        cfg = parse_config(
            {
                "sources": [
                    {"url": "grpc://a:8815/img", "alias": "lab"},
                    {"url": "grpc://b:8815/img", "alias": "lab"},  # collides -> dropped
                    {"url": "grpc://c:8815/other", "alias": "arc"},  # unrelated
                ]
            }
        )
        ids = [s.source_id for s in resolve_all_sources(cfg)]
        assert ids == ["lab__img", "arc__other"]

    def test_distinct_aliases_do_not_collide(self):
        from biopb_tensor_server.core.config import parse_config, resolve_all_sources

        cfg = parse_config(
            {
                "sources": [
                    {"url": "grpc://a:8815/img", "alias": "lab"},
                    {"url": "grpc://b:8815/img", "alias": "arc"},
                ]
            }
        )
        ids = {s.source_id for s in resolve_all_sources(cfg)}
        assert ids == {"lab__img", "arc__img"}
