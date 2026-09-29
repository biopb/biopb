"""A config entry is a request; `sources.resolve` turns it into concrete sources.

`discover_sources` expands one entry -- a typeless file or directory is typed by
whichever adapter claims it, a `grpc://` url routes to the tensor-server proxy --
and `resolve_all_sources` runs that over a whole config. Covered here: the claim
protocol is reached with a ClaimContext (not a bare Path), a file no adapter
claims is a hard error rather than a guessed type, and the tensor-server alias
namespaces `source_id` without letting a clash abort the catalog.

The `core.config` half of the same story -- `detect_source_type`, the `type`
literal, `alias` validation and parsing -- is in `config_source_type_test.py`.
"""

import logging
import os
import tempfile

import numpy as np
import pytest
import tifffile
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.core.config import SourceConfig, parse_config
from biopb_tensor_server.sources.resolve import (
    _namespaced_source_id,
    discover_sources,
    resolve_all_sources,
)


class TestDiscoverSources:
    """Claim-based expansion of a single typeless entry."""

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

    def test_unclaimed_file_raises_instead_of_guessing(self):
        """A typeless file no adapter claims is a hard error, not a guessed type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mystery = os.path.join(tmpdir, "mystery.xyz")
            with open(mystery, "wb") as f:
                f.write(b"\x00\x01\x02\x03")

            with pytest.raises(ValueError, match="Could not detect type for file"):
                discover_sources(SourceConfig(url=mystery), get_default_registry())


class TestTensorServerExpansion:
    """A ``grpc://`` entry expands into proxied sources under its alias.

    Only the single-source url form is exercised here, so nothing dials an
    upstream; bare-host expansion lives in the proxy integration tests.
    """

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
        with pytest.raises(ValueError, match="requires explicit 'type'"):
            discover_sources(SourceConfig(url="s3://bucket/key"))

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
        assert _namespaced_source_id("lab", "img") == "lab__img"
        assert _namespaced_source_id(None, "img") == "img"

    def test_alias_clash_collision_is_tolerated(self, caplog):
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
