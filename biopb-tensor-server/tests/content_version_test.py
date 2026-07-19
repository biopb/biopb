"""content_version folded into chunk_id / cache key (biopb/biopb#178).

A source's content_version is wrapped into every chunk_id it mints (a leading
0xFF-sentinel header) so the cache namespaces by it: a re-registered source with
new bytes gets a fresh cache key instead of serving a stale cached chunk. Covers:
- the codec is backward-compatible (unversioned chunk_ids / cache keys are
  byte-identical to pre-#178) and every codec function is wrapper-aware;
- cache_key_for_chunk_id keeps the version (version-sensitive) while the inner
  projection is unchanged and the reduction method stays advisory;
- _get_read_plan wraps all minted chunk_ids with a source's content_version;
- content_version_from_path yields a cheap stat signal and OmeTiffAdapter adopts it.
"""

import os
import tempfile
import time

from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.core.base import _get_read_plan
from biopb_tensor_server.core.chunk import (
    _CV_SENTINEL,
    cache_key_for_chunk_id,
    content_version_from_path,
    content_version_of,
    decode_chunk_id,
    decode_scale_info,
    encode_chunk_id,
    encode_chunk_id_with_scale,
    get_bounds_from_chunk_id,
    is_scaled_chunk,
    rewrite_chunk_id_array_id,
    wrap_content_version,
)

CV = b"1700000000000000000:4096"


def _bounds():
    return ChunkBounds(start=[0, 10], stop=[5, 20])


# ==============================================================================
# Codec: backward compatibility + wrapper-awareness
# ==============================================================================


class TestCodecWrapper:
    def test_legacy_chunk_id_is_unversioned_and_unchanged(self):
        legacy = encode_chunk_id("src/t", _bounds())
        assert content_version_of(legacy) is None
        # A legacy chunk_id never begins with the sentinel (array_id len high byte
        # is 0x00), so the discriminator is unambiguous.
        assert legacy[0] != _CV_SENTINEL

    def test_wrap_roundtrip_regular_and_scaled(self):
        for base in (
            encode_chunk_id("src/t", _bounds()),
            encode_chunk_id_with_scale("src/t", _bounds(), (2, 2), "mean"),
        ):
            wrapped = wrap_content_version(base, CV)
            assert wrapped[0] == _CV_SENTINEL
            assert content_version_of(wrapped) == CV

    def test_decode_and_bounds_through_wrapper(self):
        wrapped = wrap_content_version(encode_chunk_id("src/t", _bounds()), CV)
        aid, bounds = decode_chunk_id(wrapped)
        assert aid == "src/t"
        assert list(bounds.start) == [0, 10] and list(bounds.stop) == [5, 20]
        assert list(get_bounds_from_chunk_id(wrapped).start) == [0, 10]

    def test_scale_detection_and_decode_through_wrapper(self):
        legacy = encode_chunk_id("src/t", _bounds())
        scaled = encode_chunk_id_with_scale("src/t", _bounds(), (2, 2), "mean")
        assert is_scaled_chunk(wrap_content_version(legacy, CV)) is False
        assert is_scaled_chunk(wrap_content_version(scaled, CV)) is True
        assert decode_scale_info(wrap_content_version(scaled, CV)) == ((2, 2), "mean")

    def test_rewrite_array_id_preserves_version(self):
        wrapped = wrap_content_version(encode_chunk_id("src/t", _bounds()), CV)
        rewritten = rewrite_chunk_id_array_id(wrapped, "other/t")
        assert content_version_of(rewritten) == CV
        assert decode_chunk_id(rewritten)[0] == "other/t"
        assert list(get_bounds_from_chunk_id(rewritten).start) == [0, 10]

    def test_rewrite_unversioned_unchanged(self):
        legacy = encode_chunk_id("src/t", _bounds())
        assert rewrite_chunk_id_array_id(legacy, "src/t") == legacy


# ==============================================================================
# Cache key: backward-compat, version-sensitivity, method-advisory
# ==============================================================================


class TestCacheKey:
    def test_unversioned_key_identical_to_legacy(self):
        # An unversioned chunk_id maps to exactly its pre-#178 cache entry.
        legacy = encode_chunk_id("src/t", _bounds())
        assert cache_key_for_chunk_id(legacy) == legacy
        scaled = encode_chunk_id_with_scale("src/t", _bounds(), (2, 2), "mean")
        # legacy scaled key drops the advisory method suffix
        assert cache_key_for_chunk_id(scaled) == cache_key_for_chunk_id(
            encode_chunk_id_with_scale("src/t", _bounds(), (2, 2), "max")
        )

    def test_versioned_key_differs_from_unversioned(self):
        legacy = encode_chunk_id("src/t", _bounds())
        assert cache_key_for_chunk_id(wrap_content_version(legacy, CV)) != (
            cache_key_for_chunk_id(legacy)
        )

    def test_key_is_version_sensitive(self):
        legacy = encode_chunk_id("src/t", _bounds())
        assert cache_key_for_chunk_id(wrap_content_version(legacy, b"v1")) != (
            cache_key_for_chunk_id(wrap_content_version(legacy, b"v2"))
        )

    def test_method_stays_advisory_within_a_version(self):
        a = wrap_content_version(
            encode_chunk_id_with_scale("src/t", _bounds(), (2, 2), "mean"), CV
        )
        b = wrap_content_version(
            encode_chunk_id_with_scale("src/t", _bounds(), (2, 2), "max"), CV
        )
        assert cache_key_for_chunk_id(a) == cache_key_for_chunk_id(b)


# ==============================================================================
# read plan: minting wraps every chunk_id with the source's content_version
# ==============================================================================


def _base_desc():
    return TensorDescriptor(
        array_id="src/t",
        dim_labels=["y", "x"],
        shape=[10, 10],
        chunk_shape=[5, 5],
        dtype="uint8",
    )


class TestReadPlanWiring:
    def test_none_version_mints_legacy_chunk_ids(self):
        plan = _get_read_plan(_base_desc(), TensorDescriptor(), (5, 5))
        assert plan.chunk_endpoints
        assert all(
            content_version_of(ep.chunk_id) is None for ep in plan.chunk_endpoints
        )

    def test_version_wraps_every_chunk_id(self):
        plan = _get_read_plan(
            _base_desc(), TensorDescriptor(), (5, 5), content_version=CV
        )
        assert plan.chunk_endpoints
        assert all(content_version_of(ep.chunk_id) == CV for ep in plan.chunk_endpoints)
        # Same grid, unchanged bounds -- only the cache namespace moves.
        for ep in plan.chunk_endpoints:
            assert decode_chunk_id(ep.chunk_id)[0] == "src/t"

    def test_version_bump_changes_all_cache_keys(self):
        p1 = _get_read_plan(
            _base_desc(), TensorDescriptor(), (5, 5), content_version=b"v1"
        )
        p2 = _get_read_plan(
            _base_desc(), TensorDescriptor(), (5, 5), content_version=b"v2"
        )
        keys1 = {cache_key_for_chunk_id(ep.chunk_id) for ep in p1.chunk_endpoints}
        keys2 = {cache_key_for_chunk_id(ep.chunk_id) for ep in p2.chunk_endpoints}
        assert keys1.isdisjoint(keys2)


# ==============================================================================
# content_version_from_path + adapter adoption
# ==============================================================================


class TestContentVersionFromPath:
    def test_stat_signature_and_none_fallback(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello")
            path = f.name
        try:
            cv = content_version_from_path(path)
            assert cv is not None and b":" in cv
            # size is part of the signal
            assert cv.endswith(b":5")
            assert content_version_from_path(path + "-nope") is None
            # a remote-looking url can't be stat'd -> unversioned
            assert content_version_from_path("s3://bucket/obj") is None
        finally:
            os.unlink(path)

    def test_signal_changes_on_rewrite(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"a")
            path = f.name
        try:
            before = content_version_from_path(path)
            with open(path, "wb") as f:
                f.write(b"much longer content")  # size changes
            after = content_version_from_path(path)
            assert before != after
        finally:
            os.unlink(path)

    def test_ome_tiff_adapter_adopts_stat_version(self):
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        with tempfile.NamedTemporaryFile(suffix=".ome.tif", delete=False) as f:
            f.write(b"not-a-real-tiff-but-init-only-stats-the-path")
            path = f.name
        try:
            # __init__ only stats the path (no open), so a stub file is enough.
            adapter = OmeTiffAdapter(path, "srcid")
            assert adapter.content_version == content_version_from_path(path)
            assert adapter.content_version is not None
        finally:
            os.unlink(path)

    def test_unresolved_url_leaves_adapter_unversioned(self):
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        assert OmeTiffAdapter("", "srcid").content_version is None

    def test_directory_signal_and_member_add_changes_it(self):
        # Directory-based adapters version off the dir's own mtime, which flips on
        # member add/remove/rename -- the O(1) signal for multi-file sources.
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.bin"), "wb") as f:
                f.write(b"x")
            before = content_version_from_path(d)
            assert before is not None and b":" in before
            # A new member flips the directory's mtime. Back-to-back adds can
            # coalesce inside one FS mtime tick (documented sub-resolution blind
            # spot -- observed ~sub-20ms on Windows), so retry until the tick
            # advances; real re-registrations are seconds apart and never coalesce.
            after = before
            for i in range(50):
                with open(os.path.join(d, f"m{i}.bin"), "wb") as f:
                    f.write(b"y")
                after = content_version_from_path(d)
                if after != before:
                    break
                time.sleep(0.01)
            assert after is not None and after != before
