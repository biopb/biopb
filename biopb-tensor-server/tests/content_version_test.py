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
from biopb_tensor_server.core.adapter_base import _get_read_plan
from biopb_tensor_server.core.chunk import (
    _CV_SENTINEL,
    cache_key_for_chunk_id,
    content_version_from_path,
    content_version_of,
    decode_chunk_id,
    decode_scale_info,
    encode_chunk_id,
    encode_chunk_id_with_scale,
    encode_proxy_envelope,
    get_bounds_from_chunk_id,
    is_proxy_envelope,
    is_scaled_chunk,
    peel_proxy_envelope,
    rewrite_chunk_id_array_id,
    routing_array_id,
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
            encode_chunk_id_with_scale("src/t", _bounds(), (2, 2)),
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
        scaled = encode_chunk_id_with_scale("src/t", _bounds(), (2, 2))
        assert is_scaled_chunk(wrap_content_version(legacy, CV)) is False
        assert is_scaled_chunk(wrap_content_version(scaled, CV)) is True
        # decode_scale_info returns the scale_hint only (method left the chunk_id).
        assert decode_scale_info(wrap_content_version(scaled, CV)) == (2, 2)

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
# Cache key: backward-compat, version-sensitivity, identity-only chunk_ids
# ==============================================================================


class TestCacheKey:
    def test_unversioned_key_identical_to_legacy(self):
        # An unversioned chunk_id maps to exactly its pre-#178 cache entry.
        legacy = encode_chunk_id("src/t", _bounds())
        assert cache_key_for_chunk_id(legacy) == legacy
        # A scaled chunk_id is now pure identity, so its key is itself.
        scaled = encode_chunk_id_with_scale("src/t", _bounds(), (2, 2))
        assert cache_key_for_chunk_id(scaled) == scaled

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

    def test_legacy_method_suffix_maps_to_identity_key(self):
        # An OLD-format scaled chunk_id carried a trailing (uint16 len + method)
        # suffix; cache_key_for_chunk_id strips it, so it maps to exactly the same
        # key as the new identity chunk_id -- no cache wipe on the #178 wire change.
        identity = encode_chunk_id_with_scale("src/t", _bounds(), (2, 2))
        legacy_with_method = identity + b"\x00\x03max"  # uint16(3) + b"max"
        assert cache_key_for_chunk_id(legacy_with_method) == identity
        assert cache_key_for_chunk_id(
            wrap_content_version(legacy_with_method, CV)
        ) == cache_key_for_chunk_id(wrap_content_version(identity, CV))


# ==============================================================================
# Proxy envelope: opaque-inner wrapper for the remote-tensor proxy (#178 W1)
# ==============================================================================


class TestProxyEnvelope:
    def test_roundtrip_with_and_without_version(self):
        inner = encode_chunk_id("upstream/img", _bounds())
        for cv in (CV, None):
            env = encode_proxy_envelope(inner, "local/img", cv)
            assert is_proxy_envelope(env)
            route, got_cv, got_inner = peel_proxy_envelope(env)
            assert route == "local/img"
            assert got_cv == cv  # empty cv decodes back to None
            assert got_inner == inner  # inner forwarded verbatim

    def test_inner_is_opaque_arbitrary_bytes(self):
        # The inner can itself be a version-wrapped / scaled upstream chunk_id; the
        # envelope carries it byte-for-byte and never parses it.
        inner = wrap_content_version(
            encode_chunk_id_with_scale("upstream/img", _bounds(), (2, 2)), b"iat:99"
        )
        env = encode_proxy_envelope(inner, "local/img", CV)
        assert peel_proxy_envelope(env)[2] == inner

    def test_discriminators_are_mutually_exclusive(self):
        legacy = encode_chunk_id("src/t", _bounds())
        versioned = wrap_content_version(legacy, CV)
        envelope = encode_proxy_envelope(legacy, "src/t", CV)
        assert not is_proxy_envelope(legacy)  # 0x00 high byte
        assert not is_proxy_envelope(versioned)  # 0xFF sentinel
        assert is_proxy_envelope(envelope)  # 0xFE sentinel
        assert legacy[0] != 0xFE and versioned[0] != 0xFE

    def test_cache_key_is_envelope_verbatim_and_injective(self):
        inner = encode_chunk_id("upstream/img", _bounds())
        env = encode_proxy_envelope(inner, "local/img", CV)
        assert cache_key_for_chunk_id(env) == env  # verbatim
        # cv, route, and inner each move the key.
        assert cache_key_for_chunk_id(
            encode_proxy_envelope(inner, "local/img", b"v2")
        ) != cache_key_for_chunk_id(env)
        assert cache_key_for_chunk_id(
            encode_proxy_envelope(inner, "other/img", CV)
        ) != cache_key_for_chunk_id(env)
        assert cache_key_for_chunk_id(
            encode_proxy_envelope(
                encode_chunk_id("upstream/j", _bounds()), "local/img", CV
            )
        ) != cache_key_for_chunk_id(env)
        # Same triple -> same key (deterministic mint dedups).
        assert cache_key_for_chunk_id(
            encode_proxy_envelope(inner, "local/img", CV)
        ) == cache_key_for_chunk_id(env)

    def test_routing_array_id_uses_route_without_decoding_inner(self):
        # A deliberately un-decodable inner proves routing never parses it.
        env = encode_proxy_envelope(b"\xde\xad\xbe\xef", "local/img", CV)
        assert routing_array_id(env) == "local/img"
        # Non-envelope chunk_ids still route by their decoded array_id.
        legacy = encode_chunk_id("src/t", _bounds())
        assert routing_array_id(legacy) == "src/t"
        assert routing_array_id(wrap_content_version(legacy, CV)) == "src/t"


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
