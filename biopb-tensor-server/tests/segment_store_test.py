"""The segment store on its own: the chunk cache's format, without the cache.

``cache/segment_store`` is what an uploaded ``cache://`` member keeps its bytes
in (``docs/upload-model.md`` step 7). The member's own behaviour is
``cache_member_test``; here is the store: rotation, what a boot restores, and
the byte range a locate publishes.
"""

import numpy as np
import pytest
from biopb_tensor_server.cache.segment_store import SegmentStore
from biopb_tensor_server.core.chunk_batch import pack_chunk_batch, unpack_chunk_array


def _batch(fill, shape=(4, 4)):
    return pack_chunk_batch(np.full(shape, fill, dtype=np.uint16))


def _filled(tmp_path, count=3, **kwargs):
    store = SegmentStore(tmp_path, **kwargs)
    for i in range(count):
        store.append(f"c{i}".encode(), _batch(i))
    return store


class TestWhatItStores:
    def test_a_batch_reads_back_as_it_was_written(self, tmp_path):
        store = _filled(tmp_path)
        try:
            np.testing.assert_array_equal(
                unpack_chunk_array(store.read(b"c1")), np.full((4, 4), 1, np.uint16)
            )
        finally:
            store.close()

    def test_a_key_it_does_not_hold_is_none(self, tmp_path):
        """Not an error: the caller decides whether a miss is a gap or a loss."""
        store = _filled(tmp_path)
        try:
            assert store.read(b"absent") is None
            assert store.locate(b"absent") is None
        finally:
            store.close()

    def test_a_key_written_twice_is_refused(self, tmp_path):
        store = _filled(tmp_path, count=1)
        try:
            with pytest.raises(KeyError):
                store.append(b"c0", _batch(9))
        finally:
            store.close()

    def test_a_closed_store_takes_no_more(self, tmp_path):
        """What a write that passed the upload's gate a moment before a discard
        meets -- and without it, that write would mint the directory again
        behind the discard's back."""
        store = _filled(tmp_path, count=1)
        store.close()

        with pytest.raises(RuntimeError, match="closed"):
            store.append(b"late", _batch(9))

    def test_it_rotates_at_the_segment_bound(self, tmp_path):
        """Every entry stays readable across the rotation, and each segment
        gets its own sidecar -- a boot restores them one file at a time."""
        store = _filled(tmp_path, count=4, max_segment_bytes=64)
        try:
            store.seal()
            assert len(list(tmp_path.glob("seg_*.arrow"))) > 1
            assert len(list(tmp_path.glob("*.idx"))) == len(
                list(tmp_path.glob("seg_*.arrow"))
            )
            for i in range(4):
                np.testing.assert_array_equal(
                    unpack_chunk_array(store.read(f"c{i}".encode())),
                    np.full((4, 4), i, np.uint16),
                )
        finally:
            store.close()


class TestWhatABootRestores:
    def test_the_index_comes_back_off_the_sidecars(self, tmp_path):
        store = _filled(tmp_path)
        store.close()

        reopened = SegmentStore.open(tmp_path)
        try:
            assert sorted(reopened.stored_keys()) == [b"c0", b"c1", b"c2"]
            np.testing.assert_array_equal(
                unpack_chunk_array(reopened.read(b"c2")), np.full((4, 4), 2, np.uint16)
            )
        finally:
            reopened.close()

    def test_a_lost_sidecar_costs_a_body_walk_and_nothing_else(self, tmp_path):
        """The sidecar is the fast path, not the record: the segment body is
        authoritative, which is what makes a half-written sidecar harmless."""
        store = _filled(tmp_path)
        store.close()
        for idx in tmp_path.glob("*.idx"):
            idx.unlink()

        reopened = SegmentStore.open(tmp_path)
        try:
            assert sorted(reopened.stored_keys()) == [b"c0", b"c1", b"c2"]
            np.testing.assert_array_equal(
                unpack_chunk_array(reopened.read(b"c1")), np.full((4, 4), 1, np.uint16)
            )
        finally:
            reopened.close()

    def test_appending_after_a_reopen_does_not_reuse_a_segment_file(self, tmp_path):
        """A client may hold a mapping of any segment on disk, so a new one is
        always a new file -- nothing here truncates or rewrites in place."""
        store = _filled(tmp_path, count=1)
        store.close()
        first = {p.name for p in tmp_path.glob("seg_*.arrow")}

        reopened = SegmentStore.open(tmp_path)
        try:
            reopened.append(b"later", _batch(7))
            reopened.seal()
            assert {p.name for p in tmp_path.glob("seg_*.arrow")} > first
            np.testing.assert_array_equal(
                unpack_chunk_array(reopened.read(b"later")),
                np.full((4, 4), 7, np.uint16),
            )
        finally:
            reopened.close()


class TestTheByteRangeItPublishes:
    def test_it_names_the_entry_it_is_indexed_under(self, tmp_path):
        """A locate hands a range to another process, which reads it with the
        server out of the loop, so the range is checked before it is published."""
        store = _filled(tmp_path)
        try:
            store.seal()
            located = store.locate(b"c1")
            assert located is not None
            assert located.byte_offset > 0 and located.byte_length > 0
            assert located.segment_path.endswith(".arrow")
        finally:
            store.close()

    def test_a_corrupt_segment_publishes_nothing(self, tmp_path):
        """Its sidecar no longer matches it and its body cannot be walked, so
        it is not indexed at all -- and a client takes do_get, which is the
        designed floor of this path rather than a failure."""
        store = _filled(tmp_path)
        store.close()
        segment = next(tmp_path.glob("seg_*.arrow"))
        segment.write_bytes(b"\x00" * 64)

        reopened = SegmentStore.open(tmp_path)
        try:
            assert reopened.locate(b"c1") is None
        finally:
            reopened.close()
