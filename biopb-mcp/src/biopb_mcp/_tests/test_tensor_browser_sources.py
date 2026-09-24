"""The Tensor Browser's source list and its re-list watcher (issue #44).

Qt-free: a fake connection whose client serves a catalog and a health sequence.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from biopb_mcp.tensor_browser._sources import SourceList


def _rows(ids):
    return [
        {
            "source_id": sid,
            "source_url": f"/data/{sid}",
            "source_type": "zarr",
            "is_resolved": True,
            "tensors": [],
        }
        for sid in ids
    ]


def _listing(ids, health=()):
    """A SourceList already holding *ids*; the server then answers *health*."""
    client = MagicMock()
    client.catalog = list(ids)
    client.query_sources.side_effect = lambda sql, **kw: _rows(client.catalog)
    listing = SourceList(SimpleNamespace(client=client))
    listing.refresh()
    client.query_sources.reset_mock()
    client.health_check.side_effect = list(health)
    return listing, client


class _FakeStop:
    """The watcher's stop Event: not stopped for *allow* waits, then stopped."""

    def __init__(self, allow):
        self.allow = allow

    def wait(self, _timeout):
        self.allow -= 1
        return self.allow < 0

    def clear(self):
        pass


def _watch(listing, polls):
    listing._watch_stop = _FakeStop(polls)
    listing._watch_loop(0.0, 0.0)


class TestWatch:
    def test_relists_when_the_count_changes(self):
        listing, client = _listing(
            ["a", "b"], [{"source_count": 2}, {"source_count": 3}]
        )
        client.catalog = ["a", "b", "c"]
        changed = []
        listing.on_changed = changed.append

        _watch(listing, 2)

        assert set(listing.sources) == {"a", "b", "c"}
        assert changed == [listing.sources]

    def test_a_list_taken_mid_index_is_caught_on_the_first_poll(self):
        listing, client = _listing(["a"], [{"source_count": 18}])
        client.catalog = [str(i) for i in range(18)]

        _watch(listing, 1)

        assert len(listing.sources) == 18

    def test_a_stable_count_does_not_relist(self):
        listing, client = _listing(
            ["a", "b"], [{"source_count": 2}, {"source_count": 2}]
        )
        _watch(listing, 2)
        client.query_sources.assert_not_called()

    def test_a_health_error_is_tolerated(self):
        listing, client = _listing(["a"], [RuntimeError("blip")])
        _watch(listing, 1)
        client.query_sources.assert_not_called()

    def test_disconnected_polls_nothing(self):
        listing, client = _listing(["a"])
        listing._conn.client = None
        _watch(listing, 1)
        client.health_check.assert_not_called()

    def test_a_failed_relist_keeps_the_watcher_alive(self):
        listing, client = _listing(["a"], [{"source_count": 2}, {"source_count": 3}])
        client.query_sources.side_effect = RuntimeError("list boom")
        _watch(listing, 2)
        assert client.health_check.call_count == 2

    def test_the_health_it_saw_is_kept_for_the_paint_thread(self):
        listing, _ = _listing(
            ["a"], [{"source_count": 1, "full_scan_in_progress": True}]
        )
        _watch(listing, 1)
        assert listing.scan_in_progress()
        assert listing.scan_source_count() == 1


class TestVerbs:
    def test_add_and_remove_relist(self):
        listing, client = _listing(["a"])
        client.catalog = ["a", "b"]
        listing.add("/data/b")
        assert set(listing.sources) == {"a", "b"}

        client.catalog = ["a"]
        listing.remove("dnd://b")
        assert set(listing.sources) == {"a"}

    def test_resolve_returns_the_row_it_committed(self):
        listing, client = _listing(["a"])
        client.resolve.return_value = _rows(["a"])[0]
        assert listing.resolve("a").source_id == "a"

    def test_warm_does_not_relist(self):
        listing, client = _listing(["a"])
        listing.warm("a")
        client.query_sources.assert_not_called()

    def test_use_server_query_follows_the_size(self):
        listing, _ = _listing([str(i) for i in range(1001)])
        assert listing.use_server_query
