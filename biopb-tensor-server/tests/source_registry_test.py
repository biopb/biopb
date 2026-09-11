"""Adapter lifecycle in the source registry (biopb/biopb#944, biopb/biopb#979).

Rebuilding a live source -- what a re-drop does when the file underneath it
changed -- swaps a new adapter in over the old one. Doing that through
``register`` would leave the displaced adapter open forever (nothing closes it),
and doing it as unregister-then-register would take the source out of
ListFlights for the length of the rebuild. ``swap`` is the seam that makes the
ordering explicit.

``swap`` hands the displaced adapter back **open** where ``unregister`` closes
on the spot, and #979 read that as two policies wanting one. It is one policy:
ownership follows whether the removal can still be undone. A replace that fails
after the swap restores the displaced adapter and goes on serving from it, so
the registry must not have closed it; an unregistered id has no such rollback.
Draining an in-flight reader is not what separates them -- that is
``SourceAdapter.close``'s own obligation, which is why ``unregister`` can close
immediately.
"""


class _Adapter:
    """A duck-typed double: ``normalize_adapter`` leaves these alone."""

    def __init__(self, name):
        self.name = name
        self.closed = 0

    def close(self):
        self.closed += 1


def _registry():
    from biopb_tensor_server.core.source_registry import SourceRegistry

    return SourceRegistry()


class TestSwap:
    def test_swap_installs_the_new_adapter_and_hands_back_the_old(self):
        registry = _registry()
        old, new = _Adapter("old"), _Adapter("new")
        registry.register("src", old)

        registered, displaced = registry.swap("src", new)

        assert registered is new
        assert displaced is old
        assert registry.get("src") is new

    def test_swap_does_not_close_the_displaced_adapter(self):
        """The caller may have to restore it instead: a replace that fails
        after the swap goes on serving from this adapter, and a registry that
        had closed it would hand back a live-but-broken source."""
        registry = _registry()
        old = _Adapter("old")
        registry.register("src", old)

        registry.swap("src", _Adapter("new"))

        assert old.closed == 0

    def test_swap_of_a_free_id_reports_nothing_displaced(self):
        registry = _registry()
        new = _Adapter("new")

        registered, displaced = registry.swap("src", new)

        assert registered is new
        assert displaced is None

    def test_register_over_a_live_id_leaks_it(self):
        """Why replacement goes through ``swap``: a bare ``register`` overwrites
        the entry and drops the old adapter on the floor, still open."""
        registry = _registry()
        old = _Adapter("old")
        registry.register("src", old)

        registry.register("src", _Adapter("new"))

        assert old.closed == 0

    def test_unregister_still_closes(self):
        registry = _registry()
        old = _Adapter("old")
        registry.register("src", old)

        assert registry.unregister("src") is old
        assert old.closed == 1

    def test_unregister_closes_without_waiting_for_a_drain(self):
        """The counterpart to the swap rule: nothing about an in-flight reader
        delays this close, which is why the two methods can differ at all.

        A caller that owed a drain could not write ``unregister`` as it stands,
        and every persistent-handle adapter's ``close`` deals with an active
        read itself (drain, decline to the reaper, or take ``_io_lock``).
        """
        registry = _registry()
        old = _Adapter("old")
        registry.register("src", old)

        registry.unregister("src")

        assert old.closed == 1
        assert registry.get("src") is None


class TestShutdown:
    """``close_all`` releases the handles the process is holding.

    Required on Windows, where an open file cannot be deleted -- a source left
    open outlives the server as a directory nothing can remove.
    """

    def test_close_all_closes_every_registered_adapter(self):
        registry = _registry()
        adapters = [_Adapter(f"a{i}") for i in range(3)]
        for index, adapter in enumerate(adapters):
            registry.register(f"src{index}", adapter)

        registry.close_all()

        assert [adapter.closed for adapter in adapters] == [1, 1, 1]

    def test_a_balky_adapter_does_not_strand_the_rest(self):
        """Shutdown closes what it can: one adapter raising must not leave the
        handles after it in the map open, which is the whole failure this
        guards -- the balky one is usually the one with a handle held."""
        registry = _registry()
        first, last = _Adapter("first"), _Adapter("last")

        class _Balky(_Adapter):
            def close(self):
                super().close()
                raise RuntimeError("balky")

        balky = _Balky("balky")
        registry.register("a", first)
        registry.register("b", balky)
        registry.register("c", last)

        registry.close_all()  # must not raise

        assert (first.closed, balky.closed, last.closed) == (1, 1, 1)


class TestClosingTwiceIsSafe:
    """``_register_source_claim``'s failure path can reach the same adapter
    twice -- the rollback closes it, then the final "is this the one serving?"
    check closes what it built. Both ends of that have to tolerate it.
    """

    def test_close_adapter_calls_through_every_time(self):
        from biopb_tensor_server.core.source_registry import close_adapter

        adapter = _Adapter("a")

        close_adapter(adapter)
        close_adapter(adapter)

        assert adapter.closed == 2, "close_adapter must not dedupe for the caller"

    def test_close_adapter_swallows_the_raise(self):
        """Never raises: unregister and shutdown must not fail on a balky
        adapter, and the registry also accepts non-inheriting test doubles."""
        from biopb_tensor_server.core.source_registry import close_adapter

        class _Balky:
            def close(self):
                raise RuntimeError("balky")

        close_adapter(_Balky())

    def test_close_adapter_of_nothing_is_a_no_op(self):
        """``unregister`` of an id that was never registered hands it None."""
        from biopb_tensor_server.core.source_registry import close_adapter

        close_adapter(None)

    def test_unregister_twice_closes_once(self):
        """The second call finds nothing to pop, so the adapter is not closed
        again on its behalf."""
        registry = _registry()
        old = _Adapter("old")
        registry.register("src", old)

        assert registry.unregister("src") is old
        assert registry.unregister("src") is None
        assert old.closed == 1
