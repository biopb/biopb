"""Adapter replacement in the source registry (biopb/biopb#944).

Rebuilding a live source -- what a re-drop does when the file underneath it
changed -- swaps a new adapter in over the old one. Doing that through
``register`` would leave the displaced adapter open forever (nothing closes it),
and doing it as unregister-then-register would close it out from under readers
that already resolved it. ``swap`` is the seam that makes the ordering explicit.
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
        """A reader that resolved the old adapter through ``get`` is still
        decoding from it; closing is the caller's, after it has drained."""
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
