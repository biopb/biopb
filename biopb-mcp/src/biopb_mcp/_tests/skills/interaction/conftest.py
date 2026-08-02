"""Smoke first, and the benchmark only after it passes.

§5 is the least isolable tier in the suite: a red run's cause space is the skill
body, the model, the tool schemas, the kernel, Qt, dask and the fixture.
`test_session_smoke.py` is the mitigation — when the *stack* is what broke, it
fails deterministically and for free.

That only works if it runs first, and by default it does not: pytest collects
alphabetically, so `test_benchmark.py` sorted ahead of it and four paid
conversations went out before anything had checked whether the session could
hold a napari layer. The claim was in the README before it was true.

Two hooks make it true, and the second is the one that matters:

**Order** — smoke moves to the front of this directory's block, and only this
directory's, so nothing about the rest of the suite's order changes.

**Dependency** — a failed smoke test *skips* the benchmark rather than merely
preceding it. Ordering alone changes nothing without ``-x``: pytest would run
the smoke failure and then spend the money anyway. A skip here is the honest
outcome, because a benchmark run on a broken stack does not produce a weak
result, it produces a meaningless one that looks like a weak result.

A smoke test that *skips* (no display, no key) is not a failure and gates
nothing — `unavailable()` already reports those, with better instructions.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent
SMOKE = "test_session_smoke.py"

_SMOKE_FAILURES: list[str] = []


def smoke_failures() -> list[str]:
    """Smoke tests that failed in this session. Empty if none ran."""
    return list(_SMOKE_FAILURES)


def pytest_collection_modifyitems(items):
    """Move this directory's smoke tests to the front of its own block.

    Positions are reused rather than the whole list re-sorted: the hook is
    handed every collected item in the run, and a directory-level conftest
    reordering all of them would silently rearrange the rest of the suite.
    """
    here = [(n, item) for n, item in enumerate(items) if item.path.parent == HERE]
    if not here:
        return
    slots = [n for n, _ in here]
    ordered = sorted((item for _, item in here), key=lambda i: i.path.name != SMOKE)
    for slot, item in zip(slots, ordered, strict=True):
        items[slot] = item


def pytest_runtest_logreport(report):
    """Record a smoke failure so the benchmark can refuse to spend.

    `logreport` rather than a `makereport` wrapper: it needs no hookwrapper
    protocol, which has changed twice across pytest majors, and a report is all
    this needs. Both `setup` and `call` count — a session that will not come up
    fails in setup.
    """
    if report.failed and SMOKE in report.nodeid:
        _SMOKE_FAILURES.append(report.nodeid)
