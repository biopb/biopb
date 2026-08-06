"""Smoke first, and the paid run only after it passes.

Same reasoning as the interaction layer's conftest, and the same two hooks. A
red run here has a cause space of the task, the model, the tool schemas, the
kernel, Qt, dask and the fixture; `test_session_smoke.py` narrows it for free by
failing separately when the *stack* is what broke.

**Order** puts this directory's smoke test at the front of its own block, and
only its own — the hook is handed every collected item in the run, so a
directory-level conftest that re-sorted all of them would silently rearrange
the rest of the suite.

**Dependency** is the half that matters: a failed smoke test *skips* the run
rather than merely preceding it. Ordering alone changes nothing without ``-x``,
and a benchmark on a broken stack does not produce a weak result — it produces
a meaningless one that looks like a weak result.

A smoke test that *skips* (no display, no tree, no key) is not a failure and
gates nothing; `unavailable()` already reports those, with better instructions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ..agentbench import _plane

HERE = Path(__file__).parent
SMOKE = "test_session_smoke.py"

_SMOKE_FAILURES: list[str] = []


def smoke_failures() -> list[str]:
    """Smoke tests that failed in this session. Empty if none ran."""
    return list(_SMOKE_FAILURES)


def pytest_collection_modifyitems(items):
    here = [(n, item) for n, item in enumerate(items) if item.path.parent == HERE]
    if not here:
        return
    slots = [n for n, _ in here]
    ordered = sorted((item for _, item in here), key=lambda i: i.path.name != SMOKE)
    for slot, item in zip(slots, ordered, strict=True):
        items[slot] = item


@pytest.fixture(scope="session", autouse=True)
def _reap_the_plane():
    """Stop the run's data plane with the run, if one was ever started.

    `_plane` registers an `atexit` backstop, but the plane is a child process
    holding a port and a file-cache lock, and a teardown that runs while pytest
    can still report is worth more than one that runs as the interpreter dies.
    """
    yield
    _plane.stop_plane()


def pytest_runtest_logreport(report):
    if report.failed and SMOKE in report.nodeid:
        _SMOKE_FAILURES.append(report.nodeid)
