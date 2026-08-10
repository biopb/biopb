"""Which cases a run pays for, and smoke first — with teeth.

Two jobs, and they are here rather than in a test module because both are about
the *run* rather than about any one assertion.

**Selection.** `--bench-fixtures` and the rest are resolved once and the paid
tests are parametrized over what survives, so `--bench-fixtures=synthetic`
collects the synthetic cases and nothing else instead of printing a screen of
skips. What
was dropped is said out loud at the end of the run: a shorter table is
otherwise indistinguishable from a shorter catalogue, and this layer's whole
failure mode is a green summary that means something narrower than it looks.
The **hermetic** checks ignore the options entirely — they are free, and a case
excluded from tonight's paid run still has to be a coherent case.

**Smoke first, and the run only after it passes.** This is the least isolable
tier in the suite: a red run's cause space is the skill body, the model, the
tool schemas, the kernel, Qt, dask and the fixture. `test_session_smoke.py` is
the mitigation — when the *stack* is what broke, it fails deterministically and
for free.

That only works if it runs first, and by default it does not: pytest collects
alphabetically, so `test_bench.py` sorted ahead of it and four paid
conversations went out before anything had checked whether the session could
hold a napari layer. The claim was in the README before it was true.

Two hooks make it true, and the second is the one that matters:

**Order** — smoke moves to the front of this directory's block, and only this
directory's, so nothing about the rest of the suite's order changes.

**Dependency** — a failed smoke test *skips* the run rather than merely
preceding it. Ordering alone changes nothing without ``-x``: pytest would run
the smoke failure and then spend the money anyway. A skip here is the honest
outcome, because a run on a broken stack does not produce a weak result, it
produces a meaningless one that looks like a weak result.

A smoke test that *skips* (no display, no key, no fixture tree) is not a
failure and gates nothing — `unavailable()` already reports those, with better
instructions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ..agentbench import _plane
from ._engine import select
from ._options import Options, resolve
from .cases import CASES

HERE = Path(__file__).parent
SMOKE = "test_session_smoke.py"

#: Fixture names that mean "one paid run per selected case". Both are
#: parametrized from the same list, so the smoke pass and the run itself never
#: disagree about which cases tonight is about.
PARAMETRIZED = ("run", "bench_case")

_SMOKE_FAILURES: list[str] = []


def smoke_failures() -> list[str]:
    """Smoke tests that failed in this session. Empty if none ran."""
    return list(_SMOKE_FAILURES)


@pytest.fixture(scope="session")
def bench_options(pytestconfig) -> Options:
    return resolve(pytestconfig)


def pytest_generate_tests(metafunc):
    """Parametrize the paid tests over the cases this run asked for."""
    options = resolve(metafunc.config)
    chosen = select(CASES, options)
    for name in PARAMETRIZED:
        if name in metafunc.fixturenames:
            metafunc.parametrize(
                name,
                chosen,
                indirect=True,
                ids=[case.label for case in chosen],
                scope="module",
            )


def pytest_terminal_summary(terminalreporter, config):
    """Say which cases the options kept out. Never silently.

    A filter is a cap on coverage, and a capped run that does not report its
    cap reads afterwards exactly like a complete one. Printed even when nothing
    was dropped but a filter was set: "it matched every case" and "it dropped
    four" are different facts about the same command line, and only one of them
    is legible from a table.
    """
    options = resolve(config)
    if not options.filtered:
        return
    kept = {c.label for c in select(CASES, options)}
    dropped = [c.label for c in CASES if c.label not in kept]
    terminalreporter.write_sep(
        "-", f"bench: {options.describe()} — {len(dropped)} case(s) not run"
    )
    for label in dropped:
        terminalreporter.write_line(f"  {label}")


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


@pytest.fixture(scope="session", autouse=True)
def _reap_the_plane():
    """Stop the run's data plane with the run, if one was ever started.

    `_plane` registers an `atexit` hook as a backstop, but the plane is a child
    process holding a port and a file-cache lock, and a teardown that runs
    while pytest can still report is worth more than one that runs as the
    interpreter is going down.
    """
    yield
    _plane.stop_plane()


def pytest_runtest_logreport(report):
    """Record a smoke failure so the run can refuse to spend.

    `logreport` rather than a `makereport` wrapper: it needs no hookwrapper
    protocol, which has changed twice across pytest majors, and a report is all
    this needs. Both `setup` and `call` count — a session that will not come up
    fails in setup.
    """
    if report.failed and SMOKE in report.nodeid:
        _SMOKE_FAILURES.append(report.nodeid)
