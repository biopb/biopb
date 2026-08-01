"""Artifact plumbing for the outcome layer.

Per §2 every case emits a number *and* an artifact: the number gates, the
artifact explains. Runs land in one directory per invocation so a failure can be
paged through afterwards rather than reconstructed from an assertion message.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path

import pytest

from ._outcome import Attempt, Fixture, Outcome, artifact_root, write_report


@pytest.fixture(scope="session")
def outcome_run_dir() -> Path:
    """One directory per pytest session, named for when it started."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    here = artifact_root() / stamp
    here.mkdir(parents=True, exist_ok=True)
    return here


@pytest.fixture
def record(outcome_run_dir: Path) -> Callable[..., Outcome]:
    """`record(verify, save, fixture, attempt)` -> the scored outcome, on disk.

    Deliberately runs before any assertion in the test body, so the artifacts of
    a *failing* run are the ones written. A report only produced on success
    would be missing exactly when it is wanted.
    """

    def _record(
        verify: Callable[[Fixture, Attempt], Outcome],
        save: Callable[[Outcome, Path], None],
        fixture: Fixture,
        attempt: Attempt,
    ) -> Outcome:
        outcome = verify(fixture, attempt)
        where = write_report(outcome, outcome_run_dir)
        save(outcome, where)
        return outcome

    return _record
