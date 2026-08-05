"""The paid run: every task, N samples each, against a real session.

**No outcome here fails a test.** A task that timed out, gave up or got the
answer wrong is a *result* — the report is the deliverable, and a red build
would only teach people to stop running this. What is asserted is that a report
reached disk: a poor result is informative, a missing one is not.
"""

from __future__ import annotations

import pytest

from ._runner import run_case, samples_wanted, unavailable, where_for
from .cases import CASES
from .conftest import smoke_failures

pytestmark = pytest.mark.tasks


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_task(case):
    if failed := smoke_failures():
        pytest.skip(
            f"the stack smoke test failed ({', '.join(failed)}); a run on a "
            "broken stack does not produce a weak result, it produces a "
            "meaningless one that looks like one"
        )
    if why := unavailable(case):
        pytest.skip(why)

    run = run_case(case)
    print("\n" + run.summary)

    assert run.samples, f"{case.case_id} produced no samples at all"
    assert len(run.samples) == samples_wanted()
    root = where_for(case)
    assert (root / "summary.md").is_file(), f"no report reached {root}"
    assert (root / "summary.json").is_file()
    for sample in run.samples:
        assert (root / f"sample-{sample.index}" / "transcript.md").is_file(), (
            f"sample {sample.index} left no transcript; a run nobody can read "
            "afterwards is not evidence of anything"
        )
