"""Run every §6 case and report. A benchmark, not a gate.

`docs/skill-testing.md` §6. The engine is `_benchmark.py` and the data is
`cases/`; this file is only the pytest surface over them, so a new skill costs a
`Case` and nothing here.

**No run's outcome fails these tests.** Out of turns, wrong answer, gave up,
even a harness error — each is a row with a reason in `summary.md`. One sample
per corner against a non-deterministic agent cannot support a verdict, and
stopping at the first bad corner would discard the three rows that explain it.

Two things *are* asserted, and neither judges a skill: that the report reached
disk with a transcript per arm, and that the **ablation took effect**. The
second is not a finding — if `skills_enabled: false` stopped withholding the
catalog, the delta would read as zero for a reason unrelated to the skill, which
is a green table saying the opposite of the truth.

Costs four conversations per case. Marked `interaction`, deselected by default,
never in CI.
"""

from __future__ import annotations

import pytest

from ._benchmark import Run, run_case, unavailable, where_for
from .cases import CASES

pytestmark = pytest.mark.interaction

#: Runs are expensive, and both tests read the same one. Keyed by skill so a
#: module-scoped parametrized fixture can hand back the run it already paid for.
_RUNS: dict[str, Run] = {}


@pytest.fixture(scope="module", params=CASES, ids=lambda case: case.skill)
def run(request) -> Run:
    case = request.param
    if reason := unavailable(case):
        pytest.skip(reason)
    if case.skill not in _RUNS:
        _RUNS[case.skill] = run_case(case)
        print("\n\n" + _RUNS[case.skill].summary() + "\n")
    done = _RUNS[case.skill]
    if done.failed_to_start:
        pytest.skip(done.results[0].error)
    return done


def test_the_benchmark_ran_and_wrote_its_report(run: Run):
    """The only assertion about the deliverable rather than the result.

    An arm that recorded nothing at all cannot be interpreted later, and that is
    a harness failure rather than a finding.
    """
    where = where_for(run.case)
    assert (where / "summary.md").is_file(), "the benchmark produced no summary"
    assert (where / "summary.json").is_file()

    missing = [
        r.arm.name
        for r in run.results
        if not r.error and not (where / r.arm.name / "transcript.md").is_file()
    ]
    assert not missing, (
        f"these arms ran but left no transcript: {missing}. Their rows cannot "
        "be interpreted without one."
    )


def test_the_ablation_took_effect(run: Run):
    """Whether the table means anything at all.

    Checked on what the catalog *returned*, not on whether `find_skills` was
    called: the tool stays registered either way and it is `load_catalog()` that
    gates, so an ablated run can call it and get an empty list back.
    """
    wrong = [
        f"{r.arm.name}: skill_offered={r.arm.skills} but catalog had "
        f"{r.catalog_hits} entries"
        for r in run.results
        if not r.error and bool(r.catalog_hits) != r.arm.skills
    ]
    assert not wrong, (
        "the ablation did not take effect, so the skill delta in the report is "
        f"not real: {wrong}"
    )
