"""Run every selected case and report. A benchmark, not a gate.

`biopb-mcp/docs/skills.md` §10. The engine is `_engine.py` and the data is
`cases/`; this file is only the pytest surface over them, so a new case costs a
`Case` and nothing here.

**No run's outcome fails these tests.** Out of turns, wrong answer, gave up,
even a harness error — each is a row with a reason in `summary.md`. A handful
of samples against a non-deterministic agent cannot support a verdict, and
stopping at the first bad corner would discard the rows that explain it.

Two things *are* asserted, and neither judges a case: that the report reached
disk with a transcript per sample, and that the **catalog matched the switch**.
The second is not a finding — if `--bench-skills=false` stopped withholding the
catalog, a skill's delta against the other session would read as zero for a
reason unrelated to the skill, which is a green table saying the opposite of
the truth.

Costs one conversation per sample. Marked `bench`, deselected by default, never
in CI.
"""

from __future__ import annotations

import pytest

from ._engine import Run, run_case, unavailable, where_for
from .conftest import smoke_failures

pytestmark = pytest.mark.bench

#: Runs are expensive, and both tests read the same one. Keyed by the case's
#: full label so a subject covered two ways does not have its second case
#: handed the first's results.
_RUNS: dict[str, Run] = {}


@pytest.fixture(scope="module")
def run(request, bench_options) -> Run:
    """One case's samples, paid for once, in this invocation's one
    configuration. Parametrized in `conftest.py`, over the cases the options
    asked for."""
    case = request.param
    if reason := unavailable(case, bench_options):
        pytest.skip(reason)
    # `conftest.py` puts the smoke tests first so this is answerable. A broken
    # stack does not produce a weak benchmark, it produces a meaningless one
    # that reads like a weak one -- so refuse rather than spend.
    if broken := smoke_failures():
        pytest.skip(
            f"the session smoke tests failed ({len(broken)}), so the stack is "
            f"what broke and not the case: {broken[0]}"
        )
    if case.label not in _RUNS:
        _RUNS[case.label] = run_case(case, bench_options)
        print("\n\n" + _RUNS[case.label].report + "\n")
    done = _RUNS[case.label]
    if done.failed_to_start:
        pytest.skip(done.results[0].error)
    return done


def test_the_benchmark_ran_and_wrote_its_report(run: Run):
    """The only assertion about the deliverable rather than the result.

    A run that recorded nothing at all cannot be interpreted later, and that is
    a harness failure rather than a finding.
    """
    where = where_for(run.case)
    assert (where / "summary.md").is_file(), "the benchmark produced no summary"
    assert (where / "summary.json").is_file()

    assert run.results, f"{run.case.label} produced no runs at all"
    assert len(run.results) == run.options.samples

    missing = [
        r.name
        for r in run.results
        if not r.error and not (where / r.name / "transcript.md").is_file()
    ]
    assert not missing, (
        f"these ran but left no transcript: {missing}. Their rows cannot be "
        "interpreted without one."
    )


def test_the_catalog_matched_the_switch(run: Run):
    """Whether the report means anything at all.

    Checked on what the catalog *returned*, not on whether `find_skills` was
    called: the tool stays registered either way and it is `load_catalog()` that
    gates, so a `--bench-skills=false` run can call it and get an empty list.

    It asserts one thing and deliberately not a second: that the catalog was
    non-empty exactly when the switch said it should be. It never names an
    entry — this package does not know which skills ship — so what it catches is
    a session whose *label* is false. A run that says the catalog was offered
    and then saw nothing came up misconfigured (or shipped no `.md` files at
    all, which imports and tests clean), and its number is not the
    configuration it claims to be.
    """
    want = run.options.skills
    wrong = [
        f"{r.name}: --bench-skills={str(want).lower()} but the catalog held "
        f"{len(r.catalog)} entries"
        for r in run.results
        if not r.error and bool(r.catalog) != want
    ]
    assert not wrong, (
        "the session's catalog did not match the switch it was run under, so "
        f"this report is not of the configuration it names: {wrong}"
    )
