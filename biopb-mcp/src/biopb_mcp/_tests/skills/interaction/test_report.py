"""The engine's own tests: classification, flags, and the report.

Hermetic — no session, no model, no fixture build — and they run with the
ordinary suite. The report **is** this layer's deliverable, so the code that
writes it should not be exercised for the first time twenty minutes into a paid
run. Everything here is fed hand-built `Outcome`s, which is also the only way to
reach the corners that a real run reaches rarely and expensively: an arm that
died, one severed by a cap, one that left an unscorable array.

The metric columns are the part most worth pinning. They used to be three
hardcoded names; now they are read off whatever the verifier reported, because a
curated fixture and a synthetic one for the same skill support different
measurements and the table has to follow.
"""

from __future__ import annotations

import json

import pytest

from ..outcomes._outcome import Attempt, Fixture, Metric, Outcome
from . import _benchmark
from ._benchmark import (
    ARMS,
    FLAG_CATALOG_MISMATCH,
    FLAG_CUT_OFF,
    FLAG_NEVER_ASKED,
    FLAG_NEVER_REGISTERED,
    FLAG_OVER_BUDGET,
    GAVE_UP,
    HARNESS_ERROR,
    NO_RESULT,
    NO_SESSION,
    OK,
    OUT_OF_TURNS,
    UNSCORABLE,
    WRONG_ANSWER,
    Result,
    Run,
)
from ._conversation import FINISHED, SILENT, TURN_CAP
from .cases import CASES

FIXTURE = Fixture(
    skill_id="demo",
    case_id="demo-case",
    kind="synthetic",
    provenance="this test",
    data={},
    truth={},
    tolerance={},
    about="a stand-in, so nothing here depends on a real skill",
)


class FakeTrace:
    def __init__(self, stopped=FINISHED, turns=5, asked=(), tools=()):
        self.stopped = stopped
        self.turns_used = turns
        self.questions = list(asked)
        self.blocking_questions = [q for q in asked if "?" in q]
        self.tool_names = list(tools)


def outcome(*metrics: Metric, arrays=None) -> Outcome:
    return Outcome(
        fixture=FIXTURE,
        attempt=Attempt(subject="x", arrays=arrays or {}),
        metrics=metrics,
    )


def scored(value: float, limit: float = 1.0) -> Outcome:
    return outcome(Metric("err_px", value, limit), arrays={"corrected": object()})


def result(arm=ARMS[0], **kwargs) -> Result:
    kwargs.setdefault("trace", FakeTrace())
    kwargs.setdefault("catalog_hits", 1 if arm.skills else 0)
    return Result(arm=arm, **kwargs)


# --- classification --------------------------------------------------------


def test_a_run_within_tolerance_is_ok():
    assert result(outcome=scored(0.2)).classify() == (OK, "within every tolerance")


def test_a_run_outside_tolerance_names_the_metric_and_the_limit():
    verdict, reason = result(outcome=scored(3.0)).classify()
    assert verdict == WRONG_ANSWER
    assert "err_px 3 > 1" in reason


def test_a_cap_beats_a_bad_number():
    """A run severed mid-workflow may still have left a plausible array, and
    calling that a wrong answer blames the skill for a budget."""
    verdict, reason = result(
        trace=FakeTrace(stopped=TURN_CAP), outcome=outcome(Metric("err_px", None, 1.0))
    ).classify()
    assert verdict == OUT_OF_TURNS
    assert "turn cap" in reason


def test_an_agent_that_stopped_talking_is_not_a_wrong_answer():
    verdict, _ = result(
        trace=FakeTrace(stopped=SILENT), outcome=outcome(Metric("err_px", None, 1.0))
    ).classify()
    assert verdict == GAVE_UP


def test_leaving_nothing_and_leaving_something_unscorable_are_different():
    """Both score zero metrics and they point at different causes: one agent
    never got there, the other got there and produced the wrong shape."""
    nothing = result(outcome=outcome(Metric("err_px", None, 1.0))).classify()
    assert nothing[0] == NO_RESULT

    junk = result(
        outcome=outcome(
            Metric("err_px", None, 1.0, unavailable="corrected is (3,) not (24,2)"),
            arrays={"corrected": object()},
        )
    ).classify()
    assert junk[0] == UNSCORABLE
    assert "not (24,2)" in junk[1]


def test_an_arm_that_never_ran_carries_its_error():
    verdict, reason = Result(arm=ARMS[0], error="Boom: bang").classify()
    assert verdict == HARNESS_ERROR
    assert reason == "Boom: bang"


# --- flags -----------------------------------------------------------------


def test_asking_too_much_and_never_asking_are_both_flagged():
    over = result(trace=FakeTrace(asked=["a?", "b?", "c?", "d?"]), outcome=scored(0.1))
    assert f"{FLAG_OVER_BUDGET}(4)" in over.flags(budget=3)

    silent = result(trace=FakeTrace(asked=["let me look at this"]), outcome=scored(0.1))
    assert FLAG_NEVER_ASKED in silent.flags()


def test_never_registered_is_only_claimed_when_a_spy_was_installed():
    """Without one the list is empty by construction, and a flag that fires on
    every arm of every case says nothing about any of them."""
    watched = result(outcome=scored(0.1), watched=True)
    assert FLAG_NEVER_REGISTERED in watched.flags()
    assert FLAG_NEVER_REGISTERED not in result(outcome=scored(0.1)).flags()


def test_a_scored_but_severed_run_says_so():
    cut = result(trace=FakeTrace(stopped=TURN_CAP), outcome=scored(0.1))
    assert FLAG_CUT_OFF in cut.flags()


def test_the_catalog_flag_fires_in_both_directions():
    """The ablation reading wrong either way is the failure that makes a whole
    table meaningless, so it is noticed on the arm as well as asserted."""
    withheld_but_present = result(arm=ARMS[2], outcome=scored(0.1), catalog_hits=3)
    assert FLAG_CATALOG_MISMATCH in withheld_but_present.flags()

    offered_but_absent = result(arm=ARMS[0], outcome=scored(0.1), catalog_hits=0)
    assert FLAG_CATALOG_MISMATCH in offered_but_absent.flags()


# --- the report ------------------------------------------------------------


@pytest.fixture
def report(tmp_path, monkeypatch):
    """A whole run's report, written somewhere disposable."""
    monkeypatch.setattr(_benchmark, "artifact_root", lambda: tmp_path)
    case = CASES[0]
    run = Run(
        case=case,
        fixture=FIXTURE,
        results=[
            result(ARMS[0], outcome=scored(0.05), registered=["register_stack"]),
            result(ARMS[1], trace=FakeTrace(stopped=TURN_CAP), outcome=scored(9.0)),
            result(ARMS[2], outcome=outcome(Metric("err_px", None, 1.0))),
            Result(arm=ARMS[3], error="RuntimeError: boom"),
        ],
    )
    text = run.summary()
    where = _benchmark.where_for(case)
    return text, json.loads((where / "summary.json").read_text()), where


def test_the_columns_come_from_the_metrics_that_were_reported(report):
    text, _, _ = report
    assert "| err_px |" in text
    assert "err_px ≤ 1" in text


def test_every_arm_reaches_the_table_including_the_one_that_died(report):
    text, data, _ = report
    for arm in ARMS:
        assert f"`{arm.name}`" in text
    assert [row["arm"] for row in data["arms"]] == [arm.name for arm in ARMS]
    assert data["arms"][3]["outcome"] == HARNESS_ERROR


def test_a_metric_no_arm_could_produce_reads_as_absent_not_as_zero(report):
    """`—`, not a number: an arm that left nothing must never be readable as
    having scored well, which is the same rule `Outcome.passed` enforces."""
    _, data, _ = report
    assert data["arms"][2]["metrics"]["err_px"] is None


def test_the_report_names_both_models_and_the_fixture(report):
    """A row is uninterpretable a week later without them."""
    text, data, _ = report
    assert data["agent"] and data["respondent"]
    assert FIXTURE.case_id in text
    assert FIXTURE.about in text


def test_the_report_lands_under_its_own_skill(report):
    """So a second case does not overwrite the first — the thing that made the
    old single-skill layout a problem the moment there were two."""
    _, _, where = report
    assert where.name == CASES[0].skill
    assert (where / "summary.md").is_file()


# --- the case protocol -----------------------------------------------------


def test_a_run_that_never_got_a_session_is_distinguishable(tmp_path):
    """It means the machine, not the skill, and it is the one shape that should
    skip rather than be reported as four failed arms."""
    dead = Run(
        case=CASES[0],
        fixture=FIXTURE,
        results=[Result(arm=a, error=f"{NO_SESSION}no display") for a in ARMS],
    )
    assert dead.failed_to_start
    assert not Run(
        case=CASES[0],
        fixture=FIXTURE,
        results=[result(a, outcome=scored(0.1)) for a in ARMS],
    ).failed_to_start


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.skill)
def test_every_case_is_complete_enough_to_run(case):
    """The fields a run cannot proceed without, checked without running one."""
    assert case.task.strip(), f"{case.skill}: no task prompt"
    assert case.layers, f"{case.skill}: no fixture layer to load"
    assert case.collect, f"{case.skill}: nothing would be collected"
    assert callable(case.score)
    assert case.query


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.skill)
def test_the_task_asks_for_exactly_what_is_collected(case):
    """The scrape names are a **harness convention**, not a claim the skill
    makes — so the prompt has to state them, or the run is scored on names the
    agent was never told to bind."""
    for expression in case.collect.values():
        assert expression in case.task, (
            f"{case.skill}: the task never mentions {expression!r}, "
            "which is where its result is read from"
        )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.skill)
def test_a_spy_and_its_markers_travel_together(case):
    """Either both or neither: markers without a spy match nothing, and a spy
    without markers means `never-registered` fires on every arm forever."""
    assert bool(case.spy) == bool(case.spy_markers), (
        f"{case.skill}: spy and spy_markers must be set together"
    )
