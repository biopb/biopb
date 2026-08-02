"""Everything asserted about a case's *data*, before anyone pays for a run.

Hermetic and instant, and they run with the ordinary suite. A case is a task
prompt, a persona, a fixture and a verifier, and each of those has a way of
being quietly wrong that no amount of running would reveal:

- a persona that volunteers what it was not asked rescues a bad agent, and a
  green run looks identical;
- a fixture whose truth is visible in its data scores runs on a question they
  could read the answer to;
- a verifier that passes an empty attempt makes every arm look fine;
- a task that asks for one name while the harness scrapes another scores a run
  on something it was never told to bind.

Every check runs over `cases.CASES`, so a skill added to the benchmark is
checked by arriving rather than by someone remembering to write a test for it.
That is the property this layer needs to survive a catalogue of thirty.
"""

from __future__ import annotations

import pytest

from .._validate import NOT_SKILLS
from ..conftest import SKILLS_DIR
from ._fixture import Attempt, Fixture
from .cases import CASES, NOT_BENCHMARKED


def _ids(case):
    return case.skill


#: Built once each: a fixture is megabytes of numpy and several cases share
#: these tests.
_FIXTURES: dict[str, Fixture] = {}


@pytest.fixture(params=CASES, ids=_ids)
def case(request):
    return request.param


@pytest.fixture
def built(case) -> Fixture:
    if case.skill not in _FIXTURES:
        _FIXTURES[case.skill] = case.build()
    return _FIXTURES[case.skill]


def test_there_is_at_least_one_case():
    """The guard against this file going vacuously green, the same shape as
    `test_the_extractor_finds_pkg_tokens` in the contract layer."""
    assert CASES


# --- the catalogue is covered ----------------------------------------------


def _shipped() -> set[str]:
    return {p.stem for p in SKILLS_DIR.glob("*.md") if p.stem not in NOT_SKILLS}


def test_every_shipped_skill_is_benchmarked_or_declared_unbenchmarkable():
    """A skill outside this layer is a decision, and it has to be written down.

    Without this the honest answer to "what does the benchmark cover" is
    "whatever anyone got round to", which is indistinguishable from full
    coverage when read off a green suite.
    """
    uncovered = _shipped() - {c.skill for c in CASES} - set(NOT_BENCHMARKED)
    assert not uncovered, (
        f"these shipped skills are neither benchmarked nor listed in "
        f"cases.NOT_BENCHMARKED with a reason: {sorted(uncovered)}"
    )


def test_nothing_claims_to_cover_a_skill_that_does_not_ship():
    """A case or an exemption left behind by a deleted skill. Cheap to check,
    and it is how the contract layer's module came to be written entirely for a
    skill that no longer existed."""
    shipped = _shipped()
    stale = ({c.skill for c in CASES} | set(NOT_BENCHMARKED)) - shipped
    assert not stale, (
        f"these name skills that are not in the catalogue: {sorted(stale)}"
    )


def test_an_exemption_carries_a_reason():
    for skill, why in NOT_BENCHMARKED.items():
        assert len(why.split()) >= 5, f"{skill}: {why!r} does not say why"


# --- the case is runnable --------------------------------------------------


def test_every_case_is_complete_enough_to_run(case):
    """The fields a run cannot proceed without, checked without running one."""
    assert case.task.strip(), f"{case.skill}: no task prompt"
    assert case.layers, f"{case.skill}: no fixture layer to load"
    assert case.collect, f"{case.skill}: nothing would be collected"
    assert callable(case.score) and callable(case.build)
    assert case.query


def test_the_task_asks_for_exactly_what_is_collected(case):
    """The scrape names are a **harness convention**, not a claim the skill
    makes — so the prompt has to state them, or the run is scored on names the
    agent was never told to bind."""
    for expression in case.collect.values():
        assert expression in case.task, (
            f"{case.skill}: the task never mentions {expression!r}, "
            "which is where its result is read from"
        )


def test_every_layer_kind_is_one_the_harness_can_add(case):
    for layer in case.layers:
        assert layer.kind in ("image", "labels"), (
            f"{case.skill}: layer {layer.name!r} wants a {layer.kind!r} layer"
        )


# --- the fixture -----------------------------------------------------------


def test_the_fixture_keeps_its_truth_out_of_the_data(built):
    """The leak the whole layer depends on not happening. A truth key visible in
    `data` means every run scores perfectly on a question it could read the
    answer to."""
    shared = set(built.data) & set(built.truth)
    assert not shared, f"{built.label}: {sorted(shared)} is both given and withheld"


def test_the_fixture_provides_every_layer_the_case_loads(case, built):
    for layer in case.layers:
        assert layer.key in built.data, (
            f"{case.skill}: layer {layer.name!r} loads data[{layer.key!r}], "
            f"which the fixture does not build ({sorted(built.data)})"
        )


def test_the_fixture_says_where_it_came_from(built):
    """Free text and required. A synthetic seed needs no review; an annotation
    is someone's claim about their own data and is only as good as the review it
    got, and this is where that is recorded."""
    assert built.provenance.strip()
    assert built.about.strip()


def test_a_run_that_left_nothing_scores_nothing(case, built):
    """The anti-vacuous check on the verifier itself.

    A verifier that passes an empty attempt would make every arm look fine,
    including the ones where the agent gave up — and `Outcome.passed` refusing
    to call "nothing scored" a pass only helps if the verifier reports
    unavailable rather than inventing a zero.
    """
    outcome = case.score(built, Attempt(subject="left-nothing"))
    assert not outcome.passed
    assert outcome.metrics, f"{case.skill}: the verifier reported no metrics at all"
    assert all(not m.scored for m in outcome.metrics)
    assert all(m.unavailable for m in outcome.metrics), (
        f"{case.skill}: a metric went unscored without saying why"
    )


def test_every_metric_the_verifier_reports_has_a_tolerance(case, built):
    """Read off the verifier rather than declared twice: the report's columns
    come from the metrics, and a limit of zero would be a silent always-fail."""
    outcome = case.score(built, Attempt(subject="left-nothing"))
    for metric in outcome.metrics:
        assert metric.limit > 0, f"{case.skill}: {metric.name} has no usable limit"


# --- the persona -----------------------------------------------------------


def test_every_fact_reaches_the_prompt(case):
    """The facts are data *and* prose, and the two must not drift. A fact the
    respondent holds but was never told about cannot be asked for, so the
    fixture would be withholding something nobody can obtain."""
    prompt = case.persona.system_prompt()
    for key, value in case.persona.facts.items():
        assert value in prompt, f"{case.skill}: {key!r} never reaches the prompt"


def test_the_persona_is_told_not_to_volunteer(case):
    """The one instruction the whole tier depends on. Asserted on the rendered
    prompt rather than trusted to the template, because the template is exactly
    what a well-meaning edit would loosen."""
    from ._respondent import DONE

    prompt = case.persona.system_prompt()
    assert "never volunteer" in prompt.casefold()
    assert DONE in prompt, "no way to end the conversation"


def test_the_background_gives_nothing_away(case):
    """`background` is what the respondent may share freely, so a private fact
    that leaked into it is available without asking — the fixture would look
    like it tests interaction while handing the answer over."""
    background = case.persona.background.casefold()
    for key, value in case.persona.facts.items():
        assert value.casefold() not in background, (
            f"{case.skill}: {key!r} is in the freely-shared background"
        )


def test_the_persona_knows_the_sample_and_not_the_procedure(case):
    """The two halves of a usable respondent, declared per case.

    It has to be able to answer — a fixture that strips a fact nobody holds is
    unanswerable rather than hard — and it must not have absorbed the skill's
    own vocabulary, or it could answer a question the agent never properly
    asked and the numeric result would stop meaning what it appears to.
    """
    prompt = case.persona.system_prompt().casefold()
    for known in case.persona_must_know:
        assert known.casefold() in prompt, (
            f"{case.skill}: the respondent cannot answer about {known!r}"
        )
    for procedural in case.persona_must_not_know:
        assert procedural.casefold() not in prompt, (
            f"{case.skill}: the respondent knows {procedural!r}, "
            "which is the skill's job"
        )


def test_the_case_declares_what_its_persona_must_hold(case):
    """Both lists non-empty, because either one empty makes the check above
    vacuous — and a vacuous version of it is indistinguishable from a passing
    one from the outside, which is the failure mode this whole file is about."""
    assert case.persona_must_know, f"{case.skill}: nothing declared as askable"
    assert case.persona_must_not_know, f"{case.skill}: no procedural terms fenced off"
