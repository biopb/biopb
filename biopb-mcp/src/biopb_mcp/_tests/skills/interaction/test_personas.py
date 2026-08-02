"""The respondent is a fixture, so it gets the checks a fixture gets.

Hermetic and instant, and they run with the ordinary suite. §6 names the failure
these exist to catch: a respondent that volunteers what it was not asked rescues
a bad agent and silently invalidates every row that depends on it. Nothing
downstream can detect that — a green run looks identical.

This is the §6 analogue of `outcomes`' `test_the_fixture_keeps_its_truth_out_of
_the_data`. There, a truth key visible in the data means every subject scores
perfectly on a procedure that never worked. Here, a private fact reaching the
agent by any route other than asking means the same thing.

Every check runs over `cases.CASES`, so a skill added to the benchmark is
persona-checked by arriving, not by someone remembering to write a test for it.
"""

from __future__ import annotations

import pytest

from ._respondent import DONE, ScriptedRespondent, SilentRespondent
from .cases import CASES


def _ids(case):
    return case.skill


def test_there_is_at_least_one_case():
    """The guard against this file going vacuously green, the same shape as
    `test_the_extractor_finds_pkg_tokens` in the contract layer."""
    assert CASES


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_every_fact_reaches_the_prompt(case):
    """The facts are data *and* prose, and the two must not drift. A fact the
    respondent holds but was never told about cannot be asked for, so the
    fixture would be withholding something nobody can obtain."""
    prompt = case.persona.system_prompt()
    for key, value in case.persona.facts.items():
        assert value in prompt, f"{case.skill}: {key!r} never reaches the prompt"


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_persona_is_told_not_to_volunteer(case):
    """The one instruction the whole tier depends on. Asserted on the rendered
    prompt rather than trusted to the template, because the template is exactly
    what a well-meaning edit would loosen."""
    prompt = case.persona.system_prompt()
    assert "never volunteer" in prompt.casefold()
    assert DONE in prompt, "no way to end the conversation"


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_background_gives_nothing_away(case):
    """`background` is what the respondent may share freely, so a private fact
    that leaked into it is available without asking — the fixture would look
    like it tests interaction while handing the answer over."""
    background = case.persona.background.casefold()
    for key, value in case.persona.facts.items():
        assert value.casefold() not in background, (
            f"{case.skill}: {key!r} is in the freely-shared background"
        )


@pytest.mark.parametrize("case", CASES, ids=_ids)
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


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_case_declares_what_its_persona_must_hold(case):
    """Both lists non-empty, because either one empty makes the check above
    vacuous — and a vacuous version of it is indistinguishable from a passing
    one from the outside, which is the failure mode this whole file is about."""
    assert case.persona_must_know, f"{case.skill}: nothing declared as askable"
    assert case.persona_must_not_know, f"{case.skill}: no procedural terms fenced off"


def test_the_silent_respondent_gives_nothing_and_ends_nothing():
    """The control condition has to answer *unhelpfully*, not terminally: a user
    who says "I don't know" has not left the room, so the agent proceeds on a
    guess and the numeric verifier is what fails it."""
    respondent = SilentRespondent()
    answer = respondent.reply("Which channel is the structural one?")
    assert DONE not in answer
    assert "channel" not in answer.casefold()


def test_a_scripted_respondent_matches_on_what_was_asked():
    respondent = ScriptedRespondent([("which channel", "Channel 1, the membrane.")])
    assert respondent.reply("WHICH CHANNEL is structural?") == (
        "Channel 1, the membrane."
    )
    assert "don't know" in respondent.reply("What is the pixel size?")
