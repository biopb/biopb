"""The respondent is a fixture, so it gets the checks a fixture gets.

Hermetic and instant, and they run with the ordinary suite. §6 names the
failure these exist to catch: a respondent that volunteers what it was not
asked rescues a bad agent and silently invalidates every row that depends on
it. Nothing downstream can detect that — a green run looks identical.

This is the §6 analogue of `outcomes`' `test_the_fixture_keeps_its_truth_out_of
_the_data`. There, a truth key visible in the data means every subject scores
perfectly on a procedure that never worked. Here, a private fact reaching the
agent by any route other than asking means the same thing.
"""

from __future__ import annotations

import pytest

from . import _personas
from ._personas import DRIFT_CHANNELS
from ._respondent import DONE, Persona, ScriptedRespondent, SilentRespondent

PERSONAS = [
    getattr(_personas, name)
    for name in dir(_personas)
    if isinstance(getattr(_personas, name), Persona)
]


def _ids(p):
    return p.name


def test_there_is_at_least_one_persona():
    """The guard against this file going vacuously green, the same shape as
    `test_the_extractor_finds_pkg_tokens` in the contract layer."""
    assert PERSONAS


@pytest.mark.parametrize("persona", PERSONAS, ids=_ids)
def test_every_fact_reaches_the_prompt(persona):
    """The facts are data *and* prose, and the two must not drift. A fact the
    respondent holds but was never told about cannot be asked for, so the
    fixture would be withholding something nobody can obtain."""
    prompt = persona.system_prompt()
    for key, value in persona.facts.items():
        assert value in prompt, f"{persona.name}: {key!r} never reaches the prompt"


@pytest.mark.parametrize("persona", PERSONAS, ids=_ids)
def test_the_persona_is_told_not_to_volunteer(persona):
    """The one instruction the whole tier depends on. Asserted on the rendered
    prompt rather than trusted to the template, because the template is exactly
    what a well-meaning edit would loosen."""
    prompt = persona.system_prompt().casefold()
    assert "never volunteer" in prompt
    assert DONE in persona.system_prompt(), "no way to end the conversation"


@pytest.mark.parametrize("persona", PERSONAS, ids=_ids)
def test_the_background_gives_nothing_away(persona):
    """`background` is what the respondent may share freely, so a private fact
    that leaked into it is available without asking — the fixture would look
    like it tests interaction while handing the answer over."""
    background = persona.background.casefold()
    for key, value in persona.facts.items():
        assert value.casefold() not in background, (
            f"{persona.name}: {key!r} is in the freely-shared background"
        )


def test_the_drift_persona_knows_the_sample_and_not_the_procedure():
    """A persona that had read the skill could answer a question the agent
    never properly asked, and the numeric result would stop meaning what it
    appears to. This person knows their cells, not registration."""
    prompt = DRIFT_CHANNELS.system_prompt().casefold()
    for procedural in (
        "reference=",
        "stackreg",
        "register",
        "phase_cross_correlation",
        "structural channel",
    ):
        assert procedural not in prompt, (
            f"the respondent knows {procedural!r}, which is the skill's job"
        )


def test_the_drift_persona_holds_the_fact_the_fixture_strips():
    """The other half: it must actually be able to answer. `_drift_channels`
    withholds which channel is structural, and channel 1 is the answer."""
    # Keys and values both render into the prompt (`- key: value`), and the
    # channel identities live in the keys, so this asks the rendered thing.
    prompt = DRIFT_CHANNELS.system_prompt().casefold()
    assert "channel 0" in prompt and "channel 1" in prompt
    facts = " ".join(DRIFT_CHANNELS.facts.values()).casefold()
    assert "move" in facts, "nothing here says the objects move on their own"
    assert "drift" in facts, "nothing here says the stage drifted"


def test_the_silent_respondent_gives_nothing_and_ends_nothing():
    """The control condition has to answer *unhelpfully*, not terminally: a
    user who says "I don't know" has not left the room, so the agent proceeds
    on a guess and the numeric verifier is what fails it."""
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
