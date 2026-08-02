"""The loop, driven by a scripted agent — no model, no session, no key.

These are machinery tests and they run with the **ordinary suite**, like
`outcomes/test_outcome_protocol.py`: a break in the conversation loop should
surface as a normal red test, not be discovered by someone mid-diagnosis with
a paid run.

The session here is a stand-in, and that is the one place in this package where
that is the right answer. What is under test is the loop's own contract — route
plain text to the respondent, route tool calls to the session, record both, stop
for a stated reason — and none of that is a claim about the runtime. The real
session is covered next door by `test_session_smoke.py`, against the real thing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from ._agent import AgentTurn, ReplayAgent, ScriptedAgent, ToolCall
from ._conversation import (
    FINISHED,
    SILENT,
    TOOL_CAP,
    TURN_CAP,
    Trace,
    converse,
    scrape,
)
from ._respondent import DONE, ScriptedRespondent, SilentRespondent
from ._session import ToolResult, ToolSpec


@dataclass
class FakeSession:
    """Just enough session to exercise the loop. Not a model of the runtime."""

    tools: list[ToolSpec] = field(
        default_factory=lambda: [
            ToolSpec("execute_code", "run python", {"type": "object"}),
            ToolSpec("server_status", "status", {"type": "object"}),
        ]
    )
    results: dict[str, ToolResult] = field(default_factory=dict)
    arrays: dict[str, object] = field(default_factory=dict)
    calls: list[tuple[int, str, dict]] = field(default_factory=list)
    _turn: int = 0

    def call(self, name, /, **arguments):
        self.calls.append((self._turn, name, dict(arguments)))
        return self.results.get(name, ToolResult(name, f"{name} ok"))

    def get_array(self, expression):
        return self.arrays.get(expression)


def _says(text):
    return AgentTurn(text=text)


def _calls(name, **arguments):
    return AgentTurn(tool_calls=(ToolCall(f"c-{name}", name, arguments),))


# --- routing ---------------------------------------------------------------


def test_plain_text_goes_to_the_respondent_and_the_answer_comes_back():
    """The mechanism the whole tier rests on: the agent's only route to a fact
    it was not given is to say something to the user."""
    agent = ScriptedAgent([_says("Which channel is structural?"), _says("Done.")])
    respondent = ScriptedRespondent([("structural", "Channel 1, the membrane.")])
    session = FakeSession()

    trace = converse(session, agent, respondent, task="correct the drift")

    assert respondent.heard == ["Which channel is structural?", "Done."]
    assert "Channel 1, the membrane." in [
        e.text for e in trace.events if e.role == "user"
    ]
    # The answer must be in what the agent sees next, or asking bought nothing.
    assert agent.seen[-1][0] > agent.seen[0][0]


def test_tool_calls_go_to_the_session_and_never_to_the_respondent():
    """A turn that acted is not waiting on an answer, so the user is not
    disturbed by it. A respondent that heard tool output would be a very
    different fixture."""
    agent = ScriptedAgent([_calls("server_status"), _says("All set.")])
    respondent = ScriptedRespondent([])
    session = FakeSession()

    trace = converse(session, agent, respondent, task="t")

    assert trace.tool_names == ["server_status"]
    assert respondent.heard == ["All set."]


def test_a_turn_that_both_speaks_and_calls_is_not_a_question():
    """Models narrate while acting. Counting that as a blocking question would
    inflate every structural assertion in §5."""
    agent = ScriptedAgent(
        [
            AgentTurn(
                text="Let me look.", tool_calls=(ToolCall("i", "server_status", {}),)
            )
        ]
    )
    respondent = ScriptedRespondent([])

    converse(FakeSession(), agent, respondent, task="t")

    assert respondent.heard == [], "narration was mistaken for a question"


# --- stopping --------------------------------------------------------------


def test_the_respondent_ends_the_run_with_the_sentinel():
    agent = ScriptedAgent([_says("Here is the corrected movie.")])
    respondent = ScriptedRespondent([("corrected", DONE)])

    trace = converse(FakeSession(), agent, respondent, task="t")

    assert trace.stopped == FINISHED


def test_a_silent_agent_stops_the_run():
    """Nothing said and nothing called: there is no next move to make, and
    spinning to the turn cap would only hide it."""
    trace = converse(FakeSession(), ScriptedAgent([]), ScriptedRespondent([]), task="t")
    assert trace.stopped == SILENT


def test_the_turn_cap_is_recorded_as_the_outcome_it_is():
    """ "Finished" and "was cut off" score the same numerically and mean
    completely different things about a skill, so the reason is recorded rather
    than inferred."""
    agent = ScriptedAgent([_says("Are you there?")] * 50)
    respondent = ScriptedRespondent([("there", "Yes.")])

    trace = converse(FakeSession(), agent, respondent, task="t", max_turns=4)

    assert trace.stopped == TURN_CAP and trace.turns_used == 4


def test_the_tool_cap_bounds_a_model_stuck_in_a_loop():
    agent = ScriptedAgent([_calls("server_status")] * 50)

    trace = converse(
        FakeSession(),
        agent,
        ScriptedRespondent([]),
        task="t",
        max_tool_calls=3,
    )

    assert trace.stopped == TOOL_CAP and len(trace.tool_names) == 3


# --- the trace -------------------------------------------------------------


def test_the_trace_answers_the_gate_spy_question():
    """Structural assertion (§5): did a blocking question precede the expensive
    call, or follow it? That is an ordering question about the record."""
    asked_first = converse(
        FakeSession(),
        ScriptedAgent(
            [_says("Field or objects?"), _calls("execute_code", python_code="x")]
        ),
        ScriptedRespondent([("field", "The stage drifted.")]),
        task="t",
    )
    assert asked_first.first_question() < asked_first.first_call_of("execute_code")

    acted_first = converse(
        FakeSession(),
        ScriptedAgent(
            [_calls("execute_code", python_code="x"), _says("Field or objects?")]
        ),
        ScriptedRespondent([("field", "The stage drifted.")]),
        task="t",
    )
    assert acted_first.first_question() > acted_first.first_call_of("execute_code")


def test_questions_counts_only_what_was_said_to_the_user():
    agent = ScriptedAgent(
        [_says("One?"), _calls("server_status"), _says("Two?"), _says("Done.")]
    )
    trace = converse(
        FakeSession(),
        agent,
        ScriptedRespondent([("done", DONE)]),
        task="t",
    )
    assert trace.questions == ["One?", "Two?", "Done."]


def test_narration_is_not_a_blocking_question():
    """Models narrate their plan in a turn of their own and act in the next.
    The loop routes that to the respondent like any plain text -- but it is a
    status update, not a checkpoint, and counting it put a well-behaved run six
    over a budget of four on the first real measurement of this layer."""
    agent = ScriptedAgent(
        [
            _says("I found a drift-correction skill. Let me read it."),
            _calls("server_status"),
            _says("Which channel is structural?"),
            _says("The correction is complete."),
        ]
    )
    trace = converse(
        FakeSession(),
        agent,
        ScriptedRespondent([("channel", "Channel 1."), ("complete", DONE)]),
        task="t",
    )

    assert len(trace.questions) == 3
    assert trace.blocking_questions == ["Which channel is structural?"]


def test_the_trace_is_written_in_both_forms(tmp_path):
    """One to replay from, one to read. Written whatever the verdict, because
    an assertion message cannot tell a bad skill from a bad kernel."""
    trace = converse(
        FakeSession(),
        ScriptedAgent([_calls("server_status"), _says("Done.")]),
        ScriptedRespondent([("done", DONE)]),
        task="correct the drift",
    )
    where = trace.write(tmp_path / "run")

    lines = (where / "trace.jsonl").read_text().splitlines()
    header = json.loads(lines[0])
    assert header["stopped"] == FINISHED and header["task"] == "correct the drift"
    assert all(json.loads(line) for line in lines[1:])

    markdown = (where / "transcript.md").read_text()
    assert "server_status" in markdown and "Done." in markdown


def test_a_written_trace_replays(tmp_path):
    """A finding travels as a file: someone with the trace and no key can
    re-check every structural assertion made about it."""
    original = converse(
        FakeSession(),
        ScriptedAgent(
            [_says("Which channel?"), _calls("server_status"), _says("Bye.")]
        ),
        ScriptedRespondent([("channel", "Channel 1."), ("bye", DONE)]),
        task="t",
    )
    where = original.write(tmp_path / "run")
    records = [
        json.loads(line)
        for line in (where / "trace.jsonl").read_text().splitlines()[1:]
    ]

    replayed = converse(
        FakeSession(),
        ReplayAgent.from_trace(records),
        ScriptedRespondent([("channel", "Channel 1."), ("bye", DONE)]),
        task="t",
    )

    assert replayed.questions == original.questions
    assert replayed.tool_names == original.tool_names


# --- the control condition -------------------------------------------------


def test_the_silent_respondent_answers_nothing_but_keeps_talking():
    """The calibration for every interaction case. It must not end the run --
    a user who says "I don't know" has not left the room -- so the agent gets
    to proceed on a guess, and the numeric verifier is what fails it."""
    respondent = SilentRespondent()
    agent = ScriptedAgent(
        [_says("Which channel is structural?"), _says("OK, guessing.")]
    )

    trace = converse(FakeSession(), agent, respondent, task="t")

    assert len(respondent.said) == 2
    assert trace.stopped != FINISHED
    answers = [e.text for e in trace.events if e.role == "user"]
    assert all("don't know" in a for a in answers)


def test_an_unmatched_question_is_not_silently_answered():
    """A scripted respondent that answered everything would make the loop look
    better than it is, so anything unmatched falls through to "I don't know"."""
    respondent = ScriptedRespondent([("spacing", "0.325 µm")])
    assert "don't know" in respondent.reply("Which channel is structural?")


# --- scraping --------------------------------------------------------------


def test_scrape_takes_what_is_there_and_records_what_is_not():
    """A run that left nothing behind is an ordinary outcome. It has to arrive
    as absence, which `Outcome.passed` already refuses to read as a pass."""
    session = FakeSession(arrays={"offsets": [1, 2, 3]})
    trace = Trace(agent="a", respondent="r", task="t")

    found = scrape(session, trace, {"offsets": "offsets", "corrected": "corrected"})

    assert set(found) == {"offsets"}
    note = [e for e in trace.events if e.role == "harness"][-1].text
    assert "['offsets']" in note and "'corrected'" in note


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"python_code": "x = 1"}', {"python_code": "x = 1"}),
        ({"already": "a dict"}, {"already": "a dict"}),
        ("", {}),
        (None, {}),
    ],
)
def test_tool_arguments_arrive_however_the_provider_sent_them(raw, expected):
    from ._bridge import parse_arguments

    assert parse_arguments(raw) == expected


def test_malformed_tool_arguments_are_a_finding_not_a_crash():
    """A model that cannot form a tool call is a fact about the model, and the
    trace is where it belongs -- not a traceback out of the harness."""
    from ._bridge import parse_arguments

    assert parse_arguments("{not json") == {"__malformed__": "{not json"}
