"""The two-model loop, and the trace it leaves behind.

An agent turn that calls tools gets tool results. An agent turn that is plain
text is a **message to the user**, so it goes to the respondent and the reply
becomes the next user turn. That is the whole idea: the agent has no other way
to reach the person who knows which channel is structural, so asking is not
something the harness rewards — it is the only route to the answer.

**The trace is written before any assertion runs.** §5 is the least isolable
tier in the suite: a red run's cause space is the skill body, the model, the
tool schemas, the kernel, Qt, dask and the fixture. An assertion message cannot
tell those apart and a transcript usually can, so the transcript exists
whatever the verdict — that is the mitigation for the cost of testing against
the real environment rather than a stand-in.

Everything is bounded. A model that will not stop, or that loops on a failing
tool call, ends the run at a turn cap rather than burning a budget, and the cap
being hit is recorded as the outcome it is.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ._agent import AgentTurn, ChatAgent
from ._bridge import to_function_tools
from ._models import EmptyCompletion
from ._respondent import DONE, Respondent

#: Bounds on one run. Generous enough for the seven-step drift workflow with
#: mistakes along the way, small enough that a stuck model is cheap.
MAX_TURNS = 40
MAX_TOOL_CALLS = 60

#: Why a run stopped. Recorded rather than inferred, because "the agent
#: finished" and "the agent was cut off" score the same numerically and mean
#: entirely different things about the skill.
FINISHED = "finished"
TURN_CAP = "turn-cap"
TOOL_CAP = "tool-cap"
SILENT = "agent-said-nothing"

#: The two ways a *provider* ends a run, as opposed to a model deciding to.
#: They are named separately from `SILENT` and `FINISHED` because they look
#: exactly like them from the outside and are not the agent's doing: a
#: reasoning model bills its reasoning against `max_tokens`, so a budget that
#: holds the answer can still be spent before the answer starts. Both are
#: scored as harness errors, which is what they are.
AGENT_TRUNCATED = "agent-truncated"
RESPONDENT_FAILED = "respondent-failed"

#: The conversation stopped progressing: the agent keeps talking and never acts
#: again. `SilentRespondent` cannot end a run — it answers "I don't know" to
#: everything, including a sign-off — so an agent that finishes its work and
#: says so is answered with a non-answer and says so again, to the turn cap.
#: Measured: both silent arms did exactly this, trailing 42 and 55 tool-free
#: turns after their last real action, and scored `turn-cap` with
#: `cut-off-but-scored` on work that was complete.
STALLED = "stalled"

#: Consecutive agent turns with no tool call before the run is called stalled.
#: Set from measurement: healthy runs never exceeded **2** in either sweep, and
#: the livelocked ones ran 42 and 55 — so this sits well clear of a legitimate
#: run of questions (the ask budget is 3) and far below the pathology.
MAX_IDLE_TURNS = 8


@dataclass
class Event:
    """One thing that happened, in the order it happened."""

    turn: int
    role: str  # agent | tool | user | harness
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    name: str = ""
    is_error: bool = False
    #: Agent turns only: the provider's own word for why generation stopped.
    #: Recorded because an empty turn cannot otherwise be told apart from a
    #: truncated one after the fact, and that is the distinction a red run
    #: turns on.
    finish_reason: str = ""


@dataclass
class Trace:
    """A whole run, in a form both a human and `ReplayAgent` can read."""

    agent: str
    respondent: str
    task: str
    events: list[Event] = field(default_factory=list)
    stopped: str = ""
    turns_used: int = 0

    @property
    def questions(self) -> list[str]:
        """Everything the agent said *to the user*, asking or not."""
        return [e.text for e in self.events if e.role == "agent" and e.text]

    @property
    def answers(self) -> list[str]:
        """Everything the respondent said back. Zero of these against a
        non-empty :attr:`questions` is the shape of a broken respondent."""
        return [e.text for e in self.events if e.role == "user"]

    @property
    def blocking_questions(self) -> list[str]:
        """The subset that actually asks for something — what §5's "at most
        three blocking checkpoints" is counted against.

        Not the same as :attr:`questions`, and the difference is not pedantry.
        Models narrate: "I found a drift-correction skill, let me read it" is a
        plain-text turn with no tool call, so the loop routes it to the
        respondent like anything else — but it is a status update, not a
        checkpoint, and counting it put a well-behaved run at six over a budget
        of four on the first real measurement.

        **The test is a question mark**, which is a heuristic and is stated as
        one. It cannot see a checkpoint phrased as "let me know which channel
        is structural", and it would miscount a rhetorical question. It is
        transparent and cheap, where the alternative is asking a third model to
        classify each turn — paying for a judgement in a layer built to avoid
        judged verifiers.
        """
        return [text for text in self.questions if "?" in text]

    @property
    def tool_names(self) -> list[str]:
        return [e.name for e in self.events if e.role == "tool"]

    def first_call_of(self, name: str) -> int | None:
        """Index into :attr:`events` of the first call to *name*, or ``None``.

        The gate question: compare it against the index of a question to
        answer "did it ask before it spent".
        """
        for i, e in enumerate(self.events):
            if e.role == "tool" and e.name == name:
                return i
        return None

    def first_question(self) -> int | None:
        for i, e in enumerate(self.events):
            if e.role == "agent" and e.text.strip():
                return i
        return None

    def write(self, where: Path) -> Path:
        """Both forms: one to replay from, one to read."""
        where.mkdir(parents=True, exist_ok=True)
        with (where / "trace.jsonl").open("w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "agent": self.agent,
                        "respondent": self.respondent,
                        "task": self.task,
                        "stopped": self.stopped,
                        "turns_used": self.turns_used,
                    }
                )
                + "\n"
            )
            for event in self.events:
                fh.write(json.dumps(asdict(event)) + "\n")
        (where / "transcript.md").write_text(self._markdown(), encoding="utf-8")
        return where

    def _markdown(self) -> str:
        lines = [
            f"# {self.agent} vs {self.respondent}",
            "",
            f"Stopped: **{self.stopped}** after {self.turns_used} turns.",
            "",
            "## Task",
            "",
            self.task,
            "",
        ]
        for e in self.events:
            if e.role == "agent":
                if e.tool_calls:
                    calls = ", ".join(f"`{c['name']}`" for c in e.tool_calls)
                    lines.append(f"**agent** (turn {e.turn}) calls {calls}")
                if e.text:
                    lines.append(f"**agent → user** (turn {e.turn}): {e.text}")
                # Only when it is the surprising one: `stop`/`tool_calls` is
                # every ordinary turn and would be noise on all of them.
                if e.finish_reason in ("length", "max_tokens"):
                    lines.append(
                        f"_(turn {e.turn} cut off at the token budget: "
                        f"finish_reason={e.finish_reason})_"
                    )
            elif e.role == "tool":
                flag = " *(error)*" if e.is_error else ""
                body = e.text if len(e.text) < 1200 else e.text[:1200] + " …"
                lines.append(f"> `{e.name}`{flag}\n>\n> ```\n> {body}\n> ```")
            elif e.role == "user":
                lines.append(f"**user** (turn {e.turn}): {e.text}")
            else:
                lines.append(f"_{e.role}: {e.text}_")
            lines.append("")
        return "\n".join(lines)


def converse(
    session,
    agent: ChatAgent,
    respondent: Respondent,
    task: str,
    *,
    max_turns: int = MAX_TURNS,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> Trace:
    """Run *agent* against *session*, routing its plain text to *respondent*.

    Returns the :class:`Trace` whatever happened — a run that hit a cap, or
    that crashed a tool on every turn, still produces the record. Scoring is
    somebody else's job, and it happens after.
    """
    tools = to_function_tools(session.tools)
    messages: list[dict] = [{"role": "user", "content": task}]
    trace = Trace(
        agent=getattr(agent, "name", type(agent).__name__),
        respondent=getattr(respondent, "name", type(respondent).__name__),
        task=task,
    )
    calls_made = 0
    idle_turns = 0

    for turn in range(max_turns):
        trace.turns_used = turn + 1
        session._turn = turn
        step: AgentTurn = agent.respond(messages, tools)

        trace.events.append(
            Event(
                turn=turn,
                role="agent",
                text=step.text,
                tool_calls=[
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in step.tool_calls
                ],
                finish_reason=step.finish_reason,
            )
        )

        if step.tool_calls:
            messages.append(
                {
                    # Whatever the provider requires echoed — a reasoning
                    # model's turn is not just text and tool calls, and
                    # dropping the rest fails the *next* request, not this one.
                    **step.provider_fields,
                    "role": "assistant",
                    "content": step.text or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(c.arguments),
                            },
                        }
                        for c in step.tool_calls
                    ],
                }
            )
            for call in step.tool_calls:
                result = session.call(call.name, **call.arguments)
                calls_made += 1
                trace.events.append(
                    Event(
                        turn=turn,
                        role="tool",
                        name=call.name,
                        text=result.text,
                        is_error=result.is_error,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.text or "(no output)",
                    }
                )
            if calls_made >= max_tool_calls:
                trace.stopped = TOOL_CAP
                return trace
            idle_turns = 0

            # A question asked while acting is still a question. Routing used
            # to key on "did this turn call a tool", which conflated *should
            # the user see this* with *did the agent block* — so a model that
            # asked and kept working had the question swallowed, then said "I
            # have asked, I will wait", and the run ended on the sign-off with
            # nobody having been asked anything.
            if step.asks_something:
                try:
                    answer = respondent.reply(step.text)
                except EmptyCompletion as failure:
                    trace.stopped = RESPONDENT_FAILED
                    trace.events.append(
                        Event(
                            turn=turn, role="harness", text=str(failure), is_error=True
                        )
                    )
                    return trace
                # `DONE` here means "not a question to me", not "we are
                # finished": the agent was working, not handing off. Ending
                # the run on it would turn every rhetorical question inside a
                # working turn into a terminated arm — the same bug, mirrored.
                if answer.strip() != DONE:
                    messages.append({"role": "user", "content": answer})
                    trace.events.append(Event(turn=turn, role="user", text=answer))
            continue

        idle_turns += 1

        if step.is_empty:
            # Nothing said and nothing called: there is no next move to make.
            # Whose doing that was is the provider's to say — a model with
            # nothing left to say and a model cut off mid-reasoning produce
            # the identical turn, and only one of them is about the skill.
            if step.was_truncated:
                trace.stopped = AGENT_TRUNCATED
                trace.events.append(
                    Event(
                        turn=turn,
                        role="harness",
                        text=(
                            "agent turn was cut off at the token budget, not "
                            "finished: raise ToolCallingAgent.max_tokens"
                        ),
                        is_error=True,
                    )
                )
            else:
                trace.stopped = SILENT
            return trace

        try:
            answer = respondent.reply(step.text)
        except EmptyCompletion as failure:
            # The respondent never answered. Ending here as FINISHED would
            # report the agent as having handed off, which is the one reading
            # this failure most resembles and the most expensive to believe.
            trace.stopped = RESPONDENT_FAILED
            trace.events.append(
                Event(turn=turn, role="harness", text=str(failure), is_error=True)
            )
            return trace

        if answer.strip() == DONE:
            trace.stopped = FINISHED
            return trace

        messages.append(
            {**step.provider_fields, "role": "assistant", "content": step.text}
        )
        messages.append({"role": "user", "content": answer})
        trace.events.append(Event(turn=turn, role="user", text=answer))

        if idle_turns >= MAX_IDLE_TURNS:
            # Talking without acting, this many turns running. Nothing new is
            # entering the conversation, so the remaining budget buys nothing
            # but tokens and a `turn-cap` on work that may be finished.
            trace.stopped = STALLED
            trace.events.append(
                Event(
                    turn=turn,
                    role="harness",
                    text=(
                        f"no tool call in {idle_turns} turns — the conversation "
                        "stopped progressing; the respondent may be unable to "
                        "end it (SilentRespondent never returns DONE)"
                    ),
                    is_error=True,
                )
            )
            return trace

    trace.stopped = TURN_CAP
    return trace


def scrape(session, trace: Trace, names: dict[str, str]) -> dict[str, Any]:
    """Read the run's results out of the kernel namespace.

    *names* maps the key a verifier wants to the expression to evaluate. A
    missing one comes back absent rather than raising: a run that left nothing
    behind is an ordinary outcome, and `Outcome.passed` already refuses to read
    "nothing scored" as a pass.

    **This is a harness convention, not a claim the skill makes.** The task
    prompt asks the agent to leave its results under particular names; that
    request is the harness's, and it is not counted as something the skill was
    tested on.
    """
    found = {}
    for key, expression in names.items():
        array = session.get_array(expression)
        if array is not None:
            found[key] = array
    trace.events.append(
        Event(
            turn=-1,
            role="harness",
            text=f"scraped {sorted(found)} of {sorted(names)}",
        )
    )
    return found
