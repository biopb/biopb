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


@dataclass
class Event:
    """One thing that happened, in the order it happened."""

    turn: int
    role: str  # agent | tool | user | harness
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    name: str = ""
    is_error: bool = False


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
            )
        )

        if step.tool_calls:
            messages.append(
                {
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
            continue

        if not step.text.strip():
            # Nothing said and nothing called: there is no next move to make.
            trace.stopped = SILENT
            return trace

        answer = respondent.reply(step.text)
        if answer.strip() == DONE:
            trace.stopped = FINISHED
            return trace

        messages.append({"role": "assistant", "content": step.text})
        messages.append({"role": "user", "content": answer})
        trace.events.append(Event(turn=turn, role="user", text=answer))

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
