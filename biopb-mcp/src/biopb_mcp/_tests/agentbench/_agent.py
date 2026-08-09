"""The agent under test, behind one small protocol.

§5a: **not from the family that wrote the skill.** These bodies were
co-authored with Claude, so a Claude agent can pass by recognising its own
prose rather than by reading it — and §6 already records that blind spots
correlate within a family. The reference agent is therefore a hosted
non-Anthropic model, and the *family* is part of what makes it a valid fixture,
not just the version.

Three implementations, and the first is the one that makes this layer testable
at all:

:class:`ScriptedAgent`
    A canned sequence of turns. No model, no key, no network — and it drives
    the whole conversation loop, respondent hand-off, trace and scraping
    deterministically. Prove the machinery works before paying anything to
    exercise it.

:class:`ToolCallingAgent`
    The real one. Chat-completions with tool calling, which reaches OpenAI,
    Gemini, DeepSeek and Mistral, plus a local Ollama or vLLM, through one
    implementation. Which model, and at which address, comes from
    `_models.py` — the same table the respondent uses, and configured by a
    separate variable so the two sides can differ.

:class:`ReplayAgent`
    Re-feeds a recorded trace, so the structural assertions can be re-checked
    offline and free, and a finding travels as an attachable file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ._bridge import parse_arguments
from ._models import ModelChoice, agent_choice, echoed_fields


class RequestRejected(RuntimeError):
    """The provider refused the request, with the *shape* of what was sent.

    **A rejection names the message it dislikes; the harness has to name the
    conversation it built.** `reasoning_content in the thinking mode must be
    passed back` says a turn was missing something, not which turn or what the
    surrounding messages looked like — and reproducing it costs a paid run,
    because it depends on what the real tools returned. So the shape travels
    with the error: roles, the keys on each message, and how many tool calls
    it carried. Never the content, which is large and is already in the trace.
    """

    def __init__(self, model: str, messages: list[dict], cause: Exception) -> None:
        shape = "\n    ".join(_describe(m) for m in messages[-10:])
        super().__init__(
            f"{model} rejected the request: {cause}\n"
            f"  last {min(len(messages), 10)} of {len(messages)} messages sent:\n"
            f"    {shape}"
        )
        self.messages = messages
        self.cause = cause


def _describe(message: dict) -> str:
    """One message as role + keys + tool-call count. No content."""
    calls = len(message.get("tool_calls") or ())
    keys = ",".join(sorted(k for k in message if k != "role"))
    return f"{message.get('role'):<9} keys=[{keys}]" + (
        f" tool_calls={calls}" if calls else ""
    )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class AgentTurn:
    """One step from the agent: tool calls, a message to the user, or both.

    Both, because a model may narrate and then act in the same turn. The
    conversation loop treats tool calls as taking precedence — a turn that
    called something is not waiting on an answer.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    #: The provider's own word for why generation stopped, carried into the
    #: trace. **A turn with no text and no tool calls is ambiguous without
    #: it**: the model deciding it has nothing left to say and the model being
    #: cut off mid-reasoning are the same empty turn, and they mean opposite
    #: things about the skill. Empty for agents that do not have one.
    finish_reason: str = ""
    #: Fields the provider requires echoed back on the assistant message.
    #:
    #: **A reasoning model's conversation is not just text and tool calls.**
    #: Several providers return the reasoning alongside them and then *reject
    #: the next request* if it is not sent back — DeepSeek's is
    #: `reasoning_content`, and a request that drops it fails with "the
    #: `reasoning_content` in the thinking mode must be passed back to the
    #: API". Rebuilding the assistant turn from the parts this harness happens
    #: to care about silently discards them, so they are carried here and
    #: merged back verbatim, under whatever key they arrived on.
    provider_fields: dict[str, Any] = field(default_factory=dict)
    #: Whatever else the provider returned, for the trace. Never asserted on.
    raw: Any = None

    @property
    def is_question(self) -> bool:
        return not self.tool_calls and bool(self.text.strip())

    @property
    def is_empty(self) -> bool:
        """Nothing said and nothing called — no next move to make."""
        return not self.tool_calls and not self.text.strip()

    @property
    def asks_something(self) -> bool:
        """Whether this turn puts a question to the user, tools or not.

        **Deliberately not** :attr:`is_question`, which is about whether the
        agent *blocked*. A model that asks while it keeps working has still
        asked, and a user sitting there would see it — so routing has to read
        this and the ask budget goes on reading `is_question`.

        The test is a question mark, the same heuristic
        `Trace.blocking_questions` documents: it cannot see "let me know which
        channel is structural", and it over-fires on a rhetorical question.
        The loop's answer to the over-firing is to refuse to *end* a run on a
        turn that also called tools.
        """
        return bool(self.text.strip()) and "?" in self.text

    @property
    def was_truncated(self) -> bool:
        """Cut off at the token budget rather than finished.

        `length` is the OpenAI-compatible spelling and `max_tokens` the
        Anthropic one; both mean the budget ran out mid-generation.
        """
        return self.finish_reason in ("length", "max_tokens")


class ChatAgent(Protocol):
    """What the conversation loop needs. Deliberately not an MCP client — the
    bridge translates, so any chat model with tool calling can sit here."""

    name: str

    def respond(self, messages: list[dict], tools: list[dict]) -> AgentTurn: ...


@dataclass
class ScriptedAgent:
    """A fixed sequence of turns, for testing the harness rather than a skill.

    Running out of script is not an error: it yields a final empty turn, which
    the loop reads as "nothing more to say" and ends on. A test that wants to
    assert the loop stops for the *right* reason should say so explicitly
    rather than rely on exhaustion.
    """

    turns: list[AgentTurn]
    name: str = "scripted"
    seen: list[tuple[int, int]] = field(default_factory=list)
    #: The tool list as offered, not just its length — which list the loop
    #: hands over is itself a thing worth asserting on.
    seen_tools: list[dict] = field(default_factory=list)
    _index: int = 0

    def respond(self, messages: list[dict], tools: list[dict]) -> AgentTurn:
        self.seen.append((len(messages), len(tools)))
        self.seen_tools = list(tools)
        if self._index >= len(self.turns):
            return AgentTurn(text="")
        turn = self.turns[self._index]
        self._index += 1
        return turn


@dataclass
class ReplayAgent:
    """Replays the agent side of a recorded trace.

    What this is for: the structural assertions (§5) are about the *shape* of a
    conversation — how many blocking questions, whether one preceded the
    expensive call — and that shape is fixed once the run happened. Re-checking
    it should not cost another run, and a finding should be reproducible by
    someone who has the file and no key.
    """

    turns: list[AgentTurn]
    name: str = "replay"
    _index: int = 0

    @classmethod
    def from_trace(cls, records: list[dict]) -> ReplayAgent:
        turns = [
            AgentTurn(
                text=r.get("text", ""),
                tool_calls=tuple(
                    ToolCall(c["id"], c["name"], c.get("arguments") or {})
                    for c in r.get("tool_calls") or ()
                ),
                finish_reason=r.get("finish_reason", ""),
            )
            for r in records
            if r.get("role") == "agent"
        ]
        return cls(turns=turns)

    def respond(self, messages: list[dict], tools: list[dict]) -> AgentTurn:
        if self._index >= len(self.turns):
            return AgentTurn(text="")
        turn = self.turns[self._index]
        self._index += 1
        return turn


@dataclass
class ToolCallingAgent:
    """A hosted chat model with tool calling, over the OpenAI-compatible API.

    One implementation reaches OpenAI, Gemini, DeepSeek, Mistral and a local
    Ollama or vLLM, because they all speak this shape — `_models.py` supplies
    the address and the key, exactly as it does for the respondent.

    There is no Anthropic tool-calling agent here, and that is a *consequence*
    of §5a rather than an omission: the one family whose SDK would need its own
    implementation is the family that wrote these skills, so such an agent
    would be unusable the moment it existed. If a skill is ever authored by
    another family, this is where the second implementation goes.

    **Temperature is pinned to 0.** It does not make a run deterministic —
    nothing does, with tool calling and a second model in the loop — but it
    removes the one source of variance that is free to remove.
    """

    choice: ModelChoice | None = None
    temperature: float = 0.0
    #: Headroom for reasoning *and* the tool call that follows it. A reasoning
    #: model bills both from here, and the measured agent spent 3570 of 4096 on
    #: reasoning alone in a first turn against an almost empty context — a
    #: margin thin enough that a harder turn truncates, and a truncated turn is
    #: an empty one. The cost of the larger budget is nothing until it is used.
    #:
    #: 16384 was that reasoning applied once and it was still short. Two full
    #: sweeps truncated one case each — `strahler-ordering` at turn 54 of 90
    #: while computing a fourth variant of a ratio it had already got right
    #: (the bound answer would have passed both tolerances), and
    #: `measure-smlm-resolution` at turn 29 mid-FRC-curve. Both were deep in a
    #: long transcript, which is where the budget has to cover reasoning over
    #: the most context. Doubled again rather than measured, on the same "costs
    #: nothing until used" argument — but a *third* truncation should be
    #: measured rather than doubled a third time, because at that point the
    #: pattern is a turn shape this budget cannot hold, not a number set too
    #: low.
    #:
    #: This is not the turn cap, which is a scored outcome (`out-of-turns`)
    #: exactly like the ask budget: a run that will not converge is supposed to
    #: fail. Truncation is the harness ending a run for a reason that is not
    #: the agent's performance, which is why it scores `agent-truncated` and is
    #: worth spending headroom to avoid.
    max_tokens: int = 32768

    def __post_init__(self) -> None:
        self.choice = self.choice or agent_choice()

    @property
    def name(self) -> str:
        return self.choice.name

    def _client(self):
        from openai import OpenAI

        if why := self.choice.why_unavailable():
            raise RuntimeError(why)
        if self.choice.provider.sdk != "openai":
            raise RuntimeError(
                f"{self.choice.name} does not speak the OpenAI tool-calling API; "
                "see the class docstring on why there is no second backend"
            )
        kwargs = {"api_key": self.choice.key}
        if self.choice.base_url:
            kwargs["base_url"] = self.choice.base_url
        return OpenAI(**kwargs)

    def respond(self, messages: list[dict], tools: list[dict]) -> AgentTurn:
        try:
            completion = self._client().chat.completions.create(
                model=self.choice.model,
                messages=messages,
                tools=tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise RequestRejected(self.name, messages, exc) from exc
        choice = completion.choices[0].message
        calls = tuple(
            ToolCall(
                id=c.id,
                name=c.function.name,
                arguments=parse_arguments(c.function.arguments),
            )
            for c in (choice.tool_calls or ())
        )
        return AgentTurn(
            text=choice.content or "",
            tool_calls=calls,
            finish_reason=completion.choices[0].finish_reason or "",
            provider_fields=echoed_fields(choice),
        )
