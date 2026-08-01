"""The agent under test, behind one small protocol.

§6a: **not from the family that wrote the skill.** These bodies were
co-authored with Claude, so a Claude agent can pass by recognising its own
prose rather than by reading it — and §7 already records that blind spots
correlate within a family. The reference agent is therefore a hosted
non-Anthropic model, and the *family* is part of what makes it a valid fixture,
not just the version.

Three implementations, and the first is the one that makes this layer testable
at all:

:class:`ScriptedAgent`
    A canned sequence of turns. No model, no key, no network — and it drives
    the whole conversation loop, respondent hand-off, trace and scraping
    deterministically. The same move as the scripted subjects in §6b: prove the
    machinery works before paying anything to exercise it.

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
from ._models import ModelChoice, agent_choice


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
    #: Whatever the provider returned, for the trace. Never asserted on.
    raw: Any = None

    @property
    def is_question(self) -> bool:
        return not self.tool_calls and bool(self.text.strip())


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
    _index: int = 0

    def respond(self, messages: list[dict], tools: list[dict]) -> AgentTurn:
        self.seen.append((len(messages), len(tools)))
        if self._index >= len(self.turns):
            return AgentTurn(text="")
        turn = self.turns[self._index]
        self._index += 1
        return turn


@dataclass
class ReplayAgent:
    """Replays the agent side of a recorded trace.

    What this is for: the structural assertions (§6) are about the *shape* of a
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
    of §6a rather than an omission: the one family whose SDK would need its own
    implementation is the family that wrote these skills, so such an agent
    would be unusable the moment it existed. If a skill is ever authored by
    another family, this is where the second implementation goes.

    **Temperature is pinned to 0.** It does not make a run deterministic —
    nothing does, with tool calling and a second model in the loop — but it
    removes the one source of variance that is free to remove.
    """

    choice: ModelChoice | None = None
    temperature: float = 0.0
    max_tokens: int = 4096

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
        completion = self._client().chat.completions.create(
            model=self.choice.model,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
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
            raw={"finish_reason": completion.choices[0].finish_reason},
        )
