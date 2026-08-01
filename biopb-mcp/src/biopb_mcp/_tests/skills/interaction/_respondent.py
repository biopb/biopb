"""The simulated user: a persona, a few private facts, and nothing else.

**The respondent prompt is a fixture and gets reviewed like one.** A chatty
respondent that volunteers the structural channel rescues a bad agent and
silently invalidates the suite — the whole tier rests on the agent having to
*ask*.

Two rules hold it to that, and both are structural rather than hoped for:

**Skill-blind.** The respondent never sees the body. It cannot paraphrase step
2 back at the agent, because it does not know step 2 exists. Claude is fine
here even though it co-authored the skills (§6a) — holding a persona and
answering from a fact table is not a job where family contamination helps.

**Facts are data, not prose.** :class:`Persona` carries its private facts as a
mapping *and* renders them into the prompt, so a test can assert that none of
them appears in anything handed to the agent. A persona whose prompt drifted
from its fact table would be exactly the silent failure this layer cannot
detect from the outside.

:class:`SilentRespondent` is the control condition. A run against a respondent
that answers nothing **must fail** the numeric verifier; if it does not, the
fixture's asymmetry is decorative and the layer is measuring something else.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

#: What a respondent says instead of answering, when the agent's message is a
#: hand-off rather than a question. The loop ends on it.
DONE = "__BIOPB_DONE__"

RESPONDENT_KEY_ENV = "ANTHROPIC_API_KEY"
RESPONDENT_MODEL_ENV = "BIOPB_SKILL_RESPONDENT_MODEL"
DEFAULT_RESPONDENT_MODEL = "claude-sonnet-5"

_RULES = f"""
You are a microscopist talking to an analysis assistant about your own data.

How to answer:
- Answer only what you were actually asked, in one or two sentences.
- Never volunteer anything you were not asked. This matters more than being
  helpful: if the assistant does not ask, it does not find out.
- If you are asked something not covered by what you know, say you do not know.
  Do not guess and do not reason it out from the images.
- You cannot see the assistant's screen and you do not write code.
- If the assistant's message is a summary, a result, or a sign-off rather than
  a question to you, reply with exactly {DONE} and nothing else.

What you know about this experiment:
"""


@dataclass(frozen=True)
class Persona:
    """A respondent fixture: who they are, and what only they know.

    ``facts`` is the load-bearing field. It is rendered into the prompt *and*
    kept as data, so `test_personas` can assert that no fact reaches the agent
    by another route. Keeping the two in one object is what stops a prompt from
    quietly drifting away from the fact table it is supposed to encode.
    """

    name: str
    #: What only this person knows. Values are what the agent must not be told
    #: except by asking.
    facts: Mapping[str, str]
    #: Anything true but freely available -- context that does not give the
    #: answer away, so the conversation is not absurd.
    background: str = ""

    def system_prompt(self) -> str:
        known = "\n".join(f"- {k}: {v}" for k, v in self.facts.items())
        parts = [_RULES.strip(), known]
        if self.background:
            parts.append(f"\nBackground you may share freely:\n{self.background}")
        return "\n".join(parts)


class Respondent(Protocol):
    name: str

    def reply(self, message: str) -> str: ...


@dataclass
class SilentRespondent:
    """Answers nothing, ever. The control condition.

    Not a straw man: "I don't know" is what a real user says about half the
    metadata they are asked for, and `calibrated-measurements` specifies that
    branch explicitly. Here it does double duty as the calibration — a fixture
    whose outcome does not change when the answers stop was never testing the
    asking.
    """

    name: str = "silent"
    said: list[str] = field(default_factory=list)

    def reply(self, message: str) -> str:
        self.said.append(message)
        return "I don't know, sorry — I'd have to go back and check."


@dataclass
class ScriptedRespondent:
    """Answers by keyword match, for testing the harness without a model.

    Rules are ``(substring, answer)`` in order; the first whose substring
    appears in the agent's message, case-folded, wins. Nothing matched means
    the same "I don't know" a real respondent would give, because a scripted
    fixture that answered everything would make the loop look better than it is.
    """

    rules: list[tuple[str, str]]
    name: str = "scripted"
    fallback: str = "I don't know, sorry."
    heard: list[str] = field(default_factory=list)

    def reply(self, message: str) -> str:
        self.heard.append(message)
        low = message.casefold()
        for needle, answer in self.rules:
            if needle.casefold() in low:
                return answer
        return self.fallback


def respondent_key() -> str:
    """``ANTHROPIC_API_KEY`` if set, else ``""``. Presence only, never logged."""
    return RESPONDENT_KEY_ENV if os.environ.get(RESPONDENT_KEY_ENV) else ""


@dataclass
class ClaudeRespondent:
    """The real respondent. Persona in, one short answer out.

    Stateful across a run: it keeps its own view of the conversation so that
    "you already told me that" behaves like a person rather than a lookup
    table. It never sees the tool calls, only what the agent said to it —
    which is exactly a user's view.
    """

    persona: Persona
    model: str = ""
    name: str = ""
    max_tokens: int = 300
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = (
            self.model
            or os.environ.get(RESPONDENT_MODEL_ENV)
            or DEFAULT_RESPONDENT_MODEL
        )
        self.name = self.name or f"claude:{self.model}"

    def reply(self, message: str) -> str:
        import anthropic

        if not respondent_key():
            raise RuntimeError(f"no respondent key: set {RESPONDENT_KEY_ENV}")
        self.history.append({"role": "user", "content": message})
        response = anthropic.Anthropic().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.persona.system_prompt(),
            messages=self.history,
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        self.history.append({"role": "assistant", "content": text or DONE})
        return text
