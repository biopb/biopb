"""The simulated user: a persona, a few private facts, and nothing else.

**The respondent prompt is a fixture and gets reviewed like one.** A chatty
respondent that volunteers the structural channel rescues a bad agent and
silently invalidates the suite — the whole tier rests on the agent having to
*ask*.

Two rules hold it to that, and both are structural rather than hoped for:

**Skill-blind.** The respondent never sees the body. It cannot paraphrase step
2 back at the agent, because it does not know step 2 exists. That is also why
§5a does not constrain *which* model plays this part: holding a persona and
answering from a fact table is not a job where knowing the skills helps, so any
provider will do and `_models.py` picks it. Anthropic is one option, not the
assumption.

**Facts are data, not prose.** :class:`Persona` carries its private facts as a
mapping *and* renders them into the prompt, so a test can assert that none of
them appears in anything handed to the agent. A persona whose prompt drifted
from its fact table would be exactly the silent failure this layer cannot
detect from the outside.

:class:`SilentRespondent` is the control condition. A run against a respondent
that answers nothing **must fail** the numeric verifier; if it does not, the
fixture's asymmetry is decorative and the layer is measuring something else.

:class:`BriefedRespondent` is the *other* control, and it varies the opposite
thing. Silent withholds the fact; briefed withholds only the asking — the same
facts arrive in the task prompt through :meth:`Persona.briefing`, and nobody
answers afterwards. Between them the three respondents separate two questions a
`model` run cannot tell apart: whether the fact was obtainable from the pixels
(model against silent) and what it cost to have to elicit it rather than be
handed it (model against briefed).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ._models import EmptyCompletion, TextBackend, respondent_choice, text_backend

__all__ = [
    "DONE",
    "BriefedRespondent",
    "EmptyCompletion",
    "ModelRespondent",
    "Persona",
    "Respondent",
    "ScriptedRespondent",
    "SilentRespondent",
    "model_respondent",
]

#: What a respondent says instead of answering, when the agent's message is a
#: hand-off rather than a question. The loop ends on it.
DONE = "__BIOPB_DONE__"

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

#: The header the same fact table gets when it is handed to the *agent* instead
#: of held back.
#:
#: **It countermands the task above it**, and has to. A case's prompt tells the
#: agent the microscopist is there and how to reach them — written for the
#: condition every other switch value runs — and a brief that only added facts
#: would leave the run holding two contradictory instructions and reasonably
#: spending turns on the wrong one. Saying so here rather than editing that line
#: out of every case module keeps one prompt per case: the interactive text is
#: what a case *is*, and this switch is a configuration on top of it. The
#: respondent enforces the same thing from the other end for an agent that asks
#: anyway (:class:`BriefedRespondent`).
_BRIEFING = """
Before you start: the person who acquired this data is not available for this
session, so anything above about asking them does not apply here. Instead, here
is everything they know about it, and this is the whole of it:
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

    def _known(self) -> str:
        """The fact table as prose. One renderer, because the two audiences
        below must see the same facts — a briefing that carried a different set
        from the one `silent` withholds would make the pair of runs a comparison
        between two fixtures rather than between two conditions."""
        return "\n".join(f"- {k}: {v}" for k, v in self.facts.items())

    def system_prompt(self) -> str:
        parts = [_RULES.strip(), self._known()]
        if self.background:
            parts.append(f"\nBackground you may share freely:\n{self.background}")
        return "\n".join(parts)

    def briefing(self) -> str:
        """The same facts, disclosed to the agent up front rather than withheld.

        What `--bench-responder=briefed` folds into the task prompt.

        :attr:`facts` and **only** facts. `background` is second-person prose
        addressed to whoever is playing the part — "you acquired these fields,
        you are happy to answer questions about the camera" — so handing it to
        the agent would tell it that *it* ran the experiment, on top of
        re-offering the conversation this switch removes. Nothing is lost by
        leaving it out: `test_cases.py` already asserts no fact value appears in
        it, which is exactly the assertion that it carries no information.
        """
        return f"{_BRIEFING.strip()}\n{self._known()}"


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
class BriefedRespondent:
    """Answers nothing, because everything was already said in the brief.

    The other half of `--bench-responder=briefed`, and the reason it is a
    separate class rather than a reuse of :class:`SilentRespondent`: what it
    says back has to be *true of the configuration*. "I'd have to go back and
    check" invites the agent to wait for a person; "you already have all of it"
    tells it to proceed, which is what a briefed run is asking it to do.

    Local and constant like the silent one, so a briefed session reaches for no
    respondent model, needs no respondent key, and spends nothing proving one
    (`_engine.models_in_play`). It also cannot leak: the reply is a literal, so
    there is no route by which a fact not in the brief reaches the agent.

    Like the silent one it never returns :data:`DONE` — ending a run is the
    agent's to declare with the completion sentinel, and a respondent that
    cannot judge should not be the thing that decides.
    """

    name: str = "briefed"
    said: list[str] = field(default_factory=list)

    def reply(self, message: str) -> str:
        self.said.append(message)
        return (
            "I've told you everything I know about this already — it's all in "
            "what I sent you at the start, and I have nothing to add. Go ahead "
            "with what you have."
        )


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


@dataclass
class ModelRespondent:
    """A real model playing the persona. Provider is a parameter, not a fact.

    Anthropic is one choice among several and nothing here assumes it —
    `_models.py` owns that selection, and agent and respondent are configured
    independently so they can sit on different vendors, or on the same
    compatible API at two different addresses.

    Stateful across a run: it keeps its own view of the conversation so that
    "you already told me that" behaves like a person rather than a lookup
    table. It never sees tool calls or their output, only what the agent said
    *to it* — which is exactly a user's view, and the reason a run cannot be
    rescued by a respondent noticing a bad array.
    """

    persona: Persona
    backend: TextBackend
    #: Sized for the *reasoning*, not the answer. A persona reply is one or two
    #: sentences, and 300 tokens was set against that — but a reasoning model
    #: bills its reasoning from this same budget, and on the measured case
    #: spent 1772 of them before writing a word. Below roughly 4k the reply
    #: never starts, and the run ends at the agent's first question.
    max_tokens: int = 8192
    history: list[dict] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.backend.name}/{self.persona.name}"

    def reply(self, message: str) -> str:
        """The persona's answer, or :class:`EmptyCompletion` if there is none.

        **The failure is not caught here**, and that is deliberate. `DONE` ends
        the run and is scored as the agent finishing, so returning it for a
        provider that gave no text spends a whole arm and reports the loss
        against the skill. The loop stops on the exception with a reason of its
        own instead.
        """
        self.history.append({"role": "user", "content": message})
        reply = self.backend.complete(
            system=self.persona.system_prompt(),
            messages=self.history,
            max_tokens=self.max_tokens,
        )
        # The same echo the agent's loop does, for the same reason: this
        # history is replayed on every later question, so a turn stripped of
        # what the provider wants back fails the *next* one.
        self.history.append(
            {**reply.provider_fields, "role": "assistant", "content": reply.text}
        )
        return reply.text


def model_respondent(persona: Persona, **kwargs) -> ModelRespondent:
    """A respondent for *persona*, on whatever `BIOPB_RESPONDENT` names."""
    return ModelRespondent(
        persona=persona, backend=text_backend(respondent_choice()), **kwargs
    )
