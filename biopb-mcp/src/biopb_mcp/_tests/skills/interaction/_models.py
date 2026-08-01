"""Which model sits on each side of the conversation, and how to reach it.

Both sides of §6 are chat models, so provider selection is one concern and
lives here rather than being hardcoded into either. The two sides are named and
configured **independently**:

```
BIOPB_SKILL_AGENT=openai:gpt-5
BIOPB_SKILL_RESPONDENT=anthropic:claude-sonnet-5
```

That independence is not cosmetic. Anthropic is *a* respondent choice, not the
respondent, and pointing both sides at the same compatible API — two Ollama
models, or a hosted agent against a local respondent — has to be expressible.
An earlier version hardcoded the respondent to Anthropic and read the agent's
endpoint from the shared `OPENAI_BASE_URL`, which made "one on each of two
compatible endpoints" impossible to say.

**What §6a constrains is the agent, and only the agent**: it must not come from
the family that wrote the skill, because it could then pass by recognising its
own prose. The respondent holds a persona and answers from a fact table, which
is not a job where family contamination helps, so nothing here restricts it.
The harness records both names in the trace and lets a human judge the pairing.

A provider is a `(sdk, base_url, key_env)` triple, and most of them are the
OpenAI-compatible API with a different address in front — which is why one
implementation reaches nearly all of them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

#: Which side is being configured. Separate variables so agent and respondent
#: can sit on different providers, or the same one at different addresses.
AGENT_ENV = "BIOPB_SKILL_AGENT"
RESPONDENT_ENV = "BIOPB_SKILL_RESPONDENT"
AGENT_BASE_URL_ENV = "BIOPB_SKILL_AGENT_BASE_URL"
RESPONDENT_BASE_URL_ENV = "BIOPB_SKILL_RESPONDENT_BASE_URL"

DEFAULT_AGENT = "openai:gpt-5"
DEFAULT_RESPONDENT = "anthropic:claude-sonnet-5"


@dataclass(frozen=True)
class Provider:
    """How to reach one vendor. ``sdk`` is the only real branch."""

    name: str
    sdk: str  # "openai" | "anthropic"
    key_env: str
    base_url: str = ""
    #: True when this vendor authored the skills in this repo. §6a reads it.
    wrote_these_skills: bool = False


PROVIDERS: dict[str, Provider] = {
    "openai": Provider("openai", "openai", "OPENAI_API_KEY"),
    "anthropic": Provider(
        "anthropic", "anthropic", "ANTHROPIC_API_KEY", wrote_these_skills=True
    ),
    "gemini": Provider(
        "gemini",
        "openai",
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "deepseek": Provider(
        "deepseek", "openai", "DEEPSEEK_API_KEY", "https://api.deepseek.com"
    ),
    # Local, and the key is a placeholder the SDK insists on rather than a
    # secret. Useful for rehearsing a run without spending anything.
    "ollama": Provider(
        "ollama", "openai", "BIOPB_SKILL_OLLAMA_KEY", "http://localhost:11434/v1"
    ),
}


@dataclass(frozen=True)
class ModelChoice:
    """A resolved ``provider:model`` plus where to send it."""

    provider: Provider
    model: str
    base_url: str = ""

    @property
    def name(self) -> str:
        return f"{self.provider.name}:{self.model}"

    @property
    def key(self) -> str:
        """The API key, read at call time. Never stored, logged or traced."""
        if self.provider.name == "ollama":
            return os.environ.get(self.provider.key_env, "ollama")
        return os.environ.get(self.provider.key_env, "")

    def why_unavailable(self) -> str:
        return (
            ""
            if self.key
            else f"{self.name} needs {self.provider.key_env} in the environment"
        )


def parse_choice(spec: str, base_url: str = "") -> ModelChoice:
    """``"openai:gpt-5"`` -> a :class:`ModelChoice`.

    A bare model name is an error rather than a guess: which vendor is serving
    a model is exactly the thing §6a cares about, and inferring it from the
    model string would make the rule depend on naming conventions.
    """
    provider_name, _, model = spec.partition(":")
    if not model:
        raise ValueError(
            f"{spec!r} is not 'provider:model'. Known providers: "
            f"{', '.join(sorted(PROVIDERS))}"
        )
    if provider_name not in PROVIDERS:
        raise ValueError(
            f"unknown provider {provider_name!r}; known: {', '.join(sorted(PROVIDERS))}"
        )
    provider = PROVIDERS[provider_name]
    return ModelChoice(provider, model, base_url or provider.base_url)


def agent_choice() -> ModelChoice:
    return parse_choice(
        os.environ.get(AGENT_ENV) or DEFAULT_AGENT,
        os.environ.get(AGENT_BASE_URL_ENV, ""),
    )


def respondent_choice() -> ModelChoice:
    return parse_choice(
        os.environ.get(RESPONDENT_ENV) or DEFAULT_RESPONDENT,
        os.environ.get(RESPONDENT_BASE_URL_ENV, ""),
    )


class TextBackend(Protocol):
    """A chat model with a system prompt and **no tools** — what a respondent
    is. The agent needs tool calling and keeps its own client (`_agent.py`)."""

    name: str

    def complete(
        self, *, system: str, messages: list[dict], max_tokens: int
    ) -> str: ...


@dataclass
class OpenAICompatText:
    """Any OpenAI-compatible endpoint: OpenAI, Gemini, DeepSeek, Ollama, vLLM."""

    choice: ModelChoice
    temperature: float = 0.0

    @property
    def name(self) -> str:
        return self.choice.name

    def complete(self, *, system: str, messages: list[dict], max_tokens: int) -> str:
        from openai import OpenAI

        kwargs = {"api_key": self.choice.key}
        if self.choice.base_url:
            kwargs["base_url"] = self.choice.base_url
        completion = OpenAI(**kwargs).chat.completions.create(
            model=self.choice.model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return (completion.choices[0].message.content or "").strip()


@dataclass
class AnthropicText:
    """The Anthropic API, where the system prompt is its own argument."""

    choice: ModelChoice
    temperature: float = 0.0

    @property
    def name(self) -> str:
        return self.choice.name

    def complete(self, *, system: str, messages: list[dict], max_tokens: int) -> str:
        import anthropic

        kwargs = {"api_key": self.choice.key}
        if self.choice.base_url:
            kwargs["base_url"] = self.choice.base_url
        response = anthropic.Anthropic(**kwargs).messages.create(
            model=self.choice.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=system,
            messages=messages,
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()


def text_backend(choice: ModelChoice) -> TextBackend:
    """The backend for *choice*. One branch, on the SDK it speaks."""
    if choice.provider.sdk == "anthropic":
        return AnthropicText(choice)
    return OpenAICompatText(choice)
