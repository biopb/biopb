"""Which model sits on each side of the conversation, and how to reach it.

Both sides of §5 are chat models, so provider selection is one concern and
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

**What §5a constrains is the agent, and only the agent**: it must not come from
the family that wrote the skill, because it could then pass by recognising its
own prose. The respondent holds a persona and answers from a fact table, which
is not a job where family contamination helps, so nothing here restricts it.
The harness records both names in the trace and lets a human judge the pairing.

A provider is a `(sdk, base_url, key_env)` triple, and most of them are the
OpenAI-compatible API with a different address in front — which is why one
implementation reaches nearly all of them.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

#: Where a key may live besides the environment. A non-interactive shell does
#: not read `~/.bashrc` past its `case $- in *i*` guard, so an export added at
#: the bottom of it is invisible to anything run from a tool or a CI step --
#: which is a confusing way to spend an afternoon. A file is simply read.
#:
#: `.env` at the repo root is already gitignored. Order: an explicit
#: `BIOPB_SKILL_ENV_FILE`, then the repo root, then the biopb config dir.
ENV_FILE_ENV = "BIOPB_SKILL_ENV_FILE"
CONFIG_ENV_FILE = Path.home() / ".config" / "biopb" / "skill-harness.env"
HOME_ENV_FILE = Path.home() / ".env"

#: Which side is being configured. Separate variables so agent and respondent
#: can sit on different providers, or the same one at different addresses.
AGENT_ENV = "BIOPB_SKILL_AGENT"
RESPONDENT_ENV = "BIOPB_SKILL_RESPONDENT"
AGENT_BASE_URL_ENV = "BIOPB_SKILL_AGENT_BASE_URL"
RESPONDENT_BASE_URL_ENV = "BIOPB_SKILL_RESPONDENT_BASE_URL"

DEFAULT_AGENT = "openai:gpt-5"
DEFAULT_RESPONDENT = "anthropic:claude-sonnet-5"

#: Model-name substrings that mean "this came from the family that wrote these
#: skills", checked in addition to the provider. Needed because `provider:`
#: names a wire protocol as often as a vendor: behind an OpenAI-compatible
#: gateway a Claude model is spelled `openai:claude-...`, and §5a's whole point
#: is to notice that. Not a security boundary -- a gateway can name a model
#: anything -- but it catches the mistake somebody actually makes.
AUTHORING_FAMILY_MARKERS = ("claude", "sonnet", "opus", "haiku")


def _repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def env_file() -> Path | None:
    """The dotenv this run would read, if any. Most specific first."""
    explicit = os.environ.get(ENV_FILE_ENV, "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    root = _repo_root()
    candidates = [
        *([root / ".env"] if root else ()),
        CONFIG_ENV_FILE,
        HOME_ENV_FILE,
    ]
    return next((p for p in candidates if p.is_file()), None)


def read_env_file(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines, minus comments, blanks and surrounding quotes.

    Deliberately not a dotenv library: this needs to parse five lines, and a
    dependency whose only job is that would be carried by every developer
    running the ordinary suite.
    """
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


@functools.cache
def _dotenv() -> dict[str, str]:
    path = env_file()
    if path is None:
        return {}
    try:
        return read_env_file(path)
    except OSError:
        # Unreadable is the same as absent here; a missing key is reported by
        # name further down, which is a better message than a stat error.
        return {}


def reload_env_file() -> None:
    """Forget the cached dotenv. For tests, and for a file edited mid-session."""
    _dotenv.cache_clear()


def setting(name: str, default: str = "") -> str:
    """*name* from the environment, else the dotenv, else *default*.

    **The environment wins**, so an explicit export always overrides a file
    somebody forgot about — the direction that makes a surprising result easier
    to explain, not harder.
    """
    return os.environ.get(name) or _dotenv().get(name) or default


@dataclass(frozen=True)
class Provider:
    """How to reach one vendor. ``sdk`` is the only real branch."""

    name: str
    sdk: str  # "openai" | "anthropic"
    key_env: str
    base_url: str = ""
    #: True when this vendor authored the skills in this repo. §5a reads it.
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
            return setting(self.provider.key_env, "ollama")
        return setting(self.provider.key_env)

    @property
    def from_authoring_family(self) -> bool:
        """Whether this model comes from the family that wrote these skills.

        Two ways to be one, because `provider:` names a **wire protocol** as
        often as a vendor. Behind a gateway — an OpenAI-compatible endpoint
        serving many vendors' models — `openai:` says nothing about who trained
        what is on the other end, so a gateway-served Claude spells
        ``openai:claude-sonnet-5`` and would sail through a check that only
        read the provider. That is precisely the case §5a exists to catch, so
        the model name is read too.

        A name check is a heuristic and cannot be otherwise: a gateway may call
        a model anything at all. It catches the honest mistake, not a
        determined one, and `interaction_notes` says which of the two it is.
        """
        if self.provider.wrote_these_skills:
            return True
        low = self.model.casefold()
        return any(marker in low for marker in AUTHORING_FAMILY_MARKERS)

    @property
    def family_is_certain(self) -> bool:
        """False when a gateway makes the family unknowable from config alone.

        A run behind one is not invalid — it is unverified, which is a
        different thing and belongs in the trace rather than in an exception.
        """
        return not self.base_url or self.provider.base_url == self.base_url

    def why_unavailable(self) -> str:
        if self.key:
            return ""
        return (
            f"{self.name} needs {self.provider.key_env}: export it, or put it "
            f"in a .env at the repo root (gitignored). Note that an export at "
            f"the bottom of ~/.bashrc will NOT reach a non-interactive shell — "
            f"it returns early at the `case $- in *i*` guard."
        )


def parse_choice(spec: str, base_url: str = "") -> ModelChoice:
    """``"openai:gpt-5"`` -> a :class:`ModelChoice`.

    A bare model name is an error rather than a guess: which vendor is serving
    a model is exactly the thing §5a cares about, and inferring it from the
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


#: The conventional name, which the OpenAI SDK reads by itself. Honoured as a
#: fallback for both sides: anyone pointing a gateway at this harness reaches
#: for it first, and quietly ignoring it sends a gateway key to
#: api.openai.com, which fails with an authentication error that explains
#: nothing. The per-side variables still win, which is what keeps "one model on
#: each of two endpoints" sayable.
SHARED_BASE_URL_ENV = "OPENAI_BASE_URL"


def agent_choice() -> ModelChoice:
    return parse_choice(
        setting(AGENT_ENV, DEFAULT_AGENT),
        setting(AGENT_BASE_URL_ENV) or setting(SHARED_BASE_URL_ENV),
    )


def respondent_choice() -> ModelChoice:
    return parse_choice(
        setting(RESPONDENT_ENV, DEFAULT_RESPONDENT),
        setting(RESPONDENT_BASE_URL_ENV) or setting(SHARED_BASE_URL_ENV),
    )


class EmptyCompletion(RuntimeError):
    """A provider returned no usable text, and said so.

    **This exists so that a budget failure cannot be mistaken for a reply.** A
    reasoning model bills its reasoning against the same `max_tokens` as its
    answer, so a budget that would comfortably hold the answer can be consumed
    entirely before the answer starts: the provider returns
    `finish_reason="length"` with empty content, which is a *truncation*, not a
    short response. An earlier version returned `""` here and let the caller
    read it as a hand-off, which ended runs at the first question and scored
    them against the agent.

    ``reason`` is the provider's own word for it (`length`, `max_tokens`, or
    `no-text` when it gave none), so a trace says which happened.
    """

    def __init__(self, model: str, reason: str, max_tokens: int) -> None:
        super().__init__(
            f"{model} returned no text (reason={reason!r}, max_tokens={max_tokens}). "
            "A reasoning model spends its reasoning from this same budget; raise "
            "max_tokens rather than reading the empty result as an answer."
        )
        self.model = model
        self.reason = reason
        self.max_tokens = max_tokens


#: Keys a provider may return on an assistant message and require back on the
#: next request. Tried in order; the first one *present* is carried under its
#: own name, because a provider spelling it `reasoning` will not accept
#: `reasoning_content`. Shared with `_agent.py`: both sides of the conversation
#: hold history, so both have to echo, and one definition keeps them honest.
ECHOED_FIELDS = ("reasoning_content", "reasoning")


def echoed_fields(message: Any) -> dict:
    """What *message* carries that has to be sent back with it."""
    for key in ECHOED_FIELDS:
        value = getattr(message, key, None)
        if value is not None:
            return {key: value}
    return {}


@dataclass(frozen=True)
class Reply:
    """A completion, and whatever the provider wants back with it.

    **A reasoning turn is not just its text.** The text is what a caller
    wants; `provider_fields` is what the *next* request needs in order to be
    accepted at all, and dropping it fails a turn later than its cause.
    """

    text: str
    provider_fields: dict = field(default_factory=dict)


class TextBackend(Protocol):
    """A chat model with a system prompt and **no tools** — what a respondent
    is. The agent needs tool calling and keeps its own client (`_agent.py`).

    Raises :class:`EmptyCompletion` rather than returning an empty string: the
    caller cannot tell the two apart, and one of them is a bug.
    """

    name: str

    def complete(
        self, *, system: str, messages: list[dict], max_tokens: int
    ) -> Reply: ...


def reachable(backend: TextBackend) -> str:
    """Why *backend* cannot be used at all, or ``""``.

    **A model name that the endpoint does not serve is an environment fault,
    and it should cost one skip rather than four dead arms.** `why_unavailable`
    only knows whether a *key* is present; it cannot know that
    `BIOPB_SKILL_AGENT` names something this gateway retired, or that a shell
    export is quietly beating the dotenv — and both of those spend every arm
    before saying so, each one failing identically for a reason that has
    nothing to do with the skill.

    One negligible completion answers it. An :class:`EmptyCompletion` counts as
    *reachable*: the model answered, and having spent its budget on reasoning
    is a different problem from not existing.
    """
    try:
        backend.complete(
            system="Reply with the single word: ok",
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=16,
        )
    except EmptyCompletion:
        return ""
    except Exception as exc:  # noqa: BLE001 - any failure here is disqualifying
        return f"{type(exc).__name__}: {exc}"
    return ""


@dataclass
class OpenAICompatText:
    """Any OpenAI-compatible endpoint: OpenAI, Gemini, DeepSeek, Ollama, vLLM."""

    choice: ModelChoice
    temperature: float = 0.0

    @property
    def name(self) -> str:
        return self.choice.name

    def complete(self, *, system: str, messages: list[dict], max_tokens: int) -> Reply:
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
        choice = completion.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise EmptyCompletion(
                self.name, choice.finish_reason or "no-text", max_tokens
            )
        return Reply(text, echoed_fields(choice.message))


@dataclass
class AnthropicText:
    """The Anthropic API, where the system prompt is its own argument."""

    choice: ModelChoice
    temperature: float = 0.0

    @property
    def name(self) -> str:
        return self.choice.name

    def complete(self, *, system: str, messages: list[dict], max_tokens: int) -> Reply:
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
        # Thinking blocks are dropped here but *are* billed against max_tokens,
        # and on the current models thinking is on unless it is turned off — so
        # a budget spent thinking arrives as content with no text block in it.
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not text:
            raise EmptyCompletion(
                self.name, response.stop_reason or "no-text", max_tokens
            )
        # Anthropic needs no echo here: thinking is a content block and this
        # backend never asks for it. `Reply` stays the shape either way.
        return Reply(text)


def text_backend(choice: ModelChoice) -> TextBackend:
    """The backend for *choice*. One branch, on the SDK it speaks."""
    if choice.provider.sdk == "anthropic":
        return AnthropicText(choice)
    return OpenAICompatText(choice)
