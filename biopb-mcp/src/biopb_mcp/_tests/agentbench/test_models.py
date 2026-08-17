"""Provider selection: hermetic, and it makes §5a checkable instead of assumed.

No network and no key — every one of these is about resolution, which is the
part that decides *what* a run measured. Runs with the ordinary suite.
"""

from __future__ import annotations

import sys
import types

import pytest

from ._models import (
    AGENT_BASE_URL_ENV,
    AGENT_ENV,
    DEFAULT_AGENT,
    DEFAULT_RESPONDENT,
    ENV_FILE_ENV,
    PROVIDERS,
    RESPONDENT_BASE_URL_ENV,
    RESPONDENT_ENV,
    SHARED_BASE_URL_ENV,
    AnthropicText,
    EmptyCompletion,
    OpenAICompatText,
    agent_choice,
    parse_choice,
    reachable,
    read_env_file,
    reload_env_file,
    respondent_choice,
    text_backend,
)


@pytest.fixture
def no_dotenv(tmp_path, monkeypatch):
    """Point the loader at nothing, so a developer's real `.env` cannot decide
    the result of a test. Restores the cache on the way out — it is process-wide
    and a stale one would leak into whatever ran next.

    **The real environment has to go too**, not just the file. `setting()` ranks
    the environment above the dotenv on purpose, so a developer with
    `BIOPB_AGENT` exported from their shell profile — the ordinary way to
    run this layer — silently overrode the file each of these tests writes, and
    the assertions read a machine's configuration instead of the fixture's. CI
    exports none of them, so this passed everywhere except where it mattered.
    """
    for name in (
        AGENT_ENV,
        RESPONDENT_ENV,
        AGENT_BASE_URL_ENV,
        RESPONDENT_BASE_URL_ENV,
        SHARED_BASE_URL_ENV,
        *(provider.key_env for provider in PROVIDERS.values()),
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ENV_FILE_ENV, str(tmp_path / "absent.env"))
    reload_env_file()
    yield monkeypatch
    monkeypatch.undo()
    reload_env_file()


def test_a_bare_model_name_is_refused():
    """Which vendor serves a model is exactly what §5a cares about, so it is
    never inferred from the model string — that would make the rule depend on
    naming conventions that vendors change at will."""
    with pytest.raises(ValueError, match="provider:model"):
        parse_choice("gpt-5")


def test_an_unknown_provider_says_what_is_known():
    with pytest.raises(ValueError, match="unknown provider"):
        parse_choice("acme:something")


@pytest.mark.parametrize("name", sorted(PROVIDERS))
def test_every_provider_resolves_to_a_backend(name):
    """The table is the whole configuration surface, so nothing in it may be
    half-wired. A provider that parses but has no backend would fail only on a
    paid run."""
    choice = parse_choice(f"{name}:some-model")
    backend = text_backend(choice)
    assert backend.name == f"{name}:some-model"
    expected = AnthropicText if choice.provider.sdk == "anthropic" else OpenAICompatText
    assert isinstance(backend, expected)


def test_the_two_sides_are_configured_independently(monkeypatch):
    """The point of the refactor. Agent and respondent must be able to sit on
    different vendors -- or on the same compatible API at two addresses, which
    a single shared base-URL variable made impossible to express."""
    monkeypatch.setenv(AGENT_ENV, "deepseek:deepseek-chat")
    monkeypatch.setenv(RESPONDENT_ENV, "ollama:qwen3")
    monkeypatch.setenv("BIOPB_AGENT_BASE_URL", "https://agent.example/v1")
    monkeypatch.setenv("BIOPB_RESPONDENT_BASE_URL", "http://localhost:11434/v1")

    agent, respondent = agent_choice(), respondent_choice()

    assert agent.provider.name == "deepseek"
    assert respondent.provider.name == "ollama"
    assert agent.base_url != respondent.base_url


def test_a_base_url_override_beats_the_provider_default(monkeypatch):
    monkeypatch.setenv(RESPONDENT_ENV, "openai:gpt-4o-mini")
    monkeypatch.setenv("BIOPB_RESPONDENT_BASE_URL", "http://proxy.internal/v1")
    assert respondent_choice().base_url == "http://proxy.internal/v1"


def test_a_missing_key_is_reported_by_name_not_by_traceback(no_dotenv):
    """A run that cannot start should say which variable to set, and must not
    be the thing that discovers it three tool calls in."""
    no_dotenv.delenv("OPENAI_API_KEY", raising=False)
    why = parse_choice("openai:gpt-5").why_unavailable()
    assert "OPENAI_API_KEY" in why
    # And it names the trap, because this one cost an afternoon: an export at
    # the bottom of ~/.bashrc never reaches a non-interactive shell.
    assert ".bashrc" in why and ".env" in why

    no_dotenv.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    assert parse_choice("openai:gpt-5").why_unavailable() == ""


# --- the dotenv ------------------------------------------------------------


def test_a_key_can_come_from_a_file(tmp_path, no_dotenv):
    """The whole point: a file is read by any shell, interactive or not."""
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-from-a-file\n")
    no_dotenv.delenv("OPENAI_API_KEY", raising=False)
    no_dotenv.setenv(ENV_FILE_ENV, str(env))
    reload_env_file()

    assert parse_choice("openai:gpt-5").why_unavailable() == ""


def test_the_environment_beats_the_file(tmp_path, no_dotenv):
    """An explicit export must override a file somebody forgot about. The
    other direction makes a surprising result harder to explain, not easier."""
    env = tmp_path / ".env"
    env.write_text("BIOPB_AGENT=deepseek:deepseek-chat\n")
    no_dotenv.setenv(ENV_FILE_ENV, str(env))
    reload_env_file()
    assert agent_choice().provider.name == "deepseek"

    no_dotenv.setenv(AGENT_ENV, "gemini:gemini-2.5-pro")
    assert agent_choice().provider.name == "gemini"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("KEY=plain", "plain"),
        ("export KEY=with-export-prefix", "with-export-prefix"),
        ('KEY="double quoted"', "double quoted"),
        ("KEY='single quoted'", "single quoted"),
        ("KEY = spaced ", "spaced"),
        ("KEY=has=equals=inside", "has=equals=inside"),
    ],
)
def test_the_parser_handles_what_people_actually_write(tmp_path, line, expected):
    """Including `export ` prefixes, because the natural move is to copy the
    line straight out of a shell profile."""
    path = tmp_path / ".env"
    path.write_text(f"# a comment\n\n{line}\n")
    assert read_env_file(path) == {"KEY": expected}


def test_a_line_without_an_equals_is_skipped(tmp_path):
    path = tmp_path / ".env"
    path.write_text("NOT_A_SETTING\nKEY=value\n")
    assert read_env_file(path) == {"KEY": "value"}


def test_no_file_anywhere_is_not_an_error(tmp_path, no_dotenv):
    no_dotenv.setenv(ENV_FILE_ENV, str(tmp_path / "nope.env"))
    reload_env_file()
    assert parse_choice("openai:gpt-5").why_unavailable() != ""


def test_a_local_model_needs_no_key():
    """Rehearsing a run against Ollama should not require inventing a secret."""
    assert parse_choice("ollama:qwen3").why_unavailable() == ""


# --- §5a, as a check rather than a comment ---------------------------------


def test_exactly_one_provider_is_marked_as_having_written_these_skills():
    """§5a is a rule about a fact, and the fact lives in the table. If these
    skills are ever co-authored with another vendor, this is the line that has
    to change before the rule means anything again."""
    authors = [p.name for p in PROVIDERS.values() if p.wrote_these_skills]
    assert authors == ["anthropic"]


def test_the_default_agent_is_not_from_the_authoring_family():
    """The rule, applied to what a run does when nobody configured anything.
    A default that violated §5a would be the easiest way to get a green suite
    that measured recognition rather than reading."""
    choice = parse_choice(DEFAULT_AGENT)
    assert not choice.provider.wrote_these_skills, (
        f"the default agent {choice.name} comes from the family that wrote "
        "these skills, so it could pass by recognising its own prose (§5a)"
    )


def test_the_respondent_default_is_deliberately_unconstrained():
    """The other half of §5a, and it is a real asymmetry rather than an
    oversight: holding a persona and answering from a fact table is not a job
    where having written the skills helps, so the authoring family is a fine
    respondent -- and is in fact the default."""
    assert parse_choice(DEFAULT_RESPONDENT).provider.wrote_these_skills


def test_an_agent_configured_against_the_rule_is_still_resolvable(monkeypatch):
    """Resolution does not enforce §5a -- the scoring pass does, where it can
    be skipped or overridden deliberately. Refusing here would make it
    impossible to reproduce someone else's contaminated run in order to show
    that it was contaminated."""
    monkeypatch.setenv(AGENT_ENV, "anthropic:claude-sonnet-5")
    assert agent_choice().provider.wrote_these_skills


# --- empty completions -----------------------------------------------------
#
# The provider is faked, not the boundary: what is under test is that
# `complete()` refuses to hand back an empty string, and that only runs if a
# provider is there to return one. The SDKs are imported inside the method, so
# a module in `sys.modules` is enough and no key or network is involved.


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason):
        self.message = _Message(content)
        self.finish_reason = finish_reason


def _fake_openai(monkeypatch, content, finish_reason):
    module = types.ModuleType("openai")

    class _Completions:
        def create(self, **kwargs):
            return types.SimpleNamespace(choices=[_Choice(content, finish_reason)])

    class _OpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=_Completions())

    module.OpenAI = _OpenAI
    monkeypatch.setitem(sys.modules, "openai", module)


def _fake_anthropic(monkeypatch, blocks, stop_reason):
    module = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs):
            return types.SimpleNamespace(content=blocks, stop_reason=stop_reason)

    class _Anthropic:
        def __init__(self, **kwargs):
            self.messages = _Messages()

    module.Anthropic = _Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", module)


def _block(kind, text=""):
    return types.SimpleNamespace(type=kind, text=text)


def test_a_truncated_openai_reply_raises_instead_of_looking_like_an_answer(
    monkeypatch,
):
    """`finish_reason="length"` with empty content is a *truncation*, not a
    short reply. Returning `""` here is what let a budget failure be read as
    the agent signing off."""
    _fake_openai(monkeypatch, content="", finish_reason="length")
    backend = OpenAICompatText(parse_choice("openai:some-model"))

    with pytest.raises(EmptyCompletion) as caught:
        backend.complete(system="s", messages=[], max_tokens=300)

    assert caught.value.reason == "length"
    assert caught.value.max_tokens == 300
    assert "max_tokens" in str(caught.value), "the message has to name the lever"


def test_a_thinking_only_anthropic_reply_raises(monkeypatch):
    """Thinking blocks are dropped here but billed against `max_tokens`, and
    on the current models thinking is on unless turned off — so the budget can
    be spent on content that has no text block in it at all."""
    _fake_anthropic(
        monkeypatch, blocks=[_block("thinking", "")], stop_reason="max_tokens"
    )
    backend = AnthropicText(parse_choice("anthropic:some-model"))

    with pytest.raises(EmptyCompletion) as caught:
        backend.complete(system="s", messages=[], max_tokens=300)

    assert caught.value.reason == "max_tokens"


@pytest.mark.parametrize("reason", ["stop", "end_turn"])
def test_a_real_answer_still_comes_back_as_text(monkeypatch, reason):
    """The guard has to stay narrow: a backend that raised on ordinary replies
    would be a worse failure than the one it replaces."""
    _fake_openai(monkeypatch, content="  Channel 1.  ", finish_reason=reason)
    assert (
        OpenAICompatText(parse_choice("openai:m"))
        .complete(system="s", messages=[], max_tokens=300)
        .text
        == "Channel 1."
    )

    _fake_anthropic(
        monkeypatch,
        blocks=[_block("thinking", ""), _block("text", "Channel 1.")],
        stop_reason=reason,
    )
    assert (
        AnthropicText(parse_choice("anthropic:m"))
        .complete(system="s", messages=[], max_tokens=300)
        .text
        == "Channel 1."
    )


# --- reachability ----------------------------------------------------------


def test_a_model_the_endpoint_does_not_serve_is_reported_not_spent(monkeypatch):
    """Four arms failing identically on a 401 is an environment fault wearing
    four skill-shaped rows. One negligible request answers it first."""
    module = types.ModuleType("openai")

    class _Boom:
        def create(self, **kwargs):
            raise RuntimeError("Model GLM-5.1 is not supported")

    class _OpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=_Boom())

    module.OpenAI = _OpenAI
    monkeypatch.setitem(sys.modules, "openai", module)

    why = reachable(OpenAICompatText(parse_choice("openai:GLM-5.1")))
    assert "GLM-5.1" in why and "RuntimeError" in why


def test_a_model_that_answers_is_reachable(monkeypatch):
    _fake_openai(monkeypatch, content="ok", finish_reason="stop")
    assert reachable(OpenAICompatText(parse_choice("openai:m"))) == ""


def test_spending_the_whole_budget_on_reasoning_is_not_unreachable(monkeypatch):
    """It answered. Having nothing left to say it with is a different fault,
    and skipping the run for it would hide the one this layer measures."""
    _fake_openai(monkeypatch, content="", finish_reason="length")
    assert reachable(OpenAICompatText(parse_choice("openai:m"))) == ""


def test_the_reply_carries_what_the_provider_wants_back(monkeypatch):
    """The respondent replays its own history on every later question, so a
    turn stripped of `reasoning_content` fails the *next* one — a turn after
    its cause, which is what made this read as intermittent."""
    module = types.ModuleType("openai")

    class _Msg:
        content = "Channel 1."
        reasoning_content = "weighing the two channels"

    class _Completions:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=_Msg(), finish_reason="stop")]
            )

    class _OpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=_Completions())

    module.OpenAI = _OpenAI
    monkeypatch.setitem(sys.modules, "openai", module)

    reply = OpenAICompatText(parse_choice("openai:m")).complete(
        system="s", messages=[], max_tokens=300
    )
    assert reply.text == "Channel 1."
    assert reply.provider_fields == {"reasoning_content": "weighing the two channels"}


def test_a_provider_with_nothing_to_echo_sends_nothing_extra(monkeypatch):
    _fake_openai(monkeypatch, content="ok", finish_reason="stop")
    assert (
        OpenAICompatText(parse_choice("openai:m"))
        .complete(system="s", messages=[], max_tokens=300)
        .provider_fields
        == {}
    )
