"""Tests for the chat provider adapter (_model.py).

No network: the transport is stubbed at ``httpx.AsyncClient``. What is actually
under test is the key's provenance and the readiness contract — the two things
that decide whether a user's key ends up somewhere it should not be, and whether
a misconfigured install fails before or after it has run code in their kernel.
"""

import asyncio

import httpx
import pytest
from biopb._credentials import credential_file, remove_credential, write_credential

from biopb_mcp._config import McpConfig
from biopb_mcp.mcp import _model


@pytest.fixture
def config():
    # A dict, as the launcher threads it -- see test_mcp_chat_api.chat_config.
    cfg = {
        "observe": {"enabled": True, "chat_enabled": True},
        "chat": {"model": "test-model"},
    }
    return cfg


@pytest.fixture
def state_home(tmp_path, monkeypatch):
    """Isolate the credential directory (CI sets XDG vars; see the state dir)."""
    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("BIOPB_CHAT_API_KEY", raising=False)
    yield tmp_path
    remove_credential(_model.KEY_NAME)


class TestKeyProvenance:
    def test_read_from_the_owner_only_credential_file(self, config, state_home):
        path = write_credential("sk-from-file", _model.KEY_NAME)
        assert _model.api_key(config) == "sk-from-file"
        # Its own file, not the data plane's: one leaking must not be the other.
        assert path != credential_file()
        assert path.name == "chat-provider.token"

    def test_the_env_var_overrides_for_development(
        self, config, state_home, monkeypatch
    ):
        write_credential("sk-from-file", _model.KEY_NAME)
        monkeypatch.setenv("BIOPB_CHAT_API_KEY", "sk-from-env")
        assert _model.api_key(config) == "sk-from-env"

    def test_the_key_is_never_written_into_the_config(self, config, state_home):
        # mcp-config.json is served whole by the control's GET /api/mcp_config so
        # the admin page can edit it. A key in it would be rendered in a browser.
        import dataclasses

        write_credential("sk-secret", _model.KEY_NAME)
        rendered = str(config)
        assert "sk-secret" not in rendered
        assert not any(f.name == "key" for f in dataclasses.fields(McpConfig().chat)), (
            "a bare `key` field would be the exact mistake this avoids"
        )


class TestReadiness:
    def test_each_missing_piece_says_which(self, config, state_home):
        # Only the provider halves: whether chat is offered at all is
        # observe.chat_enabled, and it decides whether these routes exist, so
        # nothing that reaches check_ready can be switched off.
        config["chat"]["model"] = ""
        with pytest.raises(_model.ChatNotConfigured, match="model"):
            _model.check_ready(config)

        config["chat"]["model"] = "test-model"
        with pytest.raises(_model.ChatNotConfigured) as exc:
            _model.check_ready(config)
        # Actionable: the path to write, and the env var, not just "no key".
        assert "chat-provider.token" in str(exc.value)
        assert "BIOPB_CHAT_API_KEY" in str(exc.value)

    def test_ready_when_all_three_are_present(self, config, state_home):
        write_credential("sk-x", _model.KEY_NAME)
        _model.check_ready(config)  # does not raise

    def test_no_default_model(self):
        # A default would bill the user for a model they never chose, and would
        # silently be the wrong one for whatever gateway they pointed at.
        assert McpConfig().chat.model == ""
        # The pane is offered by default; the model is what stays unset, so a
        # default install serves an inert pane rather than billing anyone.
        assert McpConfig().observe.chat_enabled is True


class _FakeClient:
    """Stand-in for ``httpx.AsyncClient`` that records one POST."""

    calls = []
    reply = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        type(self).calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": self.kwargs}
        )
        return type(self).reply

    async def get(self, url, headers=None):
        type(self).gets.append({"url": url, "headers": headers})
        return type(self).reply


def _response(status=200, payload=None, text=""):
    return httpx.Response(
        status_code=status,
        json=payload if payload is not None else None,
        text=text or None,
        request=httpx.Request("POST", "http://x/chat/completions"),
    )


class TestCall:
    @pytest.fixture(autouse=True)
    def stub(self, monkeypatch):
        _FakeClient.calls = []
        _FakeClient.gets = []
        monkeypatch.setattr(_model.httpx, "AsyncClient", _FakeClient)
        yield

    def test_the_request_shape(self, config, state_home):
        write_credential("sk-x", _model.KEY_NAME)
        _FakeClient.reply = _response(
            payload={"choices": [{"message": {"content": "hi"}}]}
        )
        out = asyncio.run(_model.make_model(config)([{"role": "user"}], [{"t": 1}]))
        assert out == {"content": "hi"}

        (call,) = _FakeClient.calls
        assert call["url"] == "https://api.openai.com/v1/chat/completions"
        assert call["headers"]["Authorization"] == "Bearer sk-x"
        assert call["json"]["model"] == "test-model"
        # The loop ends a turn when no tool_calls come back, so the model has to
        # stay free to answer rather than being forced to call something.
        assert call["json"]["tool_choice"] == "auto"

    def test_a_gateways_own_headers_ride_along(self, config, state_home):
        # A gateway in front of the OpenAI shape wants something the API never
        # defined; the endpoint policy decides what, and this is the wiring.
        from biopb_mcp.mcp import _chat

        write_credential("sk-x", _model.KEY_NAME)
        config["chat"]["base_url"] = "https://opencode.ai/zen/go/v1"
        _FakeClient.reply = _response(payload={"choices": [{"message": {}}]})
        asyncio.run(_model.make_model(config)([], []))

        (call,) = _FakeClient.calls
        assert call["headers"]["x-opencode-session"] == _chat.session_id()
        assert call["headers"]["Authorization"] == "Bearer sk-x"

    def test_the_session_header_follows_the_conversation_not_the_process(
        self, config, state_home
    ):
        # It names the conversation, so a new thread is a new id -- otherwise a
        # gateway attributes this thread's traffic to the one before it.
        from biopb_mcp.mcp import _chat

        write_credential("sk-x", _model.KEY_NAME)
        config["chat"]["base_url"] = "https://opencode.ai/zen/go/v1"
        _FakeClient.reply = _response(payload={"choices": [{"message": {}}]})
        model = _model.make_model(config)
        asyncio.run(model([], []))
        _chat.reset()
        asyncio.run(model([], []))

        first, second = (c["headers"]["x-opencode-session"] for c in _FakeClient.calls)
        assert first != second
        assert second == _chat.session_id()

    def test_the_catalogue_request_is_identified_too(self, config, state_home):
        # It goes to the same gateway, and after the deadline a header-less GET
        # is a 4xx that list_models turns into an empty picker.
        write_credential("sk-x", _model.KEY_NAME)
        config["chat"]["base_url"] = "https://opencode.ai/zen/go/v1"
        _FakeClient.reply = _response(payload={"data": [{"id": "m-1"}]})
        assert asyncio.run(_model.list_models(config)) == [
            {"value": "m-1", "name": "m-1"}
        ]
        (call,) = _FakeClient.gets
        assert call["headers"]["x-opencode-session"]

    def test_a_replaced_key_takes_effect_on_the_next_turn(self, config, state_home):
        # The file is how a user *sets* their key; a session that must be
        # restarted to notice is a support question waiting to happen.
        write_credential("sk-old", _model.KEY_NAME)
        model = _model.make_model(config)
        _FakeClient.reply = _response(payload={"choices": [{"message": {}}]})
        asyncio.run(model([], []))
        write_credential("sk-new", _model.KEY_NAME)
        asyncio.run(model([], []))
        assert [c["headers"]["Authorization"] for c in _FakeClient.calls] == [
            "Bearer sk-old",
            "Bearer sk-new",
        ]

    def test_a_provider_error_carries_the_providers_own_words(self, config, state_home):
        # A 400 here is usually a payload the model rejected -- a schema it
        # dislikes, a context overflow -- and the body is what says which.
        write_credential("sk-x", _model.KEY_NAME)
        _FakeClient.reply = _response(status=400, text="context_length_exceeded")
        with pytest.raises(RuntimeError, match="context_length_exceeded"):
            asyncio.run(_model.make_model(config)([], []))

    def test_an_image_refusal_is_told_apart_from_a_400_it_cannot_act_on(
        self, config, state_home
    ):
        # The loop can do something about this one -- withdraw the images and
        # try again -- and nothing about the rest, so it has its own type.
        write_credential("sk-x", _model.KEY_NAME)
        _FakeClient.reply = _response(
            status=400, text="this model does not support image input"
        )
        payload = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "(image returned by take_screenshot)"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,x"},
                    },
                ],
            }
        ]
        with pytest.raises(_model.VisionUnsupported, match="rejected an image"):
            asyncio.run(_model.make_model(config)(payload, []))

    def test_the_same_words_without_an_image_are_an_ordinary_error(
        self, config, state_home
    ):
        # Matched only against the answer to a request that carried one: a
        # provider saying "image" about a payload with no image in it is talking
        # about something else, and withdrawing images would not fix it.
        write_credential("sk-x", _model.KEY_NAME)
        _FakeClient.reply = _response(status=400, text="unknown tool: take_image")
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(
                _model.make_model(config)([{"role": "user", "content": "hi"}], [])
            )
        assert not isinstance(caught.value, _model.VisionUnsupported)

    def test_an_empty_choices_list_is_an_error_not_an_empty_answer(
        self, config, state_home
    ):
        write_credential("sk-x", _model.KEY_NAME)
        _FakeClient.reply = _response(payload={"choices": []})
        with pytest.raises(RuntimeError, match="no choices"):
            asyncio.run(_model.make_model(config)([], []))
