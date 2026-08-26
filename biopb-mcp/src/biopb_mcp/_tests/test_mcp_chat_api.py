"""Tests for the chat HTTP surface on the session child (_chat_api.py).

Driven through Starlette's TestClient over a standalone app built from the same
route table the MCP mount uses, so the handlers are exercised as handlers rather
than as functions.
"""

import asyncio

import pytest
from biopb._credentials import remove_credential, write_credential
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from biopb_mcp._config import McpConfig
from biopb_mcp.mcp import _chat, _chat_api, _model


@pytest.fixture
def configured(tmp_path, monkeypatch):
    """A chat that is on, has a model, and has a key."""
    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("BIOPB_CHAT_API_KEY", raising=False)
    write_credential("sk-x", _model.KEY_NAME)
    cfg = McpConfig()
    cfg.observe.chat_enabled = True
    cfg.chat.model = "test-model"
    _chat_api.configure(cfg)
    _chat.reset()
    yield cfg
    _chat.reset()
    remove_credential(_model.KEY_NAME)


@pytest.fixture
def client(configured):
    app = Starlette(routes=[Route(p, h, methods=m) for p, m, h in _chat_api._ROUTES])
    # A loopback base_url because every route carries its own Host/Origin guard
    # (the kernel is RCE by design and FastMCP's transport security does not
    # cover custom routes); the default 'testserver' Host is correctly refused.
    with TestClient(app, base_url="http://127.0.0.1:8766") as c:
        yield c


class TestRootSplit:
    def test_reads_are_under_api_and_the_write_is_not(self):
        # The control proxies `api` always and `chat` only when loopback-bound,
        # and enforces that local root as POST-only. A history GET under `chat`
        # would be forwarded cross-site unchecked; a turn POST under `api` would
        # put an RCE back on the public origin. Both halves are pinned here
        # because the split is invisible from either side alone.
        by_path = {p: m for p, m, _ in _chat_api._ROUTES}
        assert by_path["/api/chat/history"] == ["GET"]
        assert by_path["/api/chat/status"] == ["GET"]
        assert by_path["/chat/turn"] == ["POST"]
        assert not any(
            p.startswith("/chat/") and m != ["POST"] for p, m, _ in _chat_api._ROUTES
        )


class TestStatus:
    def test_reports_ready_when_configured(self, client):
        body = client.get("/api/chat/status").json()
        assert body == {
            "enabled": True,
            "ready": True,
            "reason": None,
            "busy": False,
            "model": "test-model",
        }

    def test_reports_why_it_is_not_ready(self, client, configured):
        configured.chat.model = ""
        body = client.get("/api/chat/status").json()
        assert body["ready"] is False
        # The reason is carried so a view can render it once at the top of an
        # empty thread, instead of the user finding out by sending a message.
        assert "model" in body["reason"]
        assert body["model"] == ""


class TestHistory:
    def test_after_returns_only_what_the_caller_has_not_seen(self, client):
        first = _chat._append("user", "one")
        _chat._append("assistant", "two")
        body = client.get("/api/chat/history", params={"after": first["id"]}).json()
        assert [m["content"] for m in body["messages"]] == ["two"]

    def test_an_unknown_after_returns_everything(self, client):
        # A view that just loaded, or one that fell behind a reset: the whole
        # thread is the right answer, not an empty one.
        _chat._append("user", "one")
        body = client.get("/api/chat/history", params={"after": "m-999"}).json()
        assert [m["content"] for m in body["messages"]] == ["one"]


class TestTurn:
    def test_a_turn_is_accepted_not_awaited(self, client, monkeypatch):
        # A turn that runs a long cell would sit past the control proxy's 300s
        # per-read bound with no bytes in between to hold it open, so the POST
        # must return and the view must poll.
        started = asyncio.Event()

        async def slow(messages, tools):
            started.set()
            await asyncio.sleep(0.05)
            return {"content": "done"}

        monkeypatch.setattr(_model, "make_model", lambda cfg: slow)
        reply = client.post("/chat/turn", json={"text": "hello"})
        assert reply.status_code == 202
        assert reply.json() == {"accepted": True}

    def test_a_failed_turn_lands_in_the_thread(self, client, monkeypatch):
        # A background task that dies would otherwise leave the view polling a
        # conversation that simply stops growing -- which reads as a hang.
        async def boom(messages, tools):
            raise RuntimeError("provider exploded")

        monkeypatch.setattr(_model, "make_model", lambda cfg: boom)
        client.post("/chat/turn", json={"text": "hello"})
        for _ in range(200):
            if any(m.get("error") for m in _chat.history()):
                break
            import time

            time.sleep(0.01)
        errors = [m for m in _chat.history() if m.get("error")]
        assert errors and "provider exploded" in errors[0]["content"]

    def test_an_unconfigured_chat_refuses_before_taking_the_message(
        self, client, configured
    ):
        configured.chat.model = ""
        reply = client.post("/chat/turn", json={"text": "hello"})
        assert reply.status_code == 503
        assert "model" in reply.json()["error"]
        # Nothing was recorded: the user's message is not half-accepted.
        assert _chat.history() == []

    def test_a_busy_session_is_409_not_a_queue(self, client, monkeypatch):
        monkeypatch.setattr(_chat, "busy", lambda: True)
        reply = client.post("/chat/turn", json={"text": "hello"})
        assert reply.status_code == 409
        assert reply.json()["busy"] is True

    @pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}])
    def test_an_empty_message_is_rejected(self, client, payload):
        assert client.post("/chat/turn", json=payload).status_code == 400


def test_routes_are_not_mounted_when_chat_is_off():
    # Off drops the surface entirely rather than serving a refusing one, the
    # same shape the console's gate takes: "is there a way to submit here?" has
    # one answer rather than a status code to interpret.
    cfg = McpConfig()
    assert cfg.observe.chat_enabled is False
    assert _chat_api.configure(cfg) is False


def test_chat_follows_the_page_it_lives_on():
    # The pane is on the observe page, so chat routes without that page have
    # nothing that can reach them. Enforced rather than documented: the two
    # flags cannot be set to a combination that serves an unreachable surface.
    cfg = McpConfig()
    cfg.observe.chat_enabled = True
    cfg.observe.enabled = False
    assert _chat_api.configure(cfg) is False
