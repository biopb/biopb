"""Tests for the chat HTTP surface on the session child (_chat_api.py).

Driven through Starlette's TestClient over a standalone app built from the same
route table the MCP mount uses, so the handlers are exercised as handlers rather
than as functions.
"""

import asyncio
import json

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
        assert by_path["/chat/cancel"] == ["POST"]
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


class TestCancel:
    @staticmethod
    def _wait_for(predicate, timeout=2.0):
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_it_carries_the_json_guard_despite_having_no_body(self, client):
        # _json_route's rationale exempts body-less POSTs, since the content-type
        # rule is there for "the one route that submits code". Cancel takes it
        # anyway: a JSON content-type is one an HTML form cannot set, and it
        # costs a caller a header to stop a cross-site page from killing
        # someone's turn. A body-less POST is not a harmless one.
        assert (
            client.post(
                "/chat/cancel", headers={"content-type": "text/plain"}
            ).status_code
            == 400
        )

    def test_a_scheduled_turn_already_counts(self, client, monkeypatch):
        # busy() is the turn lock, taken when a turn *starts*. The task exists
        # from when it is *accepted*, one event-loop step earlier -- create_task
        # only schedules, and the 202 goes out before the coroutine runs a line.
        # Reading either signal alone leaves that step unguarded.
        async def scenario():
            async def idle():
                await asyncio.sleep(3600)

            task = asyncio.create_task(idle())
            monkeypatch.setattr(_chat_api, "_turn_task", task)
            try:
                assert _chat.busy() is False  # not started: the lock is free
                assert _chat_api._in_flight() is True  # ...but it is accepted
            finally:
                task.cancel()

        asyncio.run(scenario())

    def test_cancelling_before_the_turn_starts_says_so(self, client, monkeypatch):
        # Nothing of the turn ran, so nothing recorded it -- which is the right
        # thread, but a view waiting for a cancellation message would wait
        # forever. Hence `started`.
        async def scenario():
            async def idle():
                await asyncio.sleep(3600)

            task = asyncio.create_task(idle())
            monkeypatch.setattr(_chat_api, "_turn_task", task)
            try:
                reply = await _chat_api._chat_cancel(None)
                assert json.loads(bytes(reply.body)) == {
                    "cancelled": True,
                    "started": False,
                }
                assert _chat.history() == []
            finally:
                task.cancel()

        asyncio.run(scenario())

    def test_nothing_running_is_not_an_error(self, client):
        # A person clicking cancel on a turn that has just finished has made no
        # mistake; an error status would say they had.
        reply = client.post("/chat/cancel", json={})
        assert reply.status_code == 200
        assert reply.json()["cancelled"] is False

    def test_a_running_turn_stops_and_says_so(self, client, monkeypatch):
        started = asyncio.Event()

        async def hang(messages, tools):
            started.set()
            await asyncio.sleep(3600)

        monkeypatch.setattr(_model, "make_model", lambda cfg: hang)
        client.post("/chat/turn", json={"text": "hello"})
        assert self._wait_for(started.is_set)

        assert client.post("/chat/cancel", json={}).json()["cancelled"] is True
        # The thread carries the outcome, and the session frees up: a view is
        # polling these two, and both have to move or the cancel looks ignored.
        assert self._wait_for(lambda: any(m.get("cancelled") for m in _chat.history()))
        assert self._wait_for(lambda: not _chat.busy())
        assert client.get("/api/chat/status").json()["busy"] is False

    def test_it_does_not_reach_into_the_kernel(self, client, monkeypatch):
        # Deliberate, and the whole shape of this route: stopping the turn is
        # not stopping the user's code. An MCP client going away leaves its cell
        # running too; the chat pane must not be the one place where walking
        # away destroys work.
        started = asyncio.Event()
        touched = []

        async def hang(messages, tools):
            started.set()
            await asyncio.sleep(3600)

        monkeypatch.setattr(_model, "make_model", lambda cfg: hang)
        monkeypatch.setattr(
            _chat, "_job_call", lambda *a, **k: touched.append(a) or (None, {}, None)
        )
        client.post("/chat/turn", json={"text": "hello"})
        assert self._wait_for(started.is_set)
        client.post("/chat/cancel", json={})
        assert self._wait_for(lambda: not _chat.busy())
        assert touched == []


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
