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
from biopb_mcp.mcp import _chat, _chat_api, _model, _observe

# The shape the launcher actually threads: `load_config()` returns a **dict**,
# and every consumer reads it with `get_setting`, which falls back to
# `DEFAULT_CONFIG` per key. Tests build the same partial dicts a real
# mcp-config.json produces -- a file with no `chat` section at all is the common
# case, and it was an `McpConfig` instance here that hid the bug: attribute access
# worked in every test and raised on the real dict in production.


def chat_config(**observe):
    """A config dict with chat on, plus any observe overrides."""
    return {"observe": {"enabled": True, "chat_enabled": True, **observe}}


@pytest.fixture
def configured(tmp_path, monkeypatch):
    """A chat that is on, has a model, and has a key."""
    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("BIOPB_CHAT_API_KEY", raising=False)
    write_credential("sk-x", _model.KEY_NAME)
    cfg = chat_config()
    cfg["chat"] = {"model": "test-model"}
    _chat_api.configure(cfg, agentless=True)
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
        configured["chat"]["model"] = ""
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
        configured["chat"]["model"] = ""
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
    cfg = {}
    assert McpConfig().observe.chat_enabled is False  # the default this relies on
    assert _chat_api.configure(cfg, agentless=True) is False


def test_chat_follows_the_page_it_lives_on():
    # The pane is on the observe page, so chat routes without that page have
    # nothing that can reach them. Enforced rather than documented: the two
    # flags cannot be set to a combination that serves an unreachable surface.
    cfg = chat_config(enabled=False)
    assert _chat_api.configure(cfg, agentless=True) is False


def test_a_harness_driven_session_gets_no_chat():
    # The loop is for users *without* an MCP harness. On a session an agent is
    # already driving, a second one is not a feature: only one writer can hold
    # the kernel claim, so the pane would answer questions and then refuse to
    # run anything -- correct, and not what anyone opening it expects.
    #
    # Config alone cannot express this. Both switches are on here, and the
    # surface is still withheld, because the deciding fact is how the session
    # was launched rather than how it was configured.
    cfg = chat_config()
    cfg["chat"] = {"model": "test-model"}
    assert _chat_api.configure(cfg, agentless=False) is False
    assert _chat_api.configure(cfg, agentless=True) is True


def test_chat_cannot_be_configured_on_by_accident():
    # `agentless` is required, not defaulted: either default is wrong for one
    # of the two callers, and the failure would be silent both ways -- chat on
    # every harness-driven session, or missing from the viewer it was built for.
    with pytest.raises(TypeError):
        _chat_api.configure(chat_config())


@pytest.fixture
def no_live(monkeypatch):
    """A transport with no cell running and an empty progress buffer."""
    monkeypatch.setattr(_chat_api, "_live_job", None)
    monkeypatch.setattr(_chat_api, "_live_text", "")
    monkeypatch.setattr(_chat_api, "_live_len", 0)
    monkeypatch.setattr(_chat, "_running_job_id", None)


def _running(monkeypatch, job_id):
    """Pretend the loop is polling *job_id* right now."""
    monkeypatch.setattr(_chat, "_running_job_id", job_id)


class TestPartialOutput:
    """The running cell's stdout, published without entering the conversation.

    ``_run_code`` has always streamed partial output to ``on_progress``; nothing
    passed one, so it was computed and dropped. A long cell left the thread
    silent for its whole duration, which is the one thing the promote window was
    given up to avoid.
    """

    def test_the_turn_is_given_somewhere_to_stream_to(self, configured, monkeypatch):
        # The whole defect in one line: a turn started without a sink discards
        # every byte the cell prints while it runs.
        seen = {}

        async def capture(text, model, on_progress=None):
            seen["on_progress"] = on_progress

        monkeypatch.setattr(_chat, "run_turn", capture)
        asyncio.run(_chat_api._run_turn("go"))
        assert seen["on_progress"] is _chat_api._note_progress

    def test_a_running_cell_publishes_what_it_has_printed(
        self, client, no_live, monkeypatch
    ):
        _running(monkeypatch, "job-1")
        _chat_api._note_progress("step 1\n")
        _chat_api._note_progress("step 2\n")
        body = client.get("/api/chat/history").json()
        assert body["partial"] == {
            "job_id": "job-1",
            "stdout": "step 1\nstep 2\n",
            "truncated": False,
            "stdout_len": 14,
        }

    def test_nothing_is_published_when_no_cell_is_running(self, client, no_live):
        assert client.get("/api/chat/history").json()["partial"] is None

    def test_output_stops_being_published_once_the_cell_ends(
        self, client, no_live, monkeypatch
    ):
        # The finished cell's output is in the thread as the tool's result.
        # Publishing both would show a reader the same output twice.
        _running(monkeypatch, "job-1")
        _chat_api._note_progress("done\n")
        _running(monkeypatch, None)
        assert client.get("/api/chat/history").json()["partial"] is None

    def test_a_new_cell_does_not_inherit_the_last_one_s_output(
        self, client, no_live, monkeypatch
    ):
        # Chunks carry no job id, so a buffer that is not reset would show the
        # previous cell's output as this one's -- and it would look like output
        # this cell had already produced before printing anything.
        _running(monkeypatch, "job-1")
        _chat_api._note_progress("first cell\n")
        _running(monkeypatch, "job-2")
        _chat_api._note_progress("second cell\n")
        partial = client.get("/api/chat/history").json()["partial"]
        assert partial["job_id"] == "job-2"
        assert partial["stdout"] == "second cell\n"

    def test_a_chatty_cell_does_not_grow_the_buffer_without_limit(
        self, client, no_live, monkeypatch
    ):
        # Polled every half second while a turn runs, so an unbounded buffer is
        # both memory and payload. The tail is kept, as the job detail keeps it.
        monkeypatch.setattr(_observe, "_max_output_chars", 20)
        _running(monkeypatch, "job-1")
        for i in range(50):
            _chat_api._note_progress(f"line {i}\n")
        partial = client.get("/api/chat/history").json()["partial"]
        assert len(partial["stdout"]) == 20
        assert partial["truncated"] is True
        assert partial["stdout_len"] > 20
        assert partial["stdout"].endswith("line 49\n")

    def test_streamed_output_never_enters_the_conversation(self, no_live, monkeypatch):
        # It lives on the transport because `_llm_messages` re-projects every
        # stored message on every later turn: streamed stdout would be sent to
        # the provider again and again, duplicating what the finished tool
        # result already carries in full.
        _running(monkeypatch, "job-1")
        _chat_api._note_progress("noisy output\n")
        assert _chat.history() == []
