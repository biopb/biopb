"""Tests for the in-process chat loop (_chat.py).

The model is injected, so every test here drives a scripted stub: no key, no
network, no provider. The kernel is the same ``mock_kernel_host`` the server
tests use, so tool dispatch is real all the way down to the round trip.
"""

import asyncio
import base64
import json
from unittest.mock import MagicMock

import pytest

from biopb_mcp.mcp import _chat, _server

_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()


def _envelope(value, window_alive=True):
    """A kernel reply carrying the job runner's ``<<JOB_JSON>>`` payload."""
    return {
        "stdout": _server._JOB_DELIM
        + json.dumps({"r": value, "w": window_alive})
        + "\n",
        "result_text": "",
        "error_text": "",
        "status": "ok",
    }


class _Kernel(MagicMock):
    """A kernel host whose job answers can be scripted per call."""

    def script_submit(self, reply):
        """Make the next ``submit`` return *reply* instead of a fresh job."""
        self._submit = reply

    def script_job(self, states):
        """Answer successive polls with ``(status, stdout)`` from *states*."""
        self._states = list(states)


@pytest.fixture
def chat_host():
    host = _Kernel()
    host.is_alive.return_value = True
    host.is_busy.return_value = False
    host.health.return_value = {
        "alive": True,
        "ready": True,
        "start_error": None,
        "teardown_reason": None,
        "busy": False,
        "dead": False,
        "recent_respawns": 0,
        "watchdog_running": True,
    }
    host._submit = None
    host._states = [("ok", "")]

    def execute(code, *_args, **_kwargs):
        if "_jobs.submit(" in code:
            if host._submit is not None:
                return _envelope(host._submit)
            return _envelope({"job_id": "job-1", "status": "running"})
        if "_jobs.poll(" in code:
            status, out = (
                host._states.pop(0) if len(host._states) > 1 else host._states[0]
            )
            return _envelope(
                {
                    "job_id": "job-1",
                    "status": status,
                    "stdout": out,
                    "result_text": "",
                    "error_text": "",
                    "elapsed": 0.1,
                }
            )
        if "_jobs.foreign_digest(" in code:
            return _envelope([])
        if "_jobs.ack_foreign_digest(" in code:
            return _envelope(0)
        if _server._PNG_DELIM in code or "screenshot" in code:
            return {
                "stdout": _server._PNG_DELIM + _PNG + "\n",
                "result_text": "",
                "error_text": "",
                "status": "ok",
            }
        return {"stdout": "", "result_text": "", "error_text": "", "status": "ok"}

    host.execute.side_effect = execute

    old_host, old_poll = _server._kernel_host, _chat._POLL_INTERVAL
    _server.set_kernel_host(host)
    _chat._POLL_INTERVAL = 0.0  # the stream is scripted; no reason to wait on it
    _chat.reset()
    yield host
    _chat.reset()
    _chat._POLL_INTERVAL = old_poll
    _server._kernel_host = old_host
    _server.clear_claim()


def _call(name, **arguments):
    """One tool call in the chat-completions shape."""
    return {
        "id": f"call-{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _scripted(*replies):
    """A model that returns *replies* in order, recording what it was asked.

    The recorded messages are the point of several tests: what the loop hands a
    provider is as much a contract as what it hands the kernel.
    """
    seen = []

    async def model(messages, tools):
        seen.append({"messages": messages, "tools": tools})
        return replies[len(seen) - 1]

    model.seen = seen
    return model


class TestConversation:
    def test_the_thread_is_shared_not_per_caller(self, chat_host):
        # One conversation per session: two senders append to one thread, which
        # is what lets a second window (or a reload) render the whole exchange.
        asyncio.run(_chat.run_turn("first", _scripted({"content": "one"})))
        asyncio.run(_chat.run_turn("second", _scripted({"content": "two"})))
        assert [m["content"] for m in _chat.history()] == [
            "first",
            "one",
            "second",
            "two",
        ]
        # Ids, because the views render server state rather than their own: a
        # window has to tell a message it drew from one another window just sent.
        assert len({m["id"] for m in _chat.history()}) == 4

    def test_run_turn_returns_only_the_new_messages(self, chat_host):
        asyncio.run(_chat.run_turn("first", _scripted({"content": "one"})))
        new = asyncio.run(_chat.run_turn("second", _scripted({"content": "two"})))
        assert [m["content"] for m in new] == ["second", "two"]

    def test_the_handshake_instructions_are_the_system_prompt(self, chat_host):
        # Captured by agentbench and never used; it is where the operation
        # guardrails live, so a loop that drops it is not running the same agent
        # an MCP client would.
        model = _scripted({"content": "ok"})
        asyncio.run(_chat.run_turn("hi", model))
        system = model.seen[0]["messages"][0]
        assert system["role"] == "system"
        assert "napari" in system["content"]
        # ...and it is not in what the views render.
        assert all(m["role"] != "system" for m in _chat.history())


class TestToolSurface:
    def test_the_payload_is_generated_from_the_live_registry(self, chat_host):
        payload = asyncio.run(_chat.tool_payload())
        names = {t["function"]["name"] for t in payload}
        listed = {t.name for t in asyncio.run(_server.mcp.list_tools())}
        assert names == listed
        # $schema/title are pydantic's and several providers reject them.
        for tool in payload:
            assert "$schema" not in tool["function"]["parameters"]
            assert "title" not in tool["function"]["parameters"]

    def test_both_call_tool_shapes_collapse(self, chat_host):
        # server_status has an output schema and returns (blocks, structured);
        # find_skills has none and returns a bare block list. The loop is below
        # the layer that collapses them, so it does that job -- and must, for
        # both, or one whole class of tool comes back as a tuple.
        for name, args in (("server_status", {}), ("find_skills", {"task": "drift"})):
            text, images = asyncio.run(_chat._dispatch(name, args, None))
            assert isinstance(text, str) and text
            assert images == []

    def test_an_image_travels_as_its_own_message_not_a_tool_result(self, chat_host):
        # Chat-completions tool messages are plain strings, so a screenshot
        # cannot ride back in the tool result; it follows as a user message and
        # the tool result carries a placeholder.
        model = _scripted(
            {"content": "", "tool_calls": [_call("take_screenshot")]},
            {"content": "the blob is lower-right"},
        )
        asyncio.run(_chat.run_turn("where is it?", model))

        tool_msg = [m for m in _chat.history() if m["role"] == "tool"][0]
        assert "image" not in tool_msg
        image_msg = [m for m in _chat.history() if m.get("image")][0]
        assert image_msg["role"] == "user"
        # ...and it reaches the provider as an image part, not as base64 prose.
        sent = model.seen[1]["messages"][-1]
        assert sent["content"][1]["type"] == "image_url"
        assert sent["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )


class TestExecuteCode:
    def test_submitted_as_chat_with_no_promote_window(self, chat_host):
        model = _scripted(
            {"content": "", "tool_calls": [_call("execute_code", python_code="x = 1")]},
            {"content": "done"},
        )
        asyncio.run(_chat.run_turn("set x", model))
        (snippet,) = [
            c[0][0]
            for c in chat_host.execute.call_args_list
            if "_jobs.submit(" in c[0][0]
        ]
        # Its own origin, so the notebook export and the observe badge do not
        # read a chat cell as an MCP agent's...
        assert "origin='chat'" in snippet
        # ...and its own writer id, so the kernel's one-agent claim covers it.
        assert "writer='biopb-chat'" in snippet

    def test_intent_falls_back_to_the_users_own_words(self, chat_host):
        # The model may state a purpose closer to the cell than the turn is; when
        # it does not, the user's turn is still a truthful record of why.
        model = _scripted(
            {"content": "", "tool_calls": [_call("execute_code", python_code="x = 1")]},
            {"content": "done"},
        )
        asyncio.run(_chat.run_turn("measure the drift", model))
        (snippet,) = [
            c[0][0]
            for c in chat_host.execute.call_args_list
            if "_jobs.submit(" in c[0][0]
        ]
        assert "intent='measure the drift'" in snippet

    def test_the_models_own_intent_wins(self, chat_host):
        model = _scripted(
            {
                "content": "",
                "tool_calls": [
                    _call("execute_code", python_code="x = 1", intent="warm the cache")
                ],
            },
            {"content": "done"},
        )
        asyncio.run(_chat.run_turn("measure the drift", model))
        (snippet,) = [
            c[0][0]
            for c in chat_host.execute.call_args_list
            if "_jobs.submit(" in c[0][0]
        ]
        assert "intent='warm the cache'" in snippet

    def test_partial_output_is_reported_while_the_job_runs(self, chat_host):
        # The whole reason the promote window is dropped: a long cell must show
        # its prints as they happen instead of a stalled turn.
        chat_host.script_job(
            [
                ("running", "step 1\n"),
                ("running", "step 1\nstep 2\n"),
                ("ok", "step 1\nstep 2\ndone\n"),
            ]
        )
        seen = []
        model = _scripted(
            {"content": "", "tool_calls": [_call("execute_code", python_code="go()")]},
            {"content": "finished"},
        )
        asyncio.run(_chat.run_turn("go", model, on_progress=seen.append))
        # Deltas, not the whole buffer each time.
        assert seen == ["step 1\n", "step 2\n", "done\n"]

    def test_a_kernel_held_by_another_client_is_reported_not_retried(self, chat_host):
        chat_host.script_submit(
            {"error": "not_owner", "owner": "claude-code", "owner_id": "sess-A"}
        )
        model = _scripted(
            {"content": "", "tool_calls": [_call("execute_code", python_code="x = 1")]},
            {"content": "I cannot run code here."},
        )
        asyncio.run(_chat.run_turn("go", model))
        tool_msg = [m for m in _chat.history() if m["role"] == "tool"][0]
        assert "already in use by another client (claude-code)" in tool_msg["content"]
        # The refusal names the real holder, so the mirror is corrected rather
        # than left guessing at the loop.
        assert _server._claimed_by == "sess-A"


class TestConcurrency:
    def test_a_second_turn_is_refused_not_queued(self, chat_host):
        # Same rule as _jobs.submit, for the same reason: a queued turn would be
        # composed against a conversation its sender has not seen the end of.
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow(messages, tools):
            started.set()
            await release.wait()
            return {"content": "first"}

        async def scenario():
            first = asyncio.create_task(_chat.run_turn("one", slow))
            await started.wait()
            with pytest.raises(_chat.TurnInProgress):
                await _chat.run_turn("two", _scripted({"content": "second"}))
            release.set()
            await first
            # The refused turn left no trace: no half-written user message for a
            # view to render and no history for the next turn to inherit.
            return [m["content"] for m in _chat.history()]

        assert asyncio.run(scenario()) == ["one", "first"]

    def test_the_lock_is_released_when_a_turn_raises(self, chat_host):
        async def boom(messages, tools):
            raise RuntimeError("provider exploded")

        async def scenario():
            with pytest.raises(RuntimeError):
                await _chat.run_turn("one", boom)
            # A failed turn must not wedge the session for good.
            await _chat.run_turn("two", _scripted({"content": "ok"}))

        asyncio.run(scenario())
        assert _chat.history()[-1]["content"] == "ok"

    def test_a_running_job_does_not_block_the_event_loop(self, chat_host):
        # This process also serves /mcp to any attached client and /api/* to the
        # observe page. A turn that sleeps or blocks on kernel round trips would
        # stall both for as long as the cell runs -- minutes, for real work.
        chat_host.script_job([("running", "a\n"), ("running", "ab\n"), ("ok", "abc\n")])
        model = _scripted(
            {"content": "", "tool_calls": [_call("execute_code", python_code="go()")]},
            {"content": "done"},
        )

        async def scenario():
            ticks = 0
            stop = False

            async def other_work():
                nonlocal ticks
                while not stop:
                    ticks += 1
                    await asyncio.sleep(0)

            ticker = asyncio.create_task(other_work())
            await _chat.run_turn("go", model)
            stop = True
            await ticker
            return ticks

        # If anything in the turn blocked, the loop never came back to the
        # ticker and this is 1.
        assert asyncio.run(scenario()) > 1


class TestGuards:
    def test_the_loop_identifies_itself_to_every_tool(self, chat_host):
        # Not just to submit: interrupt_kernel and restart_kernel gate on the
        # same claim, and a loop that arrived as "no identity" would slip past
        # all of them.
        seen = {}

        async def model(messages, tools):
            seen["identity"] = _server._client_identity()
            return {"content": "ok"}

        asyncio.run(_chat.run_turn("hi", model))
        assert seen["identity"] == ("biopb-chat", "chat")
        # ...and only for the duration of the turn.
        assert _server._client_identity() == (None, "")

    def test_a_model_that_never_answers_ends_the_turn(self, chat_host):
        # A tool-call loop that does not converge is not an error the model can
        # be told about -- it is the turn ending, and the user is owed that.
        async def model(messages, tools):
            return {"content": "", "tool_calls": [_call("server_status")]}

        asyncio.run(_chat.run_turn("hi", model))
        assert _chat.history()[-1]["role"] == "assistant"
        assert "Stopped after" in _chat.history()[-1]["content"]

    def test_malformed_tool_arguments_do_not_kill_the_turn(self, chat_host):
        bad = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "server_status", "arguments": "{not json"},
        }
        model = _scripted(
            {"content": "", "tool_calls": [bad]}, {"content": "recovered"}
        )
        asyncio.run(_chat.run_turn("hi", model))
        assert _chat.history()[-1]["content"] == "recovered"


@pytest.mark.parametrize("text", ["", "   "])
def test_an_empty_turn_is_still_recorded(chat_host, text):
    # The loop does not police the input; whatever the transport accepted is
    # what the record should show.
    asyncio.run(_chat.run_turn(text, _scripted({"content": "?"})))
    assert _chat.history()[0]["content"] == text
