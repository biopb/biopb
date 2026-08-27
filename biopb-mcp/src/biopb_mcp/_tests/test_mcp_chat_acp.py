"""Tests for the ACP chat engine (_chat_acp.py) and its HTTP surface.

No subprocess and no harness. Everything here drives the *client* half of the
protocol directly -- the notification handler, the transcript, the permission
question -- because that is the half biopb owns; whether opencode honours the
handshake is a question about opencode, and no local mock can answer it.
"""

import asyncio
import json

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from biopb_mcp.mcp import _chat_acp, _chat_api, _server


@pytest.fixture(autouse=True)
def fresh():
    """A cleared transcript around every case; it is module state by design."""
    _chat_acp._items.clear()
    _chat_acp._by_id.clear()
    _chat_acp._open_message.clear()
    _chat_acp._pending.clear()
    _chat_acp._commands.clear()
    _chat_acp._usage.clear()
    _chat_acp._reset_rev = 0
    yield
    _chat_acp._pending.clear()


def update(client, **fields):
    """Deliver one session/update the way the connection would."""
    return asyncio.run(client.session_update("s", fields))


def pane(mode="ask"):
    return _chat_acp._PaneClient(mode)


class TestTranscript:
    def test_chunks_of_one_message_coalesce(self):
        """A sentence arrives a few characters at a time and is one item.

        opencode streams `agent_message_chunk` in fragments as short as three
        characters, all sharing a messageId. Appending each as its own item
        renders one reply as thirty.
        """
        c = pane()
        for part in ("Hel", "lo the", "re"):
            update(
                c,
                sessionUpdate="agent_message_chunk",
                messageId="m1",
                content={"type": "text", "text": part},
            )
        items, _ = _chat_acp.history()
        assert len(items) == 1
        assert items[0]["blocks"] == [{"type": "text", "text": "Hello there"}]

    def test_a_new_message_id_starts_a_new_item(self):
        c = pane()
        update(
            c,
            sessionUpdate="agent_message_chunk",
            messageId="m1",
            content={"type": "text", "text": "one"},
        )
        update(
            c,
            sessionUpdate="agent_message_chunk",
            messageId="m2",
            content={"type": "text", "text": "two"},
        )
        items, _ = _chat_acp.history()
        assert [i["blocks"][0]["text"] for i in items] == ["one", "two"]

    def test_tool_call_update_merges_and_does_not_erase(self):
        """The completing update carries a status and nothing else.

        Observed from opencode: `tool_call_update` at completion sends
        `title: None, kind: None`. Writing those over the fields the view
        already has would blank the tool's name at the moment it finishes.
        """
        c = pane()
        update(
            c,
            sessionUpdate="tool_call",
            toolCallId="t1",
            title="biopb_server_status",
            status="pending",
        )
        update(c, sessionUpdate="tool_call_update", toolCallId="t1", status="completed")
        items, _ = _chat_acp.history()
        assert items[0]["title"] == "biopb_server_status"
        assert items[0]["status"] == "completed"

    def test_an_update_to_an_unknown_call_is_ignored(self):
        c = pane()
        update(
            c, sessionUpdate="tool_call_update", toolCallId="nope", status="completed"
        )
        assert _chat_acp.history()[0] == []

    def test_thoughts_are_not_transcript(self):
        c = pane()
        update(
            c,
            sessionUpdate="agent_thought_chunk",
            content={"type": "text", "text": "hmm"},
        )
        assert _chat_acp.history()[0] == []


class TestWatermark:
    def test_a_delta_carries_only_what_changed(self):
        c = pane()
        update(
            c, sessionUpdate="tool_call", toolCallId="t1", title="a", status="pending"
        )
        mark = _chat_acp.revision()
        update(
            c, sessionUpdate="tool_call", toolCallId="t2", title="b", status="pending"
        )
        items, full = _chat_acp.history(mark)
        assert not full
        assert [i["title"] for i in items] == ["b"]

    def test_an_in_place_update_reaches_a_caught_up_view(self):
        """The case an id cursor cannot express.

        Nothing is appended -- a call the view already holds merely changes
        status -- so an "everything after id X" read would return nothing and
        the running tool would spin in the pane forever.
        """
        c = pane()
        update(
            c, sessionUpdate="tool_call", toolCallId="t1", title="a", status="pending"
        )
        call_id = _chat_acp.history()[0][0]["id"]
        mark = _chat_acp.revision()
        update(c, sessionUpdate="tool_call_update", toolCallId="t1", status="completed")
        items, full = _chat_acp.history(mark)
        assert not full
        assert [(i["id"], i["status"]) for i in items] == [(call_id, "completed")]

    def test_a_watermark_from_before_a_reset_gets_the_whole_thread(self):
        """A window open across a reset is holding a conversation that is gone."""
        c = pane()
        update(
            c, sessionUpdate="tool_call", toolCallId="t1", title="a", status="pending"
        )
        stale = _chat_acp.revision()
        _chat_acp._items.clear()
        _chat_acp._reset_rev = _chat_acp._bump()
        update(
            c, sessionUpdate="tool_call", toolCallId="t2", title="b", status="pending"
        )
        items, full = _chat_acp.history(stale)
        assert full
        assert [i["title"] for i in items] == ["b"]

    def test_ids_are_not_reissued_after_a_reset(self):
        """Or a stale cursor matches an item the view has never seen."""
        c = pane()
        update(
            c, sessionUpdate="tool_call", toolCallId="t1", title="a", status="pending"
        )
        first = _chat_acp.history()[0][0]["id"]
        _chat_acp._items.clear()
        _chat_acp._reset_rev = _chat_acp._bump()
        update(
            c, sessionUpdate="tool_call", toolCallId="t2", title="b", status="pending"
        )
        assert _chat_acp.history()[0][0]["id"] != first


class TestPermission:
    def test_the_question_lands_in_the_thread_and_waits(self):
        async def scenario():
            c = pane()
            asking = asyncio.ensure_future(
                c.request_permission(
                    "s",
                    {"toolCallId": "t1", "title": "run rm -rf"},
                    [
                        {"optionId": "yes", "name": "Allow", "kind": "allow_once"},
                        {"optionId": "no", "name": "Reject", "kind": "reject_once"},
                    ],
                )
            )
            await asyncio.sleep(0)
            items, _ = _chat_acp.history()
            assert items[0]["kind"] == "permission"
            assert items[0]["title"] == "run rm -rf"
            assert [o["id"] for o in items[0]["options"]] == ["yes", "no"]
            assert not asking.done()

            assert _chat_acp.answer_permission(items[0]["request_id"], "yes")
            reply = await asking
            return reply, _chat_acp.history()[0][0]["outcome"]

        reply, outcome = asyncio.run(scenario())
        assert reply.outcome.option_id == "yes"
        assert outcome == "yes"

    def test_a_null_option_is_a_refusal_not_a_missing_field(self):
        async def scenario():
            c = pane()
            asking = asyncio.ensure_future(
                c.request_permission(
                    "s",
                    {"toolCallId": "t1", "title": "run something"},
                    [{"optionId": "yes", "name": "Allow", "kind": "allow_once"}],
                )
            )
            await asyncio.sleep(0)
            request_id = _chat_acp.history()[0][0]["request_id"]
            assert _chat_acp.answer_permission(request_id, None)
            return await asking

        reply = asyncio.run(scenario())
        assert reply.outcome.outcome == "cancelled"

    def test_an_option_the_agent_did_not_offer_is_refused(self):
        async def scenario():
            c = pane()
            asking = asyncio.ensure_future(
                c.request_permission(
                    "s",
                    {"toolCallId": "t1", "title": "x"},
                    [{"optionId": "yes", "name": "Allow", "kind": "allow_once"}],
                )
            )
            await asyncio.sleep(0)
            request_id = _chat_acp.history()[0][0]["request_id"]
            # Inventing an option would answer a question the agent never asked,
            # and the agent decides what its options mean.
            assert not _chat_acp.answer_permission(request_id, "made-up")
            assert _chat_acp.answer_permission(request_id, "yes")
            await asking

        asyncio.run(scenario())

    def test_answering_a_settled_question_is_refused(self):
        """Two windows watch one conversation; the second click is not an error."""

        async def scenario():
            c = pane()
            asking = asyncio.ensure_future(
                c.request_permission(
                    "s",
                    {"toolCallId": "t1", "title": "x"},
                    [{"optionId": "yes", "name": "Allow", "kind": "allow_once"}],
                )
            )
            await asyncio.sleep(0)
            request_id = _chat_acp.history()[0][0]["request_id"]
            assert _chat_acp.answer_permission(request_id, "yes")
            await asking
            return _chat_acp.answer_permission(request_id, "yes")

        assert asyncio.run(scenario()) is False

    def test_allow_mode_never_asks(self):
        async def scenario():
            c = pane("allow")
            return await c.request_permission(
                "s",
                {"toolCallId": "t1", "title": "x"},
                [
                    {"optionId": "no", "name": "Reject", "kind": "reject_once"},
                    {"optionId": "yes", "name": "Allow", "kind": "allow_always"},
                ],
            )

        reply = asyncio.run(scenario())
        # The *allow* option, not merely the first one.
        assert reply.outcome.option_id == "yes"
        assert _chat_acp.history()[0] == []

    def test_cancelling_a_turn_settles_open_questions(self):
        """Otherwise the harness waits forever on a reply nobody will give."""

        async def scenario():
            c = pane()
            asking = asyncio.ensure_future(
                c.request_permission(
                    "s",
                    {"toolCallId": "t1", "title": "x"},
                    [{"optionId": "yes", "name": "Allow", "kind": "allow_once"}],
                )
            )
            await asyncio.sleep(0)
            _chat_acp._cancel_pending()
            return await asking

        reply = asyncio.run(scenario())
        assert reply.outcome.outcome == "cancelled"


class TestAdvertised:
    def test_commands_are_carried_as_the_agent_sends_them(self):
        c = pane()
        update(
            c,
            sessionUpdate="available_commands_update",
            availableCommands=[
                {
                    "name": "review",
                    "description": "Review code",
                    "input": {"hint": "path"},
                },
                {"description": "no name, dropped"},
            ],
        )
        assert _chat_acp.commands() == [
            {"name": "review", "description": "Review code", "hint": "path"}
        ]

    def test_a_later_update_replaces_the_set(self):
        """They are advertised by notification and may change mid-session."""
        c = pane()
        update(
            c,
            sessionUpdate="available_commands_update",
            availableCommands=[{"name": "a"}],
        )
        update(
            c,
            sessionUpdate="available_commands_update",
            availableCommands=[{"name": "b"}],
        )
        assert [x["name"] for x in _chat_acp.commands()] == ["b"]


class TestContentBlocks:
    def test_a_tool_result_is_unwrapped_from_its_envelope(self):
        assert _chat_acp._block(
            {"type": "content", "content": {"type": "text", "text": "hi"}}
        ) == {"type": "text", "text": "hi"}

    def test_a_diff_is_named_rather_than_dropped(self):
        """Dropping it silently would make an edit look like it did nothing."""
        assert _chat_acp._block({"type": "diff", "path": "/x/y.py"}) == {
            "type": "text",
            "text": "(edited /x/y.py)",
        }

    def test_an_unknown_block_is_dropped(self):
        assert _chat_acp._block({"type": "something-new"}) is None


class TestResolve:
    def test_an_unknown_agent_says_what_is_supported(self):
        with pytest.raises(_chat_acp.AcpNotConfigured) as exc:
            _chat_acp.resolve_command({"chat": {"acp_agent": "emacs"}})
        assert "opencode" in str(exc.value)

    def test_an_override_that_points_at_nothing_is_refused(self, tmp_path):
        with pytest.raises(_chat_acp.AcpNotConfigured):
            _chat_acp.resolve_command(
                {"chat": {"acp_agent": "opencode", "acp_command": str(tmp_path / "no")}}
            )

    def test_an_override_wins_over_the_lookup(self, tmp_path):
        fake = tmp_path / "opencode"
        fake.write_text("#!/bin/sh\n")
        argv, name = _chat_acp.resolve_command(
            {"chat": {"acp_agent": "opencode", "acp_command": str(fake)}}
        )
        assert argv == (str(fake), "acp")
        assert name == "opencode"


class TestHttpSurface:
    """The two routes the ACP engine adds, driven as handlers."""

    @pytest.fixture
    def acp_client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path))
        # Both engines viable, so a refused switch is refused for the reason the
        # case is about rather than because the other engine was never set up.
        monkeypatch.setenv("BIOPB_CHAT_API_KEY", "sk-test")
        fake = tmp_path / "opencode"
        fake.write_text("#!/bin/sh\n")
        _chat_api.configure(
            {
                "observe": {"enabled": True, "chat_enabled": True},
                "chat": {
                    "engine": "acp",
                    "acp_command": str(fake),
                    "model": "test-model",
                },
            },
            agentless=True,
        )
        monkeypatch.setattr(_server, "_claimed_by", None, raising=False)
        app = Starlette(
            routes=[Route(p, h, methods=m) for p, m, h in _chat_api._ROUTES]
        )
        with TestClient(app, base_url="http://127.0.0.1:8766") as c:
            yield c
        _chat_api.configure({"observe": {"enabled": True}}, agentless=True)

    def post(self, client, path, body):
        return client.post(
            path, json=body, headers={"Content-Type": "application/json"}
        )

    def test_history_is_items_and_a_revision(self, acp_client):
        body = acp_client.get("/api/chat/history").json()
        assert body["items"] == []
        assert body["full"] is True
        assert "rev" in body and "commands" in body
        # The built-in loop's key must be absent, or a view that picks its
        # adapter by which key it got would pick the wrong one.
        assert "messages" not in body

    def test_answering_an_unknown_question_is_stale_not_broken(self, acp_client):
        r = self.post(acp_client, "/chat/permission", {"request_id": "p-99"})
        assert r.status_code == 409
        assert r.json()["stale"] is True

    def test_permission_needs_a_request_id(self, acp_client):
        r = self.post(acp_client, "/chat/permission", {})
        assert r.status_code == 400

    def test_compacting_is_refused_because_the_harness_owns_its_context(
        self, acp_client
    ):
        r = self.post(acp_client, "/chat/summary", {})
        assert r.status_code == 400
        assert "context" in r.json()["error"]

    def test_switching_to_the_engine_already_running_changes_nothing(self, acp_client):
        r = self.post(acp_client, "/chat/engine", {"engine": "acp"})
        assert r.status_code == 200
        assert r.json() == {"engine": "acp", "changed": False}

    def test_a_switch_is_visible_to_a_window_that_did_not_make_it(self, acp_client):
        """Two observe pages, one session: the one that did not click has to
        find out, and the once-probed status is not where that can land."""
        assert acp_client.get("/api/chat/engine").json()["engine"] == "acp"
        assert self.post(acp_client, "/chat/engine", {"engine": "builtin"}).status_code
        assert acp_client.get("/api/chat/engine").json() == {
            "engine": "builtin",
            "model": "test-model",
        }

    def test_an_unknown_engine_is_refused(self, acp_client):
        assert (
            self.post(acp_client, "/chat/engine", {"engine": "vim"}).status_code == 400
        )

    def test_a_claimed_kernel_blocks_the_switch(self, acp_client, monkeypatch):
        """Switching into a kernel someone else holds gives a session that
        answers questions and then refuses every cell -- with the refusal buried
        in a tool result, which is the worst place to find out."""
        monkeypatch.setattr(_server, "_claimed_by", "some-other-client")
        r = self.post(acp_client, "/chat/engine", {"engine": "builtin"})
        assert r.status_code == 409
        assert r.json()["claimed_by"] == "some-other-client"
        assert "Restart the kernel" in r.json()["error"]


class TestModelSelection:
    """`chat.acp_model` is applied to the session the harness just opened."""

    class Opt:
        def __init__(self, value):
            self.value = value

    class ConfigOption:
        def __init__(self, option_id, options):
            self.id = option_id
            self.options = options

    class Session:
        session_id = "ses_1"

        def __init__(self, config_options):
            self.config_options = config_options

    class Conn:
        def __init__(self):
            self.calls = []

        async def set_config_option(self, **kw):
            self.calls.append(kw)

    def session(self, values=("openai/gpt-5.5", "opencode/big-pickle")):
        return self.Session([self.ConfigOption("model", [self.Opt(v) for v in values])])

    def test_names_the_model_when_one_is_configured(self):
        conn = self.Conn()
        asyncio.run(_chat_acp._apply_model(conn, self.session(), "openai/gpt-5.5"))
        assert conn.calls == [
            {
                "config_id": "model",
                "session_id": "ses_1",
                "value": "openai/gpt-5.5",
            }
        ]

    def test_leaves_the_default_alone_when_none_is(self):
        conn = self.Conn()
        asyncio.run(_chat_acp._apply_model(conn, self.session(), ""))
        assert conn.calls == []

    def test_a_model_the_agent_does_not_offer_is_not_sent(self):
        """A typo should say so here, not fail at the provider on turn one."""
        conn = self.Conn()
        asyncio.run(_chat_acp._apply_model(conn, self.session(), "openai/gpt-5.5-typo"))
        assert conn.calls == []

    def test_an_agent_with_no_model_setting_is_not_an_error(self):
        conn = self.Conn()
        asyncio.run(_chat_acp._apply_model(conn, self.Session([]), "openai/gpt-5.5"))
        assert conn.calls == []

    def test_a_refused_set_still_leaves_a_usable_session(self):
        """Degraded chat beats no chat: the default model still answers."""

        class Failing(self.Conn):
            async def set_config_option(self, **kw):
                raise RuntimeError("nope")

        asyncio.run(_chat_acp._apply_model(Failing(), self.session(), "openai/gpt-5.5"))


class TestPinnedConfig:
    """What biopb fixes on the harness at launch, and what it deliberately does not."""

    def cfg(self, mode, agent="opencode"):
        return {"chat": {"acp_permission": mode, "acp_agent": agent}}

    def policy(self, mode):
        env = _chat_acp._agent_env(self.cfg(mode))
        return json.loads(env["OPENCODE_CONFIG_CONTENT"])

    def test_ask_pins_a_permission_policy(self):
        assert self.policy("ask")["permission"] == {
            "edit": "ask",
            "bash": "ask",
            "webfetch": "ask",
            "websearch": "ask",
        }

    def test_it_does_not_use_a_wildcard(self):
        """`{"*": "ask"}` would prompt for biopb's own tools too, so every cell
        the agent ran would stop for a click."""
        permission = self.policy("ask")["permission"]
        assert "*" not in permission
        assert "read" not in permission

    def test_allow_pins_no_permission_policy(self):
        """ "Allow" means do not interfere -- including not overriding a stricter
        choice the user made in their own config."""
        assert "permission" not in self.policy("allow")

    def test_the_agents_own_biopb_registration_is_switched_off(self):
        """The installer writes one into the user's client config, and opencode
        merges config MCP servers into an ACP session. Without this the agent
        gets biopb twice -- ours over http on the viewer in front of the user,
        and theirs over stdio, which becomes a second napari window."""
        for mode in ("ask", "allow"):
            # Unconditional: a second viewer is a wrong session, not a
            # preference about one.
            assert self.policy(mode)["mcp"] == {"biopb": {"enabled": False}}

    def test_only_biopbs_own_entry_is_suppressed(self):
        """Any other MCP server the user configured is theirs and stays."""
        assert list(self.policy("ask")["mcp"]) == ["biopb"]

    def test_an_agent_with_no_known_settings_is_launched_untouched(self):
        assert _chat_acp._agent_env(self.cfg("ask", agent="nothing")) == {}

    def test_the_model_is_not_pinned(self):
        """It is a choice, not a guarantee. Pinned in the environment it could
        only be changed by respawning the agent and losing the conversation;
        `session/set_config_option` changes it in place."""
        assert "model" not in self.policy("ask")

    def test_the_policy_is_not_a_file_the_agent_could_edit(self, tmp_path, monkeypatch):
        """The agent can write anywhere under its cwd. A permission file living
        there is one approved edit away from turning its own prompts off; the
        environment is read once at process start and fixed for the run."""
        monkeypatch.setattr(_chat_acp, "_cwd", str(tmp_path))
        env = _chat_acp._agent_env(self.cfg("ask"))
        assert "permission" in json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert list(tmp_path.iterdir()) == []
