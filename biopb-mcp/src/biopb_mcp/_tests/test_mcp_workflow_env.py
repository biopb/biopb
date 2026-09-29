"""Tests for the public workflow environment (workflow_env.py).

This is the one call a saved workflow makes to become runnable, and the one the
scratch kernel no longer makes on its behalf -- so what it binds, and what it
says when it cannot, is the contract a notebook's first cell rests on.

No network: the connection and the ops builder are stubbed, because what is
under test is the handoff, not the data plane.
"""

import pytest

from biopb_mcp import workflow_env as we


class _Conn:
    """A Connection that connected, or did not."""

    def __init__(self, client="a-client", message=""):
        self.client = client
        self.last_message = message

    def connect(self):
        return self.client is not None


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A stubbed data plane."""
    import biopb.tensor

    from biopb_mcp.mcp import _process_ops

    monkeypatch.setattr(biopb.tensor, "Connection", lambda: _Conn(), raising=False)
    monkeypatch.setattr(
        _process_ops,
        "build_ops_from_config",
        # Keeps the getter so a test can watch ops follow a reconnect.
        lambda config, getter: {"segment": getter},
    )
    return tmp_path


def _run(source="conn, ops = workflow_env()", **kwargs):
    """Run *source* the way a notebook cell runs, and hand back its namespace.

    ``exec`` with a globals dict is the shape a cell has -- which is what
    ``workflow_env`` reads to find the caller -- so this is the real path, not
    an approximation of it.
    """
    ns = {"workflow_env": we.workflow_env, **kwargs}
    exec(source, ns)  # noqa: S102 - that is the thing being tested
    return ns


class TestHandles:
    def test_the_connection_and_ops_come_back(self, env):
        ns = _run()
        assert ns["conn"].client == "a-client"
        assert "segment" in ns["ops"]

    def test_what_came_back_tracks_a_reconnect(self, env):
        # The reason the connection comes back and not the client: a reconnect
        # swaps the client out, and both the caller and ops must follow it.
        ns = _run()
        conn = ns["conn"]
        resolve_client = ns["ops"]["segment"]
        assert resolve_client() == conn.client

        conn.client = "reconnected"
        assert resolve_client() == "reconnected"

    def test_the_documents_own_spelling_still_reaches_the_client(self, env):
        # The two-line first cell the verify_workflow guidance prints.
        ns = _run("conn, ops = workflow_env()\nclient = conn.client")
        assert ns["client"] == "a-client"

    def test_no_data_plane_raises_where_the_reader_can_act_on_it(
        self, env, monkeypatch
    ):
        # The alternative is a None client and a cell three steps later blaming
        # the workflow for the environment.
        import biopb.tensor

        monkeypatch.setattr(
            biopb.tensor, "Connection", lambda: _Conn(None, "connection refused")
        )
        with pytest.raises(we.WorkflowEnvError) as caught:
            _run()
        assert "connection refused" in str(caught.value)
        assert "biopb control start" in str(caught.value)

    def test_a_workflow_that_needs_no_data_can_say_so(self, env, monkeypatch):
        import biopb.tensor

        monkeypatch.setattr(biopb.tensor, "Connection", lambda: _Conn(client=None))
        # The connection comes back unconnected rather than not at all.
        ns = _run("conn, ops = workflow_env(require_client=False)")
        assert ns["conn"].client is None
