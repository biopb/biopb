"""Tests for the public workflow environment (workflow_env.py).

This is the one call a saved workflow makes to become runnable, and the one the
scratch kernel no longer makes on its behalf -- so what it binds, and what it
says when it cannot, is the contract a notebook's first cell rests on.

No network: the connection and the ops builder are stubbed, because what is
under test is the handoff, not the data plane.
"""

import pytest
from biopb import _locations

from biopb_mcp import workflow_env as we


class _Conn:
    """A TensorConnection that connected, or did not."""

    def __init__(self, client="a-client", message=""):
        self.client = client
        self.last_message = message

    def auto_connect(self):
        return None


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A stubbed data plane and an empty plugin dir of our own."""
    from biopb_mcp import _connection
    from biopb_mcp.mcp import _process_ops

    monkeypatch.setattr(_connection, "TensorConnection", lambda: _Conn())
    monkeypatch.setattr(
        _process_ops, "build_ops_from_config", lambda config, getter: {"segment": None}
    )
    monkeypatch.setattr(_locations, "mcp_plugin_dir", lambda: tmp_path)
    return tmp_path


def _run(source="client, ops = workflow_env()", **kwargs):
    """Run *source* the way a notebook cell runs, and hand back its namespace.

    ``exec`` with a globals dict is the shape a cell has -- which is what
    ``workflow_env`` reads to find the caller -- so this is the real path, not
    an approximation of it.
    """
    ns = {"workflow_env": we.workflow_env, **kwargs}
    exec(source, ns)  # noqa: S102 - that is the thing being tested
    return ns


class TestPlugins:
    def test_a_plugin_binds_into_the_callers_namespace(self, env):
        # Under its file's stem, as the session kernel binds it -- so a document
        # rewritten from a session keeps calling it the way it was called there.
        (env / "rolling_ball.py").write_text(
            "def subtract_background(x):\n    return x\n", encoding="utf-8"
        )
        ns = _run()
        assert ns["rolling_ball"].subtract_background(1) == 1

    def test_what_bound_is_printed(self, env, capsys):
        # The answer to the NameError that follows when it did not bind.
        (env / "rolling_ball.py").write_text("x = 1\n", encoding="utf-8")
        _run()
        assert "kernel plugins: rolling_ball" in capsys.readouterr().out

    def test_nothing_to_load_says_so_rather_than_staying_quiet(self, env, capsys):
        _run()
        assert "kernel plugins: (none)" in capsys.readouterr().out

    def test_plugins_false_binds_nothing(self, env):
        (env / "rolling_ball.py").write_text("x = 1\n", encoding="utf-8")
        ns = _run("client, ops = workflow_env(plugins=False)")
        assert "rolling_ball" not in ns

    def test_a_broken_plugin_is_not_a_broken_workflow(self, env):
        # Fail-open per unit, as in the kernel: the workflow that does not use it
        # must not fail, and the one that does fails where it uses it.
        (env / "bad.py").write_text('raise RuntimeError("boom")\n', encoding="utf-8")
        (env / "good.py").write_text("x = 1\n", encoding="utf-8")
        ns = _run()
        assert "good" in ns and "bad" not in ns
        assert ns["client"] == "a-client"


class TestHandles:
    def test_the_client_and_ops_come_back(self, env):
        ns = _run()
        assert ns["client"] == "a-client"
        assert "segment" in ns["ops"]

    def test_no_data_plane_raises_where_the_reader_can_act_on_it(
        self, env, monkeypatch
    ):
        # The alternative is a None client and a cell three steps later blaming
        # the workflow for the environment.
        from biopb_mcp import _connection

        monkeypatch.setattr(
            _connection,
            "TensorConnection",
            lambda: _Conn(client=None, message="connection refused"),
        )
        with pytest.raises(we.WorkflowEnvError) as caught:
            _run()
        assert "connection refused" in str(caught.value)
        assert "biopb control start" in str(caught.value)

    def test_a_workflow_that_needs_no_data_can_say_so(self, env, monkeypatch):
        from biopb_mcp import _connection

        monkeypatch.setattr(_connection, "TensorConnection", lambda: _Conn(client=None))
        ns = _run("client, ops = workflow_env(require_client=False)")
        assert ns["client"] is None
