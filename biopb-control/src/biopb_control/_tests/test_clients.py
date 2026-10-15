"""``ClientLauncher``: launching and stopping a client from the dashboard.

The backend is a stub whose "client" is a Python one-liner, so the tests drive
real processes without any vendor CLI.
"""

import re
import sys

import pytest
from biopb._control import _agents

from biopb_control import _clients, _logs
from biopb_control._clients import ClientLauncher, LaunchRefused


class _Backend:
    id = "stub"
    relay = False
    ready_pattern = None

    def __init__(self, code, kind="child", stop=None, ready=None):
        self._argv = [sys.executable, "-c", code]
        self.launch_kind = kind
        self.ready_pattern = ready
        self._stop = stop

    def launch_argv(self, *, remote=False):
        return list(self._argv)

    def stop_argv(self, *, remote=False):
        return self._stop


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    monkeypatch.setattr("biopb._config.locations.mcp_viewer_log_dir", lambda: tmp_path)
    monkeypatch.setattr(_clients, "GRACE", 0.3)
    monkeypatch.setattr(_clients, "READY_TIMEOUT", 3.0)
    monkeypatch.setattr(_clients, "_POLL", 0.05)
    launcher = ClientLauncher(chat_served=True, relay_clients=False)
    yield launcher
    launcher.stop_all()


def _use(monkeypatch, backend, *, relay=False):
    monkeypatch.setattr(_agents, "backend", lambda client_id: backend)
    monkeypatch.setattr(
        _agents, "status", lambda client_id: {"can_launch": True, "relay": relay}
    )


SLEEP = "import time; print('up', flush=True); time.sleep(60)"


def test_a_child_that_stays_up_is_started_and_stoppable(launcher, monkeypatch):
    _use(monkeypatch, _Backend(SLEEP))
    got = launcher.launch("stub")
    assert got["state"] == "started" and "up" in got["log"]
    assert launcher.running_ids() == {"stub"}
    # A second launch is the one that is already there, not another child.
    assert launcher.launch("stub")["already"] is True
    assert launcher.stop("stub") == {"stopped": True}
    assert launcher.running_ids() == set()


def test_a_child_that_dies_on_its_own_stops_counting_as_running(launcher, monkeypatch):
    _use(monkeypatch, _Backend("import time; time.sleep(0.6)"))
    assert launcher.launch("stub")["state"] == "started"
    launcher._launches["stub"].wait(5)
    assert launcher.running_ids() == set()


def test_a_child_that_exits_early_fails_with_its_output(launcher, monkeypatch):
    _use(monkeypatch, _Backend("print('no subscription'); raise SystemExit(3)"))
    got = launcher.launch("stub")
    assert got["state"] == "failed" and "code 3" in got["error"]
    assert "no subscription" in got["log"]
    assert got["log_path"]
    assert launcher.running_ids() == set()


def test_a_readiness_line_decides_started(launcher, monkeypatch):
    # opencode names its URL; the child counts as started once it has.
    ready = re.compile(r"http://\S+")
    _use(
        monkeypatch,
        _Backend(
            "import time; print('Web interface: http://127.0.0.1:9', flush=True);"
            "time.sleep(60)",
            ready=ready,
        ),
    )
    got = launcher.launch("stub")
    assert got["state"] == "started" and "http://127.0.0.1:9" in got["log"]


def test_a_quiet_child_is_starting_not_failed(launcher, monkeypatch):
    monkeypatch.setattr(_clients, "READY_TIMEOUT", 0.4)
    _use(monkeypatch, _Backend(SLEEP, ready=re.compile(r"never")))
    assert launcher.launch("stub")["state"] == "starting"
    assert launcher.running_ids() == {"stub"}


def test_a_client_not_launched_here_is_not_stopped(launcher, monkeypatch):
    _use(monkeypatch, _Backend(SLEEP))
    got = launcher.stop("stub")
    assert got["stopped"] is False and "not started" in got["error"]


def test_a_oneshot_returns_its_output_and_is_stopped_by_its_own_command(
    launcher, monkeypatch
):
    stop = [sys.executable, "-c", "print('disabled')"]
    _use(
        monkeypatch,
        _Backend("print('Open this link: http://x/y')", kind="oneshot", stop=stop),
    )
    got = launcher.launch("stub")
    assert got["state"] == "started" and "http://x/y" in got["log"]
    assert launcher.running_ids() == {"stub"}
    assert launcher.stop("stub") == {"stopped": True, "log": "disabled"}
    assert launcher.running_ids() == set()


def test_a_failing_oneshot_is_not_remembered(launcher, monkeypatch):
    _use(monkeypatch, _Backend("raise SystemExit(2)", kind="oneshot"))
    assert launcher.launch("stub")["state"] == "failed"
    assert launcher.running_ids() == set()


def test_a_child_does_not_inherit_the_exec_token(launcher, monkeypatch):
    from biopb._security.credentials import ENV_CONTROL_EXEC_TOKEN

    monkeypatch.setenv(ENV_CONTROL_EXEC_TOKEN, "secret")
    _use(
        monkeypatch,
        _Backend(
            "import os, time; print('tok=' + os.environ.get("
            f"'{ENV_CONTROL_EXEC_TOKEN}', 'absent'), flush=True); time.sleep(60)"
        ),
    )
    assert "tok=absent" in launcher.launch("stub")["log"]


@pytest.mark.parametrize(
    "chat_served, relay_clients, relay, refused",
    [
        (True, False, False, False),
        (False, True, False, True),  # no execute root
        (True, False, True, True),  # a relay client the operator has not allowed
        (True, True, True, False),
    ],
)
def test_the_launch_gate(monkeypatch, chat_served, relay_clients, relay, refused):
    _use(monkeypatch, _Backend(SLEEP), relay=relay)
    gate = ClientLauncher(chat_served=chat_served, relay_clients=relay_clients)
    agent = {"can_launch": True, "relay": relay}
    assert (gate.blocked(agent) is not None) is refused
    if refused:
        with pytest.raises(LaunchRefused):
            gate.launch("stub")
    assert gate.blocked({"can_launch": False}) == "not launchable here"


def test_clean_drops_escapes_and_logo_lines():
    raw = "\x1b[2m█▀▀█\x1b[0m\n█▄▄█\n\x1b[1mWeb interface: http://127.0.0.1:1/\x1b[0m\n"
    assert _logs.clean(raw) == "Web interface: http://127.0.0.1:1/"
