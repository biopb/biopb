"""Tests for the ``biopb image servers`` CLI command.

The command is a thin, read-only face over ``biopb._algorithms.statuses`` (the
algorithm-plane inspector). We stub that core so the test never dials a real gRPC
server, and assert the rendering: the human table, the ``--json`` shape, the
empty-config message, and that ``--timeout`` is threaded through to the probe.
"""

import json

import pytest
from biopb.image.cli import app
from typer.testing import CliRunner

runner = CliRunner()

_ROWS = [
    {
        "name": "a",
        "kind": "url",
        "url": "grpc://a:1",
        "target": "a:1",
        "scheme": "grpc",
        "state": "up",
        "ops": [{"name": "threshold"}, {"name": "segment"}],
        "op_count": 2,
        "fingerprint": "f",
        "error": None,
    },
    {
        "name": "b",
        "kind": "url",
        "url": "grpcs://b:2",
        "target": "b:2",
        "scheme": "grpcs",
        "state": "unreachable",
        "ops": [],
        "op_count": 0,
        "fingerprint": "",
        "error": "UNAVAILABLE: down",
    },
]


@pytest.fixture
def stub_statuses(monkeypatch):
    """Answer for the control; return a dict capturing the timeout it saw.

    ``rows`` None is no control, which falls back to probing the registry
    (``_algorithms.statuses``, answering ``fallback``).
    """
    # Widen the rich console so table cells (ops preview, error text) never wrap
    # mid-string under CliRunner's non-terminal default width of 80.
    monkeypatch.setenv("COLUMNS", "200")
    seen = {}

    def _factory(rows, fallback=()):
        def control(timeout):
            seen["timeout"] = timeout
            return rows

        def probe(*, timeout):
            seen["probed"] = timeout
            return list(fallback)

        monkeypatch.setattr("biopb.algorithms", control)
        monkeypatch.setattr("biopb._algorithms.statuses", probe)
        return seen

    return _factory


def test_servers_table_lists_configured_servers(stub_statuses):
    stub_statuses(_ROWS)
    result = runner.invoke(app, ["servers"])
    assert result.exit_code == 0
    out = result.stdout
    assert "a:1" in out and "b:2" in out
    assert "threshold, segment" in out  # ops preview for the up row
    assert "UNAVAILABLE: down" in out  # error shown for the unreachable row


def test_servers_json_emits_the_rows(stub_statuses):
    stub_statuses(_ROWS)
    result = runner.invoke(app, ["servers", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"servers": _ROWS}


def test_servers_empty_config_message(stub_statuses):
    stub_statuses([])
    result = runner.invoke(app, ["servers"])
    assert result.exit_code == 0
    # The hint is advisory, so it goes to stderr (keeping stdout clean for --json).
    assert "No algorithm servers configured" in result.stderr


def test_servers_threads_timeout_to_probe(stub_statuses):
    seen = stub_statuses(_ROWS)
    result = runner.invoke(app, ["servers", "--timeout", "1.5"])
    assert result.exit_code == 0
    # The control probes url entries under the same deadline, then answers.
    assert seen["timeout"] > 1.5


def test_servers_without_a_control_probes_the_registry(stub_statuses):
    seen = stub_statuses(None, fallback=_ROWS)
    result = runner.invoke(app, ["servers", "--timeout", "1.5"])
    assert result.exit_code == 0
    assert seen["probed"] == 1.5
    assert "No control answered" in result.stderr
    assert "a:1" in result.stdout
