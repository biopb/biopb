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
        "url": "grpc://a:1",
        "target": "a:1",
        "scheme": "grpc",
        "state": "serving",
        "ops": ["threshold", "segment"],
        "op_count": 2,
        "error": None,
        "single_op": False,
    },
    {
        "url": "grpcs://b:2",
        "target": "b:2",
        "scheme": "grpcs",
        "state": "unreachable",
        "ops": [],
        "op_count": 0,
        "error": "UNAVAILABLE: down",
        "single_op": False,
    },
]


@pytest.fixture
def stub_statuses(monkeypatch):
    """Patch _algorithms.statuses; return a dict capturing the timeout it saw."""
    # Widen the rich console so table cells (ops preview, error text) never wrap
    # mid-string under CliRunner's non-terminal default width of 80.
    monkeypatch.setenv("COLUMNS", "200")
    seen = {}

    def _factory(rows):
        def fake(*, timeout):
            seen["timeout"] = timeout
            return rows

        monkeypatch.setattr("biopb._algorithms.statuses", fake)
        return seen

    return _factory


def test_servers_table_lists_configured_servers(stub_statuses):
    stub_statuses(_ROWS)
    result = runner.invoke(app, ["servers"])
    assert result.exit_code == 0
    out = result.stdout
    assert "a:1" in out and "b:2" in out
    assert "threshold, segment" in out  # ops preview for the serving row
    assert "UNAVAILABLE: down" in out  # error shown for the unreachable row


def test_servers_json_emits_the_rows(stub_statuses):
    stub_statuses(_ROWS)
    result = runner.invoke(app, ["servers", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"servers": _ROWS}


def test_servers_single_op_rendered(stub_statuses):
    stub_statuses(
        [
            {
                "url": "grpc://s:1",
                "target": "s:1",
                "scheme": "grpc",
                "state": "serving",
                "ops": [],
                "op_count": 1,
                "error": None,
                "single_op": True,
            }
        ]
    )
    result = runner.invoke(app, ["servers"])
    assert result.exit_code == 0
    assert "(single-op)" in result.stdout


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
    assert seen["timeout"] == 1.5
