"""Tests for the ``biopb algorithm`` CLI group.

``list`` is a thin, read-only face over the control client's
``biopb._control.algorithms``. We stub that so the test never dials a control, and assert
the rendering: the human table, the ``--json`` shape, the empty-registry message,
that ``--timeout`` is threaded through, and that a missing control is an error
(the CLI never starts one).
"""

import json

import pytest
from biopb.cli import app
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
def stub_control(monkeypatch):
    """Answer for the control; return a dict capturing the timeout it saw.

    ``rows`` None is no control.
    """
    # Widen the rich console so table cells (ops preview, error text) never wrap
    # mid-string under CliRunner's non-terminal default width of 80.
    monkeypatch.setenv("COLUMNS", "200")
    seen = {}

    def _factory(rows):
        def control(timeout):
            seen["timeout"] = timeout
            return rows

        monkeypatch.setattr("biopb._control.algorithms", control)
        return seen

    return _factory


def test_list_table_lists_configured_servers(stub_control):
    stub_control(_ROWS)
    result = runner.invoke(app, ["algorithm", "list"])
    assert result.exit_code == 0
    out = result.stdout
    assert "a:1" in out and "b:2" in out
    assert "threshold, segment" in out  # ops preview for the up row
    assert "UNAVAILABLE: down" in out  # error shown for the unreachable row


def test_list_json_emits_the_rows(stub_control):
    stub_control(_ROWS)
    result = runner.invoke(app, ["algorithm", "list", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"servers": _ROWS}


def test_list_empty_config_message(stub_control):
    stub_control([])
    result = runner.invoke(app, ["algorithm", "list"])
    assert result.exit_code == 0
    # The hint is for the table view only; --json stays machine-readable.
    assert "No algorithm servers configured" in result.stdout


def test_list_threads_timeout_to_probe(stub_control):
    seen = stub_control(_ROWS)
    result = runner.invoke(app, ["algorithm", "list", "--timeout", "1.5"])
    assert result.exit_code == 0
    # The control probes url entries under the same deadline, then answers.
    assert seen["timeout"] > 1.5


def test_list_without_a_control_is_an_error(stub_control):
    stub_control(None)
    result = runner.invoke(app, ["algorithm", "list"])
    assert result.exit_code == 1
    assert "No control answered" in result.stdout
