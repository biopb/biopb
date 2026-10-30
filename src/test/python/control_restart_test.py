"""`biopb control restart` replays what the last `control start` resolved."""

from unittest.mock import MagicMock

import biopb.cli as cli
import pytest
import typer
from biopb._config import locations as _locations
from typer.testing import CliRunner

TOKEN = "tok-" + "x" * 20


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path))
    (tmp_path / "biopb").mkdir()
    monkeypatch.setattr(cli, "_require_biopb_control", lambda: None)


def _stop_exits(code, order=None):
    def stop(**_kw):
        if order is not None:
            order.append("stop")
        raise typer.Exit(code)

    return stop


def test_saved_options_round_trip_with_the_token_apart(tmp_path):
    cli._save_start_options(
        {
            "config": tmp_path / "biopb.json",
            "base_port": 9000,
            "grpc_bind": "0.0.0.0",
            "url_prefix": "/node/h/1",
            "san": ["a.example"],
            "tls": True,
            "allow_relay_clients": True,
        },
        TOKEN,
    )
    assert TOKEN not in _locations.control_start_options_file().read_text()
    opts, token = cli._load_start_options()
    assert opts["config"] == tmp_path / "biopb.json"
    assert opts["base_port"] == 9000 and opts["san"] == ["a.example"]
    assert opts["allow_relay_clients"] is True
    assert token == TOKEN


def test_a_tokenless_start_forgets_the_old_token():
    cli._save_start_options({"base_port": 1}, TOKEN)
    cli._save_start_options({"base_port": 1}, None)
    assert cli._load_start_options()[1] is None


def test_nothing_saved_or_garbage_loads_as_none():
    assert cli._load_start_options() is None
    _locations.control_start_options_file().write_text("{not json")
    assert cli._load_start_options() is None


def test_restart_stops_then_starts_with_the_saved_options(monkeypatch):
    cli._save_start_options({"base_port": 9000, "log_level": "DEBUG"}, TOKEN)
    order = []
    monkeypatch.setattr(cli, "control_stop", _stop_exits(0, order))
    start = MagicMock(side_effect=lambda **_kw: order.append("start"))
    monkeypatch.setattr(cli, "_start_control", start)
    res = CliRunner().invoke(cli.app, ["control", "restart"])
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert order == ["stop", "start"]
    kw = start.call_args.kwargs
    assert kw["base_port"] == 9000 and kw["log_level"] == "DEBUG"
    assert kw["token"] == TOKEN
    assert kw["data_plane"] is True  # unsaved keys keep their defaults


def test_restart_without_a_saved_start_uses_defaults(monkeypatch):
    monkeypatch.setattr(cli, "control_stop", _stop_exits(0))
    start = MagicMock()
    monkeypatch.setattr(cli, "_start_control", start)
    res = CliRunner().invoke(cli.app, ["control", "restart"])
    assert res.exit_code == 0, res.output
    assert "defaults" in res.output
    assert start.call_args.kwargs["token"] is None


def test_restart_does_not_start_when_the_stop_declines(monkeypatch):
    monkeypatch.setattr(cli, "control_stop", _stop_exits(1))
    start = MagicMock()
    monkeypatch.setattr(cli, "_start_control", start)
    res = CliRunner().invoke(cli.app, ["control", "restart"])
    assert res.exit_code == 1
    start.assert_not_called()
