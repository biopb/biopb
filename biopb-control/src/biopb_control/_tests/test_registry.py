"""Unit tests for ``biopb_control._registry``: the registry, its migration, and the
``Ops.Describe`` probe against a real in-process server."""

import json
from concurrent import futures
from pathlib import Path

import biopb.image as proto
import grpc
import pytest

from biopb_control import _registry


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolate the config tree under a per-test home (see xdg-test-isolation)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("BIOPB_CONFIG_HOME", "BIOPB_STATE_HOME", "BIOPB_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


@pytest.fixture
def registry(home) -> Path:
    d = _registry.registry_dir()
    d.mkdir(parents=True)
    return d


def _write_mcp_config(home: Path, data: dict) -> None:
    cfg = home / ".config" / "biopb" / "mcp-config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps(data), encoding="utf-8")


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


def test_entries_reads_scripts_and_urls(registry):
    (registry / "cellpose.py").write_text("")
    (registry / "remote.json").write_text('{"url": "grpc://gpu:50051"}')
    (registry / "_draft.py").write_text("")  # skipped
    (registry / "cellpose.py.lock").write_text("")  # uv's lock, not an entry
    (registry / "notes.txt").write_text("")
    listed = {e["name"]: e for e in _registry.entries()}
    assert set(listed) == {"cellpose", "remote"}
    assert listed["cellpose"]["kind"] == "script"
    assert listed["remote"] == {
        "name": "remote",
        "kind": "url",
        "path": str(registry / "remote.json"),
        "url": "grpc://gpu:50051",
        "error": None,
    }
    assert _registry.configured() == _registry.entries()


def test_a_bad_entry_is_listed_with_its_error(registry):
    (registry / "a.json").write_text("{nope")
    (registry / "b.json").write_text('{"host": "x"}')
    (registry / "c.py").write_text("")
    (registry / "c.json").write_text('{"url": "grpc://x:1"}')
    listed = {e["name"]: e["error"] for e in _registry.entries()}
    assert "unreadable" in listed["a"]
    assert "url" in listed["b"]
    assert "both" in listed["c"]


def test_no_registry_is_empty(home):
    assert _registry.entries() == []
    assert _registry.statuses() == []


def test_servers_from_config_normalizes():
    cfg = {"services": {"process_image_servers": ["grpc://a:1", "", 3]}}
    assert _registry.servers_from_config(cfg) == ["grpc://a:1"]
    assert _registry.servers_from_config({}) == []
    assert _registry.servers_from_config({"services": "nope"}) == []
    assert _registry.servers_from_config(None) == []


def test_migration_writes_url_entries_once(home):
    _write_mcp_config(
        home,
        {
            "services": {
                "process_image_servers": ["grpc://gpu:50051", "grpcs://b:2"],
                "docs_local_dir": "/d",
            },
            "timeout": {"process_image": 60},
        },
    )
    assert _registry.migrate_from_mcp_config() == ["gpu-50051", "b-2"]
    urls = {e["name"]: e["url"] for e in _registry.entries()}
    assert urls == {"gpu-50051": "grpc://gpu:50051", "b-2": "grpcs://b:2"}
    # The key leaves the mcp config; the rest of it stays.
    cfg = json.loads((home / ".config" / "biopb" / "mcp-config.json").read_text())
    assert cfg == {
        "services": {"docs_local_dir": "/d"},
        "timeout": {"process_image": 60},
    }

    # Once: an existing registry is the user's, even an emptied one.
    for path in _registry.registry_dir().iterdir():
        path.unlink()
    assert _registry.migrate_from_mcp_config() == []
    assert _registry.entries() == []


def test_migration_without_an_mcp_config_creates_the_registry(home):
    assert _registry.migrate_from_mcp_config() == []
    assert _registry.registry_dir().is_dir()


# --------------------------------------------------------------------------- #
# probe(): URL validation (no server needed)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("url", ["localhost:50051", "http://x:1", "grpc://", "", "x"])
def test_probe_rejects_non_grpc_urls(url):
    result = _registry.probe(url, timeout=1.0)
    assert result["state"] == "invalid"
    assert result["ops"] == []


@pytest.mark.parametrize("url", ["grpc://[::1", "grpcs://[bad"])
def test_probe_does_not_raise_on_malformed_url(url):
    result = _registry.probe(url, timeout=1.0)
    assert result["state"] == "invalid"
    assert result["error"]


def test_probe_unreachable_on_closed_port():
    result = _registry.probe("grpc://127.0.0.1:1", timeout=2.0)
    assert result["state"] == "unreachable"
    assert result["error"]


def test_probe_folds_channel_creation_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("channel init blew up")

    monkeypatch.setattr(grpc, "insecure_channel", boom)
    result = _registry.probe("grpc://host:50051", timeout=1.0)
    assert result["state"] == "error"
    assert "channel init blew up" in result["error"]


# --------------------------------------------------------------------------- #
# probe()/statuses(): against real in-process servers
# --------------------------------------------------------------------------- #


class _Ops(proto.OpsServicer):
    def __init__(self, names=(), fail=False, token=None):
        self._names = names
        self._fail = fail
        self._token = token

    def Describe(self, request, context):  # noqa: N802 - gRPC method name
        if self._token and (
            ("authorization", f"Bearer {self._token}")
            not in context.invocation_metadata()
        ):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "token")
        if self._fail:
            context.abort(grpc.StatusCode.INTERNAL, "kaboom")
        return proto.OpList(
            ops=[proto.OpInfo(name=n, description=f"does {n}") for n in self._names],
            fingerprint="abc",
        )


@pytest.fixture
def serve():
    servers = []

    def start(servicer=None) -> str:
        # No servicer: a server that does not implement Ops, as an older
        # ProcessImage server answers.
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        if servicer is not None:
            proto.add_OpsServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        servers.append(server)
        return f"grpc://127.0.0.1:{port}"

    yield start
    for server in servers:
        server.stop(None)


def test_probe_lists_ops(serve):
    result = _registry.probe(serve(_Ops(["threshold", "segment"])), timeout=5.0)
    assert result["state"] == "up"
    assert [o["name"] for o in result["ops"]] == ["threshold", "segment"]
    assert result["ops"][0]["description"] == "does threshold"
    assert result["fingerprint"] == "abc"
    assert result["error"] is None


def test_probe_sends_the_token(serve):
    url = serve(_Ops(["a"], token="secret"))
    assert _registry.probe(url, timeout=5.0)["state"] == "error"
    assert _registry.probe(url, token="secret", timeout=5.0)["state"] == "up"


def test_probe_names_the_retired_protocol(serve):
    result = _registry.probe(serve(), timeout=5.0)
    assert result["state"] == "error"
    assert "ProcessImage" in result["error"]


def test_probe_unexpected_rpc_error_is_error_state(serve):
    result = _registry.probe(serve(_Ops(fail=True)), timeout=5.0)
    assert result["state"] == "error"
    assert "INTERNAL" in result["error"]


def test_statuses_rows(registry, serve):
    up = serve(_Ops(["only"]))
    (registry / "a-up.json").write_text(json.dumps({"url": up}))
    (registry / "b-down.json").write_text('{"url": "grpc://127.0.0.1:1"}')
    (registry / "c-bad.json").write_text('{"url": "grpc://[::1"}')
    (registry / "d-script.py").write_text("")
    rows = {r["name"]: r for r in _registry.statuses(timeout=2.0)}
    assert rows["a-up"]["state"] == "up"
    assert rows["a-up"]["target"] == up.removeprefix("grpc://")
    assert rows["a-up"]["op_count"] == 1
    assert rows["b-down"]["state"] == "unreachable"
    assert rows["c-bad"]["state"] == "invalid"
    assert rows["d-script"]["state"] == "unknown"
    assert rows["d-script"]["kind"] == "script"
