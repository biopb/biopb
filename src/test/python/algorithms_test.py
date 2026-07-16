"""Unit tests for ``biopb._algorithms`` — inspecting the algorithm plane.

Two concerns, mirroring the module's two layers:

- **configured()** — a stdlib read of the biopb-mcp config's flat
  ``services.process_image_servers`` key (and total tolerance of a
  missing/malformed/mistyped config), against a monkeypatched ``$HOME`` so the
  machine's real config can't leak in.
- **probe()/statuses()** — the gRPC path, exercised end-to-end against a *real*
  in-process ``ProcessImage`` server (grpcio + the ``biopb.image`` stubs are base
  ``biopb`` deps, so no mocking of grpc itself): a server that lists ops, one that
  leaves ``GetOpNames`` unimplemented (single-op), one that errors, and a closed
  port (unreachable).
"""

import json
from concurrent import futures
from pathlib import Path

import biopb.image as proto
import grpc
import pytest
from biopb import _algorithms

# --------------------------------------------------------------------------- #
# configured(): the stdlib config read
# --------------------------------------------------------------------------- #


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolate the biopb-mcp config location under a per-test home.

    Also drops inherited ``XDG_*``: ``_config_location.config_dir`` honors
    ``$XDG_CONFIG_HOME`` when it is set (GitHub's Linux runners set it), which
    would otherwise bypass the monkeypatched ``Path.home`` and read the real
    config -- so ``configured()`` would resolve outside this per-test home.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _write_config(home: Path, data: dict) -> None:
    cfg = home / ".config" / "biopb" / "mcp-config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps(data), encoding="utf-8")


def test_configured_reads_flat_services_location(home):
    _write_config(
        home,
        {"services": {"process_image_servers": ["grpc://a:1", "grpc://b:2"]}},
    )
    assert _algorithms.configured() == ["grpc://a:1", "grpc://b:2"]


def test_configured_drops_non_string_and_blank_entries(home):
    _write_config(
        home,
        {"services": {"process_image_servers": ["grpc://a:1", "", 5, None]}},
    )
    assert _algorithms.configured() == ["grpc://a:1"]


def test_configured_is_empty_when_unset(home):
    _write_config(home, {"services": {}})
    assert _algorithms.configured() == []


def test_configured_tolerates_missing_file(home):
    assert _algorithms.configured() == []  # no config written


def test_configured_tolerates_malformed_json(home):
    cfg = home / ".config" / "biopb" / "mcp-config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("{ not json", encoding="utf-8")
    assert _algorithms.configured() == []


def test_configured_tolerates_non_object_config(home):
    cfg = home / ".config" / "biopb" / "mcp-config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("[1, 2, 3]", encoding="utf-8")
    assert _algorithms.configured() == []


# --------------------------------------------------------------------------- #
# servers_from_config(): the shared normalization seam (no disk)
# --------------------------------------------------------------------------- #
# The single source of truth the biopb-mcp kernel also calls with its live CONFIG
# dict, so the key location lives in exactly one place.


def test_servers_from_config_reads_flat():
    cfg = {"services": {"process_image_servers": ["grpc://a:1"]}}
    assert _algorithms.servers_from_config(cfg) == ["grpc://a:1"]


def test_servers_from_config_filters_and_tolerates_bad_shapes():
    assert _algorithms.servers_from_config(
        {"services": {"process_image_servers": ["grpc://a:1", "", 5, None]}}
    ) == ["grpc://a:1"]
    assert _algorithms.servers_from_config({}) == []
    assert _algorithms.servers_from_config({"services": "nope"}) == []
    assert _algorithms.servers_from_config(None) == []


# --------------------------------------------------------------------------- #
# probe(): URL validation (no server needed)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("url", ["localhost:50051", "http://x:1", "grpc://", "", "x"])
def test_probe_rejects_non_grpc_urls(url):
    result = _algorithms.probe(url, timeout=1.0)
    assert result["state"] == "invalid"
    assert result["ops"] == [] and result["op_count"] == 0


def test_probe_unreachable_on_closed_port():
    # Nothing listening -> UNAVAILABLE within the deadline, folded to "unreachable".
    result = _algorithms.probe("grpc://127.0.0.1:1", timeout=2.0)
    assert result["state"] == "unreachable"
    assert result["error"]  # a code/detail string, not None


@pytest.mark.parametrize("url", ["grpc://[::1", "grpcs://[bad"])
def test_probe_does_not_raise_on_malformed_url(url):
    # urlparse raises ValueError on a malformed bracketed IPv6 literal; probe()
    # must fold that into a result, never propagate (it would tank the sweep).
    result = _algorithms.probe(url, timeout=1.0)
    assert result["state"] == "invalid"
    assert result["error"]


def test_probe_folds_channel_creation_failure(monkeypatch):
    # Channel creation can raise (bad target / TLS init); probe() must catch it.
    import grpc

    def boom(*a, **k):
        raise RuntimeError("channel init blew up")

    monkeypatch.setattr(grpc, "insecure_channel", boom)
    result = _algorithms.probe("grpc://host:50051", timeout=1.0)
    assert result["state"] == "error"
    assert "channel init blew up" in result["error"]


# --------------------------------------------------------------------------- #
# probe()/statuses(): against a real in-process ProcessImage server
# --------------------------------------------------------------------------- #


class _Servicer(proto.ProcessImageServicer):
    """A minimal ProcessImage server. ``op_names`` -> GetOpNames returns them;
    ``op_names=None`` -> GetOpNames is left UNIMPLEMENTED (single-op server);
    ``fail`` -> GetOpNames aborts with an unexpected code (INTERNAL)."""

    def __init__(self, op_names=None, fail=False):
        self._op_names = op_names
        self._fail = fail

    def GetOpNames(self, request, context):  # noqa: N802 - gRPC method name
        if self._fail:
            context.abort(grpc.StatusCode.INTERNAL, "kaboom")
        if self._op_names is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "no GetOpNames")
        return proto.OpNames(names=list(self._op_names))


@pytest.fixture
def grpc_server():
    """Start a real insecure ProcessImage server; yields a ``start(servicer)->url``
    factory and tears the server down afterward."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))

    def start(servicer) -> str:
        proto.add_ProcessImageServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        return f"grpc://127.0.0.1:{port}"

    yield start
    server.stop(None)


def test_probe_serving_lists_ops(grpc_server):
    url = grpc_server(_Servicer(op_names=["threshold", "segment"]))
    result = _algorithms.probe(url, timeout=5.0)
    assert result["state"] == "serving"
    assert result["ops"] == ["threshold", "segment"]
    assert result["op_count"] == 2
    assert result["single_op"] is False
    assert result["error"] is None


def test_probe_single_op_server_when_get_op_names_unimplemented(grpc_server):
    url = grpc_server(_Servicer(op_names=None))  # UNIMPLEMENTED GetOpNames
    result = _algorithms.probe(url, timeout=5.0)
    assert result["state"] == "serving"
    assert result["single_op"] is True
    assert result["op_count"] == 1
    assert result["ops"] == []


def test_probe_unexpected_rpc_error_is_error_state(grpc_server):
    url = grpc_server(_Servicer(fail=True))  # aborts INTERNAL
    result = _algorithms.probe(url, timeout=5.0)
    assert result["state"] == "error"
    assert "INTERNAL" in result["error"]


def test_probe_filters_empty_op_names(grpc_server):
    url = grpc_server(_Servicer(op_names=["a", "", "b"]))
    assert _algorithms.probe(url, timeout=5.0)["ops"] == ["a", "b"]


def test_status_carries_identity_fields(grpc_server):
    url = grpc_server(_Servicer(op_names=["a"]))
    row = _algorithms.status(url, timeout=5.0)
    assert row["url"] == url
    assert row["target"] == url.removeprefix("grpc://")  # 127.0.0.1:<port>
    assert row["scheme"] == "grpc"
    assert row["state"] == "serving"


def test_statuses_probes_all_configured_in_order(home, grpc_server, monkeypatch):
    up = grpc_server(_Servicer(op_names=["only"]))
    down = "grpc://127.0.0.1:1"  # nothing listening
    # Configure both (order matters); statuses() reads them via configured().
    _write_config(home, {"services": {"process_image_servers": [up, down]}})
    rows = _algorithms.statuses(timeout=2.0)
    assert [r["url"] for r in rows] == [up, down]  # config order preserved
    assert rows[0]["state"] == "serving" and rows[0]["ops"] == ["only"]
    assert rows[1]["state"] == "unreachable"


def test_statuses_is_empty_with_no_servers(home):
    _write_config(home, {"services": {"process_image_servers": []}})
    assert _algorithms.statuses(timeout=1.0) == []


def test_statuses_survives_a_malformed_configured_url(home, grpc_server):
    # A malformed URL alongside a good one must not abort the concurrent sweep
    # (pool.map re-raises) -- it comes back as an "invalid" row instead.
    up = grpc_server(_Servicer(op_names=["ok"]))
    _write_config(home, {"services": {"process_image_servers": ["grpc://[::1", up]}})
    rows = _algorithms.statuses(timeout=2.0)
    assert [r["state"] for r in rows] == ["invalid", "serving"]
