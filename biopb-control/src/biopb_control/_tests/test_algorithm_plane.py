"""The algorithm plane: install, describe and supervise script entries.

A fake ``uv`` stands in for the real one: ``lock``/``sync`` succeed unless the
file says ``FAIL_LOCK``/``FAIL_SYNC``, and ``run`` execs the file with this
interpreter. The server files are minimal ``Ops`` servers on the SDK's stubs.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import textwrap
import time
from pathlib import Path

import pytest
from biopb import _algorithms

from biopb_control._algorithm_plane import AlgorithmPlane

_FAKE_UV = """
import subprocess, sys
args = sys.argv[1:]
verb = args[0]
i = args.index("--script")
script, rest = args[i + 1], args[i + 2:]
text = open(script).read()
if verb in ("lock", "sync"):
    if "FAIL_" + verb.upper() in text:
        print("fake uv: resolution failed", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
# Like uv: the script runs as a child, and uv exits with it.
sys.exit(subprocess.call([sys.executable, script, *rest]))
"""

_SERVER = """
import argparse, os, sys
from concurrent import futures

import grpc
import biopb.image as proto
from google.protobuf import json_format

OPS = {ops!r}
{extra}

def listing():
    return proto.OpList(ops=[proto.OpInfo(name=n) for n in OPS], fingerprint="-".join(OPS))

class Ops(proto.OpsServicer):
    def Describe(self, request, context):
        token = os.environ.get("BIOPB_ALGORITHM_TOKEN")
        if token and ("authorization", "Bearer " + token) not in context.invocation_metadata():
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "token")
        return listing()

parser = argparse.ArgumentParser()
parser.add_argument("--describe", action="store_true")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=0)
args = parser.parse_args()
if args.describe:
    print(json_format.MessageToJson(listing()))
    sys.exit(0)
print("tensor plane:", os.environ.get("BIOPB_TENSOR_URL"), flush=True)
server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
proto.add_OpsServicer_to_server(Ops(), server)
server.add_insecure_port(f"{{args.host}}:{{args.port}}")
server.start()
server.wait_for_termination()
"""


def _server(ops=("alpha",), extra="") -> str:
    return textwrap.dedent(_SERVER.format(ops=list(ops), extra=extra))


@pytest.fixture
def registry(tmp_path) -> Path:
    d = tmp_path / "algorithms"
    d.mkdir()
    return d


@pytest.fixture
def plane(tmp_path, registry, monkeypatch):
    uv = tmp_path / "fake_uv.py"
    uv.write_text(_FAKE_UV)
    src = Path(__file__).resolve().parents[4] / "src" / "main" / "python"
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")])
    )
    state = tmp_path / "state"
    state.mkdir()
    p = AlgorithmPlane(
        directory=registry,
        state_dir=state,
        uv=[sys.executable, str(uv)],
        data_plane=lambda: ("grpc://127.0.0.1:8815", None),
    )
    yield p
    p.stop_all()


def _crash(entry) -> None:
    """Kill an entry's uv and server at once, as an OOM kill would."""
    if os.name == "nt":
        from biopb._lifecycle import winjob

        winjob.terminate_job(entry._winjob)
    else:
        os.killpg(entry._proc.pid, signal.SIGKILL)


def _row(plane, name) -> dict:
    return next(r for r in plane.rows(probe=False) if r["name"] == name)


def _wait_for(plane, name, states, timeout=30.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        row = _row(plane, name)
        if row["state"] in states or time.monotonic() > deadline:
            return row
        time.sleep(0.1)


def test_refresh_installs_and_caches_the_op_list(plane, registry):
    (registry / "seg.py").write_text(_server(["alpha", "beta"]))
    assert _row(plane, "seg")["state"] == "new"
    plane.refresh()
    row = _wait_for(plane, "seg", {"stopped", "failed"})
    assert row["state"] == "stopped", row["error"]
    assert [o["name"] for o in row["ops"]] == ["alpha", "beta"]
    assert row["url"] is None and row["token"] is None


def test_ensure_starts_the_server_with_its_token(plane, registry):
    (registry / "seg.py").write_text(_server())
    row = plane.ensure("seg", wait=30.0)
    assert row["state"] == "up", row["error"]
    assert row["url"].startswith("grpc://127.0.0.1:")
    assert _algorithms.probe(row["url"], timeout=5)["state"] == "error"
    answer = _algorithms.probe(row["url"], token=row["token"], timeout=5)
    assert answer["state"] == "up"
    # It was told where the data plane is.
    assert "tensor plane: grpc://127.0.0.1:8815" in "\n".join(
        plane.logs("seg", 50)["lines"]
    )


def test_an_edit_takes_effect_on_ensure(plane, registry):
    path = registry / "seg.py"
    path.write_text(_server(["alpha"]))
    first = plane.ensure("seg", wait=30.0)
    path.write_text(_server(["alpha", "gamma"]))
    row = plane.ensure("seg", wait=30.0)
    # The old server went with its uv.
    assert _algorithms.probe(first["url"], timeout=2)["state"] == "unreachable"
    assert row["state"] == "up", row["error"]
    assert [o["name"] for o in row["ops"]] == ["alpha", "gamma"]
    assert row["url"] != first["url"] or row["token"] != first["token"]
    answer = _algorithms.probe(row["url"], token=row["token"], timeout=5)
    assert [o["name"] for o in answer["ops"]] == ["alpha", "gamma"]


def test_an_import_error_fails_without_a_restart_loop(plane, registry):
    (registry / "bad.py").write_text(
        _server(
            extra='if not os.environ.get("X") and "--describe" not in sys.argv:\n'
            "    import no_such_module_xyz"
        )
    )
    row = plane.ensure("bad", wait=30.0)
    assert row["state"] == "failed"
    assert "exited before serving" in row["error"]
    assert "no_such_module_xyz" in row["error"]  # the log tail
    for _ in range(3):
        plane.tick()
    assert _row(plane, "bad")["state"] == "failed"


@pytest.mark.parametrize("verb", ["LOCK", "SYNC"])
def test_an_install_failure_is_failed(plane, registry, verb):
    (registry / "x.py").write_text(_server() + f"\n# FAIL_{verb}\n")
    row = plane.ensure("x", wait=30.0)
    assert row["state"] == "failed"
    assert f"uv {verb.lower()} failed" in row["error"]
    assert "resolution failed" in row["error"]
    # refresh does not retry an unchanged failure; an edit does.
    plane.refresh()
    assert _row(plane, "x")["state"] == "failed"
    (registry / "x.py").write_text(_server())
    plane.refresh()
    assert _wait_for(plane, "x", {"stopped"})["state"] == "stopped"


def test_a_describe_without_an_op_list_is_failed(plane, registry):
    (registry / "x.py").write_text("print('hello')\n")
    row = plane.ensure("x", wait=30.0)
    assert row["state"] == "failed"
    assert "no op list" in row["error"]


def test_a_crash_after_serving_restarts(plane, registry):
    (registry / "seg.py").write_text(_server())
    row = plane.ensure("seg", wait=30.0)
    assert row["state"] == "up"
    _crash(plane._scripts["seg"])
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        plane.tick()
        row = _row(plane, "seg")
        if row["state"] == "up" and row["restarts"] == 1:
            break
        time.sleep(0.1)
    assert row["state"] == "up" and row["restarts"] == 1


def test_stop_and_logs(plane, registry):
    (registry / "seg.py").write_text(_server())
    url = plane.ensure("seg", wait=30.0)["url"]
    assert plane.stop("seg")["state"] == "stopped"
    assert _algorithms.probe(url, timeout=2)["state"] == "unreachable"
    logs = plane.logs("seg", 100)
    assert logs["exists"] and any(
        "starting algorithm seg" in line for line in logs["lines"]
    )


def test_a_removed_file_stops_its_server(plane, registry):
    path = registry / "seg.py"
    path.write_text(_server())
    plane.ensure("seg", wait=30.0)
    row = _row(plane, "seg")
    path.unlink()
    assert plane.rows(probe=False) == []
    # The server itself is gone, not only uv.
    assert _algorithms.probe(row["url"], timeout=2)["state"] == "unreachable"


def test_url_entries_are_not_managed(plane, registry):
    (registry / "remote.json").write_text(json.dumps({"url": "grpc://127.0.0.1:1"}))
    assert plane.ensure("remote", wait=2.0)["state"] == "unreachable"
    for verb in (
        plane.stop,
        lambda n: plane.logs(n, 10),
        lambda n: plane.restart(n, 1),
    ):
        with pytest.raises(ValueError, match="url entry"):
            verb("remote")
    with pytest.raises(KeyError):
        plane.ensure("nope", wait=1.0)


def test_no_uv_is_failed(tmp_path, registry):
    (registry / "seg.py").write_text(_server())
    state = tmp_path / "s"
    state.mkdir()
    p = AlgorithmPlane(directory=registry, state_dir=state, uv=[])
    row = p.ensure("seg", wait=5.0)
    assert row["state"] == "failed"
    assert "uv is not installed" in row["error"]


# --------------------------------------------------------------------------- #
# Over HTTP: the control's routes and biopb.control's verbs
# --------------------------------------------------------------------------- #


@pytest.fixture
def control(plane, tmp_path, monkeypatch):
    import socket

    from biopb_control._control import serve_control_api
    from biopb_control._supervisor import DataPlaneSpec, DataPlaneSupervisor

    def free_port():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_port=free_port(),
        server_log=tmp_path / "server.log",
    )
    port = free_port()
    server, _thread = serve_control_api(
        "127.0.0.1",
        port,
        DataPlaneSupervisor(spec),
        30.0,
        data_web_url="http://127.0.0.1:1",
        algorithms=plane,
    )
    monkeypatch.setattr(
        "biopb.control._client.control_base_url", lambda: f"http://127.0.0.1:{port}"
    )
    monkeypatch.delenv("BIOPB_TENSOR_TOKEN", raising=False)
    monkeypatch.setattr("biopb.control._data_plane.resolve_token", lambda: None)
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    yield
    server.shutdown()


def test_client_verbs_over_http(control, registry):
    from biopb import control as client

    (registry / "seg.py").write_text(_server())
    (registry / "remote.json").write_text(json.dumps({"url": "grpc://127.0.0.1:1"}))
    rows = {r["name"]: r for r in client.algorithms()}
    assert rows["seg"]["state"] == "new"
    assert rows["remote"]["state"] == "unreachable"

    assert client.ensure_algorithm("seg", timeout=60)["state"] == "up"
    assert any(
        "starting algorithm seg" in line for line in client.algorithm_logs("seg")
    )
    assert client.restart_algorithm("seg", timeout=60)["state"] == "up"
    assert client.stop_algorithm("seg")["state"] == "stopped"
    with pytest.raises(LookupError):
        client.ensure_algorithm("nope", timeout=10)
    with pytest.raises(ValueError, match="url entry"):
        client.stop_algorithm("remote")
    assert {r["name"] for r in client.refresh_algorithms()} == {"seg", "remote"}


def test_no_control_is_none(monkeypatch):
    from biopb import control as client

    monkeypatch.setattr(
        "biopb.control._client.control_base_url", lambda: "http://127.0.0.1:1"
    )
    assert client.algorithms(timeout=1) is None
    with pytest.raises(RuntimeError, match="no control"):
        client.ensure_algorithm("x", timeout=1)
