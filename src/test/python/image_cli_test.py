"""Tests for the ``biopb image`` CLI (``ops`` and ``call``), against an in-process
algorithm server."""

from concurrent import futures

import biopb.image as proto
import grpc
import imageio
import numpy as np
import pytest
from biopb.image.cli import app
from typer.testing import CliRunner

runner = CliRunner()


class _Ops(proto.OpsServicer):
    def __init__(self, token=None):
        self._token = token

    def Describe(self, request, context):  # noqa: N802 - gRPC method name
        if self._token and (
            ("authorization", f"Bearer {self._token}")
            not in context.invocation_metadata()
        ):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "token")
        return proto.OpList(
            ops=[
                proto.OpInfo(
                    name="invert",
                    description="255 - x",
                    tensors={"image": proto.TensorArg(axes="YX")},
                    kwargs="gain=1",
                ),
                proto.OpInfo(name="stats"),
            ]
        )

    def Call(self, request, context):  # noqa: N802 - gRPC method name
        if request.op == "invert":
            image = proto.decode_arg(request.args["image"])
            gain = (
                proto.decode_arg(request.args["gain"]) if "gain" in request.args else 1
            )
            yield proto.Event(progress="inverting")
            yield proto.Event(outputs={"result": proto.encode_arg(255 - image * gain)})
        elif request.op == "stats":
            yield proto.Event(outputs={"result": proto.json_arg({"n": 3})})


def _serve(servicer):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    proto.add_OpsServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"grpc://127.0.0.1:{port}"


@pytest.fixture
def url():
    server, address = _serve(_Ops())
    yield address
    server.stop(None)


@pytest.fixture
def secured_url():
    server, address = _serve(_Ops(token="secret"))
    yield address
    server.stop(None)


@pytest.fixture(autouse=True)
def wide_console(monkeypatch):
    # Widen the rich console so table cells never wrap under CliRunner's
    # non-terminal default width of 80.
    monkeypatch.setenv("COLUMNS", "200")


def test_ops_lists_the_server_ops(url):
    result = runner.invoke(app, ["ops", url])
    assert result.exit_code == 0
    assert "invert" in result.stdout and "255 - x" in result.stdout
    assert "image: YX" in result.stdout
    assert "gain=1" in result.stdout


def test_call_runs_an_op_on_an_image_file(url, tmp_path):
    source, out = tmp_path / "in.png", tmp_path / "out.png"
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    imageio.imwrite(source, image)
    result = runner.invoke(
        app, ["call", url, "invert", "-i", str(source), "-O", str(out)]
    )
    assert result.exit_code == 0, result.stdout
    np.testing.assert_array_equal(imageio.imread(out), 255 - image)


def test_call_takes_the_only_tensor_and_the_kwargs(url, tmp_path):
    source, out = tmp_path / "in.png", tmp_path / "out.png"
    imageio.imwrite(source, np.ones((2, 2), dtype=np.uint8))
    result = runner.invoke(
        app,
        ["call", url, "invert", "-i", str(source), "-k", '{"gain": 5}', "-O", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    assert (imageio.imread(out) == 250).all()


def test_call_cannot_place_an_input_for_an_op_without_tensors(url):
    result = runner.invoke(app, ["call", url, "stats", "-i", "-"], input=b"")
    # `stats` takes no tensor, so there is nowhere to put the input.
    assert result.exit_code == 1
    assert "--tensor is required" in result.stderr


def test_call_names_the_ops_when_the_op_is_missing(url):
    result = runner.invoke(app, ["call", url, "--tensor", "x"])
    assert result.exit_code == 1
    assert "OP is required" in result.stderr


def test_call_refuses_an_unknown_op(url):
    result = runner.invoke(app, ["call", url, "nope", "--tensor", "x"])
    assert result.exit_code == 1
    assert "no op 'nope'" in result.stderr


def test_a_registry_name_is_resolved_through_the_control(url, monkeypatch):
    monkeypatch.setattr(
        "biopb._control.ensure_algorithm",
        lambda name, timeout: {"state": "up", "url": url, "token": None},
    )
    result = runner.invoke(app, ["ops", "cellpose"])
    assert result.exit_code == 0
    assert "invert" in result.stdout


def test_an_unreachable_registry_name_is_a_clean_error(monkeypatch):
    def refuse(name, timeout):
        raise LookupError(f"no algorithm {name!r}")

    monkeypatch.setattr("biopb._control.ensure_algorithm", refuse)
    result = runner.invoke(app, ["ops", "ghost"])
    assert result.exit_code == 1
    assert "no algorithm 'ghost'" in result.stderr


@pytest.mark.parametrize("command", [["ops"], ["call", "--tensor", "image"]])
def test_the_token_option_is_sent(secured_url, command):
    refused = runner.invoke(app, [command[0], secured_url, *command[1:]])
    assert refused.exit_code == 1
    assert "UNAUTHENTICATED" in refused.stderr

    accepted = runner.invoke(
        app, [command[0], secured_url, *command[1:], "--token", "secret"]
    )
    assert "UNAUTHENTICATED" not in accepted.stderr
    if command[0] == "ops":
        assert accepted.exit_code == 0
        assert "invert" in accepted.stdout


def test_the_token_is_read_from_the_environment(secured_url, monkeypatch):
    monkeypatch.setenv("BIOPB_IMAGE_TOKEN", "secret")
    result = runner.invoke(app, ["ops", secured_url])
    assert result.exit_code == 0
