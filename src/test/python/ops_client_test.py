"""Tests for the ``biopb.image`` Ops codec and client, over an in-process server."""

import math
import time
from concurrent import futures

import biopb.image as proto
import grpc
import numpy as np
import pytest


class _Ops(proto.OpsServicer):
    """``echo`` returns its arguments; ``pair`` two outputs; ``slow`` stalls."""

    def __init__(self, token=None):
        self._token = token
        self.cancelled = False

    def _check(self, context):
        if self._token and (
            ("authorization", f"Bearer {self._token}")
            not in context.invocation_metadata()
        ):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "token")

    def Describe(self, request, context):  # noqa: N802 - gRPC method name
        self._check(context)
        return proto.OpList(
            ops=[proto.OpInfo(name="echo"), proto.OpInfo(name="pair")],
            fingerprint="f",
        )

    def Call(self, request, context):  # noqa: N802 - gRPC method name
        self._check(context)
        if request.op == "echo":
            yield proto.Event(progress="working")
            yield proto.Event(outputs={"result": request.args["x"]})
        elif request.op == "pair":
            yield proto.Event(
                outputs={
                    "1": proto.json_arg("b"),
                    "0": proto.json_arg(7),
                }
            )
        elif request.op == "slow":
            context.add_callback(lambda: setattr(self, "cancelled", True))
            time.sleep(5)
        elif request.op == "refuse":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "bad sigma")
        else:
            context.abort(grpc.StatusCode.NOT_FOUND, f"no op {request.op}")


@pytest.fixture
def serve():
    servers = []

    def start(servicer) -> str:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        proto.add_OpsServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        servers.append(server)
        return f"grpc://127.0.0.1:{port}"

    yield start
    for server in servers:
        server.stop(None)


# --- the codec -------------------------------------------------------------


@pytest.mark.parametrize(
    "value", [None, True, "text", 3, 2.5, [1, "a", None], {"k": [1, 2], "z": {"a": 1}}]
)
def test_json_round_trips(value):
    assert proto.decode_arg(proto.encode_arg(value)) == value


def test_json_keeps_nan_and_inf():
    out = proto.decode_arg(proto.encode_arg([math.nan, math.inf]))
    assert math.isnan(out[0]) and out[1] == math.inf


def test_integral_numbers_read_as_ints_unless_asked_not_to():
    arg = proto.encode_arg(6)
    assert isinstance(proto.decode_arg(arg), int)
    assert isinstance(proto.decode_arg(arg, ints=False), float)


def test_numpy_values_are_json():
    arg = proto.encode_arg({"n": np.float32(1.5), "v": np.arange(3)})
    assert proto.decode_arg(arg) == {"n": 1.5, "v": [0, 1, 2]}


def test_unserialisable_value_is_a_type_error():
    with pytest.raises(TypeError, match="not JSON"):
        proto.encode_arg(object())


def test_array_round_trips_as_pixels_with_default_labels():
    array = np.arange(12, dtype=np.uint16).reshape(3, 4)
    arg = proto.encode_arg(array)
    assert arg.WhichOneof("kind") == "eager"
    assert list(arg.eager.dim_labels) == ["Y", "X"]
    np.testing.assert_array_equal(proto.decode_arg(arg), array)


def test_array_labels_are_the_callers_when_given():
    arg = proto.encode_arg(np.zeros((2, 3, 4), np.uint8), dim_labels=["Z", "Y", "X"])
    assert list(arg.eager.dim_labels) == ["Z", "Y", "X"]


def test_an_empty_arg_has_no_value():
    with pytest.raises(ValueError, match="empty"):
        proto.decode_arg(proto.Arg())


# --- the client ------------------------------------------------------------


def test_describe_lists_the_ops(serve):
    with proto.connect(serve(_Ops())) as client:
        assert [o.name for o in client.describe().ops] == ["echo", "pair"]


def test_call_returns_the_single_output_and_reports_progress(serve):
    seen = []
    array = np.arange(6, dtype=np.uint8).reshape(2, 3)
    with proto.connect(serve(_Ops())) as client:
        out = client.call("echo", on_progress=seen.append, x=array)
    np.testing.assert_array_equal(out, array)
    assert seen == ["working"]


def test_call_orders_several_outputs(serve):
    with proto.connect(serve(_Ops())) as client:
        assert client.call("pair") == (7, "b")


def test_a_refused_call_is_a_value_error(serve):
    with proto.connect(serve(_Ops())) as client:
        with pytest.raises(ValueError, match="refuse: bad sigma"):
            client.call("refuse")


def test_an_unknown_op_is_a_lookup_error(serve):
    with proto.connect(serve(_Ops())) as client:
        with pytest.raises(LookupError):
            client.call("nope")


def test_the_token_is_sent(serve):
    url = serve(_Ops(token="secret"))
    with proto.connect(url) as client:
        with pytest.raises(grpc.RpcError):
            client.describe()
    with proto.connect(url, token="secret") as client:
        assert client.describe().ops


def test_silence_times_out_and_cancels_the_call(serve):
    servicer = _Ops()
    with proto.connect(serve(servicer), inactivity_timeout=0.3) as client:
        with pytest.raises(TimeoutError, match="slow"):
            client.call("slow")
    deadline = time.monotonic() + 3
    while not servicer.cancelled and time.monotonic() < deadline:
        time.sleep(0.05)
    assert servicer.cancelled


def test_a_url_needs_a_host_and_a_known_scheme():
    with pytest.raises(ValueError, match="no host"):
        proto.make_channel("grpc://")
    with pytest.raises(ValueError, match="grpc:// or grpcs://"):
        proto.make_channel("http://x:1")


def test_a_name_is_brought_up_through_the_control(monkeypatch, serve):
    url = serve(_Ops(token="t"))
    asked = []

    def ensure(name, timeout):
        asked.append(name)
        return {"state": "up", "url": url, "token": "t"}

    monkeypatch.setattr("biopb._control.ensure_algorithm", ensure)
    with proto.connect("cellpose") as client:
        assert client.describe().ops
    assert asked == ["cellpose"]


def test_a_name_that_is_not_up_is_a_runtime_error(monkeypatch):
    monkeypatch.setattr(
        "biopb._control.ensure_algorithm",
        lambda name, timeout: {"state": "error", "error": "no gpu"},
    )
    with pytest.raises(RuntimeError, match="error:\nno gpu"):
        proto.connect("cellpose")
