"""The kernel's ``ops`` (mcp/_process_ops.py) against a real in-process Ops
server, with the control's calls and the tensor client faked."""

import threading
import time
from concurrent import futures

import biopb.image as proto
import grpc
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
import pytest
from biopb.image.utils import deserialize_image_data, serialize_from_numpy_to_image_data
from biopb.tensor import SerializedTensor, TensorDescriptor
from google.protobuf import json_format, struct_pb2

from biopb_mcp.mcp import _process_ops
from biopb_mcp.mcp._process_ops import Ops, _make_channel, _same_plane

PLANE = "grpc://127.0.0.1:8815"


def _eager(arr, labels=None) -> proto.Arg:
    return proto.Arg(
        eager=serialize_from_numpy_to_image_data(arr, dim_labels=labels).eager_data
    )


def _json(value) -> proto.Arg:
    return proto.Arg(json=json_format.ParseDict(value, struct_pb2.Value()))


def _reference(array_id: str, location: str) -> proto.Arg:
    info = flight.FlightInfo(
        pa.schema([]),
        flight.FlightDescriptor.for_command(
            TensorDescriptor(array_id=array_id).SerializeToString()
        ),
        [],
        -1,
        -1,
    )
    return proto.Arg(
        lazy=SerializedTensor(location=location, flight_info=info.serialize())
    )


class _Servicer(proto.OpsServicer):
    def __init__(self, token=None):
        self.token = token
        self.cancelled = threading.Event()
        self.seen = {}

    def Call(self, request, context):  # noqa: N802 - gRPC method name
        if self.token and (
            ("authorization", f"Bearer {self.token}")
            not in context.invocation_metadata()
        ):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "token")
        args = request.args
        self.seen = {k: v.WhichOneof("kind") for k, v in args.items()}
        op = request.op
        if op == "double" and args["image"].WhichOneof("kind") == "lazy":
            yield proto.Event(outputs={"result": _eager(np.ones((2, 2), np.uint8))})
        elif op == "double":
            image = args["image"].eager
            arr = deserialize_image_data(proto.ImageData(eager_data=image))
            yield proto.Event(
                outputs={"result": _eager(arr * 2, list(image.dim_labels))}
            )
        elif op == "stats":
            kwargs = {
                k: json_format.MessageToDict(v.json)
                for k, v in args.items()
                if v.WhichOneof("kind") == "json"
            }
            yield proto.Event(
                outputs={
                    "0": _eager(np.ones((2, 2), np.uint16)),
                    "1": _json({"kwargs": kwargs}),
                }
            )
        elif op == "track":
            for i in range(3):
                yield proto.Event(progress=f"frame {i + 1}/3")
            yield proto.Event(outputs={"result": _json("done")})
        elif op == "items":
            for i in range(2):
                yield proto.Event(progress=str(i + 1), outputs={"result": _json(i)})
        elif op == "reference":
            yield proto.Event(
                outputs={"result": _reference("scratch/@fields/r", PLANE)}
            )
        elif op == "refuse":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "three channels expected")
        elif op == "count":
            yield proto.Event(outputs={"result": _json({"n": 6, "xs": [1, 2.5]})})
        elif op == "slow":
            context.add_callback(self.cancelled.set)
            while context.is_active():
                time.sleep(0.05)
        else:
            context.abort(grpc.StatusCode.NOT_FOUND, op)


@pytest.fixture
def serve():
    servers = []

    def start(servicer):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        proto.add_OpsServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        servers.append(server)
        return f"grpc://127.0.0.1:{port}", server

    yield start
    for server in servers:
        server.stop(None)


def _info(name, tensors=("image",), **kwargs):
    return {
        "name": name,
        "description": f"does {name}",
        "tensors": {t: {"axes": "YX"} for t in tensors},
        "kwargs": kwargs,
    }


OPS = [
    _info("double"),
    _info("stats", level=0.5),
    _info("track", tensors=()),
    _info("items", tensors=()),
    _info("reference", tensors=()),
    _info("slow", tensors=()),
    _info("refuse", tensors=()),
    _info("count", tensors=()),
]


class _Client:
    """The kernel's tensor client, as far as ops touches it."""

    location = PLANE

    def __init__(self):
        self.uploads = []

    def get_tensor_pb(self, array_id):
        return _reference(array_id, PLANE).lazy

    def add_tensor(self, array_id, template):
        return TensorDescriptor(array_id=array_id)

    def upload_array(self, desc, array):
        self.uploads.append((desc.array_id, np.asarray(array)))


@pytest.fixture
def client():
    return _Client()


def _ops(rows, client=None, timeout=10.0) -> Ops:
    ops = Ops(lambda: client, inactivity_timeout=timeout)
    ops.bind(rows)
    return ops


@pytest.fixture
def url_ops(serve, client):
    url, _server = serve(_Servicer())
    return _ops(
        [{"name": "srv", "kind": "url", "url": url, "state": "up", "ops": OPS}], client
    )


# --------------------------------------------------------------------------- #
# Binding
# --------------------------------------------------------------------------- #


def test_bind_is_a_mapping_and_attributes(url_ops):
    assert sorted(url_ops) == sorted(o["name"] for o in OPS)
    assert url_ops.double is url_ops["double"]
    assert "double" in dir(url_ops)
    assert "does double" in url_ops.double.__doc__
    assert url_ops.stats.default_kwargs == {"level": 0.5}
    with pytest.raises(AttributeError, match="ops.refresh"):
        url_ops.nope  # noqa: B018 - the attribute access is the test


def test_a_shared_op_name_is_qualified():
    rows = [
        {
            "name": "a",
            "kind": "url",
            "url": "grpc://x:1",
            "state": "up",
            "ops": [_info("seg")],
        },
        {
            "name": "b",
            "kind": "url",
            "url": "grpc://y:1",
            "state": "up",
            "ops": [_info("seg")],
        },
        {"name": "c", "kind": "script", "state": "new", "ops": []},
    ]
    assert sorted(_ops(rows)) == ["a_seg", "b_seg"]


def test_no_control_is_empty():
    ops = _ops(None)
    assert len(ops) == 0 and repr(ops) == "<ops: none>"


def test_build_ops_from_config_reads_the_control(monkeypatch):
    rows = [
        {
            "name": "a",
            "kind": "url",
            "url": "grpc://x:1",
            "state": "up",
            "ops": [_info("seg")],
        }
    ]
    monkeypatch.setattr("biopb.control.algorithms", lambda timeout: rows)
    ops = _process_ops.build_ops_from_config({}, lambda: None)
    assert list(ops) == ["seg"]
    monkeypatch.setattr("biopb.control.algorithms", lambda timeout: None)
    assert len(_process_ops.build_ops_from_config({}, lambda: None)) == 0


# --------------------------------------------------------------------------- #
# Calls
# --------------------------------------------------------------------------- #


def test_ndarray_in_ndarray_out(url_ops):
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(url_ops.double(arr), arr * 2)
    np.testing.assert_array_equal(url_ops.double(image=arr), arr * 2)


def test_json_arguments_and_several_outputs(url_ops):
    mask, table = url_ops.stats(image=np.zeros((2, 2)), level=np.float32(0.25))
    assert mask.dtype == np.uint16
    assert table == {"kwargs": {"level": 0.25}}


def test_positional_needs_a_single_tensor_argument(url_ops):
    with pytest.raises(TypeError, match="by name"):
        url_ops.track(np.zeros((2, 2)))


def test_an_array_id_in_uploads_the_result(url_ops, client, serve):
    servicer = _Servicer()
    url, _ = serve(servicer)
    ops = _ops(
        [{"name": "s", "kind": "url", "url": url, "state": "up", "ops": OPS}], client
    )
    result = ops.double("src/@fields/x")
    assert servicer.seen == {"image": "lazy"}
    assert result.startswith("cache://scratch/@fields/double-")
    assert client.uploads[0][0] == result


def test_a_reference_on_the_kernels_plane_keeps_its_id(url_ops, client):
    assert url_ops.reference() == "scratch/@fields/r"
    assert client.uploads == []


def test_progress_then_the_result(url_ops):
    assert url_ops.track() == "done"


def test_several_item_events_are_a_list(url_ops):
    # Not a dict: `items` is the op, not a method.
    assert url_ops.items() == [0, 1]
    assert "items" in url_ops


def test_a_refusal_is_a_value_error_with_the_servers_message(url_ops):
    with pytest.raises(ValueError, match="^refuse: three channels expected$"):
        url_ops.refuse()


def test_integral_json_numbers_are_ints(url_ops):
    result = url_ops.count()
    assert result == {"n": 6, "xs": [1, 2.5]}
    assert type(result["n"]) is int


def test_silence_times_out_and_cancels(serve):
    servicer = _Servicer()
    url, _ = serve(servicer)
    ops = _ops(
        [{"name": "s", "kind": "url", "url": url, "state": "up", "ops": OPS}],
        timeout=0.5,
    )
    with pytest.raises(TimeoutError, match="no word"):
        ops.slow()
    assert servicer.cancelled.wait(5)


def test_a_script_entry_is_ensured_and_found_again(serve, monkeypatch):
    first, first_server = serve(_Servicer(token="t1"))
    second, _ = serve(_Servicer(token="t2"))
    answers = iter(
        [
            {"state": "up", "url": first, "token": "t1"},
            {"state": "up", "url": second, "token": "t2"},
        ]
    )
    ensured = []

    def ensure(name, timeout):
        ensured.append(name)
        return next(answers)

    monkeypatch.setattr("biopb.control.ensure_algorithm", ensure)
    ops = _ops([{"name": "seg", "kind": "script", "state": "stopped", "ops": OPS}])
    assert ops.track() == "done"
    assert ensured == ["seg"]
    # The server restarted elsewhere: the next call finds it again.
    first_server.stop(None)
    assert ops.track() == "done"
    assert ensured == ["seg", "seg"]


def test_a_script_entry_that_fails_says_where_to_look(monkeypatch):
    monkeypatch.setattr(
        "biopb.control.ensure_algorithm",
        lambda name, timeout: {"state": "failed", "error": "ImportError: torch"},
    )
    ops = _ops([{"name": "seg", "kind": "script", "state": "stopped", "ops": OPS}])
    with pytest.raises(RuntimeError, match=r"failed:\nImportError: torch") as err:
        ops.track()
    assert "ops.logs('seg')" in str(err.value)


# --------------------------------------------------------------------------- #
# The control's verbs
# --------------------------------------------------------------------------- #


def test_refresh_rebinds_and_reports(monkeypatch):
    rows = [
        {"name": "a", "kind": "script", "state": "stopped", "ops": [_info("seg")]},
        {"name": "b", "kind": "script", "state": "installing", "ops": []},
        {"name": "c", "kind": "script", "state": "failed", "ops": [], "error": "x"},
    ]
    monkeypatch.setattr("biopb.control.refresh_algorithms", lambda: rows)
    ops = _ops(None)
    report = ops.refresh()
    assert list(ops) == ["seg"]
    assert "added: seg" in report
    assert "still installing: b" in report
    assert "failed: c" in report


def test_status_logs_restart(monkeypatch):
    rows = [
        {
            "name": "a",
            "kind": "script",
            "state": "failed",
            "ops": [_info("seg")],
            "error": "exited before serving\nTraceback...",
        }
    ]
    monkeypatch.setattr("biopb.control.algorithms", lambda: rows)
    monkeypatch.setattr(
        "biopb.control.algorithm_logs", lambda name, lines: ["l1", "l2"]
    )
    monkeypatch.setattr(
        "biopb.control.restart_algorithm",
        lambda name, timeout: {"state": "up", "error": None},
    )
    ops = _ops(None)
    assert (
        ops.status() == "a (script): failed; ops: seg\n  error: exited before serving"
    )
    assert ops.logs("a") == "l1\nl2"
    assert ops.restart("a") == "a: up"
    assert list(ops) == ["seg"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "a,b,same",
    [
        ("grpc://127.0.0.1:8815", "grpc://localhost:8815", True),
        ("grpcs://h:1", "grpc+tls://h:1", True),
        ("grpc://h:1", "grpc://h:2", False),
        ("grpc://a:1", "grpc://b:1", False),
    ],
)
def test_same_plane(a, b, same):
    assert _same_plane(a, b) is same


def test_make_channel_schemes():
    _make_channel("grpc://localhost:1").close()
    _make_channel("grpcs://localhost:1").close()
    with pytest.raises(ValueError):
        _make_channel("http://localhost:1")
    with pytest.raises(ValueError):
        _make_channel("grpc://")
