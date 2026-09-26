"""The Ops protocol served from ``@op`` functions, over a real gRPC channel."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import Annotated

import biopb.image as proto
import grpc
import numpy as np
import pytest
from biopb.image.utils import deserialize_image_data, serialize_from_numpy_to_image_data
from biopb_image_base import Tensor, op
from biopb_image_base.ops import (
    _EmbeddedSink,
    _InlineSink,
    _PlaneSink,
    build_server,
    describe,
)
from google.protobuf import empty_pb2, json_format, struct_pb2


@op(description="Mean intensity and area per label", labels=["measurement"])
def label_stats(image: Tensor("YX"), labels: Tensor("YX")) -> dict:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    return {
        "label": ids,
        "area": np.array([(labels == i).sum() for i in ids]),
        "mean_intensity": np.array([image[labels == i].mean() for i in ids]),
    }


@op
def threshold(image: Tensor("YX"), level: float = 0.5, count: int = 1):
    """Foreground mask.

    More detail that stays out of the description.
    """
    mask = (image > level).astype(np.uint8)
    return mask, {"count": count, "type": type(count).__name__}


@op(input="blocks", block_shape=16, overlap=4)
def smooth(image: Tensor("YX"), size: int = 3):
    from scipy.ndimage import uniform_filter

    return uniform_filter(image.astype(np.float32), size=size, mode="nearest")


@op(input="blocks", block_shape=16)
def shrink(image: Tensor("YX")):
    return image[::2, ::2]


@op(input="lazy")
def frames(movie: Tensor("TYX")):
    for t in range(movie.shape[0]):
        yield movie[t].compute() * 2


@op(input="lazy")
def double(image: Tensor("YX")):
    return image * 2


@op(input="eager")
def total(image: Annotated[np.ndarray, Tensor("YX")]) -> float:
    return float(image.sum())


@op
def greet(name: str) -> str:
    return f"hello {name}"


ALL = [label_stats, threshold, smooth, shrink, frames, double, total, greet]


def _defs(functions=ALL):
    return [f.__biopb_op__ for f in functions]


def _eager(array, labels=None) -> proto.Arg:
    return proto.Arg(
        eager=serialize_from_numpy_to_image_data(array, dim_labels=labels).eager_data
    )


def _json(value) -> proto.Arg:
    return proto.Arg(json=json_format.ParseDict(value, struct_pb2.Value()))


def _value(arg: proto.Arg):
    kind = arg.WhichOneof("kind")
    if kind == "json":
        return json_format.MessageToDict(arg.json)
    if kind == "eager":
        return deserialize_image_data(proto.ImageData(eager_data=arg.eager))
    from biopb.tensor.client import TensorFlightClient

    return TensorFlightClient.tensor_from_pb(arg.lazy).compute()


class _Server:
    def __init__(self, definitions, sink=None, token=None):
        self.server, port = build_server(definitions, sink=sink, token=token)
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.stub = proto.OpsStub(self.channel)

    def call(self, op_name, metadata=None, **args):
        request = proto.Call(op=op_name, args=args)
        return list(self.stub.Call(request, timeout=30, metadata=metadata))

    def close(self):
        self.channel.close()
        self.server.stop(None)


@pytest.fixture
def server():
    s = _Server(_defs())
    yield s
    s.close()


def test_describe_advertises_signature(server):
    ops = {o.name: o for o in server.stub.Describe(empty_pb2.Empty()).ops}
    assert set(ops) == {f.__name__ for f in ALL}

    stats = ops["label_stats"]
    assert stats.description == "Mean intensity and area per label"
    assert list(stats.labels) == ["measurement"]
    assert {k: (v.axes, v.mapped) for k, v in stats.tensors.items()} == {
        "image": ("YX", False),
        "labels": ("YX", False),
    }
    assert dict(ops["threshold"].kwargs) == {"level": 0.5, "count": 1}
    assert ops["threshold"].description == "Foreground mask."
    assert ops["smooth"].tensors["image"].mapped
    assert ops["greet"].kwargs.fields["name"].HasField("null_value")


def test_fingerprint_changes_with_the_ops():
    one = describe(_defs([greet, total])).fingerprint
    assert describe(_defs([total, greet])).fingerprint == one
    assert describe(_defs([greet])).fingerprint != one


def test_two_tensor_arguments_and_json_result(server):
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    labels = np.zeros((4, 4), np.uint16)
    labels[:2] = 1
    labels[3, 3] = 2
    (event,) = server.call("label_stats", image=_eager(image), labels=_eager(labels))
    assert set(event.outputs) == {"result"}
    assert _value(event.outputs["result"]) == {
        "label": [1, 2],
        "area": [8, 1],
        "mean_intensity": [3.5, 15.0],
    }


def test_tuple_outputs_and_kwarg_coercion(server):
    image = np.array([[0.0, 1.0], [0.2, 0.9]], np.float32)
    (event,) = server.call(
        "threshold", image=_eager(image), level=_json(0.5), count=_json(3)
    )
    mask = _value(event.outputs["0"])
    np.testing.assert_array_equal(mask, [[0, 1], [0, 1]])
    assert _value(event.outputs["1"]) == {"count": 3, "type": "int"}


def test_other_axes_are_put_back(server):
    image = np.random.rand(1, 5, 6, 1).astype(np.float32)
    (event,) = server.call("threshold", image=_eager(image, ["Z", "Y", "X", "C"]))
    out = event.outputs["0"].eager
    assert list(out.dim_labels) == ["Z", "Y", "X", "C"]
    assert list(out.dims) == [1, 5, 6, 1]


def test_eager_refuses_a_non_singleton_axis(server):
    image = np.zeros((3, 4, 4), np.float32)
    with pytest.raises(grpc.RpcError) as err:
        server.call("threshold", image=_eager(image, ["T", "Y", "X"]))
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "input='blocks'" in err.value.details()


def test_argument_errors(server):
    with pytest.raises(grpc.RpcError) as err:
        server.call("greet", name=_json("a"), extra=_json(1))
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "extra" in err.value.details()

    with pytest.raises(grpc.RpcError) as err:
        server.call("greet")
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    with pytest.raises(grpc.RpcError) as err:
        server.call("nope")
    assert err.value.code() == grpc.StatusCode.NOT_FOUND


def test_annotated_tensor(server):
    (event,) = server.call("total", image=_eager(np.ones((3, 4), np.float32)))
    assert _value(event.outputs["result"]) == 12.0


def test_an_op_with_no_tensor(server):
    (event,) = server.call("greet", name=_json("you"))
    assert _value(event.outputs["result"]) == "hello you"


def test_blocks_matches_the_whole_image(server):
    from scipy.ndimage import uniform_filter

    movie = np.random.rand(3, 40, 37).astype(np.float32)
    (event,) = server.call("smooth", image=_eager(movie, ["T", "Y", "X"]))
    out = event.outputs["result"].eager
    assert list(out.dim_labels) == ["T", "Y", "X"]
    expected = np.stack([uniform_filter(f, size=3, mode="nearest") for f in movie])
    np.testing.assert_allclose(_value(event.outputs["result"]), expected, rtol=1e-6)


def test_blocks_refuses_a_non_pixelwise_op(server):
    with pytest.raises(grpc.RpcError) as err:
        server.call("shrink", image=_eager(np.zeros((32, 32), np.float32)))
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "pixelwise" in err.value.details()


def test_streaming_op_sends_an_event_per_item(server):
    movie = np.arange(3 * 2 * 2, dtype=np.int32).reshape(3, 2, 2)
    events = server.call("frames", movie=_eager(movie, ["T", "Y", "X"]))
    assert [e.progress for e in events] == ["1", "2", "3"]
    for t, event in enumerate(events):
        np.testing.assert_array_equal(_value(event.outputs["result"]), movie[t] * 2)


def test_a_token_is_checked():
    s = _Server(_defs([greet]), token="secret")
    try:
        with pytest.raises(grpc.RpcError) as err:
            s.call("greet", name=_json("a"))
        assert err.value.code() == grpc.StatusCode.UNAUTHENTICATED
        (event,) = s.call(
            "greet", metadata=[("authorization", "Bearer secret")], name=_json("a")
        )
        assert _value(event.outputs["result"]) == "hello a"
    finally:
        s.close()


def test_inline_sink_refuses_an_oversized_result(monkeypatch):
    import biopb_image_base.ops as ops

    monkeypatch.setattr(ops, "_MAX_MSG_SIZE", 2 * 1024**2)
    with pytest.raises(ValueError, match="--cache-dir"):
        _InlineSink().put(np.zeros(2 * 1024**2, np.uint8), None, "x")


def test_describe_flag_prints_json_without_serving(tmp_path: Path):
    script = tmp_path / "server.py"
    script.write_text(
        textwrap.dedent(
            """
            from biopb_image_base import Tensor, op, serve

            @op(labels=["denoising"])
            def denoise(image: Tensor("YX"), sigma: float = 2.0):
                return image

            if __name__ == "__main__":
                serve()
            """
        )
    )
    done = subprocess.run(
        [sys.executable, str(script), "--describe", "--port", "1"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    listing = json.loads(done.stdout)
    assert listing["ops"][0]["name"] == "denoise"
    assert listing["ops"][0]["kwargs"] == {"sigma": 2.0}
    assert listing["fingerprint"]


def test_gzip_only_integer_outputs_off_loopback():
    from biopb_image_base.ops import _OpsServicer

    class Context:
        disabled = 0

        def disable_next_message_compression(self):
            self.disabled += 1

    servicer = _OpsServicer(_defs([greet]), _InlineSink(), compress=True)
    definition = greet.__biopb_op__
    context = Context()
    servicer._event(
        definition, np.zeros((4, 4), np.uint16), lambda a: (a, None), context
    )
    assert context.disabled == 0
    servicer._event(
        definition, np.zeros((4, 4), np.float32), lambda a: (a, None), context
    )
    servicer._event(definition, "text", lambda a: (a, None), context)
    assert context.disabled == 2


# ---------------------------------------------------------------------------
# Sinks and lazy input
# ---------------------------------------------------------------------------


@pytest.fixture
def plane(tmp_path: Path):
    """A writable tensor server with its scratch source, as the control runs."""
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.core.config import CacheConfig
    from biopb_tensor_server.serving.metadata_db import MetadataDatabase
    from biopb_tensor_server.serving.server import TensorFlightServer

    CacheManager.reset()
    CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
    server = TensorFlightServer(
        location="grpc://localhost:0",
        writable=True,
        write_dir=tmp_path,
        metadata_db=MetadataDatabase(),
    )
    server.mark_ready()
    threading.Thread(target=server.serve, daemon=True).start()
    try:
        yield f"grpc://localhost:{server.port}"
    finally:
        server.shutdown()
        CacheManager.reset()


def test_plane_sink_returns_large_results_by_reference(plane, monkeypatch):
    import biopb_image_base.ops as ops

    monkeypatch.setattr(ops, "_MAX_EAGER_SIZE", 1024)
    s = _Server(_defs([threshold, double]), sink=_PlaneSink(plane, None))
    try:
        image = np.random.rand(64, 50).astype(np.float32)
        (event,) = s.call("threshold", image=_eager(image))
        result = event.outputs["0"]
        assert result.WhichOneof("kind") == "lazy"
        np.testing.assert_array_equal(_value(result), image > 0.5)

        # Its reference is a lazy input to the next call, and a dask result is
        # written to the plane without being assembled.
        (event,) = s.call("double", image=result)
        doubled = event.outputs["result"]
        assert doubled.WhichOneof("kind") == "lazy"
        np.testing.assert_array_equal(_value(doubled), (image > 0.5) * 2)
    finally:
        s.close()


def test_small_results_stay_inline_with_a_plane(plane):
    s = _Server(_defs([total, threshold]), sink=_PlaneSink(plane, None))
    try:
        (event,) = s.call("threshold", image=_eager(np.ones((4, 4), np.float32)))
        assert event.outputs["0"].WhichOneof("kind") == "eager"
    finally:
        s.close()


def test_embedded_sink(tmp_path: Path, monkeypatch):
    import biopb_image_base.ops as ops
    from biopb_image_base.server import start_embedded_cache
    from biopb_tensor_server.cache import CacheManager

    from tests.tensor_cache_test import _free_tcp_port

    monkeypatch.setattr(ops, "_MAX_EAGER_SIZE", 1024)
    CacheManager.reset()
    port = _free_tcp_port()
    cache = start_embedded_cache(
        str(tmp_path),
        "128MiB",
        ip="127.0.0.1",
        local=True,
        tensor_port=port,
    )
    s = _Server(_defs([smooth]), sink=_EmbeddedSink(cache))
    try:
        image = np.random.rand(40, 33).astype(np.float32)
        (event,) = s.call("smooth", image=_eager(image))
        result = event.outputs["result"]
        assert result.WhichOneof("kind") == "lazy"
        assert result.lazy.auth_token  # the result's own read grant
        assert _value(result).shape == image.shape
    finally:
        s.close()
        cache._server.shutdown()
        CacheManager.reset()
