"""Tests for the dynamically built ProcessImage ops (mcp/_process_ops.py).

These exercise the op-building and serialization logic with fake gRPC stubs and
a fake tensor client, so no live server or napari viewer is required.
"""

from unittest.mock import MagicMock

import biopb.image as proto
import dask.array as da
import grpc
import numpy as np
import pytest
from biopb.image.utils import (
    serialize_from_numpy_to_image_data,
)
from biopb.tensor import SerializedTensor

from biopb_mcp.mcp import _process_ops
from biopb_mcp.mcp._process_ops import (
    _build_op,
    _infer_dim_labels,
    _make_channel,
    _server_key,
    build_ops,
)


def _eager_response(arr, dim_labels):
    return proto.ProcessResponse(
        image_data=serialize_from_numpy_to_image_data(arr, dim_labels=dim_labels)
    )


class TestInferDimLabels:
    @pytest.mark.parametrize(
        "ndim,expected",
        [
            (2, ["Y", "X"]),
            (3, ["Y", "X", "C"]),
            (4, ["Z", "Y", "X", "C"]),
            (5, ["T", "Z", "Y", "X", "C"]),
        ],
    )
    def test_known(self, ndim, expected):
        assert _infer_dim_labels(ndim) == expected

    def test_unsupported_returns_none(self):
        assert _infer_dim_labels(6) is None
        assert _infer_dim_labels(1) is None


class TestMakeChannel:
    def test_grpc_insecure(self):
        ch = _make_channel("grpc://localhost:50051")
        assert isinstance(ch, grpc.Channel)
        ch.close()

    def test_grpcs_secure(self):
        ch = _make_channel("grpcs://localhost:50051")
        assert isinstance(ch, grpc.Channel)
        ch.close()

    def test_bad_scheme_raises(self):
        with pytest.raises(ValueError):
            _make_channel("http://localhost:50051")

    def test_no_host_raises(self):
        with pytest.raises(ValueError):
            _make_channel("grpc://")


class TestServerKey:
    def test_sanitizes_netloc(self):
        assert _server_key("grpc://localhost:50051") == "localhost_50051"
        assert _server_key("grpcs://host:1/path") == "host_1"


class TestBuildOpNdarray:
    def test_ndarray_in_ndarray_out(self):
        arr = np.arange(12, dtype="uint16").reshape(3, 4)
        stub = MagicMock()
        stub.Run.side_effect = lambda req, timeout=None: proto.ProcessResponse(
            image_data=req.image_data  # echo
        )
        op = _build_op(stub, "echo", None, "grpc://h:1", lambda: None, 30.0)

        out = op(arr)

        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, arr)

    def test_forwards_op_name_kwargs_timeout(self):
        arr = np.zeros((2, 2), dtype="uint8")
        captured = {}

        def run(req, timeout=None):
            captured["req"] = req
            captured["timeout"] = timeout
            return proto.ProcessResponse(image_data=req.image_data)

        stub = MagicMock()
        stub.Run.side_effect = run
        op = _build_op(stub, "seg", None, "grpc://h:1", lambda: None, 42.0)

        op(arr, threshold=0.5, mode="fast")

        assert captured["req"].op_name == "seg"
        assert captured["timeout"] == 42.0
        kw = dict(captured["req"].kwargs)
        assert kw["threshold"] == 0.5
        assert kw["mode"] == "fast"

    def test_infers_dim_labels(self):
        arr = np.zeros((2, 3), dtype="uint8")
        captured = {}

        def run(req, timeout=None):
            captured["req"] = req
            return proto.ProcessResponse(image_data=req.image_data)

        stub = MagicMock()
        stub.Run.side_effect = run
        op = _build_op(stub, "echo", None, "grpc://h:1", lambda: None, 30.0)

        op(arr)

        assert list(captured["req"].image_data.eager_data.dim_labels) == [
            "Y",
            "X",
        ]

    def test_dim_labels_override(self):
        arr = np.zeros((2, 3), dtype="uint8")
        captured = {}

        def run(req, timeout=None):
            captured["req"] = req
            return proto.ProcessResponse(image_data=req.image_data)

        stub = MagicMock()
        stub.Run.side_effect = run
        op = _build_op(stub, "echo", None, "grpc://h:1", lambda: None, 30.0)

        op(arr, dim_labels=["X", "Y"])

        assert list(captured["req"].image_data.eager_data.dim_labels) == [
            "X",
            "Y",
        ]

    def test_annotation_only_response(self):
        stub = MagicMock()
        stub.Run.return_value = proto.ProcessResponse(annotation="cells=5")
        op = _build_op(stub, "echo", None, "grpc://h:1", lambda: None, 30.0)

        assert op(np.zeros((2, 2), dtype="uint8")) == "cells=5"


class TestBuildOpArrayId:
    def test_source_id_in_source_id_out(self):
        client = MagicMock()
        client.get_tensor_pb.return_value = SerializedTensor()
        client.upload_array.return_value = "cache_99"

        result_arr = np.ones((2, 2), dtype="uint8")
        stub = MagicMock()
        stub.Run.return_value = _eager_response(result_arr, ["Y", "X"])

        op = _build_op(stub, "seg", None, "grpc://h:1", lambda: client, 30.0)

        out = op("src_id")

        assert out == "cache_99"
        client.get_tensor_pb.assert_called_once_with("src_id")
        # Sent as lazy_data, not eager.
        sent = stub.Run.call_args[0][0]
        assert sent.image_data.WhichOneof("data") == "lazy_data"
        # Uploaded result is a dask array to an ephemeral cache source.
        up_args = client.upload_array.call_args[0]
        assert isinstance(up_args[0], da.Array)
        assert up_args[1] == "cache:"

    def test_source_id_without_client_raises(self):
        stub = MagicMock()
        op = _build_op(stub, "seg", None, "grpc://h:1", lambda: None, 30.0)
        with pytest.raises(RuntimeError):
            op("src_id")


class TestBuildOps:
    def test_multi_op_server(self, monkeypatch):
        op_names = proto.OpNames(
            names=["a", "b"],
            op_schemas={
                "a": proto.OpSchema(description="op a", labels=["seg"]),
                "b": proto.OpSchema(description="op b"),
            },
        )
        fake_stub = MagicMock()
        fake_stub.GetOpNames.return_value = op_names
        monkeypatch.setattr(
            _process_ops,
            "_make_channel",
            lambda url, options=None: MagicMock(),
        )
        monkeypatch.setattr(
            _process_ops.proto, "ProcessImageStub", lambda ch: fake_stub
        )

        ops = build_ops(lambda: None, ["grpc://h:1"])

        assert set(ops) == {"a", "b"}
        assert ops["a"].op_name == "a"
        assert ops["a"].server == "grpc://h:1"
        assert ops["a"].labels == ["seg"]
        assert "op a" in ops["a"].__doc__

    def test_single_op_server_unimplemented(self, monkeypatch):
        class _Unimplemented(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNIMPLEMENTED

        fake_stub = MagicMock()
        fake_stub.GetOpNames.side_effect = _Unimplemented()
        monkeypatch.setattr(
            _process_ops,
            "_make_channel",
            lambda url, options=None: MagicMock(),
        )
        monkeypatch.setattr(
            _process_ops.proto, "ProcessImageStub", lambda ch: fake_stub
        )

        ops = build_ops(lambda: None, ["grpc://host:50051"])

        assert "host_50051" in ops
        assert ops["host_50051"].op_name is None

    def test_unreachable_server_skipped(self, monkeypatch):
        class _Unavailable(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

        fake_stub = MagicMock()
        fake_stub.GetOpNames.side_effect = _Unavailable()
        monkeypatch.setattr(
            _process_ops,
            "_make_channel",
            lambda url, options=None: MagicMock(),
        )
        monkeypatch.setattr(
            _process_ops.proto, "ProcessImageStub", lambda ch: fake_stub
        )

        ops = build_ops(lambda: None, ["grpc://h:1"])

        assert ops == {}

    def test_invalid_url_skipped(self):
        ops = build_ops(lambda: None, ["http://bad"])
        assert ops == {}

    def test_empty_server_list(self):
        assert build_ops(lambda: None, []) == {}
        assert build_ops(lambda: None, None) == {}
