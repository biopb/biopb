"""Dynamically built ``biopb.image.ProcessImage`` operations for the agent.

Each ProcessImage servicer URL listed in
``config["mcp"]["services"]["process_image_servers"]``
(``grpc://`` for plaintext, ``grpcs://`` for TLS) is queried via ``GetOpNames``.
Every advertised op becomes a thin callable placed in the ``ops`` dict in the
``execute_code`` namespace.

Unlike the Image Processing widget, these functions add no chunking or
dimensional iteration. They expose the ``Run`` RPC almost directly so the agent
can decide how to use them, and they support tensor-server ``array_id`` (a
source_id string) on both input and output:

* ``op(ndarray)``  -> inline ``eager_data`` request -> ``np.ndarray`` result.
* ``op("src_id")`` -> ``lazy_data`` request built from ``client.get_tensor_pb``
  (the server pulls pixels from the tensor server directly, no kernel
  round-trip) -> result uploaded back to the tensor server -> new source_id str.
"""

import logging
from collections.abc import Callable
from typing import Dict, List, Optional
from urllib.parse import urlparse

import biopb.image as proto
import dask.array as da
import grpc
import numpy as np
from biopb.image.utils import (
    deserialize_image_data,
    serialize_from_numpy_to_image_data,
)
from google.protobuf import empty_pb2

logger = logging.getLogger(__name__)

# biopb's ndim -> axis-label convention (see biopb.image.utils).
_NDIM_LABELS = {
    2: ["Y", "X"],
    3: ["Y", "X", "C"],
    4: ["Z", "Y", "X", "C"],
    5: ["T", "Z", "Y", "X", "C"],
}


def _infer_dim_labels(ndim: int) -> Optional[List[str]]:
    """Return biopb's default axis labels for *ndim*, or None if unsupported."""
    return _NDIM_LABELS.get(ndim)


def _make_channel(url: str, options=None) -> grpc.Channel:
    """Build a gRPC channel from a ``grpc://`` or ``grpcs://`` URL."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    target = parsed.netloc or parsed.path
    if not target:
        raise ValueError(f"ProcessImage server URL has no host: {url!r}")
    if scheme == "grpcs":
        return grpc.secure_channel(
            target, grpc.ssl_channel_credentials(), options=options
        )
    if scheme == "grpc":
        return grpc.insecure_channel(target, options=options)
    raise ValueError(
        f"ProcessImage server URL must use grpc:// or grpcs://, got {url!r}"
    )


def _server_key(url: str) -> str:
    """Stable dict key for a server that does not implement GetOpNames."""
    parsed = urlparse(url)
    netloc = parsed.netloc or parsed.path or url
    return netloc.replace(":", "_").replace("/", "_")


def _struct_to_dict(struct) -> dict:
    """Convert a protobuf Struct to a plain dict of python values."""
    try:
        return dict(struct)
    except Exception:
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(struct)


def _build_op(
    stub: "proto.ProcessImageStub",
    op_name: Optional[str],
    schema,
    server_url: str,
    client_getter: Callable[[], object],
    run_timeout: float,
) -> Callable:
    """Build a single callable bound to (stub, op_name)."""
    default_kwargs = _struct_to_dict(schema.default_kwargs) if schema else {}
    description = schema.description if schema else ""
    labels = list(schema.labels) if schema else []
    hint = schema.input_shape_hint if schema else None

    def op(image, dim_labels=None, **kwargs):
        client = client_getter()
        is_id = isinstance(image, str)

        if is_id:
            if client is None:
                raise RuntimeError(
                    "No tensor server connected; cannot resolve array_id "
                    f"{image!r}. Connect the Tensor Browser first."
                )
            image_data = proto.ImageData(lazy_data=client.get_tensor_pb(image))
        else:
            arr = np.asarray(image)
            arr_labels = (
                dim_labels if dim_labels is not None else _infer_dim_labels(arr.ndim)
            )
            image_data = serialize_from_numpy_to_image_data(arr, dim_labels=arr_labels)

        request = proto.ProcessRequest(image_data=image_data, op_name=op_name or "")
        if kwargs:
            request.kwargs.update(kwargs)

        response = stub.Run(request, timeout=run_timeout)

        # Annotation-only responses carry no image; hand back the text.
        if response.image_data.WhichOneof("data") is None:
            return response.annotation or None

        result = deserialize_image_data(response.image_data)

        if is_id:
            # Symmetric id<->id: consolidate the result onto the agent's
            # tensor server and return a source_id for further lazy chaining.
            if not isinstance(result, da.Array):
                result = da.from_array(result, chunks=result.shape)
            return client.upload_array(result, "cache:")

        # ndarray in -> ndarray out.
        if isinstance(result, da.Array):
            result = result.compute()
        return result

    doc = [
        description
        or f"biopb.image ProcessImage op {op_name or '(single-op server)'}.",
        "",
        f"Server: {server_url}",
        f"Op name: {op_name or '(unnamed / single-op server)'}",
    ]
    if labels:
        doc.append(f"Labels: {', '.join(labels)}")
    if hint is not None and (hint.expected_singletons or hint.required_multivalue):
        doc.append(
            "Input shape hint: "
            f"expected_singletons={list(hint.expected_singletons)}, "
            f"required_multivalue={list(hint.required_multivalue)}"
        )
    if default_kwargs:
        doc.append(f"Default kwargs (override via **kwargs): {default_kwargs}")
    doc += [
        "",
        "Call: op(image, dim_labels=None, **kwargs)",
        "  image: np.ndarray (sent inline/eager) OR a tensor-server source_id",
        "    str (sent as a lazy reference; the server pulls pixels from the",
        "    tensor server directly).",
        "  dim_labels: axis labels for ndarray input; inferred from ndim when",
        "    None (2D=YX, 3D=YXC, 4D=ZYXC, 5D=TZYXC).",
        "  Returns np.ndarray when image is an array; a new source_id str when",
        "  image is a source_id (result uploaded as an ephemeral 'cache:'",
        "  source on the connected tensor server).",
    ]
    op.__doc__ = "\n".join(doc)
    op.__name__ = _sanitize_name(op_name) or "process_op"
    op.op_name = op_name
    op.server = server_url
    op.labels = labels
    op.description = description
    op.default_kwargs = default_kwargs
    return op


def _sanitize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in name)


def build_ops(
    client_getter: Callable[[], object],
    server_urls,
    op_names_timeout: float = 10.0,
    run_timeout: float = 300.0,
    channel_options=None,
) -> Dict[str, Callable]:
    """Build the ``ops`` dict from configured ProcessImage server URLs.

    Unreachable servers and invalid URLs are skipped (logged), so a bad entry
    never aborts kernel bootstrap. ``client_getter`` is called at op-call time
    (not now) so the asynchronously-connecting tensor client is picked up live.
    """
    ops: Dict[str, Callable] = {}

    for url in server_urls or []:
        try:
            channel = _make_channel(url, channel_options)
        except ValueError:
            logger.warning("Skipping invalid ProcessImage URL %r", url)
            continue

        stub = proto.ProcessImageStub(channel)
        try:
            op_names = stub.GetOpNames(empty_pb2.Empty(), timeout=op_names_timeout)
            names: List[Optional[str]] = list(op_names.names)
            schemas = dict(op_names.op_schemas)
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                # Single-op server: one nameless op (op_name="").
                names, schemas = [None], {}
                logger.info("%s has no GetOpNames; single-op mode", url)
            else:
                logger.warning(
                    "GetOpNames failed for %s (%s); skipping",
                    url,
                    exc.code(),
                )
                channel.close()
                continue

        for name in names:
            schema = schemas.get(name) if name is not None else None
            op = _build_op(stub, name, schema, url, client_getter, run_timeout)
            # Keep the channel alive for as long as the op is referenced.
            op._channel = channel
            key = name if name else _server_key(url)
            if key in ops:
                logger.warning(
                    "Duplicate op key %r (from %s) overrides earlier server",
                    key,
                    url,
                )
            ops[key] = op

    return ops
